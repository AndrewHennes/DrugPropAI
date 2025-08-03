#!/usr/bin/env python3

# ============================================================================
# 1  Imports & helpers
# ============================================================================

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Built-In Modules:
# --------------------------------------------------------------------------- #

import abc
import copy
import os
import random
import re
from functools import partial, wraps
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union
import math
import sys
import base64, io
import pathlib
from pathlib import Path
import tempfile
import inspect
from collections import defaultdict, deque
import uuid

# --------------------------------------------------------------------------- #
# External Modules:
# --------------------------------------------------------------------------- #

import types, torch.hub as hub 
utils = types.ModuleType("torchvision.models.utils") 
utils.load_state_dict_from_url = hub.load_state_dict_from_url 
sys.modules["torchvision.models.utils"] = utils

import LibMTL
from LibMTL.architecture import HPS, Cross_stitch, MMoE
from LibMTL.weighting import PCGrad as _LibPCGrad

import pandas as pd

import ray

from rdkit import Chem

import torch
from torch.utils.data import TensorDataset, IterableDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from torch_geometric.data import Data, Batch
#from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from torch_geometric.nn    import (
    MessagePassing,
    global_mean_pool, global_add_pool, global_max_pool,
    Linear as GeoLinear,
    GATConv, GCNConv                        # community layer
)
from torch_geometric.data import Batch as GeoBatch

from torchmetrics.functional import accuracy as _tm_acc
from torchmetrics.functional import auroc as _tm_auroc
from torchmetrics.functional import average_precision as _tm_ap

from torch_scatter import scatter

from ray import tune
from ray.air import RunConfig, CheckpointConfig
from ray.tune import TuneConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback

data_processing_pipeline_dir_path = "/Users/asselism/Desktop/Collins_Lab/SharedCode/Shared_Pipeline_Working_Dir/"
sys.path.insert(0, data_processing_pipeline_dir_path)
from Shared_Pipeline_Ray_v1_10 import *  # TODO: Update with the real import path.

###############################################################################
# 0. Other Functionality:
###############################################################################

# =========================================================================== #
# Environment Variables:
# =========================================================================== #

os.environ["TUNE_WARN_EXCESSIVE_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S"] = "0"

# =========================================================================== #
# Enable Module Lookup in Ray Workers:
# =========================================================================== #

_mod_name = pathlib.Path(__file__).stem          # "Model_Building_Pipeline_1_10"
sys.modules[_mod_name] = sys.modules[__name__]   # point canonical name → __main__

# =========================================================================== #
# Chemical Representations:
# =========================================================================== #

MAX_ATOMIC_NUMBER = 118
NODE_FEAT_DIM = MAX_ATOMIC_NUMBER
ATOM_TYPES  = list(range(1, MAX_ATOMIC_NUMBER + 1))                            
BOND_TYPES  = {Chem.BondType.SINGLE: 0,
               Chem.BondType.DOUBLE: 1,
               Chem.BondType.TRIPLE: 2,
               Chem.BondType.AROMATIC: 3}

# =========================================================================== #
# Table Functionality:
# =========================================================================== #

def _to_tensor(
    self,
    columns: str | Sequence[str],
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
    max_rows: int | None = 5_000_000,
) -> torch.Tensor:
    """
    Convert one or more `Table` columns to a (potentially device‑resident)  
    `torch.Tensor`.

    Parameters
    ----------
    columns : str | Sequence[str]
        Name(s) of the column(s) to materialise.  If a single string is
        supplied, the returned tensor is squeezed to 1‑D.
    dtype : torch.dtype, default=torch.float32
        Desired floating‑point / integer dtype.  If the underlying pandas
        column already matches `dtype`, the copy is avoided.
    device : torch.device | str | None, default=None
        Target device.  `None` (or `"cpu"`) leaves the tensor on CPU;
        `"cuda"`/`"cuda:0"` etc. copy (non‑blocking) to GPU.
    max_rows : int | None, default=5_000_000
        Hard cap to protect against accidental materialisation of huge
        distributed / on‑disk `Table`s.  `None` disables the safeguard.

    Returns
    -------
    torch.Tensor
        *Shape:* ``(N, len(columns))`` if multiple columns are requested,
        otherwise ``(N,)``.  The tensor is **view‑backed** when possible
        (no copy) and moved to `device` if requested.

    Notes
    -----
    * Uses ``Table.to_pandas()`` internally, so any lazy / remote rows are
      brought into memory; be mindful of `max_rows`.
    * If you plan to feed the tensor into a Lightning trainer, prefer leaving
      it on CPU and rely on the Trainer’s device placement logic instead.

    Examples
    --------
    >>> t.to_tensor(["height_cm", "weight_kg"])          # shape (N, 2)
    >>> t.to_tensor("age", device="cuda")                # shape (N,)
    """
    if isinstance(columns, str):
        columns = [columns]
    df = self.to_pandas(max_rows=max_rows)[list(columns)]
    tens = torch.as_tensor(df.to_numpy(copy=False), dtype=dtype, device=device)
    return tens if len(columns) > 1 else tens.squeeze(-1)


def _as_torch_dataset(
    self,
    x_cols: Sequence[str],
    y_cols: str | Sequence[str],
    *,
    batch_size: int = 4_096,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = "cpu",
    max_rows_in_memory: int | None = 5_000_000,
):
    """
    Produce a ready‑to‑use `torch.utils.data.Dataset` view of a `Table`.

    The helper automatically **switches between a fully‑materialised
    `TensorDataset` and a streaming `IterableDataset`** depending on table
    size and the data‑frame backend (`pandas`, `Ray`, `Dask`, …).

    Parameters
    ----------
    x_cols : Sequence[str]
        Column names to be treated as (multi‑dimensional) input features ``X``.
    y_cols : str | Sequence[str]
        Column name(s) for target(s) ``y``.  If multiple, the dataset yields a
        *tuple* of targets.
    batch_size : int, default=4096
        Chunk size used **only** in the streaming path.  Each yielded sample is
        still a *single* row – batching is left to the PyTorch `DataLoader`.
    dtype : torch.dtype, default=torch.float32
        Numerical precision for both X and y tensors.
    device : torch.device | str | None, default="cpu"
        Destination device.  `"cuda"` is honoured even in the streaming path
        (non‑blocking `.to()` during iteration).
    max_rows_in_memory : int | None, default=5_000_000
        Threshold that decides whether the eager, in‑RAM path is chosen.
        Set to `None` to **force** streaming (useful for memory profiling).

    Returns
    -------
    torch.utils.data.Dataset
        Either:

        * `TensorDataset(X, y)` if the whole selection fits in RAM, **or**
        * a custom `IterableDataset` that lazily yields ``(xi, yi)`` pairs.

    Examples
    --------
    Small table ⇒ eager path::

        train_ds = tbl.as_torch_dataset(
            x_cols=["feat1", "feat2"],
            y_cols="label",
            device="cuda"
        )
        model.fit(DataLoader(train_ds, batch_size=1024))

    Large / distributed table ⇒ streaming path::

        stream_ds = big_tbl.as_torch_dataset(
            x_cols=features, y_cols=targets,
            batch_size=8_192, max_rows_in_memory=1_000_000
        )
        for xb, yb in DataLoader(stream_ds, batch_size=None):
            ...

    Notes
    -----
    * The streaming dataset pays a small extra cost to allocate tensors on
      each chunk; this remains negligible compared to IO for ≥ 10 k rows.
    * The generator **yields individual samples**, not batches, so that a
      standard `DataLoader` can still take care of shuffling and collate.
    """
    # ---------- eager path -----------------------------------------
    if self.is_pandas() and (
        max_rows_in_memory is None or len(self) <= max_rows_in_memory
    ):
        X = self.to_tensor(
            x_cols, dtype=dtype, device=device, max_rows=max_rows_in_memory
        )
        y = self.to_tensor(
            y_cols, dtype=dtype, device=device, max_rows=max_rows_in_memory
        )
        return TensorDataset(X, y)

    # ---------- streaming path ------------------------------------
    if isinstance(y_cols, str):
        y_cols = [y_cols]

    class _Stream(IterableDataset):
        """
        Lazily iterates over table chunks and yields individual (xi, yi) pairs.

        A fresh tensor batch is allocated per chunk, then row‑by‑row values are
        yielded to keep *peak* memory low and enable non‑blocking `.to(device)`.
        """

        def __iter__(inner):  # noqa: D401  (Lightning docs false‑positive)
            for chunk in self.iter_chunks(batch_size):
                Xt = torch.as_tensor(
                    chunk[list(x_cols)].to_numpy(), dtype=dtype
                )
                yt = torch.as_tensor(
                    chunk[list(y_cols)].to_numpy(), dtype=dtype
                )
                if device not in (None, "cpu"):
                    Xt = Xt.to(device, non_blocking=True)
                    yt = yt.to(device, non_blocking=True)
                for xi, yi in zip(Xt, yt):
                    yield xi, (yi.squeeze() if yi.numel() == 1 else yi)

    return _Stream()


# Register the two helpers on the upstream `Table` class
Table.to_tensor = _to_tensor                   # type: ignore[attr-defined]
Table.as_torch_dataset = _as_torch_dataset     # type: ignore[attr-defined]

# =============================================================================
# Machine Learning Parts Registry:
# =============================================================================

_REGISTRIES: Dict[str, Dict[str, Any]] = {
    "core": {},
    "head": {},
    "arch": {},
    "weighting": {},
    "loss": {}, 
    "optim": {},
    "metric": {},
    "ptl": {},
    "callback": {},
    "config": {}
}


def register(kind: str, name: str):
    """
    Decorator that plugs a class / factory function into one of the global
    registries (``atom``, ``mixer``, ``head``, ``net``, ``loss``).

    The pattern lets us **dynamically compose** models and loss stacks purely
    from YAML / CLI config strings – without hard‑coding imports.

    Parameters
    ----------
    kind : {"atom", "mixer", "head", "net", "loss"}
        Registry namespace.
    name : str
        Key under which the object will be stored.

    Returns
    -------
    Callable
        A decorator.  Usage::

            @register("head", "classification")
            class ClassificationHead(nn.Module):
                ...

    Notes
    -----
    * Re‑decorating an existing ``kind/name`` pair silently *overwrites* the
      previous entry – handy for hot‑reloading during experiments.
    """
    def decorator(cls_or_fn):
        _REGISTRIES[kind][name] = cls_or_fn
        return cls_or_fn
    return decorator


def get_from_registry(kind: str, name: str):
    """
    Fetch an object previously registered with :func:`register`.

    Parameters
    ----------
    kind : str
        Namespace, e.g. ``"atom"``.
    name : str
        Key that identifies the class / function.

    Returns
    -------
    Any
        The registered object.

    Raises
    ------
    KeyError
        If the given ``kind/name`` pair has not been registered.
    """
    if name not in _REGISTRIES[kind]:
        raise KeyError(f"{name} not found in {kind} registry")
    return _REGISTRIES[kind][name]

get_core  = lambda name: get_from_registry("core",  name)
get_head  = lambda name: get_from_registry("head",  name)
get_arch  = lambda name: get_from_registry("arch",  name)
get_weighting  = lambda name: get_from_registry("weighting",  name)
get_loss = lambda name: get_from_registry("loss",  name)
get_optim = lambda name: get_from_registry("optim",  name)
get_metric = lambda name: get_from_registry("metric",  name)
get_ptl = lambda name: get_from_registry("ptl",  name)
get_callback = lambda name: get_from_registry("callback",  name)
get_config = lambda name: get_from_registry("config",  name)

# =============================================================================
# Data Type Conversion:
# =============================================================================

def smiles_to_data(smiles: str) -> Data | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # --- atoms ---------------------------------------------------------------
    x_list = []
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        feat = torch.zeros(len(ATOM_TYPES))
        if z in ATOM_TYPES:
            feat[z - 1] = 1.0
        x_list.append(feat)
    x = torch.stack(x_list, dim=0)                                  

    # --- bonds ---------------------------------------------------------------
    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

        # ---- one‑hot 4‑vector instead of scalar ---------------------------
        t = torch.zeros(len(BOND_TYPES), dtype=torch.float32)      # (4,)
        t[BOND_TYPES[bond.GetBondType()]] = 1.0
        edge_index.extend([[i, j], [j, i]])
        edge_attr.extend([t, t])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr  = torch.stack(edge_attr, dim=0) if edge_attr else None

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# #############################################################################
# 1. CORE:
# #############################################################################

# =============================================================================
# A. Core standard neural‑network components
# =============================================================================
class QueryInterface:
    """
    Optional mix‑in that exposes uniform introspection hooks.
    Leaf modules override selectively.
    """

    # ----- predictions / targets --------------------------------------------
    def get_preds(self) -> Dict[str, torch.Tensor]:           # noqa: D401
        return {}

    # ----- hidden activations ------------------------------------------------
    def get_activations(self) -> Dict[str, torch.Tensor]:     # noqa: D401
        return {}

    # ----- learnable parameters ---------------------------------------------
    def get_learnable_weights(self) -> Dict[str, torch.Tensor]:
        # Default: expose all trainable parameters with their *full* names.
        return {n: p for n, p in self.named_parameters() if p.requires_grad}


# --------------------------------------------------------------------------- #
# Linear and bias layers
# --------------------------------------------------------------------------- #
@register("core", "linear_transformation")
class LinearTransformation(QueryInterface, nn.Module):
    """Weight‑only `y = Wx`."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight.t())


@register("core", "bias")
class Bias(QueryInterface, nn.Module):
    """Learnable bias added to last dimension."""

    def __init__(self, out_dim: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bias


@register("core", "projection")
class Projection(QueryInterface, nn.Module):
    """Affine transform = Linear ∘ (optional) Bias."""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.linear = LinearTransformation(in_dim, out_dim)
        self.bias   = Bias(out_dim) if bias else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bias(self.linear(x))


# --------------------------------------------------------------------------- #
# Dense MLP block
# --------------------------------------------------------------------------- #
class DenseStage(QueryInterface, nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        norm: Optional[str] = None,
        act: str = "relu",
        dropout_p: float = 0.0,
        local_name: str = "dense",
    ):
        super().__init__()
        self.proj = get_from_registry("core", "projection")(in_dim, out_dim)
        self.norm = (
            get_from_registry("core", norm)(out_dim) if norm else nn.Identity()
        )
        self.act = get_from_registry("core", act)()
        self.reg = get_from_registry("core", "dropout")(dropout_p)

        self._latest_out: Optional[torch.Tensor] = None
        self._name = local_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reg(self.act(self.norm(self.proj(x))))
        self._latest_out = x
        return x

    def get_activations(self) -> Dict[str, torch.Tensor]:
        return {self._name: self._latest_out} if self._latest_out is not None else {}


class FeedForwardBlock(QueryInterface, nn.Module):
    def __init__(
        self,
        dims: Sequence[int],
        *,
        name_prefix: str,
        norm: Optional[str] = "layer_norm",
        act: str = "relu",
        dropout_p: float = 0.0,
    ):
        super().__init__()
        self.stages = nn.ModuleList(
            [
                DenseStage(
                    d_in,
                    d_out,
                    norm=norm,
                    act=act,
                    dropout_p=dropout_p,
                    local_name=f"{name_prefix}.stage{i}",
                )
                for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:]))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x)
        return x

    def get_activations(self) -> Dict[str, torch.Tensor]:
        acts: Dict[str, torch.Tensor] = {}
        for stage in self.stages:
            acts.update(stage.get_activations())
        return acts


class SharedTrunk(QueryInterface, nn.Module):
    def __init__(self, dims: Sequence[int], **kw):
        super().__init__()
        self.block = FeedForwardBlock(dims, name_prefix="trunk", **kw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

    def get_activations(self) -> Dict[str, torch.Tensor]:
        return self.block.get_activations()


class PrivateTrunk(QueryInterface, nn.Module):
    def __init__(self, task: str, dims: Sequence[int], **kw):
        super().__init__()
        self.block = FeedForwardBlock(dims, name_prefix=f"{task}.trunk", **kw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

    def get_activations(self) -> Dict[str, torch.Tensor]:
        return self.block.get_activations()


# --------------------------------------------------------------------------- #
# Output heads
# --------------------------------------------------------------------------- #
@register("head", "classification")
class ClassificationHead(QueryInterface, nn.Module):
    def __init__(self, in_dim: int, num_classes: int, task: str):
        super().__init__()
        self.proj = nn.Linear(in_dim, num_classes)
        self.task = task
        self._logits: Optional[torch.Tensor] = None
        self.out_dim = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._logits = self.proj(x)
        return self._logits

    def get_preds(self) -> Dict[str, torch.Tensor]:
        return {self.task: self._logits} if self._logits is not None else {}


@register("head", "regression")
class RegressionHead(QueryInterface, nn.Module):
    def __init__(self, in_dim: int, task: str):
        super().__init__()
        self.proj = nn.Linear(in_dim, 1)
        self.task = task
        self._out: Optional[torch.Tensor] = None
        self.out_dim = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._out = self.proj(x).squeeze(-1)
        return self._out

    def get_preds(self) -> Dict[str, torch.Tensor]:
        return {self.task: self._out} if self._out is not None else {}
    
@register("head", "mlp")
class MLPHead(QueryInterface, nn.Module):
    """
    Simple MLP → 1‑dim logit / regression output.

    Parameters
    ----------
    in_dim : int
    hidden_dims : Sequence[int] | int
        If an int, interpreted as one hidden layer of that width.
    out_dim : int, default 1
        • 1  → binary logit / regression  
        • >1 → multiclass logits
    task : str
        Used as key in the prediction dict expected by the helpers.
    """
    def __init__(
        self,
        in_dim: int,
        task: str,
        *,
        hidden_dims: Union[int, Sequence[int]] = 128,
        out_dim: int = 1,
        activate: str = "relu",
        dropout_p: float = 0.0,
    ):
        super().__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        dims = [in_dim, *hidden_dims, out_dim]
        layers = []
        act_cls = get_from_registry("core", activate)
        for d_in, d_out in zip(dims[:-2], dims[1:-1]):
            layers += [nn.Linear(d_in, d_out), act_cls(), nn.Dropout(dropout_p)]
        layers += [nn.Linear(dims[-2], dims[-1])]

        self.net = nn.Sequential(*layers)
        self.task = task
        self.in_dim  = in_dim
        self.out_dim = out_dim
        self._preds: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._preds = self.net(x)
        return self._preds

    def get_preds(self):
        return {self.task: self._preds} if self._preds is not None else {}

# --------------------------------------------------------------------------- #
# Normalisation, dropout, activations
# --------------------------------------------------------------------------- #
@register("core", "layer_norm")
class LayerNorm(QueryInterface, nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


@register("core", "dropout")
class Dropout(QueryInterface, nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x)


@register("core", "relu")
class ReLU(QueryInterface, nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x)


@register("core", "gelu")
class GELU(QueryInterface, nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)


# =========================================================================== #
# B. Core graph components (PyG)
# =========================================================================== #
STAGES: List[str] = [
    "preprocessor",
    "node_embedder", "edge_embedder", "structural_encoder",
    "message_fn", "edge_attention", "local_aggregator",
    "global_attention", "hybrid_router",
    "node_update", "edge_update",
    "feed_forward", "norm_reg",
    "readout_pool", "multiscale_agg", "task_head",
]


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GraphPreprocessor(nn.Module):
    covers = ("preprocessor",)
    out_node_dim: Optional[int] = None
    out_edge_dim: Optional[int] = None

    def forward(self, data: Data) -> Data:
        if getattr(data, "edge_attr", None) is None:
            E = data.edge_index.size(1)
            data.edge_attr = data.x.new_zeros(E, 1)
        if data.x.dim() == 1:
            data.x = data.x.unsqueeze(-1)
        if not hasattr(data, "batch"):
            data.batch = data.x.new_zeros(data.num_nodes, dtype=torch.long)
        return data


# --------------------------------------------------------------------------- #
# Identity block (renamed)
# --------------------------------------------------------------------------- #
class Identity_Transformation(nn.Identity):
    def __init__(
        self,
        out_node_dim: Optional[int] = None,
        out_edge_dim: Optional[int] = None,
    ):
        super().__init__()
        self.out_node_dim = out_node_dim
        self.out_edge_dim = out_edge_dim


# --------------------------------------------------------------------------- #
# Scatter helpers
# --------------------------------------------------------------------------- #
_SCATTER = {
    "sum":  lambda src, idx, n: scatter(src, idx, dim=0, dim_size=n, reduce="sum"),
    "mean": lambda src, idx, n: scatter(src, idx, dim=0, dim_size=n, reduce="mean"),
    "max":  lambda src, idx, n: scatter(src, idx, dim=0, dim_size=n, reduce="max"),
}


# --------------------------------------------------------------------------- #
# Node Encoder:
# --------------------------------------------------------------------------- #
class NodeEncoder(nn.Module):
    covers = ("node_embedder",)

    def __init__(self, in_dim: int, out_dim: int, scheme: str = "linear"):
        super().__init__()
        if scheme == "identity":
            assert in_dim == out_dim
            self.embed = Identity_Transformation(out_node_dim=out_dim)
        elif scheme == "onehot":
            self.embed = nn.Embedding(in_dim, out_dim)
        elif scheme == "linear":
            self.embed = GeoLinear(in_dim, out_dim, bias=False)
        elif scheme == "affine":
            self.embed = GeoLinear(in_dim, out_dim, bias=True)
        else:
            raise ValueError(scheme)
        self.out_node_dim = out_dim
        self.out_edge_dim = None

    def forward(self, data: Data) -> Data:
        """
        Idempotent encoder – converts 118-dim atom one-hot vectors to the
        hidden size exactly once per sample.  Subsequent passes in the same
        epoch see the already-embedded representation and skip re-encoding.
        """
        in_feat = getattr(
            self.embed, "in_features",
            getattr(self.embed, "in_channels", None)
        )

        # 1️⃣ First encounter – raw one-hot features ➜ embed them
        if data.x.size(-1) == in_feat:
            data.x = self.embed(data.x)

        # 2️⃣ Re-encounter within the same batch/epoch – already embedded
        elif data.x.size(-1) == self.out_node_dim:
            pass  # no-op: keep the cached embedding

        # 3️⃣ Anything else is unexpected ⇒ raise
        else:
            raise RuntimeError(
                f"Unexpected node-feature dimension {data.x.size(-1)} "
                f"(expected {in_feat} raw or {self.out_node_dim} encoded)."
            )

        return data

# --------------------------------------------------------------------------- #
# Edge Encoder:
# --------------------------------------------------------------------------- #

class EdgeEncoder(nn.Module):
    covers = ("edge_embedder",)

    def __init__(self, in_dim: int, out_dim: int, scheme: str = "linear"):
        super(). __init__()
        if scheme == "identity":
            assert in_dim == out_dim
            self.embed = Identity_Transformation(out_edge_dim=out_dim)
        elif scheme == "onehot":
            self.embed = nn.Embedding(in_dim, out_dim)
        elif scheme == "linear":
            self.embed = GeoLinear(in_dim, out_dim, bias=False)
        elif scheme == "affine":
            self.embed = GeoLinear(in_dim, out_dim, bias=True)
        else:
            raise ValueError(scheme)
            
        self.raw_dim  = in_dim
        self.out_node_dim = None
        self.out_edge_dim = out_dim

    def forward(self, data: Data) -> Data:
        if getattr(data, "edge_attr", None) is None:
            return data                              # nothing to do

        if data.edge_attr.size(-1) == self.raw_dim:  # first encounter
            data.edge_attr = self.embed(data.edge_attr)
        elif data.edge_attr.size(-1) == self.out_edge_dim:
            pass                                     # already embedded
        else:
            raise RuntimeError(
                f"Unexpected edge feature dim {data.edge_attr.size(-1)} "
                f"(expected {self.raw_dim} raw or {self.out_edge_dim} encoded)."
            )
        return data


# --------------------------------------------------------------------------- #
# Structural Encoder:
# --------------------------------------------------------------------------- #
class StructuralEncoder(nn.Module):
    covers = ("structural_encoder",)

    def __init__(self, mode: str = "none"):
        super().__init__()
        self.mode = mode
        self.out_node_dim = 1 if mode != "none" else 0
        self.out_edge_dim = None

    @torch.no_grad()
    def forward(self, data: Data) -> Data:
        if self.mode == "none":
            return data
        if self.mode == "degree":
            data.pos_enc = degree(data.edge_index[0], data.num_nodes).unsqueeze(1)
            return data
        if self.mode == "shortest_path":
            N = data.num_nodes
            src, dst = data.edge_index
            dist = torch.full((N, N), float("inf"), device=src.device)
            dist.fill_diagonal_(0.0)
            dist[src, dst] = 1.0
            dist[dst, src] = 1.0
            for k in range(N):
                dist = torch.minimum(dist, dist[:, k : k + 1] + dist[k : k + 1, :])
            data.pos_enc = dist.min(dim=-1).values.unsqueeze(1)
            return data
        raise ValueError(self.mode)


# --------------------------------------------------------------------------- #
# Message function
# --------------------------------------------------------------------------- #
class EdgeMLP(MessagePassing):
    covers = ("message_fn",)

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        out_dim: int,
    ):
        super().__init__(aggr=None)
        self.mlp = nn.Sequential(
            GeoLinear(node_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            GeoLinear(hidden_dim, out_dim),
        )
        self.edge_dim = edge_dim
        self.out_edge_dim = out_dim
        self.out_node_dim = None

    def forward(self, data: Data) -> Data:
        src, dst = data.edge_index
        if data.edge_attr is None:
            E = src.size(0)
            zeros = data.x.new_zeros(E, self.edge_dim)  # safe fallback
        else:
            zeros = data.edge_attr
        cat = torch.cat([data.x[src], data.x[dst], zeros], dim=-1)
        data.edge_attr = self.mlp(cat)
        return data


# --------------------------------------------------------------------------- #
# Edge attention, local aggregator, global attention, hybrid mixer
# --------------------------------------------------------------------------- #
class EdgeAttention(nn.Module):
    covers = ("edge_attention",)

    def __init__(self, node_dim: int, heads: int = 1, p: float = 0.0):
        super().__init__()
        self.lin_src = GeoLinear(node_dim, heads, bias=False)
        self.lin_dst = GeoLinear(node_dim, heads, bias=False)
        self.dp = nn.Dropout(p)
        self.leaky = nn.LeakyReLU(0.2)
        self.out_edge_dim = node_dim
        self.out_node_dim = None

    def forward(self, data: Data) -> Data:
        h_s = self.lin_src(data.x)
        h_d = self.lin_dst(data.x)
        e = self.leaky(h_s[data.edge_index[0]] + h_d[data.edge_index[1]])
        alpha = self.dp(torch.softmax(e, dim=0)).mean(-1, keepdim=True)
        base = (
            data.edge_attr
            if getattr(data, "edge_attr", None) is not None
            else data.x[data.edge_index[0]]
        )
        data.edge_attr = base * alpha
        return data


class LocalAggregator(nn.Module):
    covers = ("local_aggregator",)

    def __init__(self, reduce: str = "mean"):
        super().__init__()
        if reduce not in _SCATTER:
            raise ValueError(reduce)
        self.reduce = reduce
        self.out_node_dim = None
        self.out_edge_dim = None

    def forward(self, data: Data) -> Data:
        src, dst = data.edge_index
        data.x = _SCATTER[self.reduce](data.edge_attr, dst, data.num_nodes)
        return data


class GlobalSelfAttention(nn.Module):
    covers = ("global_attention",)

    def __init__(self, dim: int, heads: int = 8, drop: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.out_node_dim = dim
        self.out_edge_dim = None

    def forward(self, data: Data) -> Data:
        if not hasattr(data, "batch"):
            h, _ = self.attn(data.x[None], data.x[None], data.x[None])
            data.x = h.squeeze(0)
            return data
        batch = data.batch
        sizes = torch.bincount(batch)
        G = sizes.size(0)
        maxL = int(sizes.max())
        pad = lambda t, l: F.pad(t, (0, 0, 0, maxL - l))
        xpad = torch.stack([pad(data.x[batch == g], sizes[g]) for g in range(G)])
        h, _ = self.attn(xpad, xpad, xpad)
        data.x = torch.cat([h[g, : sizes[g]] for g in range(G)], 0)
        return data


class HybridMixer(nn.Module):
    covers = ("hybrid_router",)

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.out_node_dim = None
        self.out_edge_dim = None

    def forward(
        self,
        x_local: Optional[torch.Tensor],
        x_global: Optional[torch.Tensor]
    ) -> torch.Tensor:

        # ----- pass‑through shortcuts -----------------------------------
        if x_local is None and x_global is None:
            raise ValueError("HybridMixer received no inputs.")
        if x_local is None:
            return x_global
        if x_global is None:
            return x_local

        # ----- gated blend ----------------------------------------------
        a = torch.sigmoid(self.alpha)               # scalar in (0,1)
        return a * x_local + (1.0 - a) * x_global


# --------------------------------------------------------------------------- #
# Node & edge update
# --------------------------------------------------------------------------- #
class NodeUpdate(nn.Module):
    covers = ("node_update",)

    def __init__(self, dim: int, width: int = 2, drop: float = 0.0):
        super().__init__()
        self.ff = nn.Sequential(
            GeoLinear(dim, dim * width),
            nn.GELU(),
            GeoLinear(dim * width, dim),
            nn.Dropout(drop),
        )
        self.norm = nn.LayerNorm(dim)
        self.out_node_dim = dim
        self.out_edge_dim = None

    def forward(self, data: Data) -> Data:
        data.x = self.norm(data.x + self.ff(data.x))
        return data


class EdgeUpdate(MessagePassing):
    covers = ("edge_update",)

    def __init__(self, node_dim: int, edge_dim: int, hid: int):
        super().__init__(aggr=None)
        self.mlp = nn.Sequential(
            GeoLinear(node_dim * 2 + edge_dim, hid),
            nn.ReLU(),
            GeoLinear(hid, edge_dim),
        )
        self.out_edge_dim = edge_dim
        self.out_node_dim = None

    def forward(self, data: Data) -> Data:
        src, dst = data.edge_index
        inp = torch.cat([data.x[src], data.x[dst], data.edge_attr], dim=-1)
        data.edge_attr = self.mlp(inp)
        return data


# --------------------------------------------------------------------------- #
# Readout & multiscale aggregation
# --------------------------------------------------------------------------- #
_READ = {
    "sum": global_add_pool,
    "mean": global_mean_pool,
    "max": global_max_pool,
}


class ReadoutPooler(nn.Module):
    covers = ("readout_pool",)

    def __init__(self, reduce: str = "mean", attr_name: str = "graph_feat"):
        """
        Parameters
        ----------
        reduce     : "mean" | "sum" | "max"
        attr_name  : name of the attribute that will hold
                     the pooled graph‑level feature inside `Data`.
        """
        super().__init__()
        self.fn = _READ[reduce]
        self.attr_name = attr_name

    def forward(self, data: Data) -> Data:
        if hasattr(data, "batch"):
            pooled = self.fn(data.x, data.batch)
        else:                       # single graph case
            pooled = self.fn(data.x, torch.zeros(data.num_nodes,
                                                 dtype=torch.long,
                                                 device=data.x.device))
        setattr(data, self.attr_name, pooled)   # e.g. data.graph_feat
        return data


class MultiScaleAggregator(nn.Module):
    covers = ("multiscale_agg",)

    def __init__(self, reduce: str = "mean"):
        super().__init__()
        self.pool = ReadoutPooler(reduce)           # defines attr_name

    def forward(
        self,
        data: Data,
        frag: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        # ------------------------------------------------------------------
        # 1. Ensure `data` carries the pooled graph‑level vector exactly once
        # ------------------------------------------------------------------
        if not hasattr(data, self.pool.attr_name):
            data = self.pool(data)                  # computes & attaches

        # ------------------------------------------------------------------
        # 2. Retrieve the cached vector
        # ------------------------------------------------------------------
        g = getattr(data, self.pool.attr_name)

        # ------------------------------------------------------------------
        # 3. Optionally concatenate additional fragment features
        # ------------------------------------------------------------------
        return g if frag is None else torch.cat([g, frag], dim=-1)

# --------------------------------------------------------------------------- #
# Terminal To-Vector Adaptor:
# --------------------------------------------------------------------------- #

class GraphVectorizer(nn.Module):
    """
    Final‑stage module for a stage pipeline.
    Expects `data.graph_feat` (set by ReadoutPooler or MultiScaleAggregator)
    and returns a tensor of shape  [num_graphs, out_dim].

    Parameters
    ----------
    in_dim       : Dimension of `graph_feat` coming from the readout.
    project_dim  : If not None and different from `in_dim`,
                   applies a learnable GeoLinear projection to this size.
    normalize    : If True, L2‑normalizes the output (useful for contrastive
                   learning or cosine‑similarity fusion).
    """
    covers = ("task_head",)

    def __init__(self,
                 in_dim: int,
                 project_dim: Optional[int] = None,
                 normalize: bool = False):
        super().__init__()
        self.normalize = normalize

        same_dim = project_dim is None or project_dim == in_dim
        self.proj = nn.Identity() if same_dim else GeoLinear(in_dim, project_dim)
        self.out_dim = in_dim if same_dim else project_dim

    def forward(self, data: Data) -> torch.Tensor:
        # Grab (or lazily compute) the pooled vector
        g = getattr(data, "graph_feat", None)
        if g is None:
            raise RuntimeError(
                "GraphVectorizer expects `data.graph_feat` to be populated by "
                "`ReadoutPooler` or another readout stage placed earlier in the pipeline."
            )

        g = self.proj(g)
        if self.normalize:
            g = F.normalize(g, p=2, dim=-1)
        return g

# --------------------------------------------------------------------------- #
# Community‑provided block (example)
# --------------------------------------------------------------------------- #
class GATBlock(nn.Module):
    covers = ("message_fn", "edge_attention", "local_aggregator")

    def __init__(self, in_dim: int, out_dim: int, heads: int = 8):
        super().__init__()
        self.gat = GATConv(
            in_dim,
            out_dim // heads,
            heads=heads,
            concat=True,
            add_self_loops=False,
        )
        self.out_node_dim = out_dim
        self.out_edge_dim = out_dim

    def forward(self, data: Data) -> Data:
        data.x = self.gat(data.x, data.edge_index)
        data.edge_attr = data.x[data.edge_index[0]]
        return data


# --------------------------------------------------------------------------- #
# Helpers to validate and build a stage pipeline
# --------------------------------------------------------------------------- #
class StageConfigError(ValueError):
    ...


def _validate(stage_map: Dict[str, nn.Module]) -> None:
    missing = [s for s in STAGES if s not in stage_map]
    if missing:
        raise StageConfigError(f"Missing stages {missing}")
    unknown = [k for k in stage_map if k not in STAGES]
    if unknown:
        raise StageConfigError(f"Unknown keys {unknown}")
    cov = {k: False for k in STAGES}
    for key, mod in stage_map.items():
        covers = getattr(mod, "covers", (key,))
        for s in covers:
            if cov[s]:
                raise StageConfigError(f"Stage '{s}' covered twice")
            cov[s] = True


def build_pipeline(stage_map: Dict[str, nn.Module]) -> nn.ModuleList:
    _validate(stage_map)
    return nn.ModuleList([stage_map[key] for key in STAGES])


# --------------------------------------------------------------------------- #
# Reference GNN backbone that consumes a stage map
# --------------------------------------------------------------------------- #
class GraphNeuralNetwork(nn.Module):
    """
    A sixteen‑stage graph‑to‑something backbone whose stages are both
    (i) registered as plain attributes for easy access, and
    (ii) stored in a ModuleList to keep `nn.Module` semantics happy.
    """

    def __init__(self, stage_map: Dict[str, nn.Module], seed: int = 42):
        super().__init__()
        set_global_seed(seed)

        # ------------------------------------------------------------------ #
        # 1. Validate the map (raises if a stage is missing or duplicated)
        # ------------------------------------------------------------------ #
        _validate(stage_map)            # re‑use your helper

        # ------------------------------------------------------------------ #
        # 2. ***Explicitly*** register every stage as an attribute
        # ------------------------------------------------------------------ #
        self.preprocessor       = stage_map["preprocessor"]
        self.node_embedder      = stage_map["node_embedder"]
        self.edge_embedder      = stage_map["edge_embedder"]
        self.structural_encoder = stage_map["structural_encoder"]

        self.message_fn         = stage_map["message_fn"]
        self.edge_attention     = stage_map["edge_attention"]
        self.local_aggregator   = stage_map["local_aggregator"]

        self.global_attention   = stage_map["global_attention"]
        self.hybrid_router      = stage_map["hybrid_router"]

        self.node_update        = stage_map["node_update"]
        self.edge_update        = stage_map["edge_update"]

        self.feed_forward       = stage_map["feed_forward"]
        self.norm_reg           = stage_map["norm_reg"]

        self.readout_pool       = stage_map["readout_pool"]
        self.multiscale_agg     = stage_map["multiscale_agg"]
        self.task_head          = stage_map["task_head"]

        # ------------------------------------------------------------------ #
        # 3. Keep a ModuleList in the canonical execution order
        # ------------------------------------------------------------------ #
        self.pipe = nn.ModuleList([
            self.preprocessor,
            self.node_embedder,  self.edge_embedder,  self.structural_encoder,
            self.message_fn,     self.edge_attention, self.local_aggregator,
            self.global_attention, self.hybrid_router,
            self.node_update,    self.edge_update,
            self.feed_forward,   self.norm_reg,
            self.readout_pool,   self.multiscale_agg, self.task_head,
        ])
        
        # Handy lookup used by __getitem__/__setitem__
        self._stage_to_idx = {name: idx for idx, name in enumerate(STAGES)}

        self.output_dim = getattr(self.task_head, "out_dim", None)

    # ------------------------------------------------------------------ #
    # Dict‑style *read* access:  model["readout_pool"]  or  model[13]
    # ------------------------------------------------------------------ #
    def __getitem__(self, key: Union[str, int]) -> nn.Module:
        if isinstance(key, str):                          # by stage name
            if key not in STAGES:
                raise KeyError(f"{key!r} is not a valid stage name.")
            return getattr(self, key)
        elif isinstance(key, int):                        # by position
            return self.pipe[key]
        else:
            raise TypeError("Key must be a stage name (str) or an int index.")

    # ------------------------------------------------------------------ #
    # Dict‑style *write* access:  model["readout_pool"] = new_module
    # ------------------------------------------------------------------ #
    def __setitem__(self, key: Union[str, int], value: nn.Module) -> None:
        if not isinstance(value, nn.Module):
            raise TypeError("Value must be an instance of torch.nn.Module.")

        # --- replace by stage name ---
        if isinstance(key, str):
            if key not in STAGES:
                raise KeyError(f"{key!r} is not a valid stage name.")
            setattr(self, key, value)                     # register attribute
            self.pipe[self._stage_to_idx[key]] = value    # keep execution list aligned

        # --- replace by index ---
        elif isinstance(key, int):
            if not (0 <= key < len(self.pipe)):
                raise IndexError("Stage index out of range.")
            stage_name = STAGES[key]
            setattr(self, stage_name, value)
            self.pipe[key] = value

        else:
            raise TypeError("Key must be a stage name (str) or an int index.")

        # Keep metadata in sync if the task head changes
        if (isinstance(key, str) and key == "task_head") or \
           (isinstance(key, int) and STAGES[key] == "task_head"):
            self.output_dim = getattr(value, "out_dim", None)

    # ---------------------------------------------------------------------- #
    # Forward pass – unchanged except we iterate over self.pipe
    # ---------------------------------------------------------------------- #
    def forward(self, data: Data) -> Union[Data, torch.Tensor]:
        local_x = global_x = None

        for mod in self.pipe:
            covers = getattr(mod, "covers", (None,))

            # Special case: hybrid router blends two node tensors
            if "hybrid_router" in covers:
                data.x = mod(local_x, global_x)
                continue

            # Standard stage: consume/return a Data object
            data = mod(data)

            # Cache intermediate node states if required
            if "local_aggregator" in covers:
                local_x = data.x.clone()
            if "global_attention" in covers:
                global_x = data.x.clone()

        # The last stage decides what is returned:
        #   • a Data object with predictions     → when task_head is a normal head
        #   • a tensor (embedding)               → when task_head is GraphVectorizer
        return data
    
# =========================================================================== #
# Other Core Components:
# =========================================================================== #

# ----------------------------------------------------------------------
# 1.  MinimolEncoder  – turn 512‑length fingerprint → learnable vector
# ----------------------------------------------------------------------
@register("core", "minimol_encoder")
class MinimolEncoder(QueryInterface, nn.Module):
    """
    Pass‑through or projection on top of the 512‑dim Minimol fingerprint.

    If `proj_dim is None` the raw fingerprint (optionally normalised) is
    returned.  Otherwise:  fp → Linear(512→proj_dim) → activation → dropout.
    """
    def __init__(
        self,
        proj_dim: Optional[int] = None,
        *,
        activate: str = "relu",
        dropout_p: float = 0.0,
        l2_normalise: bool = False,
    ):
        super().__init__()
        self.proj = (
            nn.Identity() if proj_dim is None
            else nn.Linear(512, proj_dim, bias=True)
        )
        self.act  = get_from_registry("core", activate)()
        self.dp   = get_from_registry("core", "dropout")(dropout_p)
        self.l2   = l2_normalise
        self.out_dim = 512 if proj_dim is None else proj_dim

    # ------------------------------------------------------------------
    # x can be:
    #   • dict   with key "minimol"
    #   • PyG Data object with attr `.minimol_representation`
    #   • plain tensor (already the fp)
    # ------------------------------------------------------------------
    def forward(self, x):
        if isinstance(x, dict) and "minimol" in x:
            fp = x["minimol"]
        elif isinstance(x, torch.Tensor):
            fp = x
        elif hasattr(x, "minimol_representation"):
            fp = x.minimol_representation
        else:
            raise TypeError("MinimolEncoder could not locate the fingerprint.")

        # 🔧 NEW — make it resilient to the B×D×1×1 shape coming from MMoE
        if fp.ndim > 2:                          # e.g. (B, 512, 1, 1)
            fp = fp.flatten(start_dim=1)         # → (B, 512)

        if fp.dtype != torch.float32:
            fp = fp.float()

        fp = self.dp(self.act(self.proj(fp)))
        if self.l2:
            fp = F.normalize(fp, p=2, dim=-1)
        return fp
    
@register("core", "concat_identity")
class ConcatIdentity(QueryInterface, nn.Identity):
    """
    A no‑op layer – the *actual* concatenation of inputs is done by
    `MoleculeModelBuilder` when `spec.target` is listed in `_CAT_KINDS`.
    """
    def __init__(self, **_):
        super().__init__()
        self.out_dim = None          # will be inferred the first time we see data

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.out_dim is None and isinstance(x, torch.Tensor):
            self.out_dim = x.size(-1)
        return x



# ----------------------------------------------------------------------
# 3.  CompressionBlock – residual bottleneck FFN
# ----------------------------------------------------------------------
@register("core", "compression_block")
class CompressionBlock(QueryInterface, nn.Module):
    """
    A residual bottleneck:  x + FFN(x) where the FFN goes through a narrower
    hidden layer.

    Hyper‑parameters
    ----------------
    in_dim : int
        Incoming feature dimension.
    bottleneck_ratio : float, optional
        Hidden size as a fraction of `in_dim` (default 0.5).  Ignored if
        `bottleneck_dim` is given.
    depth : int, default 1
        How many sequential bottleneck residual layers to stack.
    dropout_p : float
    activate : {"relu", "gelu", …}  (registry key)
    """
    def __init__(
        self,
        in_dim: int,
        *,
        bottleneck_dim: Optional[int] = None,
        bottleneck_ratio: float = 0.5,
        depth: int = 1,
        activate: str = "relu",
        dropout_p: float = 0.0,
    ):
        super().__init__()
        hid = bottleneck_dim or max(1, int(in_dim * bottleneck_ratio))
        act_cls = get_from_registry("core", activate)

        def _layer():
            return nn.Sequential(
                nn.Linear(in_dim, hid),
                act_cls(),
                nn.Dropout(dropout_p),
                nn.Linear(hid, in_dim),
                nn.Dropout(dropout_p),
            )

        self.layers = nn.ModuleList([_layer() for _ in range(depth)])
        self.norm   = nn.LayerNorm(in_dim)
        self.out_dim = in_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for ff in self.layers:
            x = x + ff(x)
            x = self.norm(x)
        return x

###############################################################################
# 1.5 Model Constructors:
###############################################################################

# --------------------------------------------------------------------------- #
# Utility: trivial linear task head (used by the GNN constructors)
# --------------------------------------------------------------------------- #
def _linear_head(in_dim: int, n_tasks: int) -> nn.Module:
    """
    Returns an nn.Linear that maps the shared representation to `n_tasks`
    logits (classification) or real‑valued outputs (regression).

    If you need per‑task softmax / sigmoid you can wrap this later in the
    LightningModule; for architecture wiring a plain Linear layer is enough.
    """
    return nn.Linear(in_dim, n_tasks)

def _validate_kwargs(builder: Callable, supplied: Dict[str, object]) -> Dict[str, object]:
    """
    Ensure the caller provided:
      • every parameter that the builder requires (no default)
      • no unexpected / miss‑spelled parameters
    Returns the final kwargs dict (including builder defaults).
    """
    sig = inspect.signature(builder)
    params = sig.parameters

    # 1) Check for unknown kwargs
    unknown = set(supplied) - set(params)
    if unknown:
        raise TypeError(f"{builder.__name__} got unexpected kwargs: {sorted(unknown)}")

    # 2) Merge caller‑supplied & defaults; detect missing required args
    final_kwargs = {}
    missing = []
    for name, p in params.items():
        if name in supplied:
            final_kwargs[name] = supplied[name]
        elif p.default is not inspect._empty:
            final_kwargs[name] = p.default
        else:
            missing.append(name)

    if missing:
        raise TypeError(f"Missing required kwargs for {builder.__name__}: {missing}")

    return final_kwargs

def construct_gcnn(*, num_node_feats: int, num_edge_feats: int,
                   hidden: int, n_tasks: int = 1) -> GraphNeuralNetwork:
    stage_map = {
        "preprocessor":      GraphPreprocessor(),
        "node_embedder":     NodeEncoder(num_node_feats, hidden),
        "edge_embedder":     EdgeEncoder(num_edge_feats, hidden, scheme="linear"),
        "structural_encoder": StructuralEncoder("none"),

        "message_fn":       EdgeMLP(hidden, hidden, hidden, hidden),
        "edge_attention":   Identity_Transformation(),
        "local_aggregator": LocalAggregator("sum"),
        "global_attention": Identity_Transformation(),
        "hybrid_router":    HybridMixer(),
        "node_update":      NodeUpdate(hidden),
        "edge_update":      EdgeUpdate(hidden, hidden, hidden),

        "feed_forward":     Identity_Transformation(),
        "norm_reg":         Identity_Transformation(),
        "readout_pool":     ReadoutPooler("sum"),
        "multiscale_agg":   Identity_Transformation(),
        "task_head":        GraphVectorizer(hidden),
    }
    return GraphNeuralNetwork(stage_map)


def construct_mpnn(*, num_node_feats: int, num_edge_feats: int,
                   hidden: int, n_tasks: int = 1) -> GraphNeuralNetwork:
    stage_map = {
        "preprocessor":      GraphPreprocessor(),
        "node_embedder":     NodeEncoder(num_node_feats, hidden),
        "edge_embedder":     EdgeEncoder(num_edge_feats, hidden),
        "structural_encoder": StructuralEncoder("none"),

        "message_fn":       EdgeMLP(hidden, hidden, hidden, hidden),
        "edge_attention":   Identity_Transformation(),
        "local_aggregator": LocalAggregator("sum"),
        "global_attention": Identity_Transformation(),
        "hybrid_router":    HybridMixer(),
        "node_update":      NodeUpdate(hidden),
        "edge_update":      Identity_Transformation(),

        "feed_forward":     Identity_Transformation(),
        "norm_reg":         Identity_Transformation(),
        "readout_pool":     ReadoutPooler("sum"),
        "multiscale_agg":   Identity_Transformation(),
        "task_head":        GraphVectorizer(hidden),
    }
    return GraphNeuralNetwork(stage_map)


def construct_dmpnn(*, num_node_feats: int, num_edge_feats: int,
                    hidden: int, n_tasks: int = 1) -> GraphNeuralNetwork:
    stage_map = {
        "preprocessor":      GraphPreprocessor(),
        "node_embedder":     NodeEncoder(num_node_feats, hidden),
        "edge_embedder":     EdgeEncoder(num_edge_feats, hidden),
        "structural_encoder": StructuralEncoder("none"),

        "message_fn":       EdgeMLP(hidden, hidden, hidden, hidden),
        "edge_attention":   Identity_Transformation(),
        "local_aggregator": LocalAggregator("sum"),
        "global_attention": Identity_Transformation(),
        "hybrid_router":    HybridMixer(),
        "node_update":      NodeUpdate(hidden),
        "edge_update":      EdgeUpdate(hidden, hidden, hidden),

        "feed_forward":     Identity_Transformation(),
        "norm_reg":         Identity_Transformation(),
        "readout_pool":     ReadoutPooler("sum"),
        "multiscale_agg":   Identity_Transformation(),
        "task_head":        GraphVectorizer(hidden),
    }
    return GraphNeuralNetwork(stage_map)


def construct_attfp(*, num_node_feats: int, num_edge_feats: int,
                    hidden: int, n_tasks: int = 1,
                    heads: int = 4, dropout: float = 0.1) -> GraphNeuralNetwork:
    stage_map = {
        "preprocessor":      GraphPreprocessor(),
        "node_embedder":     NodeEncoder(num_node_feats, hidden),
        "edge_embedder":     EdgeEncoder(num_edge_feats, hidden),
        "structural_encoder": StructuralEncoder("none"),

        "message_fn":       Identity_Transformation(),
        "edge_attention":   EdgeAttention(hidden, heads=heads, p=dropout),
        "local_aggregator": LocalAggregator("sum"),
        "global_attention": Identity_Transformation(),
        "hybrid_router":    HybridMixer(),
        "node_update":      NodeUpdate(hidden),
        "edge_update":      Identity_Transformation(),

        "feed_forward":     Identity_Transformation(),
        "norm_reg":         Identity_Transformation(),
        "readout_pool":     ReadoutPooler("sum"),
        "multiscale_agg":   Identity_Transformation(),
        "task_head":        GraphVectorizer(hidden),
    }
    return GraphNeuralNetwork(stage_map)


def construct_gin(*, num_node_feats: int,
                  hidden: int, n_tasks: int = 1) -> GraphNeuralNetwork:
    stage_map = {
        "preprocessor":      GraphPreprocessor(),
        "node_embedder":     NodeEncoder(num_node_feats, hidden),
        "edge_embedder":     Identity_Transformation(out_edge_dim=0),
        "structural_encoder": StructuralEncoder("none"),

        "message_fn":       EdgeMLP(hidden, 0, hidden, hidden),
        "edge_attention":   Identity_Transformation(),
        "local_aggregator": LocalAggregator("sum"),
        "global_attention": Identity_Transformation(),
        "hybrid_router":    HybridMixer(),
        "node_update":      NodeUpdate(hidden),
        "edge_update":      Identity_Transformation(),

        "feed_forward":     Identity_Transformation(),
        "norm_reg":         Identity_Transformation(),
        "readout_pool":     ReadoutPooler("sum"),
        "multiscale_agg":   Identity_Transformation(),
        "task_head":        GraphVectorizer(hidden),
    }
    return GraphNeuralNetwork(stage_map)


def construct_gtr(*, num_node_feats: int, num_edge_feats: int,
                  hidden: int, n_tasks: int = 1,
                  heads: int = 8, dropout: float = 0.1) -> GraphNeuralNetwork:
    stage_map = {
        "preprocessor":      GraphPreprocessor(),
        "node_embedder":     NodeEncoder(num_node_feats, hidden),
        "edge_embedder":     EdgeEncoder(num_edge_feats, hidden),
        "structural_encoder": StructuralEncoder("shortest_path"),

        "message_fn":       Identity_Transformation(),
        "edge_attention":   Identity_Transformation(),
        "local_aggregator": Identity_Transformation(),
        "global_attention": GlobalSelfAttention(hidden, heads=heads, drop=dropout),
        "hybrid_router":    HybridMixer(),
        "node_update":      NodeUpdate(hidden),
        "edge_update":      Identity_Transformation(),

        "feed_forward":     Identity_Transformation(),
        "norm_reg":         Identity_Transformation(),
        "readout_pool":     ReadoutPooler("sum"),
        "multiscale_agg":   Identity_Transformation(),
        "task_head":        GraphVectorizer(hidden),
    }
    return GraphNeuralNetwork(stage_map)


_REGISTRIES["constructor"] = {
    # canonical keys
    "gcnn":   construct_gcnn,
    "mpnn":   construct_mpnn,
    "dmpnn":  construct_dmpnn,   # ChemProp
    "chemprop": construct_dmpnn, # alias
    "attfp":  construct_attfp,
    "gin":    construct_gin,
    "gtr":    construct_gtr,
    "graphormer": construct_gtr, # alias
}

def construct_network(name: str, /, **kwargs) -> GraphNeuralNetwork:
    """
    Generic model factory.

    Parameters
    ----------
    name : str
        Key identifying the model variant (see REGISTRY).
    **kwargs
        Hyper‑parameters forwarded to the underlying builder.

    Returns
    -------
    GraphNeuralNetwork
        A fully‑initialised model instance.

    Raises
    ------
    KeyError
        If `name` is not a registered builder.
    TypeError
        On missing or unknown hyper‑parameters.
    """
    key = name.lower()
    if key not in _REGISTRIES["constructor"]:
        known = ", ".join(sorted(_REGISTRIES["constructor"]))
        raise KeyError(f"Unknown model '{name}'. Available: {known}")

    builder = _REGISTRIES["constructor"][key]
    final_kwargs = _validate_kwargs(builder, kwargs)
    return builder(**final_kwargs)

###############################################################################
# 2. Multi-task Learning:
###############################################################################

# =========================================================================== #
# Standard Architectures:
# =========================================================================== #

@register("arch", "shared_bottom")
class SharedBottomArch(HPS):
    def __init__(self,
                 task_name,
                 encoder_class,
                 decoders,
                 rep_grad: bool = False,
                 multi_input: bool = False,
                 device="cpu",
                 **kwargs):
        """
        Hard‑parameter‑sharing baseline (HPS).

        Parameters
        ----------
        task_name : list[str]
            Ordered list of task identifiers.
        encoder_class : Callable[..., torch.nn.Module]
            Backbone class used to build the *shared* encoder.
        decoders : dict[str, torch.nn.Module]
            One decoder per task.
        rep_grad : bool, default=False
            If ``True`` the shared representation tensor keeps gradients so you
            can perform representation‑level gradient surgery.
        multi_input : bool, default=False
            Set ``True`` when each task receives its own input tensor.
        device : str or torch.device, default="cpu"
            Device where the modules are allocated.
        **kwargs
            Currently unused — retained for API symmetry with other
            architectures.

        Notes
        -----
        HPS shares *all* encoder parameters; only the decoders are
        task‑specific :contentReference[oaicite:0]{index=0}.
        """
        super().__init__(task_name, encoder_class, decoders,
                         rep_grad, multi_input, device, **kwargs)


@register("arch", "cross_stitch")
class CrossStitchArch(Cross_stitch):
    def __init__(self,
                 task_name,
                 encoder_class,
                 decoders,
                 rep_grad: bool = False,
                 multi_input: bool = False,
                 device="cpu",
                 *,
                 alpha_init: float | list[float] = 0.1,
                 **kwargs):
        """
        Cross‑Stitch Network.

        Parameters
        ----------
        (common args are identical to :class:`SharedBottomArch`)
        alpha_init : float or list[float], default=0.1
            Initial value(s) for the cross‑stitch mixing coefficients ``α``.
        **kwargs
            Passed through for future proofing.

        Warnings
        --------
        * Only ResNet‑like encoders are supported.
        * Must be used with ``multi_input=False`` :contentReference[oaicite:1]{index=1}.
        """
        kwargs.setdefault("alpha_init", alpha_init)
        super().__init__(task_name, encoder_class, decoders,
                         rep_grad, multi_input, device, **kwargs)

#TODO: Implement this.
#@register("arch", "sluice")
class SluiceArch:
    def __init__(self,
                 task_name,
                 encoder_class,
                 decoders,
                 rep_grad: bool = False,
                 multi_input: bool = False,
                 device="cpu",
                 *,
                 slice: int | tuple[int, ...] = 2,
                 block_share: tuple[bool, ...] | None = None,
                 **kwargs):
        """
        Sluice Network — a latent‑sharing architecture that *learns* which
        sub‑spaces and layers to share :contentReference[oaicite:2]{index=2}.

        Parameters
        ----------
        (common args are identical to :class:`SharedBottomArch`)
        slice : int or tuple[int, ...], default=2
            Into how many channel groups (“slices”) each feature map is split.
        block_share : tuple[bool, ...], optional
            For ResNet backbones, flags deciding whether each residual block is
            shareable (``True``) or task‑private (``False``).
        **kwargs
            Additional hyper‑parameters forwarded to the base class,
            e.g. ``alpha_init`` for the slice‑mixing gates.

        Notes
        -----
        Sluice generalises Cross‑Stitch: it can learn *both* which **layers**
        and which **sub‑spaces** to share between tasks.
        """
        kwargs.update({"slice": slice})
        if block_share is not None:
            kwargs["block_share"] = block_share
        super().__init__(task_name, encoder_class, decoders,
                         rep_grad, multi_input, device, **kwargs)


# --------------------------------------------------------------------------- #
# Vector‑friendly Mixture‑of‑Experts (MMoE)
# --------------------------------------------------------------------------- #

@register("arch", "mmoe")
class MMoEArch(MMoE):
    """
    LibMTL’s MMoE expects a 4‑D “image” tensor.  This wrapper:

        • *automatically* reshapes any 2‑D feature tensor `(B, D)` to
          `(B, D, 1, 1)` **before** LibMTL sees it;

        • runs the standard MMoE logic (experts, gates, decoders); and

        • squeezes the spatial dummy dimensions **afterwards** so that every
          task head once again receives / returns a flat `(B, D_out)` tensor.

    The outside world therefore keeps the clean **vector → vector** invariant.
    """

    # ------------- construction ------------------------------------------
    def __init__(self,
                 task_name,
                 encoder_class,
                 decoders,
                 rep_grad: bool = False,
                 multi_input: bool = False,
                 device="cpu",
                 *,
                 img_size: tuple[int, int, int] | None = None,
                 num_experts: int = 4,
                 **kwargs):
        
        if isinstance(num_experts, int):
            num_experts = [num_experts]

        # If the caller did not pass img_size we infer **C** (= feature dim)
        # from the *first* decoder; H,W are set to 1.
        if img_size is None:
            # grab the first decoder just to discover its input dimension
            sample_dec = next(iter(decoders.values()))
            in_dim = getattr(sample_dec, "in_dim", None)
            if in_dim is None:
                raise ValueError(
                    "`img_size` could not be inferred automatically because "
                    "the decoder does not expose `.in_dim`.  Pass it manually."
                )
            img_size = (in_dim, 1, 1)

        kwargs.update({"img_size": img_size, "num_experts": num_experts})
        super().__init__(task_name, encoder_class, decoders,
                         rep_grad, multi_input, device, **kwargs)

    # ------------- private helpers ---------------------------------------
    @staticmethod
    def _vec2img(x: torch.Tensor):
        # only promote *exactly* 2‑D feature matrices, leave everything else intact
        return x.unsqueeze(-1).unsqueeze(-1) if (isinstance(x, torch.Tensor) and x.ndim == 2) else x

    @staticmethod
    def _img2vec(x: torch.Tensor) -> torch.Tensor:
        """Inverse of `_vec2img` for a `(B, D, 1, 1)` tensor."""
        return x.view(x.size(0), -1) if x.ndim == 4 and x.size(2) == 1 else x

    # ------------- forward ------------------------------------------------
    def forward(self, x):
        """
        Accepts *either* a single tensor or a mapping `{task -> tensor}` if
        `multi_input=True`, exactly like the parent class – but vectors are
        silently upgraded to pseudo‑images.
        """
        
        # >>> NEW: make sure each expert gets an independent copy
        if isinstance(x, dict):
            # dict style (our hybrid encoder): deep‑copy every value
            x = {k: copy.deepcopy(v) for k, v in x.items()}
        else:                # graph or tensor
            x = copy.deepcopy(x)
        # <<< -----------------------------------------------

        
        # ---- reshape the *inputs* --------------------------------------
        if isinstance(x, dict):
            x = {t: self._vec2img(v) for t, v in x.items()}
        else:
            x = self._vec2img(x)

        # ---- call the LibMTL implementation ---------------------------
        out = super().forward(x)              # dict {task -> tensor}

        # ---- bring every task back to flat (B, D_out) ------------------
        for t, v in out.items():
            out[t] = self._img2vec(v)

        return out

# =========================================================================== #
# Non-Standard Architectures:
# =========================================================================== #

@register("arch", "identity")
class IdentityArch(HPS):
    """
    Thin wrapper that *does nothing* beyond forwarding a single shared encoder
    to independent task heads.  Functionally identical to hard sharing (HPS)
    but separated for clarity in the search‑space.
    """
    def __init__(
        self,
        task_name,
        encoder_class,
        decoders,
        rep_grad: bool = False,
        multi_input: bool = False,
        device="cpu",
        **kwargs,
    ):
        super().__init__(task_name, encoder_class, decoders,
                         rep_grad, multi_input, device, **kwargs)

###############################################################################
# 3. Architectures:
###############################################################################

# --------------------------------------------------------------------------- #
# Base helper (not registered)
# --------------------------------------------------------------------------- #
class _BaseBuilder:
    """
    Base class that factorises the common steps:

        • look up encoder / head classes
        • build the per‑task decoders
        • call the LibMTL wrapper
    """

    arch_key: str                 # override in subclasses

    # ---- public API ---------------------------------------------------------
    def __init__(
        self,
        task_names: List[str],
        *,
        encoder_name: str,
        encoder_kwargs: Dict = None,
        head_name: str,
        head_kwargs: Dict = None,
        rep_grad: bool = False,
        multi_input: bool = False,
        device: str | torch.device = "cpu",
        arch_kwargs: Dict = None,
    ):
        self.task_names   = task_names
        self.encoder_name = encoder_name
        self.encoder_kwargs = encoder_kwargs or {}
        self.head_name    = head_name
        self.head_kwargs  = head_kwargs  or {}
        self.rep_grad     = rep_grad
        self.multi_input  = multi_input
        self.device       = device
        self.arch_kwargs  = arch_kwargs  or {}

    # ---- factory ------------------------------------------------------------
    def build(self) -> nn.Module:
        
        # TODO: Remove this part of code and fix registry problem.
        
        #######################################################################
        
        if callable(self.encoder_name):            # already a class / factory
            encoder_cls: Type[nn.Module] = self.encoder_name
        else:                                      # legacy path: registry key
            encoder_cls = get_core(self.encoder_name)
        
        #######################################################################

        # 2. decoders per task
        head_cls: Type[nn.Module]    = get_head(self.head_name)
        # We defer `in_dim` inference to caller via head_kwargs
        decoders = {
            t: head_cls(task=t, **self.head_kwargs) for t in self.task_names
        }

        # 3. LibMTL architecture wrapper
        ArchWrapper: Type[nn.Module] = get_arch(self.arch_key)

        return ArchWrapper(
            self.task_names,
            encoder_cls,                 # <‑ only *class*, LibMTL instantiates
            decoders,
            rep_grad   = self.rep_grad,
            multi_input= self.multi_input,
            device     = self.device,
            **self.arch_kwargs,
        )

# --------------------------------------------------------------------------- #
# Concrete builders – each sets arch_key and can add extra arguments later
# --------------------------------------------------------------------------- #
class SharedBottomBuilder(_BaseBuilder):
    """Builder for classic hard‑parameter sharing (LibMTL.HPS)."""
    arch_key = "shared_bottom"


class CrossStitchBuilder(_BaseBuilder):
    """Builder for Cross‑Stitch Networks."""
    arch_key = "cross_stitch"


class SluiceBuilder(_BaseBuilder):
    """Builder for Sluice Networks."""
    arch_key = "sluice"


class MMoEBuilder(_BaseBuilder):
    """Builder for Multi‑gate Mixture of Experts."""
    arch_key = "mmoe"
    
class IdentityBuilder(_BaseBuilder):
    """Builder that selects the IdentityArch (no MTL sharing)."""
    arch_key = "identity"


###############################################################################
# 4.  Weighting  – task‑level gradient / loss balancers
###############################################################################

class AbsWeighting(abc.ABC):
    """
    Base API: subclasses *must* implement ``backward(task_losses)``.
    """

    def __init__(self, shared_params: Sequence[nn.Parameter], **cfg):
        self.shared_params = list(shared_params)  # keep for LibMTL wrappers

    @abc.abstractmethod
    def backward(self, task_losses: List[torch.Tensor]) -> torch.Tensor: ...


@register("weighting", "none")
class IdentityWeighting(AbsWeighting):
    """Simply sums losses and back‑propagates."""

    def backward(self, task_losses: List[torch.Tensor]) -> torch.Tensor:
        total = torch.stack(task_losses, 0).mean()
        total.backward()
        return total


@register("weighting", "pcgrad")
class PCGradWeighting(AbsWeighting):
    """
    Wrapper around LibMTL's PCGrad implementation.
    """

    def __init__(self, shared_params, reduction: str = "mean", **kw):
        super().__init__(shared_params)
        self.pcgrad = _LibPCGrad(params=shared_params, reduction=reduction)

    def backward(self, task_losses: List[torch.Tensor]) -> torch.Tensor:
        total = self.pcgrad(task_losses)             # internally calls .backward()
        return total


def build_weighting(cfg: Dict, shared_params) -> AbsWeighting:
    """
    Helper that converts a YAML / dict section into an instantiated weighting
    object, e.g.

        weighting:
            name: pcgrad
            kwargs: { reduction: mean }
    """
    cls = get_from_registry("weighting", cfg.get("name", "none"))
    return cls(shared_params, **cfg.get("kwargs", {}))

###############################################################################
# 5.  Loss functions – atomic criteria, adapters, stack
###############################################################################
# ---------- atomic criteria --------------------------------------------------
@register("loss", "bce_with_logits")
def _bce(pred, target, **_):
    target = target.float().squeeze(-1)         # BCE expects float targets
    return F.binary_cross_entropy_with_logits(pred.squeeze(-1), target)

@register("loss", "cross_entropy")
def _ce(pred: torch.Tensor, target: torch.Tensor, **_):
    return F.cross_entropy(pred, target)

@register("loss", "mse")
def _mse(pred: torch.Tensor, target: torch.Tensor, **_):
    return F.mse_loss(pred, target)

@register("loss", "l1_penalty")
def _l1_penalty(model: nn.Module, **_):
    vec = torch.cat([p.view(-1) for p in model.parameters() if p.requires_grad])
    return vec.abs().sum()

@register("loss", "l2_penalty")
def _l2_penalty(model: nn.Module, **_):
    vec = torch.cat([p.view(-1) for p in model.parameters() if p.requires_grad])
    return (vec ** 2).sum()

# ---------- adapter & stack --------------------------------------------------
@dataclass
class LossAdapter:
    name: str
    weight: float
    fn: Callable          # atomic criterion
    task: str | None = None

    def __call__(
        self,
        model: nn.Module,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        batch,
        epoch: int,
    ) -> torch.Tensor:
        pred_sel   = preds.get(self.task)   if self.task else preds
        target_sel = targets.get(self.task) if self.task else targets
        raw = self.fn(model=model, pred=pred_sel, target=target_sel,
                      preds=preds, targets=targets, batch=batch, epoch=epoch)
        return self.weight * raw


class LossStack(nn.Module):
    """
    Evaluates every LossAdapter, logs individual values, returns weighted sum.
    """

    def __init__(self, adapters: List[LossAdapter]):
        super().__init__()
        self.adapters = adapters

    def forward(
        self,
        *,
        model: nn.Module,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        batch,
        epoch: int,
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        total = torch.zeros((), device=next(model.parameters()).device)
        for term in self.adapters:
            val = term(model, preds, targets, batch, epoch)
            out[term.name] = val.detach()
            total += val
        out["total"] = total
        return out


def build_loss_stack(cfg_list: List[Dict]) -> LossStack:
    """
    Parse a YAML section such as

        losses:
          - {name: cross_entropy, task: taskA, weight: 1.0}
          - {name: cross_entropy, task: taskB, weight: 1.0}
          - {name: l2_penalty,    weight: 1e-4}

    into a LossStack instance.
    """
    adapters: List[LossAdapter] = []
    for item in cfg_list:
        fn = get_from_registry("loss", item["name"])
        adapters.append(
            LossAdapter(
                name   = item["name"],
                task   = item.get("task"),
                weight = item.get("weight", 1.0),
                fn     = fn,
            )
        )
    return LossStack(adapters)

###############################################################################
# 6.  Optimisers & schedulers – builder pattern
###############################################################################
@register("optim", "adamw")
class AdamWBuilder:
    def __init__(self, lr: float = 3e-4, weight_decay: float = 0.0, **kw):
        self.kw = dict(lr=lr, weight_decay=weight_decay, **kw)

    def __call__(self, params) -> Tuple[torch.optim.Optimizer, None]:
        return torch.optim.AdamW(params, **self.kw), None


@register("optim", "sgd")
class SGDBuilder:
    def __init__(self, lr: float = 1e-1, momentum: float = 0.9, **kw):
        self.kw = dict(lr=lr, momentum=momentum, **kw)

    def __call__(self, params) -> Tuple[torch.optim.Optimizer, None]:
        return torch.optim.SGD(params, **self.kw), None


def build_optim(cfg: Dict, params):
    """
    Example cfg:

        optim:
          name: adamw
          kwargs: { lr: 1e-4, weight_decay: 1e-2 }
    """
    cls = get_from_registry("optim", cfg["name"])
    builder = cls(**cfg.get("kwargs", {}))
    return builder(params)      # → (optimizer, scheduler | None)

###############################################################################
# 7.  Metrics – functional + manager wrapper
###############################################################################


@register("metric", "accuracy")
def _accuracy(pred: torch.Tensor, target: torch.Tensor):
    """
    TorchMetrics ≥ 1.0 requires an explicit `task=` argument.

    * Binary   → task="binary"
    * ≥ 2‑class→ task="multiclass", num_classes=C
    """
    if _tm_acc is None:                      # fallback path – metric unavailable
        return (pred.argmax(-1) == target).float().mean()

    # -------- decide task type ---------------------------------------------
    # Shape (N,) or (N,1)    → binary logits/probs
    # Shape (N,C), C ≥ 2     → multiclass logits/probs
    if pred.ndim == 1 or (pred.ndim == 2 and pred.size(-1) == 1):
        # binary‑logit input → convert to probability with sigmoid
        return _tm_acc(pred.sigmoid(), target.int(), task="binary")
    else:
        C = pred.size(-1)
        return _tm_acc(pred.softmax(-1), target.int(),
                       task="multiclass", num_classes=C)

@register("metric", "rmse")
def _rmse(pred: torch.Tensor, target: torch.Tensor):
    return torch.sqrt(F.mse_loss(pred.float(), target.float()))



@register("metric", "auroc")
def _auroc(pred: torch.Tensor, target: torch.Tensor):
    """
    Works for binary and multiclass just like `_accuracy` helper.
    Returns AUROC in [0, 1].
    """
    # Decide task type from shape, then forward to torchmetrics
    if pred.ndim == 1 or (pred.ndim == 2 and pred.size(-1) == 1):
        return _tm_auroc(pred.sigmoid(), target.int(), task="binary")
    C = pred.size(-1)
    return _tm_auroc(pred.softmax(-1), target.int(),
                     task="multiclass", num_classes=C)

@register("metric", "auprc")
def _auprc(pred: torch.Tensor, target: torch.Tensor):
    """
    Area‑under‑precision‑recall curve.
    """
    if pred.ndim == 1 or (pred.ndim == 2 and pred.size(-1) == 1):
        return _tm_ap(pred.sigmoid(), target.int(), task="binary")
    C = pred.size(-1)
    return _tm_ap(pred.softmax(-1), target.int(),
                  task="multiclass", num_classes=C)


class MetricAdapter:
    def __init__(self, name: str, fn: Callable, task: str | None = None):
        self.name, self.fn, self.task = name, fn, task
        self.value = 0.0
        self.count = 0

    def update(self, preds, targets):
        p = preds.get(self.task)   if self.task else preds
        t = targets.get(self.task) if self.task else targets
        v = self.fn(p, t).item()
        self.value += v
        self.count += 1

    def compute(self):
        return self.value / max(self.count, 1)


class MetricManager:
    def __init__(self, adapters: List[MetricAdapter]):
        self.adapters = adapters

    def update(self, preds, targets):
        for m in self.adapters:
            m.update(preds, targets)

    def compute(self) -> Dict[str, float]:
        return {m.name if m.task is None else f"{m.task}_{m.name}": m.compute()
                for m in self.adapters}


def build_metrics(cfg_list: List[Dict]) -> MetricManager:
    """
    cfg example:

        metrics:
          - { name: accuracy, task: taskA }
          - { name: accuracy, task: taskB }
          - { name: rmse }
    """
    adapters: List[MetricAdapter] = []
    for item in cfg_list:
        fn = get_from_registry("metric", item["name"])
        adapters.append(MetricAdapter(item["name"], fn, item.get("task")))
    return MetricManager(adapters)

###############################################################################
# 8. PyTorch Lightning:
###############################################################################

@register("ptl", "single_default")
class SingleTaskLightningModule(pl.LightningModule):
    """
    Minimal LightningModule for *standard* nn.Module models (one task).

    • Always uses automatic optimisation (no gradient surgery needed).  
    • Internally up‑casts the model’s tensor prediction → a `{task: tensor}`
      dict so it can share the very same LossStack / MetricManager helpers.
    • **NEW**: optional early stopping driven purely by the experiment config.
    """

    # .......................................................................
    # Construction
    # .......................................................................
    def __init__(self, cfg: Dict):
        """
        Additional cfg key
        ------------------
        early_stopping (optional) ::
            { monitor: val_loss, mode: min, patience: 10, min_delta: 0.0 }
        """
        super().__init__()
        self.save_hyperparameters(cfg)

        # --‑ model -------------------------------------------------------
        model_obj = cfg["model"]
        self.model: nn.Module = model_obj() if callable(model_obj) else model_obj
        self.task_name: str   = cfg.get("task_name", "main")

        # --‑ helpers -----------------------------------------------------
        losses_cfg = [{**item, "task": self.task_name} for item in cfg["losses"]]
        self.loss_stack  = build_loss_stack(losses_cfg)
        self.weighting   = IdentityWeighting(self.model.parameters())
        self._optim_cfg  = cfg["optim"]

        self.metric_mgr: MetricManager | None = None
        if cfg.get("metrics"):
            metrics_cfg = [{**m, "task": self.task_name} for m in cfg["metrics"]]
            self.metric_mgr = build_metrics(metrics_cfg)

        # --‑ early stopping ---------------------------------------------
        self._es_cfg: Dict | None = cfg.get("early_stopping")   # ➋ NEW

    # .......................................................................
    # Lightning plumbing
    # .......................................................................
    def configure_callbacks(self):
        """
        If *early_stopping* is present in the config, create and return a
        `pytorch_lightning.callbacks.EarlyStopping` instance.  Otherwise let
        Lightning proceed with the default (empty) callback list.
        """
        if not self._es_cfg:                # section absent ⇒ nothing to add
            return []

        es = EarlyStopping(                 # defaults follow Lightning’s API
            monitor   = self._es_cfg.get("monitor",   "val_loss"),
            mode      = self._es_cfg.get("mode",      "min"),
            patience  = self._es_cfg.get("patience",  10),
            min_delta = self._es_cfg.get("min_delta", 0.0),
            verbose   = bool(self._es_cfg.get("verbose", False)),
        )
        return [es]
    
    # ---- utility -----------------------------------------------------------
    def _wrap_preds(self, preds: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convert tensor → dict{task_name: tensor} for shared helpers."""
        return {self.task_name: preds}

    # ---- Lightning standard hooks -----------------------------------------
    def configure_optimizers(self):
        opt, sched = build_optim(self._optim_cfg, self.parameters())
        return (opt, sched) if sched else opt

    def forward(self, inputs):
        return self._wrap_preds(self.model(inputs))        # dict‑style

    def _shared_step(self, batch, stage: str):
        # unpack   (tuple style → (x, y)  •  dict style → {"inputs":..,"targets":..})
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            x, y = batch
        elif isinstance(batch, dict):
            x, y = batch["inputs"], batch["targets"]
        elif isinstance(batch, GeoBatch):                     # ← new
            x, y = batch, batch.y
        else:
            raise TypeError(f"Unsupported batch type {type(batch)}")

        # ─── NEW ─── make single‑task targets dict‑shaped
        if not isinstance(y, dict):                 # tensor -> wrap
            y = {self.hparams["arch"]["kwargs"]["task_names"][0]: y}
        # ──────────

        preds = self(x)                                    # dict

        # -------- losses -------------------------------------------
        loss_dict = self.loss_stack(
            model   = self.model,
            preds   = preds,
            targets = {self.task_name: y},
            batch   = batch,
            epoch   = self.current_epoch,
        )
        total_loss = loss_dict["total"]
        
        if isinstance(self.weighting, IdentityWeighting):          # ⚙ automatic
        
            self.log(f"{stage}_loss", total_loss, prog_bar=True,
                     on_step=False, on_epoch=True, batch_size=len(x))
            return total_loss            # <<< keep the graph intact!
        
        else:

            opt = self.optimizers()
            opt.zero_grad()
            task_losses = [v for k, v in loss_dict.items() if k != "total"]
            total_loss = self.weighting.backward(task_losses)   # does .backward()
            opt.step()
            self.log(f"{stage}_loss", total_loss.detach(), prog_bar=True,
                     on_step=False, on_epoch=True, batch_size=len(x))
            return total_loss.detach()     # safely detached here
        
        self.log(f"{stage}_loss", total_loss, prog_bar=True,
                 on_step=False, on_epoch=True, batch_size=len(x))

        # -------- metrics ------------------------------------------
        if self.metric_mgr and stage != "train":
            self.metric_mgr.update(preds, {self.task_name: y})

        # optional per‑term logging
        for k, v in loss_dict.items():
            self.log(f"{stage}_{k}", v, prog_bar=False,
                     on_step=False, on_epoch=True, batch_size=len(x))

        return total_loss

    # ---- standard Lightning entry points ----------------------------------
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    # ---- epoch‑end hooks for metrics --------------------------------------
    def on_validation_epoch_end(self):
        if self.metric_mgr:
            logs = self.metric_mgr.compute()
            for k, v in logs.items():
                self.log(f"val_{k}", v, prog_bar=True)
            # reset internal counters
            self.metric_mgr = MetricManager(self.metric_mgr.adapters)

    def on_test_epoch_end(self):
        if self.metric_mgr:
            logs = self.metric_mgr.compute()
            for k, v in logs.items():
                self.log(f"test_{k}", v, prog_bar=True)

@register("ptl", "multitask_default")
class MultiTaskLightningModule(pl.LightningModule):
    """
    End‑to‑end trainer‑agnostic LightningModule that glues together:

        • architecture  (LibMTL‑based, built via one of the *Builder* classes)
        • loss stack    (Section 5)
        • task weighting (Section 4 – PCGrad etc.; optional)
        • optimiser / LR‑scheduler (Section 6)
        • metrics       (Section 7)

    *Everything* it needs comes from a single **config dict** passed at init.
    """

    ############### -----------------------------------------------------------
    # Construction helpers
    ############### -----------------------------------------------------------
    @staticmethod
    def _build_arch(cfg: Dict) -> nn.Module:
        """
        Expecting::

            arch:
              builder: shared_bottom            # one of the *_Builder classes
              kwargs:  {...}                    # forwarded to the builder
        """
        name   = cfg["builder"]
        b_cls  = {
            "shared_bottom": SharedBottomBuilder,
            "cross_stitch":  CrossStitchBuilder,
            #"sluice":        SluiceBuilder,
            "mmoe":          MMoEBuilder,
            "identity":      IdentityBuilder,
        }[name]
        builder = b_cls(**cfg["kwargs"])
        return builder.build()

    @staticmethod
    def _maybe_build_weighting(cfg: Optional[Dict], shared_params) -> AbsWeighting:
        """
        If *cfg* is absent or {'name':'none'} → IdentityWeighting.
        """
        if cfg is None:
            cfg = {"name": "none", "kwargs": {}}
        cls = get_from_registry("weighting", cfg.get("name", "none"))
        return cls(shared_params, **cfg.get("kwargs", {}))

    ############### -----------------------------------------------------------
    # Public API
    ############### -----------------------------------------------------------
    def __init__(self, cfg: Dict):
        """
        Accepts the same optional ``early_stopping`` section as the
        single‑task variant, e.g.
    
        early_stopping:
            monitor: val_loss
            mode: min
            patience: 8
            min_delta: 0.001
        """
        # ---------------------------------------------------------
        # 1.  Keep task names *outside* hparams (will be needed
        #     at run time even after Lightning strips hparams).
        # ---------------------------------------------------------
        task_names = cfg["arch"]["kwargs"]["task_names"]
        self.task_names = task_names          # e.g. ["hit"]
        self.primary_task = task_names[0]

        # ---------------------------------------------------------
        # 2.  Save only the picklable part of the config
        # ---------------------------------------------------------
        cfg_for_save = {k: v for k, v in cfg.items() if k != "arch"}
        super().__init__()
        self.save_hyperparameters(cfg_for_save)
    
        # --‑ architecture / helpers -------------------------------------
        self.arch         = self._build_arch(cfg["arch"])
        self.loss_stack   = build_loss_stack(cfg["losses"])
        self.weighting    = self._maybe_build_weighting(
                                cfg.get("weighting"), self.arch.parameters())
        self._optim_cfg   = cfg["optim"]
    
        self.metric_mgr: MetricManager | None = None
        if cfg.get("metrics"):
            self.metric_mgr = build_metrics(cfg["metrics"])
    
        self.automatic_optimization = isinstance(
            self.weighting, IdentityWeighting)
    
        # --‑ early stopping ---------------------------------------------
        self._es_cfg: Dict | None = cfg.get("early_stopping")   # ➋ NEW
    
    # .......................................................................
    # Lightning plumbing
    # .......................................................................
    def configure_callbacks(self):
        """
        Inject EarlyStopping when requested in the config.
        """
        if not self._es_cfg:
            return []
    
        es = EarlyStopping(
            monitor   = self._es_cfg.get("monitor",   "val_loss"),
            mode      = self._es_cfg.get("mode",      "min"),
            patience  = self._es_cfg.get("patience",  10),
            min_delta = self._es_cfg.get("min_delta", 0.0),
            verbose   = bool(self._es_cfg.get("verbose", False)),
        )
        return [es]


    ############### -----------------------------------------------------------
    # Lightning hooks
    ############### -----------------------------------------------------------
    # ---- optimiser / scheduler ---------------------------------------------
    def configure_optimizers(self):
        opt, sched = build_optim(self._optim_cfg, self.parameters())
        return (opt, sched) if sched else opt

    # ---- forward pass – simply proxy to architecture -----------------------
    def forward(self, inputs):
        """
        *Contract*: **returns a dict**  {task_name → prediction tensor}
        """
        return self.arch(inputs)

    # ---- shared step (train / val / test) ----------------------------------
    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        """
        Handles *one* forward / loss / metric pass.

        Returns
        -------
        torch.Tensor
            The (scalar) total loss – used for logging.
        """
        # ---------------------------------------------------------------
        # Unpack batch  – we now **enforce** dict‑style for all new code.
        # ---------------------------------------------------------------
        if not isinstance(batch, dict) or \
           "inputs" not in batch or "targets" not in batch:
            raise TypeError(
                "Expected batch to be a dict with keys {'inputs', 'targets'} "
                f"but received {type(batch)}."
            )

        x, y = batch["inputs"], batch["targets"]
        # ---- single‑task convenience ---------------------------------------
        if not isinstance(y, dict):                # tensor → dict
            y = {self.primary_task: y}
        
        # # Allow both tuple and dict‑style batches
        # if isinstance(batch, (tuple, list)) and len(batch) == 2:
        #     x, y = batch
        # else:  # assume mapping style
        #     x, y = batch["inputs"], batch["targets"]

        # Forward
        preds = self(x)

        # ---- losses ----------------------------------------------------
        loss_dict = self.loss_stack(
            model   = self.arch,
            preds   = preds,
            targets = y,
            batch   = batch,
            epoch   = self.current_epoch,
        )
        total_loss = loss_dict["total"]

        # ---- weighting & backward -------------------------------------
        if isinstance(self.weighting, IdentityWeighting):
            # automatic optimisation path
            self.log(f"{stage}_loss", total_loss, prog_bar=True, on_step=False,
                     on_epoch=True, batch_size=len(x))
            return total_loss
        else:
            # manual optimisation + gradient surgery path
            opt = self.optimizers()
            opt.zero_grad()
            # collect per‑task losses (assume they are first‑level keys that
            # include a task name)
            task_losses = [v for k, v in loss_dict.items()
                           if k != "total"]
            total_loss = self.weighting.backward(task_losses)
            opt.step()
            self.log(f"{stage}_loss", total_loss.detach(), prog_bar=True,
                     on_step=False, on_epoch=True, batch_size=len(x))

        # ---- metrics ---------------------------------------------------
        if self.metric_mgr and stage != "train":   # usually track on val/test
            self.metric_mgr.update(preds, y)

        # individual term logging (optional)
        for k, v in loss_dict.items():
            self.log(f"{stage}_{k}", v, prog_bar=False,
                     on_step=False, on_epoch=True, batch_size=len(x))

        return total_loss.detach()

    # ---- training / validation / test steps -------------------------------
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    # ---- epoch‑end hooks for metrics --------------------------------------
    def on_validation_epoch_end(self):
        if self.metric_mgr:
            logs = self.metric_mgr.compute()
            for k, v in logs.items():
                self.log(f"val_{k}", v, prog_bar=True)
        self.metric_mgr and setattr(self, "metric_mgr", self.metric_mgr.__class__(
            self.metric_mgr.adapters))  # reset counts

    def on_test_epoch_end(self):
        if self.metric_mgr:
            logs = self.metric_mgr.compute()
            for k, v in logs.items():
                self.log(f"test_{k}", v, prog_bar=True)
                

###############################################################################
# 9. Dataset Manipulation:
###############################################################################

# =============================================================================
# Train-Test Split:
# =============================================================================

def TrainTestSplit(
    tbl: Table,
    *,
    train_frac: float = 0.8,
    val_frac:   float = 0.1,
    test_frac:  float = 0.1,
    seed: int | None = None,
) -> Tuple[Table, Table, Table]:
    """
    Randomly split *tbl* into train / val / test **without materialising**
    the whole data set on a single worker.

    Parameters
    ----------
    tbl        : Table
        Source data set.
    train_frac : float, default 0.8
    val_frac   : float, default 0.1
    test_frac  : float, default 0.1
        Fractions must be non‑negative and sum to **1.0** (within 1e‑6).
    seed       : int | None
        Deterministic RNG seed.

    Returns
    -------
    (train_tbl, val_tbl, test_tbl)  – each a **new** `Table` instance.

    Notes
    -----
    * For **Ray Dataset** the function uses `random_shuffle` + `limit/skip`,
      which streams efficiently.
    * For **Dask DataFrame** it delegates to `ddf.random_split`.
    * For **Pandas** it shuffles in‑memory with `sample(frac=1.0)`.
    * The original `tbl` is **left untouched**.
    """
    # ---- validate fractions --------------------------------------------
    fracs = [train_frac, val_frac, test_frac]
    if any(f < 0 for f in fracs):
        raise ValueError("Fractions must be non‑negative.")
    if not abs(sum(fracs) - 1.0) < 1e-6:
        raise ValueError(f"Fractions must sum to 1.0, got {fracs}")

    tr_f, va_f, te_f = fracs

    # --------------------------- Ray branch -----------------------------
    if tbl.is_ray():
        
        ds = tbl.table.random_shuffle(seed=seed)
    
        n_total = ds.count()
        n_train = int(train_frac * n_total)
        n_val   = int(val_frac   * n_total)
    
        # split_at_indices returns exactly three Ray Datasets
        train_ds, val_ds, test_ds = ds.split_at_indices(
            [n_train, n_train + n_val]
        )
    
        return (
            Table(train_ds, smiles_column_name=tbl.smiles_column_name),
            Table(val_ds,   smiles_column_name=tbl.smiles_column_name),
            Table(test_ds,  smiles_column_name=tbl.smiles_column_name),
        )

    # -------------------------- Dask branch -----------------------------
    if tbl.is_dask():
        ddf_train, ddf_val, ddf_test = tbl.table.random_split(
            [tr_f, va_f, te_f], random_state=seed
        )
        return (
            Table(ddf_train, smiles_column_name=tbl.smiles_column_name),
            Table(ddf_val,   smiles_column_name=tbl.smiles_column_name),
            Table(ddf_test,  smiles_column_name=tbl.smiles_column_name),
        )

    # ------------------------- Pandas branch ----------------------------
    if tbl.is_pandas():
        pdf = tbl.table.sample(frac=1.0, random_state=seed)   # full shuffle
        n_total = len(pdf)
        n_train = int(tr_f * n_total)
        n_val   = int(va_f * n_total)
    
        train_df = pdf.iloc[:n_train]
        val_df   = pdf.iloc[n_train : n_train + n_val]
        test_df  = pdf.iloc[n_train + n_val :]
    
        return (
            Table(train_df, smiles_column_name=tbl.smiles_column_name),
            Table(val_df,   smiles_column_name=tbl.smiles_column_name),
            Table(test_df,  smiles_column_name=tbl.smiles_column_name),
        )

###############################################################################
# 10.  Model Cofnig Parser:
###############################################################################

# --------------------------------------------------------------------------- #
# 0.  Constants & small helpers
# --------------------------------------------------------------------------- #
_RESERVED_SOURCES = {"graph", "minimol"}
_CAT_KINDS = {"core.concat_identity"}     # ← replace old entry

                                                     # concatenate along dim −1


def _split_target(t: str) -> tuple[str, str]:
    """
    "core.minimol_encoder"   →  ("core", "minimol_encoder")
    """
    if "." not in t:
        raise ValueError(f"Target '{t}' must contain exactly one '.'")
    return t.split(".", 1)


def _lookup_module(kind: str, name: str, kwargs: dict):
    """
    Registry‑aware factory that instantiates *one* block.

    Supports:
        core.*,   head.*,   arch.*,   weighting.*,   metric.*
        constructor.*       (GraphNeuralNetwork builders)
    """
    if kind == "constructor":
        return construct_network(name, **kwargs)

    if kind not in _REGISTRIES:
        raise KeyError(f"Unknown registry namespace '{kind}' for target "
                       f"'{kind}.{name}'")

    cls_or_fn = get_from_registry(kind, name)
    return cls_or_fn(**kwargs) if callable(cls_or_fn) else cls_or_fn


# --------------------------------------------------------------------------- #
# 1.  NodeSpec – canonical in‑memory representation of ONE YAML stanza
# --------------------------------------------------------------------------- #
@dataclass
class NodeSpec:
    name:   str
    target: str
    inputs: list[str] = field(default_factory=list)
    kwargs: dict      = field(default_factory=dict)
    produces_tasks: list[str] | None = None            # optional

    # ---- construction helpers ---------------------------------------------
    @classmethod
    def from_dict(cls, d: dict, *, idx: int):
        required = {"name", "target", "inputs"}
        missing  = required - set(d)
        if missing:
            raise ValueError(f"Node #{idx}: missing required fields {missing}")
        return cls(
            name   = d["name"],
            target = d["target"],
            inputs = list(d["inputs"]),
            kwargs = d.get("kwargs", {}),
            produces_tasks = d.get("produces_tasks"),
        )


# --------------------------------------------------------------------------- #
# 2.  MoleculeModelBuilder – top‑level user API
# --------------------------------------------------------------------------- #

class MoleculeModelBuilder:
    """
    Generic **DAG‑to‑nn.Module** composer.

    Usage
    -----
    >>> builder = MoleculeModelBuilder(cfg["vector_pipeline"])
    >>> model   = builder.build()
    """

    # .......................................................................
    def __init__(
        self,
        node_cfg: list[dict],
        *,
        reserved_sources: set[str] | None = None,
        auto_project: bool = True,
    ):
        self._reserved   = set(reserved_sources or _RESERVED_SOURCES)
        self._auto_proj  = auto_project
        self.specs: dict[str, NodeSpec] = {}

        # ---- 1. parse & basic validation --------------------------------
        for idx, raw in enumerate(node_cfg):
            spec = NodeSpec.from_dict(raw, idx=idx)
            if spec.name in self._reserved:
                raise ValueError(f"Node name '{spec.name}' collides with "
                                 f"reserved keyword names {self._reserved}")
            if spec.name in self.specs:
                raise ValueError(f"Duplicate node name '{spec.name}'")
            if not spec.inputs:
                raise ValueError(f"Node '{spec.name}' must list at least one "
                                 "`inputs:` entry")
            self.specs[spec.name] = spec

        # ---- 2. dependency sanity checks -------------------------------
        all_known = self._reserved | set(self.specs)
        for spec in self.specs.values():
            unknown = set(spec.inputs) - all_known
            if unknown:
                raise ValueError(f"Node '{spec.name}' references unknown input "
                                 f"names {sorted(unknown)}")

        # ---- 3. topological order (detect cycles) ----------------------
        self._order = self._toposort()

        # ---- 4. instantiate Modules ------------------------------------
        self._modules = self._instantiate_all()

    # .......................................................................
    # Public API -------------------------------------------------------------
    # .......................................................................
    def build(self) -> nn.Module:
        """Return a ready‑to‑use `torch.nn.Module` executing the DAG."""
        builder = self    # capture for inner class

        class _VectorDAG(nn.Module):
            def __init__(self):
                super().__init__()
                self._nodes = nn.ModuleDict(builder._modules)   # <‑ modules

            def forward(self, inputs):
                """
                `inputs` can be either
                  • dict  containing any of the reserved sources, or
                  • a PyG `Data` object (interpreted as '$graph')
                """
                cache: dict[str, Any] = {}

                # ----------- seed the cache with reserved sources ----------
                if isinstance(inputs, dict):
                    cache.update(inputs)
                else:
                    cache["graph"] = inputs

                # ----------- execute DAG in topological order -------------
                for name in builder._order:
                    spec   = builder.specs[name]
                    module = self._nodes[name]

                    # gather inputs
                    in_vals = [cache[key] for key in spec.inputs]

                    # Fan‑in semantics:  by default pass *first* element unless
                    # the block is in _CAT_KINDS  (then concatenate)
                    if len(in_vals) == 1:
                        x = in_vals[0]
                    elif spec.target in _CAT_KINDS:
                        if not all(isinstance(t, torch.Tensor) for t in in_vals):
                            raise TypeError("Concat layer expects tensor inputs")
                        x = torch.cat(in_vals, dim=-1)
                    else:
                        # forward tuples to modules that accept them (rare)
                        x = tuple(in_vals)

                    out = module(x)

                    # ---- per‑task sanity ---------------------------------
                    if spec.produces_tasks:
                        if not isinstance(out, dict):
                            raise TypeError(f"Node '{name}' declares "
                                            "`produces_tasks` but returned "
                                            f"{type(out)}")
                        missing = set(spec.produces_tasks) - set(out)
                        if missing:
                            raise ValueError(f"Node '{name}' did not produce "
                                             f"all requested tasks {missing}")

                    cache[name] = out

                return cache[builder._order[-1]]          # pipeline output

        return _VectorDAG()

    # .......................................................................
    # Internal helpers -------------------------------------------------------
    # .......................................................................
    def _toposort(self) -> list[str]:
        """Kahn’s algorithm – fails early on cycles."""
        in_deg  = defaultdict(int)
        edges   = defaultdict(list)
        for spec in self.specs.values():
            for parent in spec.inputs:
                if parent in self._reserved:      # reserved nodes have no edges
                    continue
                edges[parent].append(spec.name)
                in_deg[spec.name] += 1

        # queue = nodes with zero indegree
        q = deque([n for n in self.specs if in_deg[n] == 0])
        order = []

        while q:
            n = q.popleft()
            order.append(n)
            for child in edges[n]:
                in_deg[child] -= 1
                if in_deg[child] == 0:
                    q.append(child)

        if len(order) != len(self.specs):
            raise RuntimeError("Cycle detected in vector‑pipeline DAG.")
        return order

    # -----------------------------------------------------------------------
    def _instantiate_all(self) -> dict[str, nn.Module]:
        """Create every nn.Module, inject `in_dim` where obvious, check dims."""
        out_dims: dict[str, int | None] = {}   # track feature sizes when known
        modules:  dict[str, nn.Module] = {}

        # seed: reserved inputs have unknown (None) dimensionality
        for r in self._reserved:
            out_dims[r] = None

        for name in self._order:
            spec = self.specs[name]
            kind, tgt = _split_target(spec.target)

            # ---- heuristic: infer `in_dim` when exactly ONE tensor input ---
            if "in_dim" not in spec.kwargs:
                src_dims = [out_dims[i] for i in spec.inputs if out_dims[i] is not None]
                if len(src_dims) == 1:
                    spec.kwargs["in_dim"] = src_dims[0]

            mod = _lookup_module(kind, tgt, spec.kwargs)

            # ---- dimension bookkeeping ----------------------------------
            out_dims[name] = getattr(mod, "out_dim", None)
            modules[name]  = mod

            # ---- optional mismatch guard -------------------------------
            if len(spec.inputs) == 1:
                src = spec.inputs[0]
                if out_dims[src] is not None and out_dims[name] is not None \
                   and out_dims[src] != out_dims[name]:
                    msg = (f"Dim mismatch: '{src}' ({out_dims[src]}) → "
                           f"'{name}' expects {out_dims[name]}")
                    if self._auto_proj and kind != "core":
                        # auto‑insert a Projection block *before* the consumer
                        proj = get_from_registry("core", "projection")(
                            out_dims[src], out_dims[name]
                        )
                        pj_name = f"{src}→{name}.proj"
                        modules[pj_name] = proj
                        # splice it into execution order just before consumer
                        self._order.insert(self._order.index(name), pj_name)
                        # fix wiring
                        self.specs[pj_name] = NodeSpec(
                            name   = pj_name,
                            target = "core.projection",
                            inputs = [src],
                            kwargs = {"in_dim": out_dims[src],
                                      "out_dim": out_dims[name]},
                        )
                        out_dims[pj_name] = out_dims[name]
                        # redirect consumer’s input
                        spec.inputs = [pj_name]
                    else:
                        raise ValueError(msg)

        return modules
    
###############################################################################
# Other Code:
###############################################################################    

# .............................................................................
# 1.  Helper – HP‑dict  →  vector‑pipeline Node list
# .............................................................................
def build_vector_pipeline_nodes(hp: Dict[str, Any]) -> List[Dict]:
    """
    Parameters
    ----------
    hp : dict
        A *flat* hyper‑parameter dict coming from Ray‑Tune.  Expected keys
        (see the mini‑configs further below):

            encoder_type         ∈ {"gnn", "minimol", "hybrid"}
            gnn_model            (only if encoder_type contains "gnn")
            gnn_hidden_dim
            num_node_feats
            num_edge_feats
            minimol_proj_dim
            compress             ∈ {True, False}
            bottleneck_ratio, compression_depth, compression_dropout, ...

    Returns
    -------
    list[dict]
        Each element follows the NodeSpec YAML schema.
    """
    nodes: List[Dict] = []

    # ---------- 1. GNN branch (optional) ---------------------------------
    if hp["encoder_type"] in {"gnn", "hybrid"}:
        nodes.append({
            "name":   "gnn_encoder",
            "target": f"constructor.{hp['gnn_model']}",
            "inputs": ["graph"],
            "kwargs": {
                "hidden":         hp["gnn_hidden_dim"],
                "num_node_feats": hp["num_node_feats"],
                "num_edge_feats": hp["num_edge_feats"],
                # task_head will be ignored – we treat the GNN as feature maker
                "n_tasks":        1,
            },
        })

    # ---------- 2. Minimol branch (optional) -----------------------------
    if hp["encoder_type"] in {"minimol", "hybrid"}:
        nodes.append({
            "name":   "fp_encoder",
            "target": "core.minimol_encoder",
            "inputs": ["minimol"],
            "kwargs": {
                "proj_dim":    hp["minimol_proj_dim"],
                "activate":    "relu",
                "dropout_p":   0.0,
                "l2_normalise": False,
            },
        })

    # ---------- 3. Combine (if Hybrid) -----------------------------------
    head_name = None
    if hp["encoder_type"] == "hybrid":
        nodes.append({
            "name":   "concat",
            "target": "core.concat_identity",   # ← previously core.concat_encoder
            "inputs": ["gnn_encoder", "fp_encoder"],
            "kwargs": {},
        })
        head_name = "concat"
    elif hp["encoder_type"] == "gnn":
        head_name = "gnn_encoder"
    else:                                 # "minimol"
        head_name = "fp_encoder"

    # ---------- 4. Optional compression block ----------------------------
    if hp["compress"]:
        nodes.append({
            "name":   "compress",
            "target": "core.compression_block",
            "inputs": [head_name],
            "kwargs": {
                "in_dim": _repr_dim(hp),
                "bottleneck_ratio": hp["bottleneck_ratio"],
                "depth":            hp["compression_depth"],
                "dropout_p":        hp["compression_dropout"],
                "activate":         "relu",
            },
        })
        head_name = "compress"            # new tip of the DAG

    #    → the *output* of the vector pipeline is whatever `head_name` is
    #      (LibMTL will ingest it as the shared encoder class).
    return nodes

# .............................................................................
# 2.  Mini sub‑spaces (Ray‑Tune ready)
# .............................................................................
def encoder_space():
    """Search space for the *initial* molecular representation."""
    return {
        # choice of overall strategy
        "encoder_type": tune.choice(["gnn", "minimol", "hybrid"]),

        # GNN‑specific
        "gnn_model":      tune.choice(["dmpnn", "gcnn", "gtr"]),
        "gnn_hidden_dim": tune.choice([64, 128, 256]),
        "num_node_feats": 118,          # constant for atom one‑hot
        "num_edge_feats": 4,            # bond types

        # Minimol‑specific
        "minimol_proj_dim": tune.choice([None, 128, 256]),
    }


def compression_space():
    """Return a dict with conditional sampling on `compress`."""
    return {
        "compress": tune.choice([True, False]),

        # the three keys below are *ignored* downstream if compress==False
        "bottleneck_ratio": tune.sample_from(
            lambda spec: random.uniform(0.25, 0.75) if spec["compress"] else 0.5
        ),
        "compression_depth": tune.sample_from(
            lambda spec: random.choice([1, 2]) if spec["compress"] else 1
        ),
        "compression_dropout": tune.sample_from(
            lambda spec: random.uniform(0.0, 0.3) if spec["compress"] else 0.0
        ),
    }


def multitask_arch_space():
    """Which LibMTL architecture to wrap around the encoder."""
    return {
        "multitask_arch": tune.choice(
            #["identity", "shared_bottom", "sluice", "mmoe"]
            ["identity", "shared_bottom", "sluice"]
        ),
        # Example of an arch‑specific hyper‑param (only for MMoE)
        "mmoe_num_experts": tune.choice([2, 4, 6]),
    }


def decoder_space():
    """Hyper‑parameters for the per‑task MLP head."""
    return {
        "decoder_hidden_dims": tune.choice([[64], [128], [128, 64]]),
        "decoder_dropout":     tune.uniform(0.0, 0.3),
    }

# --------------------------------------------------------------------------- #
# Collate helper able to handle  (a) GNN‑only, (b) FP‑only, (c) hybrid batches
# --------------------------------------------------------------------------- #
def _collate(batch):
     """
     Parameters
     ----------
     batch : list[Data]
         Every Data object must at least contain `.y`
         Optional attribute  `.minimol_representation`  (1‑D length‑512 tensor).
 
     Returns
     -------
     dict
         {
             "inputs"  :  Data  |  {"graph": DataBatch, "minimol": fp_tensor},
             "targets" :  dict[task → tensor] **or** tensor          # same rules as before
         }
     """
     # ----------- graph ------------------------------------------------------
     g_batch = Batch.from_data_list(batch)          # always available
 
     # ----------- minimol fp (if present) ------------------------------------
     if hasattr(batch[0], "minimol_representation"):
        fp_stack = torch.stack(
            [torch.as_tensor(d.minimol_representation,
                             dtype=torch.float32, device=d.x.device)
             for d in batch], 0)          # shape (B, 512)  ✅
        inputs = {"graph": g_batch, "minimol": fp_stack}
     else:                                           # legacy GNN‑only path
         fp_stack = torch.zeros(len(batch), 512)    
         inputs = {"graph": g_batch, "minimol": fp_stack}
 
     # ----------- targets ----------------------------------------------------
     # (single tensor  or  per‑task dict retained unchanged)
     y = getattr(batch[0], "y", None)
     if y is not None and y.ndim <= 2:               # e.g. shape (1,) or (T,)
         targets = torch.stack([d.y for d in batch])
     else:                                           # already a dict inside .y
         targets = {k: torch.stack([d.y[k] for d in batch]) for k in batch[0].y}
 
     return {"inputs": inputs, "targets": targets}


if __name__ == "__main__" and os.getenv("RUN_TUNE_DEMO", "1") == "1":

    # ---------------------------------------------------------------
    # 1. Load CSV into Table (contains SMILES + Hit label)
    # ---------------------------------------------------------------
    DATA_CSV = "/Users/asselism/Desktop/Collins_Lab/Datasets/Organized_by_Data_Type/Bioactivity/Organized_by_Data_Type/Antimicrobial_Activity/Organized_by_Source/Internal_Collins_Lab_Screens/39K_Screen/Organized_by_Species/Modified_Datasets/FORMATTED_Datasets/Escherichia_coli_GI_with_Minimol_Representations.csv"
    tbl = to_table(DATA_CSV)          # factory you already use

    tbl = tbl.head(n=1000)

    # ─────────────────────────────────────────────────────────────────────────────
    # 1.  Compute / cache the graph column  (fast ‑‑ ~1 k rows here)
    # ─────────────────────────────────────────────────────────────────────────────
    if "Graph" not in tbl.to_pandas().columns:
        print("🛠️  Generating graph objects …")
        tbl = tbl.assign_rowwise_using(
            "Graph",
            lambda row: smiles_to_data(row["SMILES"])
        )
    
    # ─────────────────────────────────────────────────────────────────────────────
    # 2.  Train / val / test split
    # ─────────────────────────────────────────────────────────────────────────────
    train_tbl, val_tbl, test_tbl = TrainTestSplit(tbl, seed=42)
    
    def _repr_dim(hp: Dict[str, Any]) -> int:
        """Infer the flat representation size *before* the optional compression."""
        if hp["encoder_type"] == "gnn":
            return hp["gnn_hidden_dim"]
        if hp["encoder_type"] == "minimol":
            return 512 if hp["minimol_proj_dim"] is None else hp["minimol_proj_dim"]
        # hybrid  →  concatenate
        fp_dim = 512 if hp["minimol_proj_dim"] is None else hp["minimol_proj_dim"]
        return hp["gnn_hidden_dim"] + fp_dim


    def _register_vector_encoder(unique_name: str, nodes: List[Dict]) -> Type[nn.Module]:
        """
        Dynamically create **and return** an encoder class that executes the
        Vector‑pipeline DAG produced by `build_vector_pipeline_nodes`.
    
        The class is still pushed into the 'core' registry so any old code
        that relies on the lookup continues to work.
        """
        if unique_name in _REGISTRIES["core"]:
            return _REGISTRIES["core"][unique_name]
    
        @register("core", unique_name)                 # keeps hot‑reload ability
        class VectorEncoder(nn.Module):
            out_dim = None  # filled in __init__
    
            def __init__(self):
                super().__init__()
                self.net = MoleculeModelBuilder(nodes).build()
                # Heuristic size inference (unchanged)
                VectorEncoder.out_dim = _repr_dim_hp
    
            def forward(self, x):
                return self.net(x)
    
        return VectorEncoder            # 🔑  NEW: hand the class back to caller

    
    # ---------- data ----------------------------------------------------------------
    def _rows_to_data(pdf):
        lst = []
        for _, row in pdf.iterrows():
            g = row["Graph"].clone()
            g.y = torch.tensor([int(row["Hit"])], dtype=torch.long)  # shape (1,)
            lst.append(g)
        return lst
    
    train_graphs = _rows_to_data(train_tbl.to_pandas())
    val_graphs   = _rows_to_data(val_tbl.to_pandas())
    
    def build_loaders(hp, *, train_graphs, val_graphs):
        bs = hp["batch_size"]
        return (
            DataLoader(train_graphs, batch_size=bs, shuffle=True,  collate_fn=_collate),
            DataLoader(val_graphs,   batch_size=bs, shuffle=False, collate_fn=_collate),
        )
    
    # ----------  Lightning‑config factory -----------------------------------------
    def build_lightning_cfg(hp: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert one Tune sample (flat dict) → LightningModule config dict understood
        by `MultiTaskLightningModule`.
        """
        # 1. vector‑pipeline nodes
        nodes = build_vector_pipeline_nodes(hp)
    
        # 2. build the *class* for the encoder and register it
        enc_key  = f"vector_dag_{uuid.uuid4().hex[:8]}"
        global _repr_dim_hp                      # used by VectorEncoder.__init__
        _repr_dim_hp = _repr_dim(hp)
        VectorEncoder = _register_vector_encoder(enc_key, nodes)
    
        # 3. decoder (MLP head) kwargs  ← **this was missing**
        head_kwargs = dict(
            in_dim       = _repr_dim_hp,
            hidden_dims  = hp["decoder_hidden_dims"],
            dropout_p    = hp["decoder_dropout"],
            out_dim      = 1,                    # binary classification
        )
    
        # 4. architecture‑specific extra kwargs (only MMoE needs them)
        extra_arch_kw = {}
        if hp["multitask_arch"] == "mmoe":
            extra_arch_kw = dict(num_experts=hp["mmoe_num_experts"])
    
        # 5. assemble the full Lightning‑Module config
        cfg = {
            "module": "multitask_default",
            "arch": {
                "builder": (
                    hp["multitask_arch"]
                    if hp["multitask_arch"] != "sluice" else "shared_bottom"
                ),
                "kwargs": {
                    "task_names":   ["hit"],
                    "encoder_name": VectorEncoder,     # pass the class itself
                    "head_name":    "mlp",
                    "head_kwargs":  head_kwargs,
                    "multi_input":  True,        # <‑‑ we pass a dict to .forward()
                    "arch_kwargs":  extra_arch_kw
                },
            },
            "losses": [{"name": "bce_with_logits", "task": "hit", "weight": 1.0}],
            "metrics": [{"name": "auroc", "task": "hit"},
                        {"name": "auprc", "task": "hit"}],
            "optim":   {"name": "adamw",
                        "kwargs": {"lr": hp["lr"], "weight_decay": 0.0}},
            "trainer": {"max_epochs": hp["max_epochs"],
                        "accelerator": "cpu", "devices": 1},
            "early_stopping": {
                "monitor": "val_loss", "mode": "min",
                "patience": 5, "min_delta": 0.001,
            },
        }
        return cfg

    # ----------  Tune trial --------------------------------------------------------
    def _trainable(hp, *, train_graphs=None, val_graphs=None):
        # 1️⃣ Register the encoder *inside* the worker
        #nodes = build_vector_pipeline_nodes(hp)
        #enc_name = f"vector_dag_{uuid.uuid4().hex[:8]}"
        #_register_vector_encoder(enc_name, nodes)
    
        # 2️⃣ Build the Lightning config *after* the registration
        cfg = build_lightning_cfg(hp)   # pass name in
    
        train_loader, val_loader = build_loaders(hp,
                                                 train_graphs=train_graphs,
                                                 val_graphs=val_graphs)
        lit = MultiTaskLightningModule(cfg)
    
        tune_cb = TuneReportCheckpointCallback(
            metrics={"val_loss": "val_loss"},
            filename="checkpoint",
            on="validation_end",
        )
        trainer = pl.Trainer(
            max_epochs=hp["max_epochs"],
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_checkpointing=True,
            callbacks=[tune_cb],
        )
        trainer.fit(lit, train_loader, val_loader)
    
    # ----------  Search‑space ------------------------------------------------------
    search_space = {}
    search_space.update(encoder_space())
    search_space.update(compression_space())
    search_space.update(multitask_arch_space())
    search_space.update(decoder_space())
    # generic training H‑params
    search_space.update({
        "lr":          tune.loguniform(1e-4, 1e-2),
        "batch_size":  tune.choice([32, 64, 128]),
        "max_epochs":  tune.choice([15, 20]),
    })
    
    # ----------  Ray Tune driver ---------------------------------------------------
    _trainable_with_data = tune.with_parameters(
        _trainable,
        train_graphs=train_graphs,
        val_graphs=val_graphs,
    )
    
    tuner = tune.Tuner(
        tune.with_resources(_trainable_with_data, resources={"cpu": 1, "gpu": 0}),
        param_space=search_space,
        tune_config=TuneConfig(
            metric="val_loss", mode="min",
            num_samples=16,
            scheduler=ASHAScheduler(max_t=20, grace_period=5),
        ),
        run_config=RunConfig(
            name="full_arch_sweep",
            log_to_file=("stdout.txt", "stderr.txt"),
            storage_path="/Users/asselism/Desktop/Collins_Lab/Models/Model_Building/Organized_by_Project/Active_Learning/Pipeline_Development/",
            checkpoint_config=CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_attribute="val_loss",
                checkpoint_score_order="min",
            ),
        ),
    )
    
    result = tuner.fit()
    
    print("🎯  Best trial config ↴")
    print(result.get_best_result().config)