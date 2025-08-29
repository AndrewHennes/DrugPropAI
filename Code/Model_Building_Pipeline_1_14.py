#!/usr/bin/env python3

# ============================================================================
# 1  Imports & helpers
# ============================================================================

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Built-In Modules:
# --------------------------------------------------------------------------- #

import abc
import base64
from collections import defaultdict, deque
import copy
from dataclasses import dataclass, field, fields
from functools import partial, wraps
import inspect
import io
import math
from numbers import Number
import os
import pathlib
from pathlib import Path
import random
import re
import sys
import tempfile
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union
import uuid

# --------------------------------------------------------------------------- #
# Import from Data Processing Pipeline:
# --------------------------------------------------------------------------- #

data_processing_pipeline_dir_path = "/Users/asselism/Desktop/Collins_Lab/SharedCode/Shared_Pipeline_Working_Dir/"
sys.path.insert(0, data_processing_pipeline_dir_path)
from Shared_Pipeline_Ray_v1_14 import *  # TODO: Update with the real import path.

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

from sklearn.metrics import roc_auc_score as sklearn_metrics_auroc

from torch_geometric.data import Data, Batch
#from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree as tg_degree
from torch_geometric.nn    import (
    MessagePassing,
    global_mean_pool, global_add_pool, global_max_pool,
    Linear as GeoLinear,
    GATConv, GCNConv                        # community layer
)
from torch_geometric.data import Batch as GeoBatch

from torchmetrics.classification import MulticlassAUROC as torch_metrics_auroc
from torchmetrics.functional import accuracy as _tm_acc
from torchmetrics.functional import auroc as _tm_auroc
from torchmetrics.functional import average_precision as _tm_ap

from torch_scatter import scatter

from ray import tune
from ray.air import RunConfig, CheckpointConfig
from ray.tune import TuneConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback

###############################################################################
# 0. Other Functionality:
###############################################################################

# ########################################################################### #
# Environment Variables:
# ########################################################################### #

# =========================================================================== #
# Set Environment Variables:
# =========================================================================== #

DEFAULT_ACTIVATION_FUNCTION = "relu"
DEFAULT_OPTIMIZATION_FUNCTION = "adam"
DEFAULT_LOSS_WEIGHTING_STRATEGY = {"name": "identity", "kwargs": {}}

# =========================================================================== #
# Execute Environment Changes:
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
EDGE_FEAT_DIM = len(BOND_TYPES)

# ########################################################################### #
# Expanded General Functionality:
# ########################################################################### #

# =========================================================================== #
# Data Loading:
# =========================================================================== #

def data_to_b64(g):
    if g is None:
        return None
    import io, base64, torch
    buf = io.BytesIO()
    torch.save(g, buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def b64_to_data(s):
    if s is None or s == "" or (isinstance(s, float) and math.isnan(s)):
        return None
    raw = base64.b64decode(s.encode("ascii") if isinstance(s, str) else s)
    bio = io.BytesIO(raw)
    try:
        # Newer PyTorch: force full unpickler
        return torch.load(bio, map_location="cpu", weights_only=False)
    except TypeError:
        # Older PyTorch: argument not supported
        bio.seek(0)
        return torch.load(bio, map_location="cpu")
    except Exception as e:
        print(f"b64_to_data failed: {type(e).__name__}: {e}")
        return None

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

# =========================================================================== #
# IterableTable:
# =========================================================================== #

class IterableTable(Table, IterableDataset):
    pass

# =========================================================================== #
# Reproducability:
# =========================================================================== #

def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
# =========================================================================== #
# Tensor Property Checking:
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Tensor Shape:
# --------------------------------------------------------------------------- #

def shape_of(x: torch.Tensor, *, who: str) -> Tuple[int, ...]:
    
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{who}.forward expected torch.Tensor, got {type(x).__name__}.")
    return tuple(x.shape)

# --------------------------------------------------------------------------- #
# Tensor Shape Checks:
# --------------------------------------------------------------------------- #

def num_dims_is(x, expected, who: str) -> None:
    
    shape = shape_of(x, who=who)
    if len(shape) != expected:
        raise ValueError(f"{who}.forward expected tensor with == {expected} dims, got shape {shape}.")

def num_dims_is_at_least(x, expected, who: str) -> None:
    
    shape = shape_of(x, who=who)
    if len(shape) < expected:
        raise ValueError(f"{who}.forward expected tensor with ≥ {expected} dims, got shape {shape}.")

def last_dim_has_size(x: torch.Tensor, expected: int, who: str) -> None:
    
    shape = shape_of(x, who=who)
    if len(shape) < 1 or shape[-1] != expected:
        raise ValueError(f"{who}.forward expected last dim == {expected}, got shape {shape}.")
        
def has_dim_index(x: torch.Tensor, dim: int, who: str) -> None:
    """Ensure `dim` is a valid axis index for x (supports negatives)."""
    shape = shape_of(x, who=who)
    ndim = len(shape)
    pos = dim if dim >= 0 else dim + ndim
    if not (0 <= pos < ndim):
        raise ValueError(f"{who}.forward configured with dim={dim}, but input has {ndim} dims (shape {shape}).")

# ########################################################################### #
# Machine Learning Parts Registry:
# ########################################################################### #

_REGISTRIES: Dict[str, Dict[str, Any]] = {
    "core": {},
    "head": {},
    "mixer": {},
    "weighting": {},
    "loss": {}, 
    "optim": {},
    "metric": {},
    "ptl": {},
    "callback": {},
    "architecture": {},
    "constructor": {},
    "search": {},
    "scheduler": {},
    "collate": {}
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
get_mixer  = lambda name: get_from_registry("mixer",  name)
get_weighting  = lambda name: get_from_registry("weighting",  name)
get_loss = lambda name: get_from_registry("loss",  name)
get_optim = lambda name: get_from_registry("optim",  name)
get_metric = lambda name: get_from_registry("metric",  name)
get_ptl = lambda name: get_from_registry("ptl",  name)
get_callback = lambda name: get_from_registry("callback",  name)
get_collate = lambda name: get_from_registry("collate", name)

# =============================================================================
# Data Type Conversion:
# =============================================================================

def smiles_to_data(smiles: str) -> Data | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features (118‑dim one‑hot)
    x_list = []
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        feat = torch.zeros(len(ATOM_TYPES))
        if z in ATOM_TYPES:
            feat[z - 1] = 1.0
        x_list.append(feat)
    x = torch.stack(x_list, dim=0)

    # Edges
    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        t = torch.zeros(len(BOND_TYPES), dtype=torch.float32)  # one‑hot (4,)
        t[BOND_TYPES[bond.GetBondType()]] = 1.0
        edge_index.extend([[i, j], [j, i]])
        edge_attr.extend([t, t])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() \
                 if edge_index else torch.empty(2, 0, dtype=torch.long)
    # *** key change: never leave edge_attr as None; use an empty (0, 4) tensor ***
    edge_attr  = torch.stack(edge_attr, dim=0) if edge_attr \
                 else torch.zeros((0, EDGE_FEAT_DIM), dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# #############################################################################
# 1. CORE:
# #############################################################################

# =========================================================================== #
# Shared Foundational Components:
# =========================================================================== #

class QueryInterface:
    """
    Optional mix‑in that exposes uniform introspection hooks.
    Leaf modules override selectively.
    """

    # Default behavior is that a "core" component is not assumed to make final
    # predictions => default when called is an empty dictionary.
    
    def get_preds(self) -> Dict[str, torch.Tensor]:           # noqa: D401
        return {}

    # Default behavior is to assume no activations.
    
    def get_activations(self) -> Dict[str, torch.Tensor]:     # noqa: D401
        return {}

    # Returns all parameters that use a gradient => are in the training graph.
    
    def get_learnable_weights(self) -> Dict[str, torch.Tensor]:
        # Default: expose all trainable parameters with their *full* names.
        return {n: p for n, p in self.named_parameters() if p.requires_grad}
    
    def who(self) -> str:
        return getattr(self, "name", self.__class__.__name__)

# =========================================================================== #
# Standard Linear and Bias Layers:
# =========================================================================== #

@register("core", "linear_transformation")
class LinearTransformation(QueryInterface, nn.Module):
    """Weight‑only `y = Wx`."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        num_dims_is(x, 2, self.who())
        last_dim_has_size(x, self.weight.size(1), self.who())
        
        return torch.matmul(x, self.weight.t())


@register("core", "bias")
class Bias(QueryInterface, nn.Module):
    """Learnable bias added to last dimension."""

    def __init__(self, out_dim: int):
        super().__init__()
        
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.name = "Bias"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        num_dims_is(x, 2, self.who())
        last_dim_has_size(x, self.bias.size(0), self.who())
        
        return x + self.bias

@register("core", "affine")
@register("core", "affine_transformation")
class Affine_Transformation(QueryInterface, nn.Module):
    """Affine transform = Linear ∘ (optional) Bias."""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        
        self.linear = LinearTransformation(in_dim, out_dim)
        self.bias   = Bias(out_dim) if bias else nn.Identity()
        self.name = "Affine_Transformation"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        num_dims_is(x, 2, self.who())
        last_dim_has_size(x, self.linear.weight.size(1), self.who())
        
        return self.bias(self.linear(x))

# =========================================================================== #
# Standard Regularization Layers:
# =========================================================================== #

@register("core", "layer_norm")
class LayerNorm(QueryInterface, nn.Module):
    
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dim = dim
        self.name = "LayerNorm"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        num_dims_is(x, 2, self.who())
        last_dim_has_size(x, self.dim, self.who())
        
        return self.norm(x)


@register("core", "dropout")
class DropoutLayer(QueryInterface, nn.Module):
    
    def __init__(self, dropout_prob):
        super().__init__()
        
        self.drop = nn.Dropout(dropout_prob)
        self.name = "Dropout"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        num_dims_is(x, 2, self.who())
        
        return self.drop(x)
  
@register("core", "batch_norm")
class BatchNorm(QueryInterface, nn.Module):
    
    def __init__(self, dim: int):
        super().__init__()
        
        self.norm = nn.BatchNorm1d(dim)
        self.dim = dim
        self.name = "BatchNorm"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        num_dims_is(x, 2, self.name)
        last_dim_has_size(x, self.dim, self.name)
        
        return self.norm(x)  
  
# =========================================================================== #
# Standard Activation Functions:
# =========================================================================== #

@register("core", "identity")
class Identity(QueryInterface, nn.Module):
    def __init__(self):
        super().__init__()
        
        self.act = nn.Identity()
        self.name = "Identity"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)


@register("core", "relu")
class ReLU(QueryInterface, nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.ReLU()
        self.name = "ReLU"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)


@register("core", "leaky_relu")
class LeakyReLU(QueryInterface, nn.Module):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.act = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)


@register("core", "prelu")
class PReLU(QueryInterface, nn.Module):
    def __init__(self, num_parameters: int = 1, init: float = 0.25):
        super().__init__()
        self.act = nn.PReLU(num_parameters=num_parameters, init=init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)


@register("core", "gelu")
class GELU(QueryInterface, nn.Module):
    def __init__(self, approximate: str = "none"):  # "none" or "tanh"
    
        super().__init__()
        self.act = nn.GELU(approximate=approximate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)


@register("core", "silu")
@register("core", "swish")
class SiLU(QueryInterface, nn.Module):
    def __init__(self):
        
        super().__init__()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)


@register("core", "elu")
class ELU(QueryInterface, nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.act = nn.ELU(alpha=alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)


@register("core", "selu")
class SELU(QueryInterface, nn.Module):
    
    def __init__(self):
        
        super().__init__()
        self.act = nn.SELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)


@register("core", "tanh")
class Tanh(QueryInterface, nn.Module):
    
    def __init__(self):
        
        super().__init__()
        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)


@register("core", "sigmoid")
class Sigmoid(QueryInterface, nn.Module):
    
    def __init__(self):
        
        super().__init__()
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)


@register("core", "softplus")
class Softplus(QueryInterface, nn.Module):
    def __init__(self, beta: float = 1.0, threshold: float = 20.0):
        super().__init__()
        self.act = nn.Softplus(beta=beta, threshold=threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)


@register("core", "softsign")
class Softsign(QueryInterface, nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.Softsign()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)


@register("core", "mish")
class Mish(QueryInterface, nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.Mish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)


@register("core", "relu6")
class ReLU6(QueryInterface, nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.ReLU6()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)


@register("core", "hardswish")
class Hardswish(QueryInterface, nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.Hardswish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)


@register("core", "hardtanh")
class Hardtanh(QueryInterface, nn.Module):
    def __init__(self, min_val: float = -1.0, max_val: float = 1.0):
        super().__init__()
        self.act = nn.Hardtanh(min_val=min_val, max_val=max_val)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)


@register("core", "hardsigmoid")
class Hardsigmoid(QueryInterface, nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.Hardsigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)


@register("core", "softmax")
class Softmax(QueryInterface, nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.act = nn.Softmax(dim=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)


@register("core", "log_softmax")
class LogSoftmax(QueryInterface, nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.act = nn.LogSoftmax(dim=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)

# =========================================================================== #
# Feed-Forward Layers:
# =========================================================================== #

@register("core", "feed-forward_layer")
class FeedForwardLayer(nn.Module, QueryInterface):
    """
    Dense MLP block with optional BatchNorm, Dropout, and a configurable activation.

    Args:
        in_dim (int):  Input feature dimension.
        out_dim (int): Output feature dimension.
        batch_norm (bool): If True, apply nn.BatchNorm1d(out_dim). Default: False.
        dropout_probability (float): If 0.0, no dropout; otherwise p must satisfy 0 < p < 1.
        activation_function (str): Name in the "core" registry (e.g. "relu", "gelu", "identity").
        local_name (str): Name used when exposing activations via get_activations().
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        batch_norm: bool = False,
        dropout_probability: float = 0.0,
        activation_function: str = DEFAULT_ACTIVATION_FUNCTION,
        local_name: str = "dense",
    ):
        super().__init__()
        
        # Remember in and out dimensions.
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Linear projection
        self.proj = get_core("affine_transformation")(in_dim, out_dim)

        # Optional BatchNorm1d
        self.norm = get_core("batch_norm")(out_dim) if batch_norm else get_core("identity")()

        # Activation (strings only, resolved via registry)
        if not isinstance(activation_function, str):
            raise TypeError("activation_function must be a registry key (str).")
        self.act = get_core(activation_function)()

        # Optional Dropout
        if float(dropout_probability) == 0.0:
            self.reg = get_core("identity")()
        else:
            p = float(dropout_probability)
            if not (0.0 < p < 1.0):
                raise ValueError(f"dropout_probability must be in (0, 1); got {p}")
            self.reg = get_core("dropout")(p)

        self._latest_out: Optional[torch.Tensor] = None
        self._name = local_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        num_dims_is(x, 2, self.who())
        last_dim_has_size(x, self.in_dim, self.who())
        
        # Project → (optional BN) → Activation → (optional Dropout)
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.reg(x)
        self._latest_out = x
        return x

    def get_activations(self) -> Dict[str, torch.Tensor]:
        return {self._name: self._latest_out} if self._latest_out is not None else {}

# =========================================================================== #
# Feed-Forward Neural Networks:
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Standard FFN:
# --------------------------------------------------------------------------- #

@register("core", "ffn")
@register("core", "FFN")
class FeedForwardNeuralNetwork(nn.Module, QueryInterface):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Union[List[int], Tuple[int, ...]],
        batch_norms: Union[bool, Sequence[bool]] = False,
        dropout_probabilities: Union[float, Sequence[float]] = 0.0,
        activation_functions: Union[str, Sequence[str]] = DEFAULT_ACTIVATION_FUNCTION,
        local_name: str = "dense",
    ):
        super().__init__()
        
        dims = [in_dim] + list(hidden_dims)
        if len(dims) < 2:
            raise ValueError("dims must contain at least two entries: [in_dim, ..., out_dim].")

        num_stages = len(hidden_dims)
        in_dims, out_dims = dims[:-1], dims[1:]

        batch_norms           = broadcast_val_to_list(batch_norms,           num_stages)
        dropout_probabilities = broadcast_val_to_list(dropout_probabilities, num_stages)
        activation_functions  = broadcast_val_to_list(activation_functions,  num_stages)

        # Strict length checks to avoid silent truncation
        if not (len(batch_norms) == len(dropout_probabilities) == len(activation_functions) == num_stages):
            raise ValueError(
                "batch_norms, dropout_probabilities, and activation_functions must "
                f"all have length {num_stages} (or be scalars to broadcast)."
            )

        stages: List[FeedForwardLayer] = []
        for i, (in_dim, out_dim, bn, p, act) in enumerate(
            zip(in_dims, out_dims, batch_norms, dropout_probabilities, activation_functions), start=1
        ):
            stages.append(
                get_core("feed-forward_layer")(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    batch_norm=bool(bn),
                    dropout_probability=float(p),
                    activation_function=str(act),
                    local_name=f"{local_name}.stage{i}",
                )
            )
        
        self.in_dim = dims[0]
        self.out_dim = dims[-1]
        self.stages = nn.ModuleList(stages)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        num_dims_is(x, 2, self.who())
        last_dim_has_size(x, self.in_dim, self.who())
        
        for stage in self.stages:
            x = stage(x)
        return x

    def get_activations(self) -> Dict[str, torch.Tensor]:
        acts: Dict[str, torch.Tensor] = {}
        for stage in self.stages:
            acts.update(stage.get_activations())
        return acts

# --------------------------------------------------------------------------- #
# FFN Autoencoder:
# --------------------------------------------------------------------------- #

@register("core", "mlp_autoencoder")
@register("core", "ffn_autoencoder")
class FFNAutoencoder(nn.Module, QueryInterface):
    """
    A wrapper around the registered FeedForwardNeuralNetwork ("ffn") that constructs
    an encoder/decoder pair for a deterministic MLP autoencoder.

    Layout (example): in_dim -> ... -> latent_dim  ||  latent_dim -> ... -> in_dim

    Args:
        in_dim: input dimensionality (and reconstruction dimensionality).
        shrinkage_factor: compression ratio per stage, 0 < shrinkage_factor < 1 (default: 0.5).
        num_compressions: number of shrink steps in the encoder (>= 1).

        # Regularization / activations (applied separately to encoder & decoder):
        batch_norms: bool or sequence for stages; broadcasted per network.
        dropout_probabilities: float or sequence for stages; broadcasted per network.
        activation_functions: str or sequence; broadcasted per network.
        final_activation: optional str for the **last decoder stage** (default: 'identity').
        final_batch_norm: bool for the last decoder stage (default: False).
        final_dropout: float for the last decoder stage (default: 0.0).
        name: human-readable model name.
        local_name: base name for scoping stages in the registry ('{local_name}.enc', '{local_name}.dec').

    Properties:
        in_dim (int)
        latent_dim (int)
        out_dim (int) == in_dim
    """

    def __init__(
        self,
        *,
        in_dim: int,
        shrinkage_factor: float = 0.5,
        num_compressions: int = 2,
        batch_norms: Union[bool, Sequence[bool]] = False,
        dropout_probabilities: Union[float, Sequence[float]] = 0.0,
        activation_functions: Union[str, Sequence[str]] = DEFAULT_ACTIVATION_FUNCTION,
        final_activation: Optional[str] = "identity",
        final_batch_norm: bool = False,
        final_dropout: float = 0.0,
        name: str = "Non-Variational_Autoencoder",
        local_name: str = "mlp_autoencoder",
    ):
        super().__init__()

        # --- Validations ---
        if num_compressions < 1:
            raise ValueError("num_compressions must be >= 1")
        if not (0.0 < shrinkage_factor < 1.0):
            raise ValueError("shrinkage_factor must be in (0, 1) when using multiplicative shrinkage.")

        # --- Encoder dimensions ---
        # Use 1..num_compressions so the first hidden is actually compressed
        enc_hidden_dims: List[int] = [
            max(1, int(in_dim * (shrinkage_factor ** k)))
            for k in range(1, num_compressions + 1)
        ]
        latent_dim: int = enc_hidden_dims[-1]

        # --- Decoder dimensions ---
        # Mirror the encoder *excluding* the latent again, then output back to in_dim.
        dec_in_dim: int = latent_dim
        dec_hidden_dims: List[int] = enc_hidden_dims[:-1][::-1] + [in_dim]

        # --- Per-stage regs/acts (encoder) ---
        enc_bn  = broadcast_val_to_list(batch_norms,           len(enc_hidden_dims))
        enc_do  = broadcast_val_to_list(dropout_probabilities, len(enc_hidden_dims))
        enc_act = broadcast_val_to_list(activation_functions,  len(enc_hidden_dims))

        # --- Per-stage regs/acts (decoder) ---
        dec_bn  = broadcast_val_to_list(batch_norms,           len(dec_hidden_dims))
        dec_do  = broadcast_val_to_list(dropout_probabilities, len(dec_hidden_dims))
        dec_act = broadcast_val_to_list(activation_functions,  len(dec_hidden_dims))

        # Tweak the final decoder stage (output layer)
        dec_bn[-1]  = final_batch_norm
        dec_do[-1]  = final_dropout
        dec_act[-1] = final_activation

        # --- Build encoder & decoder via the registered FFN ---
        FFN = get_core("ffn")

        self.encoder = FFN(
            in_dim=int(in_dim),
            hidden_dims=enc_hidden_dims,
            batch_norms=enc_bn,
            dropout_probabilities=enc_do,
            activation_functions=enc_act,
            local_name=f"{local_name}.enc",
        )

        self.decoder = FFN(
            in_dim=int(dec_in_dim),
            hidden_dims=dec_hidden_dims,
            batch_norms=dec_bn,
            dropout_probabilities=dec_do,
            activation_functions=dec_act,
            local_name=f"{local_name}.dec",
        )

        # --- Public properties ---
        self._local_name = local_name
        self.in_dim = int(in_dim)
        self.latent_dim = int(latent_dim)
        self.out_dim = int(in_dim)
        self.name = name

    # --- Interface ---

    def who(self) -> str:
        return f"MLPAutoencoder({self._local_name}): {self.in_dim} -> {self.latent_dim} -> {self.out_dim}"

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        num_dims_is(x, 2, self.who())
        last_dim_has_size(x, self.in_dim, self.who())
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        num_dims_is(z, 2, self.who())
        last_dim_has_size(z, self.latent_dim, self.who())
        return self.decoder(z)

    def forward(self, x: torch.Tensor, latent: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return (x_hat, z) if latent else x_hat

    def get_activations(self) -> Dict[str, torch.Tensor]:
        acts: Dict[str, torch.Tensor] = {}
        if hasattr(self.encoder, "get_activations"):
            for k, v in self.encoder.get_activations().items():
                acts[f"encoder.{k}"] = v
        if hasattr(self.decoder, "get_activations"):
            for k, v in self.decoder.get_activations().items():
                acts[f"decoder.{k}"] = v
        return acts

# ########################################################################### #
# Output Heads:
# ########################################################################### #

# --------------------------------------------------------------------------- #
# Simple Output Heads:
# --------------------------------------------------------------------------- #

# @register("head", "classification")
# class ClassificationHead(QueryInterface, nn.Module):
#     def __init__(
#         self,
#         in_dim: int,
#         num_classes: int,
#         task: str,
#         activation_function: Optional[str] = None,
#         local_name: Optional[str] = None,
#     ):
#         super().__init__()
#         act = activation_function if isinstance(activation_function, str) else "identity"
#         self.transformation = get_core("feed-forward_layer")(
#             in_dim=in_dim,
#             out_dim=num_classes,
#             activation_function=act,
#             local_name=local_name or f"{task}.head",
#         )
#         self.task = task
#         self._logits: Optional[torch.Tensor] = None

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         self._logits = self.transformation(x)
#         return self._logits

#     def get_preds(self) -> Dict[str, torch.Tensor]:
#         return {self.task: self._logits} if self._logits is not None else {}

#     def get_activations(self) -> Dict[str, torch.Tensor]:
#         # Delegate to the inner layer so activations are still visible
#         return self.transformation.get_activations()

@register("head", "classification")
class ClassificationHead(QueryInterface, nn.Module):
    """
    Classification head that always returns 2-D logits.

    • Multiclass (C >= 2): out = (B, C)
    • Binary (C == 1):     projects to (B, 1) then expands to (B, 2) via [-z, z]
                           so downstream CE/AUROC expecting 2 logits works.
    """
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        task: str,
        activation_function: Optional[str] = None,
        local_name: Optional[str] = None,
    ):
        super().__init__()
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")

        self.task = task
        self.num_classes = int(num_classes)

        act = activation_function if isinstance(activation_function, str) else "identity"

        # We *project* to 1 when C==1, then expand to 2 logits in forward().
        proj_out_dim = 1 if self.num_classes == 1 else self.num_classes
        self.transformation = get_core("feed-forward_layer")(
            in_dim=in_dim,
            out_dim=proj_out_dim,
            activation_function=act,
            local_name=local_name or f"{task}.head",
        )

        # Public for introspection
        self.in_dim = in_dim
        # Final outward-facing dimension (after potential expansion)
        self.out_dim = 2 if self.num_classes == 1 else self.num_classes

        self._logits: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.transformation(x)              # (B, 1) if C==1 else (B, C)
        # Ensure 2-D logits outward:
        if y.dim() != 2:
            # Defensive: if the inner block ever squeezed, fix it.
            y = y.view(y.size(0), -1)

        if self.num_classes == 1:
            # Expand single logit z -> [-z, z] to emulate 2-class logits
            if y.size(-1) != 1:
                raise RuntimeError(f"Internal inconsistency: expected (B,1), got {tuple(y.shape)}")
            z = y
            y = torch.cat([-z, z], dim=-1)      # (B, 2)

        self._logits = y                         # (B, out_dim)
        return y

    def get_preds(self) -> Dict[str, torch.Tensor]:
        return {self.task: self._logits} if self._logits is not None else {}

    def get_activations(self) -> Dict[str, torch.Tensor]:
        return self.transformation.get_activations()

@register("head", "regression")
class RegressionHead(QueryInterface, nn.Module):
    def __init__(
        self,
        in_dim: int,
        task: str,
        out_dim: int = 1,
        activation_function: Optional[str] = None,
        local_name: Optional[str] = None,
    ):
        super().__init__()
        act = activation_function if isinstance(activation_function, str) else "identity"
        self.transformation = get_core("feed-forward_layer")(
            in_dim=in_dim,
            out_dim=out_dim,
            activation_function=act,
            local_name=local_name or f"{task}.head",
        )
        self.task = task
        self._out: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._out = self.transformation(x)
        return self._out

    def get_preds(self) -> Dict[str, torch.Tensor]:
        return {self.task: self._out} if self._out is not None else {}

    def get_activations(self) -> Dict[str, torch.Tensor]:
        return self.transformation.get_activations()

# ########################################################################### #
# Graph Neural Network (GNN) Components:
# ########################################################################### #

# =========================================================================== #
# Structural Encoders:
# =========================================================================== #

def _device_of(data: Data) -> torch.device:
    # prefer data.x; fall back to edge_index
    if hasattr(data, "x") and isinstance(data.x, torch.Tensor):
        return data.x.device
    return data.edge_index.device

def _split_ptr(batch: Batch) -> Sequence[Tuple[int, int]]:
    """
    For a PyG Batch, return node index slices [(s0,e0), (s1,e1), ...]
    using Batch.ptr when available; falls back to bincount otherwise.
    """
    if hasattr(batch, "ptr") and batch.ptr is not None:
        ptr = batch.ptr
        return [(int(ptr[i].item()), int(ptr[i + 1].item())) for i in range(ptr.numel() - 1)]

    # fallback: contiguous blocks per graph id
    b = batch.batch
    sizes = torch.bincount(b)
    csum = torch.cumsum(sizes, dim=0)
    starts = torch.cat([torch.zeros(1, dtype=torch.long, device=b.device), csum[:-1]])
    return [(int(s.item()), int(e.item())) for s, e in zip(starts, csum)]

def _local_edge_index(edge_index: torch.Tensor, s: int, e: int) -> torch.Tensor:
    """Return a 2×E' edge_index for the subgraph with node ids in [s, e)."""
    src, dst = edge_index
    mask = (src >= s) & (src < e) & (dst >= s) & (dst < e)
    if not mask.any():
        return torch.empty(2, 0, dtype=torch.long, device=edge_index.device)
    sub_src = src[mask] - s
    sub_dst = dst[mask] - s
    return torch.stack([sub_src, sub_dst], dim=0)

def _adjacency_from_edge_index(edge_index: torch.Tensor, n: int, *, undirected: bool = True) -> torch.Tensor:
    """
    Build a dense 0/1 adjacency matrix A (n×n) from edge_index.
    If the input already contains both directions, this is still safe.
    """
    A = torch.zeros((n, n), dtype=torch.float32, device=edge_index.device)
    if edge_index.numel() == 0:
        return A
    i, j = edge_index
    A[i, j] = 1.0
    if undirected:
        A[j, i] = 1.0
    A.clamp_(0.0, 1.0)  # in case of duplicates
    return A

def _block_cat(rows: Sequence[torch.Tensor]) -> torch.Tensor:
    """Concatenate per-graph feature blocks (along node axis)."""
    if len(rows) == 0:
        return torch.zeros(0, 0)
    return torch.cat(rows, dim=0)

@register("core", "degree_features")
@torch.no_grad()
def degree_features(data: Data, *, normalize: Optional[str] = None) -> torch.Tensor:
    """
    Degree feature per node.

    Parameters
    ----------
    normalize : None | "max" | "log" | "l2"
        Normalization per graph:
            "max": d / max(d)
            "log": log(1 + d)
            "l2" : d / ||d||_2

    Returns
    -------
    Tensor  shape (N, 1)
    """
    dev = _device_of(data)
    if isinstance(data, Batch):
        b = data.batch
        deg = tg_degree(data.edge_index[0], data.num_nodes).to(dev)
        out = deg.unsqueeze(-1)
        if normalize == "max":
            sizes = torch.bincount(b)
            out_list = []
            start = 0
            for n in sizes.tolist():
                block = out[start : start + n]
                m = block.max().clamp_min(1.0)
                out_list.append(block / m)
                start += n
            out = torch.cat(out_list, dim=0)
        elif normalize == "log":
            out = torch.log1p(out)
        elif normalize == "l2":
            sizes = torch.bincount(b)
            out_list, start = [], 0
            for n in sizes.tolist():
                block = out[start : start + n]
                denom = torch.linalg.norm(block) + 1e-8
                out_list.append(block / denom)
                start += n
            out = torch.cat(out_list, dim=0)
        return out

    # single graph
    deg = tg_degree(data.edge_index[0], data.num_nodes).to(dev).unsqueeze(-1)
    if normalize == "max":
        deg = deg / deg.max().clamp_min(1.0)
    elif normalize == "log":
        deg = torch.log1p(deg)
    elif normalize == "l2":
        deg = deg / (torch.linalg.norm(deg) + 1e-8)
    return deg


@register("core", "laplacian_positional_encoding")
@torch.no_grad()
def laplacian_positional_encoding(
    data: Data,
    k: int = 8,
    *,
    normalization: str = "sym",        # {"sym", "rw"}
    exclude_trivial: bool = True,
) -> torch.Tensor:
    """
    Laplacian eigenvector positional encodings (LapPE).

    • For each graph, compute the k smallest non-trivial eigenvectors of the
      (normalised) Laplacian and pad with zeros if needed.
    • Sign ambiguity is fixed by flipping eigenvectors so that the sum ≥ 0.

    Returns
    -------
    Tensor  shape (N, k)
    """
    def _one_graph(A: torch.Tensor, k_: int) -> torch.Tensor:
        n = A.size(0)
        if n == 0:
            return torch.zeros(0, k_, dtype=torch.float32)
        # degree
        d = A.sum(dim=1)
        I = torch.eye(n, dtype=torch.float32, device=A.device)
        if normalization == "sym":
            inv_sqrt = torch.where(d > 0, d.pow(-0.5), torch.zeros_like(d))
            Dinv2 = torch.diag(inv_sqrt)
            L = I - Dinv2 @ A @ Dinv2
        elif normalization == "rw":
            inv = torch.where(d > 0, d.reciprocal(), torch.zeros_like(d))
            Dinv = torch.diag(inv)
            L = I - Dinv @ A
        else:
            raise ValueError("normalization must be 'sym' or 'rw'")

        # eigh is more stable on CPU for very small matrices
        L_cpu = L.detach().cpu()
        evals, evects = torch.linalg.eigh(L_cpu)   # ascending
        # drop the trivial eigenvector (λ≈0) if requested
        start = 1 if exclude_trivial and n > 1 else 0
        ev = evects[:, start : start + k_]         # (n, <=k)
        # pad to k
        if ev.size(1) < k_:
            pad = torch.zeros(n, k_ - ev.size(1), dtype=torch.float32, device=L_cpu.device)
            ev = torch.cat([ev, pad], dim=1)
        # sign fix
        sign = torch.sign(ev.sum(dim=0, keepdim=True)).clamp(min=1.0)
        ev = ev * sign
        return ev.to(A.device)

    dev = _device_of(data)
    if isinstance(data, Batch):
        blocks = []
        for s, e in _split_ptr(data):
            n = e - s
            E_sub = _local_edge_index(data.edge_index, s, e)
            A = _adjacency_from_edge_index(E_sub, n)
            blocks.append(_one_graph(A, k))
        return _block_cat(blocks).to(dev)

    # single graph
    A = _adjacency_from_edge_index(data.edge_index, data.num_nodes)
    return _one_graph(A, k).to(dev)


@register("core", "random_walk_structural_encoding")
@torch.no_grad()
def random_walk_structural_encoding(
    data: Data,
    walk_lengths: Sequence[int] = (1, 2, 3, 4, 5, 6, 7, 8),
) -> torch.Tensor:
    """
    Random Walk Structural Encodings (RWSE / RW-Landing probabilities).

    For each node i and step t, feature is  (P^t)_{ii}  where
    P = D^{-1} A is the random-walk transition matrix.

    Returns
    -------
    Tensor  shape (N, len(walk_lengths))
    """
    def _one_graph(A: torch.Tensor, steps: Sequence[int]) -> torch.Tensor:
        n = A.size(0)
        if n == 0:
            return torch.zeros(0, len(steps), dtype=torch.float32)
        d = A.sum(dim=1)
        inv = torch.where(d > 0, d.reciprocal(), torch.zeros_like(d))
        P = torch.diag(inv) @ A
        # iterative powers
        feats = []
        Pk = P.clone()
        if len(steps) == 0:
            return torch.zeros(n, 0, dtype=torch.float32, device=A.device)
        max_t = max(steps)
        diag_list = {}
        for t in range(1, max_t + 1):
            if t == 1:
                Pk = P
            else:
                Pk = Pk @ P
            diag_list[t] = torch.diag(Pk)
        for t in steps:
            if t == 0:
                feats.append(torch.ones(n, device=A.device))   # (P^0)_{ii} = 1
            else:
                feats.append(diag_list[t])
        return torch.stack(feats, dim=1).float()  # (n, T)

    dev = _device_of(data)
    if isinstance(data, Batch):
        blocks = []
        for s, e in _split_ptr(data):
            n = e - s
            E_sub = _local_edge_index(data.edge_index, s, e)
            A = _adjacency_from_edge_index(E_sub, n)
            blocks.append(_one_graph(A, walk_lengths))
        return _block_cat(blocks).to(dev)
    A = _adjacency_from_edge_index(data.edge_index, data.num_nodes)
    return _one_graph(A, walk_lengths).to(dev)


@register("core", "closeness_centrality")
@torch.no_grad()
def closeness_centrality(data: Data) -> torch.Tensor:
    """
    Closeness centrality per node:  (|C(i)| - 1) / sum_{j in C(i), j≠i} d(i,j)
    where C(i) is the connected component of node i.
    Uses Floyd–Warshall (O(N^3)) — fine for molecules.

    Returns
    -------
    Tensor  shape (N, 1)
    """
    def _one_graph(A: torch.Tensor) -> torch.Tensor:
        n = A.size(0)
        if n <= 1:
            return torch.zeros(n, 1, device=A.device)
        inf = torch.tensor(float("inf"), device=A.device)
        D = torch.full((n, n), inf, device=A.device)
        D.fill_diagonal_(0.0)
        nz = (A > 0)
        D[nz] = 1.0
        # Floyd–Warshall
        for k in range(n):
            D = torch.minimum(D, D[:, k:k+1] + D[k:k+1, :])
        # per-node
        reachable = (D < inf).float()
        comp_sizes = reachable.sum(dim=1)  # includes self
        sums = (D * reachable).sum(dim=1) - torch.diag(D)  # exclude self (0)
        denom = torch.where(sums > 0, sums, torch.ones_like(sums))
        closeness = (comp_sizes - 1.0) / denom
        return closeness.unsqueeze(-1)

    dev = _device_of(data)
    if isinstance(data, Batch):
        blocks = []
        for s, e in _split_ptr(data):
            n = e - s
            E_sub = _local_edge_index(data.edge_index, s, e)
            A = _adjacency_from_edge_index(E_sub, n)
            blocks.append(_one_graph(A))
        return _block_cat(blocks).to(dev)
    A = _adjacency_from_edge_index(data.edge_index, data.num_nodes)
    return _one_graph(A).to(dev)


@register("core", "eccentricity")
@torch.no_grad()
def eccentricity(data: Data) -> torch.Tensor:
    """
    Eccentricity per node:  max_j d(i, j)  within the connected component.

    Returns
    -------
    Tensor  shape (N, 1)
    """
    def _one_graph(A: torch.Tensor) -> torch.Tensor:
        n = A.size(0)
        if n <= 1:
            return torch.zeros(n, 1, device=A.device)
        inf = torch.tensor(float("inf"), device=A.device)
        D = torch.full((n, n), inf, device=A.device)
        D.fill_diagonal_(0.0)
        nz = (A > 0)
        D[nz] = 1.0
        for k in range(n):
            D = torch.minimum(D, D[:, k:k+1] + D[k:k+1, :])
        D[D == inf] = 0.0  # ignore unreachable nodes (other components)
        ecc = D.max(dim=1).values
        return ecc.unsqueeze(-1)

    dev = _device_of(data)
    if isinstance(data, Batch):
        blocks = []
        for s, e in _split_ptr(data):
            n = e - s
            E_sub = _local_edge_index(data.edge_index, s, e)
            A = _adjacency_from_edge_index(E_sub, n)
            blocks.append(_one_graph(A))
        return _block_cat(blocks).to(dev)
    A = _adjacency_from_edge_index(data.edge_index, data.num_nodes)
    return _one_graph(A).to(dev)


@register("core", "local_clustering_coefficient")
@torch.no_grad()
def local_clustering_coefficient(data: Data) -> torch.Tensor:
    """
    Local clustering coefficient per node:
        C_i = 2 * triangles(i) / (d_i * (d_i - 1)),  undirected case.

    Returns
    -------
    Tensor  shape (N, 1)
    """
    def _one_graph(A: torch.Tensor) -> torch.Tensor:
        n = A.size(0)
        if n <= 2:
            return torch.zeros(n, 1, device=A.device)
        d = A.sum(dim=1)
        # A^3 diagonal gives 2 * triangles for undirected simple graphs
        A3 = (A @ A @ A)
        tri2 = torch.diag(A3)                     # equals 2 * triangles(i)
        denom = d * (d - 1.0) + 1e-8
        Ci = tri2 / denom
        return Ci.unsqueeze(-1).clamp_(0.0, 1.0)

    dev = _device_of(data)
    if isinstance(data, Batch):
        blocks = []
        for s, e in _split_ptr(data):
            n = e - s
            E_sub = _local_edge_index(data.edge_index, s, e)
            A = _adjacency_from_edge_index(E_sub, n)
            blocks.append(_one_graph(A))
        return _block_cat(blocks).to(dev)
    A = _adjacency_from_edge_index(data.edge_index, data.num_nodes)
    return _one_graph(A).to(dev)


@register("core", "eigenvector_centrality")
@torch.no_grad()
def eigenvector_centrality(
    data: Data,
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> torch.Tensor:
    """
    Eigenvector centrality via power iteration on the (undirected) adjacency.

    Returns
    -------
    Tensor  shape (N, 1)
    """
    def _one_graph(A: torch.Tensor) -> torch.Tensor:
        n = A.size(0)
        if n == 0:
            return torch.zeros(0, 1, dtype=torch.float32, device=A.device)
        if n == 1:
            return torch.ones(1, 1, dtype=torch.float32, device=A.device)
        x = torch.ones(n, 1, dtype=torch.float32, device=A.device) / n
        for _ in range(max_iter):
            x_next = A @ x
            norm = torch.linalg.vector_norm(x_next) + 1e-12
            x_next = x_next / norm
            if torch.max(torch.abs(x_next - x)) < tol:
                x = x_next
                break
            x = x_next
        # normalise to [0, 1]
        x = x / (x.max().clamp_min(1e-12))
        return x

    dev = _device_of(data)
    if isinstance(data, Batch):
        blocks = []
        for s, e in _split_ptr(data):
            n = e - s
            E_sub = _local_edge_index(data.edge_index, s, e)
            A = _adjacency_from_edge_index(E_sub, n)
            blocks.append(_one_graph(A))
        return _block_cat(blocks).to(dev)
    A = _adjacency_from_edge_index(data.edge_index, data.num_nodes)
    return _one_graph(A).to(dev)


@register("core", "k_core_number")
@torch.no_grad()
def k_core_number(data: Data) -> torch.Tensor:
    """
    k-core number (coreness) for each node via degeneracy ordering.
    Simple O(N^2) implementation – perfectly fine for small molecules.

    Returns
    -------
    Tensor  shape (N, 1)
    """
    def _one_graph_edges_to_lists(E: torch.Tensor, n: int) -> Dict[int, set]:
        adj: Dict[int, set] = {i: set() for i in range(n)}
        if E.numel() == 0:
            return adj
        s, d = E
        for u, v in zip(s.tolist(), d.tolist()):
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def _one_graph(E: torch.Tensor, n: int) -> torch.Tensor:
        if n == 0:
            return torch.zeros(0, 1, dtype=torch.float32, device=E.device)
        adj = _one_graph_edges_to_lists(E, n)
        deg = {i: len(adj[i]) for i in range(n)}
        core = [0] * n
        remaining = set(range(n))
        while remaining:
            u = min(remaining, key=lambda i: deg[i])
            core[u] = deg[u]
            remaining.remove(u)
            for v in list(adj[u]):
                if v in remaining:
                    adj[v].discard(u)
                    deg[v] -= 1
            adj[u].clear()
        return torch.tensor(core, dtype=torch.float32, device=E.device).unsqueeze(-1)

    dev = _device_of(data)
    if isinstance(data, Batch):
        blocks = []
        for s, e in _split_ptr(data):
            n = e - s
            E_sub = _local_edge_index(data.edge_index, s, e)
            blocks.append(_one_graph(E_sub, n))
        return _block_cat(blocks).to(dev)
    return _one_graph(data.edge_index, data.num_nodes).to(dev)

@register("core", "structural_encoder")
@torch.no_grad()
def structural_encoder(names: Sequence[str]) -> Callable[[Data], torch.Tensor]:
    """
    Given a list of registry keys (strings), look each one up with `get_core`,
    and return a function that takes a `Data`/`Batch` and returns the
    concatenated structural encodings.

    Examples
    --------
    >>> comb = combine_structural_encodings([
    ...     "degree_features",
    ...     "laplacian_positional_encoding",
    ...     "random_walk_structural_encoding",
    ... ])
    >>> feats = comb(batch)   # (sum_i N_i, 1 + k + len(walk_lengths))
    """
    if not isinstance(names, (list, tuple)):
        raise TypeError("`names` must be a list/tuple of registry keys (str).")

    # Resolve (and validate) encoder functions from the 'core' registry
    encoder_fns: List[Callable[[Data], torch.Tensor]] = []
    for name in names:
        fn = get_core(name)  # may return classes or functions; we expect callables
        if not callable(fn):
            raise TypeError(f"Registry entry 'core.{name}' is not callable.")
        encoder_fns.append(fn)

    def _combined(data: Data) -> torch.Tensor:
        outs = [fn(data) for fn in encoder_fns]
        if not outs:
            dev = _device_of(data)
            return torch.empty(0, 0, device=dev)
        n = outs[0].size(0)
        if any(t.size(0) != n for t in outs):
            raise ValueError("All encoders must return tensors with the same number of rows (nodes).")
        return torch.cat(outs, dim=-1)

    return _combined

# =========================================================================== #
# Non-Architectural Stages:
# =========================================================================== #

# --------------------------------------------------------------------------- #
# GNN-Related Variabes:
# --------------------------------------------------------------------------- #

GNN_STAGES: List[str] = [
    # Stage 1:
    "preprocessor",
    # Stage 2:
    "node_embedder",
    # Stage 3:
    "edge_embedder",
    # Stage 4:
    "structural_encoder",
    # Stage 5:
    "message_fn",
    # Stage 6:
    "edge_attention",
    # Stage 7:
    "local_aggregator",
    # Stage 8:
    "global_attention",
    # Stage 9:
    "hybrid_router",
    # Stage 10:
    "node_update",
    # Stage 11:
    "edge_update",
    # Stage 12:
    "feed_forward",
    # Stage 13:
    "norm_reg",
    # Stage 14:
    "readout_pool",
    # Stage 15:
    "multiscale_agg",
    # Stage 16:
    "task_head",
]

# --------------------------------------------------------------------------- #
# Scatter helpers
# --------------------------------------------------------------------------- #

_SCATTER = {
    "sum":  lambda src, idx, n: scatter(src, idx, dim=0, dim_size=n, reduce="sum"),
    "mean": lambda src, idx, n: scatter(src, idx, dim=0, dim_size=n, reduce="mean"),
    "max":  lambda src, idx, n: scatter(src, idx, dim=0, dim_size=n, reduce="max"),
}

# --------------------------------------------------------------------------- #
# Updating Data Object Properties:
# --------------------------------------------------------------------------- #

def get_node_embeddings(data):
    return data.x

def get_edge_embeddings(data):
    return data.edge_attr

def get_structural_embeddings(data):
    return data.pos_enc

def update_node_embeddings(data, update):
    data.x = update
    return data

def update_edge_embeddings(data, update):
    data.edge_attr = update
    return data

def update_structural_embeddings(data, update):
    data.pos_enc = update
    return data

def get_src_dst_nodes(data):
    src, dst = data.edge_index
    return src, dst

# =========================================================================== #
# Stage 1 - Graph Preprocessing:
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Graph Processor Module:
# --------------------------------------------------------------------------- #

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

# =========================================================================== #
# Stage 2 - Node Encoder:
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Identity Block:
# --------------------------------------------------------------------------- #

# Note: valid for both node and edge encoding.

@register("core", "identity_transformation")
class IdentityTransformation(nn.Identity):
    def __init__(
        self,
        out_dim: Optional[int] = None,
    ):
        super().__init__()
        self.out_dim = out_dim

# --------------------------------------------------------------------------- #
# Basic Node Encoder:
# --------------------------------------------------------------------------- #

class NodeEncoder(nn.Module):
    covers = ("node_embedder",)

    def __init__(self, out_dim: int, scheme: str = "affine", in_dim = NODE_FEAT_DIM):
        super().__init__()
        
        if scheme == "identity":
            assert in_dim == out_dim, "Identity cannot change dimension."
        
        self.embed = get_core(scheme)(in_dim=in_dim, out_dim=out_dim)
        self.out_node_dim = out_dim

    def forward(self, data: Data) -> Data:
        """
        Idempotent encoder – converts 118-dim atom one-hot vectors to the
        hidden size exactly once per sample.  Subsequent passes in the same
        epoch see the already-embedded representation and skip re-encoding.
        """
        
        update_node_embeddings(data, self.embed(get_node_embeddings(data)))
        
        return data

# =========================================================================== #
# Stage 3 - Edge Encoder:
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Basic Edge Encoder:
# --------------------------------------------------------------------------- #

class EdgeEncoder(nn.Module):
    covers = ("edge_embedder",)

    def __init__(self, out_dim: int, scheme: str = "affine", in_dim = EDGE_FEAT_DIM):
        super(). __init__()
        
        if scheme == "identity":
            assert in_dim == out_dim, "Identity cannot change dimension."

        self.embed = get_core(scheme)(in_dim=in_dim, out_dim=out_dim)
        self.out_edge_dim = out_dim

    def forward(self, data: Data) -> Data:
        
        update_edge_embeddings(data, self.embed(get_edge_embeddings(data)))
        
        return data


# =========================================================================== #
# Stage 4 - Structural Encoder:
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Basic Structural Encoder:
# --------------------------------------------------------------------------- #

class StructuralEncoder(nn.Module):
    covers = ("structural_encoder",)

    def __init__(self, modes: List[str] = "none"):
        super().__init__()
        # TODO: Add a more robust functionality.
        self.structural_encoder = structural_encoder(modes)
    
    @torch.no_grad()
    def forward(self, data: Data) -> Data:
        update_structural_embeddings(data, self.structural_encoder(data))
        return data

# =========================================================================== #
# Stage 5 - Edge Update Function:
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Base Edge Update Class:
# --------------------------------------------------------------------------- #

"""
All edge update classes should inherit from this and only overwrite __init__.
"""

class EdgeUpdateBase(MessagePassing):
    covers = ("message_fn",)

    def forward(self, data: Data) -> Data:
        
        src, dst = get_src_dst_nodes(data)
        old_edge_embeddings = get_edge_embeddings(data)
        
        cat = torch.cat([data.x[src], data.x[dst], old_edge_embeddings], dim=-1)
        
        update_edge_embeddings(data, self.update_fn(cat))
        return data


# --------------------------------------------------------------------------- #
# Basic Edge Update:
# --------------------------------------------------------------------------- #

class EdgeMessageUpdate(EdgeUpdateBase):
    covers = ("message_fn",)

    def __init__(
        self,
        update_fn: callable
    ):
        super().__init__(aggr=None)
        self.update_fn = update_fn
    
# --------------------------------------------------------------------------- #
# FeedForwardNeuralNetwork Edge Update:
# --------------------------------------------------------------------------- #

class FFNEdgeUpdate(EdgeMessageUpdate):
    covers = ("message_fn",)

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Union[List[int], Tuple[int, ...]],
        batch_norms: Union[bool, Sequence[bool]] = False,
        dropout_probabilities: Union[float, Sequence[float]] = 0.0,
        activation_functions: Union[str, Sequence[str]] = DEFAULT_ACTIVATION_FUNCTION,
    ):
        update_fn = FeedForwardNeuralNetwork(in_dim = in_dim, 
                                             hidden_dims = hidden_dims,
                                             batch_norms = batch_norms,
                                             dropout_probabilities = dropout_probabilities,
                                             activation_functions = activation_functions
                                             )
        self.update_fn = update_fn
        super().__init__(update_fn)

# =========================================================================== #
# Stage 6 - Edge Attention:
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Base Edge Attention Class:
# --------------------------------------------------------------------------- #

"""
All edge attention classes should inherit from this and only overwrite __init__.
"""

class EdgeAttentionBase(nn.Module):
    covers = ("edge_attention",)

    def forward(self, data: Data) -> Data:
        h_s = self.src_transformation(get_node_embeddings(data))
        h_d = self.dst_transformation(get_node_embeddings(data))
        e = self.src_dst_transformation(h_s[data.edge_index[0]] + h_d[data.edge_index[1]])
        alpha = self.dropout(self.logit_transformation(e)).mean(-1, keepdim=True)
        update_edge_embeddings(data, get_edge_embeddings(data) * alpha)
        return data

# --------------------------------------------------------------------------- #
# Basic Edge Update Class:
# --------------------------------------------------------------------------- #


class EdgeAttention(EdgeAttentionBase):
    covers = ("edge_attention",)

    def __init__(self, src_transform: callable, dst_transform: callable, src_dst_transform: callable, logit_transform=partial(torch.softmax, dim=0), dropout_probability: float = 0):
        super().__init__()
        self.src_transformation = src_transform
        self.dst_transformation = dst_transform
        self.src_dst_transformation = src_dst_transform
        self.logit_transformation = logit_transform
        self.dropout = DropoutLayer(dropout_prob=dropout_probability)

# --------------------------------------------------------------------------- #
# Base Edge Update Class:
# --------------------------------------------------------------------------- #

class DefaultEdgeAttention(EdgeAttentionBase):
    covers = ("edge_attention",)

    def __init__(self, node_dim, heads, dropout):
        super().__init__()
        self.src_transformation = GeoLinear(node_dim, heads, bias=False)
        self.dst_transformation = GeoLinear(node_dim, heads, bias=False)
        self.src_dst_transformation = nn.LeakyReLU(0.2)
        self.logit_transformation = partial(torch.softmax, dim=0)
        self.dropout = DropoutLayer(dropout_prob=0)

# --------------------------------------------------------------------------- #
# Deprecated Module - DO NOT USE
# --------------------------------------------------------------------------- #

class DeprecatedEdgeAttention(nn.Module):
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

# =========================================================================== #
# Stage 7 - Local Aggregator:
# =========================================================================== #

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
    
# =========================================================================== #
# Stage 8 - Global Self Attention:
# =========================================================================== #

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

# =========================================================================== #
# Stage 9 - Hybrid Mixer:
# =========================================================================== #

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

# =========================================================================== #
# Stage 10 - Node Update:
# =========================================================================== #

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

# =========================================================================== #
# Stage 11 - Node Update:
# =========================================================================== #

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




# =========================================================================== #
# Stage 12 - Readout Pooler:
# =========================================================================== #

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

# =========================================================================== #
# Stage 13 - Multi Scale Aggregator:
# =========================================================================== #

class MultiScaleAggregator(nn.Module):
    covers = ("multiscale_agg",)
    def __init__(self, reduce: str = "mean"):
        super().__init__()
        self.pool = ReadoutPooler(reduce)  # defines attr_name

    def forward(self, data: Data, frag: Optional[torch.Tensor] = None) -> Data:
        if not hasattr(data, self.pool.attr_name):
            data = self.pool(data)
        g = getattr(data, self.pool.attr_name)
        if frag is not None:
            g = torch.cat([g, frag], dim=-1)
        setattr(data, self.pool.attr_name, g)
        return data

# Somwhere I messed up with numbering - oops!

# =========================================================================== #
# Stage 16 - Multi Scale Aggregator:
# =========================================================================== #

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
    missing = [s for s in GNN_STAGES if s not in stage_map]
    if missing:
        raise StageConfigError(f"Missing stages {missing}")
    unknown = [k for k in stage_map if k not in GNN_STAGES]
    if unknown:
        raise StageConfigError(f"Unknown keys {unknown}")
    cov = {k: False for k in GNN_STAGES}
    for key, mod in stage_map.items():
        covers = getattr(mod, "covers", (key,))
        for s in covers:
            if cov[s]:
                raise StageConfigError(f"Stage '{s}' covered twice")
            cov[s] = True


def build_pipeline(stage_map: Dict[str, nn.Module]) -> nn.ModuleList:
    _validate(stage_map)
    return nn.ModuleList([stage_map[key] for key in GNN_STAGES])


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
        self._stage_to_idx = {name: idx for idx, name in enumerate(GNN_STAGES)}

        self.output_dim = getattr(self.task_head, "out_dim", None)

    # ------------------------------------------------------------------ #
    # Dict‑style *read* access:  model["readout_pool"]  or  model[13]
    # ------------------------------------------------------------------ #
    def __getitem__(self, key: Union[str, int]) -> nn.Module:
        if isinstance(key, str):                          # by stage name
            if key not in GNN_STAGES:
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
            if key not in GNN_STAGES:
                raise KeyError(f"{key!r} is not a valid stage name.")
            setattr(self, key, value)                     # register attribute
            self.pipe[self._stage_to_idx[key]] = value    # keep execution list aligned

        # --- replace by index ---
        elif isinstance(key, int):
            if not (0 <= key < len(self.pipe)):
                raise IndexError("Stage index out of range.")
            stage_name = GNN_STAGES[key]
            setattr(self, stage_name, value)
            self.pipe[key] = value

        else:
            raise TypeError("Key must be a stage name (str) or an int index.")

        # Keep metadata in sync if the task head changes
        if (isinstance(key, str) and key == "task_head") or \
           (isinstance(key, int) and GNN_STAGES[key] == "task_head"):
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
# GNN Variants:
# =========================================================================== #

# =========================================================================== #
# Pre-Built Architecture Variants (updated for current class names/signatures)
#   Notes:
#   • NodeEncoder / EdgeEncoder now take (out_dim, scheme="affine", in_dim=…)
#     so we pass named args: out_dim=hidden, in_dim=<feature-dim>.
#   • Identity_Transformation  →  IdentityTransformation
#   • EdgeMLP (old)            →  EdgeMessageUpdate(FFN(...))
#   • EdgeAttention API        →  use DefaultEdgeAttention(node_dim, heads, dropout)
#   • Structural encoder stage →  temporarily Identity (as requested)
# =========================================================================== #

# TODO: Figure out why this class was built by ChatGPT and incorporate it into the existing codebase.
def _edge_mlp_for_dims(node_dim: int, edge_dim: int, hidden: int) -> EdgeMessageUpdate:
    """
    Build a message function that maps [x_src || x_dst || e] → e'
    with output dimension == node_dim so LocalAggregator can yield node_dim features.
    """
    in_dim = (2 * node_dim) + edge_dim
    ff = FeedForwardNeuralNetwork(
        in_dim=in_dim,
        # last layer = node_dim so that edge_attr has dimension==node_dim for aggregation
        hidden_dims=[hidden, hidden, node_dim],
        batch_norms=False,
        dropout_probabilities=0.0,
        activation_functions="relu",
        local_name="edge_mlp",
    )
    return EdgeMessageUpdate(update_fn=ff)

@register("core", "gcnn")
def GCNN(*, num_node_feats: int, num_edge_feats: int,
         hidden: int, n_tasks: int = 1) -> GraphNeuralNetwork:

    stage_map = {
        "preprocessor":       GraphPreprocessor(),
        "node_embedder":      NodeEncoder(out_dim=hidden, in_dim=num_node_feats, scheme="affine"),
        "edge_embedder":      EdgeEncoder(out_dim=hidden, in_dim=num_edge_feats,   scheme="affine"),
        "structural_encoder": IdentityTransformation(),  # ignore structural encodings for now

        "message_fn":         _edge_mlp_for_dims(node_dim=hidden, edge_dim=hidden, hidden=hidden),
        "edge_attention":     IdentityTransformation(),
        "local_aggregator":   LocalAggregator("sum"),
        "global_attention":   IdentityTransformation(),
        "hybrid_router":      HybridMixer(),
        "node_update":        NodeUpdate(hidden),
        "edge_update":        EdgeUpdate(node_dim=hidden, edge_dim=hidden, hid=hidden),

        "feed_forward":       IdentityTransformation(),
        "norm_reg":           IdentityTransformation(),
        "readout_pool":       ReadoutPooler("sum"),
        "multiscale_agg":     IdentityTransformation(),
        "task_head":          GraphVectorizer(in_dim=hidden),
    }
    return GraphNeuralNetwork(stage_map)

@register("core", "mpnn")
def MPNN(*, num_node_feats: int, num_edge_feats: int,
         hidden: int, n_tasks: int = 1) -> GraphNeuralNetwork:

    stage_map = {
        "preprocessor":       GraphPreprocessor(),
        "node_embedder":      NodeEncoder(out_dim=hidden, in_dim=num_node_feats, scheme="affine"),
        "edge_embedder":      EdgeEncoder(out_dim=hidden, in_dim=num_edge_feats,   scheme="affine"),
        "structural_encoder": IdentityTransformation(),

        "message_fn":         _edge_mlp_for_dims(node_dim=hidden, edge_dim=hidden, hidden=hidden),
        "edge_attention":     IdentityTransformation(),
        "local_aggregator":   LocalAggregator("sum"),
        "global_attention":   IdentityTransformation(),
        "hybrid_router":      HybridMixer(),
        "node_update":        NodeUpdate(hidden),
        "edge_update":        IdentityTransformation(),  # classic MPNN variant keeps this as identity

        "feed_forward":       IdentityTransformation(),
        "norm_reg":           IdentityTransformation(),
        "readout_pool":       ReadoutPooler("sum"),
        "multiscale_agg":     IdentityTransformation(),
        "task_head":          GraphVectorizer(in_dim=hidden),
    }
    return GraphNeuralNetwork(stage_map)

@register("core", "dmpnn")
def DMPNN(*, num_node_feats: int, num_edge_feats: int,
          hidden: int, n_tasks: int = 1) -> GraphNeuralNetwork:

    stage_map = {
        "preprocessor":       GraphPreprocessor(),
        "node_embedder":      NodeEncoder(out_dim=hidden, in_dim=num_node_feats, scheme="affine"),
        "edge_embedder":      EdgeEncoder(out_dim=hidden, in_dim=num_edge_feats,   scheme="affine"),
        "structural_encoder": IdentityTransformation(),

        "message_fn":         _edge_mlp_for_dims(node_dim=hidden, edge_dim=hidden, hidden=hidden),
        "edge_attention":     IdentityTransformation(),
        "local_aggregator":   LocalAggregator("sum"),
        "global_attention":   IdentityTransformation(),
        "hybrid_router":      HybridMixer(),
        "node_update":        NodeUpdate(hidden),
        "edge_update":        EdgeUpdate(node_dim=hidden, edge_dim=hidden, hid=hidden),

        "feed_forward":       IdentityTransformation(),
        "norm_reg":           IdentityTransformation(),
        "readout_pool":       ReadoutPooler("sum"),
        "multiscale_agg":     IdentityTransformation(),
        "task_head":          GraphVectorizer(in_dim=hidden),
    }
    return GraphNeuralNetwork(stage_map)

@register("core", "attfp")
def ATTFP(*, num_node_feats: int, num_edge_feats: int,
          hidden: int, n_tasks: int = 1,
          heads: int = 4, dropout: float = 0.1) -> GraphNeuralNetwork:

    stage_map = {
        "preprocessor":       GraphPreprocessor(),
        "node_embedder":      NodeEncoder(out_dim=hidden, in_dim=num_node_feats, scheme="affine"),
        "edge_embedder":      EdgeEncoder(out_dim=hidden, in_dim=num_edge_feats,   scheme="affine"),
        "structural_encoder": IdentityTransformation(),

        "message_fn":         IdentityTransformation(),                     # edges already embedded
        "edge_attention":     DefaultEdgeAttention(node_dim=hidden, heads=heads, dropout=dropout),
        "local_aggregator":   LocalAggregator("sum"),
        "global_attention":   IdentityTransformation(),
        "hybrid_router":      HybridMixer(),
        "node_update":        NodeUpdate(hidden),
        "edge_update":        IdentityTransformation(),

        "feed_forward":       IdentityTransformation(),
        "norm_reg":           IdentityTransformation(),
        "readout_pool":       ReadoutPooler("sum"),
        "multiscale_agg":     IdentityTransformation(),
        "task_head":          GraphVectorizer(in_dim=hidden),
    }
    return GraphNeuralNetwork(stage_map)

@register("core", "gin")
def GIN(*, num_node_feats: int,
        hidden: int, n_tasks: int = 1) -> GraphNeuralNetwork:
    """
    Simple GIN-style setup within this staged framework:
    - no dedicated edge features (we keep the 1‑dim zeros from the preprocessor),
    - message function constructs edge messages from node pairs,
    - sum aggregate.
    """
    # Edge preprocessor leaves E×1 zeros; make message fn output node_dim
    edge_in_dim = 1

    stage_map = {
        "preprocessor":       GraphPreprocessor(),
        "node_embedder":      NodeEncoder(out_dim=hidden, in_dim=num_node_feats, scheme="affine"),
        "edge_embedder":      IdentityTransformation(),  # keep E×1 zeros coming from the preprocessor
        "structural_encoder": IdentityTransformation(),

        # in_dim = 2*hidden + 1; out_dim = hidden (last FFN size)
        "message_fn":         _edge_mlp_for_dims(node_dim=hidden, edge_dim=edge_in_dim, hidden=hidden),
        "edge_attention":     IdentityTransformation(),
        "local_aggregator":   LocalAggregator("sum"),
        "global_attention":   IdentityTransformation(),
        "hybrid_router":      HybridMixer(),
        "node_update":        NodeUpdate(hidden),
        "edge_update":        IdentityTransformation(),

        "feed_forward":       IdentityTransformation(),
        "norm_reg":           IdentityTransformation(),
        "readout_pool":       ReadoutPooler("sum"),
        "multiscale_agg":     IdentityTransformation(),
        "task_head":          GraphVectorizer(in_dim=hidden),
    }
    return GraphNeuralNetwork(stage_map)

@register("core", "gtr")
def GTR(*, num_node_feats: int, num_edge_feats: int,
        hidden: int, n_tasks: int = 1,
        heads: int = 8, dropout: float = 0.1) -> GraphNeuralNetwork:

    stage_map = {
        "preprocessor":       GraphPreprocessor(),
        "node_embedder":      NodeEncoder(out_dim=hidden, in_dim=num_node_feats, scheme="affine"),
        "edge_embedder":      EdgeEncoder(out_dim=hidden, in_dim=num_edge_feats,   scheme="affine"),
        "structural_encoder": IdentityTransformation(),  # ignoring structural encodings for now

        "message_fn":         IdentityTransformation(),
        "edge_attention":     IdentityTransformation(),
        "local_aggregator":   IdentityTransformation(),
        "global_attention":   GlobalSelfAttention(hidden, heads=heads, drop=dropout),
        "hybrid_router":      HybridMixer(),
        "node_update":        NodeUpdate(hidden),
        "edge_update":        IdentityTransformation(),

        "feed_forward":       IdentityTransformation(),
        "norm_reg":           IdentityTransformation(),
        "readout_pool":       ReadoutPooler("sum"),
        "multiscale_agg":     IdentityTransformation(),
        "task_head":          GraphVectorizer(in_dim=hidden),
    }
    return GraphNeuralNetwork(stage_map)

# ########################################################################### #
# Other Network Modules:
# ########################################################################### #

# =========================================================================== #
# Minimol Encoder:
# =========================================================================== #

@register("core", "minimol_encoder")
class MinimolEncoder(IdentityTransformation, nn.Module):
    """
    Wrapper for identity that currently just passes through a minimol embedding.
    Function is currently just for clarity later on in the pipeline because 
    Minimol fingerprints are currently precomputed.
    """
    pass

# =========================================================================== #
# Concatentation Block:
# =========================================================================== #

@register("core", "concatenate")
class Concatenate(IdentityTransformation, Identity):
    """
    Wrapper of identity that is used to designate a concatentation. This is
    just used for clarity because the concatentation is automatic.
    """
    pass

# ########################################################################### #
# 2. Multi-task Learning Mixers:
# ########################################################################### #

# =========================================================================== #
# Standard Architectures:
# =========================================================================== #

@register("mixer", "shared_bottom_mixer")
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


@register("mixer", "cross_stitch_mixer")
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
#@register("mixer", "sluice")
class SluiceArch:
    def __init__(self,
                 task_name,
                 encoder_class,
                 decoders,
                 rep_grad: bool = False,
                 multi_input: bool = False,
                 device="cpu",
                 *,
                 sliced: int | tuple[int, ...] = 2, # slice --> sliced to avoid python reserved keyword.
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
        kwargs.update({"slice": sliced})
        if block_share is not None:
            kwargs["block_share"] = block_share
        super().__init__(task_name, encoder_class, decoders,
                         rep_grad, multi_input, device, **kwargs)


# --------------------------------------------------------------------------- #
# Vector‑friendly Mixture‑of‑Experts (MMoE)
# --------------------------------------------------------------------------- #

# @register("mixer", "mmoe")
# class MMoEArch(MMoE):
#     """
#     LibMTL’s MMoE expects a 4‑D “image” tensor.  This wrapper:

#         • *automatically* reshapes any 2‑D feature tensor `(B, D)` to
#           `(B, D, 1, 1)` **before** LibMTL sees it;

#         • runs the standard MMoE logic (experts, gates, decoders); and

#         • squeezes the spatial dummy dimensions **afterwards** so that every
#           task head once again receives / returns a flat `(B, D_out)` tensor.

#     The outside world therefore keeps the clean **vector → vector** invariant.
#     """

#     # ------------- construction ------------------------------------------
#     def __init__(self,
#                  task_name,
#                  encoder_class,
#                  decoders,
#                  rep_grad: bool = False,
#                  multi_input: bool = False,
#                  device="cpu",
#                  *,
#                  img_size: tuple[int, int, int] | None = None,
#                  num_experts: int = 4,
#                  **kwargs):
        
#         if isinstance(num_experts, int):
#             num_experts = [num_experts]

#         # If the caller did not pass img_size we infer **C** (= feature dim)
#         # from the *first* decoder; H,W are set to 1.
#         if img_size is None:
#             # grab the first decoder just to discover its input dimension
#             sample_dec = next(iter(decoders.values()))
#             in_dim = getattr(sample_dec, "in_dim", None)
#             if in_dim is None:
#                 raise ValueError(
#                     "`img_size` could not be inferred automatically because "
#                     "the decoder does not expose `.in_dim`.  Pass it manually."
#                 )
#             img_size = (in_dim, 1, 1)

#         kwargs.update({"img_size": img_size, "num_experts": num_experts})
#         super().__init__(task_name, encoder_class, decoders,
#                          rep_grad, multi_input, device, **kwargs)

#     # ------------- private helpers ---------------------------------------
#     @staticmethod
#     def _vec2img(x: torch.Tensor):
#         # only promote *exactly* 2‑D feature matrices, leave everything else intact
#         return x.unsqueeze(-1).unsqueeze(-1) if (isinstance(x, torch.Tensor) and x.ndim == 2) else x

#     @staticmethod
#     def _img2vec(x: torch.Tensor) -> torch.Tensor:
#         """Inverse of `_vec2img` for a `(B, D, 1, 1)` tensor."""
#         return x.view(x.size(0), -1) if x.ndim == 4 and x.size(2) == 1 else x

#     # ------------- forward ------------------------------------------------
#     def forward(self, x):
#         """
#         Accepts *either* a single tensor or a mapping `{task -> tensor}` if
#         `multi_input=True`, exactly like the parent class – but vectors are
#         silently upgraded to pseudo‑images.
#         """
        
#         # >>> NEW: make sure each expert gets an independent copy
#         if isinstance(x, dict):
#             # dict style (our hybrid encoder): deep‑copy every value
#             x = {k: copy.deepcopy(v) for k, v in x.items()}
#         else:                # graph or tensor
#             x = copy.deepcopy(x)
#         # <<< -----------------------------------------------

        
#         # ---- reshape the *inputs* --------------------------------------
#         if isinstance(x, dict):
#             x = {t: self._vec2img(v) for t, v in x.items()}
#         else:
#             x = self._vec2img(x)

#         # ---- call the LibMTL implementation ---------------------------
#         out = super().forward(x)              # dict {task -> tensor}

#         # ---- bring every task back to flat (B, D_out) ------------------
#         for t, v in out.items():
#             out[t] = self._img2vec(v)

#         return out

# ########################################################################### #
# 3. Model Builders:
# ########################################################################### #

# =========================================================================== #
# Architecture Buidler Helper:
# =========================================================================== #

class MTLArchitectureBuilder:
    """
    Base class that factorises the common steps:

        • look up encoder / head classes
        • build the per‑task decoders
        • call the LibMTL wrapper
    """

    mixer_key: str                 # override in subclasses

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
        ArchWrapper: Type[nn.Module] = get_mixer(self.mixer_key)

        return ArchWrapper(
            self.task_names,
            encoder_cls,                 # <‑ only *class*, LibMTL instantiates
            decoders,
            rep_grad   = self.rep_grad,
            multi_input= self.multi_input,
            device     = self.device,
            **self.arch_kwargs,
        )
    
    def get_tasks(self):
        
        return self.task_names
    
    def get_default_task(self):
        
        return self.get_tasks()[0]

# =========================================================================== #
# LibMTL Multitask Model Builders:
# =========================================================================== #

@register("architecture", "shared_parameter_model")
class SharedBottomBuilder(MTLArchitectureBuilder):
    """Builder for classic hard‑parameter sharing (LibMTL.HPS)."""
    mixer_key = "shared_bottom_mixer"

@register("architecture", "cross_stitch")
class CrossStitchBuilder(MTLArchitectureBuilder):
    """Builder for Cross‑Stitch Networks."""
    mixer_key = "cross_stitch_mixer"

@register("architecture", "sluice")
class SluiceBuilder(MTLArchitectureBuilder):
    """Builder for Sluice Networks."""
    mixer_key = "sluice_network_mixer"

@register("architecture", "MMoE")
class MMoEBuilder(MTLArchitectureBuilder):
    """Builder for Multi‑gate Mixture of Experts."""
    mixer_key = "mmoe_mixer"
    
# =========================================================================== #
# Identity Mixer Builders:
# =========================================================================== #

@register("architecture", "identity")
class IdentityBuilder(MTLArchitectureBuilder):
    """Builder that selects the IdentityArch (no MTL sharing)."""
    mixer_key = "identity"
    
# =========================================================================== #
# Single-Task Model Builder:
# =========================================================================== #

# ------------------------------ STL wrapper ---------------------------------
class _SingleTaskModule(nn.Module):
    """
    Wrap a backbone/encoder and a single head; forward returns {task: preds}.
    Works whether the encoder returns a Tensor or a PyG Data with .graph_feat.
    """
    def __init__(self, encoder: nn.Module, head: nn.Module, task_name: str = "main"):
        super().__init__()
        self.encoder = encoder
        self.head    = head
        self.task    = task_name

    def forward(self, inputs):
        z = self.encoder(inputs)
        # Accept either Tensor or PyG Data with pooled features
        if isinstance(z, Data):
            if hasattr(z, "graph_feat"):
                feats = z.graph_feat
            elif hasattr(z, "x"):  # last-resort fallback
                feats = z.x
            else:
                raise RuntimeError("Encoder returned Data without 'graph_feat' or 'x'.")
        else:
            feats = z
        out = self.head(feats)
        return {self.task: out}


# ------------------------------ STL builder ---------------------------------
@register("architecture", "single_task")
class SingleTaskBuilder:
    """
    Build a single-task model from the 'core' registry:
        encoder = get_core(encoder_name)(**encoder_kwargs)
        head    = get_head(head_name)(task=..., **head_kwargs)
    Wrapped to return {task_name: preds}.
    """
    def __init__(
        self,
        *,
        task_name: str = "main",
        encoder_name: str,                 # key in the 'core' registry (e.g., "gtr", "gcnn", "gin")
        encoder_kwargs: Dict | None = None,
        head_name: str = "regression",     # key in the 'head' registry
        head_kwargs: Dict | None = None,
        device: str | torch.device = "cpu",
    ):
        self._task_name = task_name
        self._enc_name  = encoder_name
        self._enc_kw    = encoder_kwargs or {}
        self._head_name = head_name
        self._head_kw   = head_kwargs or {}
        self._device    = device

    def get_tasks(self) -> List[str]:
        return [self._task_name]

    def get_default_task(self) -> str:
        return self._task_name

    def build(self) -> nn.Module:
        encoder = self._resolve_encoder(self._enc_name, self._enc_kw).to(self._device)
        HeadCls = get_head(self._head_name)
        head    = HeadCls(task=self._task_name, **self._head_kw).to(self._device)
        return _SingleTaskModule(encoder, head, self._task_name)

    @staticmethod
    def _resolve_encoder(name: str, kwargs: Dict) -> nn.Module:
        """
        Resolve ONLY from the 'core' registry and instantiate with kwargs.
        """
        try:
            Enc = get_core(name)          # class or factory from 'core'
        except KeyError as e:
            # nicer error with available keys
            available = ", ".join(sorted(_REGISTRIES["core"].keys()))
            raise KeyError(f"Unknown core encoder '{name}'. Available: {available}") from e

        enc = Enc(**(kwargs or {}))       # instantiate (works for class or factory)
        if not isinstance(enc, nn.Module):
            raise TypeError(
                f"Core encoder '{name}' must return an nn.Module, got {type(enc).__name__}."
            )
        return enc


    
# --------------------------------------------------------------------------- #
# Process Model Config:
# --------------------------------------------------------------------------- #

def process_model_config(cfg: Dict):
    """
    Supports both MTL builders (e.g., 'shared_parameter_model', 'cross_stitch')
    and STL builder ('single_task').
    """
    Builder = get_from_registry("architecture", cfg["name"])
    return Builder(**cfg.get("kwargs", {}))

# ########################################################################### #
# 4.  Weighting  – task‑level gradient / loss balancers
# ########################################################################### #

class AbsWeighting(abc.ABC):
    """
    Base API: subclasses *must* implement ``backward(task_losses)``.
    """

    def __init__(self, shared_params: Sequence[nn.Parameter], **cfg):
        self.shared_params = list(shared_params)  # keep for LibMTL wrappers

    @abc.abstractmethod
    def backward(self, task_losses: List[torch.Tensor]) -> torch.Tensor: ...


@register("weighting", "identity")
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

# --------------------------------------------------------------------------- #
# Process Weighting Config:
# --------------------------------------------------------------------------- #

def process_weighting_config(cfg: Dict, shared_params) -> AbsWeighting:
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
# 5.  Loss functions:
###############################################################################

# =========================================================================== #
# Classification Error Loss Functions:
# =========================================================================== #

@register("loss", "bce_with_logits")
def binary_cross_entropy(pred, target, **_):
    target = target.float().squeeze(-1)         # BCE expects float targets
    return F.binary_cross_entropy_with_logits(pred.squeeze(-1), target)

@register("loss", "cross_entropy")
def cross_entropy(pred: torch.Tensor, target: torch.Tensor, **_):
    return F.cross_entropy(pred, target)

# =========================================================================== #
# Regression Error Loss Functions:
# =========================================================================== #

@register("loss", "mse")
def mean_squared_error(pred: torch.Tensor, target: torch.Tensor, **_):
    return F.mse_loss(pred, target)

# =========================================================================== #
# Model Weights Loss Functions:
# =========================================================================== #

@register("loss", "l1_penalty")
def l1_penalty(model: nn.Module, **_):
    params = (model.get_learnable_weights().values()
              if hasattr(model, "get_learnable_weights")
              else (p for p in model.parameters() if p.requires_grad))
    return sum(p.abs().sum() for p in params)

@register("loss", "l2_penalty")
def l2_penalty(model: nn.Module, **_):
    params = (model.get_learnable_weights().values()
              if hasattr(model, "get_learnable_weights")
              else (p for p in model.parameters() if p.requires_grad))
    return sum((p ** 2).sum() for p in params)

# =========================================================================== #
# Loss Aggregators:
# =========================================================================== #

@dataclass
class LossTerm:
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


class LossFunction(nn.Module):
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


def process_loss_config(cfg_list: List[Dict]) -> LossFunction:
    """
    Parse a YAML section such as

        losses:
          - {name: cross_entropy, task: taskA, weight: 1.0}
          - {name: cross_entropy, task: taskB, weight: 1.0}
          - {name: l2_penalty,    weight: 1e-4}

    into a LossStack instance.
    """
    adapters: List[LossTerm] = []
    for item in cfg_list:
        fn = get_from_registry("loss", item["name"])
        adapters.append(
            LossTerm(
                name   = item["name"],
                task   = item.get("task"),
                weight = item.get("weight", 1.0),
                fn     = fn,
            )
        )
    return LossFunction(adapters)

# ########################################################################### #
# 6. Optimizers:
# ########################################################################### #

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


def process_optimizer_config(cfg: Dict, params):
    """
    Example cfg:

        optim:
          name: adamw
          kwargs: {lr: 1e-4, weight_decay: 1e-2 }
    """
    cls = get_from_registry("optim", cfg["name"])
    builder = cls(**cfg.get("kwargs", {}))
    return builder(params)      # → (optimizer, scheduler | None)

# ########################################################################### #
# 7. Performance Metrics:
# ########################################################################### #

class MetricModule(abc.ABC):

    def __init__(self, task: str):
        self.task = task

    def get_preds(self, task, preds):
        return preds.get(task)
    
    def get_targets(self, task, targets):
        return targets.get(task)
    
    def who(self) -> str:
        return getattr(self, "name", self.__class__.__name__)

    @abc.abstractmethod
    def update(self, preds, targets) -> None: ...
    
    @abc.abstractmethod
    def metric_on_batch(self) -> float: ...
    
    @abc.abstractmethod
    def metric_on_epoch(self) -> float: ...

    @abc.abstractmethod
    def reset(self) -> None: ...
    
# 3) Online (additive) metrics
@register("metric", "accuracy")
class AccuracyMetric(MetricModule):
    def __init__(self, task: str):
        super().__init__(task=task)
        self.running_total_num_correct = 0
        self.number_instances = 0

    def update(self, preds, targets) -> None:
        preds = self.get_preds(self.task, preds)
        targets = self.get_targets(self.task, targets)
        yhat = preds.detach().float().argmax(dim=-1)
        targets = targets.detach().view(-1).to(torch.int64)
        correct = (yhat.view_as(targets) == targets).sum().item()
        n = targets.numel()
        self.running_total_num_correct += correct
        self.number_instances += n
        self.last_batch_value = float(correct / max(n, 1))

    def metric_on_batch(self) -> float:
        return self.last_batch_value

    def metric_on_epoch(self) -> float:
        return float(self.running_total_num_correct / max(self.number_instances, 1))

    def reset(self) -> None:
        self.running_total_num_correct = 0
        self.number_instances = 0
        self.last_batch_value = None


@register("metric", "rmse")
class RMSEMetric(MetricModule):
    def __init__(self, task):
        super().__init__(task=task)
        self.running_total_sum_of_squares = 0.0
        self.number_instances = 0

    def update(self, preds, targets) -> None:
        preds = self.get_preds(self.task, preds)
        targets = self.get_targets(self.task, targets)
        preds = preds.detach().float().view(-1)
        targets = targets.detach().float().view(-1)
        diff = preds - targets
        sse = float((diff * diff).sum().item())
        batch_instances = targets.numel()
        self.running_total_sum_of_squares += sse
        self.number_instances += batch_instances
        self.last_batch_mse = float(torch.sqrt(torch.tensor(sse / max(batch_instances, 1))).item())
        
    def metric_on_batch(self) -> float:
        return self.last_batch_mse

    def metric_on_epoch(self) -> float:
        return float(math.sqrt(self.running_total_sum_of_squares / max(self.number_instances, 1)))

    def reset(self) -> None:
        self.running_total_sum_of_squares = 0.0
        self.number_instances = 0
        self.last_batch_mse = None


# 4) Offline (epoch-level) metrics — AUROC / AUPRC
@register("metric", "auroc")
class AUROCMetric(MetricModule):
   
    def __init__(self,
                 *,
                 task: str,
                 epoch_size: int,
                 num_classes: int = 2,
                 dtype: torch.dtype = torch.float16,
                 average_method: str = None, 
                 multi_class_method: str = "OvR"
                 ):
        super().__init__(task=task)
        self.epoch_size = epoch_size
        self.dtype = dtype
        self.num_classes = num_classes
        self.epoch_scores = torch.empty((epoch_size, self.num_classes), dtype=self.dtype, device="cpu")
        self.epoch_targets = torch.empty(epoch_size, dtype=torch.int64, device="cpu")
        self.average_method = average_method
        self.multi_class_method = multi_class_method
        self.offset = 0
        self.last_batch_auroc = None

    def auroc(self, preds, targets):
        
        auroc_fn = torch_metrics_auroc(num_classes=self.num_classes, average="macro") 
        auroc_fn.update(preds, targets)
        return auroc_fn.compute()

    def update(self, preds, targets) -> None:
        # 1) Select task views
        preds   = self.get_preds(self.task, preds)
        targets = self.get_targets(self.task, targets)
    
        # 2) Validate preds: (B, C)
        num_dims_is(preds, 2, self.who())
        last_dim_has_size(preds, self.num_classes, self.who())
    
        # 3) Normalize targets shape first
        if targets.dim() == 2:
            if targets.size(-1) == 1:
                targets = targets.view(-1)                  # (B,)
            elif targets.size(-1) == self.num_classes:
                targets = targets.argmax(dim=-1)            # one-hot -> indices (B,)
            else:
                raise ValueError(
                    f"{self.who()}.update got targets with shape {tuple(targets.shape)}; "
                    f"expected (B,), (B,1), or (B,{self.num_classes}) one-hot."
                )
        elif targets.dim() == 1:
            pass  # OK
        else:
            raise ValueError(f"{self.who()}.update expected 1D or 2D targets, got {tuple(targets.shape)}.")
    
        # 4) Move/cast AFTER shape normalization
        preds   = preds.detach().float().softmax(dim=-1).to("cpu", non_blocking=True).to(self.dtype)
        targets = targets.detach().to("cpu", non_blocking=True).to(torch.long)
    
        # 5) Write into epoch buffers
        n = targets.numel()
        self.epoch_scores[self.offset : self.offset + n]  = preds
        self.epoch_targets[self.offset : self.offset + n] = targets
        self.offset += n
    
        # 6) Per-batch AUROC
        self.last_batch_auroc = self.auroc(preds, targets)
        
    def metric_on_batch(self) -> float:
        return self.last_batch_auroc

    def metric_on_epoch(self) -> float:
        preds = self.epoch_scores[:self.offset]
        targets = self.epoch_targets[:self.offset]
        return self.auroc(preds, targets)

    def reset(self) -> None:
        self.offset = 0
        self.last_batch_auroc = None


# 5) Manager for module-based metrics
class MetricStack:
    """
    Holds MetricModule objects. Call update(...) per batch, compute() at epoch end.
    """
    def __init__(self, modules: list[MetricModule]):
        self.modules = modules
    
    def get_modules(self):
        for module in self.modules:
            yield module
    
    def update(self, preds, targets):
        for module in self.get_modules():
            module.update(preds, targets)

    def compute(self, for_samples_in: str = "batch") -> Dict[str, float]:
        out = {}
        for module in self.get_modules():
            base = module.__class__.__name__.replace("Metric", "").lower()
            key  = f"{base}_{module.task}" if getattr(module, "task", None) else base  # <-- underscore
            if for_samples_in == "batch": 
                out[key] = module.metric_on_batch()
            elif for_samples_in == "epoch":
                out[key] = module.metric_on_epoch()
        return out

    def reset(self):
        for module in self.get_modules():
            module.reset()

def _normalize_dtype_in_kwargs(kw: Dict[str, Any]) -> Dict[str, Any]:
    # Convenience: allow dtype passed as string, e.g. "float32"
    if "dtype" in kw and isinstance(kw["dtype"], str):
        if not hasattr(torch, kw["dtype"]):
            raise ValueError(f"Unrecognized torch dtype string: {kw['dtype']}")
        kw = dict(kw)
        kw["dtype"] = getattr(torch, kw["dtype"])
    return kw

def process_metrics_config(cfg_list: List[Dict[str, Any]]) -> MetricStack:
    """
    cfg example:

        metrics:
          - { name: accuracy, task: cls }
          - { name: rmse, task: reg }
          - { name: auroc, task: cls, kwargs: { epoch_size: 5000, num_classes: 3, dtype: "float32" } }

    Notes
    -----
    • 'task' may be a string or a list of strings (one instance per task).
    • Extra 'kwargs' are passed to the metric's __init__, with signature filtering.
    """
    modules: List[MetricModule] = []

    for item in cfg_list:
        if isinstance(item, str):
            item = {"name": item}

        name: str = item["name"]
        metric_cls = get_from_registry("metric", name)

        # Support task as str | list[str] | None
        tasks = item.get("task", None)
        if tasks is None or isinstance(tasks, str):
            tasks = [tasks]

        base_kwargs: Dict[str, Any] = _normalize_dtype_in_kwargs(item.get("kwargs", {}))
        sig = inspect.signature(metric_cls.__init__)

        for task in tasks:
            call_kwargs = dict(base_kwargs)

            # If the class accepts 'task', pass it when provided
            if "task" in sig.parameters and task is not None:
                call_kwargs["task"] = task

            # Check required args (positional or keyword-only) are present
            required_params = [
                p.name for p in sig.parameters.values()
                if p.name != "self"
                and p.default is inspect._empty
                and p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                               inspect.Parameter.POSITIONAL_OR_KEYWORD,
                               inspect.Parameter.KEYWORD_ONLY)
            ]
            missing = [r for r in required_params if r not in call_kwargs]
            if missing:
                raise TypeError(
                    f"Metric '{name}' missing required init args {missing}. "
                    f"Provide them under 'kwargs' in the metrics config."
                )

            # Filter to only accepted kwargs
            accepted = set(sig.parameters.keys()) - {"self"}
            call_kwargs = {k: v for k, v in call_kwargs.items() if k in accepted}

            modules.append(metric_cls(**call_kwargs))

    return MetricStack(modules)

# ########################################################################### #
# 8. PyTorch Lightning:
# ########################################################################### #

# =========================================================================== #
# Helpers Not Instantiated Yet:
# =========================================================================== #

def extract_task_names(cfg: Dict):
    return cfg["mixer"]["kwargs"]["task_names"]

def default_task_name(cfg: Dict):
    return extract_task_names(cfg)[0]

# =========================================================================== #
# PyTorch Lightning MultiTask Learning Architecture:
# =========================================================================== #

def _infer_epoch_items_from_loader(dl) -> int | None:
    # Prefer sampler length (rank-aware under DDP)
    if getattr(dl, "sampler", None) is not None:
        try:
            return len(dl.sampler)
        except TypeError:
            pass
    # Fallback to dataset length
    if getattr(dl, "dataset", None) is not None:
        try:
            return len(dl.dataset)
        except TypeError:
            pass
    # Last resort: batches * batch_size (may be slightly off with drop_last or custom collate)
    try:
        bs = getattr(dl, "batch_size", None)
        if bs is None and getattr(dl, "batch_sampler", None) is not None:
            bs = getattr(dl.batch_sampler, "batch_size", None)
        if bs is not None:
            return len(dl) * int(bs)
    except Exception:
        pass
    return None

@register("ptl", "multitask_default")
class MultiTaskLightningModule(pl.LightningModule):
    """
    End‑to‑end LightningModule that glues together:
        • model (built by your builder)
        • loss stack (Section 5)
        • task weighting (Section 4) – Identity or PCGrad, etc.
        • optimizer / LR‑scheduler (Section 6)
        • metrics (Section 7: MetricStack)

    Everything is driven by a single config dict passed to __init__.
    """

    # ------------- Public API ------------------------------------------------
    def __init__(self, cfg: Dict):
        # Save only pickleable parts. Avoid storing heavy builder objects.
        cfg_for_save = {k: v for k, v in cfg.items() if k not in ("architecture",)}
        super().__init__()
        self.save_hyperparameters(cfg_for_save)

        # ---- Slice the config ------------------------------------------------
        self.model_cfg     = cfg["architecture"]
        self.optim_cfg     = cfg["optim"]
        self.loss_cfg      = cfg["loss"]
        self.weighting_cfg = cfg.get("weighting", DEFAULT_LOSS_WEIGHTING_STRATEGY)
        self.metrics_cfg   = cfg.get("metrics", [])
        self._early_cfg    = cfg.get("early_stopping")

        # ---- Build the model via your builder -------------------------------
        self.builder       = process_model_config(self.model_cfg)
        self.model: nn.Module = self.builder.build()

        # ---- Tasks -----------------------------------------------------------
        self.task_names: List[str] = list(self.builder.get_tasks())
        self.primary_task: str     = self.builder.get_default_task()

        # ---- Losses & metrics -----------------------------------------------
        self.loss_stack  = process_loss_config(self.loss_cfg)          # LossStack
        #self.metric_mgr  = process_metrics_config(self.metrics_cfg)    # MetricStack
        self.metrics_cfg = cfg.get("metrics", [])
        self.metric_mgr = None  # defer construction

        # ---- Weighting (Identity → automatic opt; otherwise manual) ---------
        shared_params = [p for p in self.model.parameters() if p.requires_grad]
        self.weighting = process_weighting_config(self.weighting_cfg, shared_params)
        self.automatic_optimization = isinstance(self.weighting, IdentityWeighting)

        # Nice to have for consumers
        self._built = True
        
    # -------------------------------------------------------------------------
    
    def _infer_stage_epoch_size(self, stage: str) -> int | None:
        """stage in {'val','test'}; returns number of examples *on this rank*."""
        dls = getattr(self.trainer, f"{stage}_dataloaders", None)
        if dls is None:
            return None
        if isinstance(dls, (list, tuple)):
            # If you use multiple val loaders, pick the first by default; customize if needed.
            dls = dls[0]
        return _infer_epoch_items_from_loader(dls)

    def _patched_metrics_cfg_with_epoch_size(self, stage: str) -> list[dict]:
        cfg = copy.deepcopy(self.metrics_cfg)
        n = self._infer_stage_epoch_size(stage)
        for item in cfg:
            if isinstance(item, dict) and item.get("name") == "auroc":
                item.setdefault("kwargs", {})
                if n is None:
                    raise RuntimeError(f"Could not infer {stage} epoch size for AUROC. "
                                       "Provide metrics[].kwargs.epoch_size explicitly or use a dataset with __len__.")
                item["kwargs"]["epoch_size"] = int(n)
        return cfg

    # ------------- Lightning hooks ------------------------------------------

    def forward(self, inputs):
        """
        Contract: return a dict[str, Tensor], mapping each task name to its prediction tensor.
        Your builder/model must uphold this for MTL.
        """
        return self.model(inputs)

    def configure_optimizers(self):
        """
        Build (optimizer, scheduler?) from config over *model* params.
        If you later add param_groups in the builder, swap to those here.
        """
        params = self.model.parameters()
        opt, sched = process_optimizer_config(self.optim_cfg, params)
        return ({"optimizer": opt, "lr_scheduler": sched} if sched else opt)

    # Unified shared step (train/val/test) to prevent code drift.
    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        # ----------------- Validate & unpack batch ----------------------------
        if not isinstance(batch, dict) or "inputs" not in batch or "targets" not in batch:
            raise TypeError(
                "Expected batch to be a dict with keys {'inputs', 'targets'}; "
                f"got {type(batch)} with keys {getattr(batch, 'keys', lambda: [])()}."
            )
        x, y = batch["inputs"], batch["targets"]

        # Accept single‑task tensor targets for convenience → wrap
        if not isinstance(y, dict):
            y = {self.primary_task: y}

        # ----------------- Forward -------------------------------------------
        preds = self(x)  # dict[str, Tensor]

        # ----------------- Loss computation ----------------------------------
        loss_dict = self.loss_stack(
            model   = self.model,
            preds   = preds,
            targets = y,
            batch   = batch,
            epoch   = self.current_epoch,
        )
        total_loss = loss_dict["total"]

        # ----------------- Weighting & optimisation ---------------------------
        if stage == "train":
            if self.automatic_optimization:
                # Lightning will handle backward/step; just return total_loss
                self.log(f"{stage}_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
            else:
                # Manual optimisation + gradient surgery (e.g., PCGrad)
                opt = self.optimizers()
                if isinstance(opt, (list, tuple)):
                    opt = opt[0]
                opt.zero_grad()

                # Gather per-task losses = all non-total entries
                task_losses = [v for k, v in loss_dict.items() if k != "total"]
                total_loss = self.weighting.backward(task_losses)  # does backward internally
                opt.step()

                # log detached value (we did the backward already)
                self.log(f"{stage}_loss", total_loss.detach(), prog_bar=True, on_step=False, on_epoch=True)
        else:
            # eval path: log the total (no backward)
            self.log(f"{stage}_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)

        # ----------------- Metrics -------------------------------------------
        # Track metrics on val/test; skip train unless requested.
        if stage != "train" and self.metric_mgr is not None:
            self.metric_mgr.update(preds, y)

        # ----------------- Individual loss terms (optional) -------------------
        for k, v in loss_dict.items():
            self.log(
                f"{stage}_{k}",
                v.detach() if isinstance(v, torch.Tensor) and v.requires_grad else v,
                prog_bar=False, on_step=False, on_epoch=True
            )

        # For Lightning's optimisation loop, return a tensor
        return total_loss if stage == "train" else (total_loss.detach() if isinstance(total_loss, torch.Tensor) else total_loss)

    # ------------- step entry points ----------------------------------------

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    # ------------- epoch-start/end metric logging ---------------------------

    def on_validation_epoch_start(self):
        # (Re)build metrics with the correct epoch_size for this stage
        patched = self._patched_metrics_cfg_with_epoch_size("val")
        self.metric_mgr = process_metrics_config(patched)
        self.metric_mgr.reset()

    def on_test_epoch_start(self):
        patched = self._patched_metrics_cfg_with_epoch_size("test")
        self.metric_mgr = process_metrics_config(patched)
        self.metric_mgr.reset()

    def on_validation_epoch_end(self):
        if self.metric_mgr:
            logs = self.metric_mgr.compute(for_samples_in="epoch")
            for k, v in logs.items():
                # AUROC may return a Tensor; convert safely
                if isinstance(v, torch.Tensor):
                    v = v.item() if v.numel() == 1 else float(v.mean().item())
                self.log(f"val_{k}", v, prog_bar=True, on_step=False, on_epoch=True)
            # ready for next epoch
            self.metric_mgr.reset()

    def on_test_epoch_end(self):
        if self.metric_mgr:
            logs = self.metric_mgr.compute(for_samples_in="epoch")
            for k, v in logs.items():
                if isinstance(v, torch.Tensor):
                    v = v.item() if v.numel() == 1 else float(v.mean().item())
                self.log(f"test_{k}", v, prog_bar=True, on_step=False, on_epoch=True)
            self.metric_mgr.reset()

    # ------------- device transfer ------------------------------------------

    @staticmethod
    def _move_to_device(obj, device):
        # 1) Tensors
        if isinstance(obj, torch.Tensor):
            # non_blocking is ignored unless pinned; safe to pass
            return obj.to(device, non_blocking=True)
    
        # 2) PyG Data/Batch or any custom object with a .to(device)
        #    (exclude nn.Module just in case)
        if hasattr(obj, "to") and not isinstance(obj, nn.Module):
            try:
                return obj.to(device, non_blocking=True)
            except TypeError:
                return obj.to(device)  # some .to() don't accept non_blocking
    
        # 3) Containers
        if isinstance(obj, dict):
            return {k: MultiTaskLightningModule._move_to_device(v, device) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            typ = type(obj)
            return typ(MultiTaskLightningModule._move_to_device(v, device) for v in obj)
    
        # 4) Fallback: leave as-is
        return obj

    def transfer_batch_to_device(self, batch, device, dataloader_idx: int):
        x, y = batch["inputs"], batch["targets"]
        x = self._move_to_device(x, device)
        y = self._move_to_device(y, device)
        return {"inputs": x, "targets": y}
    
    def _tune_metrics_map(self) -> dict[str, str]:
        """
        Keys are the names Ray Tune will see; values are the Lightning metric
        keys as logged via self.log(...). We always include 'val_loss' because
        TuneConfig(metric="val_loss") expects it. For convenience we also add
        'val_{name}' and 'val_{name}_{task}' for each metric in self.metrics_cfg.
        Only keys that your module actually logs will appear at runtime.
        """
        m: dict[str, str] = {"val_loss": "val_loss"}
        for item in self.metrics_cfg:
            name = item.get("name")
            if not name:
                continue
            # Plain metric (single-task or your MetricStack returns 'accuracy', 'auroc', ...)
            key_plain = f"val_{name}"
            m[key_plain] = key_plain
            # Task-qualified variant (if your MetricStack emits task-suffixed keys)
            task = item.get("task")
            if task:
                key_task = f"val_{name}_{task}"
                m[key_task] = key_task
        return m

    # ---- NEW: Provide Ray Tune + (optional) EarlyStopping callbacks -------
    def configure_callbacks(self) -> list[Callback]:
        """
        Register Ray Tune's reporting/checkpointing callback. Also honors your
        optional early_stopping config by attaching Lightning's EarlyStopping.
        """
        callbacks: list[Callback] = []

        # Report metrics after each validation epoch and checkpoint the model.
        # (If you log at other times, adjust 'on="validation_end"'.)
        tune_cb = TuneReportCheckpointCallback(
            metrics=self._tune_metrics_map(),
            on="validation_end",
        )
        callbacks.append(tune_cb)

        # Optional: early stopping from your config block
        if self._early_cfg:
            callbacks.append(
                EarlyStopping(
                    monitor=self._early_cfg.get("monitor", "val_loss"),
                    mode=self._early_cfg.get("mode", "min"),
                    patience=int(self._early_cfg.get("patience", 2)),
                    min_delta=float(self._early_cfg.get("min_delta", 0.0)),
                )
            )

        return callbacks

# ########################################################################### #
# 9. Dataset Related:
# ########################################################################### #

# =============================================================================
# Train-Test Split:
# =============================================================================

def TrainTestSplit(
    tbl: IterableTable,
    *,
    train_frac: float = 0.8,
    val_frac:   float = 0.1,
    test_frac:  float = 0.1,
    seed: int | None = None,
) -> Tuple[IterableTable, IterableTable, IterableTable]:
    """
    Randomly split *tbl* into train / val / test **without materialising**
    the whole data set on a single worker.

    Parameters
    ----------
    tbl        : IterableTable
        Source data set.
    train_frac : float, default 0.8
    val_frac   : float, default 0.1
    test_frac  : float, default 0.1
        Fractions must be non‑negative and sum to **1.0** (within 1e‑6).
    seed       : int | None
        Deterministic RNG seed.

    Returns
    -------
    (train_tbl, val_tbl, test_tbl)  – each a **new** `IterableTable` instance.

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
            IterableTable(train_ds, smiles_column_name=tbl.smiles_column_name),
            IterableTable(val_ds,   smiles_column_name=tbl.smiles_column_name),
            IterableTable(test_ds,  smiles_column_name=tbl.smiles_column_name),
        )

    # -------------------------- Dask branch -----------------------------
    if tbl.is_dask():
        ddf_train, ddf_val, ddf_test = tbl.table.random_split(
            [tr_f, va_f, te_f], random_state=seed
        )
        return (
            IterableTable(ddf_train, smiles_column_name=tbl.smiles_column_name),
            IterableTable(ddf_val,   smiles_column_name=tbl.smiles_column_name),
            IterableTable(ddf_test,  smiles_column_name=tbl.smiles_column_name),
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
            IterableTable(train_df, smiles_column_name=tbl.smiles_column_name),
            IterableTable(val_df,   smiles_column_name=tbl.smiles_column_name),
            IterableTable(test_df,  smiles_column_name=tbl.smiles_column_name),
        )

# ########################################################################### # 
# 10. Hyperparameter Tuning: 
# ########################################################################### #

# =========================================================================== #
# Hyperparameter-Related Imports:
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Search Imports:
# --------------------------------------------------------------------------- #

from ray.tune.search.basic_variant import BasicVariantGenerator as _BasicVariantGenerator
from ray.tune.search.optuna import OptunaSearch as _OptunaSearch
from ray.tune.search.hyperopt import HyperOptSearch as _HyperOptSearch
from ray.tune.search.bayesopt import BayesOptSearch as _BayesOptSearch
from ray.tune.search.nevergrad import NevergradSearch as _NevergradSearch
from ray.tune.search.bohb import TuneBOHB as _TuneBOHB
#from ray.tune.search.skopt import SkOptSearch as _SkOptSearch
_SkOptSearch = None # No longer available ) :
from ray.tune.search.ax import AxSearch as _AxSearch
from ray.tune.search.hebo import HEBOSearch as _HEBOSearch

# --------------------------------------------------------------------------- #
# Scheduling Imports:
# --------------------------------------------------------------------------- #

from ray.tune.schedulers import FIFOScheduler as _FIFOScheduler
from ray.tune.schedulers import ASHAScheduler as _ASHAScheduler
from ray.tune.schedulers import HyperBandScheduler as _HyperBandScheduler
from ray.tune.schedulers import MedianStoppingRule as _MedianStoppingRule
from ray.tune.schedulers import PopulationBasedTraining as _PopulationBasedTraining
from ray.tune.schedulers.pb2 import PB2 as _PB2
from ray.tune.schedulers import HyperBandForBOHB as _HyperBandForBOHB

# =========================================================================== #
# Sampling Options:
# =========================================================================== #

def log_uniform(lower_value: Number, upper_value: Number) -> Number:
    """
    Log-uniform float sampler on (lower_value, upper_value).

    Parameters
    ----------
    lower_value : float
        Lower bound (> 0).
    upper_value : float
        Upper bound (> lower_value).

    Returns
    -------
    object
        A Ray Tune sampling domain representing log-uniform sampling.
    """
    return tune.loguniform(lower_value, upper_value)


def uniform(lower_value: Number, upper_value: Number) -> Number:
    """
    Uniform float sampler on [lower_value, upper_value).

    Returns
    -------
    object
        A Ray Tune sampling domain representing uniform sampling.
    """
    return tune.uniform(lower_value, upper_value)


def random_int(lower_value: Number, upper_value: Number) -> Number:
    """
    Uniform integer sampler on {lower_value, ..., upper_value-1}.

    Returns
    -------
    object
        A Ray Tune sampling domain representing integer uniform sampling.
    """
    return tune.randint(lower_value, upper_value)


def log_random_int(lower_value: Number, upper_value: Number) -> Number:
    """
    Log-uniform integer sampler on {lower_value, ..., upper_value-1}.

    Notes
    -----
    Useful for parameters that scale exponentially (e.g., hidden sizes).

    Returns
    -------
    object
        A Ray Tune sampling domain representing log-uniform integer sampling.
    """
    return tune.lograndint(lower_value, upper_value)


def choice(options: Sequence[object]) -> Number:
    """
    Categorical sampler that selects a single value from a finite set.

    Parameters
    ----------
    options : Sequence[T]
        Non-empty sequence of candidate values.

    Returns
    -------
    object
        A Ray Tune sampling domain representing categorical sampling.
    """
    # Convert to list to avoid surprises with generators/iterators.
    return tune.choice(list(options))

# =========================================================================== #
# Search Space Methods:
# =========================================================================== #

@register("search", "basic")
def basic_variant_generator(*args, **kwargs):
    return _BasicVariantGenerator(*args, **kwargs)

@register("search", "optuna")
def optuna_search(*args, **kwargs):
    return _OptunaSearch(*args, **kwargs)

@register("search", "hyper_opt")
def hyperopt_search(*args, **kwargs):
    return _HyperOptSearch(*args, **kwargs)

@register("search", "bayes_opt")
def bayesopt_search(*args, **kwargs):
    return _BayesOptSearch(*args, **kwargs)

@register("search", "nevergrad")
def nevergrad_search(*args, **kwargs):
    return _NevergradSearch(*args, **kwargs)

@register("search", "bohb")
def tune_bohb(*args, **kwargs):
    return _TuneBOHB(*args, **kwargs)

@register("search", "skopt")
def skopt_search(*args, **kwargs):
    return _SkOptSearch(*args, **kwargs)

@register("search", "ax")
def ax_search(*args, **kwargs):
    return _AxSearch(*args, **kwargs)

@register("search", "hebo")
def hebo_search(*args, **kwargs):
    return _HEBOSearch(*args, **kwargs)

# =========================================================================== #
# Scheduler Methods:
# =========================================================================== #

@register("scheduler", "fifo")
def fifo_scheduler(*args, **kwargs):
    return _FIFOScheduler(*args, **kwargs)

@register("scheduler", "asha")
def asha_scheduler(*args, **kwargs):
    return _ASHAScheduler(*args, **kwargs)

@register("scheduler", "hyperband")
def hyperband_scheduler(*args, **kwargs):
    return _HyperBandScheduler(*args, **kwargs)

@register("scheduler", "median_stopping")
def median_stopping_rule(*args, **kwargs):
    return _MedianStoppingRule(*args, **kwargs)

@register("scheduler", "population_based")
def population_based_training(*args, **kwargs):
    return _PopulationBasedTraining(*args, **kwargs)

@register("scheduler", "pb2")
def pb2(*args, **kwargs):
    return _PB2(*args, **kwargs)

@register("scheduler", "hyperband_for_bohb")
def hyperband_for_bohb(*args, **kwargs):
    return _HyperBandForBOHB(*args, **kwargs)

# =========================================================================== #
# Example Configs:
# =========================================================================== #

dataset_config = {
    "col_to_task": {"Graph": "graph", "Hit": "hit"},
    "collate_tasks": {"graph": "to_graph", "hit": "to_numerical"}
    }

search_space_config = {
    "architecture": {
        "name": "single_task",
        "kwargs": {
            "task_name": "hit",
            "encoder_name": "dmpnn",  # uses your @register("core", "gtr") encoder
            "encoder_kwargs": {
                "num_node_feats": NODE_FEAT_DIM,     # 118
                "num_edge_feats": EDGE_FEAT_DIM,     # 4
                "hidden": 128,
                #"heads": 4,
                #"dropout": 0.1,
            },
            "head_name": "classification",
            "head_kwargs": {
                "in_dim": 128,   # must match encoder/GraphVectorizer output
                "num_classes": 2,       # binary classification (Hit vs not)
                # "activation_function": None,  # logits for CE loss (default)
            },
            "device": "cpu",  # model builds on CPU; Lightning will move it
        },
    },

    "optim": {
        "name": "adamw",
        "kwargs": {"lr": 1e-3, "weight_decay": 1e-4}
    },

    "loss": [
        {"name": "cross_entropy", "task": "hit", "weight": 1.0},
    ],

    "metrics": [
        {"name": "accuracy", "task": "hit"},
        {"name": "auroc",    "task": "hit"},  # works for binary logits
    ],

    "weighting": {"name": "identity"},

    # Optional: used below to build the callback for Trainer
    "early_stopping": {"monitor": "val_loss", "mode": "min", "patience": 2, "min_delta": 0.0},
}

# =========================================================================== #
# Collate Methods:
# =========================================================================== #

@register("collate", "to_graph")
def collate_to_graph(l):
    return GeoBatch.from_data_list(l)


@register("collate", "to_numerical")
def collate_to_numerical(l):
    if all(isinstance(v, Number) for v in l):
        return torch.tensor(l, dtype=torch.long)                 # shape (B,)
    if all(isinstance(v, torch.Tensor) for v in l):
        t = torch.stack([v.view(-1) for v in l], dim=0).squeeze(-1)
        return t.to(torch.long)
    raise NotImplementedError("to_numerical expects Numbers or Tensors.")

# =========================================================================== #
# Hyperparameter Optimization Methods:
# =========================================================================== #

dataset_config = {
    "col_to_task": {"Graph": "graph", "Hit": "hit"},
    "collate_tasks": {"graph": "to_graph", "hit": "to_numerical"}
    }

def collate_from_config(batch: List[Dict], dataset_config):
    
    col_to_task = dataset_config["col_to_task"]
    collate_tasks = dataset_config["collate_tasks"]
    
    # 1. Standardize all columns names -> task names.
    mapped_batch = []
    for row in batch:
        mapped_batch.append({col_to_task[key]: val for key, val in row.items() if key in col_to_task})
    
    # 2. Lookup all the collate tasks methods.
    collate_functions = {key: get_collate(val) for key, val in collate_tasks.items()}
    
    # 3. Convert into a standard format.
    tasks = col_to_task.values()
    combined_mapped_batch = {task: collate_functions[task]([row[task] for row in mapped_batch]) for task in tasks}
    
    return combined_mapped_batch

def training_function(config: Dict, train_dataloader: DataLoader, val_dataloader: DataLoader, test_dataloader: DataLoader):
    
    # TODO: change ptl --> "learner"
    model_cls = get_ptl(config["name"])
    model = model_cls(config)
    
    callbacks = []
    # if config.get("early_stopping"):
    #     callbacks.append(EarlyStopping(**param_space["early_stopping"]))

    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="cpu",   # set "cpu" if you want to force CPU
        devices="auto",
        log_every_n_steps=10,
        enable_checkpointing=True, # Changed to ensure callbacks record metrics.
        callbacks=callbacks,
    )
    
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model, dataloaders=test_dataloader)

def construct_tuner(
                training_function: Callable,
                resource_config: Dict,
                search_space_config: Dict,
                optimisation_config: Dict,
                run_config: Dict,
                train_dataloader: DataLoader,
                val_dataloader: DataLoader,
                test_dataloader: DataLoader
                   ):
    
        
    training_function = tune.with_parameters(
                                        training_function, 
                                        train_dataloader=train_dataloader, 
                                        val_dataloader=val_dataloader, 
                                        test_dataloader=test_dataloader
                                        )
    tuner = tune.Tuner(
        tune.with_resources(training_function, resource_config),
        param_space=search_space_config,
        tune_config=optimisation_config,
        run_config=run_config
    )
    return tuner

def hyperparameter_optimisation(
        resource_config: Dict,
        search_space_config: Dict,
        optimisation_config: TuneConfig, # TODO: Change this so that optimisation configs that are dicts are converted correctly.
        run_config: Dict,
        dataset_config: Dict,
        training_dataset: IterableTable,
        val_dataset: IterableTable,
        test_dataset: IterableTable,
        training_function: Callable = training_function,
        collate_function: Callable = collate_from_config
        ):
    
    # 1. Give the collate function config needed to parse the dataset.
    collate = partial(collate_function, dataset_config=dataset_config)
    
    BATCH_SIZE    = 64
    NUM_WORKERS   = 0   # keep 0 on macOS; bump on Linux if you like
    PERSISTENT    = False
    #SHUFFLE = True
    
    train_dataloader = DataLoader(
        training_dataset,
        batch_size=BATCH_SIZE,
        #shuffle=SHUFFLE, # Is IterableTable
        num_workers=NUM_WORKERS,
        persistent_workers=PERSISTENT,
        collate_fn=collate,
        pin_memory=False,
        )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        #shuffle=SHUFFLE, # Is IterableTable
        num_workers=NUM_WORKERS,
        persistent_workers=PERSISTENT,
        collate_fn=collate,
        pin_memory=False,
        )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        #shuffle=SHUFFLE, # Is IterableTable
        num_workers=NUM_WORKERS,
        persistent_workers=PERSISTENT,
        collate_fn=collate,
        pin_memory=False,
        )

    tuner = construct_tuner(
                    training_function,
                    resource_config,
                    search_space_config,
                    optimisation_config,
                    run_config,
                    train_dataloader,
                    val_dataloader,
                    test_dataloader
                       )
    
    tuner.fit()
    
    
    

# ########################################################################### #
# 11.  Model Config Parser:
# ########################################################################### #

# TODO: Implement config parser.

###############################################################################
# Other Code:
###############################################################################    

if __name__ == "__main__" and os.getenv("RUN_TUNE_DEMO", "1") == "1":
    
    DATA_CSV = "/Users/asselism/Desktop/Collins_Lab/Datasets/Organized_by_Data_Type/Bioactivity/Organized_by_Data_Type/Antimicrobial_Activity/Organized_by_Source/Internal_Collins_Lab_Screens/39K_Screen/Organized_by_Species/Modified_Datasets/Escherichia_coli_GI_with_Minimol_Representations.csv"
    new_data_csv = "/Users/asselism/Desktop/Collins_Lab/Datasets/Organized_by_Data_Type/Bioactivity/Organized_by_Data_Type/Antimicrobial_Activity/Organized_by_Source/Internal_Collins_Lab_Screens/39K_Screen/Organized_by_Species/Modified_Datasets/Escherichia_coli_GI_with_Minimol_Representations_and_Graphs.csv"
    
    tbl = to_table(DATA_CSV, to_class=IterableTable)          # factory you already use
    
    #tbl = tbl.head(n=1000)

    if "Graph" not in tbl.to_pandas().columns:
        print("🛠️  Generating graph objects …")
        tbl = tbl.assign_rowwise_using("Graph", lambda row: smiles_to_data(row["SMILES"]))
    
    # After you create `tbl` and (optionally) assign Graph via SMILES:
    pdf = tbl.to_pandas()
    
    # If the CSV already had a 'Graph' column with serialized / nullable entries,
    # this still works (pandas treats None as NA in object columns).
    bad = pdf["Graph"].isna()
    print(f"Filtering out {bad.sum()} rows with invalid or missing graphs "
          f"out of {len(pdf)} rows.")
    
    pdf = pdf.loc[~bad].reset_index(drop=True)
    
    # Re-wrap into your IterableTable (keeps the same smiles_column_name if used)
    tbl = IterableTable(pdf, smiles_column_name=getattr(tbl, "smiles_column_name", None))
    
    # ─────────────────────────────────────────────────────────────────────────────
    # 2.  Train / val / test split
    # ─────────────────────────────────────────────────────────────────────────────
    training_dataset, val_dataset, test_dataset = TrainTestSplit(tbl, seed=42)
    
    #HIDDEN_DIM = 128
    
    # ---- 1) Per‑trial resource request passed to tune.with_resources(...)
    # CPU‑only example:
    resource_config = {"cpu": 4, "gpu": 0}

    search_space_config = {
        "name": "multitask_default",
        "architecture": {
            "name": "single_task",
            "kwargs": {
                "task_name": "hit",
                "encoder_name": "dmpnn",  # uses your @register("core", "gtr") encoder
                "encoder_kwargs": {
                    "num_node_feats": NODE_FEAT_DIM,     # 118
                    "num_edge_feats": EDGE_FEAT_DIM,     # 4
                    "hidden": log_random_int(32, 1024), # HIDDEN_DIM
                    #"heads": 4,
                    #"dropout": 0.1,
                },
                "head_name": "classification",
                "head_kwargs": {
                    "in_dim": tune.sample_from(
                    lambda spec: spec.config["architecture"]["kwargs"]["encoder_kwargs"]["hidden"] # Same as HIDDEN_DIM
                ),
                    "num_classes": 2,       # binary classification (Hit vs not)
                    # "activation_function": None,  # logits for CE loss (default)
                },
                "device": "cpu",  # model builds on CPU; Lightning will move it
            },
        },

        "optim": {
            "name": "adamw",
            "kwargs": {"lr": log_uniform(1e-6, 1e-2), "weight_decay": log_uniform(1e-6, 1e-2)}
        },

        "loss": [
            {"name": "cross_entropy", "task": "hit", "weight": 1.0},
        ],

        "metrics": [
            {"name": "accuracy", "task": "hit"},
            {"name": "auroc",    "task": "hit"},  # works for binary logits
        ],

        "weighting": {"name": "identity"},

        # Optional: used below to build the callback for Trainer
        "early_stopping": {"monitor": "val_loss", "mode": "min", "patience": 2, "min_delta": 0.0},
    }
    
    optimisation_config = TuneConfig(
        metric="val_loss",         # must match what your LightningModule logs
        mode="min",
        num_samples=8,             # number of trials to run
        scheduler=ASHAScheduler(   # simple, strong default
            time_attr="training_iteration",  # advanced each validation epoch by the Tune callback
            grace_period=1,                  # don't stop until 1st epoch reported
            reduction_factor=2               # rung halving
        ),
        # search_alg=basic_variant_generator()  # optional; default is fine
    )
    
    run_config = RunConfig(
        name="dmpnn_hit_hpo_demo",
        # Stop after 5 validation epochs are reported (per trial) or after a fixed walltime, etc.
        stop={"training_iteration": 20},
        verbose=1,
        # local_dir can be omitted (Ray will choose ~/ray_results); set explicitly if you like:
        # local_dir="/tmp/ray_results",
        checkpoint_config=CheckpointConfig(
            num_to_keep=1,
            # checkpoint_at_end=True # removed to appease the ray tune gods.
            # checkpoint_frequency=0  # rely on TuneReportCheckpointCallback at val_end
        )
    )
    
    dataset_config = {
        "col_to_task": {"Graph": "inputs", "Hit": "targets"},
        "collate_tasks": {"inputs": "to_graph", "targets": "to_numerical"}
        }
    
    hyperparameter_optimisation(
            resource_config,
            search_space_config,
            optimisation_config,
            run_config,
            dataset_config,
            training_dataset,
            val_dataset,
            test_dataset,
            training_function = training_function,
            collate_function = collate_from_config
            )