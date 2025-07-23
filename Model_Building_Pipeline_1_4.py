"""
================================================================================
 SINGLE‑FILE, CLI‑FREE MODULAR ML LIBRARY  (PyTorch ≥ 2.2, Lightning ≥ 2.5)
================================================================================

This version is identical to the earlier Ray‑Tune‑enabled library **plus** a
_local, pipeline‑only patch_ that equips the upstream `Table` class with two
Torch‑helper methods:

    • Table.to_tensor(...)
    • Table.as_torch_dataset(...)

Because the patch lives **inside this file**, the original module that defines
`Table` never needs to import PyTorch.

Sections
--------
1.  Imports & helpers                      (+ Ray Tune utilities)
1A. Table‑class Torch patch                «‑‑ NEW
2.  Registry utils (decorator + lookup)
3.  Core “atom” modules (Projection, Normalisation, Activation, Dropout)
4.  DenseStage  (Projection → Norm → Act → Reg)
5.  Blocks & trunks
6.  Mixer units  (Cross‑Stitch, Sluice)
7.  Heads
8.  Model recipes  (STL_FNN, SharedMTL_FNN, CrossStitchNet, SluiceNet)
9.  Loss Components (+ registry) and LossAggregator
10. ContextCollector  (captures activations for loss terms)
11. Minimal demo usage (Cross‑Stitch + loss stack – **no CLI**)
12. Hyper‑parameter optimisation with Ray Tune
13. Minimal Ray Tune demo (guarded by env‑var)
--------------------------------------------------------------------------------
"""

# ============================================================================
# 1  Imports & helpers
# ============================================================================
from __future__ import annotations

import os
import re
from functools import wraps
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import math
import sys
import base64, io, joblib

import pandas as pd

from rdkit import Chem

import torch
from torch.utils.data import TensorDataset, IterableDataset
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from ray import tune
from ray.air import RunConfig, CheckpointConfig
from ray.tune import TuneConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback

from Data_Processing_Pipeline_v_1_9 import *  # TODO: Update with the real import path.

# ---------------------------------------------------------------------------
# 1A  Table‑class Torch patch
# ---------------------------------------------------------------------------
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

#==============================================================================
# TrainTestSplit
#==============================================================================

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

        n_total = tbl._rowcount            # cached property
        n_train = int(tr_f * n_total)
        n_val   = int(va_f * n_total)
        # test = remainder

        train_ds = ds.limit(n_train)
        val_ds   = ds.skip(n_train).limit(n_val)
        test_ds  = ds.skip(n_train + n_val)

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

# ============================================================================
# 2  Registry utilities
# ============================================================================
_REGISTRIES: Dict[str, Dict[str, Any]] = {
    "atom": {},
    "mixer": {},
    "head": {},
    "net": {},
    "loss": {},
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


# ============================================================================
# 3  Core “atom” modules
# ============================================================================

# --------------------------------------------------------------------------- #
# 3A  The universal mix‑in every module will inherit
# --------------------------------------------------------------------------- #
class QueryInterface:
    """
    Standard getters; leaf modules override selectively.
    """

    # ----- predictions / targets --------------------------------------------
    def get_preds(self) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

    # ----- hidden activations ------------------------------------------------
    def get_activations(self) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

    # ----- learnable parameters ---------------------------------------------
    def get_learnable_weights(self) -> Dict[str, torch.Tensor]:
        # Default: expose all trainable parameters with their *full* names.
        return {n: p for n, p in self.named_parameters() if p.requires_grad}


# --------------------------------------------------------------------------- #
# 3A  Low‑level linear atoms
# --------------------------------------------------------------------------- #
@register("atom", "linear_transformation")
class LinearTransformation(QueryInterface, nn.Module):
    """Weight‑only `y = Wx`."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight.t())


@register("atom", "bias")
class Bias(QueryInterface, nn.Module):
    """Learnable bias added to last dimension."""

    def __init__(self, out_dim: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bias


# --------------------------------------------------------------------------- #
# 3B  High‑level projection
# --------------------------------------------------------------------------- #
@register("atom", "projection")
class Projection(QueryInterface, nn.Module):
    """LinearTransformation ∘ Bias."""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.linear = LinearTransformation(in_dim, out_dim)
        self.bias = Bias(out_dim) if bias else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bias(self.linear(x))


# ---- Remaining atomic registry entries ------------------------------------ #
@register("atom", "layer_norm")
class LayerNorm(QueryInterface, nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)


@register("atom", "relu")
class ReLU(QueryInterface, nn.Module):
    def forward(self, x):
        return F.relu(x)


@register("atom", "gelu")
class GELU(QueryInterface, nn.Module):
    def forward(self, x):
        return F.gelu(x)


@register("atom", "dropout")
class Dropout(QueryInterface, nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.drop = nn.Dropout(p)

    def forward(self, x):
        return self.drop(x)

# =============================================================================
# 3C  Graph‑centric “atom” modules  (15‑slot menu from the schematic)
# =============================================================================
#
# All modules follow a *common forward contract* so they can be composed
# interchangeably:
#
#     forward(graph, **kwargs) → updated_graph
#
# where `graph` is a **dict** with at least
#     • "x"          : node features          – shape (N, F)
#     • "edge_index" : LongTensor [2, E]      – source & target indices
# and, if present / required by a module,
#     • "edge_attr"  : edge / bond features   – (E, G)
#     • "pos_enc"    : positional encodings   – (N, P)  *or* (E, P₂)
#     • "batch"      : graph‑ids per node     – (N,)    (for pooling)
#
# Nothing else in the existing codebase is modified – these are *new* atoms.
# The default Identity / None flavours are provided so every architecture
# column from the “menu” can be expressed by simply toggling module names
# in your YAML / config.
# =============================================================================
from typing import Literal, Dict, Any

class GraphDict:
    """
    Minimal “struct‑of‑arrays” graph container.

    The object is *just a dict underneath*, but exposes attributes
    (`.x`, `.edge_index`, …) instead of the more verbose
    `g["x"]`, `g["edge_index"]`, etc., for readability.
    All downstream modules can consume either a raw *dict* **or**
    an instance of `GraphDict`.
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def __init__(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        pos_enc: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> None:
        self.x: torch.Tensor = x
        self.edge_index: torch.Tensor = edge_index
        self.edge_attr: Optional[torch.Tensor] = edge_attr
        self.pos_enc: Optional[torch.Tensor] = pos_enc
        self.batch: Optional[torch.Tensor] = batch

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def as_dict(self) -> Dict[str, Optional[torch.Tensor]]:
        """Return a plain‐Python dict with the same field names."""
        return {
            "x": self.x,
            "edge_index": self.edge_index,
            "edge_attr": self.edge_attr,
            "pos_enc": self.pos_enc,
            "batch": self.batch,
        }

    @classmethod
    def from_any(cls, d: "GraphDict | Dict[str, Any]") -> "GraphDict":
        """
        Accept either an existing :class:`GraphDict` or a mapping with
        the same keys, and always return a :class:`GraphDict`.
        """
        return d if isinstance(d, GraphDict) else cls(**d)

    # ------------------------------------------------------------------
    # Optional: nicer debugging / logging output
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # completely optional—feel free to delete
        fields = ", ".join(f"{k}={v.shape if torch.is_tensor(v) else v}"
                           for k, v in self.as_dict().items())
        return f"{self.__class__.__name__}({fields})"

# -----------------------------------------------------------------------------#
# Utility – scatter with vanilla PyTorch (no external deps)
# -----------------------------------------------------------------------------#
def _scatter(src: torch.Tensor,
             index: torch.Tensor,
             dim_size: int,
             reduce: str) -> torch.Tensor:
    """
    Vanilla PyTorch re-implementation of the three common reductions.
    Works on CPU *and* GPU, requires no external deps.

        • src    : (E, F)  values to aggregate
        • index  : (E,)    destination row for each src row
        • dim_size          number of rows in the result tensor
    """
    out = torch.zeros(dim_size, src.size(-1),
                      dtype=src.dtype, device=src.device)

    if reduce == "sum":
        out.index_add_(0, index, src)
        return out

    if reduce == "mean":
        out.index_add_(0, index, src)
        counts = torch.bincount(index, minlength=dim_size).clamp(min=1)
        return out / counts.unsqueeze(1)

    if reduce == "max":
        # 1. start with -inf; 2. scatter-reduce via `scatter_reduce_`
        out.fill_(-torch.inf)
        out.scatter_reduce_(0, index.unsqueeze(-1).expand_as(src),
                            src, reduce="amax", include_self=True)
        return out

    raise ValueError(f"Unsupported reduce '{reduce}'")


# -----------------------------------------------------------------------------#
# 1 NodeFeatureEmbedder
# -----------------------------------------------------------------------------#
@register("atom", "node_embedder")
class NodeFeatureEmbedder(QueryInterface, nn.Module):
    """
    Maps raw node attributes (categorical IDs *or* continuous vectors) to
    a fixed‑size hidden space.
    """
    def __init__(self, in_dim: int, out_dim: int, kind: Literal["linear", "onehot", "identity"] = "linear"):
        super().__init__()
        if kind == "identity":
            assert in_dim == out_dim, "identity embedder requires in_dim == out_dim"
            self.embed = nn.Identity()
        elif kind == "onehot":
            self.embed = nn.Embedding(in_dim, out_dim)
        elif kind == "linear":
            self.embed = nn.Linear(in_dim, out_dim, bias=False)
        elif kind == "affine": #Manually added affine optionality.
            self.embed = nn.Linear(in_dim, out_dim, bias=True)
        else:
            raise NotImplementedError()

    def forward(self, graph: Dict[str, Any] | GraphDict):
        g = GraphDict.from_any(graph)
        g.x = self.embed(g.x)
        return g.as_dict()


# -----------------------------------------------------------------------------#
# 2 EdgeFeatureEmbedder
# -----------------------------------------------------------------------------#
@register("atom", "edge_embedder")
class EdgeFeatureEmbedder(QueryInterface, nn.Module):
    def __init__(self, in_dim: int, out_dim: int, kind: Literal["linear", "onehot", "identity"] = "linear"):
        super().__init__()
        if kind == "identity":
            assert in_dim == out_dim
            self.embed = nn.Identity()
        elif kind == "onehot":
            self.embed = nn.Embedding(in_dim, out_dim)
        elif kind == "linear":
            self.embed = nn.Linear(in_dim, out_dim, bias=False)
        elif kind == "affine":
            self.embed = nn.Linear(in_dim, out_dim, bias=True)
        else:
            raise NotImplementedError()

    def forward(self, graph):
        g = GraphDict.from_any(graph)
        if g.edge_attr is not None:
            g.edge_attr = self.embed(g.edge_attr)
        return g.as_dict()


# -----------------------------------------------------------------------------#
# 3 StructuralEncoder (Positional / distance encodings)
# -----------------------------------------------------------------------------#
@register("atom", "structural_encoder")
class StructuralEncoder(QueryInterface, nn.Module):
    """
    Adds / concatenates structural encodings to node (or edge) states.
    Currently supports:
      • 'none'            – no‑op
      • 'shortest_path'   – BFS distance to each node (just a scalar)
      • 'centrality'      – degree / eigen‑centrality
    """
    def __init__(self, method: str = "none"):
        super().__init__()
        self.method = method

    def forward(self, graph):
        if self.method == "none":
            return graph
        g = GraphDict.from_any(graph)
        N = g.x.size(0)
        # ---- very small / naïve CPU BFS – fine for < 5k nodes --------------
        if self.method == "shortest_path":
            src, dst = g.edge_index
            dist = torch.full((N, N), float("inf"))
            dist[torch.arange(N), torch.arange(N)] = 0.0
            dist[src, dst] = 1.0
            dist[dst, src] = 1.0
            for k in range(N):
                dist = torch.minimum(dist, dist[:, k : k + 1] + dist[k : k + 1, :])
            g.pos_enc = dist.min(dim=-1).values.unsqueeze(1)  # distance to *closest* node
        elif self.method == "centrality":
            deg = torch.bincount(g.edge_index.view(-1), minlength=N).float().unsqueeze(1)
            g.pos_enc = deg
        else:
            raise ValueError(self.method)
        return g.as_dict()


# -----------------------------------------------------------------------------#
# 4 MessageFunction
# -----------------------------------------------------------------------------#
@register("atom", "message_fn")
class MessageFunction(QueryInterface, nn.Module):
    """
    Produces per‑edge messages mᵤ→ᵥ from node & edge embeddings.
    m = φ([hᵤ || hᵥ || eᵤᵥ])  with a 2‑layer MLP by default.
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, act: str = "relu"):
        super().__init__()
        act_cls = get_from_registry("atom", act)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            act_cls(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, graph):
        g = GraphDict.from_any(graph)
        src, dst = g.edge_index
        feats = torch.cat([g.x[src], g.x[dst], g.edge_attr], dim=-1) if g.edge_attr is not None else torch.cat([g.x[src], g.x[dst]], dim=-1)
        msgs = self.net(feats)
        g.edge_attr = msgs
        return g.as_dict()


# -----------------------------------------------------------------------------#
# 5 EdgeAttention  (local, GAT‑style)
# -----------------------------------------------------------------------------#
@register("atom", "edge_attention")
class EdgeAttention(QueryInterface, nn.Module):
    """
    Computes scalar (or multi‑head) attention weights per edge and *modulates*
    the edge_attr tensor in‑place.
    """
    def __init__(self, in_dim: int, heads: int = 1, dropout_p: float = 0.0):
        super().__init__()
        self.heads = heads
        self.lin_src = nn.Linear(in_dim, heads, bias=False)
        self.lin_dst = nn.Linear(in_dim, heads, bias=False)
        self.dropout = nn.Dropout(dropout_p)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, graph):
        g = GraphDict.from_any(graph)
        h_src = self.lin_src(g.x)          # (N, H)
        h_dst = self.lin_dst(g.x)
        e_src, e_dst = g.edge_index
        e = self.leaky_relu(h_src[e_src] + h_dst[e_dst])  # (E, H)
        alpha = self.dropout(torch.softmax(e, dim=0))     # softmax over *all* edges (local later)
        # multiply existing messages (or node feats) by α
        base = g.edge_attr if g.edge_attr is not None else g.x[e_src]
        g.edge_attr = base * alpha.mean(dim=-1, keepdim=True)  # collapse heads
        return g.as_dict()


# -----------------------------------------------------------------------------#
# 6 LocalAggregator
# -----------------------------------------------------------------------------#
@register("atom", "local_aggregator")
class LocalAggregator(QueryInterface, nn.Module):
    """
    Aggregates incoming edge messages to each destination node.
    """
    def __init__(self, reduce: Literal["sum", "mean", "max"] = "mean"):
        super().__init__()
        self.reduce = reduce

    def forward(self, graph):
        g = GraphDict.from_any(graph)
        src, dst = g.edge_index
        out = _scatter(g.edge_attr, dst, g.x.size(0), self.reduce)
        g.x = out
        return g.as_dict()


# -----------------------------------------------------------------------------#
# 7 GlobalAttention (Transformer‑style)
# -----------------------------------------------------------------------------#
@register("atom", "global_attention")
class GlobalAttention(QueryInterface, nn.Module):
    """
    Full N×N self‑attention over node states.  Suitable for ≤ 10k nodes.
    """
    def __init__(self, dim: int, heads: int = 8, dropout_p: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout_p, batch_first=True)

    def forward(self, graph):
        g = GraphDict.from_any(graph)
        # unpack possible batching; we assume `batch` gives graph‑ids
        if g.batch is None:
            h, _ = self.attn(g.x.unsqueeze(0), g.x.unsqueeze(0), g.x.unsqueeze(0))
            g.x = h.squeeze(0)
            return g.as_dict()

        # mini‑batch of disjoint graphs – pad to max nodes per graph
        batch_ids = g.batch
        num_graphs = int(batch_ids.max().item()) + 1
        padded, lens = [], []
        for gid in range(num_graphs):
            idx = (batch_ids == gid).nonzero(as_tuple=False).squeeze(1)
            padded.append(g.x[idx])
            lens.append(idx.numel())
        max_len = max(lens)
        pad = lambda t: F.pad(t, (0, 0, 0, max_len - t.size(0)))
        x_pad = torch.stack([pad(p) for p in padded], dim=0)  # (G, L, D)
        h, _ = self.attn(x_pad, x_pad, x_pad)
        # remove padding
        g.x = torch.cat([h[i, :l] for i, l in enumerate(lens)], dim=0)
        return g.as_dict()


# -----------------------------------------------------------------------------#
# 8 HybridMixer / Router
# -----------------------------------------------------------------------------#
@register("atom", "hybrid_router")
class HybridMixer(QueryInterface, nn.Module):
    """
    Linearly blends local and global node updates (GraphGPS / CoAtGIN idea).
    """
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def forward(self, local_graph, global_graph):
        lg = GraphDict.from_any(local_graph)
        gg = GraphDict.from_any(global_graph)
        assert lg.x.shape == gg.x.shape
        out = lg.x * torch.sigmoid(self.alpha) + gg.x * (1.0 - torch.sigmoid(self.alpha))
        lg.x = out
        return lg.as_dict()


# -----------------------------------------------------------------------------#
# 9 NodeUpdate
# -----------------------------------------------------------------------------#
@register("atom", "node_update")
class NodeUpdate(QueryInterface, nn.Module):
    """
    Classic Residual + 2‑layer MLP (+ norm) update.
    """
    def __init__(self, dim: int, hidden_mul: int = 2, norm: str = "layer_norm", dropout_p: float = 0.0):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * hidden_mul),
            nn.GELU(),
            nn.Linear(dim * hidden_mul, dim),
            nn.Dropout(dropout_p),
        )
        self.norm = get_from_registry("atom", norm)(dim)

    def forward(self, graph):
        g = GraphDict.from_any(graph)
        g.x = self.norm(g.x + self.ff(g.x))
        return g.as_dict()


# -----------------------------------------------------------------------------#
# 10 EdgeUpdate
# -----------------------------------------------------------------------------#
@register("atom", "edge_update")
class EdgeUpdate(QueryInterface, nn.Module):
    """
    Edge‑wise MLP update that incorporates its incident node states.
    """
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_dim),
        )

    def forward(self, graph):
        g = GraphDict.from_any(graph)
        if g.edge_attr is None:
            return g.as_dict()
        src, dst = g.edge_index
        inp = torch.cat([g.x[src], g.x[dst], g.edge_attr], dim=-1)
        g.edge_attr = self.mlp(inp)
        return g.as_dict()


# -----------------------------------------------------------------------------#
# 11 FeedForwardBlock  (already exists as DenseStage / FeedForwardBlock)
# -----------------------------------------------------------------------------#
# -> nothing to add – Section 4 (DenseStage) & Section 5 (FeedForwardBlock)
#    correspond directly to menu row 11.


# -----------------------------------------------------------------------------#
# 12 Norm / Residual / Reg  (already provided in Section 3)
# -----------------------------------------------------------------------------#
# -> LayerNorm, Dropout, etc. were registered earlier.


# -----------------------------------------------------------------------------#
# 13 ReadoutPooler (set→vector)
# -----------------------------------------------------------------------------#
@register("atom", "readout_pool")
class ReadoutPooler(QueryInterface, nn.Module):
    def __init__(self, reduce: Literal["sum", "mean", "max"] = "mean"):
        super().__init__()
        self.reduce = reduce

    def forward(self, graph):
        g = GraphDict.from_any(graph)
        if g.batch is None:
            if self.reduce == "sum":
                return g.x.sum(dim=0, keepdim=True)  # (1, D)
            if self.reduce == "mean":
                return g.x.mean(dim=0, keepdim=True)
            if self.reduce == "max":
                return g.x.max(dim=0, keepdim=True).values
        # mini‑batch case – scatter per graph
        batch_ids = g.batch
        num_graphs = int(batch_ids.max().item()) + 1
        if self.reduce == "sum":
            return _scatter(g.x, batch_ids, num_graphs, "sum")
        if self.reduce == "mean":
            return _scatter(g.x, batch_ids, num_graphs, "mean")
        if self.reduce == "max":
            return _scatter(g.x, batch_ids, num_graphs, "max")
        raise ValueError(self.reduce)


# -----------------------------------------------------------------------------#
# 14 MultiScaleAggregator  (junction‑tree / fragment pooling placeholder)
# -----------------------------------------------------------------------------#
@register("atom", "multiscale_agg")
class MultiScaleAggregator(QueryInterface, nn.Module):
    """
    Two-level pooling stub:

        1. node → graph  (mean)
        2. optional concatenation with a supplied fragment-level vector
    """
    def __init__(self, reduce: Literal["sum", "mean", "max"] = "mean"):
        super().__init__()
        self.pool = ReadoutPooler(reduce)

    def forward(self,
                graph: Dict[str, Any] | "GraphDict",
                fragment_repr: Optional[torch.Tensor] = None
               ) -> torch.Tensor:
        graph_repr = self.pool(graph)                # (B, D)
        return graph_repr if fragment_repr is None else \
               torch.cat([graph_repr, fragment_repr], dim=-1)


# -----------------------------------------------------------------------------#
# 15 TaskHead  (classification / regression already implemented in Section 7)
# -----------------------------------------------------------------------------#
# -> The earlier `ClassificationHead` and `RegressionHead` correspond to
#    menu row 15.  Nothing to add.


# ============================================================================
# 4  DenseStage  (Projection → Norm → Act → Dropout)
# ============================================================================
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
        self.proj = get_from_registry("atom", "projection")(in_dim, out_dim)
        self.norm = (
            get_from_registry("atom", norm)(out_dim) if norm else nn.Identity()
        )
        self.act = get_from_registry("atom", act)()
        self.reg = get_from_registry("atom", "dropout")(dropout_p)

        self._latest_out: torch.Tensor | None = None
        self._name = local_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reg(self.act(self.norm(self.proj(x))))
        self._latest_out = x
        return x

    def get_activations(self):
        return {self._name: self._latest_out} if self._latest_out is not None else {}


# ============================================================================
# 5  Blocks & Trunks
# ============================================================================
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

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return x

    # ---- aggregate getters --------------------------------------------------
    def get_activations(self):
        acts = {}
        for stage in self.stages:
            acts.update(stage.get_activations())
        return acts


class SharedTrunk(QueryInterface, nn.Module):
    def __init__(self, dims: Sequence[int], **kw):
        super().__init__()
        self.block = FeedForwardBlock(dims, name_prefix="trunk", **kw)

    def forward(self, x):
        return self.block(x)

    def get_activations(self):
        return self.block.get_activations()


class PrivateTrunk(QueryInterface, nn.Module):
    def __init__(self, task: str, dims: Sequence[int], **kw):
        super().__init__()
        self.block = FeedForwardBlock(dims, name_prefix=f"{task}.trunk", **kw)

    def forward(self, x):
        return self.block(x)

    def get_activations(self):
        return self.block.get_activations()


# ============================================================================
# 6  Mixer units
# ============================================================================
@register("mixer", "cross_stitch")
class CrossStitchUnit(nn.Module):
    """
    **Classic Cross‑Stitch** (Misra *et al.*, ICCV 2016)

    Learns a **task‑wise convex combination** of hidden representations having
    the *exact same shape*.

    Parameters
    ----------
    num_tasks : int
        Number of tasks (and hence input / output tensors).

    Notes
    -----
    The learned parameter ``alpha`` is initialised to the identity matrix,
    meaning an *untouched* copy of each representation at the start of
    training.  Off‑diagonal entries are free to deviate during optimisation.
    """
    def __init__(self, num_tasks: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.eye(num_tasks))

    def forward(self, xs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Blend representations across tasks.

        Parameters
        ----------
        xs : list[Tensor]
            List length `num_tasks`, each ``(B, D)``.

        Returns
        -------
        list[Tensor]
            Same shapes as input, but *mixed* by ``alpha``.
        """
        stacked = torch.stack(xs, dim=0)                        # [T, B, D]
        mixed = torch.tensordot(self.alpha, stacked, dims=([1], [0]))  # [T, B, D]
        return list(mixed.unbind(dim=0))


@register("mixer", "sluice")
class SluiceUnit(nn.Module):
    """
    **Sluice Networks** (Ruder *et al.*, NAACL 2019) mixer.

    Splits each task representation into a “shared” and a “private” half,
    learns:

    1. A *cross‑task* mix of the shared halves (matrix **A**).
    2. A *within‑task* 2×2 gating of the shared/private parts (tensor **B**).

    Parameters
    ----------
    num_tasks : int
        Number of tasks.

    Implementation details
    ----------------------
    * Assumes the last dimension is even – it is halved into shared/private.
    * Both A and B start as identity transforms (no mixing → vanilla MTL).

    Returns
    -------
    list[Tensor]
        Mixed task representations, each same shape as its input.
    """
    def __init__(self, num_tasks: int):
        super().__init__()
        self.A = nn.Parameter(torch.eye(num_tasks))             # across‑task mixing
        self.B = nn.Parameter(torch.tensor([[1.0, 0.0], [0.0, 1.0]]).repeat(num_tasks, 1, 1))

    def forward(self, xs: List[torch.Tensor]) -> List[torch.Tensor]:
        shared_list, private_list = [], []
        for x in xs:
            d = x.size(-1)
            shared, private = torch.split(x, d // 2, dim=-1)
            shared_list.append(shared)
            private_list.append(private)

        shared_mixed = torch.tensordot(
            self.A, torch.stack(shared_list, dim=0), dims=([1], [0])
        )

        out = []
        for t in range(len(xs)):
            s, p = shared_mixed[t], private_list[t]
            gate = self.B[t]
            s_new = gate[0, 0] * s + gate[0, 1] * p
            p_new = gate[1, 0] * s + gate[1, 1] * p
            out.append(torch.cat([s_new, p_new], dim=-1))
        return out


# ============================================================================
# 7  Heads  (store predictions for get_preds)
# ============================================================================
@register("head", "classification")
class ClassificationHead(QueryInterface, nn.Module):
    def __init__(self, in_dim: int, num_classes: int, task: str):
        super().__init__()
        self.proj = nn.Linear(in_dim, num_classes)
        self.task = task
        self._logits: torch.Tensor | None = None

    def forward(self, x):
        self._logits = self.proj(x)
        return self._logits

    def get_preds(self):
        return {self.task: self._logits} if self._logits is not None else {}


@register("head", "regression")
class RegressionHead(QueryInterface, nn.Module):
    def __init__(self, in_dim: int, task: str):
        super().__init__()
        self.proj = nn.Linear(in_dim, 1)
        self.task = task
        self._out: torch.Tensor | None = None

    def forward(self, x):
        self._out = self.proj(x).squeeze(-1)
        return self._out

    def get_preds(self):
        return {self.task: self._out} if self._out is not None else {}


# ============================================================================
# 8  Base Lightning module + recipes (only changes: no ctx, use LossStack)
# ============================================================================
class BaseMultiTaskModel(pl.LightningModule, QueryInterface):
    def __init__(self):
        super().__init__()

    # ---- aggregated getters --------------------------------------------------
    def get_preds(self):
        preds = {}
        for m in self.modules():
            if m is not self:
                preds.update(getattr(m, "get_preds", lambda: {})())
        return preds

    def get_activations(self):
        acts = {}
        for m in self.modules():
            if m is not self:
                acts.update(getattr(m, "get_activations", lambda: {})())
        return acts

    # ---- Lightning hooks -----------------------------------------------------
    def training_step(self, batch, batch_idx):
        # Accept either (inputs, targets) or a single *inputs* object
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            inputs, targets = batch
        else:
            inputs, targets = batch, None

        self(inputs)                                          # forward pass
        loss_dict = self.loss_stack(model=self,
                                    batch=batch,
                                    epoch=self.current_epoch)
        self.log_dict(loss_dict, prog_bar=True)
        return loss_dict["total"]

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            inputs, targets = batch
        else:
            inputs, targets = batch, None

        self(inputs)
        val = self.loss_stack(model=self,
                              batch=batch,
                              epoch=self.current_epoch)
        #  align keys with Ray Tune callback: log **val_total_loss**
        self.log_dict({f"val_{k}": v for k, v in val.items()},
                      on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        lr = self.hparams.get("lr", 1e-3) if hasattr(self, "hparams") else 1e-3
        return torch.optim.Adam(self.parameters(), lr=lr)


# ---- STL_FNN (example; other recipes unchanged except for ctx removal) ---- #
@register("net", "stl_fnn")
class STL_FNN(BaseMultiTaskModel):
    def __init__(
        self,
        dims: Sequence[int],
        head_type: str,
        num_classes: int,
        loss_terms: List["LossTerm"],
    ):
        super().__init__()
        self.trunk = PrivateTrunk("A", dims)
        self.head = get_from_registry("head", head_type)(
            dims[-1], num_classes, task="A"
        )
        self.loss_stack = LossStack(loss_terms)

    def forward(self, x):
        h = self.trunk(x)
        return {"A": self.head(h)}
    
# ============================================================================ #
#  § GraphNeuralNetwork  #
# ============================================================================ #

ATOM_TYPES  = list(range(1, 101))       # one‑hot size = 100
BOND_TYPES  = {Chem.BondType.SINGLE: 0,
               Chem.BondType.DOUBLE: 1,
               Chem.BondType.TRIPLE: 2,
               Chem.BondType.AROMATIC: 3}

def _mol_to_graph_dict(mol: Chem.Mol) -> GraphDict:
    """Very small, dependency‑free featuriser (node‑one‑hot + bond‑type)."""
    # --- nodes ---------------------------------------------------------
    xs = []
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        feat = torch.zeros(len(ATOM_TYPES))
        if z in ATOM_TYPES:
            feat[z - 1] = 1.0
        xs.append(feat)
    x = torch.stack(xs, dim=0)                              # (N, 100)

    # --- edges ---------------------------------------------------------
    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        t = torch.tensor([BOND_TYPES[bond.GetBondType()]])
        edge_index.extend([[i, j], [j, i]])
        edge_attr.extend([t, t])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr  = torch.stack(edge_attr, dim=0)              # (E, 1)

    return GraphDict(x=x, edge_index=edge_index, edge_attr=edge_attr)

def _graphdict_to_bytes(g: GraphDict) -> bytes:
    """
    Lossless binary serialisation:
      • concatenate tensor byte‑streams
      • prepend int32 shapes to decode later
    """
    with io.BytesIO() as f:
        # store: 4 × int32 for shapes, then raw bytes
        for tens in (g.x, g.edge_index, g.edge_attr):
            torch.save(tens, f)            # keeps dtype/device info
        return f.getvalue()

def smiles_to_graph_bytes(smiles_list, num_workers: int = os.cpu_count() or 4):
    """
    Parallel RDKit featurisation → binary column (bytes).
    Returns the *same‑length* list of `bytes`.
    """
    def _one(smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles!r}")
        gdict = _mol_to_graph_dict(mol)
        return _graphdict_to_bytes(gdict)

    return joblib.Parallel(n_jobs=num_workers, backend="loky")(
        joblib.delayed(_one)(s) for s in smiles_list
    )

# ---- 2. Adapter –  any → GraphDict  -------------------------------------

def _bytes_to_graphdict(blob: bytes) -> GraphDict:
    with io.BytesIO(blob) as f:
        tensors = []
        while True:
            try:
                tensors.append(torch.load(f))
            except EOFError:
                break
    x, edge_index, edge_attr = tensors
    return GraphDict(x, edge_index, edge_attr)

def as_graph_dict(obj) -> GraphDict:
    """
    Accept:
      • GraphDict             → returns as‑is
      • bytes                 → decode via _bytes_to_graphdict
      • torch_geometric Data  → wrap
      • SMILES (str)          → parse & featurise on‑the‑fly
    """
    if isinstance(obj, GraphDict):
        return obj
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return _bytes_to_graphdict(obj)
    if isinstance(obj, Data):
        return GraphDict(
            x=obj.x, edge_index=obj.edge_index,
            edge_attr=getattr(obj, "edge_attr", None),
            batch=getattr(obj, "batch", None),
        )
    if isinstance(obj, str):
        # slow path – ad‑hoc featurise (should be rare if you pre‑cached)
        mol = Chem.MolFromSmiles(obj)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {obj!r}")
        return _mol_to_graph_dict(mol)
    raise TypeError(f"Unsupported graph representation: {type(obj)}")

#==============================================================================
# GraphNeuralNetwork Main Class
#==============================================================================

@register("net", "graph_nn")
class GraphNeuralNetwork(nn.Module, QueryInterface):
    """
    A *single‑layer* graph network assembled from 15 interchangeable modules.
    You can turn the layer into **any** architecture column from our menu
    simply by choosing (or zeroing‑out) the appropriate arguments.

    Parameters
    ----------
    node_embedder         : nn.Module | str | None
    edge_embedder         : nn.Module | str | None
    structural_encoder    : nn.Module | str | None
    message_fn            : nn.Module | str | None
    edge_attention        : nn.Module | str | None
    local_aggregator      : nn.Module | str | None
    global_attention      : nn.Module | str | None
    hybrid_router         : nn.Module | str | None
    node_update           : nn.Module | str | None
    edge_update           : nn.Module | str | None
    feed_forward          : nn.Module | str | None
    norm_reg              : nn.Module | str | None     (normalisation / dropout)
    readout_pool          : nn.Module | str | None
    multiscale_agg        : nn.Module | str | None
    task_head             : nn.Module | str | None     (regression / classification)

    *Any* argument may be `None` or the string `"identity"` to mean a no‑op.
    """

    def __init__(
        self,
        node_embedder,
        edge_embedder,
        structural_encoder,
        message_fn,
        edge_attention,
        local_aggregator,
        global_attention,
        hybrid_router,
        node_update,
        edge_update,
        feed_forward,
        norm_reg,
        readout_pool,
        multiscale_agg,
        task_head,
    ):
        super().__init__()

        # --- small helper ----------------------------------------------------
        def _resolve(obj_or_key, registry_kind="atom"):
            """Turn None / 'identity' / str / nn.Module into nn.Module."""
            if obj_or_key is None or obj_or_key == "identity":
                return nn.Identity()
            if isinstance(obj_or_key, str):
                return get_from_registry(registry_kind, obj_or_key)()
            return obj_or_key                                 # already Module

        # ---- store the 15 building blocks -----------------------------------
        self.node_embedder      = _resolve(node_embedder)
        self.edge_embedder      = _resolve(edge_embedder)
        self.structural_encoder = _resolve(structural_encoder)
        self.message_fn         = _resolve(message_fn)
        self.edge_attention     = _resolve(edge_attention)
        self.local_aggregator   = _resolve(local_aggregator)
        self.global_attention   = _resolve(global_attention)
        self.hybrid_router      = _resolve(hybrid_router)
        self.node_update        = _resolve(node_update)
        self.edge_update        = _resolve(edge_update)
        self.feed_forward       = _resolve(feed_forward)
        self.norm_reg           = _resolve(norm_reg)
        self.readout_pool       = _resolve(readout_pool)
        self.multiscale_agg     = _resolve(multiscale_agg)
        self.task_head          = _resolve(task_head, registry_kind="head")

    # --------------------------------------------------------------------- #
    #  forward – one full inference pass                                    #
    # --------------------------------------------------------------------- #
    def forward(self, graph):
        """
        Accepts either a raw *dict* or a `GraphDict`.  Returns a pair

            (graph‑dict after last node‑level op,
             task‑level prediction tensor | dict)

        Exactly which prediction object you get depends on the `task_head`.
        """
        graph = as_graph_dict(graph)
        g = GraphDict.from_any(graph)

        # 1‑3  embedding + structural encoding
        g = GraphDict.from_any(self.node_embedder(g))
        g = GraphDict.from_any(self.edge_embedder(g))
        g = GraphDict.from_any(self.structural_encoder(g))

        # 4‑6  message creation, local attention, neighbourhood aggregation
        g = GraphDict.from_any(self.message_fn(g))
        g = GraphDict.from_any(self.edge_attention(g))
        g_local = GraphDict.from_any(self.local_aggregator(g))

        # 7   global attention path (may be Identity)
        g_global = GraphDict.from_any(self.global_attention(g))

        # 8   hybrid router decides how to blend the two paths
        if isinstance(self.hybrid_router, nn.Identity):
            g = g_global   # either identity or global only
        else:
            g = GraphDict.from_any(self.hybrid_router(g_local, g_global))

        # 9‑10  node & edge updates
        g = GraphDict.from_any(self.node_update(g))
        g = GraphDict.from_any(self.edge_update(g))

        # 11‑12  extra per‑node feed‑forward and normalisation / dropout
        g.x = self.norm_reg(self.feed_forward(g.x))

        # 13‑15  read‑out → (optional) multi‑scale → task head
        if isinstance(self.multiscale_agg, nn.Identity):
            graph_vec = self.readout_pool(g)             # old behaviour
        else:
            # let the aggregator decide how / whether to pool
            graph_vec = self.multiscale_agg(g)
    
        prediction = self.task_head(graph_vec)
        return g.as_dict(), prediction

# ============================================================================
# 9  Functional loss system  (LossTerm + LossStack + helpers)
# ============================================================================
class LossTerm:
    """
    Lightweight record that bundles

        • name   – human-readable identifier
        • weight – scalar multiplier (λ)
        • fn     – callable (model, batch, epoch) → scalar tensor

    and makes the instance itself callable.
    """

    # optional: keep the class slim and prevent typos
    __slots__ = ("name", "weight", "fn")

    def __init__(self, name: str, weight: float, fn: callable):
        self.name = name
        self.weight = weight
        self.fn = fn

    def __call__(self, model, batch, epoch):
        """Return weighted loss value."""
        return self.weight * self.fn(model, batch, epoch)

    def __repr__(self):
        return (
            f"LossTerm(name={self.name!r}, "
            f"weight={self.weight}, "
            f"fn={self.fn.__name__ if hasattr(self.fn, '__name__') else self.fn})"
        )


class LossStack(nn.Module):
    def __init__(self, terms: List[LossTerm]):
        super().__init__()
        self.terms = terms

    def forward(self, *, model, batch, epoch):
        out = {}
        total = torch.zeros((), device=model.device)
        for term in self.terms:
            val = term(model, batch, epoch)
            out[term.name] = val.detach()
            total += val
        out["total"] = total
        return out


# ---- reusable functional cores --------------------------------------------
def ce_loss(task: str):
    criterion = nn.CrossEntropyLoss()

    def _fn(model, batch, epoch):
        preds = model.get_preds()[task]
        targets = batch[1][task]
        return criterion(preds, targets)

    return _fn


def l2_loss():
    def _fn(model, batch, epoch):
        vec = torch.cat(
            [p.view(-1) for p in model.get_learnable_weights().values()]
        )
        return (vec ** 2).sum()

    return _fn


# ============================================================================
# 10  Hyper‑parameter optimisation with Ray Tune
# ============================================================================
def tune_hyperparameters(
    build_model_fn: Callable[[Dict[str, Any]], pl.LightningModule],
    train_dataloader,
    val_dataloader,
    search_space: Dict[str, Any],
    metric: str = "val_total_loss",
    mode: str = "min",
    num_samples: int = 20,
    max_epochs: int = 10,
    resources_per_trial: Optional[Dict[str, float]] = None,
    name: str = "raytune_hpo",
    output_dir: Optional[str | Path] = None,   # <─ NEW
):
    """
    Launch a **Ray Tune** HPO loop around *any* Lightning model.

    Parameters
    ----------
    …
    name : str, default="raytune_hpo"
        Sub-directory inside ``output_dir`` (or ``~/ray_results``).
    output_dir : str | Path | None, default=None
        Root directory or cloud URI where Ray writes results and checkpoints.
        • ``None`` → Ray’s default ``~/ray_results``  
        • Local path → created if it doesn’t exist  
        • ``s3://…`` / ``gs://…`` / ``azure://…`` → synced via Ray’s Syncer
    """

    if tune is None:
        raise ImportError('Install Ray Tune with `pip install "ray[tune]"`.')

    # --------------------------------------------------------------------- #
    #  internal trainable – unchanged except for the callback               #
    # --------------------------------------------------------------------- #
    def _trainable(cfg):
        model = build_model_fn(cfg)

        tune_cb = TuneReportCheckpointCallback(
            metrics={metric: metric}, on="validation_end"
        )

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            logger=False,
            enable_checkpointing=True,
            callbacks=[tune_cb],
            enable_progress_bar=False,
        )

        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

    # --------------------------------------------------------------------- #
    #  Build RunConfig – inject storage_path only when user requested it    #
    # --------------------------------------------------------------------- #
    run_cfg_kwargs = {"name": name}
    if output_dir is not None:
        run_cfg_kwargs["storage_path"] = str(Path(output_dir).expanduser())

    tuner = tune.Tuner(
        _trainable,
        param_space=search_space,
        tune_config=TuneConfig(
            metric=metric,
            mode=mode,
            num_samples=num_samples,
            scheduler=ASHAScheduler(metric=metric, mode=mode),
        ),
        run_config=RunConfig(**run_cfg_kwargs),
        resources_per_trial=resources_per_trial or {"cpu": 1, "gpu": 0},
    )

    results = tuner.fit()
    best = results.get_best_result(metric=metric, mode=mode)
    print("Best hyper-parameters:\n", best.config)
    return best


if __name__ == "__main__" and os.getenv("RUN_TUNE_DEMO", "1") == "1":
    import pathlib, random, tempfile

    # ---------------------------------------------------------------
    # 1. Load CSV into Table (contains SMILES + Hit label)
    # ---------------------------------------------------------------
    DATA_CSV = "/Users/asselism/Desktop/Collins_Lab/Datasets/Organized_by_Data_Type/Bioactivity/Organized_by_Data_Type/Antimicrobial_Activity/Organized_by_Source/COADD/Modified_Datasets/Datasets_with_Binarized_Hits/Binarization_at_80_Percent_GI/CO_ADD_Dataset_EC_82K.csv"
    tbl = to_table(DATA_CSV)          # factory you already use

    # ---------------------------------------------------------------
    # 1 ½. Add binary graph‑tensor column  (cached, parallel)
    # ---------------------------------------------------------------
    if "Graph_Tensor" not in tbl.to_pandas().columns:
        print("🛠️  Pre‑computing graph tensors …")
        graph_bytes = smiles_to_graph_bytes(tbl["SMILES"].to_pandas())
        tbl = tbl.assign("Graph_Tensor", lambda df: graph_bytes, inplace=False)
        
    tbl.to_parquet("/Users/asselism/Desktop/Collins_Lab/Datasets/Organized_by_Data_Type/Bioactivity/Organized_by_Data_Type/Antimicrobial_Activity/Organized_by_Source/COADD/Modified_Datasets/Datasets_with_Binarized_Hits/Binarization_at_80_Percent_GI/CO_ADD_Dataset_EC_82K_with_Precomputed_Graphs.parquet")
    
    # ---------------------------------------------------------------
    # 2. Train / val / test split
    # ---------------------------------------------------------------
    train_tbl, val_tbl, test_tbl = TrainTestSplit(tbl, seed=42)

    # ---------------------------------------------------------------
    # 3. Define a tiny 1‑layer GNN (reuse registry)
    # ---------------------------------------------------------------
    my_gnn = get_from_registry("net", "graph_nn")(
        # simplest menu: only message‑passing + readout + head
        node_embedder      = "identity",
        edge_embedder      = "identity",
        structural_encoder = "identity",
        message_fn         = get_from_registry("atom", "message_fn")(in_dim=100*2+1,
                                                                     hidden_dim=128,
                                                                     out_dim=32),
        edge_attention     = "identity",
        local_aggregator   = get_from_registry("atom", "local_aggregator")("mean"),
        global_attention   = "identity",
        hybrid_router      = "identity",
        node_update        = get_from_registry("atom", "node_update")(dim=100),
        edge_update        = "identity",
        feed_forward       = "identity",
        norm_reg           = "identity",
        readout_pool       = get_from_registry("atom", "readout_pool")("mean"),
        multiscale_agg     = "identity",
        task_head          = get_from_registry("head", "classification")(in_dim=100,
                                                                         num_classes=2,
                                                                         task="A"),
    )

    # ---------------------------------------------------------------
    # 4. Wrap Table splits into PyG DataLoader (reads binary column)
    # ---------------------------------------------------------------
    def _tbl_to_loader(t: Table, shuffle: bool, bs: int = 32):
        gcol = t["Graph_Tensor"].to_pandas()
        ys   = t["Hit"].to_pandas().values.astype("float32")

        # decode on‑the‑fly but let DataLoader workers parallelise
        class _Dataset(torch.utils.data.Dataset):
            def __len__(self): return len(gcol)
            def __getitem__(self, idx):
                g = as_graph_dict(gcol.iloc[idx])
                g.batch = torch.zeros(g.x.size(0), dtype=torch.long)  # single graph
                g.y = torch.tensor([ys[idx]])
                return g
        return DataLoader(_Dataset(), batch_size=bs, shuffle=shuffle,
                          num_workers=os.cpu_count()//2)

    train_loader = _tbl_to_loader(train_tbl, shuffle=True)
    val_loader   = _tbl_to_loader(val_tbl,   shuffle=False)

    # ---------------------------------------------------------------
    # 5. Hyper‑parameter tuning stub (search lr only for brevity)
    # ---------------------------------------------------------------
    def build(cfg):                       # Ray will call this
        model = my_gnn
        model.save_hyperparameters(cfg)
        return model

    best = tune_hyperparameters(
        build_model_fn   = build,
        train_dataloader = train_loader,
        val_dataloader   = val_loader,
        search_space     = {"lr": tune.loguniform(1e-4, 1e-2)},
        metric           = "val_loss",
        mode             = "min",
        num_samples      = 5,
        max_epochs       = 3,
        resources_per_trial={"cpu": 4, "gpu": 0},
        output_dir       = tempfile.gettempdir(),
        name             = "graph_tensor_demo",
    )

    print("✅ Done. Best config:", best.config)
