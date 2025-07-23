#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ---------------------------------------------------------------------
# Unified scaffold for large-scale molecular visualisation & analysis
# using Ray for parallelism and runtime-environment isolation.
# ---------------------------------------------------------------------

from __future__ import annotations

# ── standard lib ─────────────────────────────────────────────────────
import pathlib
import cloudpickle as pickle                            # NEW – needed in several places
from typing import Iterable, Union, Dict, Callable, List, Any, Optional, Sequence, Literal, Generator
from collections import defaultdict
import math
import os
from itertools import combinations
import shutil
from functools import lru_cache, cached_property
import pickle
from pathlib import Path

# ── third-party core ─────────────────────────────────────────────────
import ray                                  # parallel execution / runtime envs
import ray.data
from ray.data import ActorPoolStrategy

import pandas as pd                         # tabular convenience
import numpy as np                          # numerics

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit import Chem, DataStructs
from rdkit.Chem import (
    Descriptors,
    QED,
    rdMolDescriptors,
    Lipinski,
    Crippen,
    AllChem,
)
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.DataStructs.cDataStructs import ExplicitBitVect

import pyarrow as pa
from pyarrow import csv as pa_csv

###############################################################################
# Paths to Environments:
###############################################################################

path_to_minimol_environment = "/Users/asselism/Desktop/Collins_Lab/Environments/Environments_for_Pipeline/Minimol_Environment/minimol_env.yml"

###############################################################################
# Shorthands:
###############################################################################

names_for_pseudomonas_aeruginosa = ('PA', "pseudomonas_aeruginosa", "Pseudomonas_aeruginosa", "Pseudomonas aeruginosa")

###############################################################################
# Common File Paths:
###############################################################################

ecbd_tox_data = "/Users/asselism/Desktop/Collins_Lab/Datasets/Organized_by_Data_Type/ADMET_Data/Organized_by_Data_Type/Toxicity/Organized_by_Source/ECBD/ECBD_HepG2_100K/Original_Datasets/ECBD_HepG2_STL_tanimoto.csv"
hundred_k_tox_data = "/Users/asselism/Desktop/Collins_Lab/Datasets/Organized_by_Data_Type/ADMET_Data/Organized_by_Data_Type/Toxicity/Organized_by_Source/Collins_Lab/100K_Toxicity_Screen/100K_Toxicity_Data_with_MM_FPs_Removed_Unfeaturized_Mols.tsv"
seventy_k_tox_data = "/Users/asselism/Desktop/Collins_Lab/Datasets/Organized_by_Data_Type/ADMET_Data/Organized_by_Data_Type/Toxicity/Organized_by_Source/Collins_Lab/70K_Toxicity_Screen/70K_Scored_and_With_MM_FPs.tsv"
tox21_data = "/Users/asselism/Desktop/Collins_Lab/Datasets/Organized_by_Source/Tox21/Unzipped_Files/Modified_Datasets/_tox21-rt-viability-hek293-p1.aggregrated.tsv"

###############################################################################
# Simple Functions:
###############################################################################

def average(items):
    
    return sum([float(val) for val in items])/len(items)

def combo(items, n):
    
    for subset in combinations(items, n):
        yield subset
        
def all_pairs(items):
    
    for subset in combo(items, 2):
        yield subset
        
def list_files(directory: str | Path, recursive: bool = False) -> List[Path]:
    """
    Return a list of file paths contained in *directory*.

    Parameters
    ----------
    directory : str | pathlib.Path
        Path to the directory you want to inspect.
    recursive : bool, default False
        • False → only files directly inside *directory*  
        • True  → include files in all nested sub-directories

    Returns
    -------
    List[pathlib.Path]
        Absolute paths for every file found.
    """
    directory = Path(directory).expanduser().resolve()

    if recursive:
        return [p for p in directory.rglob('*') if p.is_file()]
    else:
        return [p for p in directory.iterdir() if p.is_file()]

###############################################################################
# Ray helper – idempotent initialisation
###############################################################################
def _ensure_ray():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, namespace="chem-pipeline")

###############################################################################
# Data Table Functionality:
###############################################################################
try:
    import dask.dataframe as dd
except ImportError:  # pragma: no cover
    dd = None

try:
    import ray
    from ray.data import Dataset as _RayDataset
except ImportError:  # pragma: no cover
    ray = None
    _RayDataset = object            # type: ignore[assignment]
# --------------------------------------------------------------------------- #

# ........................................................................... #
#  Default: promote to Dask when a Pandas frame exceeds this many rows
#  Override via env‑var or change programmatically before constructing Tables.
# ........................................................................... #
PANDAS_PROMOTE_THRESHOLD: int = int(
    os.getenv("TABLE_PANDAS_PROMOTE_THRESHOLD", "5_000_000").replace("_", "")
)


class Table:
    """Unified DataFrame / Dask DataFrame / Ray Dataset wrapper."""

    # ====================================================================== #
    #  Constructor helpers
    # ====================================================================== #
    def __init__(
        self,
        table: Union[pd.DataFrame, "dd.DataFrame", "_RayDataset"],
        smiles_column_name: str = "SMILES",
        *,
        force_pandas: bool = False,
    ):
        """
        Parameters
        ----------
        table
            The concrete data structure to wrap.
        smiles_column_name
            Name of the canonical SMILES column (kept for caller‑level
            convenience – unused internally).
        force_pandas
            • ``True``  – keep a ``pd.DataFrame`` even if it exceeds the
              auto‑promotion threshold.  Use only when you are *certain*
              the machine has enough RAM.
        """
        # ------------------ Ray Dataset stays Ray -----------------------
        if _is_ray_dataset(table):
            self.table = table

        # ------------------ Already Dask DF ----------------------------
        elif _is_dask_dataframe(table):
            self.table = table

        # ------------------ Raw Pandas DF ------------------------------
        elif isinstance(table, pd.DataFrame):
            if (
                not force_pandas
                and dd is not None
                and len(table) > PANDAS_PROMOTE_THRESHOLD
            ):
                # Promote to an *out‑of‑core* Dask DataFrame
                self.table = dd.from_pandas(table, npartitions="auto")
            else:
                self.table = table
        else:  # pragma: no cover
            raise TypeError(f"Unsupported table type: {type(table)}")

        self.smiles_column_name = smiles_column_name

    # ------------------------------------------------------------------ #
    # Convenience factory (kept for API parity)
    # ------------------------------------------------------------------ #
    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        smiles_col: str = "SMILES",
        *,
        force_pandas: bool = False,
    ) -> "Table":
        return cls(df, smiles_column_name=smiles_col, force_pandas=force_pandas)

    # ====================================================================== #
    #  Back‑end predicates
    # ====================================================================== #
    def is_ray(self) -> bool:
        return _is_ray_dataset(self.table)

    def is_dask(self) -> bool:
        return _is_dask_dataframe(self.table)

    def is_pandas(self) -> bool:
        return isinstance(self.table, pd.DataFrame)

    # ====================================================================== #
    #  Simple helpers
    # ====================================================================== #
    def get_name(self) -> str | None:
        return getattr(self, "name", None)

    # ------------------------------------------------------------------ #
    # Constant‑time (cached) length for Ray; lazy for Dask
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:  # noqa: D401  (simple expression)
        if self.is_ray():
            return self._rowcount
        if self.is_dask():
            return int(self.table.shape[0].compute())
        return len(self.table)

    @cached_property
    def _rowcount(self) -> int:
        # Only called when the first __len__ happens on a Ray table
        return int(self.table.count())

    # ------------------------------------------------------------------ #
    #  Column access – never eager‑collect unless you *ask* for it
    # ------------------------------------------------------------------ #
    def __getitem__(self, column_name: str):
        """
        Returns a *lazy* Series / Dataset – collect with ``.compute()`` /
        ``.to_pandas()`` when you truly need it in memory.
        """
        if self.is_ray():
            return self.table.select_columns(column_name)

        if self.is_dask():
            return self.table[column_name]          # still lazy

        return self.table[column_name]              # tiny, already in RAM

    # ====================================================================== #
    #  Materialisation guard
    # ====================================================================== #
    def to_pandas(self, *, max_rows: int | None = 5_000_000) -> pd.DataFrame:
        """
        Collect the entire table into a single Pandas DataFrame **only if**
        the total number of rows does not exceed ``max_rows`` (``None`` disables
        the check – use with care!).

        Raises
        ------
        MemoryError
            If the requested materialisation is likely to overflow memory.
        """
        est = len(self)
        if max_rows is not None and est > max_rows:
            raise MemoryError(
                f"Refusing to materialise {est:,} rows in‑memory "
                "(override via max_rows=None if you *really* want that)."
            )

        if self.is_ray():
            return self.table.to_pandas()

        if self.is_dask():
            return self.table.compute()

        return self.table  # already Pandas

    # ====================================================================== #
    #  Sorting (distributed when possible)
    # ====================================================================== #
    def sort(
        self,
        key: Union[str, Callable[[pd.Series], Any]],
        *,
        ascending: bool = True,
        inplace: bool = True,
        batch_size: int = 4_096,   # Ray only
        **ray_opts,
    ) -> "Table":
        # ------------------ Ray branch (unchanged) ----------------------
        if self.is_ray():
            sorted_ds = _ray_sort(self.table, key, ascending, batch_size, **ray_opts)
            return _maybe_update(self, sorted_ds, inplace)

        # ------------------ Dask branch --------------------------------
        if self.is_dask():
            sorted_df = _dask_sort(self.table, key, ascending)
            return _maybe_update(self, sorted_df, inplace)

        # ------------------ Small Pandas branch ------------------------
        sorted_df = _pandas_sort(self.table, key, ascending)
        return _maybe_update(self, sorted_df, inplace)

    # ====================================================================== #
    #  Chunk iterator – streaming in all back‑ends
    # ====================================================================== #
    def iter_chunks(self, n: int = 1_000) -> Generator[pd.DataFrame, None, None]:
        """
        Yield batches of at most ``n`` rows as *Pandas* DataFrames.
        """
        if self.is_ray():
            yield from self.table.iter_batches(batch_size=n, batch_format="pandas")
            return

        if self.is_dask():
            # Each Dask partition is a block; iterate partition‑wise and
            # further slice into ``n``‑row sub‑frames so that the yield size
            # is predictable.
            for part in self.table.partitions:
                pdf = part.compute()
                for i in range(0, len(pdf), n):
                    yield pdf.iloc[i : i + n]
            return

        # Plain Pandas – slice directly
        for i in range(0, len(self.table), n):
            yield self.table.iloc[i : i + n]

    # ====================================================================== #
    #  Vectorised assign
    # ====================================================================== #
    def assign(
        self,
        column_name: str,
        func: Union[str, Callable[[pd.Series | pd.DataFrame], Any]],
        *,
        inplace: bool = True,
        batch_size: int = 4_096,  # Ray only
        **ray_opts,
    ) -> "Table":
        """
        *func* may be

            • str         – Pandas/Dask eval expression
            • Series UDF   – receives column Series
            • DataFrame UDF– receives partition frame
        """
        if self.is_ray():
            new_ds = _ray_assign(self.table, column_name, func,
                                 batch_size=batch_size, **ray_opts)
            return _maybe_update(self, new_ds, inplace)

        if self.is_dask():
            new_df = _dask_assign(self.table, column_name, func)
            return _maybe_update(self, new_df, inplace)

        new_df = _pandas_assign(self.table, column_name, func)
        return _maybe_update(self, new_df, inplace)
    
    # ──────────────────────────────────────────────────────────────────────
    #  assign_rowwise_using()  – guaranteed per‑row UDF assignment
    # ──────────────────────────────────────────────────────────────────────
    
    def assign_rowwise_using(
        self,
        column_name: str,
        func: Callable[[pd.Series], Any],
        *,
        inplace: bool = True,
        batch_size: int = 4_096,        # Ray only
        **ray_opts,
    ) -> "Table":
        """
        Evaluate *func* **once per row** and store the return value in
        *column_name*.

        Parameters
        ----------
        column_name : str
            Name of the destination column to create / overwrite.
        func : Callable[[pd.Series], Any]
            A Python callable that receives one row at a time
            (Pandas `Series`) and returns a scalar.
        inplace : bool, default True
            • ``True``  → mutate the current `Table`  
            • ``False`` → return a *new* `Table` instance
        batch_size : int, Ray‑only
            Forwarded to `Dataset.map_batches`.
        **ray_opts
            Extra keyword arguments for Ray’s `map_batches`
            (e.g. `runtime_env`, `compute_strategy`).

        Notes
        -----
        * This is a *strict* row‑wise helper.  
          If you pass a vectorised function, it *will* still be called
          once per row.  Use :meth:`assign` when you want the library to
          auto‑detect (and exploit) a vectorised UDF.
        * The implementation mirrors :meth:`filter_rowwise_using` for
          consistency across back‑ends.
        """
        # ------------------ Ray branch --------------------------------
        if self.is_ray():
            new_ds = _ray_assign_rowwise(self.table, column_name, func,
                                          batch_size, **ray_opts)
            return _maybe_update(self, new_ds, inplace)

        # ------------------ Dask branch -------------------------------
        if self.is_dask():
            new_df = _dask_assign_rowwise(self.table, column_name, func)
            return _maybe_update(self, new_df, inplace)

        # ------------------ Pandas branch -----------------------------
        new_df = _pandas_assign_rowwise(self.table, column_name, func)
        return _maybe_update(self, new_df, inplace)

    # ====================================================================== #
    #  Row‑wise boolean filtering (vectorised when possible)
    # ====================================================================== #
    def filter_rowwise_using(
        self,
        func: Callable[[pd.Series], bool],
        *,
        inplace: bool = False,
        batch_size: int = 4_096,  # Ray only
        **ray_opts,
    ) -> "Table":
        if self.is_ray():
            new_ds = _ray_filter(self.table, func, batch_size, **ray_opts)
            return _maybe_update(self, new_ds, inplace)

        if self.is_dask():
            mask = self.table.map_partitions(lambda df: df.apply(func, axis=1))
            new_df = self.table[mask]
            return _maybe_update(self, new_df, inplace)

        mask = self.table.apply(func, axis=1)
        new_df = self.table[mask]
        return _maybe_update(self, new_df, inplace)

    # ====================================================================== #
    #  Export – parallel for Ray & Dask, serial for Pandas
    # ====================================================================== #
    def _export(
        self,
        path: pathlib.Path | str,
        fmt: str,
        *,
        sep: str | None = None,
        single_file: bool = False,
        **kwargs,
    ) -> pathlib.Path:
        """
        Parameters
        ----------
        fmt : {"csv", "tsv", "parquet"}
        single_file
            When using Dask, write a single output file **only** for smallish
            data sets.  The default (``False``) keeps a directory of part files,
            which parallel writers can create safely.
        """
        path = pathlib.Path(path).expanduser().resolve()

        # ---------------------------- Ray ------------------------------
        if self.is_ray():
            _ray_export(self.table, path, fmt, sep, **kwargs)
            return path

        # --------------------------- Dask ------------------------------
        if self.is_dask():
            _dask_export(self.table, path, fmt, sep, single_file, **kwargs)
            return path

        # ------------------------- Pandas ------------------------------
        _pandas_export(self.table, path, fmt, sep, **kwargs)
        return path

    # Public wrappers – kept unchanged
    def to_csv(self, path: pathlib.Path | str, **kwargs) -> pathlib.Path:
        return self._export(path, fmt="csv", sep=",", **kwargs)

    def to_tsv(self, path: pathlib.Path | str, **kwargs) -> pathlib.Path:
        return self._export(path, fmt="tsv", sep="\t", **kwargs)

    def to_parquet(self, path: pathlib.Path | str, **kwargs) -> pathlib.Path:
        return self._export(path, fmt="parquet", **kwargs)

    # ====================================================================== #
    #  Head / tail / sample  (distributed + safe)
    # ====================================================================== #
    # ---- input validator ------------------------------------------------
    @staticmethod
    def _validate_n(n: int, *, method: str) -> int:
        if not isinstance(n, int) or n < 0:
            raise ValueError(f"{method}: 'n' must be a non‑negative int, got {n!r}")
        return n

    # ---- head() ---------------------------------------------------------
    def head(self, n: int = 5) -> "Table":
        n = self._validate_n(n, method="head")
        if n == 0:
            return Table(self.table.limit(0) if self.is_ray() else self.table.iloc[0:0],
                         smiles_column_name=self.smiles_column_name)

        if self.is_ray():
            return Table(self.table.limit(n), smiles_column_name=self.smiles_column_name)

        if self.is_dask():
            return Table(self.table.head(n), smiles_column_name=self.smiles_column_name)

        return Table(self.table.head(n), smiles_column_name=self.smiles_column_name)

    # ---- tail() ---------------------------------------------------------
    def tail(self, n: int = 10) -> "Table":
        n = self._validate_n(n, method="tail")
        if n == 0 or len(self) == 0:
            return Table(self.table.limit(0) if self.is_ray() else self.table.iloc[0:0],
                         smiles_column_name=self.smiles_column_name)

        if self.is_ray():
            new_ds = self.table.skip(max(len(self) - n, 0))
            return Table(new_ds, smiles_column_name=self.smiles_column_name)

        if self.is_dask():
            return Table(self.table.tail(n), smiles_column_name=self.smiles_column_name)

        return Table(self.table.tail(n), smiles_column_name=self.smiles_column_name)

    # ---- sample() -------------------------------------------------------
    def sample(self, n: int = 10, *, seed: int | None = None) -> "Table":
        n = self._validate_n(n, method="sample")
        total = len(self)
        if n > total:
            raise ValueError(f"sample: 'n' ({n}) exceeds table size ({total})")

        if n == 0:
            return Table(self.table.limit(0) if self.is_ray() else self.table.iloc[0:0],
                         smiles_column_name=self.smiles_column_name)

        if self.is_ray():
            new_ds = self.table.random_shuffle(seed=seed).limit(n)
            return Table(new_ds, smiles_column_name=self.smiles_column_name)

        if self.is_dask():
            frac = n / total
            return Table(self.table.sample(frac=frac, random_state=seed),
                         smiles_column_name=self.smiles_column_name)

        return Table(self.table.sample(n=n, random_state=seed),
                     smiles_column_name=self.smiles_column_name)

    # ====================================================================== #
    #  Convenience wrappers around sort()
    # ====================================================================== #
    def top(self, key: Union[str, Callable], n: int = 10) -> "Table":
        n = self._validate_n(n, method="top")
        return self.sort(key, ascending=False, inplace=False).head(n)

    def bottom(self, key: Union[str, Callable], n: int = 10) -> "Table":
        n = self._validate_n(n, method="least")
        return self.sort(key, ascending=True, inplace=False).head(n)

# ----------------------------------------------------------------------------- #
#  assign_rowwise_using() helpers
# ----------------------------------------------------------------------------- #
def _ray_assign_rowwise(ds: "_RayDataset", col: str, func,
                        batch_size: int, **opts):
    def _apply(batch: pd.DataFrame) -> pd.DataFrame:
        batch[col] = batch.apply(func, axis=1)
        return batch
    return ds.map_batches(_apply, batch_size=batch_size,
                          batch_format="pandas", **opts)


def _dask_assign_rowwise(ddf: "dd.DataFrame", col: str, func):
    return ddf.assign(
        **{col: ddf.map_partitions(lambda df: df.apply(func, axis=1))}
    )


def _pandas_assign_rowwise(df: pd.DataFrame, col: str, func):
    df = df.copy()
    df[col] = df.apply(func, axis=1)
    return df

# =============================================================================
#  Helper utilities – factored out for readability
# =============================================================================
def _is_dask_dataframe(obj: Any) -> bool:
    return dd is not None and isinstance(obj, dd.DataFrame)  # type: ignore[arg-type]


def _is_ray_dataset(obj: Any) -> bool:
    return ray is not None and isinstance(obj, _RayDataset)  # type: ignore[arg-type]


# ----------------------------------------------------------------------------- #
#  Sorting back‑end helpers
# ----------------------------------------------------------------------------- #
def _ray_sort(ds: "_RayDataset", key, ascending, batch_size, **opts) -> "_RayDataset":
    if isinstance(key, str):
        return ds.sort(key=key, descending=not ascending)

    SORT_COL = "__sort_key__"

    def _with_key(batch: pd.DataFrame) -> pd.DataFrame:
        batch[SORT_COL] = batch.apply(key, axis=1)
        return batch

    return (
        ds.map_batches(_with_key, batch_size=batch_size,
                       batch_format="pandas", **opts)
          .sort(key=SORT_COL, descending=not ascending)
          .drop_columns([SORT_COL])
    )


def _dask_sort(ddf: "dd.DataFrame", key, ascending: bool):
    if isinstance(key, str):
        return ddf.set_index(
            key, drop=False, sorted=True, ascending=ascending,
        )
    # Callable key – add temp column partition‑wise
    tmp = ddf.map_partitions(lambda df: df.assign(__sort_key__=df.apply(key, axis=1)))
    sorted_df = tmp.set_index("__sort_key__", sorted=True,
                              drop=True, ascending=ascending)
    return sorted_df.drop(columns="__sort_key__")


def _pandas_sort(df: pd.DataFrame, key, ascending: bool):
    if isinstance(key, str):
        return df.sort_values(by=key, ascending=ascending, kind="mergesort")
    tmp = df.assign(__sort_key__=df.apply(key, axis=1))
    return (tmp.sort_values("__sort_key__", ascending=ascending, kind="mergesort")
               .drop(columns="__sort_key__"))


# ----------------------------------------------------------------------------- #
#  assign() helpers
# ----------------------------------------------------------------------------- #
def _ray_assign(ds: "_RayDataset", col: str, func, *, batch_size, **opts):
    def _apply(batch: pd.DataFrame) -> pd.DataFrame:
        batch[col] = _apply_udf(batch, func)
        return batch

    return ds.map_batches(_apply, batch_size=batch_size,
                          batch_format="pandas", **opts)


def _dask_assign(ddf: "dd.DataFrame", col: str, func):
    if isinstance(func, str):
        return ddf.assign(**{col: ddf.eval(func)})
    # func is callable – treat as vectorised across columns/rows
    return ddf.assign(**{col: ddf.map_partitions(_apply_udf, func=func)})


def _pandas_assign(df: pd.DataFrame, col: str, func):
    df = df.copy()
    df[col] = _apply_udf(df, func)
    return df


def _apply_udf(df: pd.DataFrame, func):
    """Handles Series UDF, DataFrame UDF, or expression str (already filtered)."""
    if callable(func):
        try:
            return func(df)  # DataFrame UDF?
        except Exception:    # Fallback to row‑wise
            return df.apply(func, axis=1)
    raise TypeError("func must be a callable or expression string")


# ----------------------------------------------------------------------------- #
#  filter() helper
# ----------------------------------------------------------------------------- #
def _ray_filter(ds: "_RayDataset", func, batch_size, **opts):
    def _filter(batch: pd.DataFrame) -> pd.DataFrame:
        mask = batch.apply(func, axis=1)
        if mask.dtype != bool:
            raise ValueError("filter_rowwise_using: predicate must return booleans")
        return batch[mask]

    return ds.map_batches(_filter, batch_size=batch_size,
                          batch_format="pandas", **opts)


# ----------------------------------------------------------------------------- #
#  Export helpers
# ----------------------------------------------------------------------------- #

def _ray_export(ds, path, fmt: str, sep: str | None, **kwargs):
    path = pathlib.Path(path).expanduser().resolve()

    if fmt == "parquet":
        ds.write_parquet(str(path), **kwargs)
        return

    if fmt == "csv":
        # Rely on Arrow defaults → no pickling problems
        ds.write_csv(str(path), **kwargs)
        return

    if fmt == "tsv":
        # Arrow/Ray cannot change the delimiter without WriteOptions.
        # Fallback: materialise one partition at a time and stream to TSV.
        #
        # NOTE: works for reasonably‑sized data; for very large data you
        #       would write part‑files and post‑process with Unix tools.
        import pandas as pd, csv, tempfile, os
        tmp_dir = tempfile.mkdtemp(prefix="ray_csv_")
        ds.write_csv(tmp_dir, **kwargs)          # write default CSV parts
        out_path = str(path)
        with open(out_path, "w", newline="") as fout:
            writer = None
            for f in sorted(os.listdir(tmp_dir)):
                for chunk in pd.read_csv(os.path.join(tmp_dir, f),
                                         chunksize=1_000_000):
                    if writer is None:
                        writer = csv.writer(fout, delimiter="\t",
                                            lineterminator="\n")
                        writer.writerow(chunk.columns)
                    writer.writerows(chunk.itertuples(index=False, name=None))
        shutil.rmtree(tmp_dir)
        return

    raise ValueError(f"Unsupported format: {fmt}")



def _dask_export(ddf: "dd.DataFrame", path: pathlib.Path, fmt: str,
                 sep: str | None, single_file: bool, **kwargs):
    if fmt == "parquet":
        ddf.to_parquet(path, write_index=False, **kwargs)
    elif fmt in {"csv", "tsv"}:
        delimiter = "," if fmt == "csv" else "\t"
        if single_file:
            ddf.to_csv(str(path), single_file=True, sep=delimiter,
                       index=False, **kwargs)
        else:
            # Write part‑files under a directory
            path.mkdir(parents=True, exist_ok=True)
            ddf.to_csv(str(path / "*.csv"), sep=delimiter,
                       index=False, **kwargs)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported format: {fmt!r}")


def _pandas_export(df: pd.DataFrame, path: pathlib.Path, fmt: str,
                   sep: str | None, **kwargs):
    if fmt == "parquet":
        df.to_parquet(path, index=False, **kwargs)
    elif fmt in {"csv", "tsv"}:
        delimiter = "," if fmt == "csv" else "\t"
        df.to_csv(path, sep=delimiter, index=False, **kwargs)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported format: {fmt!r}")


# ----------------------------------------------------------------------------- #
#  Utility: mutate‑or‑return‑new
# ----------------------------------------------------------------------------- #
def _maybe_update(tbl: Table, new_obj, inplace: bool) -> Table:
    if inplace:
        tbl.table = new_obj
        return tbl
    return Table(new_obj, smiles_column_name=tbl.smiles_column_name)
    
###############################################################################
# Other Table Functionality:
###############################################################################

from ray.air.util.tensor_extensions.arrow import ArrowConversionError
from ray.data.exceptions import SystemException        # Ray ≥ 2.9
import pyarrow as pa
import ray, pandas as pd

def _arrow_csv_safely(path: str):
    """
    Try Ray’s fast CSV reader; if Arrow/Ray chokes on an irregular file,
    fall back to Pandas (engine='python', on_bad_lines='skip').
    """
    try:
        ds = ray.data.read_csv(path)
        ds.schema()            # force schema inference NOW
        return ds

    # ---- Anything that signals a schema / conversion failure ----------
    except (
        pa.ArrowInvalid,
        pa.ArrowTypeError,
        ArrowConversionError,   # Ray’s wrapper around Arrow errors
        SystemException,        # Ray task failed during read
        KeyError,               # rare: missing column header
    ) as err:
        print(
            "[path_to_table] Ray CSV reader failed: "
            f"{type(err).__name__}: {err}. Falling back to Pandas."
        )

    # ---- LAST line of defence: absolutely everything else -------------
    except Exception as err:
        print(
            "[path_to_table] Unexpected error in Ray CSV reader "
            f"({type(err).__name__}: {err}). Falling back to Pandas."
        )

    # ---------- Slow but safe path -------------------------------------
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    return ray.data.from_pandas(df)

def path_to_table(path: pathlib.Path | str):
    _ensure_ray()
    path = pathlib.Path(path)
    ext  = path.suffix.lower()

    if ext == ".csv":
        return _arrow_csv_safely(str(path))

    # ----------------------------------------------------------------
    # ❷  TSV – you already special‑case this
    # ----------------------------------------------------------------
    if ext == ".tsv":
        df = pd.read_csv(path, sep="\t")
        return ray.data.from_pandas(df)

    # ----------------------------------------------------------------
    # ❸  Parquet (unchanged)
    # ----------------------------------------------------------------
    if ext in {".parquet", ".pq"}:
        return ray.data.read_parquet(str(path))

    raise ValueError(f"Unsupported extension: {ext}")

def to_table(obj: Any) -> Table:
    """Normalise many possible inputs into a `Table`."""
    
    if isinstance(obj, (str, pathlib.Path)):
        return Table(table=path_to_table(pathlib.Path(obj)))
    
    if isinstance(obj, pd.DataFrame):
        return Table(table=obj)
    
    if isinstance(obj, np.ndarray):
        return Table(table=pd.DataFrame(obj))
    
    raise TypeError(f"Cannot convert object of type {type(obj)} to Table")
    
###############################################################################
# Chemical Filters:
###############################################################################

def _build_catalogues() -> tuple[FilterCatalog, FilterCatalog]:
    """Return (PAINS catalogue, BRENK catalogue)."""
    pains_params = FilterCatalogParams()
    pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
    pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
    pains_catalog = FilterCatalog(pains_params)

    brenk_params = FilterCatalogParams()
    brenk_params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    brenk_catalog = FilterCatalog(brenk_params)

    return pains_catalog, brenk_catalog


_PAINS_CAT, _BRENK_CAT = _build_catalogues()

# ── small, frequently used helpers ───────────────────────────────────────────
def _mol_or_none(smiles: str) -> Chem.Mol | None:
    """Lightweight SMILES→Mol converter that never raises."""
    return Chem.MolFromSmiles(smiles) if smiles else None


def _mol_required(smiles: str) -> Chem.Mol:
    """Convert SMILES to Mol or raise a *clear* ValueError."""
    mol = _mol_or_none(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles!r}")
    return mol


# ── PAINS / BRENK filters ────────────────────────────────────────────────────
def pass_pains_filter(smiles: str) -> bool:
    """True ⇢ *no* PAINS alerts.  False otherwise or on parse failure."""
    mol = _mol_or_none(smiles)
    return bool(mol) and not _PAINS_CAT.HasMatch(mol)


def pass_brenk_filter(smiles: str) -> bool:
    """True ⇢ *no* BRENK alerts."""
    mol = _mol_or_none(smiles)
    return bool(mol) and not _BRENK_CAT.HasMatch(mol)

def lipinskis_rules(smiles: str) -> bool:
    """Lipinski’s Rule‑of‑Five."""
    mol = _mol_or_none(smiles)
    if not mol:
        return False
    return (
        Descriptors.MolWt(mol) <= 500
        and Crippen.MolLogP(mol) <= 5
        and Lipinski.NumHDonors(mol) <= 5
        and Lipinski.NumHAcceptors(mol) <= 10
    )


"""
entry_rules.py  ── Test whether a molecule satisfies the Hergenrother
                  eNTRy permeability rules for Gram-negative uptake.

Rule summary (Richter et al., Nature 545, 299-304 (2017))               ──────────
    1.  ≥ 1 non-sterically-encumbered, ionisable nitrogen
        • Operationalised here as a primary amine (–NH₂) that is *not* part
          of an amide, sulfonamide, etc.

    2.  Rotatable bonds ≤ 5
        • Count single bonds not in a ring, bound to a non-terminal heavy
          atom, **excluding C–N amide bonds with restricted rotation**.

    3.  Globularity ≤ 0.25
        • Globularity ≔ λ₍min₎ / λ₍max₎, the inverse condition number of
          the covariance matrix of 3-D atomic coordinates (1 = sphere,
          0 = perfectly flat). :contentReference[oaicite:0]{index=0}

Returns
-------
entry_rules(smiles: str) → bool
    True  – all three criteria are satisfied
    False – any criterion is violated

Dependencies
------------
- RDKit  : conda install -c conda-forge rdkit
- NumPy  : pip/conda install numpy
"""

from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

def _mol_or_none(smiles: str) -> Optional[Chem.Mol]:
    """Return an RDKit Mol or None if SMILES cannot be parsed/sanitised."""
    try:
        return Chem.MolFromSmiles(smiles, sanitize=True)
    except Exception as e:
        print(f"Failed to convert {smiles} to a mol object.")
        return None


def _has_primary_amine(mol: Chem.Mol) -> bool:
    """
    Detect at least one *non-amide* primary amine (–NH₂ / –NH₃⁺).

    SMARTS:
        [N;H2;D1]              : primary N
        !$(N[*]=O)             : exclude amides
        !$(N[*]S(=O)=O)        : exclude sulfonamides
        !$(N[a])               : exclude aromatic N
    """
    primary_amine = Chem.MolFromSmarts(
        "[N;H2;D1;!$(N[*]=O);!$(N[*]S(=O)=O);!$(N[a])]"
    )
    return bool(mol.HasSubstructMatch(primary_amine))


def _count_rotatable_bonds(mol: Chem.Mol) -> int:
    """
    Count rotatable bonds **excluding amide C–N** (strict definition).

    RDKit’s API changed in 2023.09: the second positional argument
    became an enum (`NumRotatableBondsOptions`).  Earlier versions
    accept a boolean `strict`.

    This shim tries the new enum first, then falls back.
    """
    try:
        # New API (RDKit ≥ 2023.09)
        opts_enum = rdMolDescriptors.NumRotatableBondsOptions.Strict
        return int(rdMolDescriptors.CalcNumRotatableBonds(mol, opts_enum))
    except (AttributeError, TypeError):
        # Legacy API (RDKit ≤ 2023.03) – bool `strict`
        return int(rdMolDescriptors.CalcNumRotatableBonds(mol, True))


def _embed_and_get_coords(mol: Chem.Mol) -> Optional[np.ndarray]:
    """
    Embed a single ETKDGv3 conformer, return (N, 3) NumPy array of coords.
    Return None on embedding failure.

    Works with all RDKit versions ≥ 2020.03.  For new builds we set
    `maxAttempts` *inside* the `EmbedParameters`; for very old builds
    we fall back to the legacy signature.
    """
    # Add hydrogens before embedding (recommended for ETKDG)
    mol3d = Chem.AddHs(mol)

    # Always try to use the most recent ETKDG flavour available
    try:
        params = AllChem.ETKDGv3()
    except AttributeError:  # very old RDKit
        params = AllChem.ETKDGv2()

    # Reproducibility
    params.randomSeed = 0xF00D
    # Number of embedding attempts – put it *in* the params object
    if hasattr(params, "maxAttempts"):
        params.maxAttempts = 20

    # ----------------------
    # Embedding (new API first)
    # ----------------------
    try:
        # Modern overload: (mol, EmbedParameters)
        status = AllChem.EmbedMolecule(mol3d, params)
    except TypeError:
        # Old overload: (mol, maxAttempts, randomSeed, …)
        status = AllChem.EmbedMolecule(
            mol3d,
            maxAttempts=20,
            randomSeed=0xF00D,
            clearConfs=True,
        )

    if status != 0:
        return None  # embedding failed

    # Local optimisation (UFF)
    AllChem.UFFOptimizeMolecule(mol3d, maxIters=200)
    conf = mol3d.GetConformer()
    return np.asarray(conf.GetPositions(), dtype=float)


def _globularity(coords: np.ndarray) -> float:
    """
    λ_min / λ_max of covariance matrix of centred coordinates.
    0 ≤ globularity ≤ 1, where lower = flatter.
    """
    centred = coords - coords.mean(axis=0)
    cov = np.cov(centred, rowvar=False)
    eigvals = np.sort(np.linalg.eigvalsh(cov))  # ascending
    return 0.0 if eigvals[-1] < 1e-8 else float(eigvals[0] / eigvals[-1])


# ---------------------------------------------------------------------
# New helper functions for P. aeruginosa rules
# ---------------------------------------------------------------------
def _formal_charge(mol: Chem.Mol) -> int:
    """
    Return the net formal charge encoded in the SMILES.

    NOTE: This ignores pH‑dependent protonation; use with caution.
    """
    return int(Chem.GetFormalCharge(mol))


def _num_hbd(mol: Chem.Mol) -> int:
    """RDKit count of hydrogen‑bond donors."""
    return int(rdMolDescriptors.CalcNumHBD(mol))


def _approx_hbd_surface_area(mol: Chem.Mol) -> float:
    """
    VERY rough surrogate for the MOE descriptor `vsa_don`.

    Each donor contributes ≈12 Å² (empirical average for sp³ N/O).
    This gives the correct scale for the 23 Å² cut‑off in the paper.
    """
    return _num_hbd(mol) * 12.0


def _tpsa(mol: Chem.Mol) -> float:
    """
    Total polar surface area (TPSA) as an inexpensive proxy
    for the MOE ‘positive polar surface area’ descriptor.

    Empirically, almost all molecules with TPSA ≥ 80 Å² and a
    positive formal charge satisfy Q_vsa_PPos ≥ 80 Å².  The
    substitution is conservative (risk of *false negatives*).
    """
    return float(rdMolDescriptors.CalcTPSA(mol))


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def entry_rules(smiles: str, species=None) -> bool:  # noqa: C901 complexity OK for clarity
    """
    Return True iff *smiles* satisfies the published ‘entry’ rules
    for the requested *species*.

    - *species is None* → E. coli (original eNTRy rules)
    - *species == "PA"* → P. aeruginosa (porin‑independent rules)
    """
    try:
        mol = _mol_or_none(smiles)
        if mol is None:
            return False
    except Exception:
        print("Error occurred during SMILES conversion to mol object.")
        return False

    # ------------------------------------------------------------
    # E. coli (original eNTRy rules) -----------------------------
    # ------------------------------------------------------------
    if species is None:
        # Rule 1 – primary amine present
        if not _has_primary_amine(mol):
            return False

        # Rule 2 – rigid (rotatable bonds ≤ 5)
        if _count_rotatable_bonds(mol) > 5:
            return False

        # Rule 3 – sufficiently flat (globularity ≤ 0.25)
        coords = _embed_and_get_coords(mol)
        if coords is None:
            print("Coords could not be generated.")
            return False  # conservative: fail if 3‑D cannot be generated
        if _globularity(coords) > 0.25:
            return False

        return True

    # ------------------------------------------------------------
    # Pseudomonas aeruginosa (porin‑independent rules) -----------
    # Geddes et al., Nature 2023                                   
    # ------------------------------------------------------------
    elif species in ("PA",):
        # ---------- Rule 1 — net positive charge -----------------
        #
        # Accept either an *explicit* formal charge ≥ +1 **or**
        # the presence of a primary amine (most are >90 % protonated
        # at physiological pH, giving an effective +1 charge).
        #
        has_positive_charge = (_formal_charge(mol) >= 1) or _has_primary_amine(mol)
        if not has_positive_charge:
            return False

        # ---------- Rule 2 — H‑bond‑donor surface area ≥ 23 Å² ----
        #
        # Using the 12 Å²‑per‑donor heuristic, this equates to ≥2 donors.
        #
        if _approx_hbd_surface_area(mol) < 23.0:
            return False

        # ---------- Rule 3 — *either* (PPSA ≥ 80 Å²) or (formal charge ≥ +1) --
        #
        # MOE’s ‘positive polar surface area’ (Q_vsa_PPos) is approximated
        # here with total TPSA.  This is imperfect but errs on the side of
        # stringency (false negatives over false positives).
        #
        if not ((_formal_charge(mol) >= 1) or (_tpsa(mol) >= 80.0)):
            return False

        return True

    # ------------------------------------------------------------
    # Unsupported species
    # ------------------------------------------------------------
    else:
        raise ValueError(f"Unknown species '{species}'. Valid options: None (E. coli) or 'PA'.")

def veber_rules(smiles: str) -> bool:
    mol = _mol_or_none(smiles)
    if mol is None:
        return False
    rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    return rot_bonds <= 10 and (tpsa <= 140 or (hbd + hba) <= 12)


def ghose_filter(smiles: str) -> bool:
    mol = _mol_or_none(smiles)
    if mol is None:
        return False
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    mr = Descriptors.MolMR(mol)
    natoms = mol.GetNumAtoms()
    return all(
        [
            160 <= mw <= 480,
            -0.4 <= logp <= 5.6,
            40 <= mr <= 130,
            20 <= natoms <= 70,
        ]
    )


def muegge_filter(smiles: str) -> bool:
    mol = _mol_or_none(smiles)
    if mol is None:
        return False
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    ring_count = mol.GetRingInfo().NumRings()
    return all(
        [
            200 <= mw <= 600,
            -2 <= logp <= 5,
            tpsa <= 150,
            hbd <= 5,
            hba <= 10,
            rot_bonds <= 15,
            1 <= ring_count <= 7,
        ]
    )


def egan_rule(smiles: str) -> bool:
    mol = _mol_or_none(smiles)
    if mol is None:
        return False
    logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    return logp < 5.88 and tpsa < 131.6


def pfizer_3_75_filter(smiles: str) -> bool:
    mol = _mol_or_none(smiles)
    if mol is None:
        return False
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    return hbd <= 3 and tpsa <= 75


def gsk_4_400_filter(smiles: str) -> bool:
    mol = _mol_or_none(smiles)
    if mol is None:
        return False
    mw = Descriptors.MolWt(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    return hbd <= 4 and mw <= 400


def rule_of_three(smiles: str) -> bool:
    mol = _mol_or_none(smiles)
    if mol is None:
        return False
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    return mw <= 300 and logp <= 3 and hbd <= 3 and hba <= 3


# ── CNS MPO helpers ──────────────────────────────────────────────────────────
def calc_cns_mpo_score(mol: Chem.Mol) -> float:
    """Approximate 0‑to‑6 CNS MPO score (Wager et al.)."""
    clogp = Crippen.MolLogP(mol)
    clogd = clogp  # crude approximation
    mw = Descriptors.MolWt(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    pka = 8.0

    def s(x, k, x0):
        return 1.0 / (1 + math.exp(k * (x - x0)))

    score = (
        s(clogp, 0.5, 3)
        + s(clogd, 0.5, 3)
        + s(mw, 0.035, 360)
        + s(tpsa, 0.07, 60)
        + s(hbd, 1.0, 1)
        + s(pka, 1.0, 8)
    )
    return score


def cns_mpo_filter(smiles: str, threshold: float = 4.0) -> bool:
    mol = _mol_or_none(smiles)
    return bool(mol) and calc_cns_mpo_score(mol) >= threshold

# ──────────────────────────────────────────────────────────────────────────────
#  ChEMBL structural‑alert filters
#
#  Alert sets and entry counts in ChEMBL 33                                (ref)
#    • Glaxo            55   • Dundee           105
#    • BMS             180   • PAINS            479
#    • SureChEMBL      166   • MLSMR            116
#    • Inpharmatica     91   • LINT              57   :contentReference[oaicite:0]{index=0}
#
#  NB  The RDKit 2023.03 release added all eight sets under
#      `FilterCatalogParams.FilterCatalogs.CHEMBL_*`.                    :contentReference[oaicite:1]{index=1}
# ──────────────────────────────────────────────────────────────────────────────

# ---- internal helper ---------------------------------------------------------
def _build_chembl_catalog(flag_name: str) -> FilterCatalog | None:
    """
    Build a FilterCatalog containing *exactly* one ChEMBL alert set.
    Returns None if the running RDKit build does not expose the flag.
    """
    try:
        flag = getattr(FilterCatalogParams.FilterCatalogs, flag_name)
    except AttributeError as e: 
        print(f"Failed to instantiate the module {flag_name} due to {e}.")                # RDKit < 2023.03 or typo
        return None

    params = FilterCatalogParams()
    params.AddCatalog(flag)
    return FilterCatalog(params)


# ---- one singleton catalogue per alert set ----------------------------------
_CHEMBL_Glaxo_CAT         = _build_chembl_catalog("CHEMBL_Glaxo")
_CHEMBL_Dundee_CAT        = _build_chembl_catalog("CHEMBL_Dundee")
_CHEMBL_BMS_CAT           = _build_chembl_catalog("CHEMBL_BMS")
_CHEMBL_SureChEMBL_CAT    = _build_chembl_catalog("CHEMBL_SureChEMBL")
_CHEMBL_MLSMR_CAT         = _build_chembl_catalog("CHEMBL_MLSMR")
_CHEMBL_Inpharmatica_CAT  = _build_chembl_catalog("CHEMBL_Inpharmatica")
_CHEMBL_LINT_CAT          = _build_chembl_catalog("CHEMBL_LINT")

# ---- public API --------------------------------------------------------------
def matches_glaxo_filter(smiles: str) -> bool:
    """True ⇢ SMILES triggers a **Glaxo** structural alert."""
    mol = _mol_or_none(smiles)
    return bool(mol) and _CHEMBL_Glaxo_CAT is not None and _CHEMBL_Glaxo_CAT.HasMatch(mol)


def matches_dundee_filter(smiles: str) -> bool:
    """True ⇢ SMILES triggers a **Dundee** alert (NTD screening set)."""
    mol = _mol_or_none(smiles)
    return bool(mol) and _CHEMBL_Dundee_CAT is not None and _CHEMBL_Dundee_CAT.HasMatch(mol)


def matches_bms_filter(smiles: str) -> bool:
    """True ⇢ SMILES triggers a **BMS (Bristol‑Myers Squibb)** alert."""
    mol = _mol_or_none(smiles)
    return bool(mol) and _CHEMBL_BMS_CAT is not None and _CHEMBL_BMS_CAT.HasMatch(mol)


def matches_pains_filter(smiles: str) -> bool:
    """
    True ⇢ SMILES triggers at least one **PAINS** alert.
    (Opposite semantics to existing `pass_pains_filter`.)
    """
    mol = _mol_or_none(smiles)
    return bool(mol) and _PAINS_CAT.HasMatch(mol)   # _PAINS_CAT was built earlier


def matches_surechembl_filter(smiles: str) -> bool:
    """True ⇢ SMILES triggers a **SureChEMBL** alert."""
    mol = _mol_or_none(smiles)
    return bool(mol) and _CHEMBL_SureChEMBL_CAT is not None and _CHEMBL_SureChEMBL_CAT.HasMatch(mol)


def matches_mlsmr_filter(smiles: str) -> bool:
    """True ⇢ SMILES triggers an **NIH MLSMR** excluded‑functionality alert."""
    mol = _mol_or_none(smiles)
    return bool(mol) and _CHEMBL_MLSMR_CAT is not None and _CHEMBL_MLSMR_CAT.HasMatch(mol)


def matches_inpharmatica_filter(smiles: str) -> bool:
    """True ⇢ SMILES triggers an **Inpharmatica** alert."""
    mol = _mol_or_none(smiles)
    return bool(mol) and _CHEMBL_Inpharmatica_CAT is not None and _CHEMBL_Inpharmatica_CAT.HasMatch(mol)


def matches_lint_filter(smiles: str) -> bool:
    """True ⇢ SMILES triggers a **LINT** (Lead‑Investigator Negative Toolkit) alert."""
    mol = _mol_or_none(smiles)
    return bool(mol) and _CHEMBL_LINT_CAT is not None and _CHEMBL_LINT_CAT.HasMatch(mol)

all_tox_filters = [
    matches_glaxo_filter,
    matches_dundee_filter,
    matches_bms_filter,
    matches_pains_filter,
    matches_surechembl_filter,
    matches_mlsmr_filter,
    matches_inpharmatica_filter,
    matches_lint_filter
    ]

all_tox_filter_names = [
    "matches_glaxo_filter",
    "matches_dundee_filter",
    "matches_bms_filter",
    "matches_pains_filter",
    "matches_surechembl_filter",
    "matches_mlsmr_filter",
    "matches_inpharmatica_filter",
    "matches_lint_filter"
    ]

###############################################################################
# Chemical Property Prediction:
###############################################################################

def get_mw(smiles: str) -> float | None:
    mol = _mol_or_none(smiles)
    return Descriptors.MolWt(mol) if mol else None


def get_tpsa(smiles: str) -> float | None:
    mol = _mol_or_none(smiles)
    return Descriptors.TPSA(mol) if mol else None


def get_num_heavy_atoms(smiles: str) -> int | None:
    mol = _mol_or_none(smiles)
    return mol.GetNumHeavyAtoms() if mol else None


def get_clogp(smiles: str) -> float | None:
    mol = _mol_or_none(smiles)
    return Crippen.MolLogP(mol) if mol else None


def get_sascore(smiles: str) -> float | None:
    """Ertl‑Schuffenhauer synthetic accessibility score."""
    mol = _mol_or_none(smiles)
    return rdMolDescriptors.CalcSAScore(mol) if mol else None


def get_qed(smiles: str) -> float | None:
    mol = _mol_or_none(smiles)
    return QED.qed(mol) if mol else None


def get_num_hba(smiles: str) -> int | None:
    mol = _mol_or_none(smiles)
    return Descriptors.NumHAcceptors(mol) if mol else None


def get_num_hbd(smiles: str) -> int | None:
    mol = _mol_or_none(smiles)
    return Descriptors.NumHDonors(mol) if mol else None

###############################################################################
# Chemical Distance and Similarity Metrics:
###############################################################################

#==============================================================================
# Distance Metric Computation:
#==============================================================================

# -- two builtin single-pair metrics ----------------------------------
def tanimoto_distance(smi1: str, smi2: str) -> float:
    """1 − Tanimoto similarity on Morgan-2 / 2048-bit fingerprints."""
    m1, m2 = Chem.MolFromSmiles(smi1), Chem.MolFromSmiles(smi2)
    if m1 is None or m2 is None:
        return 1.0
    fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2, nBits=2048)
    return 1.0 - DataStructs.TanimotoSimilarity(fp1, fp2)

def graph_edit_distance(smi1: str, smi2: str) -> float:
    """Slow toy graph-edit distance (RDKit)."""
    from rdkit.Chem import rdFMCS
    m1, m2 = Chem.MolFromSmiles(smi1), Chem.MolFromSmiles(smi2)
    if m1 is None or m2 is None:
        return 1.0
    res = rdFMCS.FindMCS([m1, m2], completeRingsOnly=True, ringMatchesRingOnly=True)
    mcs = Chem.MolFromSmarts(res.smartsString) if res.smartsString else None
    if mcs is None:
        return 1.0
    return 1.0 - mcs.GetNumAtoms() / max(m1.GetNumAtoms(), m2.GetNumAtoms())

###############################################################################
# Fingerprint and Other Representation Computation:
###############################################################################

#==============================================================================
# Minimol Representation
#==============================================================================

def get_minimol_fingerprint(smiles: str):
    os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
    os.environ.setdefault("JOBLIB_N_JOBS",          "1")
    _force_graphium_float32()
    from minimol import Minimol
    model = Minimol()
    return model(smiles)

# ---------------------------------------------------------------------
# Graphium → float32 shim  (avoid SciPy float16 bug – local to Minimol)
# ---------------------------------------------------------------------
def _force_graphium_float32():
    """
    Monkey‑patch Graphium's featuriser so every internal call that would
    have used dtype=float16 is transparently promoted to float32.

    Called once per process **immediately before** the first Minimol()
    instantiation.  Harmless no‑op if Graphium is absent or already patched.
    """
    try:
        from graphium.features import featurizer as _gf          # only in Minimol env
        import numpy as np
        from functools import partial
    except ImportError:          # Graphium not present outside the Minimol env
        return

    if getattr(_gf, "_dtype_patched", False):   # idempotent
        return

    _gf.mol_to_adj_and_features = partial(_gf.mol_to_adj_and_features, dtype=np.float32)
    _gf.mol_to_graph_dict      = partial(_gf.mol_to_graph_dict,      dtype=np.float32)
    _gf.mol_to_pyggraph        = partial(_gf.mol_to_pyggraph,        dtype=np.float32)

    _gf._dtype_patched = True   # mark as done
    
def _init_minimol_single_thread():
    """
    Instantiate `Minimol()` once *without* the inner joblib multiprocess pool.
    This avoids Ray‑in‑Ray pickling errors.
    """
    # 1‑line opt‑out for joblib
    os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
    os.environ.setdefault("JOBLIB_N_JOBS",          "1")

    _force_graphium_float32()
    from minimol import Minimol
    model = Minimol()

    # Make absolutely sure the Graphium datamodule does not respawn processes
    try:
        dm = model.datamodule
        dm.featurization_backend = "threading"   # or "serial"
        dm.featurization_n_jobs  = 1
        dm.num_workers           = 0            # PyTorch DataLoader
    except AttributeError:
        pass

    return model

class _MinimolActor:
    """
    Ray *actor* that owns exactly one MiniMol model instance for the lifetime
    of the worker.  The class is itself pickle‑able, so Ray can ship it to
    every node, but the heavy model never crosses a process boundary.
    """
    def __init__(self, smiles_col: str, output_col: str):
        self._smiles_col  = smiles_col
        self._output_col  = output_col
        self._model       = _init_minimol_single_thread()

    def __call__(self, batch_df: "pd.DataFrame") -> "pd.DataFrame":
        fps, bad = [], []
        for idx, smi in enumerate(batch_df[self._smiles_col].astype(str)):
            try:
                fp = self._model(smi)[0]          # fingerprint only
                fps.append(fp.tolist() if hasattr(fp, "tolist") else fp)
            except Exception:
                fps.append(None)
                bad.append(idx)

        if bad:
            print(f"[Minimol] skipped {len(bad)} invalid SMILES "
                  f"out of {len(batch_df)} in batch")

        batch_df[self._output_col] = fps
        return batch_df

# ──────────────────────────────────────────────────────────────────────
# 1.  add_minimol_fingerprints – now calls that one-liner
# ──────────────────────────────────────────────────────────────────────
def add_minimol_fingerprints(
    tbl: Table,
    *,
    override_smiles_column_name_as: Optional[str] = None,
    output_column: str = "Minimol_Representation",
    batch_size: int = 1_024,
) -> Table:

    _ensure_ray()

    smiles_col = override_smiles_column_name_as or tbl.smiles_column_name
    if smiles_col is None:
        raise ValueError("No SMILES column defined on the Table object.")

    # ------------------------------------------------------------------
    # Ray path – one MiniMol instance per worker via *actors*
    # ------------------------------------------------------------------
    if tbl.is_ray():
        strategy = ActorPoolStrategy()
        tbl.table = tbl.table.map_batches(
            # The *class* (not an instance) – Ray will build one per actor
            _MinimolActor,
            # kwargs forwarded to the actor constructor
            fn_constructor_kwargs=dict(
                smiles_col = smiles_col,
                output_col = output_column,
            ),
            batch_format = "pandas",
            batch_size   = batch_size,
            compute      = strategy,                          # ← key line
            runtime_env  = {"conda": "Pipeline_Minimol_Environment"},
        )
        return tbl

    # ------------------------------------------------------------------
    # Fallback: single‑process Pandas (no Ray)
    # ------------------------------------------------------------------
    actor = _MinimolActor(smiles_col, output_column)
    processed = [actor(chunk) for chunk in tbl.iter_chunks(batch_size)]
    tbl.table = pd.concat(processed, ignore_index=True)
    return tbl

#==============================================================================
# Synthetic Assesability Score Calculator:
#==============================================================================

# ────────────────────────────── SA score ────────────────────────────────────
def get_sa_score(smiles: str) -> float | None:
    """Ertl‑Schuffenhauer SA score (1 easy → 10 hard)."""
    mol = _mol_or_none(smiles)
    if not mol:
        return None

    # Fast C++ path (RDKit ≥ 2023.09)
    try:
        return rdMolDescriptors.CalcSAScore(mol)
    except AttributeError:
        pass

    # Fallback Python path (all older RDKit builds)
    try:
        from rdkit.Chem.SA_Score import sascorer
        return float(sascorer.calculateScore(mol))
    except Exception:
        return None


# ────────────────────────────── RA score ────────────────────────────────────
_RA_ACTOR_NAME = "__ra_scorer__"
_RASCORE_RUNTIME_ENV = {
    "pip": [
        "git+https://github.com/reymond-group/RAscore.git@master",
        "xgboost>=1.7,<2.0",
        "scikit-learn>=1.0,<2.0",
    ]
}

@ray.remote(runtime_env=_RASCORE_RUNTIME_ENV)
class _RAscoreActor:
    def __init__(self):
        from RAscore import RAscore_XGB
        self._scorer = RAscore_XGB.RAScorerXGB()

    def score(self, smiles: str) -> float | None:
        try:
            return float(self._scorer.predict_from_smiles([smiles])[0])
        except Exception:
            return None


def init_ra_scorer_if_needed() -> None:
    """Create the detached RA‑scorer exactly once per Ray cluster."""
    _ensure_ray()                                    # your helper
    try:
        ray.get_actor(_RA_ACTOR_NAME)
    except ValueError:                               # not running
        _RAscoreActor.options(
            name=_RA_ACTOR_NAME,
            lifetime="detached",
        ).remote()


def get_ra_score(smiles: str) -> float | None:
    """0–1 probability that AiZynthFinder finds a synthetic route."""
    try:
        actor = ray.get_actor(_RA_ACTOR_NAME)
        return ray.get(actor.score.remote(smiles))
    except (ValueError, ray.exceptions.RayActorError):
        # scorer missing or crashed → None (caller can handle NaNs)
        return None

#==============================================================================
# Other Functionality:
#==============================================================================

def df_to_smiles_dict(dist_df: pd.DataFrame) -> Dict[int, str]:
    """
    Build `{row_index → SMILES}` from a long-form distance DataFrame.
    Prefers the side-car lookup (original order) and falls back to
    categorical codes only if needed.                           # <- fix #6
    """
    lookup = dist_df.attrs.get("smiles_lookup")
    if lookup is not None:
        return lookup.to_dict()
    
    if pd.api.types.is_categorical_dtype(dist_df["idx_i"]):
        cats = dist_df["idx_i"].cat.categories
        return {i: smi for i, smi in enumerate(cats)}
    
    raise ValueError(
        "Distance DataFrame does not contain Categorical indices or "
        "'smiles_lookup' attribute – cannot rebuild SMILES dictionary."
    )

distance_metric_defaults = defaultdict(
    lambda: None,
    {"tanimoto": tanimoto_distance, "ged": graph_edit_distance},
)

@ray.remote
def pairwise_distance_on_chunks(
    smiles_iterable_one: List[str],
    smiles_iterable_two: List[str],
    metric_function_blob: bytes,
    col_a_offset: int,
    col_b_offset: int,
) -> pd.DataFrame:
    """Compute all distances for a chunk-pair (upper-triangle only)."""
    metric_function: Callable[[str, str], float] = pickle.loads(metric_function_blob)
    
    rows: list[tuple[int, int, float]] = []
    for i, s1 in enumerate(smiles_iterable_one):
        for j, s2 in enumerate(smiles_iterable_two):
            if col_a_offset + i > col_b_offset + j:
                continue
            rows.append((col_a_offset + i,
                         col_b_offset + j,
                         metric_function(s1, s2)))
    return pd.DataFrame(rows, columns=("idx_i", "idx_j", "distance"))

def bulk_tanimoto_distance_df(
    smiles_a: list[str],
    smiles_b: list[str],
    *,
    offset_a: int = 0,
    offset_b: int = 0,
    upper_triangle: bool = True,
    n_bits: int = 1024,
    radius: int = 2,
) -> pd.DataFrame:
    # ------------------------------------------------------------------
    zero_fp = ExplicitBitVect(n_bits)                       # ➋ NEW

    def _fp(smi: str):
        m = Chem.MolFromSmiles(smi)
        return (
            AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
            if m is not None
            else None
        )

    fps_a = [_fp(s) or zero_fp for s in smiles_a]           # ➌ NEW
    fps_b = [_fp(s) or zero_fp for s in smiles_b]           # ➍ NEW

    rows = []
    for i, fp_a in enumerate(fps_a):
        # with the substitution we never have fp_a is None, so no need to skip
        sims = DataStructs.BulkTanimotoSimilarity(fp_a, fps_b)
        for j, sim in enumerate(sims):
            if upper_triangle and (offset_a + i > offset_b + j):
                continue
            dist = 1.0 - sim
            rows.append((offset_a + i, offset_b + j, dist))

    return pd.DataFrame(rows, columns=("idx_i", "idx_j", "distance"))


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Ray-remote wrapper so we can parallelise by chunk                        │
# ──────────────────────────────────────────────────────────────────────────────
import ray

@ray.remote
def pairwise_bulk_tanimoto_on_chunks(
    smiles_chunk_one: list[str],
    smiles_chunk_two: list[str],
    col_a_offset: int,
    col_b_offset: int,
) -> pd.DataFrame:
    """Ray task that delegates to `bulk_tanimoto_distance_df`."""
    return bulk_tanimoto_distance_df(
        smiles_chunk_one,
        smiles_chunk_two,
        offset_a=col_a_offset,
        offset_b=col_b_offset,
        upper_triangle=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Modified `pairwise_distances` – treats 'tanimoto' as a special-case      │
# ──────────────────────────────────────────────────────────────────────────────
def pairwise_distances(
    table: Table,
    metric: Union[str, Callable[[str, str], float]] = "tanimoto",
    *,
    chunk_size: int = 4_096,
) -> Table:
    """
    Compute a long-form distance table (`idx_i`, `idx_j`, `distance`).
    Uses **bulk Tanimoto** for the keyword ``metric="tanimoto"``; every
    other metric falls back to the generic per-pair worker.
    """
    _ensure_ray()  # existing helper ------------------------------------------------

    # ------------------------------------------------------------------
    # 0.  Slice SMILES into chunks once (needed for *both* code-paths)
    # ------------------------------------------------------------------
    smiles_chunks, pos_offsets, running_total = [], [], 0
    for chunk in table.iter_chunks(chunk_size):
        s = chunk[table.smiles_column_name].astype(str).tolist()
        smiles_chunks.append(s)
        pos_offsets.append(running_total)
        running_total += len(s)

    # ------------------------------------------------------------------
    # 1.  Choose execution strategy
    # ------------------------------------------------------------------
    is_bulk_tani = isinstance(metric, str) and metric.lower() == "tanimoto"

    if not is_bulk_tani:
        # Resolve callable / pickle for *all other* metrics -------------
        if isinstance(metric, str):
            metric_fn = distance_metric_defaults[metric.lower()]
            if metric_fn is None:
                raise ValueError(
                    f"Unknown metric '{metric}'. Choose one of "
                    f"{list(distance_metric_defaults)} or pass a callable."
                )
        elif callable(metric):
            metric_fn = metric
        else:
            raise TypeError("metric must be str or Callable[[str,str],float]")

        metric_blob = pickle.dumps(metric_fn)

    # ------------------------------------------------------------------
    # 2.  Launch Ray tasks (upper-triangle grid)
    # ------------------------------------------------------------------
    futures: list[ray.ObjectRef] = []
    for i, (chunk_a, off_a) in enumerate(zip(smiles_chunks, pos_offsets)):
        for j in range(i, len(smiles_chunks)):
            chunk_b, off_b = smiles_chunks[j], pos_offsets[j]
            if is_bulk_tani:
                futures.append(
                    pairwise_bulk_tanimoto_on_chunks.remote(
                        chunk_a, chunk_b, off_a, off_b
                    )
                )
            else:
                futures.append(
                    pairwise_distance_on_chunks.remote(
                        chunk_a, chunk_b, metric_blob, off_a, off_b
                    )
                )

    # ------------------------------------------------------------------
    # 3.  Gather → concat
    # ------------------------------------------------------------------
    full_df = pd.concat(ray.get(futures), ignore_index=True)

    # ------------------------------------------------------------------
    # 4.  Categorical encoding (store SMILES once, unchanged)
    # ------------------------------------------------------------------
    all_smi = [s for ch in smiles_chunks for s in ch]
    if len(all_smi) == len(set(all_smi)):
        cats = pd.Index(all_smi, name="SMILES")
        for col in ("idx_i", "idx_j"):
            full_df[col] = pd.Categorical.from_codes(
                full_df[col].to_numpy(np.int32), categories=cats, ordered=False
            )
    else:
        full_df.attrs["smiles_lookup"] = pd.Series(all_smi, name="SMILES")

    # ------------------------------------------------------------------
    # 5.  Wrap and return – keeps API identical
    # ------------------------------------------------------------------
    return Table(table=full_df, smiles_column_name="idx_i")

###############################################################################
# Embedding Functions:
###############################################################################

@ray.remote(runtime_env={"pip": ["umap-learn"]})                       # fix #9
def run_umap(dists: np.ndarray, n_components: int, **kwargs):
    import umap
    return umap.UMAP(metric="precomputed",
                     n_components=n_components,
                     **kwargs).fit_transform(dists)

@ray.remote(runtime_env={"pip": ["scikit-learn"]})                    # fix #9
def run_tsne(dists: np.ndarray, n_components: int, **kwargs):
    from sklearn.manifold import TSNE
    # If the caller did not specify an init, fall back to "random"
    kwargs.setdefault("init", "random")
    return TSNE(
        metric="precomputed",
        n_components=n_components,
        **kwargs,
    ).fit_transform(dists)

def embed_using_pairwise_distances(
    pairwise_distances: Union[pd.DataFrame, "ray.data.Dataset", Table],
    embedding_method: Union[str, Callable] = "umap",
    n_components: int = 2,
    groups: Optional[np.ndarray] = None,
    plot_path: str = "embedding.jpeg",
    **kwargs,
) -> pd.DataFrame:
    """
    Embed a long-form distance table; return DF with SMILES + coordinates.
    """

    #-------------------------------------------------------------------
    # 0. Normalise input to pandas
    #-------------------------------------------------------------------
    if isinstance(pairwise_distances, Table):
        dist_df = (pairwise_distances.table
                   if not pairwise_distances.is_ray()
                   else pairwise_distances.table.to_pandas())
    elif isinstance(pairwise_distances, ray.data.Dataset):
        dist_df = pairwise_distances.to_pandas()
    elif isinstance(pairwise_distances, pd.DataFrame):
        dist_df = pairwise_distances
    else:
        raise TypeError("pairwise_distances must be DataFrame, Dataset, or Table")

    #-------------------------------------------------------------------
    # 1. Dense distance matrix
    #-------------------------------------------------------------------
    id_to_smiles = df_to_smiles_dict(dist_df)
    n = len(id_to_smiles)
    D = np.zeros((n, n), dtype=np.float32)

    if pd.api.types.is_categorical_dtype(dist_df["idx_i"]):            # fix #5
        i_idx = dist_df["idx_i"].cat.codes.to_numpy(np.int32)
        j_idx = dist_df["idx_j"].cat.codes.to_numpy(np.int32)
    else:
        i_idx = dist_df["idx_i"].to_numpy(np.int32)
        j_idx = dist_df["idx_j"].to_numpy(np.int32)

    d_val = dist_df["distance"].to_numpy(np.float32)
    D[i_idx, j_idx] = d_val
    D[j_idx, i_idx] = d_val
    np.fill_diagonal(D, 0.0)

    #-------------------------------------------------------------------
    # 2. Resolve embedding method
    #-------------------------------------------------------------------
    if isinstance(embedding_method, str):
        key = embedding_method.lower()
        if key == "umap":
            embedding_method = run_umap
        elif key in {"tsne", "t-sne"}:
            embedding_method = run_tsne
        else:
            raise ValueError(f"Unsupported embedding method: {embedding_method}")

    #-------------------------------------------------------------------
    # 3. Ray task → coords
    #-------------------------------------------------------------------
    coords = ray.get(embedding_method.remote(D, n_components, **kwargs))

    #-------------------------------------------------------------------
    # 4. Build output DF
    #-------------------------------------------------------------------
    smi_order = [id_to_smiles[i] for i in range(n)]
    col_names = ["SMILES"] + [f"Dim{k+1}" for k in range(n_components)]
    embed_df = pd.DataFrame(np.column_stack([smi_order, coords]), columns=col_names)
    for k in range(1, n_components + 1):
        embed_df[f"Dim{k}"] = embed_df[f"Dim{k}"].astype(float)

    if groups is not None and len(groups) == n:
        embed_df.insert(len(embed_df.columns), "Group", groups)

    return embed_df

# ---------------------------------------------------------------------
# 4. Scatter plotting – simple Matplotlib JPEG
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Scatter‑plot helper – runs in its own Ray runtime environment
# ---------------------------------------------------------------------
_SCATTER_RUNTIME_ENV = {
    # Add whatever you need that is *not* already in the base cluster env
    "pip": [
        "matplotlib",           # plotting
        "pillow",               # image back‑end for JPEG/PNG writing
        "numpy",                # numeric
        "pandas",               # DataFrame (small)
    ]
}

@ray.remote(runtime_env=_SCATTER_RUNTIME_ENV)
def _scatter_plot_worker(
    df_bytes: bytes,
    x_col: str,
    y_col: str,
    color: str | None,
    marker: str,
    dpi: int,
    group_col: str,
    title,
    kwargs_blob: bytes,
) -> bytes:
    """
    Ray task that recreates the DataFrame & kwargs, produces the JPEG,
    and returns the raw binary buffer to the caller.
    """
    import io, pickle
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")                     # head‑less backend
    import matplotlib.pyplot as plt           # noqa: E402

    emb_df: pd.DataFrame = pickle.loads(df_bytes)
    scatter_kwargs: dict = pickle.loads(kwargs_blob)

    x = emb_df[x_col].to_numpy(float)
    y = emb_df[y_col].to_numpy(float)

    fig, ax = plt.subplots()

    if group_col in emb_df.columns:
        groups = emb_df[group_col].astype(str).to_numpy()
        for g in np.unique(groups):
            mask = groups == g
            ax.scatter(
                x[mask], y[mask], label=g, marker=marker,
                **({"c": color} if color else {}), **scatter_kwargs
            )
        ax.legend(title=group_col)
    else:
        ax.scatter(x, y, marker=marker, c=color, **scatter_kwargs)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    
    if title is None:
        title = "Molecular embedding"
    
    ax.set_title(title)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="jpeg", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def scatter_embedding(
    emb_df,
    x_column_name: str,
    y_column_name: str,
    out_path: str | pathlib.Path = "scatter.jpeg",
    *,
    # ── NEW ────────────────────────────────────────────────────────────────
    backend: Literal["matplotlib", "seaborn"] = "matplotlib",
    # ── Shared aesthetics ─────────────────────────────────────────────────
    color: str | None = None,        # constant point colour
    palette: str | Sequence | None = None,   # seaborn palette, ignored by MPL
    marker: str = "o",
    size: float | None = None,       # constant marker size
    alpha: float = 1.0,
    dpi: int = 300,
    group_col: str = "Group",        # categorical hue by default
    title: str | None = None,
    # ── Matplotlib‑specific options ───────────────────────────────────────
    grid: bool = False,
    equal_aspect: bool = False,
    style_sheet: str | None = None,
    # ── Seaborn‑specific options ──────────────────────────────────────────
    style: str | None = None,        # "whitegrid", "ticks", …
    context: str | None = None,      # "paper", "talk", …
    hue: str | None = None,          # override group_col as hue
    size_var: str | None = None,     # map variable to point size
    legend: Literal["brief", "full", False] = "brief",
    reg_line: bool = False,          # add regression fit
    # ── Catch‑all user overrides (passed verbatim) ────────────────────────
    **kwargs,
) -> pathlib.Path:
    """
    Render a 2‑D embedding to JPEG via Matplotlib **or** Seaborn.

    Parameters
    ----------
    backend : {"matplotlib","seaborn"}, default "matplotlib"
        Selects the plotting library.
    color / palette / marker / size / alpha
        Basic point aesthetics.  `palette` is only relevant when `backend="seaborn"`.
    grid, equal_aspect, style_sheet
        Additional Matplotlib‑only options.
    style, context, hue, size_var, legend, reg_line
        Additional Seaborn‑only options.
    **kwargs
        Forwarded to the underlying `plt.scatter` (Matplotlib) or
        `sns.scatterplot` (Seaborn).  This lets power‑users pass *any*
        backend‑specific keyword without the wrapper needing to know it.
    """
    # ------------------------------------------------------------------ 0
    # Normalise input → pandas.DataFrame
    # ------------------------------------------------------------------
    if isinstance(emb_df, pd.DataFrame):
        df = emb_df.copy()
    elif isinstance(emb_df, Table):             # your project’s Table
        df = emb_df.to_pandas().copy()
    elif isinstance(emb_df, ray.data.Dataset):
        df = emb_df.to_pandas().copy()
    else:
        raise TypeError("emb_df must be a pandas DataFrame, a Table, or a Ray Dataset")
    df.attrs.clear()    # Ray sometimes stores helpers in .attrs

    # ------------------------------------------------------------------ 1
    # Serialise & launch remote plotting task
    # ------------------------------------------------------------------
    _ensure_ray()       # your existing helper

    payload = {
        "df_bytes"    : pickle.dumps(df, protocol=pickle.HIGHEST_PROTOCOL),
        "x"           : x_column_name,
        "y"           : y_column_name,
        "backend"     : backend,
        # shared
        "color"       : color,
        "palette"     : palette,
        "marker"      : marker,
        "size"        : size,
        "alpha"       : alpha,
        "dpi"         : dpi,
        "group_col"   : group_col,
        "title"       : title,
        # mpl‑only
        "grid"        : grid,
        "equal_aspect": equal_aspect,
        "style_sheet" : style_sheet,
        # sns‑only
        "style"       : style,
        "context"     : context,
        "hue"         : hue,
        "size_var"    : size_var,
        "legend"      : legend,
        "reg_line"    : reg_line,
        # catch‑all
        "kwargs_blob" : pickle.dumps(kwargs, protocol=pickle.HIGHEST_PROTOCOL),
    }

    jpeg_bytes = ray.get(_scatter_plot_worker.remote(payload))

    # ------------------------------------------------------------------ 2
    # Persist the JPEG on the driver
    # ------------------------------------------------------------------
    out_path = pathlib.Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(jpeg_bytes)

    return out_path


###############################################################################
#  Ray remote worker
###############################################################################
@ray.remote
def _scatter_plot_worker(payload: dict) -> bytes:
    """Draw the figure inside a Ray worker and return it as raw JPEG bytes."""
    import io, pickle, matplotlib.pyplot as plt

    # ------------------------------------------------------------------ A
    # Unpack serialised payload
    # ------------------------------------------------------------------
    df         = pickle.loads(payload["df_bytes"])
    backend    = payload["backend"]
    x, y       = payload["x"], payload["y"]
    kwargs     = pickle.loads(payload["kwargs_blob"])

    # ------------------------------------------------------------------ B
    # Delegate to the appropriate drawing helper
    # ------------------------------------------------------------------
    if backend == "matplotlib":
        _draw_with_matplotlib(df, payload, kwargs)
    elif backend == "seaborn":
        _draw_with_seaborn(df, payload, kwargs)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    # ------------------------------------------------------------------ C
    # Serialise figure to JPEG and return
    # ------------------------------------------------------------------
    buf = io.BytesIO()
    plt.savefig(buf, format="jpeg", dpi=payload["dpi"], bbox_inches="tight")
    plt.close()
    return buf.getvalue()


###############################################################################
#  Helper: Matplotlib
###############################################################################
def _draw_with_matplotlib(df, p, kwargs):
    import matplotlib.pyplot as plt
    if p["style_sheet"]:
        plt.style.use(p["style_sheet"])
    fig, ax = plt.subplots()
    scatter = ax.scatter(
        df[p["x"]], df[p["y"]],
        c=df[p["group_col"]] if p["color"] is None else p["color"],
        marker=p["marker"],
        s=p["size"],
        alpha=p["alpha"],
        **kwargs,
    )
    if p["grid"]:
        ax.grid(True, zorder=0)
    if p["equal_aspect"]:
        ax.set_aspect("equal", adjustable="datalim")
    if p["title"]:
        ax.set_title(p["title"])
    # Matplotlib auto‑legend only if group colours used
    if p["color"] is None:
        ax.legend(*scatter.legend_elements(title=p["group_col"]),
                  title=p["group_col"], loc="best")


###############################################################################
#  Helper: Seaborn
###############################################################################
def _draw_with_seaborn(df, p, kwargs):
    import seaborn as sns, matplotlib.pyplot as plt
    # Optional global theme tweaks
    if p["style"]   is not None: sns.set_style(p["style"])
    if p["context"] is not None: sns.set_context(p["context"])

    ax = sns.scatterplot(
        data=df,
        x=p["x"],
        y=p["y"],
        hue=p["hue"] or p["group_col"],
        palette=p["palette"],
        marker=p["marker"],
        size=p["size_var"],
        sizes=(p["size"], p["size"]) if p["size"] else None,
        alpha=p["alpha"],
        legend=p["legend"],
        **kwargs,
    )
    if p["reg_line"]:
        # draw regression on the same axes
        sns.regplot(
            data=df,
            x=p["x"],
            y=p["y"],
            scatter=False,
            ax=ax,
            color="black",
            truncate=False,
        )
    if p["title"]:
        ax.set_title(p["title"])

###############################################################################
# Wrapper that Embeds Molecules in a Learned Space
###############################################################################

def embed_from_paths(
    datasets: Union[str, pathlib.Path, "Table", Sequence[Union[str, pathlib.Path, "Table"]]],
    *,
    dataset_names: Optional[Sequence[Optional[str]]] = None,     # ← NEW
    distance_metric: Union[str, Callable[[str, str], float]] = "tanimoto",
    embedding_method: Union[str, Callable] = "umap",
    chunk_size: int = 4_096,
    out_path: str = None,
    scatter_kwargs: Optional[Dict[str, Any]] = None,
    embed_kwargs: Optional[Dict[str, Any]] = None,
    n_subset = None
) -> pd.DataFrame:
    """
    Convenience wrapper:
      • load each file, tag rows with a readable name -> `Group`
      • compute global pair‑wise distances
      • embed & scatter

    Parameters
    ----------
    datasets : str | pathlib.Path | Table | iterable of those
        The data sources to embed.
    dataset_names : sequence[str | None], optional
        Custom display names, one per dataset.  If provided, its length
        **must** match the number of datasets.  Use ``None`` to fall back
        on the automatic name for a particular dataset.
    distance_metric, embedding_method, chunk_size, out_path, scatter_kwargs,
    embed_kwargs : see original signature.

    Returns
    -------
    pd.DataFrame
        Embedding with columns  SMILES  Dim1  Dim2  Group
    """
    _ensure_ray()

    scatter_kwargs = scatter_kwargs or {}
    embed_kwargs   = embed_kwargs   or {}

    # ------------------------------------------------------------
    # 1. Normalise input → list[ Table | PathLike | str ]
    # ------------------------------------------------------------
    if isinstance(datasets, (str, pathlib.Path)) or not isinstance(datasets, Sequence):
        datasets = [datasets]

    # ------------------------------------------------------------
    # 1a. Normalise / validate dataset_names
    # ------------------------------------------------------------
    if dataset_names is None:
        dataset_names = [None] * len(datasets)
    elif len(dataset_names) != len(datasets):
        raise ValueError(
            f"`dataset_names` has length {len(dataset_names)} but "
            f"{len(datasets)} datasets were supplied."
        )

    dfs, group_labels = [], []

    # ------------------------------------------------------------
    # 2. Load every dataset and assign a group name
    # ------------------------------------------------------------
    for idx, (src, custom_name) in enumerate(zip(datasets, dataset_names)):
        tbl = src if isinstance(src, Table) else to_table(src)
        
        tbl = tbl.filter_rowwise_using(
            lambda r: isinstance(r["SMILES"], str) and r["SMILES"].strip().lower() not in {"", "nan"}
        )

        if n_subset is not None:
            tbl = tbl.sample(n=n_subset)

        # Priority: user‑supplied name → automatic derivation
        if custom_name not in (None, ""):
            group = custom_name
        elif isinstance(src, (str, pathlib.Path)):
            group = pathlib.Path(src).stem                 # file stem
        else:
            group = getattr(tbl, "name", None) or f"set_{idx}"

        # materialise to pandas and tag rows
        df = tbl.to_pandas().copy()
        df["Group"] = group

        dfs.append(df)
        group_labels.extend([group] * len(df))

    # ------------------------------------------------------------
    # 3. Concatenate & compute distances
    # ------------------------------------------------------------
    merged_df = pd.concat(dfs, ignore_index=True)
    dist_tbl  = pairwise_distances(
        Table.from_df(merged_df, smiles_col="SMILES"),
        metric     = distance_metric,
        chunk_size = chunk_size,
    )

    # ------------------------------------------------------------
    # 4. Embed
    # ------------------------------------------------------------
    embed_df = embed_using_pairwise_distances(
        dist_tbl,
        embedding_method = embedding_method,
        groups           = np.asarray(group_labels),
        **embed_kwargs,
    )

    # ------------------------------------------------------------
    # 5. Scatter plot
    # ------------------------------------------------------------
    scatter_embedding(embed_df, "Dim1", "Dim2", out_path=out_path, **scatter_kwargs)

    return embed_df

###############################################################################
# Calculate Pareto Distances:
###############################################################################

###############################################################################
# Ray helper:  one‑shot Pareto computation inside its own runtime env
###############################################################################
import ray

_PARETO_RUNTIME_ENV = {
    "pip": [
        # versions are optional – pin if you need reproducibility
        "nds",                 # Buzdalov–Shalyto sorter
        "py-paretoarchive",    # Incremental BSP sorter
        "numpy",
        "scipy",
    ]
}

@ray.remote(runtime_env=_PARETO_RUNTIME_ENV)
def _pareto_distance_worker(
    pdf,                       # pandas.DataFrame
    objective_cols,
    pareto_col,
    method,
):
    """
    Identical to the *old* body of `compute_pareto_distance`, but written as
    a free function so Ray can pickle it.  Works entirely on the in‑memory
    pandas frame that the driver ships over the wire.
    """
    import numpy as np
    from scipy.spatial.distance import cdist

    obj_matrix = pdf[list(objective_cols)].to_numpy(float)
    points = -obj_matrix

    method_key = method.lower().replace(" ", "")

    if method_key.startswith("buzdalov"):
        from nds import ndomsort
        fronts = ndomsort.non_domin_sort(points, only_front_indices=True)
        frontier_idx = np.where(np.asarray(fronts) == 0)[0]

    elif method_key.startswith("incremental"):
        from paretoarchive import PyBspTreeArchive
        from scipy.spatial import cKDTree

        M = points.shape[1]
        archive = PyBspTreeArchive(numObjectives=M, maxSize=len(points))
        for p in points:
            archive.add(p)

        frontier_pts = np.asarray(archive.solutions)
        kd = cKDTree(points)
        _, frontier_idx = kd.query(frontier_pts, k=1)
        frontier_idx = np.unique(frontier_idx)

    else:
        raise ValueError(
            "Unknown method. Choose 'Buzdalov–Shalyto' or 'Incremental BSP'."
        )

    frontier_pts = obj_matrix[frontier_idx]
    dists = cdist(obj_matrix, frontier_pts, metric="euclidean").min(axis=1)
    pdf[pareto_col] = dists
    return pdf


def compute_pareto_distance(
    tbl: Table,
    objective_cols: Iterable[str],
    pareto_distance_column_name: str = "ParetoDistance",
    *,
    method: str = "Buzdalov–Shalyto",
) -> Table:
    """
    As before – but the actual work is delegated to `_pareto_distance_worker`.
    """
    # ------------------------------------------------------------------
    # 0.  Collect the data into a pandas frame once (driver side)
    #     This is required anyway because the algorithm needs the
    #     *global* objective matrix to identify the frontier.
    # ------------------------------------------------------------------
    df = tbl.to_pandas().copy()

    # ------------------------------------------------------------------
    # 1.  Launch a single Ray task in the dedicated runtime env
    # ------------------------------------------------------------------
    _ensure_ray()                                     # idempotent initialiser
    remote_ref = _pareto_distance_worker.remote(
        df,
        list(objective_cols),
        pareto_distance_column_name,
        method,
    )
    df_enriched = ray.get(remote_ref)                 # ← blocks until done

    # ------------------------------------------------------------------
    # 2.  Push results back into the original Table container
    # ------------------------------------------------------------------
    if tbl.is_ray():
        tbl.table = ray.data.from_pandas(df_enriched)
    else:
        tbl.table = df_enriched

    return tbl


###############################################################################
# Produces Powerpoint Files with SMILES Visualized:
###############################################################################

def smiles_to_pptx(
    table: Table,
    out_path: Union[str, pathlib.Path] = "smiles_slides.pptx",
    *,
    smiles_column: str = "SMILES",
    title: str = "SMILES catalogue",
    img_size: tuple[int, int] = (300, 300),        # pixels (w, h)
    columns_to_put_in_table: list[str] | None = None,
):
    """
    Create a PowerPoint deck with one slide per molecule showing

        • RDKit 2‑D structure (left)
        • SMILES string (right)
        • OPTIONAL property table (under the SMILES)

    Parameters
    ----------
    table : Table
        Your project‐level Table object.
    out_path : str | Path, default "smiles_slides.pptx"
        Destination file.
    smiles_column : str, default "SMILES"
        Column that holds the SMILES strings.
    columns_to_put_in_table : list[str] | None
        If given, these columns (must exist in *table*) are rendered as a
        2‑column table ‑- property name and value ‑- beneath the SMILES.
        Pass an empty list or None to skip the table.

    Requires
    --------
        pip install python-pptx rdkit-pypi pillow
    """
    # ── lazy imports ──────────────────────────────────────────────────
    from io import BytesIO
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from rdkit import Chem
    from rdkit.Chem import Draw
    from PIL import Image, ImageDraw

    # ------------------------------------------------------------------
    # 0.  Normalise data
    # ------------------------------------------------------------------
    df = table.table.to_pandas() if table.is_ray() else table.table
    if smiles_column not in df.columns:
        raise KeyError(f"Column '{smiles_column}' not found.")

    if columns_to_put_in_table:
        missing = [c for c in columns_to_put_in_table if c not in df.columns]
        if missing:
            raise KeyError(f"Columns not found in table: {missing}")

    # ------------------------------------------------------------------
    # 1.  Presentation scaffold
    # ------------------------------------------------------------------
    prs = Presentation()
    blank_layout = prs.slide_layouts[6]          # empty slide

    title_slide = prs.slides.add_slide(prs.slide_layouts[0])
    title_slide.shapes.title.text = title

    # Coordinates / sizes ------------------------------------------------
    LEFT_IMG   = Inches(0.5)
    TOP_IMG    = Inches(1.3)

    LEFT_TEXT  = Inches(5.0)
    TOP_TEXT   = TOP_IMG
    TEXT_W     = Inches(4.5)
    TEXT_H     = Inches(1.0)

    TOP_TABLE  = TOP_TEXT + TEXT_H + Inches(0.2)      # a bit below SMILES
    TABLE_W    = TEXT_W
    ROW_H      = Inches(0.4)                          # height per row

    # ------------------------------------------------------------------
    # 2.  Iterate rows
    # ------------------------------------------------------------------
    for _, row in df.iterrows():
        smi = str(row[smiles_column])

        slide = prs.slides.add_slide(blank_layout)

        # ---- image ----------------------------------------------------
        mol = Chem.MolFromSmiles(smi)
        if mol:
            img = Draw.MolToImage(mol, size=img_size, kekulize=True)
        else:
            img = Image.new("RGB", img_size, (255, 255, 255))
            d = ImageDraw.Draw(img)
            d.text((10, img_size[1] // 2 - 10), "Invalid SMILES", fill=(0, 0, 0))

        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        slide.shapes.add_picture(
            buf,
            LEFT_IMG,
            TOP_IMG,
            width=Inches(img_size[0] / 96),    # px ➜ in (96 dpi default)
            height=Inches(img_size[1] / 96),
        )

        # ---- SMILES text ---------------------------------------------
        txt_box = slide.shapes.add_textbox(LEFT_TEXT, TOP_TEXT, TEXT_W, TEXT_H)
        tf = txt_box.text_frame
        run = tf.paragraphs[0].add_run()
        run.text = smi
        run.font.size = Pt(20)

        # ---- optional property table ---------------------------------
        if columns_to_put_in_table:
            n_rows = len(columns_to_put_in_table)
            n_cols = 2  # property | value

            tbl_shape = slide.shapes.add_table(
                rows=n_rows,
                cols=n_cols,
                left=LEFT_TEXT,
                top=TOP_TABLE,
                width=TABLE_W,
                height=ROW_H * n_rows,
            )
            tbl = tbl_shape.table

            # column widths (roughly half‑half)
            tbl.columns[0].width = int(TABLE_W / 2)
            tbl.columns[1].width = int(TABLE_W / 2)

            for r, col_name in enumerate(columns_to_put_in_table):
                # property name
                cell_prop = tbl.cell(r, 0)
                cell_prop.text = str(col_name)

                # value (convert NaN to "")
                val = row[col_name]
                cell_val = tbl.cell(r, 1)
                cell_val.text = "" if (val != val) else str(val)

    # ------------------------------------------------------------------
    # 3.  Save
    # ------------------------------------------------------------------
    prs.save(str(pathlib.Path(out_path).expanduser().resolve()))

###############################################################################
# Area Where I Am Testing New Code:
###############################################################################


ecbd_tox_data = "/Users/asselism/Desktop/Collins_Lab/Datasets/Organized_by_Data_Type/ADMET_Data/Organized_by_Data_Type/Toxicity/Organized_by_Source/ECBD/ECBD_HepG2_100K/Original_Datasets/ECBD_HepG2_STL_tanimoto.csv"
hundred_k_tox_data = "/Users/asselism/Desktop/Collins_Lab/Datasets/Organized_by_Data_Type/ADMET_Data/Organized_by_Data_Type/Toxicity/Organized_by_Source/Collins_Lab/100K_Toxicity_Screen/100K_Toxicity_Data_with_MM_FPs_Removed_Unfeaturized_Mols.tsv"
seventy_k_tox_data = "/Users/asselism/Desktop/Collins_Lab/Datasets/Organized_by_Data_Type/ADMET_Data/Organized_by_Data_Type/Toxicity/Organized_by_Source/Collins_Lab/70K_Toxicity_Screen/70K_Scored_and_With_MM_FPs.tsv"
tox21_data = "/Users/asselism/Desktop/Collins_Lab/Datasets/Organized_by_Source/Tox21/Unzipped_Files/Modified_Datasets/_tox21-rt-viability-hek293-p1.aggregrated.tsv"
commercially_available_compounds = "/Users/asselism/Desktop/Collins_Lab/Datasets/Organized_by_Data_Type/Molecular_Structures/Molecules/Organized_by_Compound_Category/Commercially_Available/14M_deduplicated_combined.tsv"
eight_k_pseudomonas_screening_results = "/Users/asselism/Desktop/Collins_Lab/Datasets/Organized_by_Data_Type/Bioactivity/Organized_by_Data_Type/Antimicrobial_Activity/Organized_by_Source/Internal_Collins_Lab_Screens/All_PA_Screen_Results/8K-screen-PA_del6_del3_del40_WT.csv"

# training_set_dir = "/Users/asselism/Desktop/Collins_Lab/Datasets/Organized_by_Data_Type/Bioactivity/Organized_by_Data_Type/Antimicrobial_Activity/Organized_by_Source/COADD/Modified_Datasets/Datasets_with_Binarized_Hits/Binarization_at_80_Percent_GI/"
# for file in list_files(training_set_dir):
#     dataset = to_table(file)
#     dataset = add_minimol_fingerprints(dataset)
#     new_fpath = file.with_name(f"{file.stem}_with_Minimol_Representations.tsv")
#     dataset.to_tsv(new_fpath)

#path = "/Users/asselism/Desktop/Collins_Lab/Analysis/Organized_by_Project/Non-toxic_Antibiotics/Other_Analyses/Tox_Filters_Application_07212025/EC_KP_summary_with_IMR90.csv"

#ds = to_table(path)

#for filt, filt_name in zip(all_tox_filters, all_tox_filter_names):
    
#    ds.assign_rowwise_using(filt_name, lambda row, f=filt: int(f(row["SMILES"])))
    
#ds.assign_rowwise_using("any_pharma_filter", lambda row: int(any([row[name] for name in all_tox_filter_names])))
#ds.assign_rowwise_using("all_pharma_filter", lambda row: int(all([row[name] for name in all_tox_filter_names])))
    
#ds.to_csv("/Users/asselism/Desktop/Collins_Lab/Analysis/Organized_by_Project/Non-toxic_Antibiotics/Other_Analyses/Tox_Filters_Application_07212025/EC_KP_summary_with_IMR90_with_all_the_filters.csv")