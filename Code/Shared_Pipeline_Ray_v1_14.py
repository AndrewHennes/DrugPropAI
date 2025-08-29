#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ---------------------------------------------------------------------
# Unified scaffold for large-scale molecular visualisation & analysis
# using Ray for parallelism and runtime-environment isolation.
# ---------------------------------------------------------------------

###############################################################################
# Package Imports:
###############################################################################

from __future__ import annotations

#==============================================================================
# Built-In Packages:
#==============================================================================
import base64
from collections import defaultdict
import errno
from functools import cached_property, lru_cache, wraps
from copy import deepcopy

import logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

import inspect
from itertools import combinations
import math
import os
from os import PathLike
import pathlib
from pathlib import Path
import pickle
import re
import shutil
from threading import Lock
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Literal,
    Optional,
    overload,
    Sequence,
    TypeVar,
    Tuple,
    Union,
)
import unicodedata
import warnings

#==============================================================================
# Third-Party Packages:
#==============================================================================

from cachetools import cached, LRUCache

import cloudpickle  # cloud-compatible pickle implementation

import dask.dataframe as dd

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

#import pyyaml as yaml

import pyarrow as pa
from pyarrow import csv as pa_csv

import ray
from ray.air.util.tensor_extensions.arrow import ArrowConversionError
from ray.data import ActorPoolStrategy, Dataset as _RayDataset
from ray.data.exceptions import SystemException        # Ray ≥ 2.9

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, Mol, QED, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger

# ########################################################################### #
# Global State Declaration and Implementation:
# ########################################################################### #

# =========================================================================== #
# Global State Variable Definition:
# =========================================================================== #

disable_rdkit_error_messages = False
PANDAS_PROMOTE_THRESHOLD = "5_000_000"
DASK_FROM_PANDAS_CHUNKSIZE = "1_000_000"
MAX_CACHE_SIZE = "50_000"

# =========================================================================== #
# Global State Variable Execution:
# =========================================================================== #

standardize_int = lambda number: int(number.replace("_", ""))

if disable_rdkit_error_messages:
    RDLogger.DisableLog("rdApp.*")

PANDAS_PROMOTE_THRESHOLD = standardize_int(PANDAS_PROMOTE_THRESHOLD)
DASK_FROM_PANDAS_CHUNKSIZE = standardize_int(DASK_FROM_PANDAS_CHUNKSIZE)
MAX_CACHE_SIZE = standardize_int(MAX_CACHE_SIZE)

# ########################################################################### #
# Registry-Related Methods and Objects:
# ########################################################################### #

# --------------------------------------------------------------------------- #
# Registry Initialization:
# --------------------------------------------------------------------------- #

_REGISTRIES: Dict[str, Dict[str, Any]] = {
    # kept for backward compatibility
    "utility": {},
    "filter": {},
    "prop": {},

    # actively used throughout this module
    "descriptors": {},
    "smarts": {},
    "alerts": {},
    "rules": {},
    "scores": {},
    "scaffold": {},
    "fingerprint": {},
    "similarity": {},
    "embed": {},
}

# --------------------------------------------------------------------------- #
# Basic Registry Methods:
# --------------------------------------------------------------------------- #

def register(kind: str, name: str):
    
    def decorator(cls_or_fn):
        _REGISTRIES[kind][name] = cls_or_fn
        return cls_or_fn
    
    return decorator


def get(kind: str, name: str):
    
    if name not in _REGISTRIES[kind]:
        raise KeyError(f"{name} not found in {kind} registry")
        
    return _REGISTRIES[kind][name]

# --------------------------------------------------------------------------- #
# Shorthand Registry Lookup Aliases:
# --------------------------------------------------------------------------- #
get_utility     = lambda name: get("utility", name)
get_filter      = lambda name: get("filter", name)
get_prop        = lambda name: get("prop", name)

get_descriptors = lambda name: get("descriptors", name)
get_smarts      = lambda name: get("smarts", name)
get_alerts      = lambda name: get("alerts", name)
get_rules       = lambda name: get("rules", name)
get_scores      = lambda name: get("scores", name)
get_scaffold    = lambda name: get("scaffold", name)
get_fingerprint = lambda name: get("fingerprint", name)
get_similarity  = lambda name: get("similarity", name)
get_embedder    = lambda name: get("embed", name)   # returns the Ray-remote embedder

# ########################################################################### #
# Ray-Related Methods:
# ########################################################################### #

F = TypeVar("F", bound=Callable[..., Any])

# Default init options (merge-able at call time)
_DEFAULT_RAY_INIT: dict[str, Any] = {
    "ignore_reinit_error": True,
    "namespace": "chem-pipeline",
    "include_dashboard": False,
}

_ray_init_lock = Lock()

def _ensure_ray(**init_kwargs: Any) -> None:
    """
    Idempotent Ray init; safe and cheap to call repeatedly.
    """
    try:
        import ray  # already imported at top of file, but keep this robust
    except Exception as e:
        raise RuntimeError("Ray is required but not available") from e

    if ray.is_initialized():
        return

    with _ray_init_lock:
        if not ray.is_initialized():
            cfg = dict(_DEFAULT_RAY_INIT)
            cfg.update(init_kwargs)
            ray.init(**cfg)


def ensure_ray(_fn: Optional[F] = None, /, **init_kwargs: Any):
    """
    Decorator: ensure Ray is initialized on the driver before calling the function.

    Usage:
        @ensure_ray
        def f(...): ...

        @ensure_ray(namespace="chem-pipeline", include_dashboard=False)
        def g(...): ...

        class C:
            @ensure_ray
            def m(self, ...): ...

            @ensure_ray
            @classmethod
            def cm(cls, ...): ...

            @ensure_ray
            @staticmethod
            def sm(...): ...

        @ensure_ray  # async supported too
        async def af(...): ...
    """
    def _decorate(fn: F) -> F:
        # Handle classmethod/staticmethod by unwrapping, then re-wrapping.
        is_cm  = isinstance(fn, classmethod)
        is_sm  = isinstance(fn, staticmethod)
        target = fn.__func__ if (is_cm or is_sm) else fn  # type: ignore[attr-defined]

        if inspect.iscoroutinefunction(target):
            @wraps(target)
            async def async_wrapper(*args, **kwargs):
                _ensure_ray(**init_kwargs)
                return await target(*args, **kwargs)
            wrapped = async_wrapper
        else:
            @wraps(target)
            def wrapper(*args, **kwargs):
                _ensure_ray(**init_kwargs)
                return target(*args, **kwargs)
            wrapped = wrapper

        if is_cm:
            return classmethod(wrapped)  # type: ignore[return-value]
        if is_sm:
            return staticmethod(wrapped)  # type: ignore[return-value]
        return wrapped  # type: ignore[return-value]

    return _decorate if _fn is None else _decorate(_fn)

# ########################################################################### #
# File-Related Methods:
# ########################################################################### #

# =========================================================================== #
# Paths to Environments:
# =========================================================================== #

# TODO: Localize all of the environment paths.
path_to_minimol_environment = "/Users/asselism/Desktop/Collins_Lab/Environments/Environments_for_Pipeline/Minimol_Environment/minimol_env.yml"

# =========================================================================== #
# Paths to Datasets:
# =========================================================================== #

ecbd_tox_data = "/Users/asselism/Desktop/Collins_Lab/Datasets/Organized_by_Data_Type/ADMET_Data/Organized_by_Data_Type/Toxicity/Organized_by_Source/ECBD/ECBD_HepG2_100K/Original_Datasets/ECBD_HepG2_STL_tanimoto.csv"
hundred_k_tox_data = "/Users/asselism/Desktop/Collins_Lab/Datasets/Organized_by_Data_Type/ADMET_Data/Organized_by_Data_Type/Toxicity/Organized_by_Source/Collins_Lab/100K_Toxicity_Screen/100K_Toxicity_Data_with_MM_FPs_Removed_Unfeaturized_Mols.tsv"
seventy_k_tox_data = "/Users/asselism/Desktop/Collins_Lab/Datasets/Organized_by_Data_Type/ADMET_Data/Organized_by_Data_Type/Toxicity/Organized_by_Source/Collins_Lab/70K_Toxicity_Screen/70K_Scored_and_With_MM_FPs.tsv"
tox21_data = "/Users/asselism/Desktop/Collins_Lab/Datasets/Organized_by_Source/Tox21/Unzipped_Files/Modified_Datasets/_tox21-rt-viability-hek293-p1.aggregrated.tsv"
commercially_available_compounds = "/Users/asselism/Desktop/Collins_Lab/Datasets/Organized_by_Data_Type/Molecular_Structures/Molecules/Organized_by_Compound_Category/Commercially_Available/14M_deduplicated_combined.tsv"
eight_k_pseudomonas_screening_results = "/Users/asselism/Desktop/Collins_Lab/Datasets/Organized_by_Data_Type/Bioactivity/Organized_by_Data_Type/Antimicrobial_Activity/Organized_by_Source/Internal_Collins_Lab_Screens/All_PA_Screen_Results/8K-screen-PA_del6_del3_del40_WT.csv"

###############################################################################
# Commonly Referenced Concepts:
###############################################################################

# =========================================================================== #
# Biology:
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Bacterial Species Names:
# --------------------------------------------------------------------------- #

names_for_pseudomonas_aeruginosa = (
    "PA",
    "pseudomonas_aeruginosa",
    "Pseudomonas_aeruginosa",
    "Pseudomonas aeruginosa",
)

names_for_klebsiella_pneumoniae = (
    "KP",
    "klebsiella_pneumoniae",
    "Klebsiella_pneumoniae",
    "Klebsiella pneumoniae",
)

names_for_escherichia_coli = (
    "EC",
    "escherichia_coli",
    "Escherichia_coli",
    "Escherichia coli",
)

names_for_acinetobacter_baumannii = (
    "AB",
    "acinetobacter_baumannii",
    "Acinetobacter_baumannii",
    "Acinetobacter baumannii",
)

# =========================================================================== #
# Other:
# =========================================================================== #

# --------------------------------------------------------------------------- #
# File Extensions:
# --------------------------------------------------------------------------- #

names_for_csv_file_extensions = ("csv", ".csv")
names_for_tsv_file_extensions = ("tsv", ".tsv")

# ########################################################################### #
# Common Methods:
# ########################################################################### #

#==============================================================================
# Common Math Methods:
#==============================================================================

# --------------------------------------------------------------------------- #
# Mean Methods:
# --------------------------------------------------------------------------- #

def average(items):
    
    return sum([float(val) for val in items])/len(items)

# --------------------------------------------------------------------------- #
# Combinatorial Methods:
# --------------------------------------------------------------------------- #

def combo(items, n):
    
    for subset in combinations(items, n):
        yield subset
        
def all_pairs(items):
    
    for subset in combo(items, 2):
        yield subset
        
# =========================================================================== #
# Simple File and Directory-Related Methods:
# =========================================================================== #
        
# --------------------------------------------------------------------------- #
# Methods on Paths:
# --------------------------------------------------------------------------- #

def standardize_path(
    p: str | bytes | Path | PathLike, *,
    expand_vars: bool = True,
    resolve_symlinks: bool = False,
    strict: bool = False,
    normalize_unicode: bool = False,
    normalize_case: bool | None = None,
) -> Path:
    """
    Convert `p` (string/bytes/Path/os.PathLike) into a pathlib.Path with:
      • user expansion (~ or ~user)
      • optional env var expansion ($HOME, %USERPROFILE%)
      • absolute form (no relative segments)
      • optional symlink resolution

    Parameters
    ----------
    p
        Any path-like object (e.g., str, bytes, pathlib.Path, os.PathLike).
    expand_vars : bool, default True
        Expand environment variables in the string form before creating a Path.
    resolve_symlinks : bool, default False
        If True, dereference symlinks using Path.resolve().
    strict : bool, default False
        Only used when `resolve_symlinks=True`. If True, raise if the path
        (or any parent) does not exist.
    normalize_unicode : bool, default False
        If True, normalize the path string to NFC (useful on macOS where
        filenames may be NFD).
    normalize_case : bool | None, default None
        If True, case-normalize the final path string (os.path.normcase).
        If None, it applies on Windows automatically and is a no-op elsewhere.

    Returns
    -------
    pathlib.Path
        The standardized absolute path.
    """
    # Convert “path-like” → str (handles bytes + os.PathLike)
    s = os.fsdecode(os.fspath(p))

    if expand_vars:
        s = os.path.expandvars(s)

    if normalize_unicode:
        s = unicodedata.normalize("NFC", s)

    path = Path(s).expanduser()

    if resolve_symlinks:
        # Resolve symlinks (optionally strict)
        path = path.resolve(strict=strict)
    else:
        # Absolute without dereferencing symlinks
        path = path.absolute()

    # Case-normalize where appropriate
    if normalize_case is True or (normalize_case is None and os.name == "nt"):
        path = Path(os.path.normcase(str(path)))

    return path

def keep_files(files):
    
    return (p for p in files if p.is_file())

def is_ds_store(fname):
    
    return fname.name in (".DS_Store", "DS_Store")

# --------------------------------------------------------------------------- #
# Modifying Paths:
# --------------------------------------------------------------------------- #

P = TypeVar("P", bound=pathlib.PurePath)

def modify_filename(
    path: str | P,
    fn: Callable[[str], str],
) -> str | P:
    
    path = Path(path)
    parent = path.parent
    name = path.name
    suffixes = path.suffixes
    name_without_ext = path.name[:-len("".join(suffixes))]
    name_without_ext = fn(name_without_ext)
    new_name = name_without_ext + suffixes
    return parent / new_name

# --------------------------------------------------------------------------- #
# Enumerating Files and Directories:
# --------------------------------------------------------------------------- #

def list_files(directory: str | Path,
               recursive: bool = False,
               *,
               absolute: bool = True,
               ignore_ds_store: bool = True) -> List[Path]:
    
    directory = standardize_path(directory)
    
    pattern = '**/*' if recursive else '*'
    files = directory.glob(pattern)
    files = keep_files(files)
    
    # Expand paths relative to directory if not absolute paths.
    if not absolute:
        files = (p.relative_to(directory) for p in files if p.is_file())
    
    # Remove .DS_Store file if present.
    if ignore_ds_store:
        files = (p for p in files if not is_ds_store(p))

    return list(files)

    
def list_dirs(directory: Union[str, Path], recursive: bool = False) -> List[Path]:
    """
    Return a list of **directory paths** contained in *directory*.

    Parameters
    ----------
    directory : str | pathlib.Path
        Path to the directory you want to inspect.
    recursive : bool, default False
        • False → only immediate sub-directories  
        • True  → include sub-directories at *all* nested levels

    Returns
    -------
    List[pathlib.Path]
        Absolute paths for every directory found.
    """
    directory = standardize_path(directory)

    if recursive:
        # Path.rglob('*') yields everything; filter for directories
        return [p for p in directory.rglob('*') if p.is_dir()]
    else:
        # Path.iterdir() lists immediate children
        return [p for p in directory.iterdir() if p.is_dir()]

# --------------------------------------------------------------------------- #
# Data-Type Conversions:
# --------------------------------------------------------------------------- #

def file_extension_to_delimiter(extension):
    
    if extension in names_for_csv_file_extensions:
        return ","
    
    elif extension in names_for_tsv_file_extensions:
        return "\t"
    
    else:
        raise NotImplementedError()
        
def broadcast_val_to_list(x, n):
    
    from collections.abc import Sequence as _Seq
    if isinstance(x, (str, bytes)) or not isinstance(x, _Seq):
        return [x] * n
    return list(x)

# =========================================================================== #
# Data Type Transformation:
# =========================================================================== #

# def load_yaml(path: str | Path) -> dict:
#     """
#     Load a YAML file into a Python dictionary (or list, depending on the root).

#     Parameters
#     ----------
#     path : str | Path
#         Path to the .yaml file.

#     Returns
#     -------
#     dict | list
#         Parsed YAML content.
#     """
#     Loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
#     with open(path, "r") as f:
#         return yaml.load(f, Loader=Loader)

# =========================================================================== #
# OS-Like Methods:
# =========================================================================== #

# TODO: Check mv method.

def mv(
    src: Union[str, Path],
    dst: Union[str, Path],
    *,
    no_clobber: bool = False,           # like `mv -n`: fail if destination exists
    update: bool = False,               # like `mv -u`: only move if src is newer than dst
    target_is_directory: bool | None = None,  # None = auto (shutil.move); False ~ `-T`
) -> Path:
    """
    Move file or directory in a way similar to Unix `mv`.

    Behavior:
      - If target_is_directory is None: behave like shutil.move (if dst is a dir, put src inside).
      - If True: force treating dst as a directory (error if it isn't).
      - If False: treat dst as a file path (like `mv -T`; error if dst is a directory).

    Returns the resolved destination Path.
    """
    src = standardize_path(src)
    dst = standardize_path(dst)

    if target_is_directory is True:
        if not dst.exists() or not dst.is_dir():
            raise NotADirectoryError(f"Destination is not a directory: {dst}")
        dst_final = dst / src.name
    elif target_is_directory is False:  # mimic `mv -T`
        if dst.exists() and dst.is_dir():
            raise IsADirectoryError(f"Destination is a directory (use target_is_directory=True): {dst}")
        dst_final = dst
    else:
        # shutil.move semantics: if dst is an existing directory, move into it
        dst_final = (dst / src.name) if dst.is_dir() else dst

    if no_clobber and dst_final.exists():
        raise FileExistsError(f"Destination exists: {dst_final}")

    if update and dst_final.exists():
        try:
            if src.stat().st_mtime <= dst_final.stat().st_mtime:
                return dst_final
        except FileNotFoundError:
            # raced with another process; continue to move
            pass

    # Fast path: atomic same-FS replace when possible
    try:
        # Guard obvious type conflicts so os.replace doesn’t produce surprising errors
        if src.is_dir() and dst_final.exists() and dst_final.is_file():
            raise OSError(errno.EISDIR, "Destination is a file")
        if src.is_file() and dst_final.exists() and dst_final.is_dir():
            raise OSError(errno.EEXIST, "Destination is a directory")
        os.replace(src, dst_final)       # atomic within same filesystem
        return dst_final
    except OSError as e:
        # Cross-device (EXDEV) or type mismatch: fall back to shutil.move
        if getattr(e, "errno", None) not in (errno.EXDEV, errno.EISDIR, errno.EEXIST):
            # Many other errors will also be handled by shutil.move, so we still try it next.
            pass

    return Path(shutil.move(str(src), str(dst_final)))

# =========================================================================== #
# Reproducability Methods:
# =========================================================================== #

# ########################################################################### #
# Data Table Functionality:
# ########################################################################### #

# =========================================================================== #
# Checking Table Type:
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Table Class Definition:
# --------------------------------------------------------------------------- #


class Table:
    """Unified Pandas / Dask / Ray tabular wrapper."""

    # ====================================================================== #
    # Backend detectors
    # ====================================================================== #

    @staticmethod
    def _is_dask_dataframe(obj: Any) -> bool:
        return dd is not None and isinstance(obj, dd.DataFrame)  # type: ignore[arg-type]

    @staticmethod
    def _is_ray_dataset(obj: Any) -> bool:
        return ray is not None and _RayDataset is not None and isinstance(obj, _RayDataset)  # type: ignore[arg-type]

    # ====================================================================== #
    # Sort utilities
    # ====================================================================== #

    @staticmethod
    def _ray_sort(ds: "_RayDataset", key, ascending: bool, batch_size: int, **opts) -> "_RayDataset":
        if isinstance(key, str):
            return ds.sort(key=key, descending=not ascending)

        SORT_COL = "__sort_key__"

        def _with_key(batch: pd.DataFrame) -> pd.DataFrame:
            batch[SORT_COL] = batch.apply(key, axis=1)
            return batch

        return (
            ds.map_batches(_with_key, batch_size=batch_size, batch_format="pandas", **opts)
              .sort(key=SORT_COL, descending=not ascending)
              .drop_columns([SORT_COL])
        )

    @staticmethod
    def _dask_sort(ddf: "dd.DataFrame", key, ascending: bool):
        # Prefer the official dask sort_values implementation.
        if isinstance(key, str):
            return ddf.sort_values(by=key, ascending=ascending)
        # Callable key – compute a temp column per partition, then sort.
        tmp = ddf.map_partitions(lambda df: df.assign(__sort_key__=df.apply(key, axis=1)))
        sorted_ddf = tmp.sort_values("__sort_key__", ascending=ascending)
        return sorted_ddf.drop(columns="__sort_key__")

    @staticmethod
    def _pandas_sort(df: pd.DataFrame, key, ascending: bool):
        if isinstance(key, str):
            return df.sort_values(by=key, ascending=ascending, kind="mergesort")
        tmp = df.assign(__sort_key__=df.apply(key, axis=1))
        return tmp.sort_values("__sort_key__", ascending=ascending, kind="mergesort").drop(columns="__sort_key__")

    # ====================================================================== #
    # Mutation helper
    # ====================================================================== #

    @staticmethod
    def _maybe_update(tbl: "Table", new_obj, inplace: bool) -> "Table":
        self = tbl #mokeypatch
        if inplace:
            tbl.table = new_obj
            return tbl
        return self.__class__(new_obj, smiles_column_name=tbl.smiles_column_name)

    # ====================================================================== #
    # Row-wise helpers
    # ====================================================================== #

    @staticmethod
    def _ray_assign_rowwise(ds: "_RayDataset", col: str, func, batch_size: int, **opts):
        def _apply(batch: pd.DataFrame) -> pd.DataFrame:
            batch[col] = batch.apply(func, axis=1)
            return batch

        return ds.map_batches(_apply, batch_size=batch_size, batch_format="pandas", **opts)

    @staticmethod
    def _dask_assign_rowwise(ddf: "dd.DataFrame", col: str, func):
        return ddf.assign(**{col: ddf.map_partitions(lambda df: df.apply(func, axis=1))})

    @staticmethod
    def _pandas_assign_rowwise(df: pd.DataFrame, col: str, func):
        df = df.copy()
        df[col] = df.apply(func, axis=1)
        return df

    @staticmethod
    def _ray_filter(ds: "_RayDataset", func, batch_size: int, **opts):
        def _filter(batch: pd.DataFrame) -> pd.DataFrame:
            mask = batch.apply(func, axis=1)
            if mask.dtype != bool:
                raise ValueError("filter_rowwise_using: predicate must return booleans")
            return batch[mask]

        return ds.map_batches(_filter, batch_size=batch_size, batch_format="pandas", **opts)

    # ====================================================================== #
    # Export helpers
    # ====================================================================== #

    @staticmethod
    def _ray_export(ds, path, fmt: str, sep: str | None, **kwargs):
        import pathlib as _pathlib  # local to keep import optionality tight
        import shutil, tempfile, os
        import pandas as _pd
        import pyarrow as pa
        import pyarrow.types as pat

        path = _pathlib.Path(path).expanduser().resolve()

        schema = ds.schema()
        names = list(getattr(schema, "names", []))
        arrow_types = list(getattr(schema, "types", []))

        def _is_nested_arrow_type(t) -> bool:
            if not isinstance(t, pa.DataType):
                return False
            return (
                pat.is_list(t) or pat.is_large_list(t) or pat.is_fixed_size_list(t)
                or pat.is_struct(t) or pat.is_map(t)
            )

        nested_cols = [n for n, t in zip(names, arrow_types) if _is_nested_arrow_type(t)]

        if fmt in {"csv", "tsv"}:
            try:
                sample = next(iter(ds.iter_batches(batch_size=32, batch_format="pandas")), None)
            except StopIteration:
                sample = None
            if sample is not None:
                import numpy as np  # noqa: F401

                def _looks_nested_py(v):
                    return (
                        v is not None
                        and not isinstance(v, (str, bytes))
                        and (isinstance(v, (list, tuple, dict)) or hasattr(v, "tolist"))
                    )

                for c in sample.columns:
                    if c in nested_cols:
                        continue
                    col = sample[c].dropna()
                    if not col.empty and _looks_nested_py(col.iloc[0]):
                        nested_cols.append(c)

        if nested_cols and fmt in {"csv", "tsv"}:
            import json

            def _cell_to_json(x):
                if x is None:
                    return None
                try:
                    if hasattr(x, "tolist"):
                        return json.dumps(x.tolist())
                    if isinstance(x, (list, tuple, dict)):
                        return json.dumps(x)
                    return json.dumps(x)
                except Exception:
                    return str(x)

            def _stringify(batch: _pd.DataFrame) -> _pd.DataFrame:
                for c in nested_cols:
                    if c in batch.columns:
                        batch[c] = batch[c].apply(_cell_to_json)
                return batch

            ds = ds.map_batches(_stringify, batch_format="pandas")

        if fmt == "parquet":
            ds.write_parquet(str(path), **kwargs)
            return

        if fmt == "csv":
            ds.write_csv(str(path), **kwargs)
            return

        if fmt == "tsv":
            tmp_dir = tempfile.mkdtemp(prefix="ray_csv_")
            try:
                ds.write_csv(tmp_dir, **kwargs)
                with open(str(path), "w", newline="") as fout:
                    writer_started = False
                    for f in sorted(os.listdir(tmp_dir)):
                        fpath = os.path.join(tmp_dir, f)
                        for chunk in _pd.read_csv(fpath, chunksize=1_000_000):
                            chunk.to_csv(
                                fout,
                                sep="\t",
                                index=False,
                                header=(not writer_started),
                                lineterminator="\n",
                            )
                            writer_started = True
            finally:
                shutil.rmtree(tmp_dir)
            return

        raise ValueError(f"Unsupported format: {fmt}")

    @staticmethod
    def _dask_export(ddf: "dd.DataFrame", path: pathlib.Path, fmt: str,
                     sep: str | None, single_file: bool, **kwargs):
        if fmt == "parquet":
            ddf.to_parquet(path, write_index=False, **kwargs)
        elif fmt in {"csv", "tsv"}:
            delimiter = "," if fmt == "csv" else "\t"
            ext = ".csv" if fmt == "csv" else ".tsv"
            if single_file:
                ddf.to_csv(str(path), single_file=True, sep=delimiter, index=False, **kwargs)
            else:
                # directory of part files
                path.mkdir(parents=True, exist_ok=True)
                ddf.to_csv(str(path / f"*{ext}"), sep=delimiter, index=False, **kwargs)
        else:  # pragma: no cover
            raise ValueError(f"Unsupported format: {fmt!r}")

    @staticmethod
    def _pandas_export(df: pd.DataFrame, path: pathlib.Path, fmt: str,
                       sep: str | None, **kwargs):
        if fmt == "parquet":
            df.to_parquet(path, index=False, **kwargs)
        elif fmt in {"csv", "tsv"}:
            delimiter = "," if fmt == "csv" else "\t"
            df.to_csv(path, sep=delimiter, index=False, **kwargs)
        else:  # pragma: no cover
            raise ValueError(f"Unsupported format: {fmt!r}")

    # ====================================================================== #
    # Backend checks
    # ====================================================================== #

    def is_ray(self) -> bool:
        return self._is_ray_dataset(self.table)

    def is_dask(self) -> bool:
        return self._is_dask_dataframe(self.table)

    def is_pandas(self) -> bool:
        return isinstance(self.table, pd.DataFrame)

    # ====================================================================== #
    # Constructor
    # ====================================================================== #

    def __init__(
        self,
        table: Union[pd.DataFrame, "dd.DataFrame", "_RayDataset"],
        smiles_column_name: str = "SMILES",
        name: Union[str, None] = None,
        force_pandas: bool = False,
    ):
        # 1) Keep Ray if given
        if self._is_ray_dataset(table):
            self.table = table

        # 2) Keep Dask if given
        elif self._is_dask_dataframe(table):
            self.table = table

        # 3) Pandas → (optionally) Dask
        elif isinstance(table, pd.DataFrame):
            if not force_pandas and dd is not None and len(table) > PANDAS_PROMOTE_THRESHOLD:
                self.table = dd.from_pandas(table, chunksize=DASK_FROM_PANDAS_CHUNKSIZE)  # type: ignore[attr-defined]
            else:
                self.table = table

        else:  # pragma: no cover
            raise TypeError(f"Unsupported table type: {type(table)}")

        self.smiles_column_name = smiles_column_name
        self.table_name = name

    # ====================================================================== #
    # Properties / metadata
    # ====================================================================== #

    def has_name(self) -> str | None:
        return self.table_name is not None

    def get_name(self) -> str | None:
        return getattr(self, "table_name", None)

    @cached_property
    def _rowcount(self) -> int:
        if self.is_ray():
            return int(self.table.count())
        # Safe fallback if accidentally accessed
        if self.is_dask():
            return int(self.table.shape[0].compute())
        return len(self.table)

    # ====================================================================== #
    # Iteration & extraction
    # ====================================================================== #

    def iter_chunks(self, n: int = 1_000) -> Generator[pd.DataFrame, None, None]:
        """
        Yield batches of at most ``n`` rows as *Pandas* DataFrames.
        """
        # 1) Ray
        if self.is_ray():
            yield from self.table.iter_batches(batch_size=n, batch_format="pandas")
            return

        # 2) Dask
        if self.is_dask():
            for i in range(self.table.npartitions):
                pdf = self.table.get_partition(i).compute()
                for j in range(0, len(pdf), n):
                    yield pdf.iloc[j : j + n]
            return

        # 3) Pandas
        for i in range(0, len(self.table), n):
            yield self.table.iloc[i : i + n]

    # ====================================================================== #
    # Unique values
    # ====================================================================== #

    def unique(self, column_name: str) -> List[Any]:
        """
        Return **all** unique values in *column_name* as a Python list.
        """
        # Validate existence
        if self.is_ray():
            if column_name not in self.table.schema().names:
                raise KeyError(f"Column {column_name!r} not found")
        else:
            if column_name not in self.table.columns:
                raise KeyError(f"Column {column_name!r} not found")

        # Ray
        if self.is_ray():
            if hasattr(self.table, "unique"):
                ds = self.table.unique(column_name)
                return ds.to_pandas()[column_name].tolist()
            if hasattr(self.table, "select_columns") and hasattr(self.table, "distinct"):
                return (
                    self.table.select_columns(column_name)
                              .distinct()
                              .to_pandas()[column_name]
                              .tolist()
                )
            grouped = self.table.groupby(column_name).count()
            if hasattr(grouped, "drop_columns"):
                grouped = grouped.drop_columns(["count()"])
            return grouped.to_pandas()[column_name].tolist()

        # Dask
        if self.is_dask():
            return self.table[column_name].unique().compute().tolist()

        # Pandas
        return self.table[column_name].unique().tolist()

    # ====================================================================== #
    # Python protocol methods
    # ====================================================================== #

    def __getitem__(self, i: str | int):
        
        if isinstance(i, int):
            pass
            # Implement row indexing here.
        
        elif isinstance(i, str):
            if self.is_ray():
                return self.table.select_columns(i)
            if self.is_dask():
                return self.table[i]
            return self.table[i]

    def __len__(self) -> int:
        if self.is_ray():
            return self._rowcount
        if self.is_dask():
            return int(self.table.shape[0].compute())
        return len(self.table)
    
    def __iter__(self) -> Generator[Dict[str, Any], None, None]:
        """
        Iterate over the rows, yielding a dict mapping column name -> value
        for each row. Uses chunked iteration to avoid materializing the full
        dataset, and works across Ray / Dask / Pandas backends.
        """
        for pdf in self.iter_chunks():
            cols = list(pdf.columns)
            # Use itertuples for lower overhead vs. to_dict('records')
            for values in pdf.itertuples(index=False, name=None):
                # Convert the row tuple into a dict mapping col -> value.
                yield {c: v for c, v in zip(cols, values)}

    # ====================================================================== #
    # Mutations
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
        if self.is_ray():
            sorted_ds = self._ray_sort(self.table, key, ascending, batch_size, **ray_opts)
            return self._maybe_update(self, sorted_ds, inplace)

        if self.is_dask():
            sorted_df = self._dask_sort(self.table, key, ascending)
            return self._maybe_update(self, sorted_df, inplace)

        sorted_df = self._pandas_sort(self.table, key, ascending)
        return self._maybe_update(self, sorted_df, inplace)

    def filter_rowwise_using(
        self,
        func: Callable[[pd.Series], bool],
        *,
        inplace: bool = False,
        batch_size: int = 4_096,  # Ray only
        **ray_opts,
    ) -> "Table":
        if self.is_ray():
            new_ds = self._ray_filter(self.table, func, batch_size, **ray_opts)
            return self._maybe_update(self, new_ds, inplace)

        if self.is_dask():
            mask = self.table.map_partitions(lambda df: df.apply(func, axis=1))
            new_df = self.table[mask]
            return self._maybe_update(self, new_df, inplace)

        mask = self.table.apply(func, axis=1)
        new_df = self.table[mask]
        return self._maybe_update(self, new_df, inplace)

    def keep_columns(
        self,
        column_names: Iterable[str],
        in_place: bool = False,
    ) -> "Table":
        """
        Keep only the specified columns (strict: raises on miss).
        """
        seen = set()
        cols = [c for c in column_names if not (c in seen or seen.add(c))]

        existing = list(self.table.schema().names) if self.is_ray() else list(self.table.columns)
        missing = [c for c in cols if c not in existing]
        if missing:
            raise KeyError(f"keep_columns: columns not found: {missing}")

        if self.is_ray():
            if cols:
                new_obj = self.table.select_columns(cols)
            else:
                def _empty(batch: pd.DataFrame) -> pd.DataFrame:
                    return batch.iloc[:, 0:0]
                new_obj = self.table.map_batches(_empty, batch_format="pandas")
            return self._maybe_update(self, new_obj, inplace=in_place)

        if self.is_dask():
            return self._maybe_update(self, self.table[cols], inplace=in_place)

        return self._maybe_update(self, self.table.loc[:, cols].copy(), inplace=in_place)

    def drop_columns(
        self,
        column_names: Iterable[str],
        in_place: bool = False,
    ) -> "Table":
        """
        Drop the specified columns (strict: raises on miss).
        """
        cols = list(column_names)
        if not cols:
            return self._maybe_update(self, self.table, inplace=in_place)

        if self.is_ray():
            existing = set(self.table.schema().names)
            missing = [c for c in cols if c not in existing]
            if missing:
                raise KeyError(f"drop_columns: columns not found: {missing}")

            if set(cols) == existing:
                def _empty(batch: pd.DataFrame) -> pd.DataFrame:
                    return batch.iloc[:, 0:0]
                new_obj = self.table.map_batches(_empty, batch_format="pandas")
            else:
                new_obj = self.table.drop_columns(cols)
            return self._maybe_update(self, new_obj, inplace=in_place)

        existing = list(self.table.columns)
        missing = [c for c in cols if c not in existing]
        if missing:
            raise KeyError(f"drop_columns: columns not found: {missing}")

        new_obj = self.table.drop(columns=cols)
        return self._maybe_update(self, new_obj, inplace=in_place)

    def assign_rowwise_using(
        self,
        column_name: str,
        func: Callable[[pd.Series], Any],
        *,
        inplace: bool = True,
        batch_size: int = 4_096,        # Ray only
        **ray_opts,
    ) -> "Table":
        if self.is_ray():
            new_ds = self._ray_assign_rowwise(self.table, column_name, func, batch_size, **ray_opts)
            return self._maybe_update(self, new_ds, inplace)

        if self.is_dask():
            new_df = self._dask_assign_rowwise(self.table, column_name, func)
            return self._maybe_update(self, new_df, inplace)

        new_df = self._pandas_assign_rowwise(self.table, column_name, func)
        return self._maybe_update(self, new_df, inplace)

    # ====================================================================== #
    # Non-mutation table functions
    # ====================================================================== #

    @staticmethod
    def _validate_n(n: int, *, method: str) -> int:
        if not isinstance(n, int) or n < 0:
            raise ValueError(f"{method}: 'n' must be a non‑negative int, got {n!r}")
        return n

    def head(self, n: int = 5) -> "Table":
        n = self._validate_n(n, method="head")
        if n == 0:
            return self.__class__(self.table.limit(0) if self.is_ray() else self.table.iloc[0:0],
                         smiles_column_name=self.smiles_column_name)

        if self.is_ray():
            return self.__class__(self.table.limit(n), smiles_column_name=self.smiles_column_name)
        if self.is_dask():
            return self.__class__(self.table.head(n), smiles_column_name=self.smiles_column_name)
        return self.__class__(self.table.head(n), smiles_column_name=self.smiles_column_name)

    def tail(self, n: int = 10) -> "Table":
        n = self._validate_n(n, method="tail")
        if n == 0 or len(self) == 0:
            return self.__class__(self.table.limit(0) if self.is_ray() else self.table.iloc[0:0],
                         smiles_column_name=self.smiles_column_name)

        if self.is_ray():
            new_ds = self.table.skip(max(len(self) - n, 0))
            return self.__class__(new_ds, smiles_column_name=self.smiles_column_name)
        if self.is_dask():
            return self.__class__(self.table.tail(n), smiles_column_name=self.smiles_column_name)
        return self.__class__(self.table.tail(n), smiles_column_name=self.smiles_column_name)

    def sample(self, n: int = 10, *, seed: int | None = None) -> "Table":
        n = self._validate_n(n, method="sample")
        total = len(self)
        if n > total:
            raise ValueError(f"sample: 'n' ({n}) exceeds table size ({total})")
        if n == 0:
            return self.__class__(self.table.limit(0) if self.is_ray() else self.table.iloc[0:0],
                         smiles_column_name=self.smiles_column_name)
        if self.is_ray():
            new_ds = self.table.random_shuffle(seed=seed).limit(n)
            return self.__class__(new_ds, smiles_column_name=self.smiles_column_name)
        if self.is_dask():
            frac = n / total
            return self.__class__(self.table.sample(frac=frac, random_state=seed),
                         smiles_column_name=self.smiles_column_name)
        return self.__class__(self.table.sample(n=n, random_state=seed),
                     smiles_column_name=self.smiles_column_name)

    def top(self, key: Union[str, Callable], n: int = 10) -> "Table":
        n = self._validate_n(n, method="top")
        return self.sort(key, ascending=False, inplace=False).head(n)

    def bottom(self, key: Union[str, Callable], n: int = 10) -> "Table":
        n = self._validate_n(n, method="bottom")
        return self.sort(key, ascending=True, inplace=False).head(n)

    # ====================================================================== #
    # Materialization
    # ====================================================================== #

    def to_pandas(self, *, max_rows: int | None = 5_000_000) -> pd.DataFrame:
        est = len(self)
        if max_rows is not None and est > max_rows:
            raise MemoryError(
                f"Refusing to materialise {est:,} rows in‑memory "
                "(override via max_rows=None if you really want that)."
            )

        if self.is_ray():
            return self.table.to_pandas()
        if self.is_dask():
            return self.table.compute()
        return self.table

    # ====================================================================== #
    # Export
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
        path = pathlib.Path(path).expanduser().resolve()

        if self.is_ray():
            self._ray_export(self.table, path, fmt, sep, **kwargs)
            return path

        if self.is_dask():
            self._dask_export(self.table, path, fmt, sep, single_file, **kwargs)
            return path

        self._pandas_export(self.table, path, fmt, sep, **kwargs)
        return path

    def to_csv(self, path: pathlib.Path | str, **kwargs) -> pathlib.Path:
        return self._export(path, fmt="csv", sep=",", **kwargs)

    def to_tsv(self, path: pathlib.Path | str, **kwargs) -> pathlib.Path:
        return self._export(path, fmt="tsv", sep="\t", **kwargs)

    def to_parquet(self, path: pathlib.Path | str, **kwargs) -> pathlib.Path:
        return self._export(path, fmt="parquet", **kwargs)

    # ====================================================================== #
    # Merge (pandas-style API) with Ray awareness
    # ====================================================================== #

    def merge(
        self,
        right: "Table | pd.DataFrame | dd.DataFrame | _RayDataset",
        how: str = "inner",
        on: str | Sequence[str] | None = None,
        left_on: str | Sequence[str] | None = None,
        right_on: str | Sequence[str] | None = None,
        left_index: bool = False,
        right_index: bool = False,
        sort: bool = False,
        suffixes: tuple[str, str] = ("_x", "_y"),
        copy: bool = True,
        indicator: bool | str = False,
        validate: str | None = None,
        *,
        inplace: bool = False,
        **kwargs,
    ) -> "Table":
        right_obj = right.table if isinstance(right, Table) else right

        # ---- Ray path when either side is Ray --------------------------------
        if self.is_ray() or self._is_ray_dataset(right_obj):
            if left_index or right_index:
                raise NotImplementedError("Index-based joins are not supported for Ray datasets.")

            # Convert operands to Ray if needed
            def _to_ray(obj):
                if self._is_ray_dataset(obj):
                    return obj
                if self._is_dask_dataframe(obj):
                    obj = obj.compute()
                if isinstance(obj, pd.DataFrame):
                    if ray is None:
                        raise RuntimeError("ray is not available to perform a Ray merge")
                    return ray.data.from_pandas(obj)
                raise TypeError(f"Cannot convert {type(obj)} into a Ray Dataset")

            left_ds  = self.table if self.is_ray() else _to_ray(self.table)
            right_ds = _to_ray(right_obj)

            # Resolve keys
            if on is not None:
                left_on = right_on = on
            left_keys  = [left_on]  if isinstance(left_on,  str) else list(left_on or [])
            right_keys = [right_on] if isinstance(right_on, str) else list(right_on or [])
            if not left_keys and not right_keys:
                raise ValueError("Must pass 'on' or 'left_on'/'right_on' for Ray merges.")
            if len(left_keys) != len(right_keys):
                raise ValueError("'left_on' and 'right_on' must have the same number of columns.")

            # If names differ, rename right side to match left side
            if left_keys != right_keys:
                rename_map = dict(zip(right_keys, left_keys))
                if hasattr(right_ds, "rename_columns"):
                    try:
                        right_ds = right_ds.rename_columns(rename_map)  # preferred
                    except TypeError:
                        # Older Ray: needs full list of new names.
                        names = list(right_ds.schema().names)
                        new_names = [rename_map.get(n, n) for n in names]
                        right_ds = right_ds.rename_columns(new_names)
                else:
                    raise RuntimeError("Ray Dataset missing rename_columns API.")

            # Construct a version-safe join call
            join_sig = inspect.signature(left_ds.join).parameters
            join_kwargs = {}
            if "on" in join_sig:
                join_kwargs["on"] = left_keys
            elif {"left_on", "right_on"} <= set(join_sig.keys()):
                join_kwargs["left_on"] = left_keys
                join_kwargs["right_on"] = left_keys
            else:
                raise RuntimeError("Ray Dataset.join() signature changed unexpectedly.")

            if "how" in join_sig:
                join_kwargs["how"] = how
            elif how.lower() != "inner":
                warnings.warn(
                    f"Ray join fallback to pandas for how='{how}'.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                left_pdf  = left_ds.to_pandas()
                right_pdf = right_ds.to_pandas()
                merged_pdf = left_pdf.merge(
                    right_pdf,
                    how=how,
                    on=left_keys,
                    suffixes=suffixes,
                    copy=copy,
                    indicator=indicator,
                    validate=validate,
                    **kwargs,
                )
                return self._maybe_update(self, merged_pdf, inplace)

            if "lsuffix" in join_sig:
                join_kwargs["lsuffix"] = suffixes[0]
            if "rsuffix" in join_sig:
                join_kwargs["rsuffix"] = suffixes[1]

            merged_ds = left_ds.join(right_ds, **join_kwargs)

            if sort or indicator or validate:
                warnings.warn("sort / indicator / validate are ignored for Ray merges", RuntimeWarning, stacklevel=2)

            return self._maybe_update(self, merged_ds, inplace)

        # ---- Dask path --------------------------------------------------------
        if self.is_dask() and self._is_dask_dataframe(right_obj):
            merged_ddf = dd.merge(  # type: ignore[attr-defined]
                self.table,
                right_obj,
                how=how,
                on=on,
                left_on=left_on,
                right_on=right_on,
                left_index=left_index,
                right_index=right_index,
                suffixes=suffixes,
                indicator=indicator,
                **kwargs,
            )
            return self._maybe_update(self, merged_ddf, inplace)

        # ---- Fallback: materialize to pandas ---------------------------------
        left_pdf = self.to_pandas()
        if self._is_dask_dataframe(right_obj):
            right_pdf = right_obj.compute()
        elif self._is_ray_dataset(right_obj):
            right_pdf = right_obj.to_pandas()
        else:
            right_pdf = right_obj

        merged_pdf = left_pdf.merge(
            right_pdf,
            how=how,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            sort=sort,
            suffixes=suffixes,
            copy=copy,
            indicator=indicator,
            validate=validate,
            **kwargs,
        )
        return self._maybe_update(self, merged_pdf, inplace)
    
# =========================================================================== #
# Other Table Functionality:
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Table Data Type Conversions:
# --------------------------------------------------------------------------- #

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

    raise ValueError(f"Unsupported extension, {ext}, for path, {path}.")

def to_table(obj: Any, force_pandas=False, to_class=Table) -> Table:
    """Normalise many possible inputs into a `Table`."""
    
    if isinstance(obj, to_class):
        return obj
    
    if isinstance(obj, (str, pathlib.Path)):
        return to_class(table=path_to_table(pathlib.Path(obj)), force_pandas=force_pandas)
    
    if isinstance(obj, pd.DataFrame):
        return to_class(table=obj, force_pandas=force_pandas)
    
    if isinstance(obj, np.ndarray):
        return to_class(table=pd.DataFrame(obj), force_pandas=force_pandas)
    
    raise TypeError(f"Cannot convert object, {obj}, of type {type(obj)} to Table")
    
# --------------------------------------------------------------------------- #
# Combine Tables:
# --------------------------------------------------------------------------- #

def combine_tables(
    tables: Iterable[Table],
    on: Union[str, Callable[[pd.Series], Any]],
    *,
    how: str = "outer",                      # kept for compatibility; ignored
    return_long: bool = False,               # True → return long form; False → pivot once to wide
    assay_name: str = "Assay",               # name for the long "assay/column label"
    value_name: str = "Value",               # name for the long "value"
    agg: Union[str, Callable] = "max",       # pivot aggregation when duplicates exist
    dropna_values: bool = True,              # drop rows where Value is NaN in long form
    pivot_max_rows: int | None = None,       # guard when materialising to pandas for pivot; None disables
) -> Table:
    """
    Combine multiple `Table` objects by:
      1) computing a join key per table (`on`),
      2) converting each to long: [key, assay_name, value_name],
      3) concatenating all long tables (outer semantics),
      4) optionally pivoting once to wide (index=key, columns=assay_name, values=value_name).
    """
    # ------------------------------- 0) Validate & normalise -------------------------------
    tbl_list: list[Table] = list(tables)
    if not tbl_list:
        raise ValueError("combine_tables: at least one Table is required.")
    
    # TODO: Implement a column_names() method in the Table class.
    # Helper to list columns per backend
    def _cols(tbl: Table) -> list[str]:
        if tbl.is_ray():
            return list(tbl.table.schema().names)
        return list(tbl.table.columns)
    
    # Gather all existing column names to avoid name collisions
    existing = set()
    for _t in tbl_list:
        existing.update(_cols(_t))

    # Choose unique internal column names that do not collide with inputs
    def _unique_name(base: str) -> str:
        name, k = base, 1
        while name in existing:
            k += 1
            name = f"{base}_{k}"
        existing.add(name)
        return name

    key_col    = _unique_name("__combine_key__")
    assay_col  = _unique_name(assay_name if assay_name not in existing else f"__{assay_name}__")
    value_col  = _unique_name(value_name if value_name not in existing else f"__{value_name}__")

    # ------------------------------- 1) Build long version for each table -----------------
    long_tables: list[Table] = []

    for idx, tbl in enumerate(tbl_list):
        # Stable, safe tag per table to disambiguate assay labels
        # TODO: Change to use get and has name methods.
        raw_tag = getattr(tbl, "name", None) or f"t{idx}"
        tag = re.sub(r"\W+", "_", str(raw_tag)).strip("_") or f"t{idx}"

        # 1a) Attach/compute the key column *without* mutating `tbl`
        if isinstance(on, str):
            # vectorised copy of an existing column
            t_with_key = tbl.assign_rowwise_using(key_col, lambda df, _c=on: df[_c], inplace=False)
        else:
            # strict row-wise computation
            t_with_key = tbl.assign_rowwise_using(key_col, on, inplace=False)

        # 1b) Identify value columns (everything except the key AND the original `on` if str)
        cols = _cols(t_with_key)
        exclude = {key_col}
        if isinstance(on, str):
            exclude.add(on)
        value_cols = [c for c in cols if c not in exclude]

        if not value_cols:
            # nothing to melt; create empty long table with correct schema
            empty = pd.DataFrame({key_col: pd.Series(dtype="object"),
                                  assay_col: pd.Series(dtype="object"),
                                  value_col: pd.Series(dtype="float64")})
            long_tables.append(to_table(empty, smiles_col=key_col))
            continue

        # 1c) Convert to long (backend-aware)
        if t_with_key.is_ray():
            def _to_long(batch: pd.DataFrame,
                         _key=key_col, _assay=assay_col, _val=value_col,
                         _vals=value_cols, _tag=tag) -> pd.DataFrame:
                sub = batch[[_key] + _vals]
                long = sub.melt(id_vars=[_key], value_vars=_vals,
                                var_name=_assay, value_name=_val)
                long[_assay] = long[_assay].astype(str) + f"__{_tag}"
                if dropna_values:
                    long = long[~pd.isna(long[_val])]
                return long

            new_ds = t_with_key.table.map_batches(
                _to_long, batch_format="pandas", batch_size=4096
            )
            long_tables.append(Table(new_ds, smiles_column_name=key_col))

        elif t_with_key.is_dask():
            ddf = t_with_key.table[[*value_cols, key_col]].melt(
                id_vars=[key_col], var_name=assay_col, value_name=value_col
            )
            ddf = ddf.map_partitions(
                lambda df, _ac=assay_col, _tag=tag: df.assign(**{_ac: df[_ac].astype(str) + f"__{_tag}"}),
            )
            if dropna_values:
                ddf = ddf[ddf[value_col].notna()]
            long_tables.append(Table(ddf, smiles_column_name=key_col))

        else:  # Pandas
            pdf = t_with_key.table[[*value_cols, key_col]].melt(
                id_vars=[key_col], var_name=assay_col, value_name=value_col
            )
            pdf[assay_col] = pdf[assay_col].astype(str) + f"__{tag}"
            if dropna_values:
                pdf = pdf.dropna(subset=[value_col])
            long_tables.append(to_table(pdf, smiles_col=key_col))

    # ------------------------------- 2) Concatenate long tables (outer/union) --------------
    def _concat_tables(ts: list[Table]) -> Table:
        if not ts:
            return to_table(pd.DataFrame({key_col: [], assay_col: [], value_col: []}),
                                 smiles_col=key_col)
        if all(t.is_ray() for t in ts):
            ds = ts[0].table
            for t in ts[1:]:
                ds = ds.union(t.table)
            return Table(ds, smiles_column_name=key_col)
        if all(t.is_dask() for t in ts):
            ddf = dd.concat([t.table for t in ts], axis=0, interleave_partitions=True)
            return Table(ddf, smiles_column_name=key_col)
        # fallback: materialise mixed back-ends to Pandas
        pdfs = []
        for t in ts:
            if t.is_ray():
                pdfs.append(t.table.to_pandas())
            elif t.is_dask():
                pdfs.append(t.table.compute())
            else:
                pdfs.append(t.table)
        return to_table(pd.concat(pdfs, ignore_index=True), smiles_col=key_col)

    long_all = _concat_tables(long_tables)

    # ------------------------------- 2.5) Rename key BEFORE any use ------------------------
    final_key = on if isinstance(on, str) else "CombinedKey"
    if final_key != key_col:
        if long_all.is_ray():
            def _rename_key(batch: pd.DataFrame, old=key_col, new=final_key):
                return batch.rename(columns={old: new})
            long_all = Table(
                long_all.table.map_batches(_rename_key, batch_format="pandas"),
                smiles_column_name=final_key,
            )
        elif long_all.is_dask():
            long_all = Table(long_all.table.rename(columns={key_col: final_key}),
                             smiles_column_name=final_key)
        else:
            long_all = to_table(long_all.table.rename(columns={key_col: final_key}),
                                     smiles_col=final_key)

    # ------------------------------- 3) Return long, or pivot to wide ----------------------
    if return_long:
        # align public names for assay/value (key already renamed)
        if (assay_col != assay_name) or (value_col != value_name):
            if long_all.is_ray():
                def _rename_av(batch: pd.DataFrame, m={assay_col: assay_name, value_col: value_name}):
                    return batch.rename(columns=m)
                long_all = Table(
                    long_all.table.map_batches(_rename_av, batch_format="pandas"),
                    smiles_column_name=final_key,
                )
            elif long_all.is_dask():
                long_all = Table(long_all.table.rename(columns={assay_col: assay_name, value_col: value_name}),
                                 smiles_column_name=final_key)
            else:
                long_all = to_table(
                    long_all.table.rename(columns={assay_col: assay_name, value_col: value_name}),
                    smiles_col=final_key
                )
        return long_all

    # Pivot once to wide (Pandas)
    pdf_long = long_all.to_pandas(max_rows=pivot_max_rows)  # raises if > pivot_max_rows

    # Ensure compact encodings before pivot (optional but helps memory)
    if pdf_long[assay_col].dtype != "category":
        pdf_long[assay_col] = pdf_long[assay_col].astype("category")
    # If key is string-like, category can also save memory
    if pd.api.types.is_object_dtype(pdf_long[final_key]):
        pdf_long[final_key] = pdf_long[final_key].astype("category")

    wide = (
        pdf_long
        .pivot_table(
            index=final_key,
            columns=assay_col,
            values=value_col,
            aggfunc=agg,
            observed=True,
        )
        .reset_index()
    )

    # Clean up column index if it became a MultiIndex
    if isinstance(wide.columns, pd.MultiIndex):
        wide.columns = [c if isinstance(c, str) else c[-1] for c in wide.columns]
    wide.columns.name = None

    return to_table(wide)

# ########################################################################### #
# Chemistry Methods:
# ########################################################################### #

# =========================================================================== #
# Chemistry Pre-Instantiated Values:
# =========================================================================== #

_UNCHARGER = None
_TAUTOMERIZER = None

def _get_uncharger():
    global _UNCHARGER
    if _UNCHARGER is None:
        _UNCHARGER = rdMolStandardize.Uncharger()
    return _UNCHARGER

def _get_tautomerizer():
    global _TAUTOMERIZER
    if _TAUTOMERIZER is None:
        _TAUTOMERIZER = rdMolStandardize.TautomerEnumerator()
    return _TAUTOMERIZER


# =========================================================================== #
# Chemistry Helpers:
# =========================================================================== #

MolLike = Union[str, Chem.Mol]

# =========================================================================== #
# Data Type Conversions:
# =========================================================================== #

# --------------------------------------------------------------------------- #
# General Utility:
# --------------------------------------------------------------------------- #

def is_valid_smiles(s: str, *, sanitize: bool = True, quiet: bool = True) -> bool:
    """True if s parses to an RDKit Mol, else False."""
    if not isinstance(s, str) or not s.strip():
        return False
    try:
        if quiet:
            RDLogger.DisableLog('rdApp.*')
        return Chem.MolFromSmiles(s, sanitize=sanitize) is not None
    except Exception:
        return False
    finally:
        if quiet:
            RDLogger.EnableLog('rdApp.*')

# --------------------------------------------------------------------------- #
# Molecular Standardization:
# --------------------------------------------------------------------------- #

def _standardize_mol_like_objects(obj, out_type, keep_stereo=True):
    
    mol = get("utility", "to_mol")(obj)
    
    if mol is None:
        raise ValueError("Invalid molecule")

    # 1) Cleanup: disconnect metals, normalize groups, reionize, etc.
    mol = rdMolStandardize.Cleanup(mol)

    # 2) Strip to parent (remove salts/solvents)
    mol = rdMolStandardize.FragmentParent(mol, skipStandardize=True)

    # 3) Uncharge (optional; keep if you want neutral forms)
    mol = _get_uncharger().uncharge(mol)

    # 4) Canonical tautomer (dedup/search)
    mol = _get_tautomerizer().Canonicalize(mol)

    # 5) Stereo handling
    if keep_stereo:
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    else:
        Chem.RemoveStereochemistry(mol)

    if out_type in ("smiles",):
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=keep_stereo)
    elif out_type in ("mol",):
        return mol
    else:
        raise ValueError(f"out_type must be 'mol' or 'smiles'; got {out_type!r}")

def standardized_smiles(mol_like, *, keep_stereo=True):
    return _standardize_mol_like_objects(mol_like, "smiles", keep_stereo=keep_stereo)

def standardized_mol(mol_like, *, keep_stereo=True):
    return _standardize_mol_like_objects(mol_like, "mol", keep_stereo=keep_stereo)

# --------------------------------------------------------------------------- #
# To Data-Type Conversion Methods:
# --------------------------------------------------------------------------- #

@register("utility", "to_smiles")
def to_smiles(x):
    """Return SMILES; accept SMILES str or RDKit Mol."""
    
    if isinstance(x, str) and is_valid_smiles(x):
        return x
    
    elif isinstance(x, Mol):
        return Chem.MolToSmiles(x, isomericSmiles=True, canonical=True)
    
    raise TypeError("Expected SMILES str or RDKit Mol.")

@register("utility", "to_mol")
def to_mol(x: MolLike, *, sanitize: bool = True) -> Optional[Chem.Mol]:
    """
    Convert SMILES or Mol to a sanitized RDKit Mol.
    Returns None on failure (never raises).
    """
    
    # 1. If 
    if x is None:
        return None
    
    elif isinstance(x, Mol):
        return x
    
    # 2. If it isn't already converted
    else:
        try:
            return Chem.MolFromSmiles(x, sanitize=sanitize)
        except Exception as e:
            print(e)
            return None
        
# --------------------------------------------------------------------------- #
# Cache Mol-Like Objects:
# --------------------------------------------------------------------------- #

def cache_mol_like(
    *, maxsize: int = 200_000, isomeric: bool = True, arg_pos: int = 0
) -> Callable:
    """
    Cache a function by standardized (canonical, tautomer-collapsed) SMILES
    of the Mol/SMILES argument at `arg_pos`, but call the function with the
    original argument unchanged.
    """
    cache = LRUCache(maxsize=maxsize)

    def keyfunc(*args, **kwargs) -> str:
        x = args[arg_pos]  # FIX: honor arg_pos
        return standardized_smiles(x, keep_stereo=isomeric)  # FIX: honor isomeric

    def deco(func: Callable) -> Callable:
        @cached(cache, key=keyfunc)
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)  # original object passed through
        wrapper.cache_clear = cache.clear
        wrapper.cache_obj = cache
        return wrapper

    return deco

# =========================================================================== #
# Scaler Molecular Descriptors:
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Basic Structural Properties:
# --------------------------------------------------------------------------- #

@register("descriptors", "mw")
@cache_mol_like(maxsize=MAX_CACHE_SIZE, isomeric=True)
def mw(x: MolLike) -> Optional[float]:
    m = to_mol(x)
    return float(Descriptors.MolWt(m)) if m else None

@register("descriptors", "hbd")
@cache_mol_like(maxsize=MAX_CACHE_SIZE, isomeric=True)
def hbd(x: MolLike) -> Optional[int]:
    m = to_mol(x)
    return int(rdMolDescriptors.CalcNumHBD(m)) if m else None

@register("descriptors", "hba")
@cache_mol_like(maxsize=MAX_CACHE_SIZE, isomeric=True)
def hba(x: MolLike) -> Optional[int]:
    m = to_mol(x)
    return int(rdMolDescriptors.CalcNumHBA(m)) if m else None

@register("descriptors", "rot_bonds_strict")
@cache_mol_like(maxsize=MAX_CACHE_SIZE, isomeric=True)
def rot_bonds_strict(x: MolLike) -> Optional[int]:
    m = to_mol(x)
    if not m:
        return None
    try:
        opts = rdMolDescriptors.NumRotatableBondsOptions.Strict
        return int(rdMolDescriptors.CalcNumRotatableBonds(m, opts))
    except (AttributeError, TypeError):
        # Fallback for older RDKit builds
        return int(rdMolDescriptors.CalcNumRotatableBonds(m, True))

@register("descriptors", "ring_count")
@cache_mol_like(maxsize=MAX_CACHE_SIZE, isomeric=True)
def ring_count(x: MolLike) -> Optional[int]:
    m = to_mol(x)
    return int(m.GetRingInfo().NumRings()) if m else None

@register("descriptors", "nheavy")
@cache_mol_like(maxsize=MAX_CACHE_SIZE, isomeric=True)
def nheavy(x: MolLike) -> Optional[int]:
    m = to_mol(x)
    return int(m.GetNumHeavyAtoms()) if m else None

# NOTE: natoms can vary with explicit Hs / representation; do not SMILES-cache.
@register("descriptors", "natoms")
def natoms(x: MolLike) -> Optional[int]:
    m = to_mol(x)
    return int(m.GetNumAtoms()) if m else None

# --------------------------------------------------------------------------- #
# Basic Property Predictions:
# --------------------------------------------------------------------------- #

@register("descriptors", "logp")
@cache_mol_like(maxsize=MAX_CACHE_SIZE, isomeric=True)
def logp(x: MolLike) -> Optional[float]:
    m = to_mol(x)
    return float(Crippen.MolLogP(m)) if m else None

@register("descriptors", "tpsa")
@cache_mol_like(maxsize=MAX_CACHE_SIZE, isomeric=True)
def tpsa(x: MolLike) -> Optional[float]:
    m = to_mol(x)
    return float(rdMolDescriptors.CalcTPSA(m)) if m else None

# =========================================================================== #
# Categorical Molecular Descriptors:
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Substructure Matching:
# --------------------------------------------------------------------------- #

@register("smarts", "smarts_in_smiles")
def smarts_in_smiles(smarts: str, smiles: str) -> bool:
    """Safe SMARTS-in-SMILES substructure check."""
    mol = to_mol(smiles, sanitize=True)
    patt = Chem.MolFromSmarts(smarts)
    return bool(mol and patt and mol.HasSubstructMatch(patt))


# --------------------------------------------------------------------------- #
# Structural Filters:
# --------------------------------------------------------------------------- #

@register("alerts", "catalog_for")
#@lru_cache(maxsize=None)
def catalog_for(flag_name: str) -> Optional[FilterCatalog]:
    """
    Lazily build a FilterCatalog for a single RDKit flag name (e.g., 'BRENK').
    Returns None if the flag is not available in this RDKit build.
    """
    
    try:
        flag = getattr(FilterCatalogParams.FilterCatalogs, flag_name)
        
    except AttributeError:
        logger.debug("Filter flag %s not exposed by current RDKit.", flag_name)
        raise ValueError(f"The catalog {flag_name} is not available in this version of RdKit.")
        
    params = FilterCatalogParams()
    params.AddCatalog(flag)
    return FilterCatalog(params)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Assay Artifact Filters:
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

@register("alerts", "pains_catalog")
#@lru_cache(maxsize=None)
def pains_catalog() -> FilterCatalog:
    """PAINS A+B+C combined into a single catalog (validated via catalog_for)."""
    names = ("PAINS_A", "PAINS_B", "PAINS_C")

    # Validate that each subset is available (catalog_for raises ValueError if not)
    try:
        for name in names:
            catalog_for(name)
    except ValueError as e:
        raise ValueError(f"Cannot construct PAINS catalog: {e}") from e

    # Build the combined catalog via params (RDKit combines subsets this way)
    params = FilterCatalogParams()
    for name in names:
        params.AddCatalog(getattr(FilterCatalogParams.FilterCatalogs, name))
    return FilterCatalog(params)


@register("alerts", "pains")
@cache_mol_like(maxsize=MAX_CACHE_SIZE, isomeric=True)
def pains(x: MolLike) -> bool:
    m = to_mol(x)
    if not m:
        return False
    cat = get("alerts", "pains_catalog")()
    return bool(cat and cat.HasMatch(m))

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Toxicity Filters:
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

@register("alerts", "brenk")
@cache_mol_like(maxsize=MAX_CACHE_SIZE, isomeric=True)
def brenk(x: MolLike) -> bool:
    m = to_mol(x)
    if not m:
        return False
    cat = get("alerts", "catalog_for")("BRENK")
    return bool(cat and cat.HasMatch(m))

@register("alerts", "glaxo")
@cache_mol_like(maxsize=MAX_CACHE_SIZE, isomeric=True)
def glaxo(x: MolLike) -> bool:
    m = to_mol(x)
    if not m:
        return False
    cat = get("alerts", "catalog_for")("CHEMBL_Glaxo")
    return bool(cat and cat.HasMatch(m))

@register("alerts", "dundee")
@cache_mol_like(maxsize=MAX_CACHE_SIZE, isomeric=True)
def dundee(x: MolLike) -> bool:
    m = to_mol(x)
    if not m:
        return False
    cat = get("alerts", "catalog_for")("CHEMBL_Dundee")
    return bool(cat and cat.HasMatch(m))

@register("alerts", "bms")
@cache_mol_like(maxsize=MAX_CACHE_SIZE, isomeric=True)
def bms(x: MolLike) -> bool:
    m = to_mol(x)
    if not m:
        return False
    cat = get("alerts", "catalog_for")("CHEMBL_BMS")
    return bool(cat and cat.HasMatch(m))

@register("alerts", "surechembl")
@cache_mol_like(maxsize=MAX_CACHE_SIZE, isomeric=True)
def surechembl(x: MolLike) -> bool:
    m = to_mol(x)
    if not m:
        return False
    cat = get("alerts", "catalog_for")("CHEMBL_SureChEMBL")
    return bool(cat and cat.HasMatch(m))

@register("alerts", "mlsmr")
@cache_mol_like(maxsize=MAX_CACHE_SIZE, isomeric=True)
def mlsmr(x: MolLike) -> bool:
    m = to_mol(x)
    if not m:
        return False
    cat = get("alerts", "catalog_for")("CHEMBL_MLSMR")
    return bool(cat and cat.HasMatch(m))

@register("alerts", "inpharmatica")
@cache_mol_like(maxsize=MAX_CACHE_SIZE, isomeric=True)
def inpharmatica(x: MolLike) -> bool:
    m = to_mol(x)
    if not m:
        return False
    cat = get("alerts", "catalog_for")("CHEMBL_Inpharmatica")
    return bool(cat and cat.HasMatch(m))

@register("alerts", "lint")
@cache_mol_like(maxsize=MAX_CACHE_SIZE, isomeric=True)
def lint(x: MolLike) -> bool:
    m = to_mol(x)
    if not m:
        return False
    cat = get("alerts", "catalog_for")("CHEMBL_LINT")
    return bool(cat and cat.HasMatch(m))

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Drug-Likeness Filters:
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

@register("rules", "lipinski")
def lipinski(x: MolLike) -> bool:
    m = to_mol(x)
    if not m:
        return False
    return (
        mw(m)   <= 500
        and logp(m) <= 5
        and Lipinski.NumHDonors(m)    <= 5
        and Lipinski.NumHAcceptors(m) <= 10
    )

@register("rules", "veber")
def veber(x: MolLike) -> bool:
    m = to_mol(x)
    if not m:
        return False
    return (rot_bonds_strict(m) <= 10) and (tpsa(m) <= 140 or (hbd(m) + hba(m) <= 12))

@register("rules", "ghose")
def ghose(x: MolLike) -> bool:
    m = to_mol(x)
    if not m:
        return False
    mr = Descriptors.MolMR(m)  # consider caching if hot
    return all([
        160 <= mw(m)   <= 480,
        -0.4 <= logp(m) <= 5.6,
        40  <= mr <= 130,
        20  <= natoms(m) <= 70,   # not SMILES-cached
    ])

@register("rules", "muegge")
def muegge(x: MolLike) -> bool:
    m = to_mol(x)
    if not m:
        return False
    return all([
        200 <= mw(m) <= 600,
        -2  <= logp(m) <= 5,
        tpsa(m) <= 150,
        hbd(m) <= 5,
        hba(m) <= 10,
        rot_bonds_strict(m) <= 15,
        1   <= ring_count(m) <= 7,
    ])

@register("rules", "egan")
def egan(x: MolLike) -> bool:
    m = to_mol(x)
    if not m:
        return False
    return (logp(m) < 5.88) and (tpsa(m) < 131.6)

@register("rules", "egan_rule")
def egan_rule(smiles: str) -> bool:
    return egan(smiles)

@register("rules", "pfizer_3_75")
def pfizer_3_75(x: MolLike) -> bool:
    m = to_mol(x)
    if not m:
        return False
    return (hbd(m) <= 3) and (tpsa(m) <= 75)

@register("rules", "gsk_4_400")
def gsk_4_400(x: MolLike) -> bool:
    m = to_mol(x)
    if not m:
        return False
    return (hbd(m) <= 4) and (mw(m) <= 400)

@register("rules", "rule_of_three")
def rule_of_three(x: MolLike) -> bool:
    m = to_mol(x)
    if not m:
        return False
    return (mw(m) <= 300) and (logp(m) <= 3) and (hbd(m) <= 3) and (hba(m) <= 3)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Other Filters:
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

def _sigmoid(x: float, k: float, x0: float) -> float:
    return 1.0 / (1.0 + math.exp(k * (x - x0)))

@register("scores", "calc_cns_mpo_score")
@cache_mol_like(maxsize=MAX_CACHE_SIZE, isomeric=True)
def calc_cns_mpo_score(x: MolLike) -> Optional[float]:
    """
    Approximate 0–6 CNS MPO score (Wager et al.).
    Note: This is the "clogP/clogD≈clogP" approximation variant.
    """
    m = to_mol(x)
    if not m:
        return None
    clogp  = logp(m)
    clogd  = clogp  # crude approximation
    w      = mw(m)
    s      = tpsa(m)
    donors = hbd(m)
    if any(v is None for v in (clogp, w, s, donors)):
        return None
    pka    = 8.0
    return (
        _sigmoid(clogp, 0.5, 3)
        + _sigmoid(clogd, 0.5, 3)
        + _sigmoid(w, 0.035, 360)
        + _sigmoid(s, 0.07, 60)
        + _sigmoid(donors, 1.0, 1)
        + _sigmoid(pka, 1.0, 8)
    )

@register("scores", "cns_mpo_filter")
def cns_mpo_filter(smiles: str, threshold: float = 4.0) -> bool:
    score = calc_cns_mpo_score(smiles)
    return (score is not None) and (score >= threshold)

_PRIMARY_AMINE = Chem.MolFromSmarts(
    "[N;H2;D1;!$(N[*]=O);!$(N[*]S(=O)=O);!$(N[a])]"
)

def _has_primary_amine(mol: Chem.Mol) -> bool:
    return bool(_PRIMARY_AMINE and mol.HasSubstructMatch(_PRIMARY_AMINE))

def _embed_and_get_coords(mol: Chem.Mol) -> Optional[np.ndarray]:
    """
    Embed a single ETKDGv3 conformer, return (N, 3) coordinates.
    None on failure. Hydrogenated embedding (recommended).
    """
    mol3d = Chem.AddHs(mol)
    try:
        params = AllChem.ETKDGv3()
    except AttributeError:
        params = AllChem.ETKDGv2()
    params.randomSeed = 0xF00D
    if hasattr(params, "maxAttempts"):
        params.maxAttempts = 20

    try:
        status = AllChem.EmbedMolecule(mol3d, params)
    except TypeError:
        status = AllChem.EmbedMolecule(
            mol3d, maxAttempts=20, randomSeed=0xF00D, clearConfs=True
        )
    if status != 0:
        return None

    AllChem.UFFOptimizeMolecule(mol3d, maxIters=200)
    conf = mol3d.GetConformer()
    return np.asarray(conf.GetPositions(), dtype=float)

def _globularity(coords: np.ndarray) -> float:
    centred = coords - coords.mean(axis=0)
    cov = np.cov(centred, rowvar=False)
    eigvals = np.sort(np.linalg.eigvalsh(cov))
    return 0.0 if eigvals[-1] < 1e-8 else float(eigvals[0] / eigvals[-1])

#@lru_cache(maxsize=10_000)
def _coords_for_canonical_smiles(cansmi: str) -> Optional[np.ndarray]:
    m = to_mol(cansmi)
    if not m:
        return None
    return _embed_and_get_coords(m)

def _coords_for_mol(mol: Chem.Mol) -> Optional[np.ndarray]:
    cansmi = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    return _coords_for_canonical_smiles(cansmi)

def _formal_charge(mol: Chem.Mol) -> int:
    return int(Chem.GetFormalCharge(mol))

def _approx_hbd_surface_area(mol: Chem.Mol) -> float:
    return float(rdMolDescriptors.CalcNumHBD(mol)) * 12.0

@register("rules", "entry_rules")
def entry_rules(x: MolLike, species: Optional[str] = None) -> bool:
    """
    Return True iff molecule satisfies the eNTRy rules for the given species:

    - species is None → E. coli (original eNTRy rules; Richter et al., 2017)
    - species == "PA" → P. aeruginosa (porin-independent; Geddes et al., 2023)
    """
    mol = to_mol(x)
    if not mol:
        return False

    # E. coli rules
    if species is None or species in names_for_escherichia_coli:
        if not _has_primary_amine(mol):
            return False
        if rot_bonds_strict(mol) > 5:
            return False
        coords = _coords_for_mol(mol)
        if coords is None:
            logger.debug("3D embedding failed; failing closed for eNTRy.")
            return False
        if _globularity(coords) > 0.25:
            return False
        return True

    # P. aeruginosa rules
    if species in names_for_pseudomonas_aeruginosa:
        has_positive_charge = (_formal_charge(mol) >= 1) or _has_primary_amine(mol)
        if not has_positive_charge:
            return False
        if _approx_hbd_surface_area(mol) < 23.0:
            return False
        if not ((_formal_charge(mol) >= 1) or (tpsa(mol) >= 80.0)):
            return False
        return True

    raise ValueError("Unknown species %r. Use None (E. coli) or 'PA'." % (species,))

# --------------------------------------------------------------------------- #
# Pre-Defined Sets of Categorical Descriptors:
# --------------------------------------------------------------------------- #

all_structural_alerts = [
    get("alerts", "glaxo"),
    get("alerts", "dundee"),
    get("alerts", "bms"),
    get("alerts", "pains"),
    get("alerts", "surechembl"),
    get("alerts", "mlsmr"),
    get("alerts", "inpharmatica"),
    get("alerts", "lint"),
]

all_tox_filter_names = [
    "glaxo",
    "dundee",
    "bms",
    "pains",
    "surechembl",
    "mlsmr",
    "inpharmatica",
    "lint",
]

# =========================================================================== #
# Structural Descriptors:
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Scaffolds:
# --------------------------------------------------------------------------- #

@register("scaffold", "bemis_murcko_scaffold")
def bemis_murcko_scaffold(smiles: str, generic: bool = False) -> str:
    """
    Return the Bemis–Murcko scaffold as a canonical SMILES string.
    """
    mol = to_mol(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if generic:
        scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
    return Chem.MolToSmiles(scaffold, isomericSmiles=not generic, canonical=True)


# =========================================================================== #
# Vector Descriptors:
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Substructure-Based Fingerprints:
# --------------------------------------------------------------------------- #

@register("fingerprint", "morgan")
def morgan_fingerprint(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)

# --------------------------------------------------------------------------- #
# Latent-Space Based Fingerprints:
# --------------------------------------------------------------------------- #

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Minimol Representation Generation:
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

"""
Note that this implemenation of the minimol fingerprint is neither clean nor
optimal. However, minimol was very hard to implement, so this part of the code
base has been left dirty to avoid reimplementation challenges.
"""

def _force_graphium_float32() -> None:
    """
    Promote Graphium featurizer dtypes from float16 → float32 (avoid SciPy bugs).
    Safe no-op when Graphium is absent or already patched.
    """
    try:
        from graphium.features import featurizer as _gf  # type: ignore
        import numpy as _np  # local
        from functools import partial as _partial
    except Exception:
        return

    if getattr(_gf, "_dtype_patched", False):
        return

    _gf.mol_to_adj_and_features = _partial(_gf.mol_to_adj_and_features, dtype=_np.float32)
    _gf.mol_to_graph_dict = _partial(_gf.mol_to_graph_dict, dtype=_np.float32)
    _gf.mol_to_pyggraph = _partial(_gf.mol_to_pyggraph, dtype=_np.float32)
    _gf._dtype_patched = True


def _init_minimol_single_thread():
    """
    Instantiate Minimol() once without inner joblib multiprocessing.
    """
    os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
    os.environ.setdefault("JOBLIB_N_JOBS", "1")

    _force_graphium_float32()
    from minimol import Minimol  # type: ignore

    model = Minimol()
    try:
        dm = model.datamodule
        dm.featurization_backend = "threading"
        dm.featurization_n_jobs = 1
        dm.num_workers = 0
    except AttributeError:
        pass
    return model


def get_minimol_fingerprint(smiles: str):
    """
    Compute the Minimol representation for a single SMILES.
    """
    os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
    os.environ.setdefault("JOBLIB_N_JOBS", "1")
    _force_graphium_float32()
    from minimol import Minimol  # type: ignore

    model = Minimol()
    return model(smiles)


class _MinimolActor:
    """
    Ray actor that owns one Minimol model per worker.
    """

    def __init__(self, smiles_col: str, output_col: str):
        self._smiles_col = smiles_col
        self._output_col = output_col
        self._model = _init_minimol_single_thread()

    def __call__(self, batch_df: "pd.DataFrame") -> "pd.DataFrame":  # noqa: F821
        fps, bad = [], []
        for idx, smi in enumerate(batch_df[self._smiles_col].astype(str)):
            try:
                fp = self._model(smi)[0]
                fps.append(fp.tolist() if hasattr(fp, "tolist") else fp)
            except Exception:
                fps.append(None)
                bad.append(idx)

        if bad:
            logger.info(
                "[Minimol] skipped %d invalid SMILES out of %d in batch",
                len(bad),
                len(batch_df),
            )

        batch_df[self._output_col] = fps
        return batch_df

def add_minimol_fingerprints(
    tbl: "Table",  # noqa: F821
    *,
    override_smiles_column_name_as: Optional[str] = None,
    output_column: str = "Minimol_Representation",
    batch_size: int = 1_024,
):
    """
    Add Minimol representations to a Table-like object using Ray batches if available.
    Assumes `tbl` exposes: .is_ray(), .table.map_batches(...), .iter_chunks(), etc.
    """
    import pandas as pd  # type: ignore

    _ensure_ray()
    smiles_col = override_smiles_column_name_as or getattr(tbl, "smiles_column_name", None)
    if smiles_col is None:
        raise ValueError("No SMILES column defined on the Table object.")

    if getattr(tbl, "is_ray", lambda: False)():
        from ray.data import ActorPoolStrategy  # type: ignore

        strategy = ActorPoolStrategy()
        tbl.table = tbl.table.map_batches(
            _MinimolActor,
            fn_constructor_kwargs=dict(smiles_col=smiles_col, output_col=output_column),
            batch_format="pandas",
            batch_size=batch_size,
            compute=strategy,
            runtime_env={"conda": "Pipeline_Minimol_Environment"},
        )
        return tbl

    # Fallback: single-process pandas path
    actor = _MinimolActor(smiles_col, output_column)
    processed = [actor(chunk) for chunk in tbl.iter_chunks(batch_size)]
    tbl.table = pd.concat(processed, ignore_index=True)
    return tbl

# =========================================================================== #
# Chemical Distance:
# =========================================================================== #

@register("similarity", "tanimoto_distance")
def tanimoto_distance(a: MolLike, b: MolLike, *, radius: int = 2, n_bits: int = 2048) -> float:
    """
    1 − Tanimoto similarity on Morgan fingerprints. Returns 1.0 if either parse fails.
    """
    ma, mb = to_mol(a), to_mol(b)
    if not ma or not mb:
        return 1.0
    fpa = morgan_fingerprint(ma, radius=radius, n_bits=n_bits)
    fpb = morgan_fingerprint(mb, radius=radius, n_bits=n_bits)
    return 1.0 - DataStructs.TanimotoSimilarity(fpa, fpb)

@register("similarity", "graph_edit_distance")
def graph_edit_distance(a: MolLike, b: MolLike) -> float:
    """
    Slow toy graph-edit distance via MCS size.
    Returns 1.0 if either parse fails or MCS fails.
    """
    from rdkit.Chem import rdFMCS

    ma, mb = to_mol(a), to_mol(b)
    if not ma or not mb:
        return 1.0
    res = rdFMCS.FindMCS([ma, mb], completeRingsOnly=True, ringMatchesRingOnly=True)
    if not res.smartsString:
        return 1.0
    mcs = Chem.MolFromSmarts(res.smartsString)
    if not mcs:
        return 1.0
    return 1.0 - mcs.GetNumAtoms() / max(ma.GetNumAtoms(), mb.GetNumAtoms())

# -----------------------------------------
# Synthetic accessibility (SA) and RA score
# -----------------------------------------

def get_sa_score(smiles: str) -> Optional[float]:
    """
    Ertl–Schuffenhauer SA score (1 easy → 10 hard). Uses C++ path if available.
    """
    mol = to_mol(smiles)
    if not mol:
        return None
    try:
        return float(rdMolDescriptors.CalcSAScore(mol))
    except AttributeError:
        pass
    try:
        # Python fallback for older RDKit
        from rdkit.Chem.SA_Score import sascorer  # type: ignore

        return float(sascorer.calculateScore(mol))
    except Exception:
        return None


# Backwards-compatible alias (kept for users of the old name)
get_sascore = get_sa_score


_RA_ACTOR_NAME = "__ra_scorer__"
_RASCORE_RUNTIME_ENV = {
    "pip": [
        "git+https://github.com/reymond-group/RAscore.git@master",
        "xgboost>=1.7,<2.0",
        "scikit-learn>=1.0,<2.0",
    ]
}


def init_ra_scorer_if_needed() -> None:
    """
    Create the detached RA-scorer exactly once per Ray cluster.
    """
    _ensure_ray()
    try:
        import ray  # type: ignore
    except Exception as e:  # pragma: no cover
        logger.debug("Ray not available for RA scorer: %s", e)
        return

    try:
        ray.get_actor(_RA_ACTOR_NAME)
    except ValueError:
        @ray.remote(runtime_env=_RASCORE_RUNTIME_ENV)  # type: ignore
        class _RAscoreActor:
            def __init__(self):
                from RAscore import RAscore_XGB  # type: ignore

                self._scorer = RAScore = RAscore_XGB.RAScorerXGB()

            def score(self, smiles: str) -> Optional[float]:
                try:
                    return float(self._scorer.predict_from_smiles([smiles])[0])
                except Exception:
                    return None

        _RAscoreActor.options(name=_RA_ACTOR_NAME, lifetime="detached").remote()  # type: ignore


def get_ra_score(mol_like: MolLike) -> Optional[float]:
    """
    0–1 probability that AiZynthFinder finds a synthetic route.
    Requires that `init_ra_scorer_if_needed` has been called.
    """

    smiles = to_smiles(mol_like)
    actor = ray.get_actor(_RA_ACTOR_NAME)
    
    try:
        return ray.get(actor.score.remote(smiles))  # type: ignore
    
    except (ValueError, Exception):
        return None

# ########################################################################### #
# Pairwise Table Computations:
# ########################################################################### #

# =========================================================================== #
# Pairwise-Chemical Distance Calculation:
# =========================================================================== #

def combine(
    results_grid: List[List[Optional[object]]],
    chunk_lengths: Sequence[int],
    *,
    triangle: str = "lower",          # "lower", "upper", or "all"
    assume_symmetric: bool = True,    # only used if triangle="upper" with lower-grid schedule
    index_dtype: str = "int32",
    distance_dtype: str = "float32",
    include_smiles: bool = False,
    smiles: Optional[Sequence[str]] = None,  # global SMILES lookup by global index
    resolve_futures: bool = True,     # resolve Ray ObjectRefs if present
) -> pd.DataFrame:
    """
    Assemble a single long DataFrame from a grid of block results.

    Parameters
    ----------
    results_grid
        2D list where results_grid[i][j] is either:
          • a 1D numpy array of length chunk_lengths[i] * chunk_lengths[j] (row-major), or
          • a Ray ObjectRef to such an array, or
          • None (unscheduled block).
        Your current scheduler fills only i >= j (lower grid).
    chunk_lengths
        Length of each chunk in the original order; used to rebuild global indices.
    triangle
        "lower" (default, consistent with i >= j scheduling),
        "upper" (requires assume_symmetric=True to swap off-diagonal blocks), or
        "all" (keep all pairs as computed).
    assume_symmetric
        If True and triangle="upper", off-diagonal blocks (i>j) are flipped so output
        has i <= j. For diagonal blocks (i==j), only keep i_local <= j_local to avoid
        duplicates.
    include_smiles, smiles
        If True and a global SMILES list is provided, append 'smiles_i'/'smiles_j'.
    resolve_futures
        If True, attempt ray.get() on non-None entries (no-ops if Ray not installed).

    Returns
    -------
    pd.DataFrame with columns:
        idx_i[int32], idx_j[int32], distance[float32]
        (+ optional smiles_i/smiles_j if include_smiles=True)
    """
    # 1) Build global offsets from chunk lengths.
    n_chunks = len(chunk_lengths)
    if any(cl < 0 for cl in chunk_lengths):
        raise ValueError("chunk_lengths must be non-negative.")
    offsets = np.zeros(n_chunks + 1, dtype=np.int64)
    if n_chunks:
        offsets[1:] = np.cumsum(np.asarray(chunk_lengths, dtype=np.int64))
    off = offsets[:-1]  # off[i] = global start index of chunk i

    # 2) Collect per-block columns, then concatenate once.
    idx_i_parts: List[np.ndarray] = []
    idx_j_parts: List[np.ndarray] = []
    dist_parts:  List[np.ndarray] = []

    # Helper: safely resolve Ray futures (if any).
    def _resolve(x):
        if not resolve_futures or x is None:
            return x
        try:
            import ray  # local import to avoid hard dependency
            if isinstance(x, ray.ObjectRef):
                return ray.get(x)
        except Exception:
            # Not a Ray ObjectRef or ray not available; treat x as materialized.
            pass
        return x

    # Sanity on grid dimensions.
    if len(results_grid) != n_chunks or any(len(row) != n_chunks for row in results_grid):
        raise ValueError("results_grid must be a square list-of-lists matching chunk_lengths.")

    for i in range(n_chunks):
        nA = int(chunk_lengths[i])
        if nA == 0:
            continue
        for j in range(n_chunks):
            res = results_grid[i][j]
            if res is None:
                continue

            nB = int(chunk_lengths[j])
            if nB == 0:
                continue

            dists = _resolve(res)
            # Allow either bare ndarray or (array, meta). We only need the array.
            if isinstance(dists, tuple) and len(dists) >= 1:
                dists = dists[0]

            dists = np.asarray(dists, dtype=distance_dtype, copy=False)
            expected = nA * nB
            if dists.size != expected:
                raise ValueError(
                    f"Block ({i},{j}) size mismatch: got {dists.size}, expected {expected} "
                    f"(len_a={nA}, len_b={nB})."
                )

            # Local indices for this block
            i_local = np.repeat(np.arange(nA, dtype=np.int64), nB)
            j_local = np.tile  (np.arange(nB, dtype=np.int64), nA)

            # Lift to global indices
            i_global = off[i] + i_local
            j_global = off[j] + j_local

            if triangle == "all":
                idx_i_parts.append(i_global.astype(index_dtype, copy=False))
                idx_j_parts.append(j_global.astype(index_dtype, copy=False))
                dist_parts.append(dists)
                continue

            if triangle == "lower":
                # Keep i >= j
                mask = i_global >= j_global
                if mask.any():
                    idx_i_parts.append(i_global[mask].astype(index_dtype, copy=False))
                    idx_j_parts.append(j_global[mask].astype(index_dtype, copy=False))
                    dist_parts.append(dists[mask])
                continue

            if triangle == "upper":
                if not assume_symmetric:
                    raise ValueError(
                        "triangle='upper' requires assume_symmetric=True with lower-grid scheduling."
                    )
                if i == j:
                    # Diagonal block: keep only i_local <= j_local to avoid duplicates
                    mask = i_local <= j_local
                    if mask.any():
                        idx_i_parts.append(i_global[mask].astype(index_dtype, copy=False))
                        idx_j_parts.append(j_global[mask].astype(index_dtype, copy=False))
                        dist_parts.append(dists[mask])
                elif i > j:
                    # Off-diagonal lower-grid block: flip to upper orientation (i <= j)
                    # (distance assumed symmetric)
                    idx_i_parts.append(j_global.astype(index_dtype, copy=False))
                    idx_j_parts.append(i_global.astype(index_dtype, copy=False))
                    dist_parts.append(dists)
                else:
                    # i < j block shouldn't exist with your scheduler, but handle gracefully:
                    mask = i_global <= j_global
                    if mask.any():
                        idx_i_parts.append(i_global[mask].astype(index_dtype, copy=False))
                        idx_j_parts.append(j_global[mask].astype(index_dtype, copy=False))
                        dist_parts.append(dists[mask])
                continue

            raise ValueError("triangle must be one of {'lower','upper','all'}.")

    # 3) Materialize the final DataFrame (typed), optionally add SMILES.
    if not dist_parts:
        df = pd.DataFrame({
            "idx_i": np.array([], dtype=index_dtype),
            "idx_j": np.array([], dtype=index_dtype),
            "distance": np.array([], dtype=distance_dtype),
        })
    else:
        idx_i = np.concatenate(idx_i_parts, axis=0)
        idx_j = np.concatenate(idx_j_parts, axis=0)
        dist  = np.concatenate(dist_parts,  axis=0).astype(distance_dtype, copy=False)

        df = pd.DataFrame(
            {"idx_i": idx_i, "idx_j": idx_j, "distance": dist},
            copy=False,
        )
        # Ensure dtypes (concat can upcast)
        df["idx_i"] = df["idx_i"].astype(index_dtype, copy=False)
        df["idx_j"] = df["idx_j"].astype(index_dtype, copy=False)
        df["distance"] = df["distance"].astype(distance_dtype, copy=False)

    if include_smiles and smiles is not None and len(df) > 0:
        smi = np.asarray(smiles, dtype=object)
        df["smiles_i"] = smi[df["idx_i"].values]
        df["smiles_j"] = smi[df["idx_j"].values]

    return df

def pairwise_distances(
    smiles_a: Sequence[str],
    smiles_b: Sequence[str],
    metric_fn: Callable[[str, str], float],
    metric_fn_args = ("cell", "cell"),   # use metric_fn_args[1] ∈ {"cell","row"}
    *,
    dtype=np.float32,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Compute all pairwise distances for a block A×B in row-major order.

    Returns
    -------
    distances : 1D np.ndarray[dtype], length = len(A) * len(B)
    meta : dict -> {"len_a": len(A), "len_b": len(B)}
    """
    a, b, f = smiles_a, smiles_b, metric_fn
    nA, nB = len(a), len(b)

    out = np.empty(nA * nB, dtype=dtype)
    pos = 0  # write pointer into flat buffer

    mode = metric_fn_args[1]  # "cell" or "row"

    if mode == "cell":
        for i in range(nA):
            ai = a[i]
            for j in range(nB):
                out[pos] = f(ai, b[j])
                pos += 1

    elif mode == "row":
        for i in range(nA):
            row = np.asarray(f(a[i], b), dtype=dtype)
            if row.shape[0] != nB:
                raise ValueError(f"Row metric returned length {row.shape[0]} (expected {nB}).")
            out[pos:pos + nB] = row
            pos += nB
    else:
        raise ValueError("metric_fn_args[1] must be 'cell' or 'row'.")

    return out

@ray.remote
def pairwise_distances_remote(
    smiles_a: Sequence[str],
    smiles_b: Sequence[str],
    metric_fn: Callable[[str, str], float],
    metric_fn_args = ("cell", "cell"),
    *,
    dtype=np.float32,
) -> Tuple[np.ndarray, Dict[str, int]]:
    
    return pairwise_distances(
        smiles_a,
        smiles_b,
        metric_fn,
        metric_fn_args = metric_fn_args,
        dtype=np.float32,
    )

@ensure_ray
def chunked_pairwise_distances(
    table: Table,
    metric: Union[str, Callable[[str, str], float]] = "tanimoto",
    *,
    chunk_size: int = 4_096,
) -> Table:

    # 1. Get SMILES chunks.
    chunks = [chunk[table.smiles_column_name].astype(str).tolist() 
                     for chunk in table.iter_chunks(chunk_size)]

    # 2. Resolve metric function.
    if isinstance(metric, str):
        metric = get_similarity(metric)
    elif callable(metric):
        pass
    else:
        raise TypeError

    metric_blob = pickle.dumps(metric)
    
    # 3. Iterate over chunks.
    pairwise_distances = [[None for i in range(len(chunks))] for i in range(len(chunks))]
    
    for i, chunk_1 in enumerate(chunks):
        for j, chunk_2 in enumerate(chunks):
            
            if i < j: 
                continue
            
            pairwise_distances[i][j] = pairwise_distances_remote(chunk_1, chunk_2)
            
    # 4. Accumulate objects into a single long df.
    full_df = combine(pairwise_distances)

    return full_df

###############################################################################
# Embedding Functions:
###############################################################################

def df_to_smiles_dict(dist_df: pd.DataFrame) -> Dict[int, str]:
    """
    Build {row_index → SMILES} from a long-form distance DataFrame.

    - If present, uses dist_df.attrs['smiles_lookup'] (expected to map 0..n-1 → SMILES).
    - Else, if idx_i is categorical, uses its categories order as 0..n-1 → SMILES.
    """
    lookup = dist_df.attrs.get("smiles_lookup")
    if lookup is not None:
        # Accept Series or dict-like
        if isinstance(lookup, pd.Series):
            return lookup.to_dict()
        return dict(lookup)

    if pd.api.types.is_categorical_dtype(dist_df["idx_i"]):
        cats = dist_df["idx_i"].cat.categories
        return {i: smi for i, smi in enumerate(cats)}

    raise ValueError(
        "Distance DataFrame lacks a 'smiles_lookup' attribute and 'idx_i' is not categorical; "
        "cannot rebuild SMILES dictionary deterministically."
    )

def longdf_to_dense(dist_df: pd.DataFrame) -> np.ndarray:
    req = {"idx_i", "idx_j", "distance"}
    if not req.issubset(dist_df.columns):
        missing = req - set(dist_df.columns)
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    i = dist_df["idx_i"]
    j = dist_df["idx_j"]

    if i.isna().any() or j.isna().any():
        raise ValueError("idx_i/idx_j contain NA values; cannot build a dense matrix.")

    lookup = dist_df.attrs.get("smiles_lookup")
    if lookup is not None:
        # Canonical case: 0..n-1 coding must be present
        n = int(len(lookup))

        if pd.api.types.is_categorical_dtype(i) and pd.api.types.is_categorical_dtype(j):
            i_idx = i.cat.codes.to_numpy(np.int32)
            j_idx = j.cat.codes.to_numpy(np.int32)
        elif pd.api.types.is_integer_dtype(i) and pd.api.types.is_integer_dtype(j):
            i_idx = i.to_numpy(np.int32, copy=False)
            j_idx = j.to_numpy(np.int32, copy=False)
        else:
            raise ValueError(
                "With 'smiles_lookup' present, idx_i/idx_j must be integer-coded or categorical "
                "with 0..n-1 codes."
            )

        if (i_idx < 0).any() or (j_idx < 0).any() or i_idx.max() >= n or j_idx.max() >= n:
            raise ValueError("Indices out of bounds for smiles_lookup length.")
    else:
        # Fallback: use idx_i's category order as the indexer for both columns
        if not pd.api.types.is_categorical_dtype(i):
            raise ValueError(
                "Without 'smiles_lookup', idx_i must be categorical to define the canonical order."
            )
        cats = i.cat.categories
        i_idx = i.cat.codes.to_numpy(np.int32)
        j_idx = pd.Categorical(j, categories=cats).codes.astype(np.int32)
        if (i_idx < 0).any() or (j_idx < 0).any():
            unseen = j[pd.Categorical(j, categories=cats).codes < 0].unique()
            raise ValueError(f"idx_j contains ids not present in idx_i categories: {unseen!r}")
        n = len(cats)

    d_val = dist_df["distance"].to_numpy(np.float32, copy=False)

    D = np.zeros((n, n), dtype=np.float32)
    D[i_idx, j_idx] = d_val
    D[j_idx, i_idx] = d_val
    np.fill_diagonal(D, 0.0)
    return D

@register("embed", "umap")
@ray.remote(runtime_env={"pip": ["umap-learn"]})                       # fix #9
def umap(dists: np.ndarray, n_components: int, **kwargs):
    import umap
    return umap.UMAP(metric="precomputed",
                     n_components=n_components,
                     **kwargs).fit_transform(dists)

@register("embed", "tsne")                # fix #9
@ray.remote(runtime_env={"pip": ["scikit-learn"]})    
def tsne(dists: np.ndarray, n_components: int, **kwargs):
    from sklearn.manifold import TSNE
    # If the caller did not specify an init, fall back to "random"
    kwargs.setdefault("init", "random")
    return TSNE(
        metric="precomputed",
        n_components=n_components,
        **kwargs,
    ).fit_transform(dists)

def embed_longform_distances(
    dist_df: pd.DataFrame,
    embedding_method: Union[str, Callable] = "umap",
    n_components: int = 2,
    groups: Optional[np.ndarray] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Embed a long-form pairwise distance table; return a DataFrame with SMILES + coordinates.

    Required columns in dist_df: ['idx_i', 'idx_j', 'distance'].
    """
    # 1) Dense matrix
    D = longdf_to_dense(dist_df)

    # 2) Resolve remote embedder
    if isinstance(embedding_method, str):
        remote_embed = get_embedder(embedding_method)  # registry: "embed"
    elif callable(embedding_method):
        remote_embed = embedding_method
    else:
        raise ValueError(f"Unsupported embedding method: {embedding_method!r}")

    # 3) Run embedding remotely via Ray
    coords = ray.get(remote_embed.remote(D, n_components, **kwargs))

    # 4) SMILES dictionary
    id_to_smiles = df_to_smiles_dict(dist_df)
    n = len(id_to_smiles)
    if coords.shape[0] != n:
        raise ValueError(f"Coordinate count ({coords.shape[0]}) != number of items ({n}).")

    smi_order = [id_to_smiles[i] for i in range(n)]
    dim_cols = [f"Dim{k+1}" for k in range(n_components)]
    out = pd.DataFrame(coords, columns=dim_cols)
    out.insert(0, "SMILES", smi_order)

    if groups is not None:
        if len(groups) != n:
            raise ValueError(f"`groups` length ({len(groups)}) must match number of items ({n}).")
        out["Group"] = groups
    return out


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
        embedding_method = get_embedder(embedding_method)
    elif callable(embedding_method):
        pass
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

# ########################################################################### #
# Visualization Methods:
# ########################################################################### #

# =========================================================================== #
# Scatter Plots:
# =========================================================================== #

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
        to_table(merged_df, smiles_col="SMILES"),
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

# =========================================================================== #
# Powerpoint:
# =========================================================================== #

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
    
# =========================================================================== #
# PDFs:
# =========================================================================== #

from pathlib import Path
from typing import Union, Optional

import matplotlib
matplotlib.use("Agg")                      # head‑less backend (safe inside Ray)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image, ImageDraw


def smiles_to_pdf(
    tbl: "Table",
    out_path: Union[str, Path] = "smiles_catalogue.pdf",
    *,
    smiles_column: str = "SMILES",
    n_rows: int = 3,
    n_cols: int = 2,
    img_px: int = 300,
    page_inches: tuple[float, float] = (8.5, 11),     # US‑Letter portrait
    dpi: int = 150,
    title: Optional[str] = None,
) -> Path:
    """
    Render every SMILES in *tbl* into a grid of **n_rows × n_cols** images
    per PDF page (default 3 × 2 = 6).  Pages are written sequentially until
    all structures are exhausted.

    Parameters
    ----------
    tbl : Table
        Any project‑level `Table` (Pandas / Dask / Ray).  Only the SMILES
        column is accessed.
    out_path : str | Path, default "smiles_catalogue.pdf"
        Destination PDF file (parent directories are created as needed).
    smiles_column : str, default "SMILES"
        Column name that stores canonical SMILES strings.
    n_rows, n_cols : int, defaults 3 rows × 2 cols
        Grid layout per page.
    img_px : int, default 300
        Square pixel size of each RDKit depiction.
    page_inches : tuple(float, float), default (8.5, 11)
        Physical page size *in inches* passed to Matplotlib.
    dpi : int, default 150
        Resolution used by Matplotlib when embedding the raster images.
    title : str | None
        Optional document title metadata stored in the PDF.

    Returns
    -------
    pathlib.Path
        Absolute path to the written PDF.
    """
    # --------------------------- 0.  Normalise paths --------------------------
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------- 1.  Collect SMILES --------------------------
    # For huge tables we *stream* via iter_chunks() to avoid large materialise.
    smiles_iter = (
        smi
        for chunk in tbl.iter_chunks(n=4_096)
        for smi in chunk[smiles_column].astype(str).tolist()
    )

    # --------------------------- 2.  PDF writer ------------------------------
    with PdfPages(str(out_path)) as pdf:
        if title is not None:
            pdf.infodict()["Title"] = str(title)

        # --- helper: open a fresh figure with the correct grid --------------
        def _new_figure():
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=page_inches,
                dpi=dpi,
                squeeze=False,
            )
            # Remove all axis decorations
            for ax in axes.ravel():
                ax.axis("off")
            return fig, axes

        fig, axes = _new_figure()
        cell = 0  # runs 0 .. (n_rows*n_cols - 1) across pages

        for smi in smiles_iter:
            i_row, i_col = divmod(cell, n_cols)

            # ---- 2a.  Generate 2‑D depiction or placeholder -----------------
            mol = Chem.MolFromSmiles(smi)
            if mol:
                img = Draw.MolToImage(mol, size=(img_px, img_px), kekulize=True)
            else:
                img = Image.new("RGB", (img_px, img_px), "white")
                d = ImageDraw.Draw(img)
                d.text((10, img_px // 2 - 10), "Invalid\nSMILES", fill="black")

            # ---- 2b.  Draw into the corresponding subplot -------------------
            axes[i_row, i_col].imshow(img)
            axes[i_row, i_col].set_title(smi, fontsize=6, wrap=True)

            cell += 1

            # ---- 2c.  If the page is full, save & start a new page ----------
            if cell == n_rows * n_cols:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                fig, axes = _new_figure()
                cell = 0  # reset within‑page cell counter

        # ----------------- 3.  Flush any partially filled page ---------------
        if cell != 0:                     # last page not yet written
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    return out_path

# ########################################################################### #
# Heavier Algorithms:
# ########################################################################### #

# =========================================================================== #
# Pareto Frontier Calcuation:
# =========================================================================== #

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
    
# =========================================================================== #
# PCA:
# =========================================================================== #

def add_pca_components(
    tbl: Table,
    feature_func: Callable[[pd.Series], Sequence[float]],
    *,
    n_components: int = 2,
    pca_column_name: str = "PCA",
    max_rows: int = 1_000_000,
    inplace: bool = True,
) -> Table:
    """
    Compute principal components from arbitrary per‑row features and
    store the PC scores as a *list* in a new column.

    Parameters
    ----------
    tbl : Table
        Any project‑level `Table` (Pandas, Dask or Ray).
    feature_func : Callable[[pd.Series], Sequence[float]]
        A user‑supplied function that receives one **row** (Pandas
        `Series`) and returns a 1‑D vector (list / tuple / np.ndarray)
        of *equal length for every row*.
    n_components : int, default 2
        Number of principal components to keep.
    pca_column_name : str, default "PCA"
        Name of the output column that will hold the PC coordinate list
        for each row, e.g.  `[PC1, PC2, …]`.
    max_rows : int, default 1_000_000
        Safety guard – refuse to materialise > 1 M rows into memory.
        Set to `None` to disable (use with caution).
    inplace : bool, default True
        • ``True``  → mutate *tbl* and return it  
        • ``False`` → leave *tbl* untouched and return a *new* Table

    Returns
    -------
    Table
        The enriched table (same back‑end as the input).
    """
    # ------------------------------------------------------------------ 1
    # Collect the data into Pandas (with an explicit memory guard)
    # ------------------------------------------------------------------
    df = tbl.to_pandas(max_rows=max_rows).copy()

    # ------------------------------------------------------------------ 2
    # Build the feature matrix X – one row → feature_func(row)
    # ------------------------------------------------------------------
    feats = df.apply(feature_func, axis=1).tolist()
    X = np.asarray(feats, dtype=float)
    if X.ndim != 2:
        raise ValueError("feature_func must return a *1‑D* vector of "
                         "identical length for every row.")
    if n_components > X.shape[1]:
        raise ValueError(f"n_components ({n_components}) exceeds the "
                         f"feature dimension ({X.shape[1]}).")

    # ------------------------------------------------------------------ 3
    # PCA via SVD  (NumPy – avoids scikit‑learn dependency)
    # ------------------------------------------------------------------
    X_centered = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    scores = X_centered @ Vt.T[:, :n_components]          # (N, k)

    # ------------------------------------------------------------------ 4
    # Attach the PC score list to every row
    # ------------------------------------------------------------------
    df[pca_column_name] = [row.tolist() for row in scores]

    # ------------------------------------------------------------------ 5
    # Push back into the original container type
    # ------------------------------------------------------------------
    if tbl.is_ray():
        new_obj = ray.data.from_pandas(df)
    elif tbl.is_dask():
        # Preserve the original partitioning granularity when possible
        nparts = tbl.table.npartitions
        new_obj = dd.from_pandas(df, npartitions=nparts)
    else:  # already Pandas
        new_obj = df

    #return _maybe_update(tbl, new_obj, inplace=inplace)

# =========================================================================== #
# Lookup Compound SMILES:
# =========================================================================== #

# ──────────────────────────────────────────────────────────────────────────────
#  add_smiles_from_common_names                                                │
# ──────────────────────────────────────────────────────────────────────────────
#
# Populate / create a column of SMILES strings by resolving one‑or‑more
# “common‑name” columns.  The resolver tries several public services in the
# following *empirically chosen* order (≈ fastest → slowest, highest → lowest
# hit‑rate):
#
#   1.  Local OPSIN parser          (py2opsin – offline, sub‑ms)
#   2.  NCI/CADD CIR                (cirpy or bare REST)
#   3.  PubChem PUG‑REST            (PubChemPy)
#   4.  Remote OPSIN web‑service
#   5.  ChemSpider                  (ChemSpiPy; needs an API key)
#   6.  Wikidata SPARQL             (last‑chance, niche coverage)
#
# Each candidate SMILES is *validated* with RDKit; invalid strings are ignored
# and the resolver falls through to the next source.
#
# Typical throughput on a laptop (single process) is 15–20 look‑ups / s for
# unique names once the local cache has primed; Ray/Dask back‑ends scale the
# look‑ups linearly with workers.
# ──────────────────────────────────────────────────────────────────────────────
from functools import lru_cache, partial
from typing import Sequence, Union, Optional
import requests, urllib.parse

# def add_smiles_from_common_names(
#     tbl: Table,
#     common_name_columns: Union[str, Sequence[str]],
#     *,
#     smiles_column_name: str = "SMILES",
#     failed_parse_token: Optional[str] = None,
#     chemspider_api_key: Optional[str] = None,
#     inplace: bool = True,
#     batch_size: int = 4_096,
# ) -> Table:
#     """
#     Parameters
#     ----------
#     tbl
#         The `Table` to enrich.
#     common_name_columns
#         One or more columns that hold trivial / trade / IUPAC names.
#         The columns are consulted **in the order given**; the first name
#         that yields a valid SMILES wins.
#     smiles_column_name
#         Name of the destination column to create (or overwrite).
#     failed_parse_token
#         What to write when *no* resolver succeeds (default `None` →
#         the cell is left null / NaN).  Pass a string such as
#         `"PARSE_FAIL"` if you prefer an explicit sentinel.
#     chemspider_api_key
#         RSC ChemSpider API key.  If `None`, the ChemSpider step is skipped.
#     inplace, batch_size
#         Same semantics as other `Table.assign_*` helpers.

#     Returns
#     -------
#     Table
#         The enriched table (same back‑end as *tbl*).
#     """

#     # ── 0. Normalise column parameter ────────────────────────────────────
#     if isinstance(common_name_columns, str):
#         name_cols = (common_name_columns,)
#     else:
#         name_cols = tuple(common_name_columns)

#     # ── 1.  Name → SMILES resolver († lru‑cached for speed) ──────────────
#     @lru_cache(maxsize=100_000)
#     def _resolve_name_cached(name: str) -> Optional[str]:
#         """Return the first *valid* SMILES for *name*, or None on failure."""
#         name = str(name).strip()
#         if not name:
#             return None

#         # 1️⃣ Local OPSIN (py2opsin) – no network
#         try:
#             from py2opsin import py2opsin as _py2opsin            # ≈ 40 µs
#             smi = _py2opsin(name)
#             if smi and _mol_or_none(smi):
#                 return smi
#         except Exception:
#             pass

#         # 2️⃣ NCI CIR – try cirpy first, fall back to raw REST
#         try:
#             import cirpy
#             smi = cirpy.resolve(name, "smiles")
#             if smi and _mol_or_none(smi):
#                 return smi
#         except Exception:
#             pass
#         try:
#             url = (
#                 "https://cactus.nci.nih.gov/chemical/structure/"
#                 f"{urllib.parse.quote(name)}/smiles"
#             )
#             r = requests.get(url, timeout=5)
#             if r.status_code == 200:
#                 smi = r.text.strip()
#                 if smi and "Page not found" not in smi and _mol_or_none(smi):
#                     return smi
#         except Exception:
#             pass

#         # 3️⃣ PubChem PUG‑REST (PubChemPy)
#         try:
#             import pubchempy as pcp
#             hits = pcp.get_compounds(name, "name")
#             if hits:
#                 smi = hits[0].isomeric_smiles or hits[0].canonical_smiles
#                 if smi and _mol_or_none(smi):
#                     return smi
#         except Exception:
#             pass

#         # 4️⃣ Remote OPSIN web‑service
#         try:
#             url = (
#                 "https://opsin.ch.cam.ac.uk/opsin/"
#                 f"{urllib.parse.quote(name)}.json"
#             )
#             r = requests.get(url, timeout=5)
#             if r.status_code == 200:
#                 smi = r.json().get("smiles")
#                 if smi and _mol_or_none(smi):
#                     return smi
#         except Exception:
#             pass

#         # 5️⃣ ChemSpider (requires API key)
#         if chemspider_api_key:
#             try:
#                 from chemspipy import ChemSpider
#                 _cs = ChemSpider(chemspider_api_key)
#                 hits = _cs.search(name, order_by="recordId")
#                 if hits:
#                     smi = hits[0].smiles
#                     if smi and _mol_or_none(smi):
#                         return smi
#             except Exception:
#                 pass

#         # 6️⃣ Wikidata SPARQL – last resort
#         try:
#             q = (
#                 "SELECT ?smiles WHERE { "
#                 f'?compound rdfs:label "{name}"@en . '
#                 "?compound wdt:P233 ?smiles . } LIMIT 1"
#             )
#             r = requests.get(
#                 "https://query.wikidata.org/sparql",
#                 params={"format": "json", "query": q},
#                 headers={"User-Agent": "CommonNameResolver/0.1"},
#                 timeout=10,
#             )
#             if r.status_code == 200:
#                 bindings = r.json()["results"]["bindings"]
#                 if bindings:
#                     smi = bindings[0]["smiles"]["value"]
#                     if smi and _mol_or_none(smi):
#                         return smi
#         except Exception:
#             pass

#         # – all resolvers failed –
#         return None

#     # ── 2.  Per‑row helper (must be *top‑level* picklable) ───────────────
#     def _row_lookup(
#         row: "pd.Series",
#         name_cols: tuple[str, ...],
#         token: Optional[str],
#     ) -> Optional[str]:
#         # Preserve an existing SMILES if the caller already populated it
#         existing = row.get(smiles_column_name)
#         if isinstance(existing, str) and existing.strip():
#             return existing

#         for col in name_cols:
#             smi = _resolve_name_cached(row.get(col, ""))
#             if smi:
#                 return smi
#         return token   # None or user‑supplied sentinel

#     # Freeze static parameters → picklable callable
#     row_func = partial(_row_lookup, name_cols=name_cols, token=failed_parse_token)

#     # ── 3.  Vectorised assign over whichever back‑end the Table uses ─────
#     return tbl.assign_rowwise_using(
#         smiles_column_name,
#         row_func,
#         inplace=inplace,
#         batch_size=batch_size,
#     )


# ########################################################################### #
# Compute Statistics:
# ########################################################################### #

# =========================================================================== #
# Helpers:
# =========================================================================== #

def _ensure_columns_exist(tbl: "Table", cols: Sequence[str]) -> None:
    """Raise KeyError if any requested column is missing."""
    if tbl.is_ray():
        names = set(tbl.table.schema().names)
    else:
        names = set(tbl.table.columns)
    missing = [c for c in cols if c not in names]
    if missing:
        raise KeyError(f"Column(s) not found: {missing}")

def _collect_two_columns_as_pandas(
    tbl: "Table",
    score_col: str,
    truth_col: str,
    *,
    max_rows: int | None = PANDAS_PROMOTE_THRESHOLD,
) -> pd.DataFrame:
    """
    Materialise only the required columns to a Pandas DataFrame with a guard
    on total row count to avoid accidental OOMs.
    """
    _ensure_columns_exist(tbl, (score_col, truth_col))
    n = len(tbl)
    if max_rows is not None and n > max_rows:
        raise MemoryError(
            f"Refusing to materialise {n:,} rows (limit={max_rows:,}). "
            "Pass max_rows=None to override if you are sure you have enough RAM."
        )

    if tbl.is_ray():
        return tbl.table.select_columns([score_col, truth_col]).to_pandas().copy()
    if tbl.is_dask():
        return tbl.table[[score_col, truth_col]].compute().copy()
    return tbl.table[[score_col, truth_col]].copy()

def _binarise_labels(y: pd.Series, pos_label: Any | None) -> np.ndarray:
    """
    Convert a 1-D pandas Series to a binary np.ndarray of {0,1}.

    Rules:
      • bool → {0,1}
      • numeric with exactly two distinct values → map the larger value to 1
        unless pos_label is provided and present
      • otherwise, require an explicit pos_label and map (y == pos_label) → 1
    """
    # 1) Bool
    if y.dtype == bool or pd.api.types.is_bool_dtype(y):
        return y.to_numpy(dtype=np.int32)

    # 2) Numeric two-class
    y_num = pd.to_numeric(y, errors="coerce")
    if y_num.notna().all():
        uniq = np.unique(y_num.to_numpy())
        if uniq.size == 2:
            if pos_label is None or pos_label not in uniq:
                pos = uniq.max()
            else:
                pos = pos_label
            return (y_num.to_numpy() == pos).astype(np.int32)

    # 3) Fallback: require explicit pos_label
    if pos_label is None:
        raise ValueError(
            "Binary ground-truth could not be inferred. "
            "Please pass `pos_label=...` (e.g., 'active')."
        )
    return (y.astype(object).to_numpy() == pos_label).astype(np.int32)

def _clean_xy_for_classification(
    df: pd.DataFrame,
    score_col: str,
    truth_col: str,
    *,
    pos_label: Any | None,
    dropna: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (scores_float, y_binary_int) after cleaning/validation."""
    # scores → float
    s = pd.to_numeric(df[score_col], errors="coerce")
    y = df[truth_col]

    mask = s.notna()
    if dropna:
        mask &= y.notna()
    df = df.loc[mask, [score_col, truth_col]]

    scores = pd.to_numeric(df[score_col], errors="coerce").to_numpy(dtype=float)
    y_bin = _binarise_labels(df[truth_col], pos_label=pos_label)

    # finite scores only
    finite = np.isfinite(scores)
    scores = scores[finite]
    y_bin = y_bin[finite]

    if scores.size == 0:
        raise ValueError("No valid rows left after cleaning.")
    n_pos = int(y_bin.sum())
    n_neg = int(y_bin.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"Both classes are required. Found positives={n_pos}, negatives={n_neg}."
        )
    return scores, y_bin

# =========================================================================== #
# 
# =========================================================================== #

def auroc(
    tbl: "Table",
    score_column: str,
    truth_column: str,
    *,
    pos_label: Any | None = None,
    dropna: bool = True,
    max_rows: int | None = PANDAS_PROMOTE_THRESHOLD,
) -> float:
    """
    Compute **ROC AUC** on (score, ground-truth) pairs from a Table.

    - Exact, tie-aware implementation via the Mann–Whitney U / rank formula:
        AUC = (Σ ranks(positives) − n_pos·(n_pos+1)/2) / (n_pos·n_neg)

    Parameters
    ----------
    tbl : Table
    score_column : str
        Higher scores are assumed to be more likely positive.
    truth_column : str
        Binary ground truth. See `pos_label` for non-(0/1/bool) inputs.
    pos_label : Any | None, default None
        If ground-truth isn’t {0,1}/bool, specify which value is “positive”.
        For numeric two-class data, the larger value is taken as positive by default.
    dropna : bool, default True
        Drop rows where either score or truth is NA.
    max_rows : int | None, default PANDAS_PROMOTE_THRESHOLD
        Safety guard for materialising the two columns.

    Returns
    -------
    float
        ROC AUC in [0,1].
    """
    df = _collect_two_columns_as_pandas(tbl, score_column, truth_column, max_rows=max_rows)
    scores, y_bin = _clean_xy_for_classification(df, score_column, truth_column,
                                                 pos_label=pos_label, dropna=dropna)

    # Rank scores (ascending); average ranks for ties
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    n_pos = float(y_bin.sum())
    n_neg = float(len(y_bin) - n_pos)
    sum_ranks_pos = float(ranks[y_bin == 1].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    # Numerical guard
    return float(max(0.0, min(1.0, auc)))

def auprc(
    tbl: "Table",
    score_column: str,
    truth_column: str,
    *,
    pos_label: Any | None = None,
    dropna: bool = True,
    max_rows: int | None = PANDAS_PROMOTE_THRESHOLD,
) -> float:
    """
    Compute **auPRC** as **Average Precision (AP)**:
        AP = mean of precision@k at each rank k where a new positive is recalled
           = (1 / n_pos) * Σ precision[k] over all ranks k with y_k = 1
      (Equivalent to step-function area under the precision–recall curve.)

    Parameters are the same as `auroc_from_table`.

    Returns
    -------
    float
        Average Precision in [0,1].
    """
    df = _collect_two_columns_as_pandas(tbl, score_column, truth_column, max_rows=max_rows)
    scores, y_bin = _clean_xy_for_classification(df, score_column, truth_column,
                                                 pos_label=pos_label, dropna=dropna)

    # Sort by descending score (stable sort for deterministic tie-handling)
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y_bin[order]

    # Cumulative counts
    tp = np.cumsum(y_sorted, dtype=np.int64)
    ranks = np.arange(1, len(y_sorted) + 1, dtype=np.int64)  # k = 1..N
    precision_at_k = tp / ranks

    n_pos = int(tp[-1])
    if n_pos == 0:
        raise ValueError("No positive examples; auPRC/AP is undefined.")

    # Average precision: mean of precision at ranks where a positive occurs
    ap = float(precision_at_k[y_sorted == 1].sum() / n_pos)
    return ap

def r2(
    tbl: "Table",
    score_column: str,
    truth_column: str,
    *,
    dropna: bool = True,
    batch_size: int = 1_000_000,                 # kept for backward compatibility (unused)
    max_rows: int | None = PANDAS_PROMOTE_THRESHOLD,
    sample_weight: Optional[Sequence[float]] = None,
    multioutput: str = "uniform_average",
) -> float:
    """
    Wrapper around sklearn.metrics.r2_score with the project’s standard cleaning:
      • materialize only the two needed columns (guarded by max_rows)
      • coerce to numeric, drop NA (if dropna), keep only finite values
      • call r2_score(y_true, y_pred, ...)

    Returns NaN if fewer than two valid rows remain.  Note: R² is unbounded below,
    so very negative values can occur when predictions are extremely poor or
    misaligned with truth.
    """
    # 1) Collect & clean
    df = _collect_two_columns_as_pandas(tbl, score_column, truth_column, max_rows=max_rows)

    s = pd.to_numeric(df[score_column], errors="coerce")
    y = pd.to_numeric(df[truth_column], errors="coerce")

    mask = (s.notna() & y.notna()) if dropna else pd.Series(True, index=df.index)
    s = s[mask].astype(float).to_numpy()
    y = y[mask].astype(float).to_numpy()

    finite = np.isfinite(s) & np.isfinite(y)
    s = s[finite]
    y = y[finite]

    if y.size < 2:
        return float("nan")

    # 2) Optional sample weights – align to the same mask if provided
    sw = None
    if sample_weight is not None:
        sw_full = np.asarray(sample_weight, dtype=float)
        if sw_full.shape[0] == len(df):
            sw = sw_full[mask.to_numpy()][finite]
        elif sw_full.shape[0] == y.size:
            sw = sw_full
        else:
            raise ValueError("sample_weight length must match either the full table "
                             "or the cleaned vector length.")

    # 3) Delegate to scikit-learn; fall back to closed-form if sklearn is missing
    try:
        from sklearn.metrics import r2_score as _r2
        return float(_r2(y, s, sample_weight=sw, multioutput=multioutput))
    except Exception:
        # Fallback: same definition (not streaming)
        y_bar = float(y.mean())
        ss_res = float(np.sum((y - s) ** 2))
        ss_tot = float(np.sum((y - y_bar) ** 2))
        return float("nan") if ss_tot == 0.0 else float(1.0 - (ss_res / ss_tot))

def spearman(
    tbl: "Table",
    score_column: str,
    target_column: str,
    *,
    dropna: bool = True,
    max_rows: int | None = PANDAS_PROMOTE_THRESHOLD,
    rank_method: Literal["average", "min", "max", "dense", "first"] = "average",
) -> float:
    """
    Compute **Spearman rank correlation coefficient (ρ)** between a score column
    and a target column.

    Implemented as Pearson correlation of the (tie-aware) ranks.

    Parameters
    ----------
    tbl : Table
        Your unified table (Pandas / Dask / Ray).
    score_column : str
        Column of predicted/score values.
    target_column : str
        Column of ground-truth numeric targets.
    dropna : bool, default True
        Drop rows where either column is NA (or becomes NA after numeric coercion).
    max_rows : int | None, default PANDAS_PROMOTE_THRESHOLD
        Safety guard for materializing the two columns.
    rank_method : {"average","min","max","dense","first"}, default "average"
        Tie-handling strategy passed to pandas.Series.rank(...).

    Returns
    -------
    float
        Spearman’s ρ in [-1, 1]. Returns NaN if fewer than two valid rows remain
        or if either rank vector has zero variance.
    """
    # 1) Collect only the required columns (backend-aware, with guard)
    df = _collect_two_columns_as_pandas(tbl, score_column, target_column, max_rows=max_rows)

    # 2) Coerce to numeric; optionally drop NA; keep only finite values
    s = pd.to_numeric(df[score_column], errors="coerce")
    y = pd.to_numeric(df[target_column], errors="coerce")

    mask = (s.notna() & y.notna()) if dropna else pd.Series(True, index=df.index)
    s = s[mask].astype(float)
    y = y[mask].astype(float)

    s_vals = s.to_numpy(copy=False)
    y_vals = y.to_numpy(copy=False)
    finite = np.isfinite(s_vals) & np.isfinite(y_vals)
    s_vals = s_vals[finite]
    y_vals = y_vals[finite]

    if s_vals.size < 2:
        return float("nan")

    # 3) Rank both vectors (tie-aware)
    s_rank = pd.Series(s_vals).rank(method=rank_method).to_numpy(float)
    y_rank = pd.Series(y_vals).rank(method=rank_method).to_numpy(float)

    # 4) Pearson correlation on ranks
    sx = s_rank - s_rank.mean()
    sy = y_rank - y_rank.mean()
    denom = float(np.sqrt((sx * sx).sum() * (sy * sy).sum()))
    if denom == 0.0:
        return float("nan")

    rho = float((sx * sy).sum() / denom)
    # Clamp small numerical drift into [-1, 1]
    return max(-1.0, min(1.0, rho))


# ########################################################################### #
# END CODEBASE:
# ########################################################################### #

def main():
    
    # dir_path = "/Users/asselism/Desktop/Collins_Lab/Datasets/Organized_by_Data_Type/Bioactivity/Organized_by_Data_Type/Antimicrobial_Activity/Organized_by_Source/ECBD/Original_Datasets/100K/Original_Datasets/"
    # EF_path = dir_path + "export_EOS300080_67 (Enterococcus Faecilius).csv"
    # PA_path = dir_path + "export_EOS300139_129 (Pseudomonas Aeruginosa).csv"
    # EC_path = dir_path + "export_EOS300141_131 (Escherichia Coli).csv"
    # KP_path = dir_path + "export_EOS300153_145 (Klebsiella Pneumoniae).csv"
    # AB_path = dir_path + "export_EOS300157_149 (Acinetobacter Baumanii).csv"
    # SA_path = dir_path + "export_EOS300177_174 (Staphylococcus Aureus).csv"
    # tox_path = "/Users/asselism/Desktop/Collins_Lab/Datasets/Organized_by_Data_Type/ADMET_Data/Organized_by_Data_Type/Toxicity/Organized_by_Source/ECBD/ECBD_HepG2_100K/Original_Datasets/export_EOS300107_86 (ECBD HepG2 Data).csv"
    
    # EF_tbl = to_table(EF_path)
    # PA_tbl = to_table(PA_path)
    # EC_tbl = to_table(EC_path)
    # KP_tbl = to_table(KP_path)
    # AB_tbl = to_table(AB_path)
    # SA_tbl = to_table(SA_path)
    # tox_tbl = to_table(tox_path)
    
    # tables = [EF_tbl, PA_tbl, EC_tbl, KP_tbl, AB_tbl, SA_tbl]
    
    # for species, tbl in zip(("Enterococcus_faecalis", "Pseudomonas_aeruginosa", "Esherichia_coli", "Klebsiella_pneumoniae", "Acinetobacter_baumannii", "Staphylococcus_aureus"), tables):
        
    #     new_col = f"{species}_Percent_Growth_Inhibition"
        
    #     tbl.assign_rowwise_using(new_col, lambda row: row['value'])
        
    #     new_col = f"{species}_Binarized_Hit"
        
    #     tbl.assign_rowwise_using(new_col, lambda row: int(float(row['value']) > 80))
        
    # tox_tbl.assign_rowwise_using("HepG2_Percent_Growth_Inhibition", lambda row: row['value'])
    # tox_tbl.assign_rowwise_using("HepG2_Binarized_Toxicity", lambda row: int(float(row['value']) > 20))
    
    # tables += [tox_tbl]
    
    # combined_tbl = combine_tables(tables, on="smiles")
    # combined_tbl.to_pandas().to_csv(dir_path + "all_ecbd_antibiotic_and_toxicity_data.csv")
    
    #tbl = to_table("/Users/asselism/Desktop/Collins_Lab/Datasets/Organized_by_Data_Type/Bioactivity/Organized_by_Data_Type/Antimicrobial_Activity/Organized_by_Source/ECBD/Original_Datasets/100K/Modified_Datasets/Formatted_ECBD_Antibiotic_Toxicity_Data_Combined.csv")
    
    #tbl = add_minimol_fingerprints(tbl)
    
    #tbl.to_pandas().to_csv("/Users/asselism/Desktop/Collins_Lab/Analysis/Organized_by_Project/Non-toxic_Antibiotics/Analyses_for_Paper/ECBD_Cytotoxicity/Formatted_ECBD_Antibiotic_Toxicity_Data_Combined_with_Minimol_Fingerprints.tsv", sep='\t')
    
    tbl = to_table("/Users/asselism/Desktop/Collins_Lab/Analysis/Organized_by_Project/Non-toxic_Antibiotics/Analyses_for_Paper/ECBD_Cytotoxicity/Formatted_ECBD_Antibiotic_Toxicity_Data_Combined_with_Tox_and_Antibiotic_Scores.csv")
    
    #tbl= tbl.head(n=100)
    
    for alert, alert_name in zip(all_structural_alerts, all_tox_filter_names):
        
        def fn(row, alert_name):
            a = get("alerts", alert_name)
            return int(a(row["SMILES"]))
        
        g = partial(fn, alert_name=alert_name)
        
        tbl.assign_rowwise_using(alert_name, g)
        
    tbl.to_pandas().to_csv("/Users/asselism/Desktop/Collins_Lab/Analysis/Organized_by_Project/Non-toxic_Antibiotics/Analyses_for_Paper/ECBD_Cytotoxicity/Formatted_ECBD_Antibiotic_Toxicity_Data_Combined_with_Tox_and_Antibiotic_Scores_and_Filters.csv")
    
if __name__ == "__main__":
    
    main()