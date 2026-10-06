# ============================================================================

from __future__ import annotations

# --------------------------------------------------------------------------- #
# First Enable Troubleshooting Functionality:
# --------------------------------------------------------------------------- #

print("Started model building pipeline.")

# macOS + mixed conda/pip installs can load two OpenMP runtimes (e.g. conda numpy/BLAS
# and pip torch), which hard-crashes (segfaults) at import. These must be set BEFORE
# numpy/torch are imported, so they live at the very top of the script. setdefault is
# used so you can still override either one from the shell when you want to.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # allow the duplicate OpenMP runtime
os.environ.setdefault("OMP_NUM_THREADS", "1")          # avoid OpenMP oversubscription; raise for more CPU parallelism once stable
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # if accelerator: mps, let MPS-unsupported ops fall back to CPU

import faulthandler
import signal

# Conceptual Idea: 
# To help troubleshoot deadlocks or slow execution, we register a signal
# handler that will dump the Python traceback when a specific user signal (SIGUSR1) is sent.
# We also schedule it to periodically dump tracebacks every 60 seconds automatically,
# providing a continuous monitor of the program's execution state.
faulthandler.register(signal.SIGUSR1)
# faulthandler.dump_traceback_later(60, repeat=True)  # disabled: dumped all-thread tracebacks every
# 60s and looked like a hang during long epochs. Send SIGUSR1 to the process to dump on demand.

# --------------------------------------------------------------------------- #
# Shut Down Ray Data's Autoscaling:
# --------------------------------------------------------------------------- #

def silence_ray_data_autoscaler() -> None:
    """Neutralize Ray Data's autoscaler to dodge a request_resources signature
    skew and a teardown-hook mismatch seen across Ray versions.

    No-op if Ray Data is absent or its internals have moved. A single
    module-level call covers the driver and every Tune worker, since workers
    re-import this module.
    """
    try:
        from ray.data._internal.execution.autoscaler import default_autoscaler as autoscaler_module
    except Exception:
        return  # Ray Data internals absent or relocated; nothing to patch.

    actor_class = getattr(autoscaler_module, "DefaultAutoscalerActor", None)
    if actor_class is not None and hasattr(actor_class, "request_resources"):
        original_request_resources = actor_class.request_resources
        # Wrap only when the installed method is (self, reqs); our call path
        # passes an extra execution id that the older signature rejects.
        if len(inspect.signature(original_request_resources).parameters) == 2:
            def request_resources(self, requirements, execution_id=None):
                return original_request_resources(self, requirements)
            actor_class.request_resources = request_resources

    autoscaler_class = getattr(autoscaler_module, "DefaultAutoscaler", None)
    if autoscaler_class is not None:
        autoscaler_class.on_executor_shutdown = lambda self, *args, **kwargs: None


silence_ray_data_autoscaler()

# --------------------------------------------------------------------------- #
# Built-In Modules:
# --------------------------------------------------------------------------- #

import ast
from ast import literal_eval
import abc
import base64
from collections import defaultdict, deque, OrderedDict
import copy
from copy import deepcopy
from dataclasses import dataclass, field, fields
from functools import partial, wraps
import glob
import inspect
import io
import math
from numbers import Number
import os
import pathlib
from pathlib import Path
import random
import re
import re as regular_expression
import sys
import tempfile
import torchode
import threading
from typing import Any, Callable, Dict, List, Hashable, Iterable, Literal, Mapping, MutableMapping, Optional, Sequence, Tuple, Type, Union
import uuid
import functools

# --------------------------------------------------------------------------- #
# Import from Data Processing Pipeline:
# --------------------------------------------------------------------------- #

data_processing_pipeline_directory_path = os.environ.get(
    "DATA_PROCESSING_PIPELINE_DIR",
    os.path.dirname(os.path.abspath(__file__)),  # default: data-processing module sits next to this file
)
sys.path.insert(0, data_processing_pipeline_directory_path)

# --------------------------------------------------------------------------- #
# External Modules:
# --------------------------------------------------------------------------- #

import types
import torch.hub as torch_hub
torchvision_utilities = types.ModuleType("torchvision.models.utils")
torchvision_utilities.load_state_dict_from_url = torch_hub.load_state_dict_from_url
sys.modules["torchvision.models.utils"] = torchvision_utilities

# Compatibility shim for LibMTL
import sys, types
try:
    # Conceptual Idea: 
    # Torchvision models internally rely on a utility module to load state dictionaries from URLs. 
    # We mock this utility module by dynamically creating a module type and attaching the standard 
    # PyTorch hub loading function to it, ensuring legacy model loading code continues to work.
    from torch.hub import load_state_dict_from_url
    import torchvision.models as torchvision_models
    if not hasattr(torchvision_models, "utils"):
        torchvision_utilities = types.ModuleType("torchvision.models.utils")
        torchvision_utilities.load_state_dict_from_url = load_state_dict_from_url
        sys.modules["torchvision.models.utils"] = torchvision_utilities
except Exception:
    pass

import LibMTL
from LibMTL.architecture import HPS, Cross_stitch, MMoE
from LibMTL.weighting import PCGrad as LibMTLPCGrad

import numpy
import numpy as np

import pandas
import pandas as pandas_package

import ray

from ray.data import DataContext
current_data_context = DataContext.get_current()
# turn off the executor autoscaler
current_data_context.use_streaming_executor = False              # ← key line: use legacy executor
current_data_context.execution_options.autoscaling_enabled = False

from rdkit import Chem
from rdkit.Chem.rdchem import BondStereo

import torch
from torch.utils.data import TensorDataset, IterableDataset, DataLoader
import torch.nn as neural_network
import torch.nn.functional as neural_network_functional
import pytorch_lightning
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import roc_auc_score as sklearn_metrics_area_under_receiver_operating_characteristic
from sklearn.metrics import average_precision_score

from torch_geometric.data import Data, Batch
from torch_geometric.utils import softmax as geometric_softmax
#from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree as torch_geometric_degree
from torch_geometric.nn    import (
    MessagePassing,
    global_mean_pool, global_add_pool, global_max_pool,
    Linear as GeometricLinear,
    GATConv, GCNConv                        # community layer
)
from torch_geometric.data import Batch as GeometricBatch

#from torchmetrics.classification import MulticlassAUROC as torch_metrics_auroc
#from torchmetrics.classification import AveragePrecision as torch_metrics_auprc
from torchmetrics.functional import accuracy as torchmetrics_accuracy
from torchmetrics.functional import auroc as torchmetrics_area_under_receiver_operating_characteristic
from torchmetrics.functional import average_precision as torchmetrics_average_precision

from torch_geometric.utils import scatter  # was: from torch_scatter import scatter (drops the compiled torch_scatter dep; API-compatible here)

from ray import tune
try:
    from ray.air import RunConfig, CheckpointConfig
except Exception:  # newer Ray relocated these from ray.air to ray.tune
    from ray.tune import RunConfig, CheckpointConfig
from ray.tune import TuneConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback

import yaml

# ---- additional imports needed for the splitters below ----
import numpy
from sklearn.model_selection import StratifiedShuffleSplit
from rdkit import DataStructs
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

import yaml

os.environ.setdefault("RAY_LOG_TO_DRIVER", "1")
os.environ.setdefault("RAY_DISABLE_DASHBOARD", "1")
os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")

if not ray.is_initialized():
    ray.init(log_to_driver=True, ignore_reinit_error=True, include_dashboard=False)

import os
#os.environ.setdefault("RAY_TMPDIR",
#    f"/state/partition1/slurm_tmp/{os.environ.get('SLURM_JOB_ID','ray')}.0.0/ray")

from Data_Processing_Pipeline_v1_29 import tempprint, flatten, Table, to_table, broadcast_val_to_list as broadcast_val_to, listify, morgan_fingerprints, tanimoto_distance, all_unique, assert_type, make_plural_if_required
# imports tempprint, flatten, among other methods.

tempprint("Completed imports.")

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
DEFAULT_FILTER_FAILED = False
use_graphics_processing_unit = torch.cuda.is_available()

# =========================================================================== #
# Execute Environment Changes:
# =========================================================================== #

os.environ["TUNE_WARN_EXCESSIVE_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S"] = "0"

# =========================================================================== #
# Enable Module Lookup in Ray Workers:
# =========================================================================== #

current_module_name_string = pathlib.Path(__file__).stem          # "Model_Building_Pipeline_1_10"
sys.modules[current_module_name_string] = sys.modules[__name__]   # point canonical name → __main__

# =========================================================================== #
# Chemical Representations:
# =========================================================================== #

MAXIMUM_ATOMIC_NUMBER = 118
NODE_FEATURE_DIMENSION = MAXIMUM_ATOMIC_NUMBER
ATOM_TYPES  = list(range(1, MAXIMUM_ATOMIC_NUMBER + 1))                             
BOND_TYPES  = {Chem.BondType.SINGLE: 0,
               Chem.BondType.DOUBLE: 1,
               Chem.BondType.TRIPLE: 2,
               Chem.BondType.AROMATIC: 3}
EDGE_FEATURE_DIMENSION = len(BOND_TYPES)

# ########################################################################### #
# Expanded General Functionality:
# ########################################################################### #

# =========================================================================== #
# Data Conversion:
# =========================================================================== #

def convert_graph_data_to_base64_string(graph_data_object):
    if graph_data_object is None:
        return None
    import io, base64, torch
    # Conceptual Idea: 
    # To safely serialize and transport complex graph data objects across network 
    # or textual boundaries, we write the object into an in-memory byte buffer via PyTorch 
    # and then encode those bytes using the standard base64 ASCII format.
    memory_byte_buffer = io.BytesIO()
    torch.save(graph_data_object, memory_byte_buffer)
    return base64.b64encode(memory_byte_buffer.getvalue()).decode("ascii")

def convert_base64_string_to_graph_data(base64_encoded_string):
    if base64_encoded_string is None or base64_encoded_string == "" or (isinstance(base64_encoded_string, float) and math.isnan(base64_encoded_string)):
        return None
    # Conceptual Idea: 
    # The inverse of the serialization function. It takes a base64 encoded ASCII string, 
    # decodes it back into the original binary sequence, places it into an in-memory stream, 
    # and instructs PyTorch to reconstruct the exact geometric data object.
    raw_binary_bytes = base64.b64decode(base64_encoded_string.encode("ascii") if isinstance(base64_encoded_string, str) else base64_encoded_string)
    byte_input_output_stream = io.BytesIO(raw_binary_bytes)
    try:
        # Newer PyTorch: force full unpickler
        return torch.load(byte_input_output_stream, map_location="cpu", weights_only=False)
    except TypeError:
        # Older PyTorch: argument not supported
        byte_input_output_stream.seek(0)
        return torch.load(byte_input_output_stream, map_location="cpu")
    except Exception as decoding_exception:
        print(f"convert_base64_string_to_graph_data failed: {type(decoding_exception).__name__}: {decoding_exception}")
        return None
    
# =========================================================================== #
# Methods to be Overwritten:
# =========================================================================== #

def construct_neural_network_module(*args, **kwargs):
    raise NotImplementedError(
        "Placeholder called before the real construct_neural_network_module "
        "was defined; check import ordering."
    )

# =========================================================================== #
# Table Functionality:
# =========================================================================== #

def convert_table_to_tensor(
    self,
    column_names: Union[str, Sequence[str]],
    *,
    data_type: torch.dtype = torch.float32,
    target_hardware_device: Union[torch.device, str, None] = None,
    maximum_row_count: Union[int, None] = 5_000_000,
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
    if isinstance(column_names, str):
        column_names = [column_names]
    # Conceptual Idea: 
    # Extract specific columns from a potentially distributed table dataset into a memory-resident
    # Pandas DataFrame (capped at a max row count to avoid exhaustion). We then rapidly construct 
    # a PyTorch tensor directly from the continuous numpy arrays and move it to the targeted device.
    pandas_dataframe = self.to_pandas(max_rows=maximum_row_count)[list(column_names)]
    resulting_tensor = torch.as_tensor(pandas_dataframe.to_numpy(copy=False), dtype=data_type, device=target_hardware_device)
    return resulting_tensor if len(column_names) > 1 else resulting_tensor.squeeze(-1)

# Register the two helpers on the upstream `Table` class
Table.to_tensor = convert_table_to_tensor                  # type: ignore[attr-defined]
# Table.as_torch_dataset = convert_table_to_torch_dataset    # type: ignore[attr-defined]

# =========================================================================== #
# IterableTable:
# =========================================================================== #

class IterableTable(Table, IterableDataset):
    pass

def convert_to_iterable_table(plotting_table):
    return to_table(plotting_table, to_class=IterableTable)

# =========================================================================== #
# RowDataset:
# =========================================================================== #

class RowDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        self.rows = dataframe.to_dict("records")  # list of {column: value} dicts
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        return self.rows[index]

# =========================================================================== #
# Reproducability:
# =========================================================================== #

def set_global_random_seed(seed_value: int = 42) -> None:
    random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    
    numpy.random.seed(seed_value) # <-- Added NumPy Seed
        
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        
# =========================================================================== #
# Troubleshooting:
# =========================================================================== #

def print_active_python_threads():
    # Conceptual Idea: 
    # Retrieve all active threads within the current Python process and print their
    # identities and statuses. This is a critical debugging tool for identifying hung 
    # background workers or improper thread pooling behaviors.
    active_thread = threading.enumerate()
    print(f"Python sees {len(active_thread)} threads")
    for execution_thread in active_thread:
        print(
            f"name={execution_thread.name!r}, "
            f"ident={execution_thread.ident}, "
            f"is_daemon={execution_thread.daemon}"
        )
    
# =========================================================================== #
# Tensor Property Checking:
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Tensor Shape:
# --------------------------------------------------------------------------- #

def get_tensor_shape(input_tensor: torch.Tensor, *, calling_module_name: str) -> Tuple[int, ...]:
    
    if not isinstance(input_tensor, torch.Tensor):
        raise TypeError(f"{calling_module_name}.forward expected torch.Tensor, got {type(input_tensor).__name__}.")
    return tuple(input_tensor.shape)

# --------------------------------------------------------------------------- #
# Tensor Shape Checks:
# --------------------------------------------------------------------------- #

def assert_number_of_dimensions_is(input_tensor: torch.Tensor, expected_dimension_count: int, calling_module_name: str) -> None:
    # Conceptual Idea: 
    # Verify that the input tensor strictly contains the required number of axes. 
    # Failing early here prevents obscure dimensional mismatch errors during forward pass computations.
    tensor_shape_tuple = get_tensor_shape(input_tensor, calling_module_name=calling_module_name)
    if len(tensor_shape_tuple) != expected_dimension_count:
        raise ValueError(f"{calling_module_name}.forward expected tensor with == {expected_dimension_count} dims, got shape {tensor_shape_tuple}.")

def assert_number_of_dimensions_is_at_least(input_tensor: torch.Tensor, expected_minimum_dimension_count: int, calling_module_name: str) -> None:
    
    tensor_shape_tuple = get_tensor_shape(input_tensor, calling_module_name=calling_module_name)
    if len(tensor_shape_tuple) < expected_minimum_dimension_count:
        raise ValueError(f"{calling_module_name}.forward expected tensor with ≥ {expected_minimum_dimension_count} dims, got shape {tensor_shape_tuple}.")

def assert_last_dimension_has_size(input_tensor: torch.Tensor, expected_dimension_size: int, calling_module_name: str) -> None:
    # Conceptual Idea: 
    # Validate that the final axis (usually the feature axis) of the tensor matches
    # the exact dimension width that the downstream layer's internal parameters expect.
    tensor_shape_tuple = get_tensor_shape(input_tensor, calling_module_name=calling_module_name)
    if len(tensor_shape_tuple) < 1 or tensor_shape_tuple[-1] != expected_dimension_size:
        raise ValueError(f"{calling_module_name}.forward expected last dim == {expected_dimension_size}, got shape {tensor_shape_tuple}.")
        
def assert_valid_dimension_index(input_tensor: torch.Tensor, target_dimension_index: int, calling_module_name: str) -> None:
    """Ensure `target_dimension_index` is a valid axis index for input_tensor (supports negatives)."""
    # Conceptual Idea: 
    # Make sure an operation targeting a specific dimension index (like a reduction
    # or a softmax) lies safely within the mathematical boundaries of the tensor's actual geometry.
    tensor_shape_tuple = get_tensor_shape(input_tensor, calling_module_name=calling_module_name)
    total_dimension_count = len(tensor_shape_tuple)
    absolute_dimension_position = target_dimension_index if target_dimension_index >= 0 else target_dimension_index + total_dimension_count
    if not (0 <= absolute_dimension_position < total_dimension_count):
        raise ValueError(f"{calling_module_name}.forward configured with dim={target_dimension_index}, but input has {total_dimension_count} dims (shape {tensor_shape_tuple}).")

# ########################################################################### #
# Machine Learning Parts Registry:
# ########################################################################### #

COMPONENT_REGISTRIES: Dict[str, Dict[str, Any]] = {
    "architecture": {},
    "builder": {},
    "callback": {},
    "collate": {},
    "config": {},
    "constructor": {},
    "core": {},
    "dataset_splitter": {},
    "head": {},
    "loss": {},
    "metric": {},
    "mixer": {},
    "optimizer": {},
    "preprocessing": {},
    "scheduler": {},
    "search": {},
    "weighting": {},
    "interpreter": {}
}


def register(component_category: str, component_name: str):
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
            class ClassificationHead(neural_network.Module):
                ...

    Notes
    -----
    * Re‑decorating an existing ``kind/name`` pair silently *overwrites* the
      previous entry – handy for hot‑reloading during experiments.
    """
    def registration_decorator(class_or_function_reference):
        # Conceptual Idea: 
        # Map the string identifier directly to the underlying Python class
        # or function, storing it in the global registry dictionary so it can be dynamically 
        # instantiated later using only its given string name from a configuration file.
        COMPONENT_REGISTRIES[component_category][component_name] = class_or_function_reference
        return class_or_function_reference
    return registration_decorator


def retrieve_component_from_registry(component_category: str, component_name: str):
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
    if component_name not in COMPONENT_REGISTRIES[component_category]:
        raise KeyError(f"{component_name} not found in {component_category} registry")
    return COMPONENT_REGISTRIES[component_category][component_name]

retrieve_builder_component          = lambda component_name: retrieve_component_from_registry("builder", component_name)
retrieve_callback_component         = lambda component_name: retrieve_component_from_registry("callback", component_name)
retrieve_collate_component          = lambda component_name: retrieve_component_from_registry("collate", component_name)
retrieve_config_component           = lambda component_name: retrieve_component_from_registry("config", component_name)
retrieve_core_component             = lambda component_name: retrieve_component_from_registry("core", component_name)
retrieve_dataset_splitter_component = lambda component_name: retrieve_component_from_registry("dataset_splitter", component_name)
retrieve_head_component             = lambda component_name: retrieve_component_from_registry("head", component_name)
retrieve_loss_component             = lambda component_name: retrieve_component_from_registry("loss", component_name)
retrieve_metric_component           = lambda component_name: retrieve_component_from_registry("metric", component_name)
retrieve_mixer_component            = lambda component_name: retrieve_component_from_registry("mixer", component_name)
retrieve_optimizer_component        = lambda component_name: retrieve_component_from_registry("optimizer", component_name)
retrieve_preprocessing_component    = lambda component_name: retrieve_component_from_registry("preprocessing", component_name)
retrieve_scheduler_component        = lambda component_name: retrieve_component_from_registry("scheduler", component_name)
retrieve_search_component           = lambda component_name: retrieve_component_from_registry("search", component_name)
retrieve_weighting_component        = lambda component_name: retrieve_component_from_registry("weighting", component_name)
retrieve_interpreter_component        = lambda component_name: retrieve_component_from_registry("interpreter", component_name)

# =============================================================================
# Data Type Conversion:
# =============================================================================


def to_graph(chemical_object):
    return smiles_to_data(chemical_object)

def smiles_to_data(smiles_string: str) -> Data | None:
    
    # Conceptual Idea: 
    # Interpret a text string describing a chemical structure into an RDKit 
    # molecule object. If the string is invalid and building fails, return nothing.
    molecule_object = Chem.MolFromSmiles(smiles_string)
    if molecule_object is None:
        return None

    # Node features (118‑dim one‑hot)
    # Conceptual Idea: 
    # Each atom represents a node in our graph representation. Iterate over 
    # every atom in the molecule, extract its atomic number, and create a one-hot encoded vector
    # describing which element from the periodic table it is. Stack these vectors into a matrix.
    node_features = []
    for chemical_atom in molecule_object.GetAtoms():
        atomic_number = chemical_atom.GetAtomicNum()
        atom_feature_vector = torch.zeros(len(ATOM_TYPES))
        if atomic_number in ATOM_TYPES:
            atom_feature_vector[atomic_number - 1] = 1.0
        node_features.append(atom_feature_vector)
    node_features_tensor_matrix = torch.stack(node_features, dim=0)

    # Edges
    # Conceptual Idea: 
    # Chemical bonds act as edges connecting the atom nodes. Scan all bonds, 
    # extract the indices of the connected atoms, and build a one-hot representation of the bond type.
    # Add edges in both directions (start to end, and end to start) because molecular graphs are undirected.
    edge_index_connections, edge_attributes_features = [], []
    for chemical_bond in molecule_object.GetBonds():
        source_atom_index, destination_atom_index = chemical_bond.GetBeginAtomIdx(), chemical_bond.GetEndAtomIdx()
        bond_type_feature_vector = torch.zeros(len(BOND_TYPES), dtype=torch.float32)  # one‑hot (4,)
        bond_type_feature_vector[BOND_TYPES[chemical_bond.GetBondType()]] = 1.0
        
        edge_index_connections.extend([[source_atom_index, destination_atom_index], [destination_atom_index, source_atom_index]])
        edge_attributes_features.extend([bond_type_feature_vector, bond_type_feature_vector])

    edge_index_tensor = torch.tensor(edge_index_connections, dtype=torch.long).t().contiguous() \
                 if edge_index_connections else torch.empty(2, 0, dtype=torch.long)
                     
    # *** key change: never leave edge_attr as None; use an empty (0, 4) tensor ***
    edge_attributes_tensor = torch.stack(edge_attributes_features, dim=0) if edge_attributes_features \
                 else torch.zeros((0, EDGE_FEATURE_DIMENSION), dtype=torch.float32)

    geometric_data_object = Data(x=node_features_tensor_matrix, edge_index=edge_index_tensor, edge_attr=edge_attributes_tensor)
    geometric_data_object.mol = molecule_object
    return geometric_data_object

_torch_bincount_original = torch.bincount
def mps_safe_bincount(index_tensor, weights=None, minlength=0):
    """torch.bincount returns an invalid, huge buffer on Apple MPS, so run it on CPU when the
    input is on MPS and move the small counts back. A no-op wrapper on CPU/CUDA."""
    if getattr(index_tensor, "device", None) is not None and index_tensor.device.type == "mps":
        counts_on_cpu = _torch_bincount_original(index_tensor.to("cpu"),
                                                 weights=(weights.to("cpu") if weights is not None else None),
                                                 minlength=minlength)
        return counts_on_cpu.to(index_tensor.device)
    return _torch_bincount_original(index_tensor, weights=weights, minlength=minlength)


def to_numerical(input_value):
    """Try to convert input_value into int or float, else return unchanged."""
    # Conceptual Idea: 
    # Determine if an arbitrary value is fundamentally numeric. If it is already 
    # a number, return it as-is. If it is a string representation of a number, safely parse it into 
    # a float. Otherwise, signal failure by returning None.
    if isinstance(input_value, (int, float)):
        return input_value
    if isinstance(input_value, str):
        try:
            return float(input_value)
        except ValueError:
            return None
    else:
        return None
    
    return input_value

def to_integer(input_value):
    # Conceptual Idea: 
    # Try to securely force a data input into an integer type. 
    # If standard casting fails, we assume it might be a floating point string like '3.0' 
    # and perform a sequential two-step cast mapping it through a float first.
    try:
        return int(input_value)
    except Exception:
        try:
            return int(float(input_value))
        except Exception:
            return None

# =========================================================================== #
# Simple Math:
# =========================================================================== #

def multiply_values(first_value, second_value):
    """
    Defined because yaml already uses * as a reserved keyword.
    """
    return first_value * second_value

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
    
    def get_predictions(self) -> Dict[str, torch.Tensor]:           # noqa: D401
        return {}

    # Default behavior is to assume no activations.
    
    def get_activations(self) -> Dict[str, torch.Tensor]:     # noqa: D401
        return {}

    # Returns all parameters that use a gradient => are in the training graph.
    
    def get_learnable_weights(self) -> Dict[str, torch.Tensor]:
        # Default: expose all trainable parameters with their *full* names.
        return {parameter_name: parameter_tensor for parameter_name, parameter_tensor in self.named_parameters() if parameter_tensor.requires_grad}
    
    def get_component_identifier(self) -> str:
        return getattr(self, "name", self.__class__.__name__)

# =========================================================================== #
# Standard Linear and Bias Layers:
# =========================================================================== #

@register("core", "linear")
@register("core", "linear_transformation")
class LinearTransformation(QueryInterface, neural_network.Module):
    """Weight‑only `output = WeightMatrix * input`."""

    def __init__(self, input_dimension: int, output_dimension: int):
        super().__init__()
        
        # Conceptual Idea: 
        # Initialize a learnable weight matrix that will project input feature 
        # vectors into the desired output dimensionality. We apply Kaiming uniform initialization 
        # to ensure signal variance is maintained safely during the forward pass computation.
        self.transformation_weight_matrix = neural_network.Parameter(torch.empty(output_dimension, input_dimension))
        neural_network.init.kaiming_uniform_(self.transformation_weight_matrix, a=math.sqrt(5))

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        
        assert_number_of_dimensions_is(input_tensor, 2, self.get_component_identifier())
        assert_last_dimension_has_size(input_tensor, self.transformation_weight_matrix.size(1), self.get_component_identifier())
        
        # Conceptual Idea: 
        # Perform the linear transformation by calculating the mathematical dot product 
        # between the input tensor and the transposed weight matrix.
        return torch.matmul(input_tensor, self.transformation_weight_matrix.t())


@register("core", "bias")
class BiasLayer(QueryInterface, neural_network.Module):
    """Learnable bias added to last dimension."""

    def __init__(self, output_dimension: int):
        super().__init__()
        
        # Conceptual Idea: 
        # Maintain a simple vector of learnable offsets that will be added 
        # element-wise to the output signals. Initialized entirely to zero by default.
        self.bias_offset_vector = neural_network.Parameter(torch.zeros(output_dimension))
        self.name = "Bias"

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        
        assert_number_of_dimensions_is(input_tensor, 2, self.get_component_identifier())
        assert_last_dimension_has_size(input_tensor, self.bias_offset_vector.size(0), self.get_component_identifier())
        
        return input_tensor + self.bias_offset_vector

@register("core", "affine")
@register("core", "affine_transformation")
class AffineTransformation(QueryInterface, neural_network.Module):
    """Affine transform = Linear ∘ (optional) Bias."""

    def __init__(self, input_dimension: int, output_dimension: int, use_additive_bias: bool = True):
        super().__init__()
        
        # Conceptual Idea: 
        # An affine transformation combines a linear projection and an additive bias.
        # We compose these two fundamental structural sub-layers together sequentially.
        self.linear_projection_layer = LinearTransformation(input_dimension, output_dimension)
        self.bias_addition_layer     = BiasLayer(output_dimension) if use_additive_bias else neural_network.Identity()
        self.name = "Affine_Transformation"

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        
        assert_number_of_dimensions_is(input_tensor, 2, self.get_component_identifier())
        assert_last_dimension_has_size(input_tensor, self.linear_projection_layer.transformation_weight_matrix.size(1), self.get_component_identifier())
        
        return self.bias_addition_layer(self.linear_projection_layer(input_tensor))

# =========================================================================== #
# Standard Regularization Layers:
# =========================================================================== #

@register("core", "layer_norm")
class LayerNormalization(QueryInterface, neural_network.Module):
    
    def __init__(self, expected_feature_dimension: int):
        super().__init__()
        # Conceptual Idea: 
        # Normalize the feature activations independently for each data sample 
        # within the batch, improving mathematical learning stability regardless of batch size.
        self.normalization_layer = neural_network.LayerNorm(expected_feature_dimension)
        self.expected_feature_dimension = expected_feature_dimension
        self.name = "LayerNorm"

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        
        assert_number_of_dimensions_is(input_tensor, 2, self.get_component_identifier())
        assert_last_dimension_has_size(input_tensor, self.expected_feature_dimension, self.get_component_identifier())
        
        return self.normalization_layer(input_tensor)


@register("core", "dropout")
class DropoutLayer(QueryInterface, neural_network.Module):
    
    def __init__(self, active_dropout_probability: float):
        super().__init__()
        
        # Conceptual Idea: 
        # To prevent over-reliance on specific neurons and reduce overfitting,
        # we stochastically zero out a specified fraction of the layer's output activations during training.
        self.dropout_regularization_layer = neural_network.Dropout(active_dropout_probability)
        self.name = "Dropout"

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        
        assert_number_of_dimensions_is(input_tensor, 2, self.get_component_identifier())
        
        return self.dropout_regularization_layer(input_tensor)
  
@register("core", "batch_norm")
class BatchNormalization(QueryInterface, neural_network.Module):

    def __init__(self, expected_feature_dimension: int):
        super().__init__()
        self.normalization_layer = neural_network.BatchNorm1d(expected_feature_dimension)
        self.expected_feature_dimension = expected_feature_dimension
        self.name = "BatchNorm"

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        assert_number_of_dimensions_is(input_tensor, 2, self.name)
        assert_last_dimension_has_size(input_tensor, self.expected_feature_dimension, self.name)

        # A single-sample batch has no within-batch variance, so BatchNorm1d
        # raises in training mode. Normalize that one forward with the running
        # statistics instead (eval-mode behavior), applying the learned affine
        # but leaving the running stats untouched.
        if self.training and input_tensor.size(0) < 2:
            was_training = self.normalization_layer.training
            self.normalization_layer.eval()
            try:
                return self.normalization_layer(input_tensor)
            finally:
                self.normalization_layer.train(was_training)

        return self.normalization_layer(input_tensor)
  
# =========================================================================== #
# Standard Activation Functions:
# =========================================================================== #

@register("core", "identity")
class IdentityActivation(QueryInterface, neural_network.Module):
    def __init__(self):
        super().__init__()
        self.activation_function = neural_network.Identity()
        self.name = "Identity"

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.activation_function(input_tensor)


@register("core", "relu")
class ReLUActivation(QueryInterface, neural_network.Module):
    def __init__(self):
        super().__init__()
        self.activation_function = neural_network.ReLU()
        self.name = "ReLU"

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.activation_function(input_tensor)


@register("core", "leaky_relu")
class LeakyReLUActivation(QueryInterface, neural_network.Module):
    def __init__(self, negative_slope_coefficient: float = 0.01):
        super().__init__()
        self.activation_function = neural_network.LeakyReLU(negative_slope=negative_slope_coefficient)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.activation_function(input_tensor)


@register("core", "prelu")
class PReLUActivation(QueryInterface, neural_network.Module):
    def __init__(self, number_of_parameters: int = 1, initial_value: float = 0.25):
        super().__init__()
        self.activation_function = neural_network.PReLU(num_parameters=number_of_parameters, init=initial_value)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.activation_function(input_tensor)


@register("core", "gelu")
class GELUActivation(QueryInterface, neural_network.Module):
    def __init__(self, approximation_method: str = "none"):  # "none" or "tanh"
    
        super().__init__()
        self.activation_function = neural_network.GELU(approximate=approximation_method)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.activation_function(input_tensor)


@register("core", "silu")
@register("core", "swish")
class SiLUActivation(QueryInterface, neural_network.Module):
    def __init__(self):
        
        super().__init__()
        self.activation_function = neural_network.SiLU()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.activation_function(input_tensor)


@register("core", "elu")
class ELUActivation(QueryInterface, neural_network.Module):
    def __init__(self, alpha_coefficient: float = 1.0):
        super().__init__()
        self.activation_function = neural_network.ELU(alpha=alpha_coefficient)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.activation_function(input_tensor)


@register("core", "selu")
class SELUActivation(QueryInterface, neural_network.Module):
    
    def __init__(self):
        
        super().__init__()
        self.activation_function = neural_network.SELU()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.activation_function(input_tensor)


@register("core", "tanh")
class TanhActivation(QueryInterface, neural_network.Module):
    
    def __init__(self):
        
        super().__init__()
        self.activation_function = neural_network.Tanh()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.activation_function(input_tensor)


@register("core", "sigmoid")
class SigmoidActivation(QueryInterface, neural_network.Module):
    
    def __init__(self):
        
        super().__init__()
        self.activation_function = neural_network.Sigmoid()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.activation_function(input_tensor)


@register("core", "softplus")
class SoftplusActivation(QueryInterface, neural_network.Module):
    def __init__(self, beta_coefficient: float = 1.0, threshold_limit: float = 20.0):
        super().__init__()
        self.activation_function = neural_network.Softplus(beta=beta_coefficient, threshold=threshold_limit)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.activation_function(input_tensor)


@register("core", "softsign")
class SoftsignActivation(QueryInterface, neural_network.Module):
    def __init__(self):
        super().__init__()
        self.activation_function = neural_network.Softsign()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.activation_function(input_tensor)


@register("core", "mish")
class MishActivation(QueryInterface, neural_network.Module):
    def __init__(self):
        super().__init__()
        self.activation_function = neural_network.Mish()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.activation_function(input_tensor)


@register("core", "relu6")
class ReLU6Activation(QueryInterface, neural_network.Module):
    def __init__(self):
        super().__init__()
        self.activation_function = neural_network.ReLU6()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.activation_function(input_tensor)


@register("core", "hardswish")
class HardswishActivation(QueryInterface, neural_network.Module):
    def __init__(self):
        super().__init__()
        self.activation_function = neural_network.Hardswish()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.activation_function(input_tensor)


@register("core", "hardtanh")
class HardtanhActivation(QueryInterface, neural_network.Module):
    def __init__(self, minimum_boundary_value: float = -1.0, maximum_boundary_value: float = 1.0):
        super().__init__()
        self.activation_function = neural_network.Hardtanh(min_val=minimum_boundary_value, max_val=maximum_boundary_value)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.activation_function(input_tensor)


@register("core", "hardsigmoid")
class HardsigmoidActivation(QueryInterface, neural_network.Module):
    def __init__(self):
        super().__init__()
        self.activation_function = neural_network.Hardsigmoid()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.activation_function(input_tensor)


@register("core", "softmax")
class SoftmaxActivation(QueryInterface, neural_network.Module):
    def __init__(self, dimension_index: int = -1):
        super().__init__()
        self.activation_function = neural_network.Softmax(dim=dimension_index)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.activation_function(input_tensor)


@register("core", "log_softmax")
class LogSoftmaxActivation(QueryInterface, neural_network.Module):
    def __init__(self, dimension_index: int = -1):
        super().__init__()
        self.activation_function = neural_network.LogSoftmax(dim=dimension_index)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.activation_function(input_tensor)
    
# =========================================================================== #
# Element-wise Operations:
# =========================================================================== #

@register("core", "elementwise_sum")
class ElementwiseSummation(neural_network.Module):
    def forward(self, *input_tensors_sequence):
        # Conceptual Idea: 
        # Convert arbitrary lists of arrays into PyTorch tensors, stack them along a new leading
        # dimension to form an aggregate batch, and then compute the mathematical sum across that batch dimension.
        return torch.stack([tensor_entry if isinstance(tensor_entry, torch.Tensor) else torch.as_tensor(tensor_entry) for tensor_entry in input_tensors_sequence], 0).sum(0)


@register("core", "elementwise_subtract")
class ElementwiseSubtraction(neural_network.Module):
    def forward(self, *input_tensors_sequence):
        # Conceptual Idea: 
        # Subtract all subsequent tensors sequentially from the very first tensor located in the provided sequence.
        stacked_tensors_matrix = torch.stack([tensor_entry if isinstance(tensor_entry, torch.Tensor) else torch.as_tensor(tensor_entry) for tensor_entry in input_tensors_sequence], 0)
        return stacked_tensors_matrix[0] - stacked_tensors_matrix[1:].sum(0)


@register("core", "elementwise_average")
class ElementwiseAverage(neural_network.Module):
    def forward(self, *input_tensors_sequence):
        # Conceptual Idea: 
        # Compute the point-wise arithmetic mean across an arbitrary sequence of identical-shape tensors.
        return torch.stack([tensor_entry if isinstance(tensor_entry, torch.Tensor) else torch.as_tensor(tensor_entry) for tensor_entry in input_tensors_sequence], 0).mean(0)


@register("core", "elementwise_geometric_mean")
class ElementwiseGeometricMean(neural_network.Module):
    def forward(self, *input_tensors_sequence):
        # Conceptual Idea: 
        # Calculate the geometric mean point-wise. We apply a small clamping threshold
        # to prevent calculating log(0), sum the logarithms (via mean), and then exponentiate.
        return torch.exp(torch.log(torch.clamp(torch.stack([tensor_entry if isinstance(tensor_entry, torch.Tensor) else torch.as_tensor(tensor_entry) for tensor_entry in input_tensors_sequence], 0), min=1e-12)).mean(0))


@register("core", "elementwise_harmonic_mean")
class ElementwiseHarmonicMean(neural_network.Module):
    def forward(self, *input_tensors_sequence):
        # Conceptual Idea: 
        # Calculate the harmonic mean point-wise safely avoiding mathematical zero division singularities.
        stacked_tensors_matrix = torch.stack([tensor_entry if isinstance(tensor_entry, torch.Tensor) else torch.as_tensor(tensor_entry) for tensor_entry in input_tensors_sequence], 0)
        return stacked_tensors_matrix.size(0) / (1 / torch.clamp(stacked_tensors_matrix, min=1e-12)).sum(0)


@register("core", "elementwise_multiply")
class ElementwiseMultiplication(neural_network.Module):
    def forward(self, *input_tensors_sequence):
        # Conceptual Idea: 
        # Perform a point-wise associative multiplication across all provided tensors in the stack.
        return torch.stack([tensor_entry if isinstance(tensor_entry, torch.Tensor) else torch.as_tensor(tensor_entry) for tensor_entry in input_tensors_sequence], 0).prod(0)


@register("core", "elementwise_divide")
class ElementwiseDivision(neural_network.Module):
    def forward(self, numerator_tensor, denominator_tensor):
        # Conceptual Idea: 
        # Apply element-wise mathematical division structurally between exactly two tensors.
        numerator_tensor = numerator_tensor if isinstance(numerator_tensor, torch.Tensor) else torch.as_tensor(numerator_tensor)
        denominator_tensor = denominator_tensor if isinstance(denominator_tensor, torch.Tensor) else torch.as_tensor(denominator_tensor)
        return numerator_tensor / denominator_tensor


@register("core", "elementwise_ordered_choice")
class ElementwiseOrderedChoice(neural_network.Module):
    def __init__(self, selection_rank_position: int = 1):
        super().__init__()
        self.selection_rank_position = selection_rank_position
    def forward(self, *input_tensors_sequence):
        # Conceptual Idea: 
        # Treat the list of identically shaped tensors as an ensemble, and for 
        # each element coordinate position, pick the k-th highest value among the ensemble members.
        return torch.topk(torch.stack([tensor_entry if isinstance(tensor_entry, torch.Tensor) else torch.as_tensor(tensor_entry) for tensor_entry in input_tensors_sequence], 0), self.selection_rank_position, dim=0).values[-1]

# =========================================================================== #
# Feed-Forward Layers:
# =========================================================================== #

@register("core", "feed-forward_layer")
class FeedForwardLayer(neural_network.Module, QueryInterface):
    """
    Dense MLP block with optional BatchNorm, Dropout, and a configurable activation.

    Args:
        input_dimension (int):  Input feature dimension.
        output_dimension (int): Output feature dimension.
        use_batch_normalization (bool): If True, apply neural_network.BatchNorm1d(output_dimension). Default: False.
        dropout_probability (float): If 0.0, no dropout; otherwise p must satisfy 0 < p < 1.
        activation_function_name (str): Name in the "core" registry (e.g. "relu", "gelu", "identity").
        local_layer_name (str): Name used when exposing activations via get_activations().
    """

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        use_batch_normalization: bool = False,
        dropout_probability: float = 0.0,
        activation_function_name: str = DEFAULT_ACTIVATION_FUNCTION,
        use_residual_connection: bool = False,
        local_layer_name: str = "dense",
    ):
        super().__init__()

        # Remember in and out dimensions.
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.use_residual_connection = use_residual_connection
        self.name = local_layer_name

        # A residual add requires matching input and output widths.
        if use_residual_connection and input_dimension != output_dimension:
            raise ValueError(
                f"use_residual_connection=True requires input_dimension == output_dimension, "
                f"got {input_dimension} != {output_dimension}."
            )

        # Linear projection
        # Conceptual Idea:
        # Fetch our standardized affine transformation layer builder from the global component registry.
        self.linear_projection_layer = retrieve_core_component("affine_transformation")(input_dimension, output_dimension)

        # Optional BatchNorm1d
        # Conceptual Idea:
        # Optionally insert a batch normalization layer to stabilize internal parameter gradients.
        self.batch_normalization_layer = retrieve_core_component("batch_norm")(output_dimension) if use_batch_normalization else retrieve_core_component("identity")()

        # Activation (strings only, resolved via registry)
        if not isinstance(activation_function_name, str):
            raise TypeError("activation_function_name must be a registry key (str).")
        self.activation_function_layer = retrieve_core_component(activation_function_name)()

        # Optional Dropout
        # Conceptual Idea:
        # Optionally insert a dropout layer for regularization, randomly scaling the remaining
        # active neurons by their probability of retention to prevent overfitting.
        if float(dropout_probability) == 0.0:
            self.dropout_regularization_layer = retrieve_core_component("identity")()
        else:
            probability_float_value = float(dropout_probability)
            if not (0.0 < probability_float_value < 1.0):
                raise ValueError(f"dropout_probability must be in (0, 1); got {probability_float_value}")
            self.dropout_regularization_layer = retrieve_core_component("dropout")(probability_float_value)

        self.latest_computed_output_tensor: Optional[torch.Tensor] = None
        self.local_layer_name = local_layer_name

    def forward(self, input_features_tensor: torch.Tensor) -> torch.Tensor:

        assert_number_of_dimensions_is(input_features_tensor, 2, self.get_component_identifier())
        assert_last_dimension_has_size(input_features_tensor, self.input_dimension, self.get_component_identifier())

        # Project → (optional BN) → Activation → (optional Dropout)
        # Conceptual Idea:
        # We sequentially pass the incoming data through the standardized order of internal sub-layers.
        # We cache the terminal result so we can expose the post-activation state to the user later if needed.
        latent_features_tensor = self.linear_projection_layer(input_features_tensor)
        latent_features_tensor = self.batch_normalization_layer(latent_features_tensor)
        latent_features_tensor = self.activation_function_layer(latent_features_tensor)
        output_features_tensor = self.dropout_regularization_layer(latent_features_tensor)

        self.latest_computed_output_tensor = output_features_tensor

        # Conceptual Idea:
        # Optionally add the original input features directly to the output to form a skip connection,
        # helping analytical gradients safely bypass non-linearities in deep networks.
        return output_features_tensor + input_features_tensor if self.use_residual_connection else output_features_tensor

    def get_activations(self) -> Dict[str, torch.Tensor]:
        return {self.local_layer_name: self.latest_computed_output_tensor} if self.latest_computed_output_tensor is not None else {}

# =========================================================================== #
# Feed-Forward Neural Networks:
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Standard FFN:
# --------------------------------------------------------------------------- #

@register("core", "ffn")
@register("core", "FFN")
class FeedForwardNeuralNetwork(neural_network.Module, QueryInterface):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimensions: Union[List[int], Tuple[int, ...]],
        use_batch_normalizations: Union[bool, Sequence[bool]] = False,
        dropout_probabilities: Union[float, Sequence[float]] = 0.0,
        activation_functions: Union[str, Sequence[str]] = DEFAULT_ACTIVATION_FUNCTION,
        final_activation_name: Optional[str] = None,
        local_network_name: str = "dense",
        normalize_input: bool = False,
    ):
        super().__init__()

        # Conceptual Idea:
        # Treat the overall network structure logically as a sequential list of dimension widths.
        all_network_dimensions = [input_dimension] + list(hidden_dimensions)
        if len(all_network_dimensions) < 2:
            raise ValueError("dims must contain at least two entries: [input_dimension, ..., output_dimension].")

        number_of_stages = len(hidden_dimensions)
        stage_input_dimensions, stage_output_dimensions = all_network_dimensions[:-1], all_network_dimensions[1:]

        # Conceptual Idea:
        # Expand scalar hyperparameters into a list format so they can be zipped efficiently
        # across all the individual sub-layers in the network sequence evenly.
        use_batch_normalizations = broadcast_val_to(use_batch_normalizations, number_of_stages)
        dropout_probabilities    = broadcast_val_to(dropout_probabilities,    number_of_stages)
        activation_functions     = broadcast_val_to(activation_functions,     number_of_stages)

        # Strict length checks to avoid silent truncation
        if not (len(use_batch_normalizations) == len(dropout_probabilities) == len(activation_functions) == number_of_stages):
            raise ValueError(
                "use_batch_normalizations, dropout_probabilities, and activation_functions must "
                f"all have length {number_of_stages} (or be scalars to broadcast)."
            )

        # Optionally override the activation on the final stage only. Lets a head
        # emit raw logits (final_activation_name="identity") while keeping the
        # default activation on every earlier stage.
        if final_activation_name is not None:
            activation_functions = list(activation_functions)
            activation_functions[-1] = final_activation_name

        # Conceptual Idea:
        # Iteratively build the feed-forward network by appending initialized feed-forward layer
        # objects into a PyTorch ModuleList, precisely matching up the dimensions and configurations for each stage.
        network_stages: List[FeedForwardLayer] = []
        for sequential_stage_index, (stage_input_dimension, stage_output_dimension, use_batch_norm_flag, dropout_probability_value, activation_function_name_string) in enumerate(
            zip(stage_input_dimensions, stage_output_dimensions, use_batch_normalizations, dropout_probabilities, activation_functions), start=1
        ):
            network_stages.append(
                retrieve_core_component("feed-forward_layer")(
                    input_dimension=stage_input_dimension,
                    output_dimension=stage_output_dimension,
                    use_batch_normalization=bool(use_batch_norm_flag),
                    dropout_probability=float(dropout_probability_value),
                    activation_function_name=str(activation_function_name_string),
                    local_layer_name=f"{local_network_name}.stage{sequential_stage_index}",
                )
            )

        self.input_dimension = all_network_dimensions[0]
        self.output_dimension = all_network_dimensions[-1]
        self.network_stages_module = neural_network.ModuleList(network_stages)
        # Optional input normalization. Pretrained embeddings (e.g. MiniMol) arrive un-standardized
        # (all-positive, large common mode), which makes the first Linear converge very slowly.
        # LayerNorm normalizes per-sample with NO running-statistics train/eval mismatch (unlike
        # BatchNorm). No-op (identity) when normalize_input is False.
        self.input_normalization_layer = (
            neural_network.LayerNorm(self.input_dimension) if bool(normalize_input)
            else retrieve_core_component("identity")()
        )

    def forward(self, input_features_tensor: torch.Tensor) -> torch.Tensor:

        assert_number_of_dimensions_is(input_features_tensor, 2, self.get_component_identifier())
        assert_last_dimension_has_size(input_features_tensor, self.input_dimension, self.get_component_identifier())

        # Normalize raw input features first (LayerNorm if enabled, else identity) so an
        # un-standardized embedding does not stall the first Linear.
        input_features_tensor = self.input_normalization_layer(input_features_tensor)

        # Conceptual Idea:
        # Push the data feature vectors sequentially deeper through each instantiated block layer.
        for sequential_network_stage in self.network_stages_module:
            input_features_tensor = sequential_network_stage(input_features_tensor)
        return input_features_tensor

    def get_activations(self) -> Dict[str, torch.Tensor]:
        accumulated_activations_dictionary: Dict[str, torch.Tensor] = {}
        for sequential_network_stage in self.network_stages_module:
            accumulated_activations_dictionary.update(sequential_network_stage.get_activations())
        return accumulated_activations_dictionary

# --------------------------------------------------------------------------- #
# FFN Autoencoder:
# --------------------------------------------------------------------------- #

@register("core", "mlp_autoencoder")
@register("core", "ffn_autoencoder")
class FeedForwardAutoencoder(neural_network.Module, QueryInterface):
    """
    A wrapper around the registered FeedForwardNeuralNetwork ("ffn") that constructs
    an encoder/decoder pair for a deterministic MLP autoencoder.

    Layout (example): input_dimension -> ... -> latent_dimension  ||  latent_dimension -> ... -> input_dimension

    Args:
        input_dimension: input dimensionality (and reconstruction dimensionality).
        shrinkage_factor: compression ratio per stage, 0 < shrinkage_factor < 1 (default: 0.5).
        number_of_compressions: number of shrink steps in the encoder (>= 1).

        # Regularization / activations (applied separately to encoder & decoder):
        use_batch_normalizations: bool or sequence for stages; broadcasted per network.
        dropout_probabilities: float or sequence for stages; broadcasted per network.
        activation_functions: str or sequence; broadcasted per network.
        final_decoder_activation_name: optional str for the **last decoder stage** (default: 'identity').
        use_final_decoder_batch_normalization: bool for the last decoder stage (default: False).
        final_decoder_dropout_probability: float for the last decoder stage (default: 0.0).
        human_readable_model_name: human-readable model name.
        local_network_name: base name for scoping stages in the registry ('{local_network_name}.enc', '{local_network_name}.dec').

    Properties:
        input_dimension (int)
        latent_dimension (int)
        output_dimension (int) == input_dimension
    """

    def __init__(
        self,
        *,
        input_dimension: int,
        shrinkage_factor: float = 0.5,
        number_of_compressions: int = 2,
        use_batch_normalizations: Union[bool, Sequence[bool]] = True,
        dropout_probabilities: Union[float, Sequence[float]] = 0.0,
        activation_functions: Union[str, Sequence[str]] = DEFAULT_ACTIVATION_FUNCTION,
        final_decoder_activation_name: Optional[str] = "identity",
        use_final_decoder_batch_normalization: bool = False,
        final_decoder_dropout_probability: float = 0.0,
        human_readable_model_name: str = "Non-Variational_Autoencoder",
        local_network_name: str = "mlp_autoencoder",
    ):
        super().__init__()

        # --- Validations ---
        # Conceptual Idea: 
        # Ensure mathematical feasibility. We cannot have less than one compression layer, 
        # and the shrinkage factor must strictly reduce the structural dimensions exponentially.
        if number_of_compressions < 1:
            raise ValueError("number_of_compressions must be >= 1")
        if not (0.0 < shrinkage_factor < 1.0):
            raise ValueError("shrinkage_factor must be in (0, 1) when using multiplicative shrinkage.")

        # --- Encoder dimensions ---
        # Use 1..number_of_compressions so the first hidden is actually compressed
        # Conceptual Idea: 
        # Calculate the sizes of each subsequent layer in the encoder pathway by iteratively 
        # multiplying the initial input dimension width by the scalar shrinkage factor.
        encoder_hidden_dimensions: List[int] = [
            max(1, int(input_dimension * (shrinkage_factor ** compression_step_index)))
            for compression_step_index in range(1, number_of_compressions + 1)
        ]
        latent_representation_dimension: int = encoder_hidden_dimensions[-1]

        # --- Decoder dimensions ---
        # Mirror the encoder *excluding* the latent again, then output back to input_dimension.
        # Conceptual Idea: 
        # The decoder path is meant to perfectly symmetrically mirror the encoder. We 
        # reverse the sequence of the encoder's intermediate layers and set the final decoder 
        # output width equal to the original uncompressed input dimensional width.
        decoder_input_dimension: int = latent_representation_dimension
        decoder_hidden_dimensions: List[int] = encoder_hidden_dimensions[:-1][::-1] + [input_dimension]

        # --- Per-stage regs/acts (encoder) ---
        encoder_batch_normalizations  = broadcast_val_to(use_batch_normalizations, len(encoder_hidden_dimensions))
        encoder_dropout_probabilities = broadcast_val_to(dropout_probabilities,    len(encoder_hidden_dimensions))
        encoder_activation_functions  = broadcast_val_to(activation_functions,     len(encoder_hidden_dimensions))

        # --- Per-stage regs/acts (decoder) ---
        decoder_batch_normalizations  = broadcast_val_to(use_batch_normalizations, len(decoder_hidden_dimensions))
        decoder_dropout_probabilities = broadcast_val_to(dropout_probabilities,    len(decoder_hidden_dimensions))
        decoder_activation_functions  = broadcast_val_to(activation_functions,     len(decoder_hidden_dimensions))

        # Tweak the final decoder stage (output layer)
        # Conceptual Idea: 
        # The very last layer that spits out the final reconstruction typically requires
        # fundamentally different properties (e.g. omitting normalization and using a linear identity activation).
        decoder_batch_normalizations[-1]  = use_final_decoder_batch_normalization
        decoder_dropout_probabilities[-1] = final_decoder_dropout_probability
        decoder_activation_functions[-1]  = final_decoder_activation_name

        # --- Build encoder & decoder via the registered FFN ---
        RegisteredFeedForwardNetworkClass = retrieve_core_component("ffn")

        self.encoder_network_module = RegisteredFeedForwardNetworkClass(
            input_dimension=int(input_dimension),
            hidden_dimensions=encoder_hidden_dimensions,
            use_batch_normalizations=encoder_batch_normalizations,
            dropout_probabilities=encoder_dropout_probabilities,
            activation_functions=encoder_activation_functions,
            local_network_name=f"{local_network_name}.enc",
        )

        self.decoder_network_module = RegisteredFeedForwardNetworkClass(
            input_dimension=int(decoder_input_dimension),
            hidden_dimensions=decoder_hidden_dimensions,
            use_batch_normalizations=decoder_batch_normalizations,
            dropout_probabilities=decoder_dropout_probabilities,
            activation_functions=decoder_activation_functions,
            local_network_name=f"{local_network_name}.dec",
        )

        # --- Public properties ---
        self.local_network_name = local_network_name
        self.input_dimension = int(input_dimension)
        self.latent_dimension = int(latent_representation_dimension)
        self.output_dimension = int(input_dimension)
        self.name = human_readable_model_name

    # --- Interface ---

    def get_component_identifier(self) -> str:
        return f"MLPAutoencoder({self.local_network_name}): {self.input_dimension} -> {self.latent_dimension} -> {self.output_dimension}"

    def encode(self, input_features_tensor: torch.Tensor) -> torch.Tensor:
        assert_number_of_dimensions_is(input_features_tensor, 2, self.get_component_identifier())
        assert_last_dimension_has_size(input_features_tensor, self.input_dimension, self.get_component_identifier())
        return self.encoder_network_module(input_features_tensor)

    def decode(self, latent_features_tensor: torch.Tensor) -> torch.Tensor:
        assert_number_of_dimensions_is(latent_features_tensor, 2, self.get_component_identifier())
        assert_last_dimension_has_size(latent_features_tensor, self.latent_dimension, self.get_component_identifier())
        return self.decoder_network_module(latent_features_tensor)

    def forward(self, input_features_tensor: torch.Tensor, return_latent_representation: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Conceptual Idea: 
        # Standard autoencoder execution loop. Pass input features into the compression layer structure
        # to obtain a smaller representational vector, then logically decompress that vector back to its original size.
        latent_features_tensor = self.encode(input_features_tensor)
        reconstructed_features_tensor = self.decode(latent_features_tensor)
        return (reconstructed_features_tensor, latent_features_tensor) if return_latent_representation else reconstructed_features_tensor

    def get_activations(self) -> Dict[str, torch.Tensor]:
        accumulated_activations_dictionary: Dict[str, torch.Tensor] = {}
        if hasattr(self.encoder_network_module, "get_activations"):
            for layer_dictionary_key, layer_dictionary_value in self.encoder_network_module.get_activations().items():
                accumulated_activations_dictionary[f"encoder.{layer_dictionary_key}"] = layer_dictionary_value
        if hasattr(self.decoder_network_module, "get_activations"):
            for layer_dictionary_key, layer_dictionary_value in self.decoder_network_module.get_activations().items():
                accumulated_activations_dictionary[f"decoder.{layer_dictionary_key}"] = layer_dictionary_value
        return accumulated_activations_dictionary

# ########################################################################### #
# Graph Neural Network (GNN) Components:
# ########################################################################### #

# =========================================================================== #
# Structural Encoders:
# =========================================================================== #


def determine_hardware_device(graph_data: Data) -> torch.device:
    # prefer data.x; fall back to edge_index
    # Conceptual Idea: 
    # Safely identify which hardware device (CPU or specific GPU) the graph data is currently residing on
    # so we can initialize new tensors or structural encodings on that exact same hardware automatically.
    if hasattr(graph_data, "x") and isinstance(graph_data.x, torch.Tensor):
        return graph_data.x.device
    return graph_data.edge_index.device

def calculate_batch_node_index_ranges(batch_data: Batch) -> Sequence[Tuple[int, int]]:
    """
    For a PyG Batch, return node index slices [(start_0, end_0), (start_1, end_1), ...]
    using Batch.ptr when available; falls back to bincount otherwise.
    """
    # Conceptual Idea: 
    # In PyTorch Geometric, multiple graphs are packed into a single giant disconnected "batch graph". 
    # This helper isolates the start and end node indices for every individual graph within that batch 
    # so we can process node-level spectral features (like Laplacians) on a per-graph basis.
    if hasattr(batch_data, "ptr") and batch_data.ptr is not None:
        batch_pointers = batch_data.ptr
        return [(int(batch_pointers[node_index].item()), int(batch_pointers[node_index + 1].item())) for node_index in range(batch_pointers.numel() - 1)]

    # fallback: contiguous blocks per graph id
    batch_assignments = batch_data.batch
    graph_node_counts = mps_safe_bincount(batch_assignments)
    cumulative_node_counts = torch.cumsum(graph_node_counts, dim=0)
    starting_indices = torch.cat([torch.zeros(1, dtype=torch.long, device=batch_assignments.device), cumulative_node_counts[:-1]])
    return [(int(start_index.item()), int(end_index.item())) for start_index, end_index in zip(starting_indices, cumulative_node_counts)]

def extract_subgraph_edge_indices(global_edge_indices: torch.Tensor, start_node_index: int, end_node_index: int) -> torch.Tensor:
    """Return a 2×E' edge_index for the subgraph with node ids in [s, e)."""
    # Conceptual Idea: 
    # Given the edges for an entire batch, extract only the edges belonging to a single specific graph, 
    # and re-index them so they range locally from 0 to N instead of the global batch index range.
    source_nodes, destination_nodes = global_edge_indices
    subgraph_mask = (source_nodes >= start_node_index) & (source_nodes < end_node_index) & (destination_nodes >= start_node_index) & (destination_nodes < end_node_index)
    if not subgraph_mask.any():
        return torch.empty(2, 0, dtype=torch.long, device=global_edge_indices.device)
    local_source_nodes = source_nodes[subgraph_mask] - start_node_index
    local_destination_nodes = destination_nodes[subgraph_mask] - start_node_index
    return torch.stack([local_source_nodes, local_destination_nodes], dim=0)

def construct_dense_adjacency_matrix(edge_indices: torch.Tensor, number_of_nodes: int, *, make_undirected: bool = True) -> torch.Tensor:
    """
    Build a dense 0/1 adjacency matrix A (n×n) from edge_index.
    If the input already contains both directions, this is still safe.
    """
    # Conceptual Idea: 
    # Convert a sparse edge list into a dense mathematical adjacency matrix representation 
    # required for matrix operations like eigenvalue decomposition or computing random walk transitions.
    adjacency_matrix = torch.zeros((number_of_nodes, number_of_nodes), dtype=torch.float32, device=edge_indices.device)
    if edge_indices.numel() == 0:
        return adjacency_matrix
    row_indices, column_indices = edge_indices
    adjacency_matrix[row_indices, column_indices] = 1.0
    if make_undirected:
        adjacency_matrix[column_indices, row_indices] = 1.0
    adjacency_matrix.clamp_(0.0, 1.0)  # in case of duplicates
    return adjacency_matrix

def concatenate_feature_blocks(sequence_of_feature_blocks: Sequence[torch.Tensor]) -> torch.Tensor:
    """Concatenate per-graph feature blocks (along node axis)."""
    # Conceptual Idea: 
    # After calculating structural features on a per-graph basis, stitch them all 
    # back together into a single master batch tensor so standard batch processing can continue.
    if len(sequence_of_feature_blocks) == 0:
        return torch.zeros(0, 0)
    return torch.cat(sequence_of_feature_blocks, dim=0)

@register("core", "degree_features")
@torch.no_grad()
def degree_features(graph_data: Data, *, normalization_scheme: Optional[str] = None) -> torch.Tensor:
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
    hardware_device = determine_hardware_device(graph_data)
    
    # Conceptual Idea: 
    # Calculate how many edges connect to each node. We optionally apply normalizations 
    # (like max or L2) on a *per-graph* basis so that small and large molecules have comparable scale signals.
    if isinstance(graph_data, Batch):
        batch_assignments = graph_data.batch
        node_degrees = torch_geometric_degree(graph_data.edge_index[0], graph_data.num_nodes).to(hardware_device)
        output_degree_tensor = node_degrees.unsqueeze(-1)
        if normalization_scheme == "max":
            graph_sizes = mps_safe_bincount(batch_assignments)
            normalized_output = []
            current_start_index = 0
            for current_graph_size in graph_sizes.tolist():
                graph_specific_block = output_degree_tensor[current_start_index : current_start_index + current_graph_size]
                maximum_degree_value = graph_specific_block.max().clamp_min(1.0)
                normalized_output.append(graph_specific_block / maximum_degree_value)
                current_start_index += current_graph_size
            output_degree_tensor = torch.cat(normalized_output, dim=0)
        elif normalization_scheme == "log":
            output_degree_tensor = torch.log1p(output_degree_tensor)
        elif normalization_scheme == "l2":
            graph_sizes = mps_safe_bincount(batch_assignments)
            normalized_output, current_start_index = [], 0
            for current_graph_size in graph_sizes.tolist():
                graph_specific_block = output_degree_tensor[current_start_index : current_start_index + current_graph_size]
                l2_norm_denominator = torch.linalg.norm(graph_specific_block) + 1e-8
                normalized_output.append(graph_specific_block / l2_norm_denominator)
                current_start_index += current_graph_size
            output_degree_tensor = torch.cat(normalized_output, dim=0)
        return output_degree_tensor

    # single graph
    node_degrees = torch_geometric_degree(graph_data.edge_index[0], graph_data.num_nodes).to(hardware_device).unsqueeze(-1)
    if normalization_scheme == "max":
        node_degrees = node_degrees / node_degrees.max().clamp_min(1.0)
    elif normalization_scheme == "log":
        node_degrees = torch.log1p(node_degrees)
    elif normalization_scheme == "l2":
        node_degrees = node_degrees / (torch.linalg.norm(node_degrees) + 1e-8)
    return node_degrees


@register("core", "laplacian_positional_encoding")
@torch.no_grad()
def laplacian_positional_encoding(
    graph_data: Data,
    target_eigenvector_count: int = 8,
    *,
    normalization_type: str = "sym",        # {"sym", "rw"}
    exclude_trivial_eigenvector: bool = True,
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
    def compute_laplacian_positional_encoding_for_single_graph(dense_adjacency_matrix: torch.Tensor, requested_k_eigenvectors: int) -> torch.Tensor:
        # Conceptual Idea: 
        # Convert the structural topology into a normalized Laplacian matrix, which acts 
        # like a wave-equation operator on the graph. By extracting its smallest eigenvectors, 
        # we assign every node a coordinate in a continuous spectral layout space, helping the GNN 
        # understand global structure and distance geometry.
        number_of_nodes = dense_adjacency_matrix.size(0)
        if number_of_nodes == 0:
            return torch.zeros(0, requested_k_eigenvectors, dtype=torch.float32)
        # degree
        degree_vector = dense_adjacency_matrix.sum(dim=1)
        identity_matrix = torch.eye(number_of_nodes, dtype=torch.float32, device=dense_adjacency_matrix.device)
        
        if normalization_type == "sym":
            inverse_square_root_degrees = torch.where(degree_vector > 0, degree_vector.pow(-0.5), torch.zeros_like(degree_vector))
            inverse_degree_matrix = torch.diag(inverse_square_root_degrees)
            laplacian_matrix = identity_matrix - inverse_degree_matrix @ dense_adjacency_matrix @ inverse_degree_matrix
        elif normalization_type == "rw":
            inverse_degrees = torch.where(degree_vector > 0, degree_vector.reciprocal(), torch.zeros_like(degree_vector))
            inverse_degree_diagonal_matrix = torch.diag(inverse_degrees)
            laplacian_matrix = identity_matrix - inverse_degree_diagonal_matrix @ dense_adjacency_matrix
        else:
            raise ValueError("normalization_type must be 'sym' or 'rw'")

        # eigh is more stable on CPU for very small matrices
        cpu_laplacian_matrix = laplacian_matrix.detach().cpu()
        eigenvalues, eigenvectors = torch.linalg.eigh(cpu_laplacian_matrix)   # ascending
        
        # drop the trivial eigenvector (λ≈0) if requested
        start_index = 1 if exclude_trivial_eigenvector and number_of_nodes > 1 else 0
        selected_eigenvectors = eigenvectors[:, start_index : start_index + requested_k_eigenvectors]          # (n, <=k)
        
        # pad to k
        if selected_eigenvectors.size(1) < requested_k_eigenvectors:
            zero_padding_matrix = torch.zeros(number_of_nodes, requested_k_eigenvectors - selected_eigenvectors.size(1), dtype=torch.float32, device=cpu_laplacian_matrix.device)
            selected_eigenvectors = torch.cat([selected_eigenvectors, zero_padding_matrix], dim=1)
            
        # sign fix
        sign_multiplier = torch.sign(selected_eigenvectors.sum(dim=0, keepdim=True)).clamp(min=1.0)
        selected_eigenvectors = selected_eigenvectors * sign_multiplier
        return selected_eigenvectors.to(dense_adjacency_matrix.device)

    hardware_device = determine_hardware_device(graph_data)
    if isinstance(graph_data, Batch):
        feature_blocks = []
        for start_node_index, end_node_index in calculate_batch_node_index_ranges(graph_data):
            node_count = end_node_index - start_node_index
            subgraph_edge_indices = extract_subgraph_edge_indices(graph_data.edge_index, start_node_index, end_node_index)
            dense_adjacency = construct_dense_adjacency_matrix(subgraph_edge_indices, node_count)
            feature_blocks.append(compute_laplacian_positional_encoding_for_single_graph(dense_adjacency, target_eigenvector_count))
        return concatenate_feature_blocks(feature_blocks).to(hardware_device)

    # single graph
    dense_adjacency = construct_dense_adjacency_matrix(graph_data.edge_index, graph_data.num_nodes)
    return compute_laplacian_positional_encoding_for_single_graph(dense_adjacency, target_eigenvector_count).to(hardware_device)


@register("core", "random_walk_structural_encoding")
@torch.no_grad()
def random_walk_structural_encoding(
    graph_data: Data,
    walk_length_steps: Sequence[int] = (1, 2, 3, 4, 5, 6, 7, 8),
) -> torch.Tensor:
    """
    Random Walk Structural Encodings (RWSE / RW-Landing probabilities).

    For each node i and step t, feature is  (P^t)_{ii}  where
    P = D^{-1} A is the random-walk transition matrix.

    Returns
    -------
    Tensor  shape (N, len(walk_lengths))
    """
    def compute_random_walk_for_single_graph(dense_adjacency_matrix: torch.Tensor, target_walk_steps: Sequence[int]) -> torch.Tensor:
        # Conceptual Idea: 
        # Capture the local topology around a node by calculating the probability that a random walker 
        # starting at that node returns exactly back to it after `t` steps. We iterate matrix multiplication
        # of the transition probabilities and extract the diagonal elements at each requested step.
        number_of_nodes = dense_adjacency_matrix.size(0)
        if number_of_nodes == 0:
            return torch.zeros(0, len(target_walk_steps), dtype=torch.float32)
        degree_vector = dense_adjacency_matrix.sum(dim=1)
        inverse_degrees = torch.where(degree_vector > 0, degree_vector.reciprocal(), torch.zeros_like(degree_vector))
        transition_probability_matrix = torch.diag(inverse_degrees) @ dense_adjacency_matrix
        
        # iterative powers
        structural_features = []
        current_transition_power_matrix = transition_probability_matrix.clone()
        if len(target_walk_steps) == 0:
            return torch.zeros(number_of_nodes, 0, dtype=torch.float32, device=dense_adjacency_matrix.device)
            
        maximum_walk_steps = max(target_walk_steps)
        diagonal_elements_dictionary = {}
        
        for time_step in range(1, maximum_walk_steps + 1):
            if time_step == 1:
                current_transition_power_matrix = transition_probability_matrix
            else:
                current_transition_power_matrix = current_transition_power_matrix @ transition_probability_matrix
            diagonal_elements_dictionary[time_step] = torch.diag(current_transition_power_matrix)
            
        for target_step in target_walk_steps:
            if target_step == 0:
                structural_features.append(torch.ones(number_of_nodes, device=dense_adjacency_matrix.device))   # (P^0)_{ii} = 1
            else:
                structural_features.append(diagonal_elements_dictionary[target_step])
        return torch.stack(structural_features, dim=1).float()  # (n, T)

    hardware_device = determine_hardware_device(graph_data)
    if isinstance(graph_data, Batch):
        feature_blocks = []
        for start_node_index, end_node_index in calculate_batch_node_index_ranges(graph_data):
            node_count = end_node_index - start_node_index
            subgraph_edge_indices = extract_subgraph_edge_indices(graph_data.edge_index, start_node_index, end_node_index)
            dense_adjacency = construct_dense_adjacency_matrix(subgraph_edge_indices, node_count)
            feature_blocks.append(compute_random_walk_for_single_graph(dense_adjacency, walk_length_steps))
        return concatenate_feature_blocks(feature_blocks).to(hardware_device)
        
    dense_adjacency = construct_dense_adjacency_matrix(graph_data.edge_index, graph_data.num_nodes)
    return compute_random_walk_for_single_graph(dense_adjacency, walk_length_steps).to(hardware_device)


@register("core", "closeness_centrality")
@torch.no_grad()
def closeness_centrality(graph_data: Data) -> torch.Tensor:
    """
    Closeness centrality per node:  (|C(i)| - 1) / sum_{j in C(i), j≠i} d(i,j)
    where C(i) is the connected component of node i.
    Uses Floyd–Warshall (O(N^3)) — fine for molecules.

    Returns
    -------
    Tensor  shape (N, 1)
    """
    def compute_closeness_for_single_graph(dense_adjacency_matrix: torch.Tensor) -> torch.Tensor:
        # Conceptual Idea: 
        # Evaluate how "central" a node is by calculating its average shortest path distance 
        # to all other nodes. We use the all-pairs shortest path Floyd-Warshall algorithm.
        number_of_nodes = dense_adjacency_matrix.size(0)
        if number_of_nodes <= 1:
            return torch.zeros(number_of_nodes, 1, device=dense_adjacency_matrix.device)
            
        infinite_distance_value = torch.tensor(float("inf"), device=dense_adjacency_matrix.device)
        distance_matrix = torch.full((number_of_nodes, number_of_nodes), infinite_distance_value, device=dense_adjacency_matrix.device)
        distance_matrix.fill_diagonal_(0.0)
        
        non_zero_edges_mask = (dense_adjacency_matrix > 0)
        distance_matrix[non_zero_edges_mask] = 1.0
        
        # Floyd–Warshall
        for intermediate_node_index in range(number_of_nodes):
            distance_matrix = torch.minimum(distance_matrix, distance_matrix[:, intermediate_node_index:intermediate_node_index+1] + distance_matrix[intermediate_node_index:intermediate_node_index+1, :])
            
        # per-node
        reachable_nodes_mask = (distance_matrix < infinite_distance_value).float()
        component_sizes_vector = reachable_nodes_mask.sum(dim=1)  # includes self
        sum_of_distances_vector = (distance_matrix * reachable_nodes_mask).sum(dim=1) - torch.diag(distance_matrix)  # exclude self (0)
        safe_denominator_vector = torch.where(sum_of_distances_vector > 0, sum_of_distances_vector, torch.ones_like(sum_of_distances_vector))
        
        closeness_scores_vector = (component_sizes_vector - 1.0) / safe_denominator_vector
        return closeness_scores_vector.unsqueeze(-1)

    hardware_device = determine_hardware_device(graph_data)
    if isinstance(graph_data, Batch):
        feature_blocks = []
        for start_node_index, end_node_index in calculate_batch_node_index_ranges(graph_data):
            node_count = end_node_index - start_node_index
            subgraph_edge_indices = extract_subgraph_edge_indices(graph_data.edge_index, start_node_index, end_node_index)
            dense_adjacency = construct_dense_adjacency_matrix(subgraph_edge_indices, node_count)
            feature_blocks.append(compute_closeness_for_single_graph(dense_adjacency))
        return concatenate_feature_blocks(feature_blocks).to(hardware_device)
        
    dense_adjacency = construct_dense_adjacency_matrix(graph_data.edge_index, graph_data.num_nodes)
    return compute_closeness_for_single_graph(dense_adjacency).to(hardware_device)


@register("core", "eccentricity")
@torch.no_grad()
def eccentricity(graph_data: Data) -> torch.Tensor:
    """
    Eccentricity per node:  max_j d(i, j)  within the connected component.

    Returns
    -------
    Tensor  shape (N, 1)
    """
    def compute_eccentricity_for_single_graph(dense_adjacency_matrix: torch.Tensor) -> torch.Tensor:
        # Conceptual Idea: 
        # For each node, find the longest shortest-path to any other node. 
        # This tells us how "far away" the node is from the deepest fringes of the network.
        number_of_nodes = dense_adjacency_matrix.size(0)
        if number_of_nodes <= 1:
            return torch.zeros(number_of_nodes, 1, device=dense_adjacency_matrix.device)
            
        infinite_distance_value = torch.tensor(float("inf"), device=dense_adjacency_matrix.device)
        distance_matrix = torch.full((number_of_nodes, number_of_nodes), infinite_distance_value, device=dense_adjacency_matrix.device)
        distance_matrix.fill_diagonal_(0.0)
        
        non_zero_edges_mask = (dense_adjacency_matrix > 0)
        distance_matrix[non_zero_edges_mask] = 1.0
        
        for intermediate_node_index in range(number_of_nodes):
            distance_matrix = torch.minimum(distance_matrix, distance_matrix[:, intermediate_node_index:intermediate_node_index+1] + distance_matrix[intermediate_node_index:intermediate_node_index+1, :])
            
        distance_matrix[distance_matrix == infinite_distance_value] = 0.0  # ignore unreachable nodes (other components)
        eccentricity_scores_vector = distance_matrix.max(dim=1).values
        return eccentricity_scores_vector.unsqueeze(-1)

    hardware_device = determine_hardware_device(graph_data)
    if isinstance(graph_data, Batch):
        feature_blocks = []
        for start_node_index, end_node_index in calculate_batch_node_index_ranges(graph_data):
            node_count = end_node_index - start_node_index
            subgraph_edge_indices = extract_subgraph_edge_indices(graph_data.edge_index, start_node_index, end_node_index)
            dense_adjacency = construct_dense_adjacency_matrix(subgraph_edge_indices, node_count)
            feature_blocks.append(compute_eccentricity_for_single_graph(dense_adjacency))
        return concatenate_feature_blocks(feature_blocks).to(hardware_device)
        
    dense_adjacency = construct_dense_adjacency_matrix(graph_data.edge_index, graph_data.num_nodes)
    return compute_eccentricity_for_single_graph(dense_adjacency).to(hardware_device)


@register("core", "local_clustering_coefficient")
@torch.no_grad()
def local_clustering_coefficient(graph_data: Data) -> torch.Tensor:
    """
    Local clustering coefficient per node:
        C_i = 2 * triangles(i) / (d_i * (d_i - 1)),  undirected case.

    Returns
    -------
    Tensor  shape (N, 1)
    """
    def compute_clustering_for_single_graph(dense_adjacency_matrix: torch.Tensor) -> torch.Tensor:
        # Conceptual Idea: 
        # Evaluate graph cliquishness. For each node, check what fraction of its direct neighbors 
        # are also neighbors with each other. We use matrix multiplication cubed to find topological triangles.
        number_of_nodes = dense_adjacency_matrix.size(0)
        if number_of_nodes <= 2:
            return torch.zeros(number_of_nodes, 1, device=dense_adjacency_matrix.device)
            
        degree_vector = dense_adjacency_matrix.sum(dim=1)
        # A^3 diagonal gives 2 * triangles for undirected simple graphs
        adjacency_matrix_cubed = (dense_adjacency_matrix @ dense_adjacency_matrix @ dense_adjacency_matrix)
        twice_triangle_count_vector = torch.diag(adjacency_matrix_cubed)                      # equals 2 * triangles(i)
        
        safe_denominator = degree_vector * (degree_vector - 1.0) + 1e-8
        clustering_coefficients = twice_triangle_count_vector / safe_denominator
        return clustering_coefficients.unsqueeze(-1).clamp_(0.0, 1.0)

    hardware_device = determine_hardware_device(graph_data)
    if isinstance(graph_data, Batch):
        feature_blocks = []
        for start_node_index, end_node_index in calculate_batch_node_index_ranges(graph_data):
            node_count = end_node_index - start_node_index
            subgraph_edge_indices = extract_subgraph_edge_indices(graph_data.edge_index, start_node_index, end_node_index)
            dense_adjacency = construct_dense_adjacency_matrix(subgraph_edge_indices, node_count)
            feature_blocks.append(compute_clustering_for_single_graph(dense_adjacency))
        return concatenate_feature_blocks(feature_blocks).to(hardware_device)
        
    dense_adjacency = construct_dense_adjacency_matrix(graph_data.edge_index, graph_data.num_nodes)
    return compute_clustering_for_single_graph(dense_adjacency).to(hardware_device)


@register("core", "eigenvector_centrality")
@torch.no_grad()
def eigenvector_centrality(
    graph_data: Data,
    *,
    maximum_iterations: int = 100,
    tolerance_threshold: float = 1e-6,
) -> torch.Tensor:
    """
    Eigenvector centrality via power iteration on the (undirected) adjacency.

    Returns
    -------
    Tensor  shape (N, 1)
    """
    def compute_eigenvector_centrality_for_single_graph(dense_adjacency_matrix: torch.Tensor) -> torch.Tensor:
        # Conceptual Idea: 
        # A node's centrality depends on the centrality of its neighbors. We use 
        # the power iteration method to find the primary eigenvector of the adjacency matrix.
        number_of_nodes = dense_adjacency_matrix.size(0)
        if number_of_nodes == 0:
            return torch.zeros(0, 1, dtype=torch.float32, device=dense_adjacency_matrix.device)
        if number_of_nodes == 1:
            return torch.ones(1, 1, dtype=torch.float32, device=dense_adjacency_matrix.device)
            
        centrality_scores_vector = torch.ones(number_of_nodes, 1, dtype=torch.float32, device=dense_adjacency_matrix.device) / number_of_nodes
        
        for iteration_step in range(maximum_iterations):
            next_centrality_scores_vector = dense_adjacency_matrix @ centrality_scores_vector
            vector_norm_value = torch.linalg.vector_norm(next_centrality_scores_vector) + 1e-12
            next_centrality_scores_vector = next_centrality_scores_vector / vector_norm_value
            
            if torch.max(torch.abs(next_centrality_scores_vector - centrality_scores_vector)) < tolerance_threshold:
                centrality_scores_vector = next_centrality_scores_vector
                break
            centrality_scores_vector = next_centrality_scores_vector
            
        # normalise to [0, 1]
        centrality_scores_vector = centrality_scores_vector / (centrality_scores_vector.max().clamp_min(1e-12))
        return centrality_scores_vector

    hardware_device = determine_hardware_device(graph_data)
    if isinstance(graph_data, Batch):
        feature_blocks = []
        for start_node_index, end_node_index in calculate_batch_node_index_ranges(graph_data):
            node_count = end_node_index - start_node_index
            subgraph_edge_indices = extract_subgraph_edge_indices(graph_data.edge_index, start_node_index, end_node_index)
            dense_adjacency = construct_dense_adjacency_matrix(subgraph_edge_indices, node_count)
            feature_blocks.append(compute_eigenvector_centrality_for_single_graph(dense_adjacency))
        return concatenate_feature_blocks(feature_blocks).to(hardware_device)
        
    dense_adjacency = construct_dense_adjacency_matrix(graph_data.edge_index, graph_data.num_nodes)
    return compute_eigenvector_centrality_for_single_graph(dense_adjacency).to(hardware_device)


@register("core", "k_core_number")
@torch.no_grad()
def k_core_number(graph_data: Data) -> torch.Tensor:
    """
    k-core number (coreness) for each node via degeneracy ordering.
    Simple O(N^2) implementation – perfectly fine for small molecules.

    Returns
    -------
    Tensor  shape (N, 1)
    """
    def convert_edge_index_to_adjacency(edge_indices: torch.Tensor, number_of_nodes: int) -> Dict[int, set]:
        # Conceptual Idea: 
        # Map out the connections for rapid pruning. Converting edge indices to sets allows 
        # fast O(1) removal of neighbors when pruning the graph shell-by-shell.
        adjacencys: Dict[int, set] = {node_index: set() for node_index in range(number_of_nodes)}
        if edge_indices.numel() == 0:
            return adjacencys
        source_nodes, destination_nodes = edge_indices
        for source_node, destination_node in zip(source_nodes.tolist(), destination_nodes.tolist()):
            adjacencys[source_node].add(destination_node)
            adjacencys[destination_node].add(source_node)
        return adjacencys

    def compute_k_core_for_single_graph(edge_indices: torch.Tensor, number_of_nodes: int) -> torch.Tensor:
        # Conceptual Idea: 
        # Successively prune the lowest-degree node from the graph. The degree of that node 
        # exactly at the moment it is removed is its coreness.
        if number_of_nodes == 0:
            return torch.zeros(0, 1, dtype=torch.float32, device=edge_indices.device)
            
        adjacency = convert_edge_index_to_adjacency(edge_indices, number_of_nodes)
        node_degrees_dictionary = {node_index: len(adjacency[node_index]) for node_index in range(number_of_nodes)}
        coreness_values = [0] * number_of_nodes
        remaining_nodes_set = set(range(number_of_nodes))
        
        while remaining_nodes_set:
            lowest_degree_node = min(remaining_nodes_set, key=lambda node_index: node_degrees_dictionary[node_index])
            coreness_values[lowest_degree_node] = node_degrees_dictionary[lowest_degree_node]
            remaining_nodes_set.remove(lowest_degree_node)
            for connected_neighbor in list(adjacency[lowest_degree_node]):
                if connected_neighbor in remaining_nodes_set:
                    adjacency[connected_neighbor].discard(lowest_degree_node)
                    node_degrees_dictionary[connected_neighbor] -= 1
            adjacency[lowest_degree_node].clear()
            
        return torch.tensor(coreness_values, dtype=torch.float32, device=edge_indices.device).unsqueeze(-1)

    hardware_device = determine_hardware_device(graph_data)
    if isinstance(graph_data, Batch):
        feature_blocks = []
        for start_node_index, end_node_index in calculate_batch_node_index_ranges(graph_data):
            node_count = end_node_index - start_node_index
            subgraph_edge_indices = extract_subgraph_edge_indices(graph_data.edge_index, start_node_index, end_node_index)
            feature_blocks.append(compute_k_core_for_single_graph(subgraph_edge_indices, node_count))
        return concatenate_feature_blocks(feature_blocks).to(hardware_device)
        
    return compute_k_core_for_single_graph(graph_data.edge_index, graph_data.num_nodes).to(hardware_device)

@register("core", "structural_encoder")
@torch.no_grad()
def structural_encoder(list_of_encoder_names: Sequence[str]) -> Callable[[Data], torch.Tensor]:
    """
    Given a list of registry keys (strings), look each one up with `retrieve_core_component`,
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
    if not isinstance(list_of_encoder_names, (list, tuple)):
        raise TypeError("`list_of_encoder_names` must be a list/tuple of registry keys (str).")

    # Resolve (and validate) encoder functions from the 'core' registry
    # Conceptual Idea: 
    # Dynamically locate the requested mathematical descriptor functions from the component registry
    # and bundle them together so they can be executed seamlessly in sequence.
    encoder_callable_functions: List[Callable[[Data], torch.Tensor]] = []
    for encoder_name in list_of_encoder_names:
        encoder_function = retrieve_core_component(encoder_name)  # may return classes or functions; we expect callables
        if not callable(encoder_function):
            raise TypeError(f"Registry entry 'core.{encoder_name}' is not callable.")
        encoder_callable_functions.append(encoder_function)

    def compute_combined_structural_encodings(graph_data_object: Data) -> torch.Tensor:
        # Conceptual Idea: 
        # Execute each individual topology feature extractor independently on the graph object, 
        # and stack their resulting feature representations horizontally along the node dimension.
        encoder_outputs = [encoder_function(graph_data_object) for encoder_function in encoder_callable_functions]
        if not encoder_outputs:
            hardware_device = determine_hardware_device(graph_data_object)
            return torch.empty(0, 0, device=hardware_device)
        total_number_of_nodes = encoder_outputs[0].size(0)
        
        if any(tensor_output.size(0) != total_number_of_nodes for tensor_output in encoder_outputs):
            raise ValueError("All encoders must return tensors with the same number of rows (nodes).")
        return torch.cat(encoder_outputs, dim=-1)

    return compute_combined_structural_encodings

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

SCATTER_OPERATIONS = {
    "sum":  lambda source_features, index_pointers, dimension_size: scatter(source_features, index_pointers, dim=0, dim_size=dimension_size, reduce="sum"),
    "mean": lambda source_features, index_pointers, dimension_size: scatter(source_features, index_pointers, dim=0, dim_size=dimension_size, reduce="mean"),
    "max":  lambda source_features, index_pointers, dimension_size: scatter(source_features, index_pointers, dim=0, dim_size=dimension_size, reduce="max"),
}

# --------------------------------------------------------------------------- #
# Updating Data Object Properties:
# --------------------------------------------------------------------------- #

def get_node_embeddings(graph_data):
    # Conceptual Idea: 
    # Encapsulate access to PyTorch Geometric's standard attribute "x" 
    # for cleaner, intention-revealing variable access across stages.
    return graph_data.x

def get_edge_embeddings(graph_data):
    return graph_data.edge_attr

def get_structural_embeddings(graph_data):
    return graph_data.pos_enc

def update_node_embeddings(graph_data, updated_features):
    graph_data.x = updated_features
    return graph_data

def update_edge_embeddings(graph_data, updated_features):
    graph_data.edge_attr = updated_features
    return graph_data

def update_structural_embeddings(graph_data, updated_features):
    graph_data.pos_enc = updated_features
    return graph_data

def get_source_and_destination_nodes(graph_data):
    source_nodes, destination_nodes = graph_data.edge_index
    return source_nodes, destination_nodes

# =========================================================================== #
# Stage 1 - Graph Preprocessing:
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Graph Processor Module:
# --------------------------------------------------------------------------- #

class GraphPreprocessor(neural_network.Module):
    covers = ("preprocessor",)
    output_node_dimension: Optional[int] = None
    output_edge_dimension: Optional[int] = None

    def forward(self, graph_data: Data) -> Data:
        # Conceptual Idea: 
        # Normalize incoming graph data formats by ensuring edge attributes and batch 
        # indices are securely populated, preventing downstream null-reference exceptions.
        if getattr(graph_data, "edge_attr", None) is None:
            number_of_edges = graph_data.edge_index.size(1)
            graph_data.edge_attr = graph_data.x.new_zeros(number_of_edges, 1)
        if graph_data.x.dim() == 1:
            graph_data.x = graph_data.x.unsqueeze(-1)
        if not hasattr(graph_data, "batch"):
            graph_data.batch = graph_data.x.new_zeros(graph_data.num_nodes, dtype=torch.long)
        return graph_data
    
# --------------------------------------------------------------------------- #
# Universal Node Graph PreProcessor:
# --------------------------------------------------------------------------- #


# class GlobalNodeGraphPreprocessor(neural_network.Module):
#     covers = ("preprocessor",)
#     output_node_dimension: Optional[int] = None
#     output_edge_dimension: Optional[int] = None

#     def forward(self, graph_data: Data) -> Data:
        
#         # Conceptual Idea: 
#         # First ensure standard attributes exist. Then inject an artificial "super node" 
#         # connected universally to every single real node in the graph, serving as a global information sink/source.
#         if getattr(graph_data, "edge_attr", None) is None: 
#             graph_data.edge_attr = graph_data.x.new_zeros(graph_data.edge_index.size(1), 1)
        
#         if graph_data.x.dim() == 1: 
#             graph_data.x = graph_data.x.unsqueeze(-1)
        
#         if not hasattr(graph_data, "batch"): 
#             graph_data.batch = graph_data.x.new_zeros(graph_data.num_nodes, dtype=torch.long)
        
#         if getattr(graph_data, "has_global_node", False): 
#             return graph_data
        
#         original_node_count = graph_data.num_nodes
#         node_feature_dimension = graph_data.x.size(1)
        
#         graph_data.x = torch.cat([graph_data.x, graph_data.x.new_zeros(1, node_feature_dimension)], 0)
        
#         original_node_indices = torch.arange(original_node_count, device=graph_data.edge_index.device)
#         global_node_index_array = original_node_indices.new_full((original_node_count,), original_node_count)
        
#         graph_data.edge_index = torch.cat([graph_data.edge_index, torch.stack([global_node_index_array, original_node_indices]), torch.stack([original_node_indices, global_node_index_array])], 1)
        
#         edge_feature_dimension = graph_data.edge_attr.size(1)
#         graph_data.edge_attr = torch.cat([graph_data.edge_attr, graph_data.edge_attr.new_zeros(2 * original_node_count, edge_feature_dimension)], 0)
        
#         new_graph_identifier = int(graph_data.batch.max()) + 1
#         graph_data.batch = torch.cat([graph_data.batch, graph_data.batch.new_tensor([new_graph_identifier])])
        
#         if hasattr(graph_data, "ptr"): 
#             graph_data.ptr = torch.cat([graph_data.ptr, graph_data.ptr[-1:] + 1])
        
#         graph_data.has_global_node = True
#         return graph_data

# =========================================================================== #
# Stage 2 - Node Encoder:
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Identity Block:
# --------------------------------------------------------------------------- #

# Note: valid for both node and edge encoding.

@register("core", "identity_transformation")
class IdentityTransformation(neural_network.Identity):
    def __init__(
        self,
        output_dimension: Optional[int] = None,
    ):
        super().__init__()
        self.output_dimension = output_dimension

# --------------------------------------------------------------------------- #
# Basic Node Encoder:
# --------------------------------------------------------------------------- #

class NodeEncoder(neural_network.Module):
    covers = ("node_embedder",)

    def __init__(self, output_dimension: int, mapping_scheme_name: str = "affine", input_dimension = NODE_FEATURE_DIMENSION):
        
        super().__init__()
        
        # Conceptual Idea: 
        # Linearly project the raw sparse node descriptors into a dense, continuous hidden mathematical space.
        if mapping_scheme_name == "identity":
            assert input_dimension == output_dimension, "Identity cannot change dimension."
        
        self.embedding_layer = retrieve_core_component(mapping_scheme_name)(input_dimension=input_dimension, output_dimension=output_dimension)
        self.output_node_dimension = output_dimension

    def forward(self, graph_data: Data) -> Data:
        """
        Idempotent encoder – converts 118-dim atom one-hot vectors to the
        hidden size exactly once per sample.  Subsequent passes in the same
        epoch see the already-embedded representation and skip re-encoding.
        """
        
        update_node_embeddings(graph_data, self.embedding_layer(get_node_embeddings(graph_data)))
        
        return graph_data
    
# --------------------------------------------------------------------------- #
# ChemProp Node Encoder:
# --------------------------------------------------------------------------- #

class ChemPropNodeEncoder(neural_network.Module):
    covers = ("node_embedder",)
    
    # FIX: Added `output_dimension` argument to prevent NameErrors if scheme != "identity"
    def __init__(self, mapping_scheme_name: str = "identity", active_dropout_probability: float = 0.0, output_dimension: Optional[int] = None):
        super().__init__()
        
        # Conceptual Idea: 
        # ChemProp requires a very specific robust set of chemical rules to encode node states. 
        # We define a mapping list of all permitted RDKit chemical descriptors (e.g. atom type, 
        # hybridization, chirality) including an explicit fallback to an "unknown" state for robust generalization.
        atomic_number = list(range(1, MAXIMUM_ATOMIC_NUMBER + 1)) + [None] # atomic number (no unknown pad): 118
        degree = list(range(6)) + [None] # degree + unknown
        formal_charge = [-2, -1, 0, 1, 2] + [None] # formal charge + unknown
        chiral_tag = [0, 1, 2, 3] + [None] # chiral tag + unknown
        total_hydrogen_atoms = [0, 1, 2, 3, 4] + [None] # total Hs + unknown
        hybridization = [Chem.rdchem.HybridizationType.S,                       # hybridization (incl. S) + unknown
                         Chem.rdchem.HybridizationType.SP,
                         Chem.rdchem.HybridizationType.SP2,
                         Chem.rdchem.HybridizationType.SP3,
                         Chem.rdchem.HybridizationType.SP3D,
                         Chem.rdchem.HybridizationType.SP3D2] + [None]
        
        self.chemical_encodings_map_tuple = (
            atomic_number, degree, formal_charge,
            chiral_tag, total_hydrogen_atoms, hybridization
        )
        
        total_feature_dimensions = sum(len(descriptor_category) for descriptor_category in self.chemical_encodings_map_tuple) + 2                        # + aromatic(1) + mass(1)
        
        self.embedding_layer = neural_network.Identity() if mapping_scheme_name == "identity" else retrieve_core_component(mapping_scheme_name)(input_dimension=total_feature_dimensions, output_dimension=output_dimension)
        
    def forward(self, graph_data: Data) -> Data:
        molecule_objects = graph_data.mol if isinstance(getattr(graph_data, "mol", None), (list, tuple)) else [graph_data.mol]
        
        def safely_one_hot_encode_value_in_array(target_value, validation_array):
            one_hot_binary_array = [0]*len(validation_array)
            one_hot_binary_array[validation_array.index(target_value) if target_value in validation_array else -1] = 1
            return one_hot_binary_array
        
        # Conceptual Idea: 
        # Iterate over all real chemical atom objects attached to the batch, 
        # extract their precise chemical properties, explicitly one-hot encode them matching 
        # the predefined vocabulary, and concatenate them into a singular dense list.
        encoded_chemical_features = [
             safely_one_hot_encode_value_in_array(chemical_atom.GetAtomicNum(), self.chemical_encodings_map_tuple[0]) + 
             safely_one_hot_encode_value_in_array(chemical_atom.GetDegree(), self.chemical_encodings_map_tuple[1]) + 
             safely_one_hot_encode_value_in_array(chemical_atom.GetFormalCharge(), self.chemical_encodings_map_tuple[2]) +
             safely_one_hot_encode_value_in_array(int(chemical_atom.GetChiralTag()), self.chemical_encodings_map_tuple[3]) + 
             safely_one_hot_encode_value_in_array(int(chemical_atom.GetTotalNumHs()), self.chemical_encodings_map_tuple[4]) +
             safely_one_hot_encode_value_in_array(chemical_atom.GetHybridization(), self.chemical_encodings_map_tuple[5]) + 
             [int(chemical_atom.GetIsAromatic())] + 
             [chemical_atom.GetMass()/100]
             for chemical_molecule in molecule_objects for chemical_atom in chemical_molecule.GetAtoms()
             ]
        
        encoded_features_tensor = torch.as_tensor(encoded_chemical_features, dtype=torch.float32, device=determine_hardware_device(graph_data))
        graph_data.original_node_embeddings = encoded_features_tensor
        update_node_embeddings(graph_data, self.embedding_layer(encoded_features_tensor))
        return graph_data

# =========================================================================== #
# Stage 3 - Edge Encoder:
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Basic Edge Encoder:
# --------------------------------------------------------------------------- #

class EdgeEncoder(neural_network.Module):
    covers = ("edge_embedder",)

    def __init__(self, output_dimension: int, mapping_scheme_name: str = "affine", input_dimension = EDGE_FEATURE_DIMENSION):
        super().__init__()
        
        # Conceptual Idea: 
        # Similar to the basic node encoder, this projects basic edge features
        # (like simple bond type) into a wider continuous embedding vector space.
        if mapping_scheme_name == "identity":
            assert input_dimension == output_dimension, "Identity cannot change dimension."

        self.embedding_layer = retrieve_core_component(mapping_scheme_name)(input_dimension=input_dimension, output_dimension=output_dimension)
        self.output_edge_dimension = output_dimension

    def forward(self, graph_data: Data) -> Data:
        
        update_edge_embeddings(graph_data, self.embedding_layer(get_edge_embeddings(graph_data)))
        
        return graph_data

# --------------------------------------------------------------------------- #
# ChemProp Edge Encoder:
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# ChemProp Edge Encoder (produces h0 and stores a copy)
# --------------------------------------------------------------------------- #

class ChemPropEdgeEncoder(neural_network.Module):
    covers = ("edge_embedder",)
    
    # FIX: Changed `mapping_scheme_name` from "linear" to "affine" to ensure initial W_i uses a bias
    def __init__(self, output_dimension: int, activation_function_class = neural_network.ReLU, mapping_scheme_name: str = "affine", dropout_probability = 0):
        super().__init__()
        # Conceptual Idea: 
        # ChemProp computes its initial edge state (often called h0) by concatenating 
        # the raw chemical bond features with the raw chemical features of the source node.
        self.output_edge_dimension = output_dimension
        self.projection_scheme = retrieve_core_component(mapping_scheme_name)
        self.activation_function = activation_function_class()
        self.stereo_configurations = [BondStereo.STEREONONE, BondStereo.STEREOANY, BondStereo.STEREOZ,
                       BondStereo.STEREOE, BondStereo.STEREOCIS, BondStereo.STEREOTRANS, None]
        
        self.dropout_regularization_layer = neural_network.Identity() if not dropout_probability else neural_network.Dropout(p=dropout_probability)
        self.initial_projection_layer = self.projection_scheme(input_dimension=152 + EDGE_FEATURE_DIMENSION + 2 + len(self.stereo_configurations), output_dimension=self.output_edge_dimension)
    
    def forward(self, graph_data: Data) -> Data:
        base_edge_features_tensor = get_edge_embeddings(graph_data)
        source_nodes_indices, destination_nodes_indices = graph_data.edge_index
        source_node_embeddings_tensor = get_node_embeddings(graph_data)[source_nodes_indices]
        
        molecule_objects = graph_data.mol if isinstance(getattr(graph_data,'mol',None),(list,tuple)) else [graph_data.mol]
        batch_indices_tensor = getattr(graph_data,'batch', torch.zeros(graph_data.num_nodes, dtype=torch.long, device=base_edge_features_tensor.device))
        existing_batch_pointers_tensor = getattr(graph_data, 'ptr', None)
        if existing_batch_pointers_tensor is not None:
            batch_pointers_tensor = existing_batch_pointers_tensor
        else:
            batch_pointers_tensor = torch.cat([torch.zeros(1, device=batch_indices_tensor.device, dtype=torch.long), mps_safe_bincount(batch_indices_tensor).cumsum(0)])
        
        number_of_edges = base_edge_features_tensor.size(0)
        stereo_length = len(self.stereo_configurations)
        encoded_edge_rows = []
        
        # Conceptual Idea: 
        # Extract RDKit-specific structural details per bond (conjugation, rings, stereochemistry)
        # by cross-referencing the flat edge arrays against their parent molecule objects.
        for edge_index in range(number_of_edges):
            parent_graph_index = int(batch_indices_tensor[destination_nodes_indices[edge_index]])
            first_node_index_in_graph_offset = int(batch_pointers_tensor[parent_graph_index])
            
            chemical_bond = molecule_objects[parent_graph_index].GetBondBetweenAtoms(int(source_nodes_indices[edge_index])-first_node_index_in_graph_offset,int(destination_nodes_indices[edge_index])-first_node_index_in_graph_offset)
            is_conjugated_flag = float(chemical_bond.GetIsConjugated()) if chemical_bond else 0.0
            is_in_ring_flag = float(chemical_bond.IsInRing()) if chemical_bond else 0.0
            
            stereo_chemistry_identifier = self.stereo_configurations.index(chemical_bond.GetStereo()) if chemical_bond and chemical_bond.GetStereo() in self.stereo_configurations else stereo_length-1
            one_hot_stereo_encoding_vector = [0]*stereo_length
            one_hot_stereo_encoding_vector[stereo_chemistry_identifier] = 1
            
            encoded_edge_rows.append([is_conjugated_flag, is_in_ring_flag] + one_hot_stereo_encoding_vector)
        
        advanced_edge_features_tensor = torch.tensor(encoded_edge_rows,dtype=base_edge_features_tensor.dtype,device=base_edge_features_tensor.device) if number_of_edges > 0 else base_edge_features_tensor.new_zeros((0, 2+stereo_length))
        concatenated_edge_node_tensor = torch.cat([advanced_edge_features_tensor,base_edge_features_tensor,source_node_embeddings_tensor],-1)
        
        initial_hidden_state_tensor = self.dropout_regularization_layer(self.activation_function(self.initial_projection_layer(concatenated_edge_node_tensor)))

        graph_data.edge_attr = initial_hidden_state_tensor
        graph_data.edge_attr_0 = initial_hidden_state_tensor.clone()
        return graph_data


# =========================================================================== #
# Stage 4 - Structural Encoder:
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Basic Structural Encoder:
# --------------------------------------------------------------------------- #

class StructuralEncoder(neural_network.Module):
    covers = ("structural_encoder",)

    def __init__(self, requested_encoder_modes: Optional[Sequence[str]] = None):

        # Conceptual Idea: 
        # Instantiate the dynamically configured structural encoding sub-pipeline, allowing 
        # multiple mathematical graph properties (degrees, eigenvectors) to be aggregated into the embedding payload.
        if requested_encoder_modes is None or requested_encoder_modes == "none":
            requested_encoder_modes = ()
        self.structural_encoder_function = structural_encoder(requested_encoder_modes)
    
    @torch.no_grad()
    def forward(self, graph_data: Data) -> Data:
        update_structural_embeddings(graph_data, self.structural_encoder_function(graph_data))
        return graph_data

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

    def forward(self, graph_data: Data) -> Data:
        
        # Conceptual Idea: 
        # A basic edge update relies on combining information. We merge the representations of 
        # the two nodes attached to the edge alongside the previous edge representation itself to compute the next step.
        source_nodes, destination_nodes = get_source_and_destination_nodes(graph_data)
        previous_edge_embeddings_tensor = get_edge_embeddings(graph_data)
        
        concatenated_triplet_features = torch.cat([graph_data.x[source_nodes], graph_data.x[destination_nodes], previous_edge_embeddings_tensor], dim=-1)
        
        update_edge_embeddings(graph_data, self.update_function(concatenated_triplet_features))
        return graph_data


# --------------------------------------------------------------------------- #
# Basic Edge Update:
# --------------------------------------------------------------------------- #

class EdgeMessageUpdate(EdgeUpdateBase):
    covers = ("message_fn",)

    def __init__(
        self,
        target_update_function: callable
    ):
        super().__init__(aggr=None)
        self.update_function = target_update_function
    
# --------------------------------------------------------------------------- #
# FeedForwardNeuralNetwork Edge Update:
# --------------------------------------------------------------------------- #

class FFNEdgeUpdate(EdgeMessageUpdate):
    covers = ("message_fn",)

    def __init__(
        self,
        input_dimension: int,
        hidden_dimensions: Union[List[int], Tuple[int, ...]],
        use_batch_normalizations: Union[bool, Sequence[bool]] = False,
        dropout_probabilities: Union[float, Sequence[float]] = 0.0,
        activation_functions: Union[str, Sequence[str]] = DEFAULT_ACTIVATION_FUNCTION,
    ):
        feed_forward_network_function = retrieve_core_component("ffn")(input_dimension = input_dimension, 
                                             hidden_dimensions = hidden_dimensions,
                                             use_batch_normalizations = use_batch_normalizations,
                                             dropout_probabilities = dropout_probabilities,
                                             activation_functions = activation_functions
                                             )
        super().__init__(feed_forward_network_function)

# --------------------------------------------------------------------------- #
# ChemProp Edge Update:
# --------------------------------------------------------------------------- #

class ChemPropEdgeUpdate(neural_network.Module):
    covers = ("edge_update",)

    def __init__(self, edge_feature_dimension: int, active_dropout_probability: float = 0.0):
        super().__init__()
        
        # Conceptual Idea: 
        # The ChemProp framework updates edge representations by linearly projecting the accumulated messages 
        # and then folding them additively back onto the absolute starting baseline edge features (a resilient skip connection).
        # FIX: Explicitly disable bias (ChemProp W_m has no bias, relying on h0 instead)
        self.message_transformation_layer = GeometricLinear(edge_feature_dimension, edge_feature_dimension, bias=False)
        self.activation_function = neural_network.ReLU()
        self.dropout_regularization_layer = neural_network.Dropout(active_dropout_probability)

    def forward(self, graph_data: Data) -> Data:
        current_aggregated_messages_tensor = graph_data.messages
        original_baseline_messages_tensor = graph_data.edge_attr_0
        transformed_messages_tensor = self.message_transformation_layer(current_aggregated_messages_tensor)              # W_m * m
        graph_data.edge_attr = self.dropout_regularization_layer(self.activation_function(original_baseline_messages_tensor + transformed_messages_tensor))
        return graph_data

# =========================================================================== #
# Stage 6 - Edge Attention:
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Base Edge Attention Class:
# --------------------------------------------------------------------------- #

"""
All edge attention classes should inherit from this and only overwrite __init__.
"""

class EdgeAttentionBase(neural_network.Module):
    covers = ("edge_attention",)

    def forward(self, graph_data: Data) -> Data:
        source_node_features_tensor = self.source_transformation_function(get_node_embeddings(graph_data))
        destination_node_features_tensor = self.destination_transformation_function(get_node_embeddings(graph_data))
        edge_attention_logits_tensor = self.source_destination_combination_function(source_node_features_tensor[graph_data.edge_index[0]] + destination_node_features_tensor[graph_data.edge_index[1]])
        
        # USE GEOMETRIC SOFTMAX INDEXED BY DESTINATION NODES
        normalized_attention_tensor = geometric_softmax(edge_attention_logits_tensor, index=graph_data.edge_index[1], dim=0)
        attention_weights_tensor = self.dropout_layer(normalized_attention_tensor).mean(-1, keepdim=True)
        
        update_edge_embeddings(graph_data, get_edge_embeddings(graph_data) * attention_weights_tensor)
        return graph_data

# --------------------------------------------------------------------------- #
# Basic Edge Update Class:
# --------------------------------------------------------------------------- #


class EdgeAttention(EdgeAttentionBase):
    covers = ("edge_attention",)

    def __init__(self, source_transformation_function: callable, destination_transformation_function: callable, source_destination_combination_function: callable, logit_normalization_function=partial(torch.softmax, dim=0), active_dropout_probability: float = 0):
        super().__init__()
        self.source_transformation_function = source_transformation_function
        self.destination_transformation_function = destination_transformation_function
        self.source_destination_combination_function = source_destination_combination_function
        self.logit_normalization_function = logit_normalization_function
        self.dropout_layer = retrieve_core_component("dropout")(active_dropout_probability)

# --------------------------------------------------------------------------- #
# Base Edge Update Class:
# --------------------------------------------------------------------------- #

class DefaultEdgeAttention(EdgeAttentionBase):
    covers = ("edge_attention",)

    def __init__(self, node_feature_dimension: int, number_of_attention_heads: int, active_dropout_probability: float):
        super().__init__()
        self.source_transformation_function = GeometricLinear(node_feature_dimension, number_of_attention_heads, bias=False)
        self.destination_transformation_function = GeometricLinear(node_feature_dimension, number_of_attention_heads, bias=False)
        self.source_destination_combination_function = neural_network.LeakyReLU(0.2)
        self.logit_normalization_function = partial(torch.softmax, dim=0)
        self.dropout_layer = retrieve_core_component("dropout")(active_dropout_probability)

# --------------------------------------------------------------------------- #
# Deprecated Module - DO NOT USE
# --------------------------------------------------------------------------- #

class DeprecatedEdgeAttention(neural_network.Module):
    covers = ("edge_attention",)

    def __init__(self, node_feature_dimension: int, number_of_attention_heads: int = 1, active_dropout_probability: float = 0.0):
        super().__init__()
        self.linear_source_layer = GeometricLinear(node_feature_dimension, number_of_attention_heads, bias=False)
        self.linear_destination_layer = GeometricLinear(node_feature_dimension, number_of_attention_heads, bias=False)
        self.dropout_layer = neural_network.Dropout(active_dropout_probability)
        self.leaky_relu_activation = neural_network.LeakyReLU(0.2)
        self.output_edge_dimension = node_feature_dimension
        self.output_node_dimension = None

    def forward(self, graph_data: Data) -> Data:
        source_node_features_tensor = self.linear_source_layer(graph_data.x)
        destination_node_features_tensor = self.linear_destination_layer(graph_data.x)
        edge_attention_logits_tensor = self.leaky_relu_activation(source_node_features_tensor[graph_data.edge_index[0]] + destination_node_features_tensor[graph_data.edge_index[1]])
        attention_weights_tensor = self.dropout_layer(torch.softmax(edge_attention_logits_tensor, dim=0)).mean(-1, keepdim=True)
        base_edge_attributes_tensor = (
            graph_data.edge_attr
            if getattr(graph_data, "edge_attr", None) is not None
            else graph_data.x[graph_data.edge_index[0]]
        )
        graph_data.edge_attr = base_edge_attributes_tensor * attention_weights_tensor
        return graph_data

# =========================================================================== #
# Stage 7 - Local Aggregator:
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Simple Aggregator:
# --------------------------------------------------------------------------- #

class LocalAggregator(neural_network.Module):
    covers = ("local_aggregator",)

    def __init__(self, reduction_method_name: str = "mean"):
        super().__init__()
        # Conceptual Idea: 
        # Standard graph network aggregation. Collect all incoming messages traveling along 
        # edges pointing into a given node, and mathematically summarize them (sum, mean, max) 
        # into a single vector to represent the neighborhood state.
        if reduction_method_name not in SCATTER_OPERATIONS:
            raise ValueError(reduction_method_name)
        self.reduction_method_name = reduction_method_name
        self.output_node_dimension = None
        self.output_edge_dimension = None

    def forward(self, graph_data: Data) -> Data:
        source_nodes_indices, destination_nodes_indices = graph_data.edge_index
        graph_data.x = SCATTER_OPERATIONS[self.reduction_method_name](graph_data.edge_attr, destination_nodes_indices, graph_data.num_nodes)
        return graph_data

# --------------------------------------------------------------------------- #
# D-MPNN Aggregator (writes messages, preserves states)
# --------------------------------------------------------------------------- #
class DMPNNLocalAggregator(neural_network.Module):
    """
    Non-backtracking messages via the node-sum trick:
        m_{v->w} = sum_{u->v} h_{u->v}  -  h_{w->v}
    Writes:
        • data.messages ← m
    Leaves:
        • data.edge_attr as the current state h^t
    Optionally (off by default), can export node features by summing edges.
    """

    covers = ("local_aggregator",)

    def __init__(self,
                 export_node_features_flag: bool = False,
                 node_feature_source_stage: str = "pre",   # "pre" uses h, "post" uses m
                 reduction_method_name: str = "sum"):
        super().__init__()
        
        # Conceptual Idea: 
        # In a Directed-Message Passing Neural Network, we specifically avoid message "tottering" 
        # where a signal bounces back and forth across a single bond infinitely. We achieve this 
        # by gathering all incoming messages to an intermediate node, then explicitly subtracting out 
        # the message that just arrived from the destination node, ensuring flow remains strictly forward.
        if reduction_method_name not in SCATTER_OPERATIONS:
            raise ValueError(reduction_method_name)
        
        if node_feature_source_stage not in ("pre", "post"):
            raise ValueError("node_feature_source_stage must be 'pre' or 'post'.")
        
        self.reduction_method_name = reduction_method_name
        self.export_node_features_flag = export_node_features_flag
        self.node_feature_source = node_feature_source_stage
        self.output_node_dimension = None
        self.output_edge_dimension = None

    @torch.no_grad()
    def build_reverse_edge_mapping(self, source_nodes_indices: torch.Tensor, destination_nodes_indices: torch.Tensor, number_of_nodes: int) -> torch.Tensor:
        total_node_count = int(number_of_nodes)
        forward_edge_keys_tensor = (source_nodes_indices.to(torch.long) * total_node_count + destination_nodes_indices.to(torch.long))
        reverse_edge_keys_tensor = (destination_nodes_indices.to(torch.long) * total_node_count + source_nodes_indices.to(torch.long))
        sorted_ordering_indices = torch.argsort(forward_edge_keys_tensor)
        sorted_forward_edge_keys_tensor = forward_edge_keys_tensor[sorted_ordering_indices]
        search_position_indices = torch.searchsorted(sorted_forward_edge_keys_tensor, reverse_edge_keys_tensor)
        # Clamp indices to prevent Out-Of-Bounds exceptions on the right side of the bitwise '&'
        clamped_indices = torch.clamp(search_position_indices, max=sorted_forward_edge_keys_tensor.numel() - 1)
        valid_matching_mask = (search_position_indices < sorted_forward_edge_keys_tensor.numel()) & \
                              (sorted_forward_edge_keys_tensor[clamped_indices] == reverse_edge_keys_tensor)
        if not bool(torch.all(valid_matching_mask)):
            raise ValueError("Reverse edge missing; ensure bidirected edges or precompute data.rev.")
        reverse_edge_mapping_indices = sorted_ordering_indices[search_position_indices]
        return reverse_edge_mapping_indices

    def forward(self, graph_data: Data) -> Data:
        
        source_nodes_indices, destination_nodes_indices = graph_data.edge_index
        current_edge_states_tensor = graph_data.edge_attr  # [E, F] current edge states
        number_of_edges = current_edge_states_tensor.size(0); number_of_nodes = graph_data.num_nodes

        if number_of_edges == 0:
            if self.export_node_features_flag:
                graph_data.x = SCATTER_OPERATIONS[self.reduction_method_name](current_edge_states_tensor, destination_nodes_indices, number_of_nodes)
            graph_data.messages = current_edge_states_tensor.new_zeros((0, current_edge_states_tensor.size(-1)))
            return graph_data

        # 1) Sum incoming edges at each node
        # Conceptual Idea: Consolidate all edge signals pouring into every destination node simultaneously.
        summed_incoming_edge_states_tensor = SCATTER_OPERATIONS["sum"](current_edge_states_tensor, destination_nodes_indices, number_of_nodes)  # [N, F]

        # 2) Non-backtracking messages for each directed edge
        # Conceptual Idea: To prevent echoes, subtract the direct reverse edge state from the aggregated incoming state.
        reverse_edge_indices_tensor = getattr(graph_data, "rev", None)
        if reverse_edge_indices_tensor is None:
            reverse_edge_indices_tensor = self.build_reverse_edge_mapping(source_nodes_indices, destination_nodes_indices, number_of_nodes)
        non_backtracking_messages_tensor = summed_incoming_edge_states_tensor[source_nodes_indices] - current_edge_states_tensor[reverse_edge_indices_tensor]              # [E, F]

        # 3) Expose messages (do NOT overwrite h^t)
        graph_data.messages = non_backtracking_messages_tensor

        # 4) Optional node export (Chemprop does this only after the final step)
        if self.export_node_features_flag:
            graph_data.x = SCATTER_OPERATIONS[self.reduction_method_name](current_edge_states_tensor if self.node_feature_source == "pre" else non_backtracking_messages_tensor, destination_nodes_indices, number_of_nodes)

        return graph_data


# =========================================================================== #
# Stage 8 - Global Self Attention:
# =========================================================================== #

class GlobalSelfAttention(neural_network.Module):
    covers = ("global_attention",)

    def __init__(self, expected_feature_dimension: int, number_of_attention_heads: int = 8, active_dropout_probability: float = 0.0):
        super().__init__()
        
        # Conceptual Idea: 
        # Bypass topological constraints. We initialize a multi-head transformer 
        # block allowing all atom-nodes in the entire graph to cross-attend and interact globally.
        self.multihead_attention_layer = neural_network.MultiheadAttention(expected_feature_dimension, number_of_attention_heads, dropout=active_dropout_probability, batch_first=True)
        self.output_node_dimension = expected_feature_dimension
        self.output_edge_dimension = None

    def forward(self, graph_data: Data) -> Data:
        if not hasattr(graph_data, "batch"):
            attention_output_tensor, _ = self.multihead_attention_layer(graph_data.x[None], graph_data.x[None], graph_data.x[None])
            graph_data.x = attention_output_tensor.squeeze(0)
            return graph_data
            
        batch_assignment_indices = graph_data.batch
        graph_sizes_tensor = mps_safe_bincount(batch_assignment_indices)
        total_number_of_graphs = graph_sizes_tensor.size(0)
        maximum_graph_size = int(graph_sizes_tensor.max())
        
        pad_sequence_function = lambda sequence_tensor, actual_length: neural_network_functional.pad(
            sequence_tensor, (0, 0, 0, int(maximum_graph_size) - int(actual_length.item()))
        )
        padded_node_features_tensor = torch.stack([pad_sequence_function(graph_data.x[batch_assignment_indices == graph_index], graph_sizes_tensor[graph_index]) for graph_index in range(total_number_of_graphs)])

        # Mask padded positions so real nodes do not attend to padding.
        position_indices = torch.arange(int(maximum_graph_size), device=graph_data.x.device).unsqueeze(0)
        key_padding_mask = position_indices >= graph_sizes_tensor.unsqueeze(1)   # (num_graphs, max_nodes), True == ignore

        attention_output_tensor, _ = self.multihead_attention_layer(padded_node_features_tensor, padded_node_features_tensor, padded_node_features_tensor, key_padding_mask=key_padding_mask)
        graph_data.x = torch.cat([attention_output_tensor[graph_index, : graph_sizes_tensor[graph_index]] for graph_index in range(total_number_of_graphs)], 0)
        return graph_data

# =========================================================================== #
# Stage 9 - Hybrid Mixer:
# =========================================================================== #

class HybridMixer(neural_network.Module):
    covers = ("hybrid_router",)

    def __init__(self, initial_mixing_alpha_value: float = 0.5):
        super().__init__()
        # Conceptual Idea: 
        # A parameterized routing layer. It receives the locally aggregated features and 
        # globally aggregated features distinctly, passing the mixing coefficient scalar through 
        # a sigmoid gate to learn the optimal proportionate combination of the two topological perspectives.
        self.mixing_alpha_weight_tensor = neural_network.Parameter(torch.tensor(initial_mixing_alpha_value))
        self.output_node_dimension = None
        self.output_edge_dimension = None

    def forward(
        self,
        local_node_features_tensor: Optional[torch.Tensor],
        global_node_features_tensor: Optional[torch.Tensor]
    ) -> torch.Tensor:

        # ----- pass‑through shortcuts -----------------------------------
        if local_node_features_tensor is None and global_node_features_tensor is None:
            raise ValueError("HybridMixer received no inputs.")
        if local_node_features_tensor is None:
            return global_node_features_tensor
        if global_node_features_tensor is None:
            return local_node_features_tensor

        # ----- gated blend ----------------------------------------------
        normalized_mixing_alpha_scalar = torch.sigmoid(self.mixing_alpha_weight_tensor)               # scalar in (0,1)
        return normalized_mixing_alpha_scalar * local_node_features_tensor + (1.0 - normalized_mixing_alpha_scalar) * global_node_features_tensor

# =========================================================================== #
# Stage 10 - Node Update:
# =========================================================================== #

class NodeUpdate(neural_network.Module):
    covers = ("node_update",)

    def __init__(self, node_feature_dimension: int, expansion_width_multiplier: int = 2, active_dropout_probability: float = 0.0):
        super().__init__()
        # Conceptual Idea: 
        # Provide dedicated non-linear modeling capacity exclusively addressing the node representation. 
        # Usually built via an expanded projection matching transformer structures, combined with a residual layer normalization block.
        self.feed_forward_network_block = neural_network.Sequential(
            GeometricLinear(node_feature_dimension, node_feature_dimension * expansion_width_multiplier),
            neural_network.GELU(),
            GeometricLinear(node_feature_dimension * expansion_width_multiplier, node_feature_dimension),
            neural_network.Dropout(active_dropout_probability),
        )
        self.normalization_layer = neural_network.LayerNorm(node_feature_dimension)
        self.output_node_dimension = node_feature_dimension
        self.output_edge_dimension = None

    def forward(self, graph_data: Data) -> Data:
        graph_data.x = self.normalization_layer(graph_data.x + self.feed_forward_network_block(graph_data.x))
        return graph_data

# =========================================================================== #
# Stage 11 - Edge Update:
# =========================================================================== #

class EdgeUpdate(MessagePassing):
    covers = ("edge_update",)

    def __init__(self, node_feature_dimension: int, edge_feature_dimension: int, hidden_dimension: int):
        super().__init__(aggr=None)
        
        self.multilayer_perceptron = neural_network.Sequential(
            GeometricLinear(node_feature_dimension * 2 + edge_feature_dimension, hidden_dimension),
            neural_network.ReLU(),
            GeometricLinear(hidden_dimension, edge_feature_dimension),
        )
        self.output_edge_dimension = edge_feature_dimension
        self.output_node_dimension = None

    def forward(self, graph_data: Data) -> Data:
        # Conceptual Idea: 
        # Fully rewrite the hidden parameters on each edge explicitly based structurally 
        # on the newly resolved surrounding node states connected to it at the tail of the interaction sequence step.
        source_nodes_indices, destination_nodes_indices = graph_data.edge_index
        concatenated_input_features_tensor = torch.cat([graph_data.x[source_nodes_indices], graph_data.x[destination_nodes_indices], graph_data.edge_attr], dim=-1)
        graph_data.edge_attr = self.multilayer_perceptron(concatenated_input_features_tensor)
        return graph_data

# =========================================================================== #
# Stage 12 - Readout Pooler:
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Readout & multiscale aggregation
# --------------------------------------------------------------------------- #
GRAPH_READOUT_OPERATIONS = {
    "sum": global_add_pool,
    "mean": global_mean_pool,
    "max": global_max_pool,
}

class ReadoutPooler(neural_network.Module):
    covers = ("readout_pool",)

    def __init__(self, reduction_method_name: str = "mean", target_attribute_name: str = "graph_feat"):
        """
        Parameters
        ----------
        reduce     : "mean" | "sum" | "max"
        attr_name  : name of the attribute that will hold
                     the pooled graph‑level feature inside `Data`.
        """
        super().__init__()
        # Conceptual Idea: 
        # At the termination of graph traversal operations, we structurally collapse the varying 
        # dimensionally-sized node tensors into a fixed-size vector representing the entire graph 
        # globally via symmetrical aggregation operators defined on the batch segmentation markers.
        self.pooling_function = GRAPH_READOUT_OPERATIONS[reduction_method_name]
        self.target_attribute_name = target_attribute_name

    def forward(self, graph_data: Data) -> Data:
        if hasattr(graph_data, "batch"):
            pooled_features_tensor = self.pooling_function(graph_data.x, graph_data.batch)
        else:                       # single graph case
            pooled_features_tensor = self.pooling_function(graph_data.x, torch.zeros(graph_data.num_nodes,
                                                 dtype=torch.long,
                                                 device=graph_data.x.device))
        setattr(graph_data, self.target_attribute_name, pooled_features_tensor)   # e.g. data.graph_feat
        return graph_data

# --------------------------------------------------------------------------- #
# ChemProp Readout Pooler:
# --------------------------------------------------------------------------- #

class ChempropReadoutPooler(neural_network.Module):
    covers = ("readout_pool",)

    def __init__(self, 
                 edge_feature_dimension: int,
                 node_feature_dimension: int = 152,
                 activation_function_name: str = "relu",
                 reduction_method_name: str = "sum", 
                 target_attribute_name: str = "graph_feat",
                 active_dropout_probability: float = 0.0):
        super().__init__()
        if reduction_method_name not in GRAPH_READOUT_OPERATIONS:
            raise ValueError(f"reduction_method_name must be one of {list(GRAPH_READOUT_OPERATIONS.keys())}")
        self.pooling_function = GRAPH_READOUT_OPERATIONS[reduction_method_name]
        self.target_attribute_name = target_attribute_name

        # FIX: Instantiate BEFORE the forward pass so it's registered in model.parameters()
        self.projection_layer_module = GeometricLinear(node_feature_dimension + edge_feature_dimension, edge_feature_dimension)
        
        # Resolve the activation function gracefully
        self.activation_function_module = retrieve_core_component(activation_function_name)() if isinstance(activation_function_name, str) else neural_network.ReLU()
        self.dropout_regularization_module = neural_network.Dropout(active_dropout_probability) if active_dropout_probability > 0 else neural_network.Identity()

        self.output_node_dimension = edge_feature_dimension
        self.output_edge_dimension = None

    def forward(self, graph_data: Data) -> Data:
        # Conceptual Idea: 
        # The ChemProp pooler uniquely constructs final representations by combining the raw input
        # node features directly with the mathematically summed array of all adjacent incoming edges' hidden representations.
        _, destination_nodes_indices = graph_data.edge_index
        edge_hidden_states_tensor = graph_data.edge_attr
        total_number_of_nodes      = graph_data.num_nodes
        summed_edge_vectors_tensor    = SCATTER_OPERATIONS["sum"](edge_hidden_states_tensor, destination_nodes_indices, total_number_of_nodes)     # [N, d_h]

        node_representations_tensor = graph_data.x                            # [N, d_v]

        concatenated_vectors_tensor    = torch.cat([node_representations_tensor, summed_edge_vectors_tensor], dim=-1) # [N, d_v + d_h]
        atom_hidden_states_tensor = self.dropout_regularization_module(self.activation_function_module(self.projection_layer_module(concatenated_vectors_tensor))) # [N, d_h]
        graph_data.x = atom_hidden_states_tensor

        if hasattr(graph_data, "batch"):
            graph_level_features_tensor = self.pooling_function(atom_hidden_states_tensor, graph_data.batch)
        else:
            fake_batch_indices_tensor = torch.zeros(total_number_of_nodes, dtype=torch.long, device=atom_hidden_states_tensor.device)
            graph_level_features_tensor = self.pooling_function(atom_hidden_states_tensor, fake_batch_indices_tensor)

        setattr(graph_data, self.target_attribute_name, graph_level_features_tensor)
        return graph_data

# =========================================================================== #
# Stage 13 - Multi Scale Aggregator:
# =========================================================================== #

class MultiScaleAggregator(neural_network.Module):
    covers = ("multiscale_agg",)
    def __init__(self, reduction_method_name: str = "mean"):
        super().__init__()
        self.pooling_module = ReadoutPooler(reduction_method_name)  # defines attr_name

    def forward(self, graph_data: Data, fragment_features_tensor: Optional[torch.Tensor] = None) -> Data:
        # Conceptual Idea: 
        # Accumulate pooled outputs across differing hierarchical spatial sizes or multiple iterative 
        # message-passing steps simultaneously, concatenating them together sequentially.
        if not hasattr(graph_data, self.pooling_module.target_attribute_name):
            graph_data = self.pooling_module(graph_data)
        graph_level_features_tensor = getattr(graph_data, self.pooling_module.target_attribute_name)
        if fragment_features_tensor is not None:
            graph_level_features_tensor = torch.cat([graph_level_features_tensor, fragment_features_tensor], dim=-1)
        setattr(graph_data, self.pooling_module.target_attribute_name, graph_level_features_tensor)
        return graph_data


# =========================================================================== #
# Stage 16 - Multi Scale Aggregator:
# =========================================================================== #

class GraphVectorizer(neural_network.Module):
    """
    Final‑stage module for a stage pipeline.
    Expects `data.graph_feat` (set by ReadoutPooler or MultiScaleAggregator)
    and returns a tensor of shape  [num_graphs, output_dimension].

    Parameters
    ----------
    input_dimension       : Dimension of `graph_feat` coming from the readout.
    project_dim  : If not None and different from `input_dimension`,
                   applies a learnable GeometricLinear projection to this size.
    normalize    : If True, L2‑normalizes the output (useful for contrastive
                   learning or cosine‑similarity fusion).
    """
    covers = ("task_head",)

    def __init__(self,
                 input_dimension: int,
                 projection_dimension: Optional[int] = None,
                 apply_l2_normalization: bool = False):
        super().__init__()
        self.apply_l2_normalization = apply_l2_normalization

        # Conceptual Idea: 
        # Strip away the complex geometric scaffolding structure holding the graphs, linearly 
        # map the final continuous embedded vector if required, and spit it out as an independent 
        # standard 2D feature matrix (Batch_Size, Dimensions) ready to be fed directly into task MLPs.
        use_same_dimension = projection_dimension is None or projection_dimension == input_dimension
        self.projection_layer_module = neural_network.Identity() if use_same_dimension else GeometricLinear(input_dimension, projection_dimension)
        self.output_dimension = input_dimension if use_same_dimension else projection_dimension

    def forward(self, graph_data: Data) -> torch.Tensor:
        # Grab (or lazily compute) the pooled vector
        graph_level_features_tensor = getattr(graph_data, "graph_feat", None)
        if graph_level_features_tensor is None:
            raise RuntimeError(
                "GraphVectorizer expects `data.graph_feat` to be populated by "
                "`ReadoutPooler` or another readout stage placed earlier in the pipeline."
            )

        graph_level_features_tensor = self.projection_layer_module(graph_level_features_tensor)
        if self.apply_l2_normalization:
            graph_level_features_tensor = neural_network_functional.normalize(graph_level_features_tensor, p=2, dim=-1)
        return graph_level_features_tensor

# --------------------------------------------------------------------------- #
# Community‑provided block (example)
# --------------------------------------------------------------------------- #
class GATBlock(neural_network.Module):
    covers = ("message_fn", "edge_attention", "local_aggregator")

    def __init__(self, input_dimension: int, output_dimension: int, number_of_attention_heads: int = 8):
        super().__init__()
        self.graph_attention_layer_module = GATConv(
            input_dimension,
            output_dimension // number_of_attention_heads,
            heads=number_of_attention_heads,
            concat=True,
            add_self_loops=False,
        )
        self.output_node_dimension = output_dimension
        self.output_edge_dimension = output_dimension

    def forward(self, graph_data: Data) -> Data:
        graph_data.x = self.graph_attention_layer_module(graph_data.x, graph_data.edge_index)
        graph_data.edge_attr = graph_data.x[graph_data.edge_index[0]]
        return graph_data


# --------------------------------------------------------------------------- #
# Helpers to validate and build a stage pipeline
# --------------------------------------------------------------------------- #
class StageConfigurationError(ValueError):
    ...


def validate_pipeline_stage_configuration(stage_mapping_dictionary: Dict[str, neural_network.Module]) -> None:
    # Conceptual Idea: 
    # Safely verify that every abstract step declared within our overarching standard GNN layout is
    # precisely accounted for by one matching user-supplied layer module mapping, rejecting duplicates and omittances.
    missing_stages = [stage_name for stage_name in GNN_STAGES if stage_name not in stage_mapping_dictionary]
    if missing_stages:
        raise StageConfigurationError(f"Missing stages {missing_stages}")
    unknown_stages = [dictionary_key for dictionary_key in stage_mapping_dictionary if dictionary_key not in GNN_STAGES]
    if unknown_stages:
        raise StageConfigurationError(f"Unknown keys {unknown_stages}")
    stage_coverage_tracker_dictionary = {stage_name: False for stage_name in GNN_STAGES}
    for dictionary_key, module_instance in stage_mapping_dictionary.items():
        covered_stages = getattr(module_instance, "covers", (dictionary_key,))
        for specific_stage_name in covered_stages:
            if stage_coverage_tracker_dictionary[specific_stage_name]:
                raise StageConfigurationError(f"Stage '{specific_stage_name}' covered twice")
            stage_coverage_tracker_dictionary[specific_stage_name] = True


def build_pipeline_sequence(stage_mapping_dictionary: Dict[str, neural_network.Module]) -> neural_network.ModuleList:
    validate_pipeline_stage_configuration(stage_mapping_dictionary)
    return neural_network.ModuleList([stage_mapping_dictionary[dictionary_key] for dictionary_key in GNN_STAGES])


# --------------------------------------------------------------------------- #
# Reference GNN backbone that consumes a stage map
# --------------------------------------------------------------------------- #
class GraphNeuralNetwork(neural_network.Module):
    """
    A sixteen‑stage graph‑to‑something backbone whose stages are both
    (i) registered as plain attributes for easy access, and
    (ii) stored in a ModuleList to keep `neural_network.Module` semantics happy.
    """

    def __init__(self, stage_mapping_dictionary: Dict[str, neural_network.Module], number_of_propagation_steps = 1, random_seed_value: int = 42):
        super().__init__()
        #set_global_random_seed(random_seed_value)

        # ------------------------------------------------------------------ #
        # 1. Validate the map (raises if a stage is missing or duplicated)
        # ------------------------------------------------------------------ #
        validate_pipeline_stage_configuration(stage_mapping_dictionary)             # re‑use your helper

        # ------------------------------------------------------------------ #
        # 2. ***Explicitly*** register every stage as an attribute
        # ------------------------------------------------------------------ #
        
        # Pre-repeat steps
        self.preprocessor       = stage_mapping_dictionary["preprocessor"]
        self.node_embedder      = stage_mapping_dictionary["node_embedder"]
        self.edge_embedder      = stage_mapping_dictionary["edge_embedder"]
        self.structural_encoder = stage_mapping_dictionary["structural_encoder"]
        
        pre_repeat_steps = [self.preprocessor, self.node_embedder,  self.edge_embedder,  self.structural_encoder]

        self.message_fn         = stage_mapping_dictionary["message_fn"]
        self.edge_attention     = stage_mapping_dictionary["edge_attention"]
        self.local_aggregator   = stage_mapping_dictionary["local_aggregator"]
        self.global_attention   = stage_mapping_dictionary["global_attention"]
        self.hybrid_router      = stage_mapping_dictionary["hybrid_router"]
        self.node_update        = stage_mapping_dictionary["node_update"]
        self.edge_update        = stage_mapping_dictionary["edge_update"]
        self.feed_forward       = stage_mapping_dictionary["feed_forward"]
        self.norm_reg           = stage_mapping_dictionary["norm_reg"]

        repeat_steps = [self.message_fn,      self.edge_attention, self.local_aggregator, 
                         self.global_attention, self.hybrid_router, self.node_update,   
                         self.edge_update, self.feed_forward,   self.norm_reg]

        self.readout_pool       = stage_mapping_dictionary["readout_pool"]
        self.multiscale_agg     = stage_mapping_dictionary["multiscale_agg"]
        self.task_head          = stage_mapping_dictionary["task_head"]
        
        post_repeat_steps = [self.readout_pool, self.multiscale_agg, self.task_head]

        # ------------------------------------------------------------------ #
        # 3. Keep a ModuleList in the canonical execution order
        # ------------------------------------------------------------------ #
        # Conceptual Idea: 
        # Dynamically map the defined component objects linearly into the neural network's explicit 
        # tracked state parameters, repeating the central message passing sequence exactly as many 
        # times as requested to achieve the designated target graph traversal propagation depth.
        self.execution_pipeline_module = neural_network.ModuleList(
            pre_repeat_steps +
            repeat_steps * number_of_propagation_steps +
            post_repeat_steps
            )
        

    # ---------------------------------------------------------------------- #
    # Forward pass – unchanged except we iterate over self.pipe
    # ---------------------------------------------------------------------- #
    def forward(self, graph_data: Data) -> Union[Data, torch.Tensor]:
        cached_local_node_features_tensor = cached_global_node_features_tensor = None

        for pipeline_module in self.execution_pipeline_module:
            covered_stages_tuple = getattr(pipeline_module, "covers", (None,))

            # Special case: hybrid router blends two node tensors
            if "hybrid_router" in covered_stages_tuple:
                graph_data.x = pipeline_module(cached_local_node_features_tensor, cached_global_node_features_tensor)
                continue

            # Standard stage: consume/return a Data object
            graph_data = pipeline_module(graph_data)

            # Cache intermediate node states if required
            if "local_aggregator" in covered_stages_tuple:
                cached_local_node_features_tensor = graph_data.x.clone()
            if "global_attention" in covered_stages_tuple:
                cached_global_node_features_tensor = graph_data.x.clone()

        # The last stage decides what is returned:
        #   • a Data object with predictions     → when task_head is a normal head
        #   • a tensor (embedding)               → when task_head is GraphVectorizer
        return graph_data
    
# =========================================================================== #
# GNN Variants:
# =========================================================================== #

# =========================================================================== #
# Pre-Built Architecture Variants (updated for current class names/signatures)
#   Notes:
#   • NodeEncoder / EdgeEncoder now take (out_dim, scheme="affine", input_dimension=…)
#     so we pass named args: output_dimension=hidden, input_dimension=<feature-dim>.
#   • Identity_Transformation  →  IdentityTransformation
#   • EdgeMLP (old)            →  EdgeMessageUpdate(FFN(...))
#   • EdgeAttention API        →  use DefaultEdgeAttention(node_dim, heads, dropout)

# =========================================================================== #


def construct_edge_multilayer_perceptron_message_function(node_feature_dimension: int, edge_feature_dimension: int, hidden_layer_dimension: int) -> EdgeMessageUpdate:
    """
    Build a message function that maps [x_src || x_dst || e] → e'
    with output dimension == node_dim so LocalAggregator can yield node_dim features.
    """
    # Conceptual Idea: 
    # Defines a custom multi-layer perceptron to explicitly interpret features mapped 
    # directly between two node points combined directly alongside the bridging connecting link.
    calculated_input_dimension = (2 * node_feature_dimension) + edge_feature_dimension
    feed_forward_network_module = retrieve_core_component("ffn")(
        input_dimension=calculated_input_dimension,
        # last layer = node_dim so that edge_attr has dimension==node_dim for aggregation
        hidden_dimensions=[hidden_layer_dimension, hidden_layer_dimension, node_feature_dimension],
        use_batch_normalizations=False,
        dropout_probabilities=0.0,
        activation_functions="relu",
        local_network_name="edge_mlp",
    )
    return EdgeMessageUpdate(target_update_function=feed_forward_network_module)

@register("core", "gcnn")
def configure_graph_convolutional_neural_network_variant(*, number_of_node_features: int, number_of_edge_features: int,
         hidden_layer_dimension: int, number_of_tasks: int = 1) -> GraphNeuralNetwork:

    # Conceptual Idea: 
    # A standard graph convolutional network configuration composed modularly using our abstracted stages.
    stage_mapping_dictionary = {
        "preprocessor":       GraphPreprocessor(),
        "node_embedder":      NodeEncoder(output_dimension=hidden_layer_dimension, input_dimension=number_of_node_features, mapping_scheme_name="affine"),
        "edge_embedder":      EdgeEncoder(output_dimension=hidden_layer_dimension, input_dimension=number_of_edge_features,   mapping_scheme_name="affine"),
        "structural_encoder": IdentityTransformation(),

        "message_fn":         construct_edge_multilayer_perceptron_message_function(node_feature_dimension=hidden_layer_dimension, edge_feature_dimension=hidden_layer_dimension, hidden_layer_dimension=hidden_layer_dimension),
        "edge_attention":     IdentityTransformation(),
        "local_aggregator":   LocalAggregator("sum"),
        "global_attention":   IdentityTransformation(),
        "hybrid_router":      HybridMixer(),
        "node_update":        NodeUpdate(hidden_layer_dimension),
        "edge_update":        EdgeUpdate(node_feature_dimension=hidden_layer_dimension, edge_feature_dimension=hidden_layer_dimension, hidden_dimension=hidden_layer_dimension),

        "feed_forward":       IdentityTransformation(),
        "norm_reg":           IdentityTransformation(),
        "readout_pool":       ReadoutPooler("sum"),
        "multiscale_agg":     IdentityTransformation(),
        "task_head":          GraphVectorizer(input_dimension=hidden_layer_dimension),
    }
    return GraphNeuralNetwork(stage_mapping_dictionary)

@register("core", "mpnn")
def configure_message_passing_neural_network_variant(*, number_of_node_features: int, number_of_edge_features: int,
         hidden_layer_dimension: int, number_of_tasks: int = 1) -> GraphNeuralNetwork:

    stage_mapping_dictionary = {
        "preprocessor":       GraphPreprocessor(),
        "node_embedder":      NodeEncoder(output_dimension=hidden_layer_dimension, input_dimension=number_of_node_features, mapping_scheme_name="affine"),
        "edge_embedder":      EdgeEncoder(output_dimension=hidden_layer_dimension, input_dimension=number_of_edge_features,   mapping_scheme_name="affine"),
        "structural_encoder": IdentityTransformation(),

        "message_fn":         construct_edge_multilayer_perceptron_message_function(node_feature_dimension=hidden_layer_dimension, edge_feature_dimension=hidden_layer_dimension, hidden_layer_dimension=hidden_layer_dimension),
        "edge_attention":     IdentityTransformation(),
        "local_aggregator":   LocalAggregator("sum"),
        "global_attention":   IdentityTransformation(),
        "hybrid_router":      HybridMixer(),
        "node_update":        NodeUpdate(hidden_layer_dimension),
        "edge_update":        IdentityTransformation(),  # classic MPNN variant keeps this as identity

        "feed_forward":       IdentityTransformation(),
        "norm_reg":           IdentityTransformation(),
        "readout_pool":       ReadoutPooler("sum"),
        "multiscale_agg":     IdentityTransformation(),
        "task_head":          GraphVectorizer(input_dimension=hidden_layer_dimension),
    }
    return GraphNeuralNetwork(stage_mapping_dictionary)

@register("core", "dmpnn")
def configure_directed_message_passing_neural_network_variant(*, number_of_node_features: int, number_of_edge_features: int,
          hidden_layer_dimension: int, number_of_tasks: int = 1) -> GraphNeuralNetwork:

    stage_mapping_dictionary = {
        "preprocessor":       GraphPreprocessor(),
        "node_embedder":      NodeEncoder(output_dimension=hidden_layer_dimension, input_dimension=number_of_node_features, mapping_scheme_name="affine"),
        "edge_embedder":      EdgeEncoder(output_dimension=hidden_layer_dimension, input_dimension=number_of_edge_features,   mapping_scheme_name="affine"),
        "structural_encoder": IdentityTransformation(),

        "message_fn":         construct_edge_multilayer_perceptron_message_function(node_feature_dimension=hidden_layer_dimension, edge_feature_dimension=hidden_layer_dimension, hidden_layer_dimension=hidden_layer_dimension),
        "edge_attention":     IdentityTransformation(),
        "local_aggregator":   DMPNNLocalAggregator(export_node_features_flag=True),
        "global_attention":   IdentityTransformation(),
        "hybrid_router":      HybridMixer(),
        "node_update":        NodeUpdate(hidden_layer_dimension),
        "edge_update":        EdgeUpdate(node_feature_dimension=hidden_layer_dimension, edge_feature_dimension=hidden_layer_dimension, hidden_dimension=hidden_layer_dimension),

        "feed_forward":       IdentityTransformation(),
        "norm_reg":           IdentityTransformation(),
        "readout_pool":       ReadoutPooler("sum"),
        "multiscale_agg":     IdentityTransformation(),
        "task_head":          GraphVectorizer(input_dimension=hidden_layer_dimension),
    }
    return GraphNeuralNetwork(stage_mapping_dictionary)

@register("core", "chemprop")
def configure_chemprop_variant(
             edge_feature_dimension: int,
             message_aggregation_method_name = "sum",
             readout_pool_aggregation_method_name = "sum",
             node_embedding_dropout_probability = 0,
             edge_embedding_dropout_probability = 0,
             edge_update_dropout_probability = 0,
             depth: int = 2
             ) -> GraphNeuralNetwork:
    
    number_of_propagation_steps = depth - 1
    
    stage_mapping_dictionary = {
        "preprocessor":       GraphPreprocessor(),
        "node_embedder":      ChemPropNodeEncoder(active_dropout_probability=node_embedding_dropout_probability, mapping_scheme_name="identity"),
        "edge_embedder":      ChemPropEdgeEncoder(output_dimension=edge_feature_dimension, dropout_probability=edge_embedding_dropout_probability, mapping_scheme_name="affine"),
        "structural_encoder": IdentityTransformation(),

        "message_fn":         IdentityTransformation(),
        "edge_attention":     IdentityTransformation(),
        "local_aggregator":   DMPNNLocalAggregator(reduction_method_name=message_aggregation_method_name),
        "global_attention":   IdentityTransformation(),
        "hybrid_router":      IdentityTransformation(),
        "node_update":        IdentityTransformation(),
        "edge_update":        ChemPropEdgeUpdate(edge_feature_dimension=edge_feature_dimension, active_dropout_probability=edge_update_dropout_probability),
        "feed_forward":       IdentityTransformation(),
        "norm_reg":           IdentityTransformation(),
        
        # FIX: pass node_dim (152) and dropout to the readout pooler appropriately
        "readout_pool":       ChempropReadoutPooler(edge_feature_dimension=edge_feature_dimension, node_feature_dimension=152, reduction_method_name=readout_pool_aggregation_method_name, active_dropout_probability=node_embedding_dropout_probability),
        "multiscale_agg":     IdentityTransformation(),
        "task_head":          GraphVectorizer(input_dimension=edge_feature_dimension),
    }
    return GraphNeuralNetwork(stage_mapping_dictionary, number_of_propagation_steps=number_of_propagation_steps)

@register("core", "attfp")
def configure_attentive_fingerprint_variant(*, number_of_node_features: int, number_of_edge_features: int,
          hidden_layer_dimension: int, number_of_tasks: int = 1,
          heads: int = 4, dropout: float = 0.1) -> GraphNeuralNetwork:

    stage_mapping_dictionary = {
        "preprocessor":       GraphPreprocessor(),
        "node_embedder":      NodeEncoder(output_dimension=hidden_layer_dimension, input_dimension=number_of_node_features, mapping_scheme_name="affine"),
        "edge_embedder":      EdgeEncoder(output_dimension=hidden_layer_dimension, input_dimension=number_of_edge_features,   mapping_scheme_name="affine"),
        "structural_encoder": IdentityTransformation(),

        "message_fn":         IdentityTransformation(),                                   # edges already embedded
        "edge_attention":     DefaultEdgeAttention(node_feature_dimension=hidden_layer_dimension, number_of_attention_heads=heads, active_dropout_probability=dropout),
        "local_aggregator":   LocalAggregator("sum"),
        "global_attention":   IdentityTransformation(),
        "hybrid_router":      HybridMixer(),
        "node_update":        NodeUpdate(hidden_layer_dimension),
        "edge_update":        IdentityTransformation(),

        "feed_forward":       IdentityTransformation(),
        "norm_reg":           IdentityTransformation(),
        "readout_pool":       ReadoutPooler("sum"),
        "multiscale_agg":     IdentityTransformation(),
        "task_head":          GraphVectorizer(input_dimension=hidden_layer_dimension),
    }
    return GraphNeuralNetwork(stage_mapping_dictionary)

@register("core", "gin")
def configure_graph_isomorphism_network_variant(*, number_of_node_features: int,
        hidden_layer_dimension: int, number_of_tasks: int = 1) -> GraphNeuralNetwork:
    """
    Simple GIN-style setup within this staged framework:
    - no dedicated edge features (we keep the 1‑dim zeros from the preprocessor),
    - message function constructs edge messages from node pairs,
    - sum aggregate.
    """
    # Edge preprocessor leaves E×1 zeros; make message fn output node_dim
    initial_edge_input_dimension = 1

    stage_mapping_dictionary = {
        "preprocessor":       GraphPreprocessor(),
        "node_embedder":      NodeEncoder(output_dimension=hidden_layer_dimension, input_dimension=number_of_node_features, mapping_scheme_name="affine"),
        "edge_embedder":      IdentityTransformation(),  # keep E×1 zeros coming from the preprocessor
        "structural_encoder": IdentityTransformation(),

        # input_dimension = 2*hidden + 1; output_dimension = hidden (last FFN size)
        "message_fn":         construct_edge_multilayer_perceptron_message_function(node_feature_dimension=hidden_layer_dimension, edge_feature_dimension=initial_edge_input_dimension, hidden_layer_dimension=hidden_layer_dimension),
        "edge_attention":     IdentityTransformation(),
        "local_aggregator":   LocalAggregator("sum"),
        "global_attention":   IdentityTransformation(),
        "hybrid_router":      HybridMixer(),
        "node_update":        NodeUpdate(hidden_layer_dimension),
        "edge_update":        IdentityTransformation(),

        "feed_forward":       IdentityTransformation(),
        "norm_reg":           IdentityTransformation(),
        "readout_pool":       ReadoutPooler("sum"),
        "multiscale_agg":     IdentityTransformation(),
        "task_head":          GraphVectorizer(input_dimension=hidden_layer_dimension),
    }
    return GraphNeuralNetwork(stage_mapping_dictionary)

@register("core", "gtr")
def configure_graph_transformer_variant(*, number_of_node_features: int, number_of_edge_features: int,
        hidden_layer_dimension: int, number_of_tasks: int = 1,
        heads: int = 8, dropout: float = 0.1) -> GraphNeuralNetwork:

    stage_mapping_dictionary = {
        "preprocessor":       GraphPreprocessor(),
        "node_embedder":      NodeEncoder(output_dimension=hidden_layer_dimension, input_dimension=number_of_node_features, mapping_scheme_name="affine"),
        "edge_embedder":      EdgeEncoder(output_dimension=hidden_layer_dimension, input_dimension=number_of_edge_features,   mapping_scheme_name="affine"),
        "structural_encoder": IdentityTransformation(),

        "message_fn":         IdentityTransformation(),
        "edge_attention":     IdentityTransformation(),
        "local_aggregator":   IdentityTransformation(),
        "global_attention":   GlobalSelfAttention(expected_feature_dimension=hidden_layer_dimension, number_of_attention_heads=heads, active_dropout_probability=dropout),
        "hybrid_router":      HybridMixer(),
        "node_update":        NodeUpdate(hidden_layer_dimension),
        "edge_update":        IdentityTransformation(),

        "feed_forward":       IdentityTransformation(),
        "norm_reg":           IdentityTransformation(),
        "readout_pool":       ReadoutPooler("sum"),
        "multiscale_agg":     IdentityTransformation(),
        "task_head":          GraphVectorizer(input_dimension=hidden_layer_dimension),
    }
    return GraphNeuralNetwork(stage_mapping_dictionary)

# ########################################################################### #
# PBPK MODELING BLOCK                                                         #
#                                                                             #
# Reading order:                                                              #
#   1.  Raw physiology lookup tables (GastroPlus historical, PK-Sim)          #
#   2.  Physiology harmonization (raw -> canonical schema the ODE reads)      #
#   3.  Species lookups (physiology / GFR / MPPGL) and accessors              #
#   4.  ODE state-layout helpers                                              #
#   5.  Partition-coefficient method                                          #
#   6.  PBPK ODE right-hand side                                              #
#   7.  Numerical ODE integration (torchode)                                  #
#   8.  Closed-form AUC, single species (exact linear solve)                  #
#   9.  Neural drug-parameter prediction head                                 #
#   10. Unit-scaling nodes (single species)                                   #
#   11. Multi-species dispatch (one-hot, scale, closed-form)                  #
#   12. Trajectory-based path (trapezoidal AUC, simulator, organ selector)    #
# ########################################################################### #


# =========================================================================== #
# 1. RAW PHYSIOLOGY LOOKUP TABLES                                             #
# =========================================================================== #

# --------------------------------------------------------------------------- #
# 1a. GastroPlus tables (HISTORICAL).
#     Kept for provenance only. The two-pool ODE consumes the PK-Sim schema
#     (vascular / extracellular / intracellular volumes, surface areas, and the
#     canonical "blood_flow" key), none of which these tables provide, so they
#     are never fed to the ODE. GastroPlus is valid only in the non-perfusion-
#     limited mode; PK-Sim parameters are the standard ones used in simulation.
# --------------------------------------------------------------------------- #

human_physiology_from_gastroplus_pear = {
    'lung': {'volume': 1141.0, 'perfusion_rate': 383.2},
    'arterial supply': {'volume': 2228.0, 'perfusion_rate': 383.2},
    'venous return': {'volume': 4456.0, 'perfusion_rate': 383.2},
    'adipose': {'volume': 31100.0, 'perfusion_rate': 37.28},
    'muscle': {'volume': 27620.0, 'perfusion_rate': 49.72},
    'liver': {'volume': 1708.0, 'perfusion_rate': 94.16},
    'hepatic artery': {'volume': 0.0, 'perfusion_rate': 33.67},
    'tissue gut': {'volume': 0.0, 'perfusion_rate': 50.29},
    'spleen': {'volume': 170.0, 'perfusion_rate': 10.2},
    'heart': {'volume': 367.8, 'perfusion_rate': 16.11},
    'brain': {'volume': 1493.0, 'perfusion_rate': 45.67},
    'kidney': {'volume': 384.3, 'perfusion_rate': 84.86},
    'skin': {'volume': 3037.0, 'perfusion_rate': 21.87},
    'reproductive organ': {'volume': 57.66, 'perfusion_rate': 0.7265},
    'red marrow': {'volume': 1185.0, 'perfusion_rate': 21.33},
    'yellow marrow': {'volume': 3293.0, 'perfusion_rate': 5.928},
    'other tissues': {'volume': 3054.0, 'perfusion_rate': 5.498},
}

human_physiology_from_gastroplus_v510 = {
    'lung': {'volume': 1.11, 'perfusion_rate': 6.08},
    'spleen': {'volume': 0.184, 'perfusion_rate': 0.184},
    'liver': {'volume': 1.63, 'perfusion_rate': 1.5},
    'acat gut': {'volume': None, 'perfusion_rate': 0.836},
    'adipose': {'volume': 30.3, 'perfusion_rate': 0.605},
    'muscle': {'volume': 20.7, 'perfusion_rate': 0.622},
    'heart': {'volume': 0.315, 'perfusion_rate': 0.23},
    'brain': {'volume': 1.73, 'perfusion_rate': 0.882},
    'kidney': {'volume': 0.277, 'perfusion_rate': 1.02},
    'skin': {'volume': 1.96, 'perfusion_rate': 0.235},
    'testes': {'volume': 0.032, 'perfusion_rate': 0.007},
    'red marrow': {'volume': 1.18, 'perfusion_rate': 0.354},
    'yellow marrow': {'volume': 3.28, 'perfusion_rate': 0.098},
    'rest of body': {'volume': 17.6, 'perfusion_rate': 0.529},
    'arterial blood': {'volume': 2.21, 'perfusion_rate': 6.08},
    'venous blood': {'volume': 4.42, 'perfusion_rate': 6.08},
}

# --------------------------------------------------------------------------- #
# 1b. PK-Sim tables (PRODUCTION). Open Systems Pharmacology Suite physiology.
#     Human defaults to the male table; the female table is available for a sex
#     split. PK-Sim does not sex-differentiate the animal species.
# --------------------------------------------------------------------------- #

human_physiology_from_open_icrp_pksim_male = {'arterial blood': {'total_volume_L': 0.419,
                    'perfusion_rate_L_per_min': None,
                    'vascular_volume_L': 0.419,
                    'extracellular_volume_L': 0.0,
                    'intracellular_volume_L': 0.0,
                    'surface_area_m2': None},
 'venous blood': {'total_volume_L': 0.964,
                  'perfusion_rate_L_per_min': None,
                  'vascular_volume_L': 0.964,
                  'extracellular_volume_L': 0.0,
                  'intracellular_volume_L': 0.0,
                  'surface_area_m2': None},
 'portal vein': {'total_volume_L': 1.037,
                 'perfusion_rate_L_per_min': None,
                 'vascular_volume_L': 1.037,
                 'extracellular_volume_L': 0.0,
                 'intracellular_volume_L': 0.0,
                 'surface_area_m2': None},
 'adipose': {'total_volume_L': 14.65,
             'perfusion_rate_L_per_min': 0.32,
             'vascular_volume_L': 0.2637,
             'extracellular_volume_L': 2.344,
             'intracellular_volume_L': 12.0423,
             'surface_area_m2': 25.0515},
 'brain': {'total_volume_L': 1.509,
           'perfusion_rate_L_per_min': 0.78,
           'vascular_volume_L': 0.058851,
           'extracellular_volume_L': 0.006036,
           'intracellular_volume_L': 1.444113,
           'surface_area_m2': 5.590845},
 'bone': {'total_volume_L': 11.82,
          'perfusion_rate_L_per_min': 0.325,
          'vascular_volume_L': 0.40188,
          'extracellular_volume_L': 1.182,
          'intracellular_volume_L': 10.23612,
          'surface_area_m2': 38.1786},
 'gonads': {'total_volume_L': 0.04,
            'perfusion_rate_L_per_min': 0.003,
            'vascular_volume_L': 0.0022,
            'extracellular_volume_L': 0.00276,
            'intracellular_volume_L': 0.03504,
            'surface_area_m2': 0.209},
 'heart': {'total_volume_L': 0.417,
           'perfusion_rate_L_per_min': 0.26,
           'vascular_volume_L': 0.05838,
           'extracellular_volume_L': 0.0417,
           'intracellular_volume_L': 0.31692,
           'surface_area_m2': 5.5461},
 'kidney': {'total_volume_L': 0.438,
            'perfusion_rate_L_per_min': 1.327,
            'vascular_volume_L': 0.10074,
            'extracellular_volume_L': 0.0876,
            'intracellular_volume_L': 0.24966,
            'surface_area_m2': 9.5703},
 'liver': {'total_volume_L': 2.377,
           'perfusion_rate_L_per_min': 0.426,
           'vascular_volume_L': 0.40409,
           'extracellular_volume_L': 0.387451,
           'intracellular_volume_L': 1.585459,
           'surface_area_m2': 38.38855},
 'large intestine': {'total_volume_L': 0.413,
                     'perfusion_rate_L_per_min': 0.26,
                     'vascular_volume_L': 0.009912,
                     'extracellular_volume_L': 0.038822,
                     'intracellular_volume_L': 0.364266,
                     'surface_area_m2': 0.94164},
 'lung': {'total_volume_L': 1.215,
          'perfusion_rate_L_per_min': 6.088,
          'vascular_volume_L': 0.7047,
          'extracellular_volume_L': 0.22842,
          'intracellular_volume_L': 0.28188,
          'surface_area_m2': 66.9465},
 'muscle': {'total_volume_L': 32.65,
            'perfusion_rate_L_per_min': 1.116,
            'vascular_volume_L': 0.81625,
            'extracellular_volume_L': 5.224,
            'intracellular_volume_L': 26.60975,
            'surface_area_m2': 77.54375},
 'pancreas': {'total_volume_L': 0.19,
              'perfusion_rate_L_per_min': 0.065,
              'vascular_volume_L': 0.038,
              'extracellular_volume_L': 0.0228,
              'intracellular_volume_L': 0.1292,
              'surface_area_m2': 3.61},
 'small intestine': {'total_volume_L': 0.725,
                     'perfusion_rate_L_per_min': 0.65,
                     'vascular_volume_L': 0.0174,
                     'extracellular_volume_L': 0.06815,
                     'intracellular_volume_L': 0.63945,
                     'surface_area_m2': 1.653},
 'skin': {'total_volume_L': 3.763,
          'perfusion_rate_L_per_min': 0.325,
          'vascular_volume_L': 0.173098,
          'extracellular_volume_L': 1.136426,
          'intracellular_volume_L': 2.453476,
          'surface_area_m2': 16.44431},
 'spleen': {'total_volume_L': 0.207,
            'perfusion_rate_L_per_min': 0.166,
            'vascular_volume_L': 0.06831,
            'extracellular_volume_L': 0.03105,
            'intracellular_volume_L': 0.10764,
            'surface_area_m2': 6.48945},
 'stomach': {'total_volume_L': 0.169,
             'perfusion_rate_L_per_min': 0.065,
             'vascular_volume_L': 0.005408,
             'extracellular_volume_L': 0.0169,
             'intracellular_volume_L': 0.146692,
             'surface_area_m2': 0.51376}}

human_physiology_from_pksim_female = {'arterial blood': {'total_volume_L': 0.32,
                    'perfusion_rate_L_per_min': None,
                    'vascular_volume_L': 0.32,
                    'extracellular_volume_L': 0.0,
                    'intracellular_volume_L': 0.0,
                    'surface_area_m2': None},
 'venous blood': {'total_volume_L': 0.737,
                  'perfusion_rate_L_per_min': None,
                  'vascular_volume_L': 0.737,
                  'extracellular_volume_L': 0.0,
                  'intracellular_volume_L': 0.0,
                  'surface_area_m2': None},
 'portal vein': {'total_volume_L': 0.793,
                 'perfusion_rate_L_per_min': None,
                 'vascular_volume_L': 0.793,
                 'extracellular_volume_L': 0.0,
                 'intracellular_volume_L': 0.0,
                 'surface_area_m2': None},
 'adipose': {'total_volume_L': 19.42,
             'perfusion_rate_L_per_min': 0.503,
             'vascular_volume_L': 0.34956,
             'extracellular_volume_L': 3.1072,
             'intracellular_volume_L': 15.96324,
             'surface_area_m2': 33.2082},
 'brain': {'total_volume_L': 1.355,
           'perfusion_rate_L_per_min': 0.706,
           'vascular_volume_L': 0.052845,
           'extracellular_volume_L': 0.00542,
           'intracellular_volume_L': 1.296735,
           'surface_area_m2': 5.020275},
 'bone': {'total_volume_L': 9.122,
          'perfusion_rate_L_per_min': 0.295,
          'vascular_volume_L': 0.310148,
          'extracellular_volume_L': 0.9122,
          'intracellular_volume_L': 7.899652,
          'surface_area_m2': 29.46406},
 'gonads': {'total_volume_L': 0.013,
            'perfusion_rate_L_per_min': 0.001,
            'vascular_volume_L': 0.000715,
            'extracellular_volume_L': 0.000897,
            'intracellular_volume_L': 0.011388,
            'surface_area_m2': 0.067925},
 'heart': {'total_volume_L': 0.329,
           'perfusion_rate_L_per_min': 0.295,
           'vascular_volume_L': 0.04606,
           'extracellular_volume_L': 0.0329,
           'intracellular_volume_L': 0.25004,
           'surface_area_m2': 4.3757},
 'kidney': {'total_volume_L': 0.404,
            'perfusion_rate_L_per_min': 1.122,
            'vascular_volume_L': 0.09292,
            'extracellular_volume_L': 0.0808,
            'intracellular_volume_L': 0.23028,
            'surface_area_m2': 8.8274},
 'liver': {'total_volume_L': 1.918,
           'perfusion_rate_L_per_min': 0.386,
           'vascular_volume_L': 0.32606,
           'extracellular_volume_L': 0.312634,
           'intracellular_volume_L': 1.279306,
           'surface_area_m2': 30.9757},
 'large intestine': {'total_volume_L': 0.417,
                     'perfusion_rate_L_per_min': 0.295,
                     'vascular_volume_L': 0.010008,
                     'extracellular_volume_L': 0.039198,
                     'intracellular_volume_L': 0.367794,
                     'surface_area_m2': 0.95076},
 'lung': {'total_volume_L': 0.946,
          'perfusion_rate_L_per_min': 5.482,
          'vascular_volume_L': 0.54868,
          'extracellular_volume_L': 0.177848,
          'intracellular_volume_L': 0.219472,
          'surface_area_m2': 52.1246},
 'muscle': {'total_volume_L': 20.29,
            'perfusion_rate_L_per_min': 0.666,
            'vascular_volume_L': 0.50725,
            'extracellular_volume_L': 3.2464,
            'intracellular_volume_L': 16.53635,
            'surface_area_m2': 48.18875},
 'pancreas': {'total_volume_L': 0.17,
              'perfusion_rate_L_per_min': 0.059,
              'vascular_volume_L': 0.034,
              'extracellular_volume_L': 0.0204,
              'intracellular_volume_L': 0.1156,
              'surface_area_m2': 3.23},
 'small intestine': {'total_volume_L': 0.695,
                     'perfusion_rate_L_per_min': 0.649,
                     'vascular_volume_L': 0.01668,
                     'extracellular_volume_L': 0.06533,
                     'intracellular_volume_L': 0.61299,
                     'surface_area_m2': 1.5846},
 'skin': {'total_volume_L': 2.724,
          'perfusion_rate_L_per_min': 0.296,
          'vascular_volume_L': 0.125304,
          'extracellular_volume_L': 0.822648,
          'intracellular_volume_L': 1.776048,
          'surface_area_m2': 11.90388},
 'spleen': {'total_volume_L': 0.186,
            'perfusion_rate_L_per_min': 0.15,
            'vascular_volume_L': 0.06138,
            'extracellular_volume_L': 0.0279,
            'intracellular_volume_L': 0.09672,
            'surface_area_m2': 5.8311},
 'stomach': {'total_volume_L': 0.163,
             'perfusion_rate_L_per_min': 0.059,
             'vascular_volume_L': 0.005216,
             'extracellular_volume_L': 0.0163,
             'intracellular_volume_L': 0.141484,
             'surface_area_m2': 0.49552}}

mouse_physiology_from_pksim = {
 'arterial blood': {'total_volume_L': 0.000228182,
                    'perfusion_rate_L_per_min': None,
                    'vascular_volume_L': 0.000228182,
                    'extracellular_volume_L': 0.0,
                    'intracellular_volume_L': 0.0,
                    'surface_area_m2': None},
 'venous blood': {'total_volume_L': 0.000524818,
                  'perfusion_rate_L_per_min': None,
                  'vascular_volume_L': 0.000524818,
                  'extracellular_volume_L': 0.0,
                  'intracellular_volume_L': 0.0,
                  'surface_area_m2': None},
 'portal vein': {'total_volume_L': 0.00015,
                 'perfusion_rate_L_per_min': None,
                 'vascular_volume_L': 0.00015,
                 'extracellular_volume_L': 0.0,
                 'intracellular_volume_L': 0.0,
                 'surface_area_m2': None},
 'adipose': {'total_volume_L': 0.001,
             'perfusion_rate_L_per_min': 4e-05,
             'vascular_volume_L': 1e-05,
             'extracellular_volume_L': 0.000135,
             'intracellular_volume_L': 0.000855,
             'surface_area_m2': 0.00095},
 'brain': {'total_volume_L': 0.00017,
           'perfusion_rate_L_per_min': 0.00013,
           'vascular_volume_L': 6.29e-06,
           'extracellular_volume_L': 6.8e-07,
           'intracellular_volume_L': 0.00016303,
           'surface_area_m2': 0.00059755},
 'bone': {'total_volume_L': 0.00158,
          'perfusion_rate_L_per_min': 0.000253,
          'vascular_volume_L': 6.478e-05,
          'extracellular_volume_L': 0.000158,
          'intracellular_volume_L': 0.00135722,
          'surface_area_m2': 0.0061541},
 'gonads': {'total_volume_L': 0.00025,
            'perfusion_rate_L_per_min': 4.8e-05,
            'vascular_volume_L': 3.5e-05,
            'extracellular_volume_L': 1.725e-05,
            'intracellular_volume_L': 0.00019775,
            'surface_area_m2': 0.003325},
 'heart': {'total_volume_L': 9.5e-05,
           'perfusion_rate_L_per_min': 0.00028,
           'vascular_volume_L': 2.489e-05,
           'extracellular_volume_L': 9.5e-06,
           'intracellular_volume_L': 6.061e-05,
           'surface_area_m2': 0.00236455},
 'kidney': {'total_volume_L': 0.00034,
            'perfusion_rate_L_per_min': 0.0013,
            'vascular_volume_L': 3.57e-05,
            'extracellular_volume_L': 6.8e-05,
            'intracellular_volume_L': 0.0002363,
            'surface_area_m2': 0.0033915},
 'liver': {'total_volume_L': 0.0013,
           'perfusion_rate_L_per_min': 0.00035,
           'vascular_volume_L': 0.0001495,
           'extracellular_volume_L': 0.0002119,
           'intracellular_volume_L': 0.0009386,
           'surface_area_m2': 0.0142025},
 'large intestine': {'total_volume_L': 0.000627,
                     'perfusion_rate_L_per_min': 0.0005,
                     'vascular_volume_L': 1.5048e-05,
                     'extracellular_volume_L': 5.8938e-05,
                     'intracellular_volume_L': 0.000553014,
                     'surface_area_m2': 0.00142956},
 'lung': {'total_volume_L': 0.0001,
          'perfusion_rate_L_per_min': 0.005473,
          'vascular_volume_L': 6.26e-05,
          'extracellular_volume_L': 1.88e-05,
          'intracellular_volume_L': 1.86e-05,
          'surface_area_m2': 0.005947},
 'muscle': {'total_volume_L': 0.01,
            'perfusion_rate_L_per_min': 0.00091,
            'vascular_volume_L': 0.00026,
            'extracellular_volume_L': 0.0012,
            'intracellular_volume_L': 0.00854,
            'surface_area_m2': 0.0247},
 'pancreas': {'total_volume_L': 0.00013,
              'perfusion_rate_L_per_min': 5.2e-05,
              'vascular_volume_L': 2.34e-05,
              'extracellular_volume_L': 1.56e-05,
              'intracellular_volume_L': 9.1e-05,
              'surface_area_m2': 0.002223},
 'small intestine': {'total_volume_L': 0.001406,
                     'perfusion_rate_L_per_min': 0.001,
                     'vascular_volume_L': 3.3744e-05,
                     'extracellular_volume_L': 0.000132164,
                     'intracellular_volume_L': 0.00124009,
                     'surface_area_m2': 0.00320568},
 'skin': {'total_volume_L': 0.0029,
          'perfusion_rate_L_per_min': 0.00041,
          'vascular_volume_L': 5.51e-05,
          'extracellular_volume_L': 0.0008758,
          'intracellular_volume_L': 0.0019691,
          'surface_area_m2': 0.0052345},
 'spleen': {'total_volume_L': 0.0001,
            'perfusion_rate_L_per_min': 9e-05,
            'vascular_volume_L': 2.82e-05,
            'extracellular_volume_L': 1.5e-05,
            'intracellular_volume_L': 5.68e-05,
            'surface_area_m2': 0.002679},
 'stomach': {'total_volume_L': 0.00011,
             'perfusion_rate_L_per_min': 0.00011,
             'vascular_volume_L': 3.52e-06,
             'extracellular_volume_L': 1.1e-05,
             'intracellular_volume_L': 9.548e-05,
             'surface_area_m2': 0.0003344}}

rat_physiology_from_pksim = {
 'arterial blood': {'total_volume_L': 0.00293939,
                    'perfusion_rate_L_per_min': None,
                    'vascular_volume_L': 0.00293939,
                    'extracellular_volume_L': 0.0,
                    'intracellular_volume_L': 0.0,
                    'surface_area_m2': None},
 'venous blood': {'total_volume_L': 0.00676061,
                  'perfusion_rate_L_per_min': None,
                  'vascular_volume_L': 0.00676061,
                  'extracellular_volume_L': 0.0,
                  'intracellular_volume_L': 0.0,
                  'surface_area_m2': None},
 'portal vein': {'total_volume_L': 0.0015,
                 'perfusion_rate_L_per_min': None,
                 'vascular_volume_L': 0.0015,
                 'extracellular_volume_L': 0.0,
                 'intracellular_volume_L': 0.0,
                 'surface_area_m2': None},
 'adipose': {'total_volume_L': 0.01,
             'perfusion_rate_L_per_min': 0.0004,
             'vascular_volume_L': 0.0001,
             'extracellular_volume_L': 0.00135,
             'intracellular_volume_L': 0.00855,
             'surface_area_m2': 0.0095},
 'brain': {'total_volume_L': 0.0017,
           'perfusion_rate_L_per_min': 0.00133333,
           'vascular_volume_L': 6.29e-05,
           'extracellular_volume_L': 6.8e-06,
           'intracellular_volume_L': 0.0016303,
           'surface_area_m2': 0.0059755},
 'bone': {'total_volume_L': 0.0158,
          'perfusion_rate_L_per_min': 0.00253333,
          'vascular_volume_L': 0.0006478,
          'extracellular_volume_L': 0.00158,
          'intracellular_volume_L': 0.0135722,
          'surface_area_m2': 0.061541},
 'gonads': {'total_volume_L': 0.0025,
            'perfusion_rate_L_per_min': 0.00048,
            'vascular_volume_L': 0.00035,
            'extracellular_volume_L': 0.0001725,
            'intracellular_volume_L': 0.0019775,
            'surface_area_m2': 0.03325},
 'heart': {'total_volume_L': 0.0008,
           'perfusion_rate_L_per_min': 0.00391667,
           'vascular_volume_L': 0.0002096,
           'extracellular_volume_L': 8e-05,
           'intracellular_volume_L': 0.0005104,
           'surface_area_m2': 0.019912},
 'kidney': {'total_volume_L': 0.0023,
            'perfusion_rate_L_per_min': 0.00923333,
            'vascular_volume_L': 0.0002415,
            'extracellular_volume_L': 0.00046,
            'intracellular_volume_L': 0.0015985,
            'surface_area_m2': 0.0229425},
 'liver': {'total_volume_L': 0.0103,
           'perfusion_rate_L_per_min': 0.002,
           'vascular_volume_L': 0.0011845,
           'extracellular_volume_L': 0.0016789,
           'intracellular_volume_L': 0.0074366,
           'surface_area_m2': 0.112528},
 'large intestine': {'total_volume_L': 0.002177,
                     'perfusion_rate_L_per_min': 0.005,
                     'vascular_volume_L': 5.2248e-05,
                     'extracellular_volume_L': 0.000204638,
                     'intracellular_volume_L': 0.00192011,
                     'surface_area_m2': 0.00496356},
 'lung': {'total_volume_L': 0.001,
          'perfusion_rate_L_per_min': 0.0430133,
          'vascular_volume_L': 0.000626,
          'extracellular_volume_L': 0.000188,
          'intracellular_volume_L': 0.000186,
          'surface_area_m2': 0.05947},
 'muscle': {'total_volume_L': 0.122,
            'perfusion_rate_L_per_min': 0.0075,
            'vascular_volume_L': 0.003172,
            'extracellular_volume_L': 0.01464,
            'intracellular_volume_L': 0.104188,
            'surface_area_m2': 0.30134},
 'pancreas': {'total_volume_L': 0.0013,
              'perfusion_rate_L_per_min': 0.000516667,
              'vascular_volume_L': 0.000234,
              'extracellular_volume_L': 0.000156,
              'intracellular_volume_L': 0.00091,
              'surface_area_m2': 0.02223},
 'small intestine': {'total_volume_L': 0.005,
                     'perfusion_rate_L_per_min': 0.0025,
                     'vascular_volume_L': 0.00012,
                     'extracellular_volume_L': 0.00047,
                     'intracellular_volume_L': 0.00441,
                     'surface_area_m2': 0.0114},
 'skin': {'total_volume_L': 0.04,
          'perfusion_rate_L_per_min': 0.00583333,
          'vascular_volume_L': 0.00076,
          'extracellular_volume_L': 0.01208,
          'intracellular_volume_L': 0.02716,
          'surface_area_m2': 0.0722},
 'spleen': {'total_volume_L': 0.0006,
            'perfusion_rate_L_per_min': 0.000633333,
            'vascular_volume_L': 0.0001692,
            'extracellular_volume_L': 9e-05,
            'intracellular_volume_L': 0.0003408,
            'surface_area_m2': 0.016074},
 'stomach': {'total_volume_L': 0.0011,
             'perfusion_rate_L_per_min': 0.00113333,
             'vascular_volume_L': 3.52e-05,
             'extracellular_volume_L': 0.00011,
             'intracellular_volume_L': 0.0009548,
             'surface_area_m2': 0.003344}}


# =========================================================================== #
# 2. PHYSIOLOGY HARMONIZATION (raw PK-Sim table -> canonical schema)          #
# =========================================================================== #
# The ODE reads a canonical schema: per-organ "volume" and "blood_flow" (plus
# vascular / extracellular / intracellular volumes and surface_area for the
# permeability-limited model). Raw PK-Sim keys are renamed on load; the raw
# table literals above are left untouched so they still document their source.

PKSIM_KEY_RENAME = {
    "total_volume_L": "volume",
    "perfusion_rate_L_per_min": "blood_flow",
    "vascular_volume_L": "vascular_volume",
    "extracellular_volume_L": "extracellular_volume",
    "intracellular_volume_L": "intracellular_volume",
    "surface_area_m2": "surface_area",
}
PKSIM_ORGAN_RENAME = {"arterial blood": "arterial_blood", "venous blood": "venous_blood"}


def harmonize_physiology_table(raw_physiology, dropped_compartments=("portal vein",)):
    """Map PK-Sim physiology to the model schema.

    Retain arterial and venous blood pools, omit the portal-vein compartment,
    and default the permeability scaling factor to one.
    """
    dropped = set(dropped_compartments)
    harmonized = {}
    for organ_name, organ_properties in raw_physiology.items():
        if organ_name in dropped:
            continue
        canonical = {PKSIM_KEY_RENAME.get(key, key): value for key, value in organ_properties.items()}
        canonical_name = PKSIM_ORGAN_RENAME.get(organ_name, organ_name.replace(" ", "_"))
        if canonical_name in ("arterial_blood", "venous_blood"):
            # Blood pools carry only a volume in the ODE state (no tissue sub-volumes).
            harmonized[canonical_name] = {"volume": canonical["volume"]}
        else:
            canonical.setdefault("permeability_scaling_factor", 1.0)
            harmonized[canonical_name] = canonical
    return harmonized


canonical_human_physiology_pksim_male   = harmonize_physiology_table(human_physiology_from_open_icrp_pksim_male)
canonical_human_physiology_pksim_female = harmonize_physiology_table(human_physiology_from_pksim_female)
canonical_mouse_physiology_pksim        = harmonize_physiology_table(mouse_physiology_from_pksim)
canonical_rat_physiology_pksim          = harmonize_physiology_table(rat_physiology_from_pksim)


# --------------------------------------------------------------------------- #
# GastroPlus PEAR unit conversion (HISTORICAL).
# Puts the PEAR table into SI-ish units (volumes in L, perfusion in L/min). Not
# consumed by the ODE (see the note on the GastroPlus tables above); kept only as
# documentation of provenance.
# --------------------------------------------------------------------------- #

PEAR_VOLUME_ML_TO_L = 1.0 / 1000.0           # mL  -> L
PEAR_FLOW_L_PER_H_TO_L_PER_MIN = 1.0 / 60.0  # L/h -> L/min


def convert_pear_units(pear_table):
    """Return the PEAR physiology with volumes in L and perfusion in L/min.

    The original dictionary is left intact; this builds and returns a new one.
    """
    return {
        organ: {
            'volume': values['volume'] * PEAR_VOLUME_ML_TO_L,            # mL  -> L
            'perfusion_rate': values['perfusion_rate'] * PEAR_FLOW_L_PER_H_TO_L_PER_MIN,  # L/h -> L/min
        }
        for organ, values in pear_table.items()
    }


human_physiology_from_gastroplus_pear_SI = convert_pear_units(human_physiology_from_gastroplus_pear)


# =========================================================================== #
# 3. SPECIES LOOKUPS (physiology table / glomerular filtration rate / MPPGL)   #
# =========================================================================== #
# A drug measured in a given species must be simulated with that species'
# physiology, renal filtration rate, and microsomal scale-up factor. These three
# tables plus their accessors are the single place those per-species constants
# live; the multi-species nodes below resolve everything through them.

# Microsomal protein per gram of liver (mg protein / g liver), for the
# microsomal -> whole-liver intrinsic-clearance scale-up. Keyed by binomial name.
MPPGL = {
    'Homo sapiens'          : 39.5,  # Zhang et al., Sci. Rep. 2015, 5, 17671.
    'Mus musculus'          : 45,    # cross-species default; Houston, Biochem. Pharmacol. 1994, 47, 1469.
    'Rattus norvegicus'     : 45,    # Houston, Biochem. Pharmacol. 1994, 47, 1469.
    'Canis lupus familiaris': 55,    # Smith et al., Xenobiotica 2008, 38, 1386.
    'Macaca fascicularis'   : 45,    # cross-species default; Houston, Biochem. Pharmacol. 1994, 47, 1469.
    'Macaca mulatta'        : 45,    # cross-species default; Houston, Biochem. Pharmacol. 1994, 47, 1469.
    'Sus scrofa'            : 32.6,  # Achour et al., Drug Metab. Dispos. 2011, 39, 2130.
    'Bos taurus'            : 45,    # cross-species default; Houston, Biochem. Pharmacol. 1994, 47, 1469.
    'Not specified'         : 45,    # cross-species default; Houston, Biochem. Pharmacol. 1994, 47, 1469.
    'Sus scrofa domesticus' : 32.6,  # Achour et al., Drug Metab. Dispos. 2011, 39, 2130.
    'Oryctolagus cuniculus' : 45,    # cross-species default; Houston, Biochem. Pharmacol. 1994, 47, 1469.
    'Cavia porcellus'       : 45,    # cross-species default; Houston, Biochem. Pharmacol. 1994, 47, 1469.
    'Cebus capucinus'       : 45,    # cross-species default; Houston, Biochem. Pharmacol. 1994, 47, 1469.
    'Cricetulus griseus'    : 45,    # cross-species default; Houston, Biochem. Pharmacol. 1994, 47, 1469.
    'Helix pomatia'         : None,  # not applicable: gastropod digestive gland, not a vertebrate liver.
    'Chlorocebus aethiops'  : 45,    # cross-species default; Houston, Biochem. Pharmacol. 1994, 47, 1469.
    'Non human primate'     : 45,    # cross-species default; Houston, Biochem. Pharmacol. 1994, 47, 1469.
    'Simiiformes'           : 45,    # cross-species default; Houston, Biochem. Pharmacol. 1994, 47, 1469.
}

# The dataset's Standardized_Species column uses common names; map them to the
# binomial keys MPPGL is indexed by.
SPECIES_COMMON_NAME_TO_BINOMIAL = {
    "Human": "Homo sapiens",
    "Rat": "Rattus norvegicus",
    "Mouse": "Mus musculus",
}

# Raw (un-harmonized) PK-Sim physiology table per species. Human defaults to the
# male table; the female table is available for a sex split. PK-Sim does not
# sex-differentiate the animals.
SPECIES_TO_RAW_PHYSIOLOGY = {
    "Human": human_physiology_from_open_icrp_pksim_male,
    "Rat": rat_physiology_from_pksim,
    "Mouse": mouse_physiology_from_pksim,
}

# Whole-organism glomerular filtration rates in L/min, using approximate literature values.
SPECIES_TO_GLOMERULAR_FILTRATION_RATE_L_PER_MIN = {
    "Human": 0.12,
    "Rat": 0.0013,
    "Mouse": 0.00028,
}


def pbpk_raw_physiology_for_species(species):
    if species not in SPECIES_TO_RAW_PHYSIOLOGY:
        raise ValueError(f"No physiology table for species '{species}'. Known: {list(SPECIES_TO_RAW_PHYSIOLOGY)}.")
    return SPECIES_TO_RAW_PHYSIOLOGY[species]


def pbpk_glomerular_filtration_rate_for_species(species):
    if species not in SPECIES_TO_GLOMERULAR_FILTRATION_RATE_L_PER_MIN:
        raise ValueError(f"No GFR for species '{species}'. Known: {list(SPECIES_TO_GLOMERULAR_FILTRATION_RATE_L_PER_MIN)}.")
    return SPECIES_TO_GLOMERULAR_FILTRATION_RATE_L_PER_MIN[species]


def pbpk_microsomal_protein_per_gram_liver_for_species(species):
    # Resolve the common name to a binomial, then look up MPPGL. None means the
    # microsomal scale-up is not applicable for that species (e.g. an invertebrate).
    binomial_name = SPECIES_COMMON_NAME_TO_BINOMIAL.get(species, species)
    microsomal_protein_per_gram_liver = MPPGL.get(binomial_name)
    if microsomal_protein_per_gram_liver is None:
        raise ValueError(f"No usable MPPGL for species '{species}' (binomial '{binomial_name}').")
    return float(microsomal_protein_per_gram_liver)


def pbpk_liver_volume_litres_for_species(species):
    # Liver volume (L) straight from that species' raw PK-Sim table; with density
    # ~1 g/mL this doubles as the liver mass in grams via a factor of 1000.
    return float(pbpk_raw_physiology_for_species(species)["liver"]["total_volume_L"])


# =========================================================================== #
# 4. ODE STATE-LAYOUT HELPERS                                                 #
# =========================================================================== #
# These two helpers are the single source of truth for "which slot in the flat
# state tensor holds which compartment". Both the ODE right-hand side and the
# simulator wrapper consult them, so the layout can never drift out of sync.
# Because every species' table shares the same organ set in the same order, the
# layout these produce is identical across species, which is what lets the
# multi-species nodes reuse one solve per species without reshaping.

def pbpk_tissue_names(physiology, arterial_blood_name="arterial_blood", venous_blood_name="venous_blood"):
    # The "tissues" are every compartment that exchanges drug with the blood.
    # That excludes the two blood pools themselves, the portal vein (a pure
    # conduit we drop in this version), and a bile compartment (only present
    # if biliary excretion is ever turned on). Everything else (lung, liver,
    # kidney, brain, muscle, ...) is a tissue and keeps its insertion order,
    # which is the order their states appear in the flat state vector.
    excluded = {arterial_blood_name, venous_blood_name, "portal_vein", "bile"}
    return [key for key in physiology.keys() if key not in excluded]


def pbpk_state_layout(physiology, tissue_model_architecture):
    """Map compartment name -> index of its first state in the flat ODE vector.

    The state vector is laid out as:
        index 0                     arterial blood
        index 1                     venous blood
        index 2, 3, ...             the tissues, in order, each occupying
                                    compartments_per_tissue consecutive slots
    For a perfusion-limited tissue that block is a single slot (the tissue
    total concentration). For a permeability-limited tissue it is two slots
    (extracellular then intracellular), and this helper returns the index of
    the first (extracellular) slot.

    Returns the ordered tissue names, the name -> first-index map, and the
    total state dimension.
    """
    tissue_names = pbpk_tissue_names(physiology)
    # One state per tissue for the perfusion model, two (ECS, ICS) otherwise.
    compartments_per_tissue = 1 if tissue_model_architecture == "perfusion_limited" else 2
    # The two blood pools always occupy the first two slots.
    compartment_index = {"arterial_blood": 0, "venous_blood": 1}
    # Each tissue's first slot starts after the two blood pools, spaced by the
    # number of compartments the previous tissues consumed.
    for tissue_position, tissue_name in enumerate(tissue_names):
        compartment_index[tissue_name] = 2 + tissue_position * compartments_per_tissue
    # Total length: two blood pools plus every tissue's compartment block.
    state_dimension = 2 + len(tissue_names) * compartments_per_tissue
    return tissue_names, compartment_index, state_dimension


# =========================================================================== #
# 5. PARTITION-COEFFICIENT METHOD                                             #
# =========================================================================== #

def default_unit_partition_coefficient_method(tissue_names, molecular_input=None, device=None, dtype=torch.float32):
    """Return unit blood/plasma and tissue/plasma partition coefficients."""
    blood_plasma_ratio = torch.ones((), dtype=dtype, device=device)
    tissue_partition_coefficients = torch.ones(len(tissue_names), dtype=dtype, device=device)
    return blood_plasma_ratio, tissue_partition_coefficients


# =========================================================================== #
# 6. PBPK ODE RIGHT-HAND SIDE                                                 #
# =========================================================================== #
class PBPKPlusTorchODE(neural_network.Module):
    """ODE right-hand side f(t, y) for a two-blood-pool PBPK model.

    Circulatory loop: systemic tissues are perfused by ARTERIAL blood and drain
    to VENOUS blood; the lung is perfused by venous blood and drains to arterial,
    carrying the full cardiac output Qco (defined here as the sum of systemic
    tissue flows so the loop closes exactly). State = [arterial, venous, tissue
    states...]. Drug parameters are arguments (so they can be per-molecule and
    carry gradients); physiology is fixed. Architectures: 'perfusion_limited'
    (1 state/tissue) and 'permeability_limited' (ECS/ICS). Linear intrinsic
    (metabolic) clearance acts in the eliminating tissues (liver by default,
    representing hepatic metabolism from microsomal data). Renal elimination is
    modeled separately as glomerular filtration of unbound drug at the rate GFR,
    giving a renal clearance of fu*GFR on the kidney.
    """
    def __init__(self, physiology, fraction_unbound, blood_plasma_ratio, partition_coefficients,
                 intrinsic_clearance=0.0, cellular_apparent_permeability=0.0, pampa_apparent_permeability=0.0,
                 barrier_tissue_names=("brain",), tissue_model_architecture="permeability_limited",
                 eliminating_tissue_names=("liver",), glomerular_filtration_rate=0.0,
                 renal_filtration_tissue_name="kidney", arterial_blood_name="arterial_blood",
                 venous_blood_name="venous_blood", lung_name="lung", device=None, dtype=torch.float32):
        super().__init__()
        # Pick the device from the drug parameters if the caller did not specify
        # one. The drug parameters arrive from upstream already on the right
        # device, so the physiology constants are built to match them.
        if device is None:
            device = fraction_unbound.device if isinstance(fraction_unbound, torch.Tensor) else torch.device("cpu")
        # Two small constructors. as_constant materializes fixed physiology as a
        # tensor on the chosen device/dtype. as_drug_parameter leaves an incoming
        # tensor exactly as-is (so gradients keep flowing back to whatever
        # produced it) and only wraps plain Python scalars.
        def as_constant(value):
            return torch.as_tensor(value, dtype=dtype, device=device)
        def as_drug_parameter(value):
            return value if isinstance(value, torch.Tensor) else as_constant(value)
        # Architecture switch. Perfusion-limited tissues are a single well-mixed
        # pool; permeability-limited tissues split into extracellular (ECS) and
        # intracellular (ICS) compartments separated by a membrane.
        self.is_perfusion_limited = tissue_model_architecture == "perfusion_limited"
        self.compartments_per_tissue = 1 if self.is_perfusion_limited else 2
        # Resolve the tissue list (everything that is not a blood pool / conduit)
        # and lock in its order; this order defines the state layout.
        tissue_names = pbpk_tissue_names(physiology, arterial_blood_name, venous_blood_name)
        self.tissue_names = tuple(tissue_names)
        self.number_of_tissues = len(tissue_names)
        # The lung is the series element that connects venous to arterial. Without
        # it the loop cannot close, so refuse to build a model that lacks one.
        if lung_name not in tissue_names:
            raise ValueError(f"The PBPK circulatory loop requires a '{lung_name}' tissue compartment.")
        # Total size of the tissue block and of the full state vector (two blood
        # pools plus the tissue block).
        self.number_of_tissue_states = self.number_of_tissues * self.compartments_per_tissue
        self.state_dimension = 2 + self.number_of_tissue_states
        # ---- fixed physiology (constants, no gradients) ----
        # Per-tissue total volume, used as the divisor in the perfusion model.
        self.tissue_volume = as_constant([physiology[t]["volume"] for t in tissue_names])
        # Per-tissue blood flow Q. The lung's own tabulated flow is replaced below
        # by the cardiac output, so we keep the raw vector only to derive masks.
        tissue_blood_flow = as_constant([physiology[t]["blood_flow"] for t in tissue_names])
        # Boolean mask marking which tissue is the lung, and the complementary
        # "systemic" flows (lung entry zeroed). Summing the systemic flows gives
        # the cardiac output, and we route that full amount through the lung so
        # the loop conserves mass exactly.
        self.is_lung_mask = torch.tensor([t == lung_name for t in tissue_names], device=device)
        self.non_lung_blood_flow = tissue_blood_flow * (~self.is_lung_mask)
        self.cardiac_output = self.non_lung_blood_flow.sum()
        # Effective throughput per tissue: its own flow for systemic tissues, the
        # full cardiac output for the lung.
        self.effective_blood_flow = torch.where(self.is_lung_mask, self.cardiac_output, tissue_blood_flow)
        # The two blood-pool volumes (divisors for the arterial/venous balances).
        self.arterial_volume = as_constant(physiology[arterial_blood_name]["volume"])
        self.venous_volume = as_constant(physiology[venous_blood_name]["volume"])
        # ---- drug parameters (live tensors so gradients flow) ----
        # fu (plasma unbound fraction), Rb:p (blood:plasma ratio), and the
        # per-tissue tissue:plasma partition coefficients Kp. Each is a scalar or
        # a [batch, 1] / [n_tissues] tensor and broadcasts against the tissue
        # quantities in forward().
        self.fraction_unbound = as_drug_parameter(fraction_unbound)
        self.blood_plasma_ratio = as_drug_parameter(blood_plasma_ratio)
        self.partition_coefficients = as_drug_parameter(partition_coefficients)
        # Linear intrinsic (metabolic) clearance, applied only in the eliminating
        # tissues. We build a per-tissue mask (1 in the liver, 0 elsewhere) and
        # scale the supplied CLint by it, so the result is a per-tissue clearance
        # vector that is zero everywhere drug is not metabolized. The default is
        # liver-only because the predicted CLint represents hepatic metabolism
        # (supervised against microsomal clearance); renal elimination is handled
        # by the separate glomerular-filtration term below, so the kidney is not
        # included here (doing so would double-count renal clearance).
        eliminating = set(eliminating_tissue_names)
        eliminating_mask = as_constant([1.0 if t in eliminating else 0.0 for t in tissue_names])
        self.intrinsic_clearance_base = as_drug_parameter(intrinsic_clearance) * eliminating_mask
        # ---- renal glomerular filtration (a distinct mechanism from metabolism) ----
        # The glomerulus filters plasma at the rate GFR, carrying unbound drug into
        # urine; the resulting renal clearance is fu*GFR (PK-Sim's glomerular-
        # filtration process with filtration fraction = 1). GFR is a species
        # physiological constant in L/min, NOT a per-molecule prediction, so it is
        # materialized as a fixed constant. The kidney mask restricts filtration to
        # the kidney compartment. Defaulting GFR to 0.0 leaves the model's behaviour
        # unchanged until a value is supplied (e.g. ~0.0013 L/min for a 0.25 kg rat).
        self.glomerular_filtration_rate = as_constant(glomerular_filtration_rate)
        self.renal_filtration_kidney_mask = as_constant(
            [1.0 if t == renal_filtration_tissue_name else 0.0 for t in tissue_names])
        # ---- permeability-limited extras (only needed when ECS/ICS are split) ----
        if not self.is_perfusion_limited:
            # The three sub-volumes that make up a permeability-limited tissue.
            self.extracellular_volume = as_constant([physiology[t]["extracellular_volume"] for t in tissue_names])
            self.intracellular_volume = as_constant([physiology[t]["intracellular_volume"] for t in tissue_names])
            self.vascular_volume = as_constant([physiology[t]["vascular_volume"] for t in tissue_names])
            # Membrane area and an empirical in-vitro-to-in-vivo scaling factor
            # (defaults to 1.0 when the table does not supply one).
            surface_area = as_constant([physiology[t]["surface_area"] for t in tissue_names])
            scaling = as_constant([physiology[t].get("permeability_scaling_factor", 1.0) for t in tissue_names])
            # Barrier tissues (the brain by default) use the PAMPA apparent
            # permeability; all other tissues use the cellular apparent
            # permeability. The mask selects which Papp applies per tissue.
            is_barrier = torch.tensor([t in set(barrier_tissue_names) for t in tissue_names], device=device)
            apparent_permeability = torch.where(
                is_barrier, as_drug_parameter(pampa_apparent_permeability),
                as_drug_parameter(cellular_apparent_permeability))
            # Permeability-surface-area product PS = Papp * area * scaling, the
            # coefficient on the passive membrane flux in forward().
            self.permeability_surface_area_product = apparent_permeability * surface_area * scaling

    def forward(self, continuous_time, current_system_state_tensor):
        # torchode calls this as f(t, y). The model is autonomous (no explicit
        # time dependence), so continuous_time is unused; it stays in the
        # signature only because the solver requires the (t, y) convention.
        #
        # current_system_state_tensor is [batch, state_dimension]. We return
        # d(state)/dt with the same shape, filled slot by slot.
        derivative = torch.zeros_like(current_system_state_tensor)
        # The two blood pools live at the first two fixed slots.
        arterial_concentration = current_system_state_tensor[..., 0]   # [batch]
        venous_concentration = current_system_state_tensor[..., 1]     # [batch]
        # View the tissue portion as [batch, n_tissues, comp_per_tissue] so each
        # tissue's compartments are addressable on the last axis (slot 0 is the
        # tissue total or ECS; slot 1, if present, is the ICS).
        tissue_state = current_system_state_tensor[..., 2:2 + self.number_of_tissue_states].reshape(
            *current_system_state_tensor.shape[:-1], self.number_of_tissues, self.compartments_per_tissue)
        tissue_derivative = torch.zeros_like(tissue_state)
        # Drug-parameter aliases. Scalars or [batch, 1] tensors, both of which
        # broadcast against the [batch, n_tissues] tissue quantities, which is
        # what lets one integration handle a batch of different molecules.
        blood_plasma_ratio = self.blood_plasma_ratio      # Rb:p
        partition = self.partition_coefficients           # Kp per tissue
        fraction_unbound = self.fraction_unbound           # fu (plasma)
        # The concentration the blood exchanges with: tissue total (perfusion) or
        # extracellular (permeability).
        first_compartment = tissue_state[..., :, 0]        # [batch, n_tissues]
        # Whole-blood concentration leaving each tissue. At equilibrium the tissue
        # is Kp times plasma, so plasma out is C/Kp and whole blood out is that
        # times Rb:p: C_out = C * Rb:p / Kp.
        tissue_outflow = first_compartment * blood_plasma_ratio / partition
        # Which blood pool perfuses each tissue: arterial for the systemic tissues,
        # venous for the lung. The [n_tissues] mask broadcasts against the
        # [batch, 1] blood concentrations to give [batch, n_tissues].
        incoming_blood = torch.where(self.is_lung_mask, venous_concentration.unsqueeze(-1),
                                     arterial_concentration.unsqueeze(-1))
        # Net perfusion flux per tissue, Q * (C_in - C_out), in amount per time.
        # effective_blood_flow is the tissue's own Q (systemic) or Qco (lung).
        blood_exchange = self.effective_blood_flow * (incoming_blood - tissue_outflow)
        # Unbound concentration that drives membrane transfer and metabolism.
        first_compartment_unbound = first_compartment * fraction_unbound / partition
        # Renal glomerular filtration. The glomerulus filters the ARTERIAL plasma
        # perfusing the kidney, removing unbound drug at the rate GFR. The filtrate
        # concentration is the arterial unbound plasma concentration, i.e. arterial
        # whole blood divided by Rb:p. Referencing filtration to arterial plasma
        # (rather than the well-mixed kidney concentration) makes the systemic renal
        # clearance exactly fu*GFR, independent of renal blood flow, which is the
        # textbook filtration clearance and matches PK-Sim. The kidney mask zeros
        # this everywhere except the kidney; broadcasting the [batch, 1] arterial
        # plasma and fu against the [n_tissues] mask gives [batch, n_tissues],
        # nonzero only in the kidney column.
        arterial_plasma_concentration = (arterial_concentration / blood_plasma_ratio).unsqueeze(-1)
        renal_filtration_amount = (self.renal_filtration_kidney_mask * self.glomerular_filtration_rate
                                   * fraction_unbound * arterial_plasma_concentration)
        if self.is_perfusion_limited:
            # Single well-mixed tissue. Metabolism (nonzero only in eliminating
            # tissues) removes CLint,u * unbound concentration.
            elimination = self.intrinsic_clearance_base * first_compartment_unbound
            # dC/dt = (perfusion in/out - metabolism - renal filtration) / tissue
            # volume. The renal term is nonzero only in the kidney slot.
            tissue_derivative[..., :, 0] = (
                blood_exchange - elimination - renal_filtration_amount) / self.tissue_volume
        else:
            # Two-compartment tissue: ECS (slot 0) exchanges with blood, ICS
            # (slot 1) is where metabolism happens.
            intracellular = tissue_state[..., :, 1]
            intracellular_unbound = intracellular * fraction_unbound / partition
            # Passive flux across the membrane, proportional to the unbound
            # concentration gradient times the permeability-surface-area product.
            passive_diffusion = self.permeability_surface_area_product * (
                first_compartment_unbound - intracellular_unbound)
            # Metabolism acts on the intracellular unbound concentration.
            elimination = self.intrinsic_clearance_base * intracellular_unbound
            # Effective extracellular volume: the true ECS volume plus the
            # vascular sub-volume, the latter converted into the same
            # concentration reference via Rb:p / Kp.
            extracellular_effective_volume = (
                self.extracellular_volume + self.vascular_volume * blood_plasma_ratio / partition)
            # ECS gains perfusion, loses what crosses into the cell, and loses the
            # renal filtrate (glomerular filtration draws from the plasma-accessible
            # space; nonzero only in the kidney).
            tissue_derivative[..., :, 0] = (
                blood_exchange - passive_diffusion - renal_filtration_amount) / extracellular_effective_volume
            # ICS gains what crossed the membrane, loses metabolism.
            tissue_derivative[..., :, 1] = (passive_diffusion - elimination) / self.intracellular_volume
        # ---- close the loop through the two blood pools ----
        # Lung outflow concentration (feeds the arterial pool). Multiplying by the
        # lung mask and summing extracts the single lung entry as [batch].
        lung_outflow = (tissue_outflow * self.is_lung_mask).sum(-1)
        # Total drug returning to venous blood: sum over systemic tissues of
        # Q_tissue * C_tissue_out (non_lung_blood_flow is zero at the lung).
        venous_return = (self.non_lung_blood_flow * tissue_outflow).sum(-1)
        # Total systemic outflow from the arterial pool equals the cardiac output.
        systemic_flow = self.non_lung_blood_flow.sum()
        # Arterial balance: gains the lung outflow at cardiac output, loses what it
        # distributes to the systemic tissues.
        derivative[..., 0] = (self.cardiac_output * lung_outflow
                              - systemic_flow * arterial_concentration) / self.arterial_volume
        # Venous balance: gains the systemic returns, loses what it sends to the lung.
        derivative[..., 1] = (venous_return - self.cardiac_output * venous_concentration) / self.venous_volume
        # Fold the tissue derivatives back into the flat state vector.
        derivative[..., 2:2 + self.number_of_tissue_states] = tissue_derivative.reshape(
            *current_system_state_tensor.shape[:-1], self.number_of_tissue_states)
        return derivative


# =========================================================================== #
# 7. NUMERICAL ODE INTEGRATION (torchode)                                     #
# =========================================================================== #
def integrate_pbpk_ode(ode_right_hand_side, initial_state, time_points,
                       absolute_tolerance=1e-7, relative_tolerance=1e-5):
    """Integrate the PBPK ODE over time_points for a whole batch at once.

    initial_state is [batch, state_dimension]; time_points is a 1-D grid of
    evaluation times shared by the batch. Returns the trajectory as
    [batch, n_timepoints, state_dimension]. The solver is differentiable, so
    gradients flow from a loss on the trajectory back to the drug parameters.
    """
    # torchode wants the evaluation times per batch element, so broadcast the
    # shared 1-D grid up to [batch, n_timepoints].
    evaluation_times = time_points.unsqueeze(0).expand(initial_state.shape[0], -1)
    # Wrap the right-hand side as an ODE term.
    term = torchode.ODETerm(ode_right_hand_side)
    # Dormand-Prince 5(4): an adaptive explicit Runge-Kutta step.
    step_method = torchode.Dopri5(term=term)
    # Adapt the step size to hit the requested absolute/relative tolerances.
    step_size_controller = torchode.IntegralController(
        atol=absolute_tolerance, rtol=relative_tolerance, term=term)
    # AutoDiffAdjoint backpropagates through the solve (needed for training).
    solver = torchode.AutoDiffAdjoint(step_method, step_size_controller)
    # Solve the batched initial value problem and return the sampled trajectory.
    solution = solver.solve(torchode.InitialValueProblem(y0=initial_state, t_eval=evaluation_times))
    return solution.ys   # [batch, n_timepoints, state_dimension]


# =========================================================================== #
# 8. CLOSED-FORM AUC, SINGLE SPECIES (exact linear solve)                     #
# =========================================================================== #
@register("core", "pbpk_closed_form_auc")
class PBPKClosedFormAUC(neural_network.Module):
    """Exact plasma AUC of the linear PBPK model in one differentiable solve: AUC = -A^{-1} y0.

    Conceptual Idea:
    The perfusion-limited PBPK model with first-order clearance is a linear, constant-coefficient system
    dy/dt = A y, y(0) = y0. Integrating over [0, inf) with y(inf) = 0 gives the whole-body AUC vector
    exactly as AUC = -A^{-1} y0, i.e. the single linear solve A x = -y0 (no time stepping, immune to the
    system's stiffness, no quadrature error). A is recovered from the SAME ODE the integrator uses
    (A[:,:,j] = f(e_j), since f is linear with f(0)=0), so the closed form and the integrator can never
    silently disagree. The solve runs in float64 and casts back; torch.linalg.solve is differentiable, so
    gradients flow from the AUC to fu, CLint, dose, and the partition coefficients.

    Species: pass species (e.g. "Rat") and the physiology table and glomerular filtration rate are looked
    up from it; an explicit physiology_table / glomerular_filtration_rate_L_per_min override either.

    Inputs (forward):
      drug_parameters  : [B,5] = [dose, fu, CLint(L/min, already hepatic-scaled), Papp_cell, Papp_pampa].
      partition_bundle : optional (blood_plasma_ratio, Kp[n_tissues]) from an external partition module;
                         if None the configured partition method is used (unit Rb:p and Kp by default).

    Output: plasma AUC [B,1], or log10(AUC) if return_log10=True, read at observed_compartment_name and
            converted blood -> plasma via the blood:plasma ratio.
    """
    def __init__(self, species=None, physiology_table=None, tissue_model_architecture="perfusion_limited",
                 eliminating_tissue_names=("liver",), glomerular_filtration_rate_L_per_min=None,
                 renal_filtration_tissue_name="kidney", partition_coefficient_method=None,
                 dosing_compartment_name="venous_blood", observed_compartment_name="arterial_blood",
                 return_log10=False, linear_solve_dtype=torch.float64):
        super().__init__()
        # Resolve physiology and GFR from the species unless explicitly overridden.
        if physiology_table is None:
            if species is None:
                raise ValueError("PBPKClosedFormAUC requires either species or physiology_table.")
            physiology_table = pbpk_raw_physiology_for_species(species)
        if glomerular_filtration_rate_L_per_min is None:
            glomerular_filtration_rate_L_per_min = (
                pbpk_glomerular_filtration_rate_for_species(species) if species is not None else 0.0)
        self.physiology_table = harmonize_physiology_table(physiology_table)
        self.tissue_model_architecture = tissue_model_architecture
        self.eliminating_tissue_names = tuple(eliminating_tissue_names)
        self.glomerular_filtration_rate_L_per_min = float(glomerular_filtration_rate_L_per_min)
        self.renal_filtration_tissue_name = renal_filtration_tissue_name
        self.partition_coefficient_method = partition_coefficient_method or default_unit_partition_coefficient_method
        self.dosing_compartment_name = dosing_compartment_name
        self.observed_compartment_name = observed_compartment_name
        self.return_log10 = return_log10
        self.linear_solve_dtype = linear_solve_dtype
        self.tissue_names, self.compartment_index, self.state_dimension = pbpk_state_layout(
            self.physiology_table, tissue_model_architecture)

    def forward(self, drug_parameters, partition_bundle=None):
        batch_size = drug_parameters.shape[0]
        original_device = drug_parameters.device
        solve_dtype = self.linear_solve_dtype
        # MPS cannot hold float64 tensors, so run this float64 solve on CPU when the input is on MPS
        # (CPU/CUDA both support float64). Inputs move over and the scalar AUC moves back at the end;
        # autograd flows through the device hops and this module has no learnable params of its own.
        solve_device = torch.device('cpu') if original_device.type == 'mps' else original_device
        drug_parameters = drug_parameters.to(solve_device)
        if partition_bundle is not None:
            partition_bundle = tuple(
                component.to(solve_device) if torch.is_tensor(component) else component
                for component in partition_bundle)
        device = solve_device
        fraction_unbound = drug_parameters[:, 1:2].to(solve_dtype)
        intrinsic_clearance = drug_parameters[:, 2:3].to(solve_dtype)
        dose_amount = drug_parameters[:, 0:1].to(solve_dtype)

        if partition_bundle is not None:
            blood_plasma_ratio, partition_coefficients = partition_bundle
        else:
            blood_plasma_ratio, partition_coefficients = self.partition_coefficient_method(
                self.tissue_names, device=device, dtype=solve_dtype)

        pbpk_ode = PBPKPlusTorchODE(
            self.physiology_table, fraction_unbound=fraction_unbound,
            blood_plasma_ratio=blood_plasma_ratio, partition_coefficients=partition_coefficients,
            intrinsic_clearance=intrinsic_clearance, tissue_model_architecture=self.tissue_model_architecture,
            eliminating_tissue_names=self.eliminating_tissue_names,
            glomerular_filtration_rate=self.glomerular_filtration_rate_L_per_min,
            renal_filtration_tissue_name=self.renal_filtration_tissue_name, dtype=solve_dtype)

        state_dimension = self.state_dimension
        identity = torch.eye(state_dimension, dtype=solve_dtype, device=device)
        system_matrix_columns = [
            pbpk_ode.forward(0.0, identity[column_index].unsqueeze(0).expand(batch_size, state_dimension))
            for column_index in range(state_dimension)]
        system_matrix = torch.stack(system_matrix_columns, dim=-1)

        initial_condition = torch.zeros(batch_size, state_dimension, dtype=solve_dtype, device=device)
        dosing_index = self.compartment_index[self.dosing_compartment_name]
        dosing_volume = self.physiology_table[self.dosing_compartment_name]["volume"]
        initial_condition[:, dosing_index] = dose_amount.squeeze(-1) / dosing_volume

        area_under_curve_vector = torch.linalg.solve(system_matrix, -initial_condition.unsqueeze(-1)).squeeze(-1)
        observed_index = self.compartment_index[self.observed_compartment_name]
        observed_blood_auc = area_under_curve_vector[:, observed_index]
        blood_plasma_ratio_scalar = (blood_plasma_ratio if torch.is_tensor(blood_plasma_ratio)
                                     else torch.as_tensor(blood_plasma_ratio, dtype=solve_dtype, device=device))
        plasma_auc = (observed_blood_auc / blood_plasma_ratio_scalar).to(drug_parameters.dtype).unsqueeze(-1)

        if self.return_log10:
            plasma_auc = torch.log10(plasma_auc.clamp_min(1e-12))
        return plasma_auc.to(original_device)   # back to the model's device (e.g. MPS)


# =========================================================================== #
# 9. NEURAL DRUG-PARAMETER PREDICTION HEAD                                    #
# =========================================================================== #
@register("core", "pbpk_parameter_head")
class PBPKParameterHead(neural_network.Module):
    """Thin head: map a shared-trunk latent to the PBPK drug-parameter vector plus log10 supervision ports.

    Conceptual Idea:
    A thin output head (no internal trunk) on top of the shared FeedForwardNeuralNetwork. Each mechanistic
    parameter gets its own linear projection. The quantities span orders of magnitude and the supervised
    targets are log10-transformed, so the head predicts in log10 and emits BOTH representations:
      * a LINEAR drug-parameter vector that feeds (the MPPGL scale, then) the ODE, and
      * LOG10 values for fu and CLint, routed to the L1/L2 supervision losses against the log10 targets.
    fu is produced as sigmoid(logit) so it stays in (0,1); its log10 is exact from the same logit. CLint is
    predicted directly in log10 (microsomal uL/min/mg, matching the supervised microsomal column) and the
    linear value for the ODE is 10**log10. The microsomal -> whole-liver unit conversion happens downstream
    in the MPPGL scale node, so the supervised CLint here stays in its native microsomal units.

    Output ports (tuple, mapped to named outputs in declaration order):
      0 'drug_parameters'            : [B,5] = [dose, fu(linear), CLint(linear, microsomal), Papp_cell, Papp_pampa]
      1 'fraction_unbound'           : [B,1] fu linear (kept for optional linear-space supervision)
      2 'intrinsic_clearance'        : [B,1] CLint linear, microsomal units
      3 'fraction_unbound_log10'     : [B,1] log10(fu),   for supervision against log10 fu
      4 'intrinsic_clearance_log10'  : [B,1] log10(CLint), for supervision against log10 microsomal CLint
    (ports 3,4 are present only when return_log10_supervision_ports=True; perfusion-limited v1 emits the
    two Papp columns as zeros.)
    """
    def __init__(self, input_dimension, dose_value=1.0, predict_permeability=False,
                 return_log10_supervision_ports=True, log10_clamp=(-9.0, 9.0), logK_clamp=(-6.0, 6.0),
                 predict_per_species=False, species_vocabulary=("Human", "Rat", "Mouse")):
        super().__init__()
        self.dose_value = float(dose_value)
        self.predict_permeability = bool(predict_permeability)
        self.return_log10_supervision_ports = bool(return_log10_supervision_ports)
        # Clamp log10 values before exponentiating so 10**x cannot overflow to inf.
        self.log10_minimum, self.log10_maximum = log10_clamp
        # Clamp logK so the derived fu stays strictly inside (0, 1) and 10**logK cannot overflow.
        self.logK_minimum, self.logK_maximum = logK_clamp

        # --- Species handling, toggled by predict_per_species -------------------------------------- #
        # The head ALWAYS receives the per-row species list, so the config wiring is identical for both
        # modes; flip this one kwarg to switch behaviour:
        #   predict_per_species=False ("species bit"):   concat the species one-hot onto the latent and
        #                                                predict ONE value (species-conditioned).
        #   predict_per_species=True  ("per-species"):   predict one value PER species from structure
        #                                                ([B, n_species]) and gather the row's own species,
        #                                                so only the relevant species' value is used.
        self.predict_per_species = bool(predict_per_species)
        self.species_vocabulary = tuple(species_vocabulary)
        self.species_name_to_index = {label: index for index, label in enumerate(self.species_vocabulary)}
        number_of_species = len(self.species_vocabulary)
        self.register_buffer("species_identity_matrix", torch.eye(number_of_species))

        # Canonical learned variables: logK (for fu) and log10(CLint). Everything else is derived.
        head_output_width = number_of_species if self.predict_per_species else 1
        head_input_width = input_dimension if self.predict_per_species else (input_dimension + number_of_species)
        self.binding_logK_head = neural_network.Linear(head_input_width, head_output_width)
        self.intrinsic_clearance_log10_head = neural_network.Linear(head_input_width, head_output_width)
        if self.predict_permeability:
            self.cellular_permeability_log10_head = neural_network.Linear(head_input_width, head_output_width)
            self.pampa_permeability_log10_head = neural_network.Linear(head_input_width, head_output_width)

    def _species_row_indices(self, species_per_row, device):
        return torch.tensor([self.species_name_to_index[str(label)] for label in species_per_row],
                            dtype=torch.long, device=device)

    def _project_parameter_head(self, parameter_head, molecular_latent, species_one_hot, species_row_indices):
        # Reduce a parameter head to a [B, 1] value for each row's species, in either mode.
        if self.predict_per_species:
            per_species_values = parameter_head(molecular_latent)                        # [B, n_species]
            return per_species_values.gather(1, species_row_indices.view(-1, 1))          # [B, 1]
        return parameter_head(torch.cat([molecular_latent, species_one_hot], dim=-1))     # [B, 1]

    def forward(self, molecular_latent, species_per_row):
        batch_size = molecular_latent.shape[0]
        dtype, device = molecular_latent.dtype, molecular_latent.device
        species_row_indices = self._species_row_indices(species_per_row, device)
        species_one_hot = self.species_identity_matrix.to(dtype)[species_row_indices]      # [B, n_species]

        # Unbound quantity: predict logK = log10((1 - fu)/fu) as the single canonical variable, then
        # derive fu = 1/(1 + K) in (0,1) and log10(fu) = -log10(1 + K) (exact; 1 + K >= 1).
        binding_logK = self._project_parameter_head(self.binding_logK_head, molecular_latent, species_one_hot, species_row_indices)
        binding_logK = binding_logK.clamp(self.logK_minimum, self.logK_maximum)
        binding_ratio = 10.0 ** binding_logK                                               # K = 10**logK  (> 0)
        fraction_unbound_linear = 1.0 / (1.0 + binding_ratio)                              # fu in (0,1), into the ODE
        fraction_unbound_log10 = -torch.log10(1.0 + binding_ratio)                         # log10(fu) <= 0

        # CLint: predict log10(CLint) as the canonical variable; linear value for the ODE is 10**log10.
        intrinsic_clearance_log10 = self._project_parameter_head(self.intrinsic_clearance_log10_head, molecular_latent, species_one_hot, species_row_indices)
        intrinsic_clearance_log10 = intrinsic_clearance_log10.clamp(self.log10_minimum, self.log10_maximum)
        intrinsic_clearance_linear = 10.0 ** intrinsic_clearance_log10                      # > 0, into the ODE

        dose_column = torch.full((batch_size, 1), self.dose_value, dtype=dtype, device=device)
        if self.predict_permeability:
            cellular_permeability_linear = 10.0 ** self._project_parameter_head(self.cellular_permeability_log10_head, molecular_latent, species_one_hot, species_row_indices).clamp(self.log10_minimum, self.log10_maximum)
            pampa_permeability_linear = 10.0 ** self._project_parameter_head(self.pampa_permeability_log10_head, molecular_latent, species_one_hot, species_row_indices).clamp(self.log10_minimum, self.log10_maximum)
        else:
            cellular_permeability_linear = torch.zeros(batch_size, 1, dtype=dtype, device=device)
            pampa_permeability_linear = torch.zeros(batch_size, 1, dtype=dtype, device=device)

        drug_parameters = torch.cat([dose_column, fraction_unbound_linear, intrinsic_clearance_linear,
                                     cellular_permeability_linear, pampa_permeability_linear], dim=-1)  # [B,5]

        if self.return_log10_supervision_ports:
            # Port order MUST match the config `outputs:` list for pbpk_head.
            return (drug_parameters, fraction_unbound_linear, intrinsic_clearance_linear,
                    fraction_unbound_log10, intrinsic_clearance_log10, binding_logK)
        return drug_parameters, fraction_unbound_linear, intrinsic_clearance_linear


# =========================================================================== #
# 10. UNIT-SCALING NODES (single species)                                     #
# =========================================================================== #
@register("core", "scale_by_constant")
class ScaleByConstant(neural_network.Module):
    """Multiply the input by a fixed constant (scalar, or a vector broadcast over the last dim).

    Conceptual Idea:
    Converts the drug-parameter vector from measurement units into the ODE's units WITHOUT rescaling
    the physiology tables (which stay in clean L and L/min). For the [B,5] parameter vector
    [dose, fu, CLint, Papp_cell, Papp_pampa] we pass factor=[1, 1, CLINT_UNIT_SCALE, 1, 1] so only
    intrinsic clearance is converted (e.g. microsomal uL/min/mg -> whole-liver L/min). The factor is a
    non-learnable buffer (a unit constant, not a parameter); gradients pass straight through to the
    input so upstream parameter predictions still train.
    """
    def __init__(self, factor):
        super().__init__()
        self.register_buffer("factor", torch.as_tensor(factor, dtype=torch.float32))

    def forward(self, input_tensor):
        return input_tensor * self.factor.to(input_tensor.dtype)


@register("core", "pbpk_microsomal_clearance_scale")
class PBPKMicrosomalClearanceScale(neural_network.Module):
    """Convert the microsomal CLint column of the drug-parameter vector to whole-liver L/min (one species).

    Conceptual Idea:
    The head predicts CLint in microsomal units (uL/min/mg), matching the supervised microsomal target.
    The ODE needs a whole-liver intrinsic clearance in L/min. The IVIVE scale-up is
        CLint_liver [L/min] = CLint_micro [uL/min/mg] * MPPGL [mg/g] * liver_mass [g] * 1e-6,
    and with liver_mass = liver_volume_L * 1000 (density ~1 g/mL) this is a single factor
    MPPGL * liver_volume_L * 1e-3. MPPGL is species-specific (looked up by species), so this is NOT one
    universal constant. Only the CLint column (index 2) is scaled; dose, fu, and the two Papp columns pass
    through unchanged. The factor is a non-learnable buffer, so gradients pass straight through to the head.
    Use PBPKMultiSpeciesMicrosomalClearanceScale instead when a batch mixes species.
    """
    def __init__(self, species, clint_column_index=2, number_of_drug_parameters=5):
        super().__init__()
        microsomal_protein_per_gram_liver = pbpk_microsomal_protein_per_gram_liver_for_species(species)
        liver_volume_in_litres = pbpk_liver_volume_litres_for_species(species)
        self.hepatic_clearance_scale_factor = microsomal_protein_per_gram_liver * liver_volume_in_litres * 1e-3
        scale_factor_vector = torch.ones(number_of_drug_parameters, dtype=torch.float32)
        scale_factor_vector[clint_column_index] = self.hepatic_clearance_scale_factor
        self.register_buffer("factor", scale_factor_vector)

    def forward(self, drug_parameters):
        return drug_parameters * self.factor.to(drug_parameters.dtype)


# =========================================================================== #
# 11. MULTI-SPECIES DISPATCH (per-row species within a batch)                  #
# =========================================================================== #
# When rows in a batch belong to different species, three places need the per-row
# species label: the prediction head (so fu / CLint can be species-specific, via
# the one-hot below concatenated onto the trunk latent), the microsomal scale (a
# per-row IVIVE factor), and the closed-form AUC (a per-row physiology + GFR).
# The closed-form reuses the validated single-species solve unchanged: rows are
# grouped by species, each group solved once, and the results scattered back in
# the original order. All species share the ODE state layout, so this needs no
# reshaping of the solve.

@register("core", "species_one_hot")
class SpeciesOneHotEncoder(neural_network.Module):
    """Turn the per-row species strings into a [B, n_species] one-hot tensor on the model's device.

    Conceptual Idea:
    fu and microsomal CLint are measured in species-specific systems, so the same molecule has different
    targets across species. A structure-only head cannot fit both. Concatenating this one-hot onto the
    shared latent (before the shared FFN) lets the trunk and head produce species-specific fu / CLint.
    The buffered identity matrix both stores the one-hot rows and carries the device, since the input is a
    list of strings rather than a tensor.
    """
    def __init__(self, species_vocabulary=("Human", "Rat", "Mouse")):
        super().__init__()
        self.species_to_index = {species_label: index for index, species_label in enumerate(species_vocabulary)}
        self.register_buffer("identity_matrix", torch.eye(len(species_vocabulary)))

    def forward(self, species_per_row):
        species_indices = torch.tensor([self.species_to_index[s] for s in species_per_row],
                                       dtype=torch.long, device=self.identity_matrix.device)
        return self.identity_matrix.index_select(0, species_indices)


@register("core", "pbpk_multispecies_microsomal_clearance_scale")
class PBPKMultiSpeciesMicrosomalClearanceScale(neural_network.Module):
    """Per-row microsomal -> whole-liver CLint scaling, with the MPPGL factor chosen per row by species.

    Conceptual Idea:
    The single-species scale baked one MPPGL * liver_volume * 1e-3 factor into a buffer. Here the factor is
    a per-row scalar: a small table holds the factor for every species in the vocabulary, and each row's
    factor is gathered by its species label. Only the CLint column is scaled; everything else passes
    through. No grouping is needed because this is a plain per-row multiply.

    forward(drug_parameters [B,5], species_per_row [list of B strings]) -> [B,5].
    """
    def __init__(self, species_vocabulary=("Human", "Rat", "Mouse"), clint_column_index=2):
        super().__init__()
        self.clint_column_index = clint_column_index
        self.species_to_index = {species_label: index for index, species_label in enumerate(species_vocabulary)}
        per_species_factor = [
            pbpk_microsomal_protein_per_gram_liver_for_species(species_label)
            * pbpk_liver_volume_litres_for_species(species_label) * 1e-3
            for species_label in species_vocabulary]
        self.register_buffer("species_factor_table", torch.tensor(per_species_factor, dtype=torch.float32))

    def forward(self, drug_parameters, species_per_row):
        species_indices = torch.tensor([self.species_to_index[s] for s in species_per_row],
                                       dtype=torch.long, device=drug_parameters.device)
        per_row_factor = self.species_factor_table.to(drug_parameters.dtype).index_select(0, species_indices)  # [B]
        scale_vector = torch.ones(drug_parameters.shape, dtype=drug_parameters.dtype, device=drug_parameters.device)
        scale_vector[:, self.clint_column_index] = per_row_factor
        return drug_parameters * scale_vector


@register("core", "pbpk_multispecies_closed_form_auc")
class PBPKMultiSpeciesClosedFormAUC(neural_network.Module):
    """Plasma AUC for a batch whose rows may be different species.

    Conceptual Idea:
    All species share the ODE state layout, so each species differs only in the numbers inside A. Rather
    than vectorize A over per-row physiology (which would mean rewriting the validated ODE forward for
    batched constants), this groups the batch by species, runs each group through that species' already-
    validated single-species closed form, and scatters the per-group AUCs back into the original row order
    with index_copy (out-of-place, differentiable). With three species this is at most three small solves.
    One PBPKClosedFormAUC is built per species in the vocabulary at construction; only the species actually
    present in a batch are solved.

    forward(drug_parameters [B,5], species_per_row [list of B strings], partition_bundle=None)
      -> plasma AUC [B,1] (or log10 if return_log10=True). The partition_bundle slot is reserved for a
         future shared Kp module; pass it as a third DAG input when that lands.
    """
    def __init__(self, species_vocabulary=("Human", "Rat", "Mouse"), tissue_model_architecture="perfusion_limited",
                 eliminating_tissue_names=("liver",), renal_filtration_tissue_name="kidney",
                 dosing_compartment_name="venous_blood", observed_compartment_name="arterial_blood",
                 return_log10=False, linear_solve_dtype=torch.float64):
        super().__init__()
        self.species_modules = neural_network.ModuleDict({
            species_label: PBPKClosedFormAUC(
                species=species_label, tissue_model_architecture=tissue_model_architecture,
                eliminating_tissue_names=eliminating_tissue_names,
                renal_filtration_tissue_name=renal_filtration_tissue_name,
                dosing_compartment_name=dosing_compartment_name, observed_compartment_name=observed_compartment_name,
                return_log10=return_log10, linear_solve_dtype=linear_solve_dtype)
            for species_label in species_vocabulary})

    def forward(self, drug_parameters, species_per_row, partition_bundle=None):
        batch_size = drug_parameters.shape[0]
        area_under_curve_output = drug_parameters.new_zeros(batch_size, 1)
        # Group row indices by species so each species is solved exactly once.
        grouped_row_indices = defaultdict(list)
        for row_index, species_label in enumerate(species_per_row):
            grouped_row_indices[species_label].append(row_index)
        for species_label, row_index_list in grouped_row_indices.items():
            row_index_tensor = torch.tensor(row_index_list, dtype=torch.long, device=drug_parameters.device)
            species_drug_parameters = drug_parameters.index_select(0, row_index_tensor)
            species_area_under_curve = self.species_modules[species_label](species_drug_parameters)
            # Scatter this species' results back to their original rows (out-of-place; differentiable).
            area_under_curve_output = area_under_curve_output.index_copy(0, row_index_tensor, species_area_under_curve)
        return area_under_curve_output


# =========================================================================== #
# 12. TRAJECTORY-BASED PATH (kept for generality: full time course)           #
# =========================================================================== #
# The closed form above is the production path for whole-body AUC. The pieces
# below integrate the full concentration-time trajectory with torchode and read
# AUC (or any organ's curve) from it. They are kept for endpoints the closed form
# does not give directly (e.g. Cmax / Tmax, individual organ time courses).

@register("core", "trapezoidal_auc")
class TrapezoidalAUC(neural_network.Module):
    """Plasma AUC from a concentration trajectory C[B, T] over a fixed time grid.

    Conceptual Idea:
    AUC = (trapezoid over the sampled grid) + (analytical terminal tail). The tail matters because a
    finite grid truncates the elimination phase. We estimate the terminal rate constant k from the
    last two grid points (C ~ C_last * exp(-k*(t - t_last))) and add AUC_tail = C_last / k, the
    closed-form integral of that mono-exponential to infinity. The tail is added only where the curve
    is genuinely decaying (C_last < C_prev) and C_last > 0; otherwise it is zero. Everything is
    differentiable (torch.trapz, the log/divide, and the masked select via torch.where), so gradients
    flow from AUC back through the ODE to the drug parameters. return_log=True returns log(AUC),
    matched to a log-transformed AUC target.
    """
    def __init__(self, time_points, add_terminal_tail=True, return_log=False, numerical_floor=1e-12):
        super().__init__()
        self.register_buffer("time_points", torch.as_tensor(list(time_points), dtype=torch.float32))
        self.add_terminal_tail = add_terminal_tail
        self.return_log = return_log
        self.numerical_floor = numerical_floor

    def forward(self, concentration_trajectory):  # [B, T]
        time_grid = self.time_points.to(concentration_trajectory.dtype)
        area_on_grid = torch.trapz(concentration_trajectory, time_grid, dim=-1)  # [B]
        if self.add_terminal_tail:
            concentration_last = concentration_trajectory[..., -1]
            concentration_previous = concentration_trajectory[..., -2]
            final_step_width = time_grid[-1] - time_grid[-2]
            decay_ratio = concentration_previous.clamp_min(self.numerical_floor) / concentration_last.clamp_min(self.numerical_floor)
            terminal_rate_constant = (torch.log(decay_ratio.clamp_min(1.0 + 1e-6)) / final_step_width).clamp_min(1e-6)
            curve_is_decaying = (concentration_last < concentration_previous) & (concentration_last > self.numerical_floor)
            terminal_tail_area = torch.where(curve_is_decaying, concentration_last / terminal_rate_constant,
                                             torch.zeros_like(concentration_last))
            area_on_grid = area_on_grid + terminal_tail_area
        if self.return_log:
            return torch.log(area_on_grid.clamp_min(self.numerical_floor)).unsqueeze(-1)
        return area_on_grid.unsqueeze(-1)


@register("core", "pbpk_simulator")
class PBPKSimulator(neural_network.Module):
    """DAG node wrapping PBPKPlusTorchODE for the full time course.

    forward() input: ONE tensor of shape [batch, N_DRUG_PARAMETERS]. Assumed
    column order (PROVISIONAL; will grow as predicted inputs are added):
        0 dose                            amount placed in the dosing compartment at t=0
        1 fraction_unbound                fu in plasma
        2 intrinsic_clearance             CLint,u in eliminating tissues
        3 cellular_apparent_permeability  Papp, non-barrier tissues
        4 pampa_apparent_permeability     Papp, barrier tissues (e.g. brain)
    Blood:plasma and tissue:plasma partition coefficients come from
    partition_coefficient_method, not from these columns.

    forward() output: dict {compartment_name: tensor[batch, n_timepoints]}. Route
    a single compartment to a head/loss/metric with an organ_concentration_selector node.
    """
    def __init__(self, *, raw_physiology, tissue_model_architecture="permeability_limited",
                 time_points, dosing_compartment="venous_blood", partition_coefficient_method=None,
                 barrier_tissue_names=("brain",), eliminating_tissue_names=("liver", "kidney"),
                 dtype=torch.float32):
        super().__init__()
        # Normalize the raw PK-Sim table to the canonical schema the ODE reads.
        # harmonize is idempotent, so passing an already-canonical table is fine.
        self.physiology = harmonize_physiology_table(raw_physiology)
        self.tissue_model_architecture = tissue_model_architecture
        # Which compartment the dose lands in at t=0 (venous blood for an IV bolus).
        self.dosing_compartment = dosing_compartment
        self.barrier_tissue_names = tuple(barrier_tissue_names)
        self.eliminating_tissue_names = tuple(eliminating_tissue_names)

        self.partition_coefficient_method = partition_coefficient_method or default_unit_partition_coefficient_method
        self.solver_dtype = dtype
        # Store the evaluation time grid as a buffer so it moves with .to(device).
        self.register_buffer("time_points", torch.as_tensor(list(time_points), dtype=dtype))
        # Precompute the state layout once (tissue order, name -> index, total size).
        self.tissue_names, self.compartment_index, self.state_dimension = pbpk_state_layout(
            self.physiology, tissue_model_architecture)

    def forward(self, drug_parameter_tensor):
        # Match the solver dtype, then slice the input tensor by the documented
        # column order. Columns 1..4 are kept as [batch, 1] so they broadcast
        # cleanly against the per-tissue quantities inside the ODE.
        drug_parameter_tensor = drug_parameter_tensor.to(self.solver_dtype)
        dose = drug_parameter_tensor[:, 0]


        blood_plasma_ratio, tissue_partition_coefficients = self.partition_coefficient_method(
            self.tissue_names, molecular_input=None, device=dose.device, dtype=self.solver_dtype)

        # Build the right-hand side for this batch. It is rebuilt every forward so
        # the per-molecule drug parameters (and their gradients) take effect; the
        # physiology constants are cheap to re-materialize.
        ode_right_hand_side = PBPKPlusTorchODE(
            self.physiology, fraction_unbound=drug_parameter_tensor[:, 1:2],
            blood_plasma_ratio=blood_plasma_ratio, partition_coefficients=tissue_partition_coefficients,
            intrinsic_clearance=drug_parameter_tensor[:, 2:3],
            cellular_apparent_permeability=drug_parameter_tensor[:, 3:4],
            pampa_apparent_permeability=drug_parameter_tensor[:, 4:5],
            barrier_tissue_names=self.barrier_tissue_names, eliminating_tissue_names=self.eliminating_tissue_names,
            tissue_model_architecture=self.tissue_model_architecture, device=dose.device, dtype=self.solver_dtype)

        # Initial condition: everything starts empty except the dosing compartment,
        # which receives dose as a concentration (amount divided by its volume).
        initial_state = torch.zeros(dose.shape[0], self.state_dimension,
                                    dtype=self.solver_dtype, device=dose.device)
        dosing_index = self.compartment_index[self.dosing_compartment]
        initial_state[:, dosing_index] = dose / self.physiology[self.dosing_compartment]["volume"]

        # Integrate, then unpack the flat trajectory into a name -> [batch, n_time]
        # dict using the precomputed first-state index of each compartment.
        trajectory = integrate_pbpk_ode(ode_right_hand_side, initial_state, self.time_points)
        return {name: trajectory[..., index] for name, index in self.compartment_index.items()}


@register("core", "organ_concentration_selector")
class OrganConcentrationSelector(neural_network.Module):
    """Pull one compartment's trajectory out of the PBPK output dict as a tensor.

    The PBPK simulator returns a dict, which the DAG carries intact on its single
    output port but tensor-consuming nodes (heads, losses, metrics) cannot read.
    This node bridges the gap: give it an organ name and it returns that organ's
    [batch, n_timepoints] trajectory.
    """
    def __init__(self, organ_name):
        super().__init__()
        self.organ_name = organ_name

    def forward(self, pbpk_output_dictionary):
        return pbpk_output_dictionary[self.organ_name]

# ########################################################################### #
# Other Network Modules:
# ########################################################################### #

# =========================================================================== #
# Concatentation Block:
# =========================================================================== #

@register("core", "concat_tensors")
class ConcatTensors(neural_network.Module):
    """
    Concatenate a sequence of tensors along the last dimension.
    """

    def __init__(self, target_dimension_index: int = -1):
        super().__init__()
        self.target_dimension_index = target_dimension_index

    def forward(self, *input_tensors_sequence: torch.Tensor) -> torch.Tensor:
        # Conceptual Idea: 
        # Safely wrap tensor concatenation so that multiple parallel structural streams can be
        # merged logically within a unified configuration pipeline mapping string dictionary.
        return torch.cat(input_tensors_sequence, dim=self.target_dimension_index)

# =========================================================================== #
# Add Tensors Block:
# =========================================================================== #

@register("core", "add_tensors")
class AddTensors(neural_network.Module):
    covers = ("add_tensors",)
    def forward(self, *input_tensors_sequence: torch.Tensor) -> torch.Tensor:
        # Conceptual Idea: 
        # Create an explicit module to mathematically stack parallel geometric tensors and sum them 
        # directly along a new vertical axis. Used when combining independent representations into one shared space.
        return torch.stack(input_tensors_sequence, 0).sum(0)

# ########################################################################### #
# 2. Multi-task Learning Mixers:
# ########################################################################### #

# =========================================================================== #
# Standard Architectures:
# =========================================================================== #

@register("mixer", "shared_bottom_mixer")
class SharedBottomArchitecture(HPS):
    def __init__(self,
                 target_task_names,
                 shared_encoder_class_reference,
                 task_specific_decoders_dictionary,
                 keep_representation_gradients: bool = False,
                 allow_multiple_inputs: bool = False,
                 hardware_device_target="cpu",
                 **keyword_arguments):
        # Conceptual Idea:
        # Implement the absolute simplest form of Multi-Task Learning. All incoming data pushes
        # completely through an identically shared foundational encoder backbone, splitting structurally
        # into separate independently trained linear decoders solely at the very final prediction step.
        super().__init__(target_task_names, shared_encoder_class_reference, task_specific_decoders_dictionary,
                         keep_representation_gradients, allow_multiple_inputs, hardware_device_target, **keyword_arguments)


@register("mixer", "cross_stitch_mixer")
class CrossStitchArchitecture(Cross_stitch):
    def __init__(self,
                 target_task_names,
                 shared_encoder_class_reference,
                 task_specific_decoders_dictionary,
                 keep_representation_gradients: bool = False,
                 allow_multiple_inputs: bool = False,
                 hardware_device_target="cpu",
                 *,
                 initial_alpha_mixing_coefficients: Union[float, List[float]] = 0.1,
                 **keyword_arguments):
        # Conceptual Idea:
        # Maintain independent parallel feature extractors for each distinct task,
        # but allow them to mathematically share features through learnable soft linear
        # combinations ("cross-stitches") at specific layers.
        keyword_arguments.setdefault("alpha_init", initial_alpha_mixing_coefficients)
        super().__init__(target_task_names, shared_encoder_class_reference, task_specific_decoders_dictionary,
                         keep_representation_gradients, allow_multiple_inputs, hardware_device_target, **keyword_arguments)


@register("mixer", "mmoe_mixer")
class MMoEArchitecture(MMoE):
    def __init__(self,
                 target_task_names,
                 shared_encoder_class_reference,
                 task_specific_decoders_dictionary,
                 keep_representation_gradients: bool = False,
                 allow_multiple_inputs: bool = False,
                 hardware_device_target="cpu",
                 **keyword_arguments):
        # Conceptual Idea:
        # Multi-gate Mixture-of-Experts allows multiple neural network blocks to specialize as
        # independent "experts". Different tasks learn their own unique gating networks to selectively
        # weight and assemble the experts' outputs in a way optimally suited for that particular task.
        if "num_experts" in keyword_arguments and isinstance(keyword_arguments["num_experts"], int):
            keyword_arguments["num_experts"] = [keyword_arguments["num_experts"]]

        expected_image_size_tuple = keyword_arguments.get("img_size", None)
        if not (isinstance(expected_image_size_tuple, (list, tuple)) and len(expected_image_size_tuple) == 1 and isinstance(expected_image_size_tuple[0], int) and expected_image_size_tuple[0] > 0):
            raise ValueError("MMoE expects 2D inputs. Provide img_size=[feature_dim].")

        super().__init__(target_task_names, shared_encoder_class_reference, task_specific_decoders_dictionary, keep_representation_gradients, allow_multiple_inputs, hardware_device_target, **keyword_arguments)

# =========================================================================== #
# LibMTL Mixer (Pure-Mixer Wrapper)
# =========================================================================== #

# --------------------------------------------------------------------------- #
# 2D -> 4D and 4D -> 2D mapping:
# --------------------------------------------------------------------------- #

class MapFourDimensionsToTwoDimensions(neural_network.Module):
    """
    (B, C, H, W) -> AdaptiveAvgPool2d(1) -> (B, C)   (and passthrough for (B, D))

    Use when you want a safe 4D->2D shim; if input is already 2D, it is returned unchanged.
    """
    def __init__(self):
        super().__init__()
        self.adaptive_pooling_layer = neural_network.AdaptiveAvgPool2d(1)
        self.flattening_layer = neural_network.Flatten(start_dim=1)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # Conceptual Idea:
        # Safely reduce any convolutional feature map with trailing spatial dimensions
        # down to a standard flat (Batch, Features) tensor required by linear classification heads.
        if input_tensor.dim() == 4:        # (B, C, H, W) -> (B, C, 1, 1) -> (B, C)
            return self.flattening_layer(self.adaptive_pooling_layer(input_tensor))
        if input_tensor.dim() == 2:        # already (B, D)
            return input_tensor
        raise TypeError(f"Expected 4D or 2D tensor, got shape {tuple(input_tensor.shape)}")


class MapTwoDimensionsToFourDimensions(neural_network.Module):
    """
    Add two trailing singleton spatial dimensions.

    Behavior:
      - (B, D)           -> (B, D, 1, 1)
      - (B, C, H, W)     -> unchanged (passthrough)

    Any other rank raises a TypeError.
    """
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # Conceptual Idea:
        # Inverse to Map4Dto2D. Artificially expand a flat feature vector with synthetic height
        # and width boundaries so that older 2D convolutional networks can process it without crashing.
        if input_tensor.dim() == 2:          # (B, D) -> (B, D, 1, 1)
            return input_tensor.unsqueeze(-1).unsqueeze(-1)
        if input_tensor.dim() == 4:          # already NCHW
            return input_tensor
        raise TypeError(f"Expected 2D or 4D tensor, got shape {tuple(input_tensor.shape)}")


# --------------------------------------------------------------------------- #
# Identity Encoders and Decoders:
# --------------------------------------------------------------------------- #

class IdentityEncoderWrapper(neural_network.Module):
    """No-op encoder so LibMTL mixers see the API they expect."""
    def __init__(self, *positional_arguments, **keyword_arguments):
        super().__init__()
    def forward(self, input_tensor):
        return input_tensor


class IdentityDecoderWrapper(neural_network.Module):
    """No-op decoder per task so LibMTL mixers see the API they expect."""
    def __init__(self, *positional_arguments, **keyword_arguments):
        super().__init__()
    def forward(self, input_tensor):
        return input_tensor


# --------------------------------------------------------------------------- #
# General Encoder and Decoder Classes:
# --------------------------------------------------------------------------- #

class GeneralEncoderWrapper(neural_network.Module):
    """Thin wrapper around an arbitrary neural_network.Module acting as the encoder."""
    def __init__(self, target_module: neural_network.Module):
        super().__init__()
        self.wrapped_module = target_module

    def forward(self, input_tensor):
        return self.wrapped_module(input_tensor)


class GeneralDecoderWrapper(neural_network.Module):
    """Thin wrapper around an arbitrary neural_network.Module acting as the decoder."""
    def __init__(self, target_module: neural_network.Module):
        super().__init__()
        self.wrapped_module = target_module

    def forward(self, input_tensor):
        return self.wrapped_module(input_tensor)


# --------------------------------------------------------------------------- #
# ResNet-Specific Encoder and Decoder Classes:
# --------------------------------------------------------------------------- #

def wrap_two_dimensional_module_around_four_dimensional_input_output(target_module):

    return neural_network.Sequential(
        MapFourDimensionsToTwoDimensions(),
        target_module,
        MapTwoDimensionsToFourDimensions()
        )


class ResNetLikeEncoder(neural_network.Module):
    """
    ResNet-shaped encoder for LibMTL Cross-stitch:
      - conv1 / bn1 / relu / maxpool (identity) as the stem
      - layer1..layer4 as Module/Sequential stages
    Uses LazyConv2d so it can be constructed without knowing input channels.

    IMPORTANT: LibMTL's Cross_stitch does not call this encoder's forward. Its
    internal _transform_resnet_cross decomposes the encoder by attribute name,
    reading encoder.conv1, encoder.bn1, encoder.relu, encoder.maxpool and the four
    encoder.layerN stages directly. The stem attributes below are therefore named
    conv1 / bn1 / relu / maxpool to match exactly; renaming them breaks Cross-stitch
    with an AttributeError.

    Expected input to the stem is NCHW (B, C, H, W). If you start from vectors (B, D),
    the stem's MapTwoDimensionsToFourDimensions reshapes to (B, D, 1, 1) before the conv.
    """
    def __init__(
        self,
        stem_output_dimension,
        use_batch_normalization = True,
        residual_layer_1 = None,
        residual_layer_2 = None,
        residual_layer_3 = None,
        residual_layer_4 = None
    ):
        super().__init__()

        # ---- Stem ----
        # LazyConv2d infers in_channels on first call (so Cross-stitch can build the stem and run it).
        # Conceptual Idea:
        # Construct an artificial scaffold mirroring ResNet's internal four-stage layout. This
        # lets our abstract 1D MLPs plug directly into legacy multi-task architectures designed
        # strictly for images. The stem names match what LibMTL's Cross_stitch reads by attribute.
        self.conv1 = neural_network.Sequential(
            MapTwoDimensionsToFourDimensions(),
            neural_network.LazyConv2d(out_channels=stem_output_dimension, kernel_size=1, bias=False),
        )
        self.bn1 = neural_network.BatchNorm2d(stem_output_dimension)
        self.relu = neural_network.ReLU(inplace=True) if use_batch_normalization else neural_network.Identity()
        self.maxpool = neural_network.Identity()  # H=W=1 in our pseudo-image setup

        sequential_network_layers = [residual_layer_1, residual_layer_2, residual_layer_3, residual_layer_4]

        # Default is the identity function if a layer is not defined.
        sequential_network_layers = [construct_neural_network_module(layer_definition) if layer_definition is not None else neural_network.Identity() for layer_definition in sequential_network_layers]

        # Wrap 4D -> 2D and then 2D -> 4D around module
        sequential_network_layers = [wrap_two_dimensional_module_around_four_dimensional_input_output(layer_instance) for layer_instance in sequential_network_layers]

        self.layer1 = sequential_network_layers[0]
        self.layer2 = sequential_network_layers[1]
        self.layer3 = sequential_network_layers[2]
        self.layer4 = sequential_network_layers[3]

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class ResNetLikeIdentityDecoder(neural_network.Module):
    """
    Minimal decoder that converts a stage-4 feature map to a flat vector:
        AdaptiveAvgPool2d(1) -> Flatten -> (optional) Linear

    If you want a specific output size but don't know the input channels a priori,
    set `proj_out_features` and we use LazyLinear so construction stays arg-free compatible.
    """
    def __init__(self, projected_output_features_dimension: Union[int, None] = None):
        super().__init__()
        self.adaptive_pooling_layer    = neural_network.AdaptiveAvgPool2d(1)
        self.flattening_layer = neural_network.Flatten(start_dim=1)
        self.projection_layer    = neural_network.Identity() if projected_output_features_dimension is None else neural_network.LazyLinear(projected_output_features_dimension)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        input_tensor = self.adaptive_pooling_layer(input_tensor)         # (B, C, 1, 1)
        input_tensor = self.flattening_layer(input_tensor)       # (B, C)
        input_tensor = self.projection_layer(input_tensor)         # (B, C) or (B, proj_out_features)
        return input_tensor


# --------------------------------------------------------------------------- #
# LibMTL MixerBase:
# --------------------------------------------------------------------------- #

class LibMTLMixerBase(neural_network.Module):

    target_mixer_type_identifier = None # Must be overridden in inherited classes.

    def __init__(self,
                 *,
                 encoder_modules: Sequence[neural_network.Module],
                 decoder_modules: Sequence[neural_network.Module],
                 target_output_task_names: Sequence[str],
                 multiple_inputs_enabled: bool = False,
                 representation_gradients_enabled: bool = False,
                 hardware_device_target: Union[str, torch.device] = "cpu",
                 **mixer_keyword_arguments):
        super().__init__()

        # Conceptual Idea:
        # Provide a seamlessly structured unifying adapter connecting our external
        # framework logically with the internal backend implementations of the LibMTL library tools.
        self.mixer_class_reference = retrieve_mixer_component(self.target_mixer_type_identifier)
        self.multiple_inputs_enabled = bool(multiple_inputs_enabled)
        self.representation_gradients_enabled = bool(representation_gradients_enabled)
        self.hardware_device_target = torch.device(hardware_device_target) if not isinstance(hardware_device_target, torch.device) else hardware_device_target
        self.mixer_keyword_arguments = mixer_keyword_arguments

        # Will finalize on construction or infer on first forward (multi_input & dict).
        self.target_output_task_names: List[str] = list(target_output_task_names)

        # Construct backend; keep purely as a mixing module.
        # NOTE: the device is passed as hardware_device_target (the name the architecture
        # wrappers above expect). They forward it positionally into LibMTL's `device` slot,
        # so it must not also be passed as device= here or LibMTL gets it twice.
        self.internal_multi_task_model_backend = self.mixer_class_reference(
            target_task_names=self.target_output_task_names,
            shared_encoder_class_reference=encoder_modules,
            task_specific_decoders_dictionary=decoder_modules,
            keep_representation_gradients=self.representation_gradients_enabled,
            allow_multiple_inputs=self.multiple_inputs_enabled,
            hardware_device_target=self.hardware_device_target,
            **self.mixer_keyword_arguments
        ).to(self.hardware_device_target)

    def forward(self, input_features_tensor) -> List[torch.Tensor]:
        # LibMTL architectures return a {task_name: prediction} dict. We re-emit it as an
        # ordered list aligned with target_output_task_names for the rest of the pipeline.
        predictions_dictionary = self.internal_multi_task_model_backend(input_features_tensor)
        sequenced_outputs = [predictions_dictionary[target_output_name] for target_output_name in self.target_output_task_names]
        return sequenced_outputs


# --------------------------------------------------------------------------- #
# Specific Mixer Architectures:
# --------------------------------------------------------------------------- #

def construct_encoders_and_decoders(target_output_task_names, encoder_configurations, decoder_configurations, multiple_inputs_enabled):

    # 1) Build decoders into an neural_network.ModuleDict
    # Conceptual Idea:
    # Iterate through the expected task outputs. If a specific decoder isn't mapped,
    # substitute an identity layer safely so the forward pass network logic doesn't crash on null properties.
    if decoder_configurations is None or (isinstance(decoder_configurations, type) and issubclass(decoder_configurations, neural_network.Identity)) or isinstance(decoder_configurations, neural_network.Identity):
        decoder_modules_dictionary = neural_network.ModuleDict({target_task_name: neural_network.Identity() for target_task_name in target_output_task_names})
    else:
        normalized_decoder_configurations = decoder_configurations if isinstance(decoder_configurations, (list, tuple)) else [decoder_configurations]
        if len(normalized_decoder_configurations) == 1 and len(target_output_task_names) > 1:
            normalized_decoder_configurations = normalized_decoder_configurations * len(target_output_task_names)
        decoder_modules_dictionary = neural_network.ModuleDict({target_task_name: construct_neural_network_module(decoder_subconfiguration, should_instantiate_flag=True) for target_task_name, decoder_subconfiguration in zip(target_output_task_names, normalized_decoder_configurations)})

    # 2) Build encoders as a factory function
    # LibMTL calls encoder_class() with no args, so each encoder is left uninstantiated
    # (a functools.partial) and instantiated by LibMTL itself.
    normalized_encoder_configurations = encoder_configurations if isinstance(encoder_configurations, (list, tuple)) else [encoder_configurations]
    encoder_modules = [construct_neural_network_module(encoder_subconfiguration, should_instantiate_flag=False) for encoder_subconfiguration in normalized_encoder_configurations]

    if not multiple_inputs_enabled:
        encoder_modules = encoder_modules[0]

    return encoder_modules, decoder_modules_dictionary


@register("core", "Shared_Parameter_Mixer")
class SharedParameterMixer(LibMTLMixerBase):
    target_mixer_type_identifier = "shared_bottom_mixer"

    def __init__(self,
                 outputs: Sequence[str],
                 encoders, # Now required
                 decoders = None,
                 multiple_inputs: bool = False,
                 representation_gradients: bool = False,
                 device: Union[str, torch.device] = "cpu",
                 **mixer_keyword_arguments):

        encoder_modules, decoder_modules_dictionary = construct_encoders_and_decoders(outputs, encoders, decoders, multiple_inputs)

        super().__init__(
                 encoder_modules = encoder_modules,
                 decoder_modules = decoder_modules_dictionary,
                 target_output_task_names = outputs,
                 multiple_inputs_enabled = multiple_inputs,
                 representation_gradients_enabled = representation_gradients,
                 hardware_device_target = device,
                 **mixer_keyword_arguments
                 )


@register("core", "Cross_Stitch_Mixer")
class CrossStitchMixer(LibMTLMixerBase):
    target_mixer_type_identifier = "cross_stitch_mixer"

    def __init__(self,
                 outputs: Sequence[str],
                 stem_output_dimension = 128,
                 use_batch_normalization = True,
                 residual_layer_1 = None,
                 residual_layer_2 = None,
                 residual_layer_3 = None,
                 residual_layer_4 = None,
                 decoders = None,
                 multiple_inputs: bool = False,
                 representation_gradients: bool = False,
                 device: Union[str, torch.device] = "cpu",
                 **mixer_keyword_arguments):

        encoder_modules_factory = partial(ResNetLikeEncoder,
            stem_output_dimension, use_batch_normalization=use_batch_normalization,
            residual_layer_1=residual_layer_1, residual_layer_2=residual_layer_2, residual_layer_3=residual_layer_3, residual_layer_4=residual_layer_4
        )

        if decoders is None:
            decoder_modules_dictionary = neural_network.ModuleDict({target_task_name: ResNetLikeIdentityDecoder() for target_task_name in outputs})
        else:
            raise NotImplementedError()

        super().__init__(
                 encoder_modules = encoder_modules_factory, decoder_modules = decoder_modules_dictionary,
                 target_output_task_names = outputs, multiple_inputs_enabled = multiple_inputs,
                 representation_gradients_enabled = representation_gradients, hardware_device_target = device, **mixer_keyword_arguments
                 )


@register("core", "MMoE_Mixer")
class MMoEMixer(LibMTLMixerBase):
    target_mixer_type_identifier = "mmoe_mixer"

    def __init__(self,
                 outputs: Sequence[str],
                 encoders,
                 decoders = None,
                 multiple_inputs: bool = False,
                 representation_gradients: bool = False,
                 device: Union[str, torch.device] = "cpu",
                 **mixer_keyword_arguments):

        encoder_modules, decoder_modules_dictionary = construct_encoders_and_decoders(outputs, encoders, decoders, multiple_inputs)

        super().__init__(encoder_modules=encoder_modules, decoder_modules=decoder_modules_dictionary,
                         target_output_task_names=outputs, multiple_inputs_enabled=multiple_inputs,
                         representation_gradients_enabled=representation_gradients, hardware_device_target=device, **mixer_keyword_arguments)

# ########################################################################### #
# 4.  Weighting  – task‑level gradient / loss balancers
# ########################################################################### #

"""
Some comments on the weighting classes:
    - Assumes that losses per task are returned as a list that, by convention, 
    has a different element for each task (so that combination of task-specific
    tensors can be done outside of a model's training and kept in this class).

"""


class AbstractWeightingStrategy(abc.ABC):
    """
    Base API: subclasses *must* implement ``backward(task_losses)``.
    """

    def __init__(self, shared_parameters: Sequence[neural_network.Parameter], **configuration_keyword_arguments):
        self.shared_parameters = list(shared_parameters)  # keep for LibMTL wrappers

    @abc.abstractmethod
    def backward(self, task_loss_tensors: List[torch.Tensor]) -> torch.Tensor: ...


@register("weighting", "identity")
class IdentityWeighting(AbstractWeightingStrategy):
    """Simply sums losses and back‑propagates."""

    def backward(self, task_loss_tensors: List[torch.Tensor]) -> torch.Tensor:
        # Conceptual Idea: 
        # Combine independently aggregated task losses symmetrically by purely mapping scalar addition across them linearly.
        total_loss_tensor = torch.stack(task_loss_tensors, 0).sum()
        total_loss_tensor.backward()
        return total_loss_tensor


@register("weighting", "pcgrad")
class PCGradWeighting(AbstractWeightingStrategy):
    """
    Project Conflicting Gradients (PCGrad), Yu et al., NeurIPS 2020.

    Self-contained implementation. LibMTL's PCGrad is an AbsWeighting mixin that only works
    once fused into LibMTL's own architecture and Trainer (its backward() reads host-provided
    self.task_num, self.device, self.get_share_params(), self.zero_grad_share_params(), ...),
    so it cannot be driven standalone the way this pipeline needs. We therefore perform the
    gradient-surgery step directly over the parameters passed in.

    backward() takes a list of per-task scalar losses, computes each task's gradient w.r.t.
    the parameters, removes pairwise conflicting components (negative inner products), writes
    the resolved gradient into each parameter's .grad, and returns the summed loss (detached)
    for logging. The caller is responsible for calling optimizer.step().
    """

    def __init__(self, shared_parameters, reduction_method_name: str = "mean", **additional_keyword_arguments):
        super().__init__(shared_parameters)
        # Accepted for config compatibility; PCGrad sums the de-conflicted per-task gradients,
        # so no additional reduction is applied here.
        self.reduction_method_name = reduction_method_name

    def _flatten_task_gradient(self, task_loss_tensor):
        # Gradient of one task loss w.r.t. every parameter, flattened to one vector.
        # allow_unused=True because a task does not touch the other tasks' heads, so those
        # parameters return None and are treated as zeros.
        per_parameter_gradients = torch.autograd.grad(
            task_loss_tensor, self.shared_parameters, retain_graph=True, allow_unused=True)
        flattened_gradient_segments = [
            (gradient_segment if gradient_segment is not None else torch.zeros_like(parameter_tensor)).reshape(-1)
            for gradient_segment, parameter_tensor in zip(per_parameter_gradients, self.shared_parameters)]
        return torch.cat(flattened_gradient_segments)

    def backward(self, task_loss_tensors: List[torch.Tensor]) -> torch.Tensor:
        # 1) One flat gradient vector per task.
        original_task_gradients = [self._flatten_task_gradient(task_loss) for task_loss in task_loss_tensors]

        # 2) Surgery: for each task i, project out the component of its gradient that conflicts
        #    (negative dot product) with each other task j, visiting the others in a fresh random
        #    order each time, as in the paper.
        surgically_adjusted_gradients = [gradient_vector.clone() for gradient_vector in original_task_gradients]
        number_of_tasks = len(original_task_gradients)
        for task_index_i in range(number_of_tasks):
            visiting_order = list(range(number_of_tasks))
            random.shuffle(visiting_order)
            for task_index_j in visiting_order:
                if task_index_j == task_index_i:
                    continue
                inner_product = torch.dot(surgically_adjusted_gradients[task_index_i], original_task_gradients[task_index_j])
                if inner_product < 0:
                    squared_norm = original_task_gradients[task_index_j].pow(2).sum().clamp_min(1e-12)
                    surgically_adjusted_gradients[task_index_i] = surgically_adjusted_gradients[task_index_i] - (inner_product / squared_norm) * original_task_gradients[task_index_j]

        # 3) Sum the de-conflicted task gradients into one vector.
        combined_gradient_vector = torch.stack(surgically_adjusted_gradients, dim=0).sum(dim=0)

        # 4) Scatter the combined vector back into each parameter's .grad, reshaping each
        #    contiguous slice to that parameter's shape. The caller then steps the optimizer.
        current_flat_offset = 0
        for parameter_tensor in self.shared_parameters:
            number_of_elements = parameter_tensor.numel()
            gradient_slice = combined_gradient_vector[current_flat_offset:current_flat_offset + number_of_elements]
            parameter_tensor.grad = gradient_slice.view_as(parameter_tensor).detach().clone()
            current_flat_offset += number_of_elements

        # 5) Report the (unweighted) total loss for logging; gradients are already set.
        return torch.stack([task_loss.detach() for task_loss in task_loss_tensors]).sum()

# --------------------------------------------------------------------------- #
# Process Weighting Config:
# --------------------------------------------------------------------------- #

def construct_weighting_object_from_configuration(configuration_dictionary: Dict, shared_parameters) -> AbstractWeightingStrategy:
    """
    Helper that converts a YAML / dict section into an instantiated weighting
    object, e.g.

        weighting:
            name: pcgrad
            kwargs: { reduction: mean }
    """
    WeightingStrategyClass = retrieve_weighting_component(configuration_dictionary.get("name", "identity"))
    return WeightingStrategyClass(shared_parameters, **configuration_dictionary.get("kwargs", {}))

###############################################################################
# 5.  Loss functions:
###############################################################################

# =========================================================================== #
# Classification Error Loss Functions:
# =========================================================================== #

@register("loss", "bce_with_logits")
def calculate_binary_cross_entropy_with_logits(predictions, targets, sample_importance_weights, **ignored_keyword_arguments):
    # Conceptual Idea: 
    # Evaluate binary classification alignment using unnormalized logits (for numerical stability).
    # Then element-wise weight the individual prediction errors explicitly to account for class imbalance or noisy labels.
    targets = targets.float().squeeze(-1)         # BCE expects float targets
    computed_loss_tensor = neural_network_functional.binary_cross_entropy_with_logits(predictions.squeeze(-1), targets, reduction="none")
    computed_loss_tensor = computed_loss_tensor.view(computed_loss_tensor.shape[0], -1).mean(-1)
    sample_importance_weights_tensor = torch.as_tensor(sample_importance_weights, dtype=computed_loss_tensor.dtype, device=computed_loss_tensor.device)
    return (computed_loss_tensor * sample_importance_weights_tensor).sum() / sample_importance_weights_tensor.sum().clamp_min(1e-6)

@register("loss", "cross_entropy")
def calculate_cross_entropy(predictions: torch.Tensor, targets: torch.Tensor, sample_importance_weights, **ignored_keyword_arguments):
    computed_loss_tensor = neural_network_functional.cross_entropy(predictions, targets, reduction="none").view(predictions.shape[0], -1).mean(-1)
    sample_importance_weights_tensor = torch.as_tensor(sample_importance_weights, dtype=computed_loss_tensor.dtype, device=computed_loss_tensor.device)
    return (computed_loss_tensor * sample_importance_weights_tensor).sum() / sample_importance_weights_tensor.sum().clamp_min(1e-6)

@register("loss", "focal")
def calculate_focal_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    sample_importance_weights,
    alpha_scaling_factor=0.5,                 # keep your default
    gamma_focus_factor=0.0,                # keep your default
    reduction_method_name: str = "mean",
    **ignored_keyword_arguments,
):
    """
    Focal loss using logits in `predictions` and labels in `targets`.
    - Binary: predictions shape [N] or [N,1], targets float/bool/int -> {0,1}
    - Multiclass: predictions shape [..., C], targets either class indices [...], or one-hot [..., C]
    The signature matches the other losses (`predictions, targets, **_`) so it plays nice with LossTerm.
    """
    # Conceptual Idea: 
    # Focal loss naturally down-weights easily classified ("confident") samples and forces the network 
    # to concentrate its gradient focus disproportionately onto hard, misclassified edge cases.
    alpha_tensor = torch.as_tensor(alpha_scaling_factor, dtype=predictions.dtype, device=predictions.device)
    
    if predictions.ndim == 1 or predictions.shape[-1] == 1:
        raw_prediction_logits, target_class_labels = predictions.squeeze(-1), targets.to(predictions).squeeze(-1)
        baseline_cross_entropy_loss = neural_network_functional.binary_cross_entropy_with_logits(raw_prediction_logits, target_class_labels, reduction="none")
        class_weighting_factor = target_class_labels * alpha_tensor + (1 - target_class_labels) * (1 - alpha_tensor)
    else:
        target_class_indices_tensor = (targets.argmax(-1) if targets.shape == predictions.shape else targets).to(predictions.device).long()
        baseline_cross_entropy_loss = -neural_network_functional.log_softmax(predictions, -1).gather(-1, target_class_indices_tensor.unsqueeze(-1)).squeeze(-1)
        class_weighting_factor = alpha_tensor[target_class_indices_tensor] if alpha_tensor.numel() == predictions.shape[-1] else alpha_tensor
        
    computed_focal_loss_tensor = class_weighting_factor * (1 - torch.exp(-baseline_cross_entropy_loss)).pow(gamma_focus_factor) * baseline_cross_entropy_loss
    sample_importance_weights_tensor = torch.as_tensor(sample_importance_weights, dtype=computed_focal_loss_tensor.dtype, device=computed_focal_loss_tensor.device)
    
    if reduction_method_name == "none":
        return computed_focal_loss_tensor * sample_importance_weights_tensor.view(-1, *([1] * (computed_focal_loss_tensor.ndim - 1)))
    if reduction_method_name not in {"mean", "sum"}:
        raise ValueError(f"Invalid reduction: {reduction_method_name}")
        
    computed_focal_loss_tensor = computed_focal_loss_tensor.view(computed_focal_loss_tensor.shape[0], -1).mean(-1) * sample_importance_weights_tensor
    return computed_focal_loss_tensor.sum() if reduction_method_name == "sum" else computed_focal_loss_tensor.sum() / sample_importance_weights_tensor.sum().clamp_min(1e-6)

# =========================================================================== #
# Regression Error Loss Functions:
# =========================================================================== #

@register("loss", "mse")
def calculate_mean_squared_error(predictions: torch.Tensor, targets: torch.Tensor, sample_importance_weights, **ignored_keyword_arguments):
    computed_loss_tensor = neural_network_functional.mse_loss(
        predictions.squeeze(-1), 
        targets, 
        reduction="none"
    ).view(predictions.shape[0], -1).mean(-1)
    sample_importance_weights_tensor = torch.as_tensor(sample_importance_weights, dtype=computed_loss_tensor.dtype, device=computed_loss_tensor.device)
    return (computed_loss_tensor * sample_importance_weights_tensor).sum() / sample_importance_weights_tensor.sum().clamp_min(1e-6)

# =========================================================================== #
# Triplet Hinge Loss:
# =========================================================================== #

@register("loss", "property_distance_hinge")
def calculate_property_distance_hinge_loss(
    predictions: torch.Tensor,     # Graph Embeddings: shape (Batch_Size, Feature_Dimension)
    targets: torch.Tensor,         # Property Labels: shape (Batch_Size, 1) or (Batch_Size,)
    sample_importance_weights: torch.Tensor, 
    margin_threshold_value: float = 0.0, 
    enforce_two_sided_penalty: bool = False,
    **ignored_keyword_arguments
):
    current_batch_size = predictions.size(0)
    
    # Safe-guard against heavily masked batches where pairs cannot be formed
    if current_batch_size < 2:
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)

    if targets.dim() == 1:
        targets = targets.unsqueeze(1)
        
    # Conceptual Idea: 
    # Enforce a structural property manifold. We want the Euclidean distances between learned embeddings 
    # to proportionally mirror the absolute distance between their actual physical properties in reality.
    
    # 1. Pairwise L2 distance between embeddings: shape (Batch_Size, Batch_Size)
    pairwise_embedding_distance_matrix = torch.cdist(predictions.float(), predictions.float(), p=2.0)
    
    # 2. Pairwise absolute differences in the target property: shape (Batch_Size, Batch_Size)
    pairwise_property_difference_matrix = torch.cdist(targets.float(), targets.float(), p=1.0)
    
    # 3. Hinge loss: max(0, |y_i - y_j| - ||h_i - h_j|| + margin)
    # This pushes embeddings apart if their property difference is large.
    pairwise_hinge_loss_matrix = neural_network_functional.relu(pairwise_property_difference_matrix - pairwise_embedding_distance_matrix + margin_threshold_value)
    
    if enforce_two_sided_penalty:
        # Optionally, also penalize if embeddings are further apart than properties
        pairwise_hinge_loss_matrix += neural_network_functional.relu(pairwise_embedding_distance_matrix - pairwise_property_difference_matrix - margin_threshold_value)
    
    # 4. Mask out the diagonal (a datapoint compared to itself shouldn't be penalized)
    self_comparison_exclusion_mask = ~torch.eye(current_batch_size, dtype=torch.bool, device=predictions.device)
    
    # 5. Apply per-sample weighting (outer product so both items in pair contribute)
    reshaped_per_sample_weight_vector = torch.as_tensor(sample_importance_weights, dtype=pairwise_hinge_loss_matrix.dtype, device=pairwise_hinge_loss_matrix.device).view(-1, 1)
    pairwise_importance_weight_matrix = reshaped_per_sample_weight_vector @ reshaped_per_sample_weight_vector.t()
    
    valid_off_diagonal_losses = pairwise_hinge_loss_matrix[self_comparison_exclusion_mask]
    valid_off_diagonal_weights = pairwise_importance_weight_matrix[self_comparison_exclusion_mask]
    
    if valid_off_diagonal_losses.numel() == 0:
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)
        
    return (valid_off_diagonal_losses * valid_off_diagonal_weights).sum() / valid_off_diagonal_weights.sum().clamp_min(1e-6)

# =========================================================================== #
# Model Weights Loss Functions:
# =========================================================================== #


def extract_learnable_parameters_sequence(model: neural_network.Module):
    # Conceptual Idea: 
    # Extract only the active, unfrozen parameters dynamically attached to the computational graph.
    learnable_parameters_sequence = (model.get_learnable_weights().values()
              if hasattr(model, "get_learnable_weights")
              else (parameter_tensor for parameter_tensor in model.parameters() if parameter_tensor.requires_grad))
    return learnable_parameters_sequence

@register("loss", "l1_penalty")
def calculate_l1_regularization_penalty(model: neural_network.Module, **ignored_keyword_arguments):
    learnable_parameters_sequence = extract_learnable_parameters_sequence(model)
    return sum(parameter_tensor.abs().sum() for parameter_tensor in learnable_parameters_sequence)

@register("loss", "l2_penalty")
def calculate_l2_regularization_penalty(model: neural_network.Module, **ignored_keyword_arguments):
    learnable_parameters_sequence = extract_learnable_parameters_sequence(model)
    return sum((parameter_tensor ** 2).sum() for parameter_tensor in learnable_parameters_sequence)

# =========================================================================== #
# Loss Aggregators:
# =========================================================================== #

syntax_for_task_mask_key = "mask"
syntax_for_task_weighting_key = "weighting"
syntax_for_task_targets_key = "values"

def retrieve_weighting_vector_from_batch_for_task(batch, target_task_name: str):
    task_specific_weighting_tensor = batch[target_task_name][syntax_for_task_weighting_key]
    return task_specific_weighting_tensor

def retrieve_mask_vector_from_batch_for_task(batch, target_task_name: str):
    task_specific_mask_tensor = batch[target_task_name][syntax_for_task_mask_key]
    return task_specific_mask_tensor

def retrieve_target_values_from_batch_for_task(batch, target_task_name: str):
    task_specific_target_values_tensor = batch[target_task_name][syntax_for_task_targets_key]
    return task_specific_target_values_tensor

# Assuming metric_weighting_column_name is defined upstream in the pipeline
def retrieve_metric_weights_from_batch_for_task(batch, target_task_name: str):
    task_specific_metric_weights_tensor = batch[syntax_for_metric_weighting_column_name]
    return task_specific_metric_weights_tensor

def apply_masks_to_targets_weights_and_predictions(targets, predictions, sample_importance_weights_tensor, batch, target_task_name, prediction_source_name=None):
    
    # --- NEW: Fallback to the task name if prediction_source isn't provided ---
    target_prediction_dictionary_key = prediction_source_name if prediction_source_name is not None else target_task_name
    model_prediction_logits_tensor = predictions[target_prediction_dictionary_key] 
    # --------------------------------------------------------------------------
    
    # Conceptual Idea: 
    # In sparse multi-task learning datasets, not every molecule has data for every assay. 
    # We dynamically select and calculate loss using strictly only the data points that are fully populated.
    
    # 1) Select the targets and predictions for the loss term's task.
    # model_prediction_logits_tensor is a dictionary mapping task names to their outputs.
    targets = retrieve_target_values_from_batch_for_task(batch, target_task_name)
    
    # 2) Get and apply the mask that removes empty values for the task.
    task_availability_mask_tensor = retrieve_mask_vector_from_batch_for_task(batch, target_task_name)
    masked_prediction_logits_tensor, masked_target_values_tensor, masked_sample_importance_weights_tensor = model_prediction_logits_tensor[task_availability_mask_tensor], targets[task_availability_mask_tensor], sample_importance_weights_tensor[task_availability_mask_tensor]
    
    # 3) Return masked targets and predictions.
    return masked_target_values_tensor, masked_prediction_logits_tensor, masked_sample_importance_weights_tensor

@dataclass
class LossTerm:
    """
    Some notes about this class:
        - It is assumed that each loss term can be described by a few features.
        Those are, first, a name that enables it to be designated. Currently,
        this functionality isn't really used anywhere, but it might be used in
        the future in error messages or something else. Second, it has a
        loss_term_weight. In principle, this could have been rolled into the
        loss term functions themselves, however it seems like essentially all
        kinds of loss terms might have the need for different weighting, so
        this is probably a good thing to process for all. Third, this saves the
        loss term function itself. This variable is sampled from one of the
        functions provided previously in this section. Finally, it has a task
        variable which declares which task it should be calculated on. Note
        that if a loss_term should work on multiple tasks, multiple should be
        created, one for each task.
    """
    
    loss_term_name: str
    loss_term_weighting_factor: float
    loss_term_mathematical_function: Callable          # atomic criterion
    target_task_name: str | None = None
    prediction_source_name: str | None = None  # <--- NEW FIELD

    def retrieve_task_name_for_loss_term(self):
        return self.target_task_name
    
    def retrieve_loss_term_name(self):
        # functools.partial (used when a loss has kwargs in the config) has no __name__, so
        # fall back to the wrapped function's name, then to the type name.
        loss_function_reference = self.loss_term_mathematical_function
        base_function_name = getattr(loss_function_reference, "__name__", None) \
            or getattr(getattr(loss_function_reference, "func", None), "__name__", type(loss_function_reference).__name__)
        return f"{base_function_name}_for_{self.retrieve_task_name_for_loss_term()}"
    
    # Prediction source lets you override the assumption that the prediction
    # are the task - the triplet loss violates this assumption.
    
    def __call__(
        self,
        model: neural_network.Module,
        predictions,
        batch,
        training_epoch_number: int,
        prediction_source_override: str | None = None  # <--- NEW FIELD
    ) -> torch.Tensor:
        
        # 0) Commonly used variables.
        resolved_task_name = self.retrieve_task_name_for_loss_term()
        
        # If it has a task (most losses).
        if resolved_task_name is not None:
            targets = retrieve_target_values_from_batch_for_task(batch, resolved_task_name)
            
            # Conceptual Idea: 
            # Retrieve identically shaped mask parameters selectively cleaning mathematical inputs logically perfectly scaled linearly.
            # 1) Mask targets and predictions.
            per_sample_importance_weights_tensor = retrieve_weighting_vector_from_batch_for_task(batch, resolved_task_name)
            masked_target_values_tensor, masked_prediction_logits_tensor, masked_per_sample_weights_tensor = apply_masks_to_targets_weights_and_predictions(targets, predictions, per_sample_importance_weights_tensor, batch, resolved_task_name, prediction_source_override or self.prediction_source_name)
            
            # 2) Compute the loss for the loss term, before reweighting. If every sample for this
            # task was masked out of the batch (sparse multi-task labels), the pointwise losses
            # would reshape a 0-row tensor and raise, so short-circuit to a graph-connected zero.
            if masked_prediction_logits_tensor.shape[0] == 0:
                pre_weighting_computed_loss_term_value = masked_prediction_logits_tensor.sum() * 0.0
            else:
                pre_weighting_computed_loss_term_value = self.loss_term_mathematical_function(model=model,
                              predictions=masked_prediction_logits_tensor, targets=masked_target_values_tensor,
                              batch=batch, epoch=training_epoch_number, sample_importance_weights=masked_per_sample_weights_tensor)
            
        else:
            pre_weighting_computed_loss_term_value = self.loss_term_mathematical_function(model=model)
            
        # 3) Compute and return the weight loss_term loss.
        weighted_final_loss_term_value = self.loss_term_weighting_factor * pre_weighting_computed_loss_term_value
            
        return weighted_final_loss_term_value

class LossFunction(neural_network.Module):
    """
    Evaluates every LossTerm, logs individual values, returns weighted sum.
    """

    def __init__(self, loss_term_adapters: List[LossTerm]):
        super().__init__()
        self.loss_term_adapters = loss_term_adapters

    def forward(
        self,
        *,
        model: neural_network.Module,
        predictions: Dict[str, torch.Tensor],
        batch,
        training_epoch_number: int,
    ) -> Dict[str, torch.Tensor]:
        
        # Conceptual Idea: 
        # Calculate isolated objective metrics correctly mathematically aggregating identical task representations cleanly globally.
        # 1) Instantiate output.
        accumulated_task_losses_dictionary: Dict[str, torch.Tensor] = {}
        total_global_accumulated_loss = torch.zeros((), device=next(model.parameters()).device)
        
        for individual_loss_term_adapter in self.loss_term_adapters:
            associated_task_name = individual_loss_term_adapter.retrieve_loss_term_name()
            calculated_loss_for_task_term = individual_loss_term_adapter(model, predictions, batch, training_epoch_number)
            accumulated_task_losses_dictionary[associated_task_name] = calculated_loss_for_task_term
            total_global_accumulated_loss = total_global_accumulated_loss + calculated_loss_for_task_term
            
        accumulated_task_losses_dictionary["total"] = total_global_accumulated_loss
        return accumulated_task_losses_dictionary


def construct_loss_function_from_configuration(loss_configuration: List[Dict]) -> LossFunction:
    # Conceptual Idea: 
    # Parse a declarative configuration list containing user-defined objective definitions, 
    # fetch the actual implementations from the global registry, and compose them into a unified evaluation block.
    loss_term_adapters: List[LossTerm] = []
    for loss_configuration_item_dictionary in loss_configuration:
        selected_loss_term_mathematical_function = retrieve_loss_component(loss_configuration_item_dictionary["type"])
        
        # --- NEW: Inject kwargs into the loss function dynamically ---
        loss_function_keyword_arguments = loss_configuration_item_dictionary.get("kwargs", {})
        if loss_function_keyword_arguments:
            selected_loss_term_mathematical_function = partial(selected_loss_term_mathematical_function, **loss_function_keyword_arguments)
        # -------------------------------------------------------------
        
        default_loss_term_name_string = loss_configuration_item_dictionary["type"]
        associated_target_tasks = listify(loss_configuration_item_dictionary.get("task", [None]))
        for individual_target_task_name in associated_target_tasks:
            loss_term_adapters.append(
                LossTerm(
                    loss_term_name  = loss_configuration_item_dictionary.get("name", default_loss_term_name_string),
                    target_task_name  = individual_target_task_name,
                    loss_term_weighting_factor = loss_configuration_item_dictionary.get("weight", 1.0),
                    loss_term_mathematical_function = selected_loss_term_mathematical_function,
                    prediction_source_name = loss_configuration_item_dictionary.get("prediction_source") # <--- Extract from YAML
                )
            )
    return LossFunction(loss_term_adapters)

# ########################################################################### #
# 6. Optimizers:
# ########################################################################### #

class OptimizerBase:
    optimizer_class: type[torch.optim.Optimizer]

    def __init__(self, **keyword_arguments):
        # Conceptual Idea: 
        # Safely validate incoming configurations by inspecting the actual 
        # underlying PyTorch optimizer signature, dynamically raising explicit 
        # errors if a user passes unsupported hyperparameters.
        supported_parameters = set(inspect.signature(self.optimizer_class).parameters)
        unsupported_parameters = sorted(set(keyword_arguments) - supported_parameters)
        if unsupported_parameters:
            raise TypeError(
                f"{self.__class__.__name__} received unsupported optimizer kwargs {unsupported_parameters}. "
                f"These were intended for {self.optimizer_class.__name__}, which only accepts: {sorted(supported_parameters)}."
            )
        self.keyword_arguments = keyword_arguments

    def __call__(self, model_parameters) -> torch.optim.Optimizer:
        return self.optimizer_class(model_parameters, **self.keyword_arguments)


@register("optimizer", "adamw")
class AdamWBuilder(OptimizerBase):
    optimizer_class = torch.optim.AdamW

    def __init__(self, learning_rate: float = 3e-4, weight_decay_coefficient: float = 0.0, **keyword_arguments):
        super().__init__(lr=learning_rate, weight_decay=weight_decay_coefficient, **keyword_arguments)


@register("optimizer", "sgd")
class SGDBuilder(OptimizerBase):
    optimizer_class = torch.optim.SGD

    def __init__(self, learning_rate: float = 1e-1, momentum_coefficient: float = 0.9, **keyword_arguments):
        super().__init__(lr=learning_rate, momentum=momentum_coefficient, **keyword_arguments)


def optimizer_from_optimizer_config(model_parameters, optimizer_configuration_dictionary: Dict) -> torch.optim.Optimizer:
    """
    Example cfg:

        optim:
          type: adamw
          kwargs: {lr: 1e-4, weight_decay: 1e-2}
    """
    # Conceptual Idea: 
    # Read the text configuration mapping, fetch the corresponding builder block 
    # from the global component registry, and instantiate the PyTorch optimizer attached to the model parameters.
    OptimizerBuilderClass = retrieve_optimizer_component(optimizer_configuration_dictionary["type"])
    optimizer_builder_instance = OptimizerBuilderClass(**optimizer_configuration_dictionary.get("kwargs", {}))
    return optimizer_builder_instance(model_parameters)      # returns the constructed optimizer

# ########################################################################### #
# 7. Performance Metrics:
# ########################################################################### #

def process_targets_predictions_logits(true_target_labels, predicted_logits, number_of_classes, requested_return_type: str = "probabilities"):
    # Conceptual Idea: 
    # Standardize neural network output formats before metrics calculation. Depending on the task 
    # (binary vs multiclass) and the requested format, safely map raw logits into probabilities 
    # (via sigmoid/softmax) or hard class predictions (via argmax).
    predicted_logits = predicted_logits.detach().float()
    if predicted_logits.dim() == 1:
        predicted_logits = predicted_logits.unsqueeze(-1)
    if predicted_logits.size(-1) == 1:
        prediction_probabilities = torch.sigmoid(predicted_logits.squeeze(-1))
        predicted_logits = torch.stack([1.0 - prediction_probabilities, prediction_probabilities], dim=-1)
    else:
        predicted_logits = predicted_logits.softmax(dim=-1)

    if true_target_labels.dim() == 2 and true_target_labels.size(-1) == number_of_classes:
        target_class_indices = true_target_labels.argmax(dim=-1)
    elif true_target_labels.dim() == 2 and true_target_labels.size(-1) == 1:
        target_class_indices = true_target_labels.view(-1)
    else:
        target_class_indices = true_target_labels

    if requested_return_type == "class":
        return target_class_indices.long(), predicted_logits.argmax(dim=-1)
    if requested_return_type == "probabilities":
        true_target_labels = neural_network_functional.one_hot(target_class_indices.long(), num_classes=predicted_logits.size(-1)).to(dtype=predicted_logits.dtype)
        return true_target_labels, predicted_logits
    raise ValueError(f"Invalid return_type: {requested_return_type}")

class MetricModule(abc.ABC):

    def __init__(self, target_task_name: str):
        self.target_task_name = target_task_name

    def get_predictions_for_task(self, target_task_name, model_predictions_dictionary):
        return model_predictions_dictionary.get(target_task_name)
    
    def get_targets_for_task(self, target_task_name, true_target_labels):
        return true_target_labels.get(target_task_name)
    
    def get_task(self):
        return self.target_task_name
    
    def get_metric_module_name(self) -> str:
        default_metric_name_string = self.__class__.__name__
        return getattr(self, "name", default_metric_name_string)

    def update(self, model_predictions_dictionary, true_target_labels, data_batch_dictionary, sample_metric_weighting=None, missing_targets_default_return_value=0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, bool]:        
        """
        The base update method essentially just dose two things all the metrics
        need:
            1. It masks the targets and predictions given the task information.
            2. It checks if there aren't any samples. If so, it updates the
            last batch info with a default symbol (can vary based on metric)
            and return a flag (honored in the metric-specific call) that tells
            the method-specific update call to potentially stop execution of
            update().
        """
        
        # 1) Mask targets and predictions.
        # Conceptual Idea: 
        # Safely slice down the batch to only look at values where ground truth data 
        # actually exists for this specific task metric, filtering out structural blanks in sparse multitask datasets.
        sample_metric_weighting = (
            torch.ones(model_predictions_dictionary[self.target_task_name].size(0), device=model_predictions_dictionary[self.target_task_name].device)
            if sample_metric_weighting is None
            else torch.as_tensor(sample_metric_weighting, device=model_predictions_dictionary[self.target_task_name].device, dtype=torch.float)
        )
        masked_target_values, masked_prediction_logits, masked_sample_weights = apply_masks_to_targets_weights_and_predictions(
            true_target_labels, model_predictions_dictionary, sample_metric_weighting, data_batch_dictionary, self.target_task_name
        )
        
        # 2) If targets is empty return the default value. Furthermore,
        # return a bool stating whether the remaining method should continue
        # or halt.
        total_weight_sum = float(masked_sample_weights.sum().item())
        if masked_sample_weights.numel() == 0 or total_weight_sum == 0:
            self.last_computed_batch_value = missing_targets_default_return_value
            should_continue_execution = False
        else:
            should_continue_execution = True
        return masked_target_values, masked_prediction_logits, masked_sample_weights, total_weight_sum, should_continue_execution
    
    @abc.abstractmethod
    def metric_on_batch(self) -> float: ...
    
    @abc.abstractmethod
    def metric_on_epoch(self) -> float: ...

    @abc.abstractmethod
    def reset(self) -> None: ...


@register("metric", "accuracy")
class AccuracyMetric(MetricModule):
    def __init__(self, target_task_name: str):
        
        super().__init__(target_task_name=target_task_name)
        self.running_total_weighted_correct_predictions = 0.0
        self.running_total_accumulated_weights_sum = 0.0

    def update(self, model_predictions_dictionary, data_batch_dictionary, sample_metric_weighting=None) -> None:
        
        sample_metric_weighting = retrieve_metric_weights_from_batch_for_task(data_batch_dictionary, self.get_task())
        true_target_labels = retrieve_target_values_from_batch_for_task(data_batch_dictionary, self.get_task())
        
        masked_target_values, masked_prediction_logits, masked_sample_weights, total_weight_sum, should_continue_execution = super().update(
            model_predictions_dictionary, true_target_labels, data_batch_dictionary, sample_metric_weighting
        )
        if not should_continue_execution: return
        
        # 1) Process targets and predictions.
        # Conceptual Idea: 
        # Evaluate hard classification accuracy by directly comparing the highest 
        # probability output bin directly against the index of the true label.
        masked_target_values, masked_hard_class_predictions = process_targets_predictions_logits(
            masked_target_values, masked_prediction_logits, number_of_classes=2, requested_return_type="class"
        )
        
        # 2) Compute accuracy. 
        weighted_correct_predictions_sum = float(
            ((masked_hard_class_predictions == masked_target_values).float() * masked_sample_weights).sum().item()
        )
        self.running_total_weighted_correct_predictions += weighted_correct_predictions_sum
        self.running_total_accumulated_weights_sum        += total_weight_sum
        self.last_computed_batch_value = float(weighted_correct_predictions_sum / total_weight_sum)

    def metric_on_batch(self) -> float:
        return self.last_computed_batch_value

    def metric_on_epoch(self) -> float:
        return float(self.running_total_weighted_correct_predictions / max(self.running_total_accumulated_weights_sum, 1e-12))

    def reset(self) -> None:
        self.running_total_weighted_correct_predictions = 0.0
        self.running_total_accumulated_weights_sum = 0.0
        self.last_computed_batch_value = None


@register("metric", "rmse")
class RMSEMetric(MetricModule):
    def __init__(self, target_task_name):
        super().__init__(target_task_name=target_task_name)
        self.running_total_weighted_mean_squared_error_sum = 0.0
        self.running_total_accumulated_weights_sum = 0.0

    def update(self, model_predictions_dictionary, data_batch_dictionary, sample_metric_weighting=None) -> None:
        
        sample_metric_weighting = retrieve_metric_weights_from_batch_for_task(data_batch_dictionary, self.get_task())
        true_target_labels = retrieve_target_values_from_batch_for_task(data_batch_dictionary, self.get_task())
        
        masked_target_values, masked_prediction_logits, masked_sample_weights, total_weight_sum, should_continue_execution = super().update(
            model_predictions_dictionary, true_target_labels, data_batch_dictionary, sample_metric_weighting
        )
        if not should_continue_execution: return
        
        # 1) Compute RMSE.
        # Conceptual Idea: 
        # Calculate the mathematical mean squared error independently per row, weight it logically 
        # against our dataset importance factors, and hold the summed accumulation so we can root it properly at epoch end.
        # Align shapes before subtracting. A prediction of [N, 1] (e.g. a regression head) minus a
        # target of [N] broadcasts to [N, N] and silently computes the wrong error. Reshaping both to
        # [N, K] (K=1 for scalar regression) forces a strictly element-wise difference.
        prediction_values_2d = masked_prediction_logits.reshape(masked_sample_weights.size(0), -1)
        target_values_2d = masked_target_values.reshape(masked_sample_weights.size(0), -1)
        per_sample_squared_error = (prediction_values_2d - target_values_2d).pow(2).mean(dim=1)
        
        weighted_mean_squared_error_sum = float((per_sample_squared_error * masked_sample_weights).sum().item())
        self.running_total_weighted_mean_squared_error_sum += weighted_mean_squared_error_sum
        self.running_total_accumulated_weights_sum        += total_weight_sum
        self.last_computed_batch_value = float(math.sqrt(weighted_mean_squared_error_sum / total_weight_sum))
        
    def metric_on_batch(self) -> float:
        return self.last_computed_batch_value

    def metric_on_epoch(self) -> float:
        return float(math.sqrt(self.running_total_weighted_mean_squared_error_sum / max(self.running_total_accumulated_weights_sum, 1e-12)))

    def reset(self) -> None:
        self.running_total_weighted_mean_squared_error_sum = 0.0
        self.running_total_accumulated_weights_sum = 0.0
        self.last_computed_batch_value = None


@register("metric", "auroc")
class AUROCMetric(MetricModule):
    def __init__(self, *, target_task_name: str, number_of_classes: int = 2, averaging_method_name: str = "macro"):
        super().__init__(target_task_name=target_task_name)
        self.number_of_classes = int(number_of_classes)
        self.averaging_method_name = averaging_method_name
        self.accumulated_scores, self.accumulated_targets, self.accumulated_weights, self.last_computed_batch_value = [], [], [], None

    def compute_area_under_receiver_operating_characteristic(self, predicted_probabilities, true_target_labels, sample_importance_weights=None):
        # Conceptual Idea: 
        # Connect to Scikit-Learn's robust AUROC mathematical calculation. We safely offload 
        # tensors to numpy matrices directly ensuring the shapes match what the downstream function expects.
        if self.number_of_classes not in (1, 2) or predicted_probabilities.ndim != 2 or predicted_probabilities.size(-1) != 2:
            raise ValueError("AUROCMetric (sklearn) supports only binary classification (1 or 2 logits).")
        true_target_labels = (true_target_labels[:, -1] if true_target_labels.ndim == 2 else true_target_labels).detach().numpy()
        predicted_probabilities = predicted_probabilities[:, -1].detach().numpy()
        numpy_sample_weights = None if sample_importance_weights is None else sample_importance_weights.detach().numpy()
        # AUROC is undefined unless both classes are present. With rare hits, a batch (or even an entire
        # fold for a sparse task) can be single-class, in which case sklearn emits a warning and returns
        # nan (or raises, depending on version). Return nan explicitly so the result is deterministic.
        if np.unique(true_target_labels).size < 2:
            return float("nan")
        return float(sklearn_metrics_area_under_receiver_operating_characteristic(true_target_labels, predicted_probabilities, sample_weight=numpy_sample_weights))

    def update(self, model_predictions_dictionary, data_batch_dictionary) -> None:
        sample_metric_weighting = retrieve_metric_weights_from_batch_for_task(data_batch_dictionary, self.get_task())
        true_target_labels = retrieve_target_values_from_batch_for_task(data_batch_dictionary, self.get_task())

        masked_target_values, masked_prediction_logits, masked_sample_weights, _, should_continue_execution = super().update(
            model_predictions_dictionary, true_target_labels, data_batch_dictionary, sample_metric_weighting
        )
        if not should_continue_execution:
            return

        # 1) Process targets and predictions.
        masked_target_values, masked_prediction_logits = process_targets_predictions_logits(
            masked_target_values, masked_prediction_logits, 2, requested_return_type="probabilities"
        )

        masked_target_values = masked_target_values.detach().cpu()
        masked_prediction_logits = masked_prediction_logits.detach().cpu()
        masked_sample_weights = masked_sample_weights.detach().cpu()

        self.last_computed_batch_value = self.compute_area_under_receiver_operating_characteristic(masked_prediction_logits, masked_target_values, masked_sample_weights)
        self.accumulated_scores.append(masked_prediction_logits)
        self.accumulated_targets.append(masked_target_values)
        self.accumulated_weights.append(masked_sample_weights)

    def metric_on_batch(self) -> float:
        return self.last_computed_batch_value

    def metric_on_epoch(self) -> float:
        # Conceptual Idea: 
        # The AUROC metric is globally non-decomposable, meaning we cannot safely average 
        # independent batch AUROC scores. We must instead aggregate all predictions and targets across 
        # the entire epoch, and perform one global integration calculation at the end.
        if not self.accumulated_targets:
            return float("nan")
        concatenated_scores_tensor = torch.cat(self.accumulated_scores, dim=0)
        concatenated_labels_tensor = torch.cat(self.accumulated_targets, dim=0)
        concatenated_weights_tensor = torch.cat(self.accumulated_weights, dim=0)
        return self.compute_area_under_receiver_operating_characteristic(concatenated_scores_tensor, concatenated_labels_tensor, concatenated_weights_tensor)

    def reset(self) -> None:
        self.accumulated_scores.clear()
        self.accumulated_targets.clear()
        self.accumulated_weights.clear()
        self.last_computed_batch_value = None


@register("metric", "auprc")
class AUPRCMetric(MetricModule):
    def __init__(self, *, target_task_name: str, number_of_classes: int = 2, averaging_method_name: str = "macro"):
        super().__init__(target_task_name=target_task_name)
        self.number_of_classes = int(number_of_classes)
        self.averaging_method_name = averaging_method_name
        self.accumulated_scores, self.accumulated_targets, self.accumulated_weights, self.last_computed_batch_value = [], [], [], None

    def compute_area_under_precision_recall_curve(self, predicted_probabilities, true_target_labels, sample_importance_weights=None):
        # Conceptual Idea: 
        # Connect directly to Scikit-Learn's AUPRC implementation. This metric is especially 
        # useful for highly imbalanced class distributions where AUROC can be overly optimistic.
        if self.number_of_classes not in (1, 2) or predicted_probabilities.ndim != 2 or predicted_probabilities.size(-1) != 2:
            raise ValueError("AUPRCMetric (sklearn) supports only binary classification (1 or 2 logits).")
        true_target_labels = (true_target_labels[:, -1] if true_target_labels.ndim == 2 else true_target_labels).detach().numpy()
        predicted_probabilities = predicted_probabilities[:, -1].detach().numpy()
        numpy_sample_weights = None if sample_importance_weights is None else sample_importance_weights.detach().numpy()
        # Average precision is undefined with no positive labels (and degenerate when all are positive).
        # With rare hits this happens often per batch, so return nan rather than sklearn's warning plus a
        # misleading 0.0 / 1.0.
        if np.unique(true_target_labels).size < 2:
            return float("nan")
        return float(average_precision_score(true_target_labels, predicted_probabilities, sample_weight=numpy_sample_weights))

    def update(self, model_predictions_dictionary, data_batch_dictionary) -> None:
        sample_metric_weighting = retrieve_metric_weights_from_batch_for_task(data_batch_dictionary, self.get_task())
        true_target_labels = retrieve_target_values_from_batch_for_task(data_batch_dictionary, self.get_task())

        masked_target_values, masked_prediction_logits, masked_sample_weights, _, should_continue_execution = super().update(
            model_predictions_dictionary, true_target_labels, data_batch_dictionary, sample_metric_weighting
        )
        if not should_continue_execution:
            return

        # 1) Process targets and predictions.
        masked_target_values, masked_prediction_logits = process_targets_predictions_logits(
            masked_target_values, masked_prediction_logits, 2, requested_return_type="probabilities"
        )

        masked_target_values = masked_target_values.detach().cpu()
        masked_prediction_logits = masked_prediction_logits.detach().cpu()
        masked_sample_weights = masked_sample_weights.detach().cpu()

        self.last_computed_batch_value = self.compute_area_under_precision_recall_curve(masked_prediction_logits, masked_target_values, masked_sample_weights)
        self.accumulated_scores.append(masked_prediction_logits)
        self.accumulated_targets.append(masked_target_values)
        self.accumulated_weights.append(masked_sample_weights)

    def metric_on_batch(self) -> float:
        return self.last_computed_batch_value

    def metric_on_epoch(self) -> float:
        if not self.accumulated_targets:
            return float("nan")
        concatenated_scores_tensor = torch.cat(self.accumulated_scores, dim=0)
        concatenated_labels_tensor = torch.cat(self.accumulated_targets, dim=0)
        concatenated_weights_tensor = torch.cat(self.accumulated_weights, dim=0)
        return self.compute_area_under_precision_recall_curve(concatenated_scores_tensor, concatenated_labels_tensor, concatenated_weights_tensor)

    def reset(self) -> None:
        self.accumulated_scores.clear()
        self.accumulated_targets.clear()
        self.accumulated_weights.clear()
        self.last_computed_batch_value = None

@register("metric", "r2")
class R2Metric(MetricModule):
    """Weighted coefficient of determination R^2 = 1 - SS_res/SS_tot, per task.

    Accumulates weighted SS_res and the weighted target moments across the whole validation epoch, so
    R^2 is computed globally (NOT as the mean of per-batch R^2). Returns nan when the task has no
    labeled rows in the window or the targets carry no variance (R^2 is undefined there).
    """
    def __init__(self, target_task_name):
        super().__init__(target_task_name=target_task_name)
        self.reset()

    @staticmethod
    def _r2_from_moments(weighted_sum_squared_residuals, weighted_sum_targets, weighted_sum_squared_targets, total_weight):
        if total_weight <= 0:
            return float("nan")
        total_sum_of_squares = weighted_sum_squared_targets - (weighted_sum_targets * weighted_sum_targets) / total_weight
        if total_sum_of_squares <= 1e-12:
            return float("nan")
        return float(1.0 - weighted_sum_squared_residuals / total_sum_of_squares)

    def update(self, model_predictions_dictionary, data_batch_dictionary, sample_metric_weighting=None) -> None:
        sample_metric_weighting = retrieve_metric_weights_from_batch_for_task(data_batch_dictionary, self.get_task())
        true_target_labels = retrieve_target_values_from_batch_for_task(data_batch_dictionary, self.get_task())
        masked_target_values, masked_prediction_logits, masked_sample_weights, total_weight_sum, should_continue_execution = super().update(
            model_predictions_dictionary, true_target_labels, data_batch_dictionary, sample_metric_weighting
        )
        if not should_continue_execution:
            return
        # Align shapes to [N, K] (K=1 for scalar regression) so the subtraction is strictly element-wise.
        prediction_values_2d = masked_prediction_logits.reshape(masked_sample_weights.size(0), -1)
        target_values_2d = masked_target_values.reshape(masked_sample_weights.size(0), -1)
        per_sample_squared_error = (prediction_values_2d - target_values_2d).pow(2).mean(dim=1)
        per_sample_target = target_values_2d.mean(dim=1)
        batch_weighted_sum_squared_residuals = float((per_sample_squared_error * masked_sample_weights).sum().item())
        batch_weighted_sum_targets = float((per_sample_target * masked_sample_weights).sum().item())
        batch_weighted_sum_squared_targets = float((per_sample_target.pow(2) * masked_sample_weights).sum().item())
        self.running_weighted_sum_squared_residuals += batch_weighted_sum_squared_residuals
        self.running_weighted_sum_targets += batch_weighted_sum_targets
        self.running_weighted_sum_squared_targets += batch_weighted_sum_squared_targets
        self.running_total_accumulated_weights_sum += total_weight_sum
        self.last_computed_batch_value = self._r2_from_moments(
            batch_weighted_sum_squared_residuals, batch_weighted_sum_targets,
            batch_weighted_sum_squared_targets, total_weight_sum)

    def metric_on_batch(self) -> float:
        return self.last_computed_batch_value

    def metric_on_epoch(self) -> float:
        return self._r2_from_moments(
            self.running_weighted_sum_squared_residuals, self.running_weighted_sum_targets,
            self.running_weighted_sum_squared_targets, self.running_total_accumulated_weights_sum)

    def reset(self) -> None:
        self.running_weighted_sum_squared_residuals = 0.0
        self.running_weighted_sum_targets = 0.0
        self.running_weighted_sum_squared_targets = 0.0
        self.running_total_accumulated_weights_sum = 0.0
        self.last_computed_batch_value = None


class MetricStack:
    """
    Holds MetricModule objects. Call update(...) per batch, compute() at epoch end.
    
    
    """
    def __init__(self, metric_modules: list[MetricModule]):
        self.metric_modules = metric_modules
    
    def iterate_over_modules(self):
        for metric_module in self.metric_modules:
            yield metric_module
    
    def update(self, model_predictions_dictionary, data_batch_dictionary):
        # Conceptual Idea: 
        # Pipeline execution block. Simply pass the predictions sequentially to every 
        # registered metric module so they can internally update their continuous running states.
        for metric_module in self.iterate_over_modules():
            metric_module.update(model_predictions_dictionary, data_batch_dictionary)
    
    def compute(self, requested_sample_scope: str = "epoch") -> Dict[str, float]:
        computed_output_dictionary = {}
        for metric_module in self.iterate_over_modules():
            
            # Given a module, determine what it's name in the output should be.
            base_metric_name_string = metric_module.__class__.__name__.replace("Metric", "").lower()
            formatted_metric_key  = f"{base_metric_name_string}_{metric_module.target_task_name}" if getattr(metric_module, "target_task_name", None) else base_metric_name_string  # <-- underscore
            
            if requested_sample_scope == "batch": 
                computed_output_dictionary[formatted_metric_key] = metric_module.metric_on_batch()
            elif requested_sample_scope == "epoch":
                computed_output_dictionary[formatted_metric_key] = metric_module.metric_on_epoch()
        
        return computed_output_dictionary
    
    def reset(self):
        for metric_module in self.iterate_over_modules():
            metric_module.reset()
            
def construct_metric_stack_from_configs(list_of_metric_configuration_dictionaries: List[Dict[str, Any]]) -> MetricStack:
    """
    cfg example:

        metrics:
          - { type: accuracy, task: cls }
          - { type: rmse, task: reg }
          - { type: auroc, task: cls, kwargs: { epoch_size: 5000, num_classes: 3, dtype: "float32" } }

    Notes
    -----
    • 'task' may be a string or a list of strings (one instance per task).
    • Extra 'kwargs' are passed to the metric's __init__, with signature filtering.
    """
    # Conceptual Idea: 
    # Read the provided configuration map, lookup the corresponding metric math functions from the 
    # internal class registry dynamically, and instantiate an array wrapping them cleanly for execution.
    instantiated_modules = []
    for metric_configuration in list_of_metric_configuration_dictionaries:
        
        # 1) Get metric methods.
        MetricClassReference = retrieve_metric_component(metric_configuration.get("type"))
        
        # 2) For each metric get the keyword arguments.
        extracted_keyword_arguments = dict(metric_configuration.get("kwargs") or {})
        
        # 3) For each task, add a metric for that task.
        target_tasks = metric_configuration.get("task")
        for target_task_name in target_tasks if isinstance(target_tasks, list) else [target_tasks]:
            instantiated_modules.append(MetricClassReference(target_task_name=target_task_name, **extracted_keyword_arguments))
    
    return MetricStack(instantiated_modules)

# ########################################################################### #
# 9. Dataset Splitting and Processing Related:
# ########################################################################### #

# =========================================================================== #
# General Syntax for Dataset Splitting
# =========================================================================== #

standard_group_column_name = "group_key"
standard_subset_column_name = "subset_key"

# # =========================================================================== #
# # Dataset Splitting Helper Functions:
# # =========================================================================== #

# # --------------------------------------------------------------------------- #
# # Base Functions:
# # --------------------------------------------------------------------------- #

# def subset_sizes_from_relative_sizes(
#     total_row_count: int, relative_split_proportions: Sequence[float]
# ) -> List[int]:
#     """Convert relative sizes into integer subset sizes that sum exactly to row_count."""
#     # Conceptual Idea: 
#     # Transforms continuous percentage ratios (like [0.8, 0.1, 0.1]) strictly into integer-valued 
#     # hard split bounds summing absolutely perfectly to the underlying dataframe's exact sample limit.
#     relative_size_numpy_array = numpy.asarray(relative_split_proportions, dtype=float)
#     total_relative_size_sum = float(relative_size_numpy_array.sum())
#     if total_relative_size_sum <= 0:
#         raise ValueError("relative_split_proportions must sum to a positive value.")

#     expected_absolute_sizes = relative_size_numpy_array / total_relative_size_sum * int(total_row_count)
#     computed_integer_subset_sizes = numpy.floor(expected_absolute_sizes).astype(int)

#     remaining_unallocated_rows = int(total_row_count) - int(computed_integer_subset_sizes.sum())
#     if remaining_unallocated_rows:
#         fractional_remainders = expected_absolute_sizes - computed_integer_subset_sizes
#         computed_integer_subset_sizes[numpy.argsort(fractional_remainders)[-remaining_unallocated_rows:]] += 1

#     return computed_integer_subset_sizes.tolist()

# # --------------------------------------------------------------------------- #
# # Group assignment functions (Table -> Table)
# # --------------------------------------------------------------------------- #

# def assign_to_groups_using_all_unique(
#     data_table: Table,
#     target_group_column_name: str = standard_group_column_name,
# ) -> Table:
#     """Each row is its own group (identity grouping)."""
#     # Conceptual Idea: 
#     # Create a random split where every single molecule is mathematically treated entirely uniquely, 
#     # possessing no underlying topological relationship constraining assignment clustering explicitly.
#     total_row_count = len(data_table)
#     sequential_identity_values = numpy.arange(total_row_count)

#     if data_table.column_in_table(target_group_column_name):
#         data_table.drop_columns([target_group_column_name], in_place=True)

#     return data_table.merge_with_pandas_dataframe(
#         pandas.DataFrame({target_group_column_name: sequential_identity_values})
#     )


# def assign_to_groups_using_prexisting_column(
#     data_table: Table,
#     source_reference_column_name: str,
#     target_group_column_name: str = standard_group_column_name,
# ) -> Table:
#     """
#     Group assignment that copies an existing column into group_column_name.

#     Usage: group_assignment_function = partial(make_existing_column_group_assigner, source_column_name="...")
#     """
#     data_table.assert_column_in_table(source_reference_column_name)
#     return data_table.assign_rowwise_using(
#         target_group_column_name, lambda row_data: row_data[source_reference_column_name], inplace=True
#     )


# def assign_to_groups_by_scaffold(
#     data_table: Table,
#     target_group_column_name: str = standard_group_column_name,
# ) -> Table:
#     """Assign Bemis–Murcko scaffold string as the grouping key."""
#     # Conceptual Idea: 
#     # To test model generalization outside its comfort zone, cluster molecules functionally based on 
#     # their underlying core carbon scaffold geometry so structurally related analogs are isolated firmly inside identical splits.
#     smiles_feature_column_name = data_table.get_smiles_column_name()
#     if smiles_feature_column_name is None:
#         raise ValueError("Table object does not have a SMILES column.")

#     def extract_bemis_murcko_scaffold_from_row(row_data):
#         return MurckoScaffold.MurckoScaffoldSmiles(smiles=row_data[smiles_feature_column_name])

#     return data_table.assign_rowwise_using(target_group_column_name, extract_bemis_murcko_scaffold_from_row, inplace=True)


# def assign_to_groups_using_butina_clustering(
#     data_table: Table,
#     *,
#     clustering_similarity_cutoff: float = 0.5,
#     fingerprint_radius: int = 2,
#     fingerprint_bit_count: int = 2048,
#     target_group_column_name: str = standard_group_column_name,
# ) -> Table:
#     """
#     Assign Butina cluster id (1..K) as the grouping key.

#     IMPORTANT (per request):
#       - Do NOT call morgan_fingerprints here.
#       - Compute pairwise distances by calling tanimoto_similarity or tanimoto_distance
#         (which caches fingerprints internally).
#     """
#     # Conceptual Idea: 
#     # Segment data by identifying completely disconnected dense spherical structural 
#     # similarity clusters using the unsupervised iterative Butina topological algorithm.
#     extracted_smiles_strings = list(data_table.get_smiles())
#     total_row_count = len(extracted_smiles_strings)
    
#     preconfigured_distance_function = partial(tanimoto_distance, radius=fingerprint_radius, bit_count=fingerprint_bit_count)
    
#     computed_pairwise_distances = [
#         preconfigured_distance_function(extracted_smiles_strings[row_index], extracted_smiles_strings[column_index])
#         for row_index in range(total_row_count)
#         for column_index in range(row_index)
#     ]

#     identified_clusters_tuple = Butina.ClusterData(computed_pairwise_distances, total_row_count, clustering_similarity_cutoff, isDistData=True)

#     cluster_identifier_by_row_array = numpy.empty(total_row_count, dtype=int)
#     for cluster_identifier_index, cluster_member_indices in enumerate(identified_clusters_tuple, start=1):
#         cluster_identifier_by_row_array[list(cluster_member_indices)] = cluster_identifier_index

#     if data_table.column_in_table(target_group_column_name):
#         data_table.drop_columns([target_group_column_name], in_place=True)

#     return data_table.merge_with_pandas_dataframe(
#         pandas.DataFrame({target_group_column_name: cluster_identifier_by_row_array})
#     )


# # --------------------------------------------------------------------------- #
# # Subset Identity Assignment:
# # --------------------------------------------------------------------------- #

# standard_dataset_group_key = "Group"

# def make_contiguous_identity_assigner(
#     data_table: Table,
#     target_absolute_subset_sizes: Sequence[float],
#     *,
#     row_ordering_function: Callable[[Table, int], Sequence[int]],
#     target_subset_column_name: str = "group_identity",
# ) -> Table:
#     """Assign subset ids by contiguous slices along a provided row order."""
#     # 1) Basic values.
#     # Conceptual Idea: 
#     # Using a predefined sort (e.g., temporally by publication date), carve the 
#     # sequential list sequentially exactly where the chunk dimension limits cap out.
#     total_row_count = len(data_table)

#     computed_row_order_array = numpy.asarray(row_ordering_function(data_table, total_row_count), dtype=int)
    
#     assigned_group_identity_array = numpy.empty(total_row_count, dtype=int)
#     current_start_index = 0
#     for subset_identifier_index, current_subset_size in enumerate(target_absolute_subset_sizes, start=1):
#         current_stop_index = current_start_index + int(current_subset_size)
#         assigned_group_identity_array[computed_row_order_array[current_start_index:current_stop_index]] = subset_identifier_index
#         current_start_index = current_stop_index

#     if data_table.column_in_table(target_subset_column_name):
#         data_table.drop_columns([target_subset_column_name], in_place=True)

#     return data_table.merge_with_pandas_dataframe(
#         pandas.DataFrame({target_subset_column_name: assigned_group_identity_array})
#     )


# def make_first_fit_group_identity_assigner(
#     data_table: Table,
#     target_absolute_subset_sizes: Sequence[float],
#     *,
#     source_group_column_name: str = standard_dataset_group_key,
#     target_subset_column_name: str = "group_identity",
#     randomization_seed: Union[int, None] = 42,
#     sort_groups_by_size_descending_flag: bool = False,
# ) -> Table:
#     """Assign subset ids by whole-group first-fit, with optional shuffling or size-sorting."""
    
#     # 1) Basic values.
#     total_row_count = len(data_table)
    
#     # 2) Create a dictionary mapping each group to a list of row indices.
#     # Conceptual Idea: 
#     # When preserving scaffolds or clusters, we cannot fracture a group across different splits. 
#     # We solve a packing problem by treating each logical group as a single indivisible block of data.
#     group_indices_by_value_dictionary: dict[str, list[int]] = {}
#     for row_index, specific_group_value in enumerate(data_table[source_group_column_name]):
#         group_indices_by_value_dictionary.setdefault(str(specific_group_value), []).append(row_index)
    
#     # 3) Get all groups IDs.
#     extracted_group_keys = list(group_indices_by_value_dictionary.keys())
    
#     # 4) Sort so the largest groups are looked at first...
#     if sort_groups_by_size_descending_flag:
#         extracted_group_keys.sort(key=lambda group_key: len(group_indices_by_value_dictionary[group_key]), reverse=True)
#     # else if a seed is provided randomize the list ...
#     elif randomization_seed is not None:
#         numpy.random.RandomState(randomization_seed).shuffle(extracted_group_keys)
#     # else just leave the order of groups as is.
#     else:
#         pass

#     # 5) Initialize the occupancy of all of the subsets and the final subset
#     # occupancy intiialization.
#     current_subset_fill_counts = [0] * len(target_absolute_subset_sizes)
#     assigned_group_identity_array = numpy.empty(total_row_count, dtype=int)

#     # 6) Look over each subset. If that subset doesn't can have a group added 
#     # to it, do so. If it can't be added to any but it is the last subset, add
#     # it to the last subset anyway. Might want to consider having it add the
#     # group to the smallest subset if no perfectly fitting subset can be found.
#     for current_group_key in extracted_group_keys:
#         associated_group_indices = group_indices_by_value_dictionary[current_group_key]
#         current_group_size = len(associated_group_indices)

#         for subset_index, maximum_subset_capacity in enumerate(target_absolute_subset_sizes, start=1):
#             if (
#                 current_subset_fill_counts[subset_index - 1] + current_group_size <= maximum_subset_capacity
#                 or subset_index == len(target_absolute_subset_sizes)
#             ):
#                 current_subset_fill_counts[subset_index - 1] += current_group_size
#                 assigned_group_identity_array[associated_group_indices] = subset_index
#                 break

#     return data_table.merge_with_pandas_dataframe(
#         pandas.DataFrame({target_subset_column_name: assigned_group_identity_array})
#     )

# def make_sequential_multilabel_stratified_identity_assigner(
#     data_table: Table,
#     target_absolute_subset_sizes: Sequence[float],
#     *,
#     target_label_column_names: Union[Sequence[str], str],
#     randomization_seed: int = 42,
#     target_subset_column_name: str = "group_identity",
# ) -> Table:
#     """Sequential multi-label stratified carve-out (iterstrat).

#     IMPORTANT (per request):
#       - Do NOT use _materialize_columns_as_numpy.
#       - Materialize label columns via Table.__getitem__ (i.e., table[col]).
#     """
#     # 1) Wrap target_label_column_names in a list if it is not already one.
#     target_label_column_names = listify(target_label_column_names)

#     # 2) Confirm all label columns exist.
#     for label_column_name in target_label_column_names:
#         data_table.assert_column_in_table(label_column_name)

#     # 3) Get the label columns into a numpy matrix.
#     materialized_label_columns = [numpy.asarray(list(data_table[label_column_name])) for label_column_name in target_label_column_names]
#     total_row_count = int(len(materialized_label_columns[0]))
#     concatenated_label_matrix = numpy.column_stack(materialized_label_columns)
    

#     # Conceptual Idea:
#     # Uses the iterstrat library to perform stratified splitting on multi-task sparse datasets, ensuring 
#     # that the proportions of positive hits across multiple classification tasks are optimally balanced in each split.
#     dummy_feature_matrix = numpy.zeros((total_row_count, 1))
#     assigned_group_identity_array = numpy.empty(total_row_count, dtype=int)
#     remaining_row_indices_array = numpy.arange(total_row_count)

#     for subset_numerical_identifier, current_subset_size in enumerate(target_absolute_subset_sizes[:-1], start=1):

#         stratified_splitting_module = MultilabelStratifiedShuffleSplit(
#             n_splits=1,
#             test_size=current_subset_size,
#             random_state=randomization_seed + subset_numerical_identifier,
#         )
#         remaining_relative_indices, selected_relative_indices = next(
#             stratified_splitting_module.split(
#                 dummy_feature_matrix[remaining_row_indices_array],
#                 concatenated_label_matrix[remaining_row_indices_array],
#             )
#         )
#         selected_absolute_row_indices = remaining_row_indices_array[selected_relative_indices]
#         assigned_group_identity_array[selected_absolute_row_indices] = subset_numerical_identifier
#         remaining_row_indices_array = remaining_row_indices_array[remaining_relative_indices]

#     assigned_group_identity_array[remaining_row_indices_array] = len(target_absolute_subset_sizes)


#     return data_table.merge_with_pandas_dataframe(
#         pandas.DataFrame({target_subset_column_name: assigned_group_identity_array})
#     )

# def make_sequential_maxmin_identity_assigner(
#     data_table: Table,
#     target_absolute_subset_sizes: Sequence[float],
#     *,
#     randomization_seed: int = 0,
#     fingerprint_radius: int = 2,
#     fingerprint_bit_count: int = 2048,
#     target_subset_column_name: str = "group_identity",
# ) -> Table:
#     """
#     Sequential MaxMin: choose subset K from all rows, then subset K-1 from remainder, ..., leaving subset 1 last.
#     """
    
#     # 1) Assign some general variables.
#     extracted_smiles_strings = list(data_table.get_smiles())
#     total_row_count = len(extracted_smiles_strings)

#     # 2) 
#     # Conceptual Idea:
#     # Sequentially extracts validation and test sets by greedily picking molecules that are mathematically 
#     # as far apart from each other as possible. This forces the model to extrapolate rather than merely interpolate.
#     computed_morgan_fingerprints = morgan_fingerprints(extracted_smiles_strings, fingerprint_radius, fingerprint_bit_count)

#     tanimoto_similarity_function = DataStructs.TanimotoSimilarity

#     remaining_row_indices = list(range(total_row_count))
#     assigned_group_identity_array = numpy.ones(total_row_count, dtype=int)

#     for selection_round_index, target_subset_index in enumerate(range(len(target_absolute_subset_sizes), 1, -1)):
#         current_target_subset_size = int(target_absolute_subset_sizes[target_subset_index - 1])

#         remaining_smiles_subset = [extracted_smiles_strings[row_index] for row_index in remaining_row_indices]

#         selected_relative_indices = list(
#             MaxMinPicker().LazyPick(
#                 tanimoto_distance,
#                 len(remaining_row_indices),
#                 current_target_subset_size,
#                 randomization_seed + selection_round_index,
#             )
#         )

#         selected_absolute_row_indices_set = {remaining_row_indices[relative_index] for relative_index in selected_relative_indices}
#         for selected_row_index in selected_absolute_row_indices_set:
#             assigned_group_identity_array[selected_row_index] = target_subset_index

#         remaining_row_indices = [
#             row_index
#             for row_index in remaining_row_indices
#             if row_index not in selected_absolute_row_indices_set
#         ]

#     if data_table.column_in_table(target_subset_column_name):
#         data_table.drop_columns([target_subset_column_name], in_place=True)

#     return data_table.merge_with_pandas_dataframe(
#         pandas.DataFrame({target_subset_column_name: assigned_group_identity_array})
#     )

# def make_joint_maxmin_identity_assigner(
#     data_table: Table,
#     target_absolute_subset_sizes: Sequence[float],
#     *,
#     randomization_seed: int = 0,
#     fingerprint_radius: int = 2,
#     fingerprint_bit_count: int = 2048,
#     target_subset_column_name: str = "group_identity",
# ) -> Table:
#     """
#     Joint MaxMin selection for all non-first subsets (subset 2..K) in one selection phase,
#     implemented per request as:

#       1) Collapse all holdout subsets [2:] into a single large holdout.
#       2) Call make_sequential_maxmin_identity_assigner to pick that large holdout.
#       3) Randomly split the selected holdout indices into subsets 2..K according to
#          the original subset sizes.

#     The first subset is the complement (commonly used as the "training" bucket).
#     """
#     # Conceptual Idea:
#     # A fairer alternative to sequential max-min. We bundle all holdout partitions (validation + testing) 
#     # into a single maximum-diversity pool first, then logically split it back out randomly. This avoids pushing the test set into tighter corners.
#     size_of_primary_training_subset = target_absolute_subset_sizes[0]
#     size_of_aggregated_holdout_subsets = sum(target_absolute_subset_sizes[1:])
#     temporary_joint_subset_sizes = [size_of_primary_training_subset, size_of_aggregated_holdout_subsets]
    
#     data_table_with_temporary_subsets = make_sequential_maxmin_identity_assigner(data_table, temporary_joint_subset_sizes, 
#                                              randomization_seed=randomization_seed, fingerprint_radius=fingerprint_radius,
#                                              fingerprint_bit_count=fingerprint_bit_count, 
#                                              target_subset_column_name=
#                                              target_subset_column_name
#                                              )


# # =========================================================================== #
# # Dataset Splitting Methods:
# # =========================================================================== #

# # --------------------------------------------------------------------------- #
# # Dataset Splitting Helper:
# # --------------------------------------------------------------------------- #

# def split_dataset_into_subsets(
#     data_table: Table,
#     relative_fold_sizes_sequence: Sequence[float],
#     group_assignment_function_reference: Callable[[Table], Table],
#     subset_assignment_function_reference: Callable[[Table, Sequence[float]], Table],
#     target_group_column_name = standard_group_column_name,
#     target_subset_column_name = standard_subset_column_name
# ) -> Table:
#     """
#     Scaffolding function: (1) assign groups, then (2) assign group identities (subset ids 1..K).
#     Returns a Table with the group identity column added.
#     """
#     # Conceptual Idea:
#     # General data pipeline execution wrapper. Resolves floating point size ratios to precise integer limits,
#     # executes a user-specified grouping logic function, then executes a subset assignment logic function, keeping the Table architecture clean.
#     # 1) Normalize all relative fold sizes.
#     total_row_count = len(data_table)
#     absolute_computed_subset_sizes = subset_sizes_from_relative_sizes(total_row_count, relative_fold_sizes_sequence)
    
#     # 2) Drop any old group and subset columns if present.
#     for existing_column_name in (target_group_column_name, target_subset_column_name):
#         if data_table.column_in_table(existing_column_name):
#             print(f"Warning: column {existing_column_name} is already present. Dropping and recomputing.")
#             data_table.drop_columns([existing_column_name], in_place=True)
    
#     # 3) Assign groups to molecules.
#     data_table_with_groups_assigned = group_assignment_function_reference(data_table)
#     data_table_with_groups_assigned.assert_column_in_table(target_group_column_name)
    
#     # 4) Assign subset occupancies for each molecule.
#     data_table_with_subsets_assigned = subset_assignment_function_reference(data_table_with_groups_assigned, absolute_computed_subset_sizes)
#     data_table_with_subsets_assigned.assert_column_in_table(target_subset_column_name)
    
#     # 5) Return aformentioned.
#     return data_table_with_subsets_assigned

# # --------------------------------------------------------------------------- #
# # Dataset Splitting Methods:
# # --------------------------------------------------------------------------- #

# '''
# These methods all essentially take in a table and return a Table with a
# subset column name calculated.
# '''

# def calculate_equal_subdivision_ratios(requested_number_of_equal_subdivisions):
#     return [1/requested_number_of_equal_subdivisions for iteration_index in range(requested_number_of_equal_subdivisions)]

# def subdivide_using_precalculated_subset_column(
#         data_table: Table,
#         precalculated_subset_column_name: str,
#         requested_number_of_equal_subdivisions: int,
#         target_group_column_name = standard_group_column_name,
#         target_subset_column_name = standard_subset_column_name
#     ):
    
#     # Conceptual Idea:
#     # A bypass method allowing external custom data splits to be preserved perfectly. It merely points 
#     # the dataset loading structure towards a column that is already statically initialized inside the Table.
#     relative_fold_sizes_sequence = calculate_equal_subdivision_ratios(requested_number_of_equal_subdivisions)
    
#     precalculated_group_column_name = precalculated_subset_column_name
    
#     configured_group_assignment_method = partial(assign_to_groups_using_prexisting_column,
#                                         source_reference_column_name = precalculated_group_column_name,
#                                         target_group_column_name = target_group_column_name
#                                         )
#     identity_subset_assignment_method = lambda input_data_table, absolute_sizes: input_data_table
    
#     subdivided_data_table = split_dataset_into_subsets(
#         data_table,
#         relative_fold_sizes_sequence = relative_fold_sizes_sequence,
#         group_assignment_function_reference = configured_group_assignment_method,
#         subset_assignment_function_reference = identity_subset_assignment_method,
#         target_group_column_name = target_group_column_name,
#         target_subset_column_name = target_subset_column_name
#     )
    
#     return subdivided_data_table

# def subdivide_using_murcko_scaffolds(
#         data_table: Table,
#         requested_number_of_equal_subdivisions: int,
#         target_group_column_name = standard_group_column_name,
#         target_subset_column_name = standard_subset_column_name
#         ):
    
#     relative_fold_sizes_sequence = calculate_equal_subdivision_ratios(requested_number_of_equal_subdivisions)
    
#     configured_group_assignment_method = partial(assign_to_groups_by_scaffold, target_group_column_name=target_group_column_name)
    
#     configured_subset_assignment_method = partial(make_first_fit_group_identity_assigner, source_group_column_name=target_group_column_name, target_subset_column_name=target_subset_column_name)
    
#     subdivided_data_table = split_dataset_into_subsets(
#         data_table,
#         relative_fold_sizes_sequence = relative_fold_sizes_sequence,
#         group_assignment_function_reference = configured_group_assignment_method,
#         subset_assignment_function_reference = configured_subset_assignment_method,
#         target_group_column_name = target_group_column_name,
#         target_subset_column_name = target_subset_column_name
#     )
    
#     return subdivided_data_table

# # --------------------------------------------------------------------------- #
# # Cross Validation Splits Generator.
# # --------------------------------------------------------------------------- #

# def split_dataset_into_test_and_non_test(data_table, target_split_column_name, specified_target_values):
    
#     # Conceptual Idea:
#     # A generic explicit binary filter parsing the dataframe logically to perfectly isolate test validation chunks from general model evaluation rows.
#     test_subset_table = data_table.filter_rowwise_using(lambda row_data: row_data[target_split_column_name] in specified_target_values)
#     non_test_subset_table = data_table.filter_rowwise_using(lambda row_data: row_data[target_split_column_name] not in specified_target_values)
#     return [test_subset_table, non_test_subset_table]

# def cross_validation_splits_generator(
#         data_table: Table,
#         inner_splitting_method_reference, 
#         requested_number_of_inner_splits,
#         outer_splitting_method_reference,
#         requested_number_of_outer_splits,
#         cross_validation_architecture_method: Literal["normal", "fixed_held_out", "nested"],
#         ):
#     """
#     Test dataset status can be:
#         None - no test dataset generated.
#         Single_Split - test dataset is one of the splits. In this case the data
#         is split and the last dataset split is used as the test set.
#         Multi_Split - each split is used as the test set once.
        
#     """
#     # Conceptual Idea:
#     # Complex iterator yielding data tuple views configured structurally to represent multi-tier cross validation processes 
#     # where hyperparameter selection naturally executes across nested holdout chunks.
#     inner_split_group_column_name_string = "inner_split_group"
#     inner_split_subset_column_name_string = "inner_split_subset"
#     outer_split_group_column_name_string = "outer_split_group"
#     outer_split_subset_column_name_string = "outer_split_subset"
    
#     if cross_validation_architecture_method in ("normal",):
        
#         subdivided_data_table = inner_splitting_method_reference(data_table, target_group_column_name=inner_split_group_column_name_string, target_subset_column_name=inner_split_subset_column_name_string)
    
#     elif cross_validation_architecture_method in ("fixed_held_out",):
        
#         subdivided_data_table = outer_splitting_method_reference(data_table, requested_number_of_equal_subdivisions=requested_number_of_outer_splits, target_group_column_name=outer_split_group_column_name_string, target_subset_column_name=outer_split_subset_column_name_string)
#         test_dataset_table, non_test_dataset_table = split_dataset_into_test_and_non_test(subdivided_data_table, outer_split_subset_column_name_string, specified_target_values=[0])
#         inner_subdivided_table = inner_splitting_method_reference(non_test_dataset_table, requested_number_of_equal_subdivisions=requested_number_of_inner_splits, target_group_column_name=inner_split_group_column_name_string, target_subset_column_name=inner_split_subset_column_name_string)
#         numerical_indices = list(range(1, requested_number_of_inner_splits + 1))
#         subset_index_exclusion_sets = [[index_value for index_value in numerical_indices if index_value != excluded_index_value] for excluded_index_value in numerical_indices]
#         # list of [(val_1, train_1), (val_2, train_2), ...(val_n, train_n)]
#         dataset_splits_matrix = [split_dataset_into_test_and_non_test(inner_subdivided_table, inner_split_subset_column_name_string, subset_index_set) + [test_dataset_table] for subset_index_set in subset_index_exclusion_sets]
#         return map(list, zip(*dataset_splits_matrix))
        
#     else:
#         raise NotImplementedError()
        
# =========================================================================== #
# Enamine Split Generator:
# =========================================================================== #

def generate_enamine_train_validation_test_splits(
    data_table: Table,
    target_split_column_name: str = "Split_Using_This_Column",
):
    """
    Return (training_datasets, validation_datasets, test_datasets) as lists of pandas DataFrames.

    If the largest label present is Enamine_Split_N, this returns N+1 cyclic splits.
    "Include_In_All_Splits" rows are always included in training only.
    """
    # Conceptual Idea:
    # Processes strictly mapped text flags produced explicitly inside the Enamine commercial hit-finding system. 
    # Handles continuous testing rotational requirements cleanly while securing unchanging foundational active compounds purely within training boundaries.
    data_table.assert_column_in_table(target_split_column_name)
    materialized_pandas_dataframe = data_table.to_pandas(max_rows=None)
    split_identifiers_column_series = materialized_pandas_dataframe[target_split_column_name].astype(str)

    # 14 == len("Enamine_Split_")
    total_number_of_splits_required = 1 + max(int(string_value[14:]) for string_value in split_identifiers_column_series.unique() if string_value.startswith("Enamine_Split_"))
    include_always_boolean_mask, enamine_prefix_boolean_mask = split_identifiers_column_series == "Include_In_All_Splits", split_identifiers_column_series.str.startswith("Enamine_Split_")
    training_datasets, validation_datasets, test_datasets = [], [], []

    for split_iteration_index in range(total_number_of_splits_required):
        expected_test_split_string_value = f"Enamine_Split_{(total_number_of_splits_required - 1 + split_iteration_index) % total_number_of_splits_required}"
        expected_validation_split_string_value = f"Enamine_Split_{(total_number_of_splits_required - 2 + split_iteration_index) % total_number_of_splits_required}"
        dynamic_training_boolean_mask = include_always_boolean_mask | (enamine_prefix_boolean_mask & ~split_identifiers_column_series.isin([expected_validation_split_string_value, expected_test_split_string_value]))
        training_datasets.append(materialized_pandas_dataframe.loc[dynamic_training_boolean_mask].reset_index(drop=True))
        validation_datasets.append(materialized_pandas_dataframe.loc[split_identifiers_column_series == expected_validation_split_string_value].reset_index(drop=True))
        test_datasets.append(materialized_pandas_dataframe.loc[split_identifiers_column_series == expected_test_split_string_value].reset_index(drop=True))

    return [convert_to_iterable_table(d) for d in training_datasets], [convert_to_iterable_table(d) for d in validation_datasets], [convert_to_iterable_table(d) for d in test_datasets]


def generate_random_cross_validation_splits(
    data_table,
    number_of_folds: int = 10,
    include_test_split: bool = True,
    maximum_total_rows: Optional[int] = None,
    supported_species: Tuple[str, ...] = ("Human", "Rat", "Mouse"),
    species_column_name: str = "Standardized_Species",
    smiles_column_name: str = "SMILES",
    deduplicate_on_columns: Tuple[str, ...] = ("SMILES", "Standardized_Species", "Task"),
    random_seed: int = 42,
):
    """
    Build molecule-grouped cross-validation splits.

    Steps:
      1) Materialize the table to pandas.
      2) Keep only the species the species-one-hot / multi-species PBPK modules support
         (Human / Rat / Mouse) so an unknown label can never KeyError those nodes.
      3) Drop rows with a missing SMILES.
      4) Deduplicate rows that share the same (SMILES, species, task) so identical
         measurements are not repeated within / across folds.
      5) Optionally cap the total number of rows for a fast smoke test.
      6) Shuffle and partition into `number_of_folds` folds. For fold i:
            validation = fold i
            test       = fold (i + 1) % number_of_folds   (if include_test_split, else = validation)
            training   = every row not held out as this fold's validation/test
    Returns (training_tables, validation_tables, test_tables): three equal-length lists of
    IterableTables, exactly like generate_enamine_train_validation_test_splits.
    """
    materialized_dataframe = data_table.to_pandas(max_rows=None)

    # 2) restrict to supported species
    if species_column_name in materialized_dataframe.columns:
        materialized_dataframe = materialized_dataframe[
            materialized_dataframe[species_column_name].isin(list(supported_species))
        ]

    # 3) drop missing SMILES
    if smiles_column_name in materialized_dataframe.columns:
        materialized_dataframe = materialized_dataframe[
            materialized_dataframe[smiles_column_name].notna()
        ]

    # Numeric preprocessing represents censored measurements by their thresholds.
    identifier_columns = {species_column_name, smiles_column_name, "Task"}
    for measurement_column_name in materialized_dataframe.columns:
        if measurement_column_name in identifier_columns:
            continue
        if materialized_dataframe[measurement_column_name].dtype == object:
            materialized_dataframe[measurement_column_name] = materialized_dataframe[measurement_column_name].map(
                lambda cell_value: re.sub(r"[<>=≤≥~]", "", cell_value).strip()
                if isinstance(cell_value, str) else cell_value
            )


    if "Task" in materialized_dataframe.columns:
        materialized_dataframe = materialized_dataframe.drop(columns=["Task"])

    # Merge each molecule/species group using the first non-null value per column.
    deduplicate_columns_present = [c for c in deduplicate_on_columns if c in materialized_dataframe.columns]
    if deduplicate_columns_present:
        rows_before_dedup = len(materialized_dataframe)
        materialized_dataframe = materialized_dataframe.replace(r"^\s*$", np.nan, regex=True)
        materialized_dataframe = (
            materialized_dataframe
            .groupby(deduplicate_columns_present, as_index=False, sort=False)
            .first()
        )
        tempprint(
            f"[random_cv_splitter] collapsed {rows_before_dedup - len(materialized_dataframe)} duplicate "
            f"rows into one-per-{deduplicate_columns_present} (first non-null value per column)."
        )

    materialized_dataframe = materialized_dataframe.reset_index(drop=True)

    # 5) optional row cap for quick tests
    if maximum_total_rows is not None and len(materialized_dataframe) > int(maximum_total_rows):
        materialized_dataframe = materialized_dataframe.sample(
            n=int(maximum_total_rows), random_state=random_seed
        ).reset_index(drop=True)

    total_number_of_rows = len(materialized_dataframe)
    if total_number_of_rows < number_of_folds:
        raise ValueError(
            f"Only {total_number_of_rows} rows remain after filtering/dedup, which is fewer "
            f"than number_of_folds={number_of_folds}."
        )

    tempprint(
        f"[random_cv_splitter] {total_number_of_rows} rows after species-filter / SMILES-drop / dedup; "
        f"building {number_of_folds} folds (include_test_split={include_test_split})."
    )

    # Keep all observations of each molecule in the same fold.
    random_number_generator = np.random.default_rng(random_seed)
    if smiles_column_name in materialized_dataframe.columns:
        row_group_keys = materialized_dataframe[smiles_column_name].to_numpy()
    else:
        row_group_keys = np.arange(total_number_of_rows)
    unique_group_keys = np.unique(row_group_keys)
    shuffled_unique_group_keys = random_number_generator.permutation(unique_group_keys)
    group_key_to_fold_index = {
        group_key: (assignment_position % number_of_folds)
        for assignment_position, group_key in enumerate(shuffled_unique_group_keys)
    }
    fold_index_of_each_row = np.array([group_key_to_fold_index[key] for key in row_group_keys])
    per_fold_index_arrays = [
        np.nonzero(fold_index_of_each_row == fold_number)[0]
        for fold_number in range(number_of_folds)
    ]

    training_datasets, validation_datasets, test_datasets = [], [], []
    for fold_iteration_index in range(number_of_folds):
        validation_indices = per_fold_index_arrays[fold_iteration_index]
        if include_test_split:
            test_indices = per_fold_index_arrays[(fold_iteration_index + 1) % number_of_folds]
        else:
            test_indices = validation_indices

        held_out_mask = np.zeros(total_number_of_rows, dtype=bool)
        held_out_mask[validation_indices] = True
        held_out_mask[test_indices] = True
        training_indices = np.nonzero(~held_out_mask)[0]

        random_number_generator.shuffle(training_indices)

        training_datasets.append(materialized_dataframe.iloc[training_indices].reset_index(drop=True))
        validation_datasets.append(materialized_dataframe.iloc[validation_indices].reset_index(drop=True))
        test_datasets.append(materialized_dataframe.iloc[test_indices].reset_index(drop=True))

    return (
        [convert_to_iterable_table(d) for d in training_datasets],
        [convert_to_iterable_table(d) for d in validation_datasets],
        [convert_to_iterable_table(d) for d in test_datasets],
    )


# =========================================================================== #
# Dataset Collate Methods:
# =========================================================================== #

@register("collate", "graph")
def collate_to_graph(input_elements):
    # Conceptual Idea:
    # Maps isolated discrete PyTorch Geometric Data structures into a robust multi-molecule Batch representation.
    return GeometricBatch.from_data_list(input_elements)


@register("collate", "numerical")
def collate_to_numerical(input_elements):
    # Conceptual Idea:
    # Enforces standard 32-bit floating point hardware arrays for heterogeneous model target regression labels safely.
    if all(isinstance(element_value, Number) for element_value in input_elements):
        return torch.tensor(input_elements, dtype=torch.float32)                 # shape (Batch_Size,)
    if all(isinstance(element_value, torch.Tensor) for element_value in input_elements):
        stacked_tensor_result = torch.stack([element_value.view(-1) for element_value in input_elements], dim=0).squeeze(-1)
        return stacked_tensor_result.to(torch.float32)
    raise NotImplementedError("to_numerical expects Numbers or Tensors.")
    
@register("collate", "integer")
def collate_to_integer(input_elements):
    # Conceptual Idea:
    # Converts arrays holding class indices or numerical counts directly into PyTorch's required long integers 
    # to naturally avoid destructive downstream parameter cast faults during execution.
    try:
        if all(isinstance(element_value, Number) for element_value in input_elements):
            return torch.tensor(input_elements, dtype=torch.long)  # (Batch_Size,)
        if all(isinstance(element_value, torch.Tensor) for element_value in input_elements):
            stacked_tensor_result = torch.stack([element_value.view(-1) for element_value in input_elements], dim=0).squeeze(-1)
            return stacked_tensor_result.to(torch.long)
        # Mixed types? Coerce elementwise through safely cast method (last resort):
        coerced_integer_elements = [to_integer(element_value) for element_value in input_elements]
        return torch.tensor([0 if coerced_value is None else int(coerced_value) for coerced_value in coerced_integer_elements], dtype=torch.long)
    except Exception as tensorization_error:
        print("[collate_to_integer] tensorization failed; first 32 items:")
        for index_value, element_value in enumerate(input_elements[:32]):
            print(f"  idx {index_value}: {repr(element_value)}  type={type(element_value).__name__}")
        print(f"[collate_to_integer] error: {type(tensorization_error).__name__}: {tensorization_error}")
        raise

@register("collate", "tensor")
def collate_to_tensor(input_elements):
    return torch.stack(input_elements, dim=0)

@register("collate", "string")
def collate_strings(list_of_values):
    # Conceptual Idea: collate a column of strings into a plain Python list, NOT a stacked tensor.
    # The model receives the list and hands it to the partition module, which featurizes per batch.
    return list(list_of_values)

# =========================================================================== #
# Dataset Preprocessing Methods:
# =========================================================================== #

@register("preprocessing", "graph")
def preprocess_to_graph(input_object):
    return smiles_to_data(input_object)    

@register("preprocessing", "integer")
def preprocess_to_integer(input_object):
    return to_integer(input_object)

@register("preprocessing", "numerical")
def preprocess_to_numerical(input_object):
    return to_numerical(input_object)

@register("preprocessing", "tensor")
def preprocess_to_tensor(input_object):
    # Conceptual Idea:
    # Reverse string encapsulation mappings for array lists perfectly safely directly loading them back into PyTorch matrices.
    if isinstance(input_object, torch.Tensor):
        return input_object
    if isinstance(input_object, str):
        try:
            evaluated_python_object = literal_eval(input_object)            # handles "[...]", "(...)", etc.
            parsed_embedding_tensor = torch.as_tensor(evaluated_python_object, dtype=torch.float32)
            # A few MiniMol rows can carry NaN/Inf; one non-finite input poisons the whole batch.
            return torch.nan_to_num(parsed_embedding_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as evaluation_error:
            print(f"Couldn't process this object {input_object[:100]}")
            print(f"Got this error {evaluation_error} when trying to process this object {input_object}")
            return None
        
@register("preprocessing", "string")
def to_string(raw_value):
    # Conceptual Idea: pass-through for text columns such as SMILES. No tensorization; the raw
    # string is carried so a downstream module (the partition-coefficient calculator) can parse it.
    # is_missing_value already treats None / empty strings as missing, so no extra guard is needed here.
    if raw_value is None:
        return None
    return str(raw_value)

# ########################################################################### # 
# 10. Hyperparameter Tuning Methods: 
# ########################################################################### #

# =========================================================================== #
# Hyperparameter-Related Imports:
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Search Imports:
# --------------------------------------------------------------------------- #

from ray.tune.search.basic_variant import BasicVariantGenerator as RayTuneBasicVariantGenerator
from ray.tune.search.optuna import OptunaSearch as RayTuneOptunaSearch
from ray.tune.search.hyperopt import HyperOptSearch as RayTuneHyperOptSearch
from ray.tune.search.bayesopt import BayesOptSearch as RayTuneBayesOptSearch
from ray.tune.search.nevergrad import NevergradSearch as RayTuneNevergradSearch
from ray.tune.search.bohb import TuneBOHB as RayTuneBOHBSearch
RayTuneSkOptSearch = None
from ray.tune.search.ax import AxSearch as RayTuneAxSearch
from ray.tune.search.hebo import HEBOSearch as RayTuneHEBOSearch

# --------------------------------------------------------------------------- #
# Scheduling Imports:
# --------------------------------------------------------------------------- #

from ray.tune.schedulers import FIFOScheduler as RayTuneFIFOScheduler
from ray.tune.schedulers import ASHAScheduler as RayTuneASHAScheduler
from ray.tune.schedulers import HyperBandScheduler as RayTuneHyperBandScheduler
from ray.tune.schedulers import MedianStoppingRule as RayTuneMedianStoppingRule
from ray.tune.schedulers import PopulationBasedTraining as RayTunePopulationBasedTraining
from ray.tune.schedulers.pb2 import PB2 as RayTunePB2Scheduler
from ray.tune.schedulers import HyperBandForBOHB as RayTuneHyperBandForBOHBScheduler

# =========================================================================== #
# Search Space Methods:
# =========================================================================== #

# Conceptual Idea:
# Defines mapping wrappers tying powerful Ray Tune search algorithm frameworks (e.g. Optuna, HyperOpt, Bayesian Optimization) 
# seamlessly directly into the local architecture registry via straightforward lookup dictionaries.

@register("search", "basic")
def construct_basic_variant_generator_search(*positional_arguments, **keyword_arguments):
    return RayTuneBasicVariantGenerator(*positional_arguments, **keyword_arguments)

@register("search", "optuna")
def construct_optuna_search(*positional_arguments, **keyword_arguments):
    return RayTuneOptunaSearch(*positional_arguments, **keyword_arguments)

@register("search", "hyper_opt")
def construct_hyper_opt_search(*positional_arguments, **keyword_arguments):
    return RayTuneHyperOptSearch(*positional_arguments, **keyword_arguments)

@register("search", "bayes_opt")
def construct_bayes_opt_search(*positional_arguments, **keyword_arguments):
    return RayTuneBayesOptSearch(*positional_arguments, **keyword_arguments)

@register("search", "nevergrad")
def construct_nevergrad_search(*positional_arguments, **keyword_arguments):
    return RayTuneNevergradSearch(*positional_arguments, **keyword_arguments)

@register("search", "bohb")
def construct_bohb_search(*positional_arguments, **keyword_arguments):
    return RayTuneBOHBSearch(*positional_arguments, **keyword_arguments)

@register("search", "skopt")
def construct_skopt_search(*positional_arguments, **keyword_arguments):
    # SkOpt was removed from Ray Tune, so RayTuneSkOptSearch is None. Fail with a clear message
    # rather than the opaque "'NoneType' object is not callable" if this option is ever selected.
    if RayTuneSkOptSearch is None:
        raise NotImplementedError("The 'skopt' search algorithm is no longer available in this Ray version.")
    return RayTuneSkOptSearch(*positional_arguments, **keyword_arguments)

@register("search", "ax")
def construct_ax_search(*positional_arguments, **keyword_arguments):
    return RayTuneAxSearch(*positional_arguments, **keyword_arguments)

@register("search", "hebo")
def construct_hebo_search(*positional_arguments, **keyword_arguments):
    return RayTuneHEBOSearch(*positional_arguments, **keyword_arguments)

# =========================================================================== #
# Scheduler Methods:
# =========================================================================== #

# Conceptual Idea:
# Standardizes integration bindings specifically bridging dynamic trial resource scheduling controls 
# (e.g., ASHA early stopping or Population Based Training sweeps) smoothly into the project's native string registration maps.

@register("scheduler", "fifo")
def construct_fifo_scheduler(*positional_arguments, **keyword_arguments):
    return RayTuneFIFOScheduler(*positional_arguments, **keyword_arguments)

@register("scheduler", "asha")
def construct_asha_scheduler(*positional_arguments, **keyword_arguments):
    return RayTuneASHAScheduler(*positional_arguments, **keyword_arguments)

@register("scheduler", "hyperband")
def construct_hyperband_scheduler(*positional_arguments, **keyword_arguments):
    return RayTuneHyperBandScheduler(*positional_arguments, **keyword_arguments)

@register("scheduler", "median_stopping")
def construct_median_stopping_rule_scheduler(*positional_arguments, **keyword_arguments):
    return RayTuneMedianStoppingRule(*positional_arguments, **keyword_arguments)

@register("scheduler", "population_based")
def construct_population_based_training_scheduler(*positional_arguments, **keyword_arguments):
    return RayTunePopulationBasedTraining(*positional_arguments, **keyword_arguments)

@register("scheduler", "pb2")
def construct_pb2_scheduler(*positional_arguments, **keyword_arguments):
    return RayTunePB2Scheduler(*positional_arguments, **keyword_arguments)

@register("scheduler", "hyperband_for_bohb")
def construct_hyperband_for_bohb_scheduler(*positional_arguments, **keyword_arguments):
    return RayTuneHyperBandForBOHBScheduler(*positional_arguments, **keyword_arguments)
    
# ########################################################################### #
# 11. Pipeline Execution:
# ########################################################################### #

# =========================================================================== #
# Example Config:
# =========================================================================== #

'''
interpreter_is: model_training

write_results_to: "/path/to/results.csv"

dataset:
  load_dataset_at: "/path/to/dataset.csv"
  inputs:
    - named_as: graph_input
      from_column: Standardized_SMILES
      object_type: graph
      filter_failed: True

    - named_as: minimol_fp
      from_column: Minimol_Representation
      object_type: tensor
      filter_failed: False

  tasks:
    - {named_as: EC_Hit, from_column: Escherichia_coli_Hit, object_type: integer}
    - {named_as: AB_Hit, from_column: Acinetobacter_baumannii_Hit, object_type: integer}
    - {named_as: KP_Hit, from_column: Klebsiella_pneumoniae_Hit, object_type: integer}
    - {named_as: PA_Hit, from_column: Pseudomonas_aeruginosa_Hit, object_type: integer}
    - {named_as: Enterobacteraceae_Hit, from_column: Enterobacteraceae_Hit, object_type: integer}
    - {named_as: Broad_Spectrum_Hit, from_column: Broad_Spectrum_Hit, object_type: integer}

split_data_using:
  type: enamine 

# Repetitive tasks nicely collapsed into arrays
loss:
  - type: cross_entropy
    task: [EC_Hit, AB_Hit, KP_Hit, PA_Hit]
    weight: 1.0 

  - type: l1_penalty
    weight: ${l1_norm_weight}

  - type: l2_penalty
    weight: ${l2_norm_weight}

variable_bindings:
  gnn_hidden_dim: log_random_int(16, 1024)
  concat_layer_dim: gnn_hidden_dim + 512
  gnn_propagation_steps: random_int(2, 4)

  num_shared_layers: random_int(1, 4)
  shared_layer_width: choice([16, 32, 64, 128, 256, 512, 1024, 2048])
  hidden_layer_sizes: "[shared_layer_width] * num_shared_layers"
  dropout_prob: uniform(0, 1)

  l1_norm_weight: log_uniform(1e-12, 1e-2)
  l2_norm_weight: log_uniform(1e-8, 1e-2)
  learning_rate: log_uniform(1e-6, 1e-1)
  amount_of_patience: random_int(1, 4)

modules:
  - name: ChemPropGNN
    type: chemprop
    inputs: [{from: graph_input/out}]
    params:
      edge_feature_dimension: ${gnn_hidden_dim}
      depth: ${gnn_propagation_steps}

  - name: concat_layer
    type: concat_tensors
    inputs:
      - {from: ChemPropGNN/out}
      - {from: minimol_fp/out}

  - name: shared_parameter_mixer
    type: ffn
    inputs: [{from: concat_layer/out}]
    params:
      input_dimension: ${concat_layer_dim}
      hidden_dimensions: ${hidden_layer_sizes}
      dropout_probabilities: ${dropout_prob}

  # Replace unregistered `classification_head` with registered `ffn` modules
  - name: EC_Classification_Head
    type: ffn
    inputs: [{from: shared_parameter_mixer/out}]
    params:
      input_dimension: ${shared_layer_width}
      hidden_dimensions: [2]
    predictions:
      - for_task: EC_Hit

  - name: AB_Classification_Head
    type: ffn
    inputs: [{from: shared_parameter_mixer/out}]
    params:
      input_dimension: ${shared_layer_width}
      hidden_dimensions: [2]
    predictions:
      - for_task: AB_Hit

  - name: KP_Classification_Head
    type: ffn
    inputs: [{from: shared_parameter_mixer/out}]
    params:
      input_dimension: ${shared_layer_width}
      hidden_dimensions: [2]
    predictions:
      - for_task: KP_Hit

  - name: PA_Classification_Head
    type: ffn
    inputs: [{from: shared_parameter_mixer/out}]
    params:
      input_dimension: ${shared_layer_width}
      hidden_dimensions: [2]
    predictions:
      - for_task: PA_Hit

  - name: Enterobacteraceae_Score
    type: elementwise_ordered_choice
    inputs:
      - {from: EC_Classification_Head/out}
      - {from: KP_Classification_Head/out}
    params:
      selection_rank_position: 1  # Corrected kwarg mapping
    predictions:
      - for_task: Enterobacteraceae_Hit

  - name: Broad_Spectrum_Score
    type: elementwise_ordered_choice
    inputs:
      - {from: PA_Classification_Head/out}
      - {from: AB_Classification_Head/out}
      - {from: EC_Classification_Head/out}
    params:
      selection_rank_position: 2  # Corrected kwarg mapping
    predictions:
      - for_task: Broad_Spectrum_Hit

optimizer:
  type: adamw
  kwargs:
    learning_rate: ${learning_rate}

weighting:
  name: pcgrad
  kwargs: {}

early_stopping:
  monitor: validation_total 
  mode: min
  patience: ${amount_of_patience}
  min_delta: 0.0

# Tasks properly grouped into list variables
metrics:
  - type: accuracy
    task: [EC_Hit, AB_Hit, KP_Hit, PA_Hit, Enterobacteraceae_Hit, Broad_Spectrum_Hit]
  - type: auroc
    task: [EC_Hit, AB_Hit, KP_Hit, PA_Hit, Enterobacteraceae_Hit, Broad_Spectrum_Hit]
  - type: auprc
    task: [EC_Hit, AB_Hit, KP_Hit, PA_Hit, Enterobacteraceae_Hit, Broad_Spectrum_Hit]

resources:
  cpu: 3
  gpu: 0

optimization:
  optimization_criterion:
    metric: validation_auprc_EC_Hit
    mode: max
    num_samples: 125
    max_concurrent_trials: 13

  optimization_scheduler_method:
    type: asha
    time_attr: training_iteration
    grace_period: 2
    reduction_factor: 2

  optimization_search_method:
    type: optuna

run_configuration:
  name: chemprop_hit_hpo_demo
  stop: {training_iteration: 30}
  verbose: 2
  storage_path: "/path/to/storage"
  log_to_file:
    - stdout.log
    - stderr.log

checkpoint_configuration:
  num_to_keep: 1
'''

# =========================================================================== #
# Syntax for config file:
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Syntax for Main Config Keys:
# --------------------------------------------------------------------------- #

config_syntax_for_recursive_config_execution = "execute_configs_at"
config_syntax_for_the_config_interpreter = "interpreter_is"

config_syntax_for_variable_expressions = "variable_bindings"
config_syntax_for_variable_search_space = "variable_search_space"

config_syntax_for_weight_optimization = "optimizer"

config_syntax_for_dataset_annotation = "dataset"
config_syntax_for_dataset_input_annotation = "inputs"
config_syntax_for_dataset_task_annotation = "tasks"

config_syntax_for_dataset_spliting_method = "split_data_using"
config_syntax_for_the_loss_function = "loss"
config_syntax_for_model_modules = "modules"
config_syntax_for_task_specific_weighting = "weighting"
config_syntax_for_early_stopping_criterion = "early_stopping"
config_syntax_for_model_performance_metrics = "metrics"
config_syntax_for_resources = "resources"

config_syntax_for_optimization = "optimization"
config_syntax_for_optimization_criteria = "optimization_criterion"
config_syntax_for_optimization_scheduler_method = "optimization_scheduler_method"
config_syntax_for_optimization_search_method = "optimization_search_method"
config_syntax_for_run_configuration = "run_configuration"
config_syntax_for_checkpoint_configuration = "checkpoint_configuration"

# --------------------------------------------------------------------------- #
# Syntax for Auxiliary Config Keys:
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Syntax for Different Interpreters:
# --------------------------------------------------------------------------- #

config_syntax_for_model_training_interpreter = "model_training"
config_syntax_for_fine_tuning_interpreter = "fine_tuning"
config_syntax_for_dataset_scoring_interpreter = "dataset_scoring"

# --------------------------------------------------------------------------- #
# Syntax for Secondary Config Keys:
# --------------------------------------------------------------------------- #

config_syntax_for_type_of_method = "type"

# --------------------------------------------------------------------------- #
# Commonly-Used Groups of Variables:
# --------------------------------------------------------------------------- #

# Note: config_syntax_for_recursive_config_execution ("execute_configs_at") is intentionally NOT
# listed here. It is optional and only appears in parent configs that point at other config files,
# and collect_configs_to_execute consumes it before require_keys runs. Listing it as a mandatory
# base field would make require_keys reject every leaf config (and the extraction loop KeyError on it).
base_config_syntax_variables = [
    config_syntax_for_the_config_interpreter,
]

training_dataset_syntax_variables = [
    config_syntax_for_dataset_annotation,
    config_syntax_for_dataset_spliting_method
]

model_training_syntax_variables = [
    config_syntax_for_weight_optimization,
    config_syntax_for_the_loss_function,
    config_syntax_for_model_modules,
    config_syntax_for_task_specific_weighting,
    config_syntax_for_early_stopping_criterion,
    config_syntax_for_model_performance_metrics,
    config_syntax_for_resources,
    config_syntax_for_optimization,
    config_syntax_for_run_configuration,
    config_syntax_for_checkpoint_configuration,
]


all_config_syntax_variables = base_config_syntax_variables + training_dataset_syntax_variables + model_training_syntax_variables

# =========================================================================== #
# Mapping Interpreters to Expected Fields:
# =========================================================================== #

interpreter_name_to_expected_fields = {
    config_syntax_for_model_training_interpreter:
        base_config_syntax_variables +
        model_training_syntax_variables +
        training_dataset_syntax_variables,
    config_syntax_for_fine_tuning_interpreter:
        base_config_syntax_variables +
        model_training_syntax_variables +
        training_dataset_syntax_variables,
    config_syntax_for_dataset_scoring_interpreter:
        []
    }
    
def get_expected_fields_for_an_interpreter(interpreter_name: str):
    return interpreter_name_to_expected_fields[interpreter_name]

# =========================================================================== #
# Extracting Sub-Configs:
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Methods for Extracting/Processing Subconfigs:
# --------------------------------------------------------------------------- #

def extract_optimization_config(config):
    '''
    Config organization:
    ...
    optimization:
        
        optimization_criterion:
            metric: val_auprc_EC_Hit
            mode: max
            num_samples: 125
            max_concurrent_trials: 13
        
        optimization_scheduler_method:
            type: asha
            time_attr: training_iteration
            grace_period: 2
            reduction_factor: 2
  
        optimization_search_method:
          type: optuna
    ...
    '''

    configuration_subconfig = config[config_syntax_for_optimization]

    optimization_configuration = configuration_subconfig[config_syntax_for_optimization_criteria]
    scheduler_configuration = configuration_subconfig[config_syntax_for_optimization_scheduler_method]
    search_method_configuration = configuration_subconfig[config_syntax_for_optimization_search_method]

    scheduler_class = scheduler_configuration[config_syntax_for_type_of_method]
    scheduler_keyword_arguments = {
        key: value
        for key, value in scheduler_configuration.items()
        if key not in (config_syntax_for_type_of_method,)
    }
    scheduler = retrieve_scheduler_component(scheduler_class)(**scheduler_keyword_arguments)

    search_method_class = search_method_configuration[config_syntax_for_type_of_method]
    search_method_keyword_arguments = {
        key: value
        for key, value in search_method_configuration.items()
        if key not in (config_syntax_for_type_of_method,)
    }
    search_method = retrieve_search_component(search_method_class)(**search_method_keyword_arguments)

    optimization_configuration["scheduler"] = scheduler
    optimization_configuration["search_alg"] = search_method
    return TuneConfig(**optimization_configuration)

def extract_run_config(configuration):
    
    '''
    Config organization:
    ...
    run_configuration:
      name: chemprop_hit_hpo_dem
      stop: {training_iteration: 30}
      verbose: 2
      storage_path: /Users/asselism/Desktop/Collins_Lab/Models/Model_Building/Organized_by_Project/Active_Learning/Active_Learnign_Round_2_Model_Development/Hybrid_Model/Temp_Storage/ 
      # capture per-trial stdout/stderr to files in each trial directory
      log_to_file:
        - stdout.log
        - stderr.log
    ...
    '''
    
    run_configuration = configuration[config_syntax_for_run_configuration].copy()
    checkpoint_keyword_arguments = (configuration.get(config_syntax_for_checkpoint_configuration) or {}).copy()

    # Derive defaults for checkpoint scoring from early_stopping or optimization_criteria
    optimization_configuration = configuration.get(config_syntax_for_optimization) or {}
    optimization_criteria_configuration = optimization_configuration.get(config_syntax_for_optimization_criteria) or {}
    early_stopping_configuration = configuration.get(config_syntax_for_early_stopping_criterion) or {}
    monitor_key = early_stopping_configuration.get("monitor", optimization_criteria_configuration.get("metric", "validation_total"))
    mode = str(early_stopping_configuration.get("mode", optimization_criteria_configuration.get("mode", "min"))).lower()
    monitor_key = "validation_total" if monitor_key == "val_loss" else monitor_key
    monitor_key = "validation_" + monitor_key[4:] if monitor_key.startswith("val_") else monitor_key

    # Keep only the *best* checkpoint over time (by instantaneous metric)
    checkpoint_keyword_arguments.setdefault("num_to_keep", 1)
    checkpoint_keyword_arguments.setdefault("checkpoint_score_attribute", monitor_key)
    checkpoint_keyword_arguments.setdefault("checkpoint_score_order", "max" if mode == "max" else "min")

    checkpoint_config = CheckpointConfig(**checkpoint_keyword_arguments)
    run_configuration["checkpoint_config"] = checkpoint_config

    # Keep your sync behavior
    run_configuration["sync_config"] = tune.SyncConfig(sync_artifacts=True, sync_artifacts_on_checkpoint=True)
    return RunConfig(**run_configuration)

config_syntax_for_dataset_input_columns_initial_name = "from_column"
config_syntax_for_dataset_input_columns_modified_name = "named_as"

def extract_dataset_config(configuration):

    path_to_dataset = configuration[config_syntax_for_dataset_annotation]["load_dataset_at"]
    inputs_configuration = configuration[config_syntax_for_dataset_annotation][config_syntax_for_dataset_input_annotation]
    tasks_configuration = configuration[config_syntax_for_dataset_annotation][config_syntax_for_dataset_task_annotation]

    dataset_configuration = {
        "path_to_dataset": path_to_dataset,
        "source_column_name_to_final_column_name": {},
        "final_column_name_to_source_column_name": {},
        "object_type": {},
        "filter_failed": {},
        "in_names": [],
        "final_column_names": [],
        "input_column_names": [],
        "task_column_names": [],
    }

    # Record all output fields (inputs + tasks) so the collate can materialize them.
    for field_specification in [*inputs_configuration, *tasks_configuration]:
        source_column_name = field_specification[config_syntax_for_dataset_input_columns_initial_name]
        output_field_name = field_specification[config_syntax_for_dataset_input_columns_modified_name]
        dataset_configuration["source_column_name_to_final_column_name"][source_column_name] = output_field_name
        dataset_configuration["final_column_name_to_source_column_name"][output_field_name] = source_column_name
        dataset_configuration["object_type"][output_field_name] = field_specification["object_type"]
        dataset_configuration["filter_failed"][output_field_name] = field_specification.get("filter_failed", DEFAULT_FILTER_FAILED)
        dataset_configuration["final_column_names"].append(output_field_name)

    # NEW: split by role for mask construction & validation.
    dataset_configuration["input_column_names"] = [input_field[config_syntax_for_dataset_input_columns_modified_name] for input_field in inputs_configuration]
    dataset_configuration["task_column_names"] = [task_field[config_syntax_for_dataset_input_columns_modified_name] for task_field in tasks_configuration]

    return dataset_configuration

def extract_the_loss_function_config(config: Mapping[str, Any]) -> Any:
    return config[config_syntax_for_the_loss_function]


def extract_model_modules_config(config: Mapping[str, Any]) -> Any:
    return config[config_syntax_for_model_modules]


def extract_task_specific_weighting_config(config: Mapping[str, Any]) -> Any:
    return config[config_syntax_for_task_specific_weighting]


def extract_early_stopping_criterion_config(config: Mapping[str, Any]) -> Any:
    return config[config_syntax_for_early_stopping_criterion]


def extract_model_performance_metrics_config(config: Mapping[str, Any]) -> Any:
    return config[config_syntax_for_model_performance_metrics]


def extract_resources_config(config: Mapping[str, Any]) -> Any:
    return config[config_syntax_for_resources]


def extract_optimization_criteria_config(config: Mapping[str, Any]) -> Any:
    return config[config_syntax_for_optimization_criteria]


def extract_optimization_scheduler_method_config(config: Mapping[str, Any]) -> Any:
    return config[config_syntax_for_optimization_scheduler_method]


def extract_optimization_search_method_config(config: Mapping[str, Any]) -> Any:
    return config[config_syntax_for_optimization_search_method]


def extract_run_configuration_config(config: Mapping[str, Any]) -> Any:
    return config[config_syntax_for_run_configuration]


def extract_checkpoint_configuration_config(config: Mapping[str, Any]) -> Any:
    return config[config_syntax_for_checkpoint_configuration]


# --------------------------------------------------------------------------- #
# Mapping Interpreters to the Relevant Extraction Method:
# --------------------------------------------------------------------------- #

field_name_to_extraction_method = {
    config_syntax_for_recursive_config_execution: None,
    config_syntax_for_the_config_interpreter: None,
    config_syntax_for_variable_expressions: None,
    config_syntax_for_dataset_annotation: extract_dataset_config,
    config_syntax_for_dataset_spliting_method: None,
    config_syntax_for_the_loss_function: extract_the_loss_function_config,
    config_syntax_for_model_modules: extract_model_modules_config,
    config_syntax_for_task_specific_weighting: extract_task_specific_weighting_config,
    config_syntax_for_early_stopping_criterion: extract_early_stopping_criterion_config,
    config_syntax_for_model_performance_metrics: extract_model_performance_metrics_config,
    config_syntax_for_resources: extract_resources_config,
    config_syntax_for_optimization_criteria: extract_optimization_config,
    config_syntax_for_optimization_scheduler_method: extract_optimization_scheduler_method_config,
    config_syntax_for_optimization_search_method: extract_optimization_search_method_config,
    config_syntax_for_run_configuration: extract_run_config,
    config_syntax_for_checkpoint_configuration: extract_checkpoint_configuration_config,
    config_syntax_for_weight_optimization: None,
    config_syntax_for_optimization: extract_optimization_config
}

def get_extraction_method_from_field_name(field_name):
    identity = lambda subconfig, field_name=field_name: subconfig[field_name]
    method = field_name_to_extraction_method[field_name]
    if method is None:
        return identity
    else:
        return method

# =========================================================================== #
# Pipeline Execution Helper Methods:
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Collate from Config:
# --------------------------------------------------------------------------- #

def is_missing_value(x) -> bool:
    """Return True if x should be treated as missing/absent."""
    # None
    if x is None:
        return True

    # Torch tensor
    if isinstance(x, torch.Tensor):
        if x.numel() == 0:
            return True
        if x.dtype.is_floating_point and torch.isnan(x).any():
            return True
        return False

    if isinstance(x, np.ndarray):
        if x.size == 0:
            return True
        if np.issubdtype(x.dtype, np.floating) and np.isnan(x).any():
            return True
        return False
    if isinstance(x, np.floating):
        return np.isnan(x) or np.isinf(x)

    # Python floats
    if isinstance(x, float):
        return math.isnan(x) or math.isinf(x)

    # Strings
    if isinstance(x, str):
        s = x.strip().lower()
        if s == "" or s in {"nan", "na", "null", "none"}:
            return True
        return False

    return False

def get_weighting_column_name_from_value_column_name(column_name):
    return "weights"
    # return column_name + "_weights"

def get_placeholder_for_object_type(object_type):
    
    if object_type in ("numerical", "integer"):
        return 0
    elif object_type in ("tensor", ):
        return torch.zeros(512)
    elif object_type in ("graph", ):
        return smiles_to_data("C")
    elif object_type == "string":
        return ""   # empty string; is_missing_value flags it so the row is masked like any other missing input
    else:
        raise NotImplementedError()

syntax_for_metric_weighting_column_name = "Metric_Weighting"

def collate_from_config(batch: List[Dict], dataset_config):
    """
    Collate a list of row dicts into a batch dict, *always* emitting a boolean
    mask per task: '<task>__mask' with shape (B,). Inputs remain strict (missing
    inputs raise); task targets may be missing and will be masked out.

    This function:
      1) gathers per-output lists from rows,
      2) for task outputs, builds per-row boolean masks and fills placeholders
         for missing entries (so collation never fails),
      3) applies the registered collate function for each output,
      4) appends '<task>__mask' tensors to the result.
    """

    final_column_names   = dataset_config["final_column_names"]
    object_types   = dataset_config["object_type"]

    # 1) gather per-output lists from rows
    collated_batch: Dict[str, list] = {column_name: {"values": [], "mask": [], "weighting": []} for column_name in final_column_names}
    
    default_metric_weight = 1
    collated_batch[syntax_for_metric_weighting_column_name] = torch.tensor([row.get(syntax_for_metric_weighting_column_name, default_metric_weight) for row in batch])
    
    for value_column_name in final_column_names:
        column_information = collated_batch[value_column_name]
        object_type = object_types[value_column_name]
        weighting_column_name = get_weighting_column_name_from_value_column_name(value_column_name)
        
        for row in batch:
            default_weighting = 1
            weight = row.get(weighting_column_name, default_weighting)
            source_col = dataset_config["final_column_name_to_source_column_name"][value_column_name]
            value = row.get(source_col)
            value_is_present = not is_missing_value(value)
            
            # These are not broken up into an if-else statement because the 
            # preprocessing could fail, in which case the second if statement
            # would also be used.
            if value_is_present:
                # Pre-processing inputs.
                preprocess_function = retrieve_preprocessing_component(object_type)
                try:
                    processed_value = preprocess_function(value)
                    if processed_value is None:
                        value_is_present = False
                    else:
                        column_information["values"].append(processed_value)
                        column_information["mask"].append(True)
                except Exception:
                    value_is_present = False
                    
            if not value_is_present:
                column_information["mask"].append(False)
                placeholder = get_placeholder_for_object_type(object_type)
                column_information["values"].append(placeholder)
                
            column_information["weighting"].append(weight)
        
        for column_name in ("mask", "weighting"):
            column_information[column_name] = torch.tensor(column_information[column_name])
            
        collate_function = retrieve_collate_component(object_type)
        column_information["values"] = collate_function(column_information["values"])
    
    return collated_batch

# --------------------------------------------------------------------------- #
# Constructing Tuner:
# --------------------------------------------------------------------------- #

def construct_tuner(
                training_function: Callable,
                resource_config: Dict,
                search_space_config: Dict,
                optimisation_config: Dict,
                run_config: Dict,
                training_datasets: DataLoader,
                validation_datasets: DataLoader,
                test_datasets: DataLoader,
                dataloader_kwargs
                   ):
        
    training_function = tune.with_parameters(
                                        training_function, 
                                        training_datasets=training_datasets, 
                                        validation_datasets=validation_datasets, 
                                        test_datasets=test_datasets,
                                        dataloader_kwargs=dataloader_kwargs
                                        )
    
    tuner = tune.Tuner(
        tune.with_resources(training_function, resource_config),
        param_space=search_space_config,
        tune_config=optimisation_config,
        run_config=run_config
    )
    
    return tuner

# --------------------------------------------------------------------------- #
# Recursive config extraction:
# --------------------------------------------------------------------------- #

def config_does_recursive_execution(config):
    return config_syntax_for_recursive_config_execution in config

def get_config_paths_to_recursively_execute(config):
    config_paths = config[config_syntax_for_recursive_config_execution]
    return config_paths

def collect_configs_to_execute(config_path):
    
    # 1) Load the config.
    config = load_yaml(config_path)
    
    # 2) Compile all of the configs that need to be executed.
    if config_does_recursive_execution(config):
        config_paths = get_config_paths_to_recursively_execute(config)
        configs = flatten([collect_configs_to_execute(config_path) for config_path in config_paths])
        assert len(configs) >= 1, "At least one config must be provided when recursively executing configs."
    else:
        configs = [config]
    
    # 3) Return the config list.
    return configs

# --------------------------------------------------------------------------- #
# Other helpers:
# --------------------------------------------------------------------------- #

def get_interpreter_name(config):
    return config[config_syntax_for_the_config_interpreter]

def require_keys(mapping: Mapping[Hashable, Any], required_keys: Iterable[Hashable]) -> None:
    """Raise ValueError listing any required keys that are missing from the mapping."""
    missing_keys = [key for key in required_keys if key not in mapping]
    if missing_keys:
        raise ValueError(f"Missing required keys: {missing_keys}")

def expand_globs(text, recurse=True):
    return glob.glob(text, recurse=recurse)

# =========================================================================== #
# Load yaml
# =========================================================================== #

def load_yaml(path):
    
    # Load a yaml file from its file location.
    with Path(path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)


    return config

# =========================================================================== #
# Compile Variable Bindings:
# =========================================================================== #

def compile_variable_bindings(variable_expressions: Mapping[str, Any]) -> dict[str, Any]:
    """
    Compiles the ordered mapping at `config_syntax_for_variable_binding` into Ray Tune
    `tune.sample_from(...)` objects intended to be placed under `config[config_syntax_for_variable_search_space]`.

    - Configuration expressions are trusted Python expressions evaluated with `eval()`.
    - `${name}` is treated as a reference to a previously-defined variable `name`.
    - Variables are assumed to live under `specification.config[config_syntax_for_variable_search_space]`.
    - Every variable is represented via `tune.sample_from(...)` (even constants).
    """
    placeholder_pattern = regular_expression.compile(r"\$\{([A-Za-z_]\w*)\}")

    def log_uniform(lower: float, upper: float) -> float:
        return math.exp(random.uniform(math.log(lower), math.log(upper)))

    def random_int(lower: Any, upper: Any) -> int:
        return random.randrange(int(lower), int(upper))

    def log_random_int(lower: Any, upper: Any) -> int:
        lower_integer, upper_integer = int(lower), int(upper)
        sample_value = int(math.exp(random.uniform(math.log(lower_integer), math.log(upper_integer))))
        return max(lower_integer, min(sample_value, upper_integer - 1))

    sampling_functions = {
        "uniform": random.uniform,
        "log_uniform": log_uniform,
        "random_int": random_int,
        "log_random_int": log_random_int,
        "choice": random.choice,
    }
    evaluation_globals = {"__builtins__": __builtins__, "math": math, "random": random}
    compiled_variables: dict[str, Any] = {}
    prior_variable_names: list[str] = []

    for variable_name, expression_value in variable_expressions.items():
        expression_source = placeholder_pattern.sub(r"\1", str(expression_value))

        def expression_function(
            specification,
            expression_source=expression_source,
            referenced_variable_names=tuple(prior_variable_names),
        ):
            variables_namespace = (
                specification.config.get(config_syntax_for_variable_search_space)
                or specification.config.get("variables", {})
            )
            local_namespace = {name: variables_namespace[name] for name in referenced_variable_names} | sampling_functions
            return eval(expression_source, evaluation_globals, local_namespace)

        compiled_variables[variable_name] = tune.sample_from(expression_function)
        prior_variable_names.append(variable_name)

    return compiled_variables

# def substitute_variables_throughout_config(config):
#     """
#     Replaces ${var} placeholders throughout `config` with references to
#     `spec.config.variables[var]` (or attribute access), where `config["variables"]`
#     defines the available variable names.
#
#     - Assumes placeholders are written as ${var}.
#     - Leaves the root `config["variables"]` section unchanged.
#     - Performs substitution recursively through dicts/lists.
#     """
#     placeholder_pattern = regular_expression.compile(r"\$\{([A-Za-z_]\w*)\}")
#     variable_names = set(config[config_syntax_for_variable_search_space].keys())
#
#     def get_value(mapping_or_object, name):
#         try:
#             return mapping_or_object[name]
#         except (TypeError, KeyError):
#             return getattr(mapping_or_object, name)
#
#     def transform(node, is_root=False):
#         if isinstance(node, dict):
#             for key, value in node.items():
#                 if is_root and key == config_syntax_for_variable_search_space:
#                     continue
#                 node[key] = transform(value)
#             return node
#         if isinstance(node, list):
#             for index, value in enumerate(node):
#                 node[index] = transform(value)
#             return node
#         if isinstance(node, str) and placeholder_pattern.search(node):
#             def expression_function(specification, template=node):
#                 variables_namespace = specification.config.variables
#                 full_match = placeholder_pattern.fullmatch(template)
#                 if full_match and full_match.group(1) in variable_names:
#                     return get_value(variables_namespace, full_match.group(1))
#                 return placeholder_pattern.sub(
#                     lambda match: str(get_value(variables_namespace, match.group(1)))
#                     if match.group(1) in variable_names
#                     else match.group(0),
#                     template,
#                 )
#
#             return tune.sample_from(expression_function)
#         return node
#
#     return transform(config, is_root=True)

def substitute_variables_throughout_config(config):
    """
    Replaces ${var} placeholders throughout `config` with the (already sampled)
    concrete values stored in `config[config_syntax_for_variable_search_space]`.

    - Assumes placeholders are written as ${var}.
    - Leaves the root `config[config_syntax_for_variable_search_space]` section unchanged.
    - Performs substitution recursively through dicts/lists.
    - If a string is exactly "${var}", the substituted value preserves its type.
      Otherwise, placeholders are substituted into the string via str(...).
    """
    placeholder_pattern = regular_expression.compile(r"\$\{([A-Za-z_]\w*)\}")
    variables_mapping = config.get(config_syntax_for_variable_search_space) or {}
    variable_names = set(variables_mapping.keys())

    work_stack = [(None, None, config, True)]
    while work_stack:
        parent, key, node, is_root = work_stack.pop()

        if isinstance(node, dict):
            for child_key, child_value in node.items():
                if is_root and child_key == config_syntax_for_variable_search_space:
                    continue
                work_stack.append((node, child_key, child_value, False))
            continue

        if isinstance(node, list):
            for index, child_value in enumerate(node):
                work_stack.append((node, index, child_value, False))
            continue

        if not isinstance(node, str) or not placeholder_pattern.search(node):
            continue

        full_match = placeholder_pattern.fullmatch(node)
        if full_match and full_match.group(1) in variable_names:
            replacement = variables_mapping[full_match.group(1)]
        else:
            replacement = placeholder_pattern.sub(
                lambda match: str(variables_mapping[match.group(1)])
                if match.group(1) in variable_names
                else match.group(0),
                node,
            )

        if parent is not None:
            parent[key] = replacement

    return config

# =========================================================================== #
# Model Training Code:
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Cross Validation:
# --------------------------------------------------------------------------- #

def calculate_mean_of_metrics(metric_dictionaries_iterable: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute per-metric means across folds.

    - Average the available numeric values other than NaN for each metric.
    - Torch / NumPy scalars are converted via .item() when available.
    """
    # Conceptual Idea:
    # In k-fold cross validation, we accumulate metrics dictionary returns from each iteration.
    # This safely aggregates the dictionaries, discarding missing metrics, and computes the mathematical expectation correctly.
    accumulated_metric_totals_dictionary = defaultdict(float)
    accumulated_metric_counts_dictionary = defaultdict(int)
    for individual_metric_dictionary in metric_dictionaries_iterable:
        for metric_name_string, metric_numerical_value in (individual_metric_dictionary or {}).items():
            if metric_numerical_value is None:
                continue
            if hasattr(metric_numerical_value, "item"):
                metric_numerical_value = metric_numerical_value.item()
            try:
                metric_numerical_value = float(metric_numerical_value)
            except (TypeError, ValueError):
                continue

            if math.isnan(metric_numerical_value):
                continue
            accumulated_metric_totals_dictionary[metric_name_string] += metric_numerical_value
            accumulated_metric_counts_dictionary[metric_name_string] += 1
    return {metric_name_string: accumulated_metric_totals_dictionary[metric_name_string] / accumulated_metric_counts_dictionary[metric_name_string] for metric_name_string in accumulated_metric_totals_dictionary if accumulated_metric_counts_dictionary[metric_name_string]}


def execute_cross_validation(
    configuration_dictionary,
    model_builder_class_reference,
    training_dataloaders_iterable: Iterable[Any],
    validation_dataloaders_iterable: Iterable[Any],
    test_dataloaders_iterable: Iterable[Any],
    target_checkpoint_file_path: Optional[str] = "best",
    evaluate_training_split_flag: bool = True,
    load_model_from_checkpoint_path = None
) -> Dict[str, Dict[str, float]]:
    """
    Run cross-validation when splits are already produced as fold-specific dataloaders.

    Assumes a PyTorch Lightning-like API:
      - trainer.fit(model, train_dataloaders=..., val_dataloaders=...)
      - trainer.validate(model, dataloaders=..., ckpt_path=...) -> List[Dict] or Dict
      - trainer.test(model, dataloaders=..., ckpt_path=...) -> List[Dict] or Dict

    Returns mean metrics across folds for each split:
      {"train": {...}, "val": {...}, "test": {...}}
    """
    # Conceptual Idea:
    # Automates sequential robust cross-validation modeling over segmented partitions of datasets cleanly.
    # Instantiates distinct trainers and models per-fold, executes fitting, and securely aggregates out resulting performance metrics.
    training_metric_dictionaries: List[Dict[str, Any]] = []
    validation_metric_dictionaries: List[Dict[str, Any]] = []
    test_metric_dictionaries: List[Dict[str, Any]] = []

    for training_dataloader_object, validation_dataloader_object, test_dataloader_object in zip(
        training_dataloaders_iterable, validation_dataloaders_iterable, test_dataloaders_iterable
    ):
        instantiated_neural_network_model = model_builder_class_reference(configuration_dictionary)
        pytorch_lightning_trainer_instance = construct_pytorch_lightning_trainer(configuration_dictionary)
        
        # NOTE: External API keywords like train_dataloaders, val_dataloaders, ckpt_path must be preserved exactly for Lightning compatibility.
        pytorch_lightning_trainer_instance.fit(instantiated_neural_network_model, train_dataloaders=training_dataloader_object, val_dataloaders=validation_dataloader_object, ckpt_path=load_model_from_checkpoint_path)

        if evaluate_training_split_flag:
            training_results_output = pytorch_lightning_trainer_instance.validate(instantiated_neural_network_model, dataloaders=training_dataloader_object, ckpt_path=target_checkpoint_file_path)
            raw_train_metrics = training_results_output[0] if isinstance(training_results_output, list) else training_results_output
            
            # Remap 'validation_' keys back to 'train_'
            remapped_train_metrics = {}
            for k, v in (raw_train_metrics or {}).items():
                new_key = k.replace("validation_", "train_").replace("val_", "train_")
                remapped_train_metrics[new_key] = v
                
            training_metric_dictionaries.append(remapped_train_metrics)

        validation_results_output = pytorch_lightning_trainer_instance.validate(instantiated_neural_network_model, dataloaders=validation_dataloader_object, ckpt_path=target_checkpoint_file_path)
        validation_extracted_metrics = validation_results_output[0] if isinstance(validation_results_output, list) else validation_results_output
        validation_metric_dictionaries.append(validation_extracted_metrics or {})

        test_results_output = pytorch_lightning_trainer_instance.test(instantiated_neural_network_model, dataloaders=test_dataloader_object, ckpt_path=target_checkpoint_file_path)
        test_extracted_metrics = test_results_output[0] if isinstance(test_results_output, list) else test_results_output
        test_metric_dictionaries.append(test_extracted_metrics or {})
        
        del pytorch_lightning_trainer_instance, instantiated_neural_network_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "train": calculate_mean_of_metrics(training_metric_dictionaries) if evaluate_training_split_flag else {},
        "val": calculate_mean_of_metrics(validation_metric_dictionaries),
        "test": calculate_mean_of_metrics(test_metric_dictionaries),
    }

# --------------------------------------------------------------------------- #
# Training Function:
# --------------------------------------------------------------------------- #

def disable_ray_autoscaler() -> None:
    # Conceptual Idea:
    # Defensively intercept Ray Data's internal scaling framework which often conflicts with 
    # strictly partitioned SLURM allocations on computing clusters, preventing abrupt crash shutdowns.
    
    # 1) Patch the DefaultAutoscaler shutdown hook (if present)
    try:
        import ray.data._internal.execution.autoscaler.default_autoscaler.default_autoscaler as default_autoscaler_module
        if hasattr(default_autoscaler_module, "DefaultAutoscaler"):
            default_autoscaler_module.DefaultAutoscaler.on_executor_shutdown = lambda self, *positional_arguments, **keyword_arguments: None
            print("[shim][trial] Patched DefaultAutoscaler.on_executor_shutdown -> no-op")
    except Exception as default_autoscaler_exception:
        print(f"[shim][trial] Autoscaler patch not applied: {default_autoscaler_exception}")

    # 2) Tune DataContext flags (if present)
    try:
        from ray.data import DataContext
        current_data_context = DataContext.get_current()
        if hasattr(current_data_context, "use_streaming_executor"):
            current_data_context.use_streaming_executor = False
        if hasattr(getattr(current_data_context, "execution_options", None), "autoscaling_enabled"):
            current_data_context.execution_options.autoscaling_enabled = False
    except Exception as data_context_exception:
        print(f"[trial] DataContext tuning skipped: {data_context_exception}")


def write_average_metrics_to_file(average_metrics_dictionary: dict, target_metrics_file_path: str) -> None:
    """Append metrics to metrics_file_path, prefixing keys with train_/val_/test_."""
    # Conceptual Idea:
    # Recursively format all hierarchical metric evaluations logically separated by dataset fold splits 
    # down into a flat text-file JSON line logging structure cleanly appending progressively over trials.
    flattened_metrics_dictionary = {f"{dataset_split_name}_{metric_name_string}": metric_numerical_value for dataset_split_name in ("train", "val", "test") for metric_name_string, metric_numerical_value in average_metrics_dictionary.get(dataset_split_name, {}).items()}
    with open(target_metrics_file_path, "a", encoding="utf-8") as target_metrics_file_stream:
        target_metrics_file_stream.write(json.dumps(flattened_metrics_dictionary, sort_keys=True, default=float) + "\n")

# TODO: Add to imports.
import datetime
import pathlib
import secrets
import shutil
import subprocess
import sys
import tempfile
import json
import yaml

def write_checkpoint_directory(parent_directory_path_string, configuration_dictionary):
    # Conceptual Idea:
    # Archives robust exact binary replicas securing deterministic rollback dependencies cleanly linking the 
    # local execution source code state alongside all conda pip environment definitions at execution time exactly.
    resolved_parent_directory_object = pathlib.Path(parent_directory_path_string).expanduser().resolve()
    checkpoint_source_file_path = resolved_parent_directory_object / "checkpoint.ckpt"
    if not checkpoint_source_file_path.is_file():
        raise FileNotFoundError(f"Expected {checkpoint_source_file_path} to exist.")
    current_timestamp_string = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    unique_checkpoint_directory_path = resolved_parent_directory_object / f"{current_timestamp_string}_{secrets.randbelow(10000):04d}_Checkpoint"
    unique_checkpoint_directory_path.mkdir(parents=True, exist_ok=False)
    shutil.copy2(checkpoint_source_file_path, unique_checkpoint_directory_path / "checkpoint.ckpt")
    (unique_checkpoint_directory_path / "config.yml").write_text(yaml.safe_dump(configuration_dictionary, sort_keys=False), encoding="utf-8")
    wheel_destination_file_path = unique_checkpoint_directory_path / "original_code.whl"
    with tempfile.TemporaryDirectory() as temporary_wheel_build_directory:
        subprocess.run([sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "-w", temporary_wheel_build_directory], check=True)
        compiled_wheel_file_paths = list(pathlib.Path(temporary_wheel_build_directory).glob("*.whl"))
        if len(compiled_wheel_file_paths) != 1:
            raise RuntimeError(f"Expected 1 wheel; found {len(compiled_wheel_file_paths)}: {compiled_wheel_file_paths}")
        shutil.copy2(compiled_wheel_file_paths[0], wheel_destination_file_path)
    environment_evaluation_command = ["conda", "list", "--show-channel-urls"] if shutil.which("conda") else [sys.executable, "-m", "pip", "list", "--verbose"]
    environment_output_text_string = subprocess.run(environment_evaluation_command, check=True, capture_output=True, text=True).stdout
    (unique_checkpoint_directory_path / "original_environment.txt").write_text(environment_output_text_string, encoding="utf-8")
    return unique_checkpoint_directory_path

def construct_pytorch_lightning_trainer(configuration_dictionary):
    # Conceptual Idea:
    # Parses the physical hardware configuration requirements (GPUs, CPU threading) and translates them 
    # directly into a strictly configured PyTorch Lightning Trainer object managing the actual execution loop.
    requested_hardware_graphics_processing_units = int((configuration_dictionary.get("resources") or {}).get("gpu", 0) or 0)

    # Accelerator selection, controlled by the optional top-level `accelerator` config key:
    #   "auto" (default): CUDA GPU if resources.gpu > 0 and CUDA is present, else CPU (original behaviour).
    #   "cpu": force CPU.   "gpu"/"cuda": force CUDA.   "mps": use the Apple-Silicon GPU (Metal).
    # Ray does not track MPS, so it still allocates CPU for the trial; PYTORCH_ENABLE_MPS_FALLBACK=1
    # (set at the top of this file) lets MPS-unsupported ops fall back to CPU.
    requested_accelerator_name = str((configuration_dictionary or {}).get("accelerator", "auto")).strip().lower()
    mps_backend_is_available = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    cuda_requested_and_available = requested_hardware_graphics_processing_units > 0 and torch.cuda.is_available()
    if requested_accelerator_name == "mps":
        if not mps_backend_is_available:
            raise RuntimeError("config accelerator: mps requested, but torch.backends.mps.is_available() is False on this machine.")
        hardware_accelerator_type, devices_allocation_strategy, distributed_data_parallel_strategy = "mps", 1, "auto"
    elif requested_accelerator_name in ("gpu", "cuda") or (requested_accelerator_name == "auto" and cuda_requested_and_available):
        hardware_accelerator_type = "gpu"
        devices_allocation_strategy = "auto"
        distributed_data_parallel_strategy = "ddp" if requested_hardware_graphics_processing_units > 1 else "auto"
    else:
        hardware_accelerator_type, devices_allocation_strategy, distributed_data_parallel_strategy = "cpu", "auto", "auto"
    
    # Added: allow the config to cap training epochs (top-level `max_epochs`) for quick smoke tests.
    maximum_number_of_epochs = int((configuration_dictionary or {}).get("max_epochs", 50))
    return pytorch_lightning.Trainer(
        max_epochs=maximum_number_of_epochs,
        accelerator=hardware_accelerator_type,
        devices=devices_allocation_strategy,
        strategy=distributed_data_parallel_strategy,
        log_every_n_steps=10,
        enable_checkpointing=True,
        callbacks=[],
        # Optional but common on GPU:
        # precision="16-mixed" if use_graphics_processing_unit else 32,
    )

def execute_model_training_function(
        configuration_dictionary,
        training_datasets,
        validation_datasets,
        test_datasets,
        dataloader_kwargs
        ):
    
    # Conceptual Idea:
    # The main training execution routine for a single trial. It sets up the environment, resolves config variables,
    # and orchestrates the cross-validation process across all provided data splits.
    
    # 1) Turn off autoscalar so ray doesn't break.
    disable_ray_autoscaler()
    
    # Instantiate DataLoaders inside the Trial Worker!
    training_dataloaders = [DataLoader(ds, **dataloader_kwargs) for ds in training_datasets]
    validation_dataloaders = [DataLoader(ds, **dataloader_kwargs) for ds in validation_datasets]
    test_dataloaders = [DataLoader(ds, **dataloader_kwargs) for ds in test_datasets]
    
    # 2) Substitute all sampled variables throughout config (this should be
    # done automatically before training_function is called).
    configuration_dictionary = substitute_variables_throughout_config(configuration_dictionary)
    
    # 3) Compute performance metrics.
    assert (total_number_of_folds := len(training_dataloaders)) == len(validation_dataloaders) == len(test_dataloaders), "There must be an equal number of train, validation, and test dataset splits."
    ArchitectureBuilderClassReference = retrieve_interpreter_component(get_interpreter_name(configuration_dictionary))    
    computed_performance_metrics_dictionary = execute_cross_validation(configuration_dictionary, ArchitectureBuilderClassReference, training_dataloaders, validation_dataloaders, test_dataloaders)
    
    # 4) Write out performance metrics results.
    target_path_for_results_file = configuration_dictionary["write_results_to"]
    write_average_metrics_to_file(computed_performance_metrics_dictionary, target_path_for_results_file)
# --------------------------------------------------------------------------- #
# Hyperparameter Optimization:
# --------------------------------------------------------------------------- #

def execute_hyperparameter_optimization(configuration_dictionary,
                                training_datasets_iterable, 
                                validation_datasets_iterable, 
                                test_datasets_iterable,
                                dataloader_kwargs):
    
    # Conceptual Idea:
    # Wrapper function delegating the training loop to Ray Tune. It slices the config correctly 
    # to feed hardware resources, algorithmic bounds, and data structures straight to the optimization engine.
    compute_resource_configuration_dictionary: Dict = configuration_dictionary[config_syntax_for_resources]
    hyperparameter_search_space_configuration_dictionary: Dict = configuration_dictionary # This will be passed into the training function.
    search_optimization_configuration_dictionary: TuneConfig = configuration_dictionary[config_syntax_for_optimization]
    execution_run_configuration_dictionary: Dict = configuration_dictionary[config_syntax_for_run_configuration]
    
    constructed_ray_tuner_object = construct_tuner(
        execute_model_training_function,
        compute_resource_configuration_dictionary,
        hyperparameter_search_space_configuration_dictionary,
        search_optimization_configuration_dictionary,
        execution_run_configuration_dictionary,
        training_datasets_iterable,
        validation_datasets_iterable,
        test_datasets_iterable,
        dataloader_kwargs # NEW ARGUMENT
    )
    
    tuning_results_output = constructed_ray_tuner_object.fit()

# =========================================================================== #
# Pipeline Execution Methods:
# =========================================================================== #

def execute_pipeline_from_configuration_path(configuration_file_path_string):
    
    tempprint("Running pipeline.")
    
    # Conceptual Idea:
    # Main entry point. Traverses the user-defined yaml map, discovers all active execution 
    # sub-configs, parses out the hyperparameter placeholder bindings, and begins executing the runs sequentially.
    
    # 1) Collect all configs into a single list.
    extracted_configuration_dictionaries = collect_configs_to_execute(configuration_file_path_string)
    print(f"Identified a total of {len(extracted_configuration_dictionaries)} {make_plural_if_required('config', len(extracted_configuration_dictionaries))} files to execute.")
    
    # 2) Process each into a final executable form.
    for individual_configuration_dictionary in extracted_configuration_dictionaries:
        individual_configuration_dictionary[config_syntax_for_variable_search_space] = compile_variable_bindings(individual_configuration_dictionary[config_syntax_for_variable_expressions])
    
    # 3) Execute each config.
    for individual_configuration_dictionary in extracted_configuration_dictionaries:
        print(f"Beginning execution of config at: {individual_configuration_dictionary}")
        execute_pipeline_from_configuration(individual_configuration_dictionary)

def execute_pipeline_from_configuration(configuration_dictionary):
    
    # Conceptual Idea:
    # Handles dynamic dispatch based on what the config is asking to do (e.g. Model Training vs Scoring).
    # Ensures prerequisites are met, processes dataset paths into proper loaders, and executes the designated interpreter.
    
    # 0) Keep a copy of the original.
    original_unmodified_configuration_dictionary = deepcopy(configuration_dictionary)
    
    # 1) Determine which interpreter must be used to read rest of file.
    target_interpreter_name_string = get_interpreter_name(configuration_dictionary)
    
    # 2) See what fields the interpreter needs to run correctly.
    expected_mandatory_fields = get_expected_fields_for_an_interpreter(target_interpreter_name_string)
    
    # 3) Ensure all info is present in the config file that is necessary.
    require_keys(configuration_dictionary, expected_mandatory_fields)
    
    # 4) Process all subconfig components.
    for expected_configuration_field_name in expected_mandatory_fields:
        field_extraction_method_reference = get_extraction_method_from_field_name(expected_configuration_field_name)
        processed_extracted_field_dictionary = field_extraction_method_reference(deepcopy(original_unmodified_configuration_dictionary))
        configuration_dictionary[expected_configuration_field_name] = processed_extracted_field_dictionary
    
    # 5) Run all relevant pre-model steps.
    # extra_keyword_arguments = {}
    
    if config_syntax_for_dataset_annotation in expected_mandatory_fields:
        
        # 1) Define a colate method for the dataset.
        dataset_configuration_subdictionary = configuration_dictionary[config_syntax_for_dataset_annotation]
        assigned_dataset_collate_method = partial(collate_from_config, dataset_config=dataset_configuration_subdictionary)
        
        # 2) Load a dataset.
        target_dataset_file_path = configuration_dictionary[config_syntax_for_dataset_annotation]["path_to_dataset"]
        loaded_data_table_object = to_table(target_dataset_file_path, to_class=IterableTable)
    
    if config_syntax_for_dataset_spliting_method in expected_mandatory_fields:
    
        # 3) Split dataset into train, val, and test splits.
        dataset_splitting_subconfiguration_dictionary = configuration_dictionary[config_syntax_for_dataset_spliting_method]
        
        number_of_test_datasets_requested = int(dataset_splitting_subconfiguration_dictionary.get("include_test_dataset", False))
        number_of_training_validation_subsets_requested = dataset_splitting_subconfiguration_dictionary.get("cross_fold_validation_number", 2)
        total_number_of_dataset_splitting_folds = number_of_training_validation_subsets_requested + number_of_test_datasets_requested
        

        training_datasets, validation_datasets, test_datasets = generate_random_cross_validation_splits(
            loaded_data_table_object,
            number_of_folds=int(dataset_splitting_subconfiguration_dictionary.get("cross_fold_validation_number", 10)),
            include_test_split=bool(dataset_splitting_subconfiguration_dictionary.get("include_test_dataset", True)),
            maximum_total_rows=dataset_splitting_subconfiguration_dictionary.get("maximum_total_rows", None),
            random_seed=int(dataset_splitting_subconfiguration_dictionary.get("random_seed", 42)),
        )
    
        use_graphics_processing_unit = torch.cuda.is_available()
        pytorch_dataloader_keyword_arguments = {
            "batch_size": 256,
            "num_workers": 4 if use_graphics_processing_unit else 0,          # tune for your machine
            "persistent_workers": True if use_graphics_processing_unit else False,
            "collate_fn": assigned_dataset_collate_method,
            "pin_memory": True if use_graphics_processing_unit else False,
        }
    
    if target_interpreter_name_string == config_syntax_for_model_training_interpreter:
        execute_hyperparameter_optimization(configuration_dictionary,
                                            training_datasets, # PASS DATASETS, NOT DATALOADERS
                                            validation_datasets,
                                            test_datasets,
                                            pytorch_dataloader_keyword_arguments # PASS KWARGS
                                            )
    elif target_interpreter_name_string == config_syntax_for_fine_tuning_interpreter:
            pass
    elif target_interpreter_name_string == config_syntax_for_dataset_scoring_interpreter:
            pass

# ########################################################################### #
#
# ########################################################################### #

# =========================================================================== #
# Module Builder
# =========================================================================== #

from graphlib import TopologicalSorter

EXTRA_FIELDS = ["encoders", "layer_1", "layer_2", "layer_3", "layer_4"]

def normalize_reference(target_module_name_string):
    return f"{target_module_name_string}/out"

def construct_neural_network_module(module_configuration_dictionary, should_instantiate_flag=True):
    
    # Conceptual Idea:
    # Factory function for resolving a sub-dictionary from the configuration into an actual Python object.
    
    # 1) Get the class of module that is to be made.
    neural_network_module_class = retrieve_core_component(module_configuration_dictionary["type"])
    
    # 2) Instantiate each module with the appropriate arguments.
    keyword_parameters_dictionary = dict(module_configuration_dictionary.get("model_parameters") or module_configuration_dictionary.get("params") or {})
    for configuration_field_name in EXTRA_FIELDS:
        if configuration_field_name in module_configuration_dictionary and configuration_field_name not in keyword_parameters_dictionary:
            keyword_parameters_dictionary[configuration_field_name] = module_configuration_dictionary[configuration_field_name]
            
    # 3) Instantiate or return a callable factory
    if should_instantiate_flag:
        return neural_network_module_class(**keyword_parameters_dictionary)
    else:
        # LibMTL needs an uninstantiated constructor to dynamically spin up experts.
        return functools.partial(neural_network_module_class, **keyword_parameters_dictionary)

class DirectedAcyclicGraphModule(neural_network.Module):
    def __init__(self, topological_execution_order, input_wiring_dictionary, output_ports_dictionary, prediction_wiring_dictionary, instantiated_submodules_dictionary):
        super().__init__()
        self.topological_execution_order, self.input_wiring_dictionary, self.output_ports_dictionary, self.prediction_wiring_dictionary, self.instantiated_submodules_dictionary = (
            topological_execution_order, input_wiring_dictionary, output_ports_dictionary, prediction_wiring_dictionary, instantiated_submodules_dictionary
        )

    # Returns all intermediate outputs.
    def forward(self, input_tensors_dictionary):
        """Returns all intermediate outputs."""
        
        # Conceptual Idea:
        # Graph execution engine. Dynamically pushes tensor data through uniquely connected modules, 
        # saving all intermediate values in a localized state dictionary to be referenced by downstream layers.
        intermediate_results_store_dictionary = {}
        
        # 1) Inputs are required.
        for input_name_string, input_tensor_value in input_tensors_dictionary.items():
            intermediate_results_store_dictionary[normalize_reference(input_name_string)] = input_tensor_value
        
        # 2) Iterate over each node and...
        for current_node_name_string in self.topological_execution_order:
            
            node_input_wiring = self.input_wiring_dictionary[current_node_name_string]
            current_module_instance = self.instantiated_submodules_dictionary[current_node_name_string]

            fetched_node_input_tensors = [intermediate_results_store_dictionary[input_wire_dictionary["from"]] for input_wire_dictionary in node_input_wiring]
            
            # I want to convert node_inputs such that multiple 1d tensors that
            # get passed in are concatenated in order into a larger tensor.
            # if len(fetched_node_input_tensors) != 1:
            #     concatenated_node_inputs_tensor = torch.cat(fetched_node_input_tensors, dim=-1) if fetched_node_input_tensors else None
            # else:
            #     concatenated_node_inputs_tensor = fetched_node_input_tensors[0]
            # module_execution_results = current_module_instance(concatenated_node_inputs_tensor) if concatenated_node_inputs_tensor is not None else current_module_instance()
            
            if len(fetched_node_input_tensors) > 1:
                module_execution_results = current_module_instance(*fetched_node_input_tensors)
            elif len(fetched_node_input_tensors) == 1:
                module_execution_results = current_module_instance(fetched_node_input_tensors[0])
            else:
                module_execution_results = current_module_instance()
                
            expected_node_output_ports = self.output_ports_dictionary[current_node_name_string]
            
            if isinstance(module_execution_results, (list, tuple)):
                assert len(module_execution_results) == len(expected_node_output_ports)
            else:
                module_execution_results = listify(module_execution_results)
            
            for execution_result_tensor, output_port_name in zip(module_execution_results, expected_node_output_ports):
                intermediate_results_store_dictionary.update({f"{current_node_name_string}/{output_port_name}": execution_result_tensor})
                
        # 3)
        
        final_model_predictions_dictionary = {}
        for target_task_name, stored_reference_key in self.prediction_wiring_dictionary.items():
            final_model_predictions_dictionary[target_task_name] = intermediate_results_store_dictionary[f"predictions/{target_task_name}"] = intermediate_results_store_dictionary[stored_reference_key]
        return final_model_predictions_dictionary


def construct_module_based_model(module_configurations):
    
    # Conceptual Idea:
    # Converts a declarative YAML topological layout into a strictly ordered PyTorch module graph. 
    # Validates dependencies using a topological sorter to ensure inputs are generated strictly before they are consumed.
    
    # 1) Extract modules subconfig and ensure it is uniquely named.
    assert all_unique([module_configuration_dictionary["name"] for module_configuration_dictionary in deepcopy(module_configurations)])
    
    configuration_node_by_name_dictionary = {configuration_node["name"]: configuration_node for configuration_node in module_configurations}
    
    input_wiring_dictionary, output_ports_dictionary, topological_dependencies_dictionary = {}, {}, {}
    for node_name_string, node_configuration_dictionary in configuration_node_by_name_dictionary.items():
        
        node_inputs = assert_type(node_configuration_dictionary.get("inputs"), (dict, list))
        node_inputs = listify(node_inputs)
        
        required_predecessors_set = set()
        for predecessor_input_dictionary in node_inputs:
            if "/" not in predecessor_input_dictionary["from"]:
                predecessor_input_dictionary["from"] = normalize_reference(predecessor_input_dictionary["from"])
            predecessor_node_name_string = predecessor_input_dictionary["from"].split("/", 1)[0]
            if predecessor_node_name_string in configuration_node_by_name_dictionary:
                required_predecessors_set.add(predecessor_node_name_string)
        input_wiring_dictionary[node_name_string] = node_inputs
        
        # Either the module defines multiple outputs (returns a list of them)
        # or it has a single output in which case that single output has the
        # value /out added to the end of it.
        node_outputs = listify(node_configuration_dictionary.get("outputs", "out"))
        output_ports_dictionary[node_name_string] = node_outputs
        topological_dependencies_dictionary[node_name_string] = required_predecessors_set
    
    topological_execution_order = list(TopologicalSorter(topological_dependencies_dictionary).static_order())
    instantiated_modules_dictionary = neural_network.ModuleDict()
    for sorted_node_name_string in topological_execution_order:
        sorted_module_configuration_dictionary = configuration_node_by_name_dictionary[sorted_node_name_string]
        instantiated_modules_dictionary[sorted_node_name_string] = construct_neural_network_module(sorted_module_configuration_dictionary)
    
    prediction_task_to_output_source_dictionary = {}
    for target_module_name_string, target_module_configuration_dictionary in configuration_node_by_name_dictionary.items():
        
        if "predictions" not in target_module_configuration_dictionary:
            continue
        
        module_predictions = listify(target_module_configuration_dictionary["predictions"])
        for prediction_configuration_dictionary in module_predictions:
            
            # If from output is provided, 
            if "from_output" in prediction_configuration_dictionary:
                source_output_reference_string = f"{target_module_name_string}/{prediction_configuration_dictionary['from_output']}"
            else:
                source_output_reference_string = normalize_reference(target_module_name_string)
            
            associated_target_task_name = prediction_configuration_dictionary["for_task"]
            prediction_task_to_output_source_dictionary[associated_target_task_name] = source_output_reference_string
            
    return DirectedAcyclicGraphModule(topological_execution_order, input_wiring_dictionary, output_ports_dictionary, prediction_task_to_output_source_dictionary, instantiated_modules_dictionary)

# ########################################################################### #
# Builders:
# ########################################################################### #

# =========================================================================== #
# Modular_Builder:
# =========================================================================== #

@register("interpreter", "model_training")
class ModuleBuilderTrainer(pytorch_lightning.LightningModule):
    
# --------------------------------------------------------------------------- #
# Init Methods:
# --------------------------------------------------------------------------- #
    
    def __init__(self, config):
        super().__init__()

        # Manual optimization so the weighting strategy (e.g. PCGrad) actually drives the backward
        # and optimizer step in reweight_losses. Under automatic optimization Lightning would
        # backpropagate the plain summed loss and the weighting object would never be used.
        self.automatic_optimization = False
        
        self.save_hyperparameters(config)
        
        # Objects that can be made during __init__.
        self.metrics_manager = construct_metric_stack_from_configs(config[config_syntax_for_model_performance_metrics])
        self.loss_function = construct_loss_function_from_configuration(config[config_syntax_for_the_loss_function])
        self.model = construct_module_based_model(config[config_syntax_for_model_modules])
        self.weighting_object = construct_weighting_object_from_configuration(config.get("weighting", DEFAULT_LOSS_WEIGHTING_STRATEGY), list(self.model.parameters()))
        
        #Must be fully instantiated after weights are available.
        self.make_optimizer_using_model_parameters = partial(optimizer_from_optimizer_config, optimizer_configuration_dictionary=config[config_syntax_for_weight_optimization])
        
        # callbacks
        self.early_stopping_configuration = config["early_stopping"]
        
        # References needed later.
        self.input_column_names = config[config_syntax_for_dataset_annotation]["input_column_names"]
        self.task_column_names = config[config_syntax_for_dataset_annotation]["task_column_names"]
    
# --------------------------------------------------------------------------- #
# Helper Methods:
# --------------------------------------------------------------------------- #
    
    def get_values_from_batch(self, batch, inputs_or_targets):
        
        match inputs_or_targets:
            case "inputs":
                relevant_columns = self.input_column_names 
            case "targets":
                relevant_columns = self.task_column_names
        filter_function = lambda key: key in relevant_columns
        
        return_values = {}
        for target_column in filter(filter_function, batch):
            return_values[target_column] = batch[target_column]["values"]
        return return_values
    
    def get_input_values_from_batch(self, batch):
        return self.get_values_from_batch(batch, "inputs")
    
    def get_target_values_from_batch(self, batch):
        return self.get_values_from_batch(batch, "targets")

    def get_model_parameters(self):
        return self.model.parameters()
    
    def get_batch_size(self, batch):
        for target_column_name in batch:
            try:
                return len(batch[target_column_name]["mask"])
            except Exception:
                continue
        return None
        
    def reweight_losses(self, loss_dictionary, batch_size):
        total_loss = loss_dictionary["total"]
        if self.automatic_optimization:
            self.log("train_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size)
        else:
            optimizer_instance = self.optimizers()
            if isinstance(optimizer_instance, (list, tuple)):
                optimizer_instance = optimizer_instance[0]
            optimizer_instance.zero_grad()
            task_losses = [value for key, value in loss_dictionary.items() if key != "total"]
            total_loss = self.weighting_object.backward(task_losses)
            optimizer_instance.step()
            self.log("train_loss", total_loss.detach(), prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size)
            
    def log_loss_dictionary(self, loss_dictionary, stage, prog_bar=False):
        
        for loss_term, loss_value in loss_dictionary.items():

            loss_value = loss_value.detach() if isinstance(loss_value, torch.Tensor) and loss_value.requires_grad else loss_value
            self.log(f"{stage}_{loss_term}", loss_value, prog_bar=prog_bar, on_step=False, on_epoch=True) #batch_size=batch_size)
            
    def log_metrics(self, metrics, stage):
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item() if value.numel() == 1 else float(value.mean().item())
            self.log(f"{stage}_{key}", value, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)
            
    def move_to_device(self, object_value, device):
        
        if isinstance(object_value, torch.Tensor):
            return object_value.to(device, non_blocking=True)
        if hasattr(object_value, "to") and not isinstance(object_value, neural_network.Module):
            try:
                return object_value.to(device, non_blocking=True)
            except TypeError:
                return object_value.to(device)
        if isinstance(object_value, dict):
            return {key: self.move_to_device(value, device) for key, value in object_value.items()}
        if isinstance(object_value, (list, tuple)):
            object_type = type(object_value)
            return object_type(self.move_to_device(value, device) for value in object_value)
        return object_value

# --------------------------------------------------------------------------- #
# Main Methods:
# --------------------------------------------------------------------------- #

    def forward(self, batch):
        model_inputs = self.get_input_values_from_batch(batch)
        model_outputs = self.model(model_inputs)
        return model_outputs
    
    def configure_optimizers(self):
        model_parameters = self.get_model_parameters()
        optimizer = self.make_optimizer_using_model_parameters(model_parameters)
        return optimizer

    def on_fit_start(self):
        # Lightning may move the model to its device after __init__, which replaces the parameter
        # tensors captured by the weighting object. Re-point it at the live parameters so gradient
        # surgery and the optimizer operate on the same tensors (matters on GPU; a no-op on CPU).
        self.weighting_object.shared_parameters = list(self.model.parameters())
    
    def training_step(self, batch, batch_index):
        
        # 1) Make predictions.
        predictions = self.forward(batch)
        # Predictions has the structure of a dictionary mapping task names to predictions on those tasks.
        
        # 2) Calculate loss.
        # targets_dictionary = self.get_target_values_from_batch(batch)
        loss_dictionary = self.loss_function(model=self.model, predictions=predictions, batch=batch, training_epoch_number=self.current_epoch)
        total_loss = loss_dictionary["total"]
        
        # 3) Do loss reweighting.
        batch_size = self.get_batch_size(batch)
        self.reweight_losses(loss_dictionary, batch_size)
        
        # 4) Log losses.
        self.log_loss_dictionary(loss_dictionary, "train")
        
        # 5) Return total loss.
        return total_loss
    
    def validation_step(self, batch, batch_index):
        
        # 1) Make predictions.
        predictions = self.forward(batch)
        # Predictions has the structure of a dictionary mapping task names to predictions on those tasks.
        
        # 2) Calculate loss.
        # targets_dictionary = self.get_target_values_from_batch(batch)
        loss_dictionary = self.loss_function(model=self.model, predictions=predictions, batch=batch, training_epoch_number=self.current_epoch)
        
        # 3) Update metrics calculations.
        self.metrics_manager.update(predictions, batch)
        
        # 4) Log losses.
        self.log_loss_dictionary(loss_dictionary, "validation")
    
    def test_step(self, batch, batch_index):
        
        # 1) Make predictions.
        predictions = self.forward(batch)
        # Predictions has the structure of a dictionary mapping task names to predictions on those tasks.
        
        # 2) Calculate loss.
        # targets_dictionary = self.get_target_values_from_batch(batch)
        loss_dictionary = self.loss_function(model=self.model, predictions=predictions, batch=batch, training_epoch_number=self.current_epoch)
        
        # 3) Update metrics calculations.
        self.metrics_manager.update(predictions, batch)
        
        # 4) Log losses.
        self.log_loss_dictionary(loss_dictionary, "test")
        
    def on_validation_epoch_start(self):
        
        # 1) Reset metrics between validation runs.
        self.metrics_manager.reset()
        
    def on_validation_epoch_end(self):
        
        # 1) Calculate and log per-epoch metrics.
        per_epoch_metrics = self.metrics_manager.compute(requested_sample_scope="epoch")
        self.log_metrics(per_epoch_metrics, stage="validation")
    
    def on_test_epoch_start(self):
        
        # 1) Reset metrics before test run.
        self.metrics_manager.reset()
        
    def on_test_epoch_end(self):
        
        # 1) Calculate and log per-epoch metrics.
        per_epoch_metrics = self.metrics_manager.compute(requested_sample_scope="epoch")
        self.log_metrics(per_epoch_metrics, stage="test")
        
    def transfer_batch_to_device(self, batch, device, dataloader_idx: int):
        
        # 1) General recursive transfer device call.
        return self.move_to_device(batch, device)
    
    def create_early_stopping_config(self):

        early_stopping_configuration = getattr(self, "early_stopping_configuration", None)
        if not isinstance(early_stopping_configuration, dict) or not early_stopping_configuration:
            return None

        monitor_key = str(early_stopping_configuration.get("monitor", "validation_total"))
        if monitor_key == "val_loss":
            monitor_key = "validation_total"
        elif monitor_key.startswith("val_"):
            monitor_key = "validation_" + monitor_key[4:]

        return EarlyStopping(
            monitor=monitor_key,
            mode=str(early_stopping_configuration.get("mode", "min")).lower(),
            patience=int(early_stopping_configuration.get("patience", 1)),
            min_delta=float(early_stopping_configuration.get("min_delta", 0.0)),
            check_on_train_epoch_end=False,
            strict=False,
        )


    def configure_callbacks(self):
        # Local imports keep this module self-contained
        from pytorch_lightning.callbacks import ModelCheckpoint
        from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback

        self.callbacks = []

        # Report these to Ray every validation epoch
        metric_keys = ["validation_total"]
        for module in self.metrics_manager.iterate_over_modules():
            base = module.__class__.__name__.replace("Metric", "").lower()
            key = f"{base}_{module.target_task_name}" if getattr(module, "target_task_name", None) else base
            metric_keys.append(f"validation_{key}")

        # Determine the monitor key/mode
        monitor_key = "validation_total"
        monitor_mode = "min"
        if isinstance(self.early_stopping_configuration, dict):
            monitor_key = self.early_stopping_configuration.get("monitor", monitor_key)
            monitor_mode = self.early_stopping_configuration.get("mode", monitor_mode)

        monitor_key = str(monitor_key)         
        monitor_mode = str(monitor_mode).lower()
        if monitor_key == "val_loss":
            monitor_key = "validation_total"
        elif monitor_key.startswith("val_"):
            monitor_key = "validation_" + monitor_key[4:]

        # (C) Ensure Ray sees the exact monitor key and the monotonic best version
        if monitor_key not in metric_keys:
            metric_keys.append(monitor_key)

        # Report metrics and write a Ray checkpoint each validation
        self.callbacks.append(TuneReportCheckpointCallback(metrics=metric_keys, on="validation_end"))

        # (A) Save only the best Lightning checkpoint on disk (by the same monitor)
        self.callbacks.append(ModelCheckpoint(
            monitor=monitor_key,
            mode=monitor_mode,
            save_top_k=1,
            save_last=False,
            save_on_train_epoch_end=False,
            filename=f"best-{{epoch:02d}}-{{{monitor_key}:.5f}}",
        ))

        # Keep your existing EarlyStopping if configured
        early_stopping_callback = self.create_early_stopping_config()
        if early_stopping_callback is not None:
            self.callbacks.append(early_stopping_callback)

        return self.callbacks

"""
delta_pipeline_additions.py
===========================

Everything the Model Building Pipeline needs to train the concat-Minimol delta FFN
and benchmark it on Biogen. It adds three things and edits nothing:

  1. a `minimol_lookup` input object_type  -> maps a SMILES column to its 512-d
     Minimol vector via a prebuilt {smiles: vector} lookup, so the pair CSVs stay
     small (they hold SMILES, not 512-float strings) and there is no per-epoch
     re-parsing.
  2. a predefined-column splitter          -> train/val/test taken straight from a
     `Split` column, so we can train on ChEMBL rows and test on Biogen rows in one
     file. Installed by wrapping the (hardcoded) Enamine splitter so it auto-detects
     a train/val/test `Split` column; every existing Enamine dataset is untouched.
  3. `pearson` and `spearman` metrics       -> the delta benchmark numbers,
     accumulated across the epoch and computed once at the end (non-decomposable,
     same treatment as AUROC).

HOW TO USE
----------
Set MINIMOL_LOOKUP_PATH to the absolute path of the MiniMol lookup file.
"""

# ======================================================================= #
# 1. Minimol lookup input object_type
# ======================================================================= #
import os
import pickle

_MINIMOL_LOOKUP = None


def _load_minimol_lookup():
    """Lazy-load the {smiles: float32[512]} Minimol lookup once per process."""
    global _MINIMOL_LOOKUP
    if _MINIMOL_LOOKUP is None:
        lookup_path = os.environ.get("MINIMOL_LOOKUP_PATH", "")
        if not lookup_path or not os.path.isfile(lookup_path):
            raise FileNotFoundError(
                "Minimol lookup not found. Set MINIMOL_LOOKUP_PATH to the pickle "
                f"produced during data prep (got MINIMOL_LOOKUP_PATH={lookup_path!r})."
            )
        with open(lookup_path, "rb") as handle:
            _MINIMOL_LOOKUP = pickle.load(handle)
    return _MINIMOL_LOOKUP


@register("preprocessing", "minimol_lookup")
def preprocess_minimol_lookup(smiles_string):
    # Map a SMILES to its precomputed 512-d Minimol vector. Returning None on a
    # miss lets collate_from_config mask that row exactly like any missing value.
    lookup_table = _load_minimol_lookup()
    representation_vector = lookup_table.get(smiles_string)
    if representation_vector is None:
        return None
    return torch.as_tensor(np.asarray(representation_vector, dtype=np.float32))


@register("collate", "minimol_lookup")
def collate_minimol_lookup(list_of_vectors):
    return torch.stack(list_of_vectors, dim=0).float()   # (batch, 512)


# get_placeholder_for_object_type is a plain module function; wrap it so the new
# object_type has a correctly-sized (512) placeholder for the rare masked row.
_original_get_placeholder_for_object_type = get_placeholder_for_object_type


def get_placeholder_for_object_type(object_type):          # noqa: F811 (intentional override)
    if object_type == "minimol_lookup":
        return torch.zeros(512)
    return _original_get_placeholder_for_object_type(object_type)


# ======================================================================= #
# 2. Predefined-column splitter (train / val / test from a `Split` column)
# ======================================================================= #
def generate_predefined_column_splits(
    data_table,
    split_column: str = "Split",
    train_label: str = "train",
    validation_label: str = "val",
    test_label: str = "test",
):
    """Return ([train], [val], [test]) IterableTables from a precomputed Split column.

    Mirrors generate_enamine_train_validation_test_splits' return shape (three lists,
    one fold each) so it drops straight into execute_cross_validation.
    """
    data_table.assert_column_in_table(split_column)
    materialized_frame = data_table.to_pandas(max_rows=None)
    split_labels = materialized_frame[split_column].astype(str)
    train_frame = materialized_frame.loc[split_labels == train_label].reset_index(drop=True)
    validation_frame = materialized_frame.loc[split_labels == validation_label].reset_index(drop=True)
    test_frame = materialized_frame.loc[split_labels == test_label].reset_index(drop=True)
    return (
        [convert_to_iterable_table(train_frame)],
        [convert_to_iterable_table(validation_frame)],
        [convert_to_iterable_table(test_frame)],
    )


# execute_pipeline_from_configuration calls generate_enamine_train_validation_test_splits
# directly (the split registry is not consulted there). Rather than edit that function,
# wrap it: if the loaded table carries a train/val/test `Split` column, use the
# predefined splitter; otherwise fall back to the original Enamine behaviour. Enamine
# datasets have no such column, so they are completely unaffected (and pay no cost:
# the guard is a column-presence check before any materialization).
_original_generate_enamine_splits = generate_enamine_train_validation_test_splits


def generate_enamine_train_validation_test_splits(data_table, *args, **kwargs):   # noqa: F811
    if data_table.column_in_table("Split"):
        unique_labels = set(
            str(value) for value in data_table.to_pandas(max_rows=None)["Split"].unique()
        )
        if unique_labels and unique_labels <= {"train", "val", "test"}:
            return generate_predefined_column_splits(data_table, split_column="Split")
    return _original_generate_enamine_splits(data_table, *args, **kwargs)


# ======================================================================= #
# 3. Correlation metrics (Pearson / Spearman) for the delta benchmark
# ======================================================================= #
@register("metric", "pearson")
class PearsonMetric(MetricModule):
    """Pearson r between predicted and observed delta, computed over the whole epoch.

    Correlation is not batch-decomposable (like AUROC), so predictions and targets
    are accumulated and the coefficient is computed once in metric_on_epoch.
    """

    def __init__(self, target_task_name: str):
        super().__init__(target_task_name=target_task_name)
        self.accumulated_predictions = []
        self.accumulated_targets = []
        self.last_computed_batch_value = None

    @staticmethod
    def _correlation(predicted_values, observed_values):
        if predicted_values.numel() < 2:
            return float("nan")
        predicted_values = predicted_values.float()
        observed_values = observed_values.float()
        centered_predicted = predicted_values - predicted_values.mean()
        centered_observed = observed_values - observed_values.mean()
        denominator = (centered_predicted.norm() * centered_observed.norm()).clamp_min(1e-12)
        return float((centered_predicted @ centered_observed) / denominator)

    def update(self, model_predictions_dictionary, data_batch_dictionary, sample_metric_weighting=None) -> None:
        sample_metric_weighting = retrieve_metric_weights_from_batch_for_task(data_batch_dictionary, self.get_task())
        true_target_labels = retrieve_target_values_from_batch_for_task(data_batch_dictionary, self.get_task())
        masked_targets, masked_predictions, masked_weights, _, should_continue = super().update(
            model_predictions_dictionary, true_target_labels, data_batch_dictionary, sample_metric_weighting
        )
        if not should_continue:
            return
        prediction_scalars = masked_predictions.detach().reshape(masked_weights.size(0), -1).mean(-1).cpu()
        target_scalars = masked_targets.detach().reshape(masked_weights.size(0), -1).mean(-1).cpu()
        self.accumulated_predictions.append(prediction_scalars)
        self.accumulated_targets.append(target_scalars)
        self.last_computed_batch_value = self._correlation(prediction_scalars, target_scalars)

    def metric_on_batch(self) -> float:
        return self.last_computed_batch_value

    def metric_on_epoch(self) -> float:
        if not self.accumulated_predictions:
            return float("nan")
        all_predictions = torch.cat(self.accumulated_predictions)
        all_targets = torch.cat(self.accumulated_targets)
        return self._correlation(all_predictions, all_targets)

    def reset(self) -> None:
        self.accumulated_predictions = []
        self.accumulated_targets = []
        self.last_computed_batch_value = None


@register("metric", "spearman")
class SpearmanMetric(PearsonMetric):
    """Spearman rho: Pearson on the ranks (ties handled by scipy when available)."""

    @staticmethod
    def _correlation(predicted_values, observed_values):
        if predicted_values.numel() < 2:
            return float("nan")
        try:
            from scipy.stats import spearmanr
            coefficient = spearmanr(predicted_values.numpy(), observed_values.numpy()).correlation
            return float(coefficient) if coefficient == coefficient else float("nan")
        except Exception:
            def _fractional_rank(values):
                ordering = values.argsort()
                ranks = torch.empty_like(ordering, dtype=torch.float)
                ranks[ordering] = torch.arange(values.numel(), dtype=torch.float)
                return ranks
            return PearsonMetric._correlation(
                _fractional_rank(predicted_values.float()), _fractional_rank(observed_values.float())
            )

###############################################################################
# Other Code:
###############################################################################    

if __name__ == "__main__":
   
    import sys
    execute_pipeline_from_configuration_path("/Users/asselism/Desktop/Collins_Lab/Agentic_Environments/Preliminary_ADME_Test/delta_configs/delta_ffn_hlm_benchmark.yaml")