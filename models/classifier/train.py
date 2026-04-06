"""
Train the accident edge classifier with PyTorch.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models.classifier.model import (
    ARCHITECTURE_PRESETS,
    AccidentClassifier,
    build_branch_dims,
    parse_branch_widths,
)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUTS_ROOT = (PROJECT_ROOT / "outputs").resolve()
DEFAULT_DATA_ROOT = (PROJECT_ROOT / "data").resolve()

EDGE_N2V_FEATURES = ("maxspeed", "time", "length", "lanes", "ref")
DEFAULT_BRANCH_NAMES = (
    "emb_maxspeed",
    "emb_time",
    "emb_length",
    "emb_lanes",
    "emb_ref",
    "bridge",
    "highway",
    "junction",
    "lanes",
    "slope",
    "angle",
    "oneway",
)
LEGACY_BRANCH_NAMES = (
    "emb_maxspeed",
    "emb_time",
    "emb_length",
    "bridge",
    "highway",
    "junction",
    "lanes",
    "slope",
    "angle",
    "oneway",
)
KNOWN_BRANCH_NAMES = tuple(dict.fromkeys(DEFAULT_BRANCH_NAMES + LEGACY_BRANCH_NAMES))
N2V_BRANCH_TO_KEY = {
    "emb_maxspeed": "maxspeed",
    "emb_time": "time",
    "emb_length": "length",
    "emb_lanes": "lanes",
    "emb_ref": "ref",
}
N2V_BRANCH_LABELS = {
    "maxspeed": "node2vec edge_maxspeed",
    "time": "node2vec edge_time",
    "length": "node2vec edge_length",
    "lanes": "node2vec edge_lanes",
    "ref": "node2vec edge_ref",
}

CATEGORY_BRANCHES = ("bridge", "highway", "junction", "lanes", "oneway")
NUMERIC_BRANCHES = ("slope", "angle")

LEGACY_CAT_FILES = {
    "bridge": "Feature-bridge.csv",
    "highway": "Feature-highway.csv",
    "junction": "Feature-junction.csv",
    "lanes": "Feature-lanes.csv",
    "angle": "Feature-angle.csv",
    "slope": "Feature-slope.csv",
    "oneway": "Feature-oneway.csv",
    "accident": "Feature-have_accident.csv",
}

UNKNOWN_TOKEN = "__unknown__"


@dataclass(frozen=True)
class DataManifest:
    """
    Minimal dataset provenance record persisted alongside checkpoints.

    Attributes:
        embeddings: Mapping from embedding key to the source ``.npy`` path.
        edge_info_csv: Interim edge metadata CSV path, or ``None`` in legacy
            mode.
        branch_names: Ordered list of branch names used by the model.
    """

    embeddings: dict[str, str]
    edge_info_csv: str | None = None
    branch_names: list[str] | None = None


# --------------------------------------------------------------------------- #
# Path helpers
# --------------------------------------------------------------------------- #


def _resolve_root(
    raw_root: str | None,
    fallback_root: Path,
) -> Path:
    """
    Resolve a root directory argument to an absolute path.

    Args:
        raw_root: Raw CLI value; may be ``None``, relative, or absolute.
        fallback_root: Directory returned when ``raw_root`` is ``None``.

    Returns:
        Absolute :class:`Path` for the requested root.
    """
    if raw_root is None:
        return fallback_root
    root = Path(raw_root).expanduser()
    if root.is_absolute():
        return root
    return (PROJECT_ROOT / root).resolve()


def _resolve_path(
    raw_path: str | None,
    root: Path,
    default_path: Path,
) -> Path:
    """
    Resolve a file path argument to an absolute path.

    Searches, in order, the current working directory, ``root``, then the
    project root. If none of the candidates exist, the ``root``-relative
    candidate is returned so the caller can raise a clear error.

    Args:
        raw_path: Raw CLI value; ``None`` falls back to ``default_path``.
        root: Base directory used when ``raw_path`` is relative.
        default_path: Path returned when ``raw_path`` is ``None``.

    Returns:
        Resolved :class:`Path`.
    """
    if raw_path is None:
        return default_path
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    if path.exists():
        return path
    candidate = (root / path).resolve()
    if candidate.exists():
        return candidate
    project_candidate = (PROJECT_ROOT / path).resolve()
    if project_candidate.exists():
        return project_candidate
    return candidate


def _ensure_file_exists(
    path: Path,
    label: str,
) -> None:
    """
    Raise :class:`FileNotFoundError` if ``path`` does not exist.

    Args:
        path: Filesystem path to check.
        label: Human-readable description used in the error message.
    """
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def _make_logger(
    log_path: Path,
) -> logging.Logger:
    """
    Create a logger that writes INFO logs to stdout and ``log_path``.

    Args:
        log_path: File path for the file handler (opened in write mode).

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger("classifier_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger


def _validate_and_log_embeddings(
    embeddings: dict[str, Path],
    logger: logging.Logger,
) -> None:
    """
    Log and verify that every required node2vec embedding file exists.

    Args:
        embeddings: Mapping from embedding key to ``.npy`` path.
        logger: Logger used to emit status lines.

    Raises:
        FileNotFoundError: If any required embedding path does not exist.
    """
    logger.info("Loading node2vec embeddings:")
    for key in EDGE_N2V_FEATURES:
        path = embeddings[key]
        label = N2V_BRANCH_LABELS.get(key, key)
        status = "OK" if path.exists() else "MISSING"
        logger.info(f"  - {label}: {path} [{status}]")
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")


# --------------------------------------------------------------------------- #
# Feature processing helpers
# --------------------------------------------------------------------------- #


def _set_seed(
    seed: int,
) -> None:
    """
    Seed Python, NumPy, and PyTorch (including CUDA) RNGs.

    Args:
        seed: Integer seed applied to all random-number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize_seeded_value(
    values: Iterable[Any],
) -> list[str]:
    """
    Coerce raw categorical values to stripped strings, using a NaN sentinel.

    Args:
        values: Raw iterable of categorical values (may include NaN / None).

    Returns:
        List of stripped strings with ``UNKNOWN_TOKEN`` substituted for NaN.
    """
    return [UNKNOWN_TOKEN if pd.isna(v) else str(v).strip() for v in values]


def _read_one_column_csv(
    path: Path,
) -> np.ndarray:
    """
    Load a single column from a CSV file, preferring known header names.

    Args:
        path: Path to the CSV file to read.

    Returns:
        1-D NumPy array of the selected column's values.

    Raises:
        ValueError: If the feature column cannot be inferred.
    """
    frame = pd.read_csv(path)
    if frame.shape[1] == 1:
        return frame.iloc[:, 0].to_numpy()
    if "feature" in frame.columns:
        return frame["feature"].to_numpy()
    if "_col0" in frame.columns:
        return frame["_col0"].to_numpy()
    raise ValueError(f"Cannot infer feature column in {path}")


def _to_one_hot(
    values: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """
    One-hot encode integer class codes as a ``float32`` matrix.

    Args:
        values: 1-D array of integer class indices.
        num_classes: Total number of classes (width of the output matrix).

    Returns:
        ``float32`` array of shape ``(len(values), num_classes)``.
    """
    values = np.asarray(values, dtype=np.int64)
    return np.eye(num_classes, dtype=np.float32)[values]


def _build_category_mapping(
    raw_values: Iterable[Any],
) -> dict[str, int]:
    """
    Build a stable category->code map that always contains ``UNKNOWN_TOKEN``.

    Args:
        raw_values: Raw values to derive categories from.

    Returns:
        Ordered mapping from category string to integer code, with
        ``UNKNOWN_TOKEN`` always present.
    """
    mapping: dict[str, int] = {}
    for token in _normalize_seeded_value(raw_values):
        if token not in mapping:
            mapping[token] = len(mapping)
    if UNKNOWN_TOKEN not in mapping:
        mapping[UNKNOWN_TOKEN] = len(mapping)
    return mapping


def _apply_category_mapping(
    raw_values: Iterable[Any],
    mapping: dict[str, int],
) -> np.ndarray:
    """
    Encode values using ``mapping``; unseen tokens fall back to ``UNKNOWN_TOKEN``.

    Args:
        raw_values: Raw categorical values to encode.
        mapping: Category-to-code mapping; mutated in place to include
            ``UNKNOWN_TOKEN`` when missing.

    Returns:
        ``int64`` array of integer codes aligned with ``raw_values``.
    """
    if UNKNOWN_TOKEN not in mapping:
        mapping[UNKNOWN_TOKEN] = len(mapping)
    unknown_idx = mapping[UNKNOWN_TOKEN]
    return np.array(
        [
            mapping.get(token, unknown_idx)
            for token in _normalize_seeded_value(raw_values)
        ],
        dtype=np.int64,
    )


def _ensure_float_series(
    series: pd.Series,
) -> np.ndarray:
    """
    Convert a pandas Series to a ``(N, 1)`` ``float32`` numpy array.

    Args:
        series: Input Series; non-numeric values are coerced to ``0.0``.

    Returns:
        ``float32`` array of shape ``(N, 1)``.
    """
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(np.float32)
    return values.to_numpy().reshape(-1, 1)


def _load_legacy_features(
    path: Path,
) -> dict[str, np.ndarray]:
    """
    Load every legacy ``Feature-*.csv`` file from ``path``.

    Args:
        path: Directory containing the legacy ``Feature-*.csv`` files.

    Returns:
        Mapping from feature name to the loaded 1-D NumPy array.
    """
    return {
        name: _read_one_column_csv(path / filename)
        for name, filename in LEGACY_CAT_FILES.items()
    }


# --------------------------------------------------------------------------- #
# Dataset construction
# --------------------------------------------------------------------------- #


def _resolve_requested_branches(
    requested: list[str] | None,
) -> list[str]:
    """
    Validate and normalise a requested branch-name list.

    Args:
        requested: Explicit branch names, or ``None`` for the default set.

    Returns:
        Validated list of branch names.
    """
    if requested is None:
        return list(DEFAULT_BRANCH_NAMES)
    if not requested:
        raise ValueError(
            "requested_branch_names must contain at least one branch name."
        )
    requested_names = [str(name) for name in requested]
    invalid = [name for name in requested_names if name not in KNOWN_BRANCH_NAMES]
    if invalid:
        raise ValueError(f"Unknown branch names: {invalid}")
    return requested_names


def _load_embedding_arrays(
    embeddings: dict[str, Path],
    requested_names: list[str],
) -> tuple[dict[str, np.ndarray], int]:
    """
    Load node2vec embedding arrays for the requested branches.

    Args:
        embeddings: Mapping from embedding key to ``.npy`` path. Must contain
            every key in :data:`EDGE_N2V_FEATURES`.
        requested_names: Branch names requested by the caller.

    Returns:
        Tuple ``(branch_arrays, n_samples)`` where ``branch_arrays`` is keyed
        by branch name and ``n_samples`` is the row count shared across arrays.
    """
    missing = [key for key in EDGE_N2V_FEATURES if key not in embeddings]
    if missing:
        raise ValueError(f"Missing required embedding(s): {missing}")

    branch_data: dict[str, np.ndarray] = {}
    n_samples: int | None = None
    for branch_name in requested_names:
        embedding_key = N2V_BRANCH_TO_KEY.get(branch_name)
        if embedding_key is None:
            continue
        array = np.asarray(np.load(str(embeddings[embedding_key])), dtype=np.float32)
        if n_samples is None:
            n_samples = len(array)
        elif len(array) != n_samples:
            raise ValueError("Embedding arrays must contain the same number of rows.")
        branch_data[branch_name] = array

    if n_samples is None:
        raise ValueError("No embedding features were loaded.")
    return branch_data, n_samples


def _load_raw_tabular_features(
    legacy_feature_dir: Path | None,
    edge_info_csv: Path | None,
    n_samples: int,
) -> tuple[dict[str, Any], np.ndarray]:
    """
    Load raw categorical/numeric feature columns and the accident label.

    Args:
        legacy_feature_dir: Directory of legacy ``Feature-*.csv`` files; when
            provided, legacy mode is used.
        edge_info_csv: Interim edge metadata CSV; required when
            ``legacy_feature_dir`` is ``None``.
        n_samples: Expected row count (validated against the loaded data).

    Returns:
        Tuple ``(raw_columns, labels)`` where ``raw_columns`` maps feature
        names to raw columns and ``labels`` is a binary ``int64`` accident
        array.
    """
    if legacy_feature_dir is not None:
        legacy = _load_legacy_features(legacy_feature_dir)
        if len(legacy["bridge"]) != n_samples:
            raise ValueError(
                "Legacy feature files have different row count than embeddings."
            )
        raw_columns: dict[str, Any] = {
            "bridge": legacy["bridge"],
            "highway": legacy["highway"],
            "junction": legacy["junction"],
            "lanes": legacy["lanes"],
            "oneway": legacy["oneway"],
            "angle": legacy["angle"],
            "slope": legacy["slope"],
        }
        accident: Any = legacy["accident"]
    else:
        if edge_info_csv is None:
            raise ValueError(
                "edge_info_csv must be provided when not using legacy features."
            )
        edge_df = pd.read_csv(edge_info_csv)
        if len(edge_df) != n_samples:
            raise ValueError(
                "Interim edge table and embeddings have different row counts."
            )

        # Legacy pipeline treated `lanes` as a categorical label; preserve that.
        lanes_raw = (
            pd.to_numeric(
                edge_df.get("lanes", pd.Series([UNKNOWN_TOKEN] * n_samples)),
                errors="coerce",
            )
            .fillna(UNKNOWN_TOKEN)
            .astype(str)
        )

        raw_columns = {
            "bridge": edge_df.get("bridge", pd.Series([UNKNOWN_TOKEN] * n_samples)),
            "highway": edge_df.get("highway", pd.Series([UNKNOWN_TOKEN] * n_samples)),
            "junction": edge_df.get("junction", pd.Series([UNKNOWN_TOKEN] * n_samples)),
            "lanes": lanes_raw,
            "oneway": edge_df.get("oneway", pd.Series([UNKNOWN_TOKEN] * n_samples)),
            "angle": edge_df.get("angle", pd.Series([0.0] * n_samples)),
            "slope": edge_df.get("slope", pd.Series([0.0] * n_samples)),
        }

        if "have_accident" in edge_df.columns:
            accident = edge_df["have_accident"]
        elif "number_of_accident" in edge_df.columns:
            accident = (
                pd.to_numeric(edge_df["number_of_accident"], errors="coerce") > 0
            ).astype(int)
        else:
            raise ValueError(
                "No accident label column found. Expected have_accident or number_of_accident."
            )

    if isinstance(accident, np.ndarray):
        labels_arr = accident
    else:
        labels_arr = pd.Series(accident).astype(int).to_numpy()
    return raw_columns, labels_arr.astype(np.int64)


def _resolve_category_maps(
    raw_columns: dict[str, Any],
    category_maps: dict[str, dict[str, int]] | None,
    requested_names: list[str],
) -> dict[str, dict[str, int]]:
    """
    Build or reuse a category->code mapping for each categorical branch.

    Args:
        raw_columns: Mapping from feature name to raw column values.
        category_maps: Pre-built mappings from a checkpoint, or ``None`` to
            derive fresh mappings from ``raw_columns``.
        requested_names: Branch names selected for the current run.

    Returns:
        Mapping from each categorical branch name to its category->code dict.
    """
    if category_maps is None:
        return {
            name: _build_category_mapping(raw_columns[name])
            for name in CATEGORY_BRANCHES
        }

    resolved: dict[str, dict[str, int]] = {}
    for name in CATEGORY_BRANCHES:
        if name in requested_names:
            resolved[name] = dict(category_maps[name])
        else:
            resolved[name] = _build_category_mapping([UNKNOWN_TOKEN])
    return resolved


def load_dataset(
    embeddings: dict[str, Path],
    legacy_feature_dir: Path | None,
    edge_info_csv: Path | None = None,
    category_maps: dict[str, dict[str, int]] | None = None,
    requested_branch_names: list[str] | None = None,
) -> tuple[
    list[np.ndarray],
    np.ndarray,
    dict[str, dict[str, int]],
    dict[str, int],
    DataManifest,
    list[str],
]:
    """
    Assemble per-branch feature arrays, labels, and dataset metadata.

    Args:
        embeddings: Mapping from embedding key to ``best_model_edges.npy`` path.
        legacy_feature_dir: Optional directory of legacy ``Feature-*.csv``
            files. When provided, legacy mode is used.
        edge_info_csv: Interim edge metadata CSV; required when
            ``legacy_feature_dir`` is ``None``.
        category_maps: Optional pre-built category mappings (e.g. from a
            checkpoint). When provided, unseen tokens map to ``UNKNOWN_TOKEN``.
        requested_branch_names: Subset of branches to build. Defaults to
            :data:`DEFAULT_BRANCH_NAMES`.

    Returns:
        ``(features, labels, category_maps, input_dims, manifest, branch_names)``
        where ``features`` is one array per branch in ``branch_names`` order
        and ``input_dims`` is a mapping from branch name to feature width.
    """
    requested_names = _resolve_requested_branches(requested_branch_names)
    branch_data, n_samples = _load_embedding_arrays(embeddings, requested_names)
    raw_columns, labels = _load_raw_tabular_features(
        legacy_feature_dir, edge_info_csv, n_samples
    )

    category_maps_out = _resolve_category_maps(
        raw_columns, category_maps, requested_names
    )

    one_hot_features = {
        name: _to_one_hot(
            _apply_category_mapping(raw_columns[name], category_maps_out[name]),
            len(category_maps_out[name]),
        )
        for name in CATEGORY_BRANCHES
    }
    numeric_features = {
        name: _ensure_float_series(pd.Series(raw_columns[name]))
        for name in NUMERIC_BRANCHES
    }

    empty = np.empty((n_samples, 0), dtype=np.float32)
    candidate_branch_data: dict[str, np.ndarray] = {
        "emb_maxspeed": branch_data.get("emb_maxspeed", empty),
        "emb_time": branch_data.get("emb_time", empty),
        "emb_length": branch_data.get("emb_length", empty),
        "emb_lanes": branch_data.get("emb_lanes", empty),
        "emb_ref": branch_data.get("emb_ref", empty),
        **one_hot_features,
        **numeric_features,
    }

    features = [candidate_branch_data[name] for name in requested_names]
    input_dims = {
        name: int(candidate_branch_data[name].shape[1]) for name in requested_names
    }

    manifest = DataManifest(
        embeddings={k: str(v) for k, v in embeddings.items()},
        edge_info_csv=str(edge_info_csv) if edge_info_csv is not None else None,
        branch_names=requested_names,
    )

    return features, labels, category_maps_out, input_dims, manifest, requested_names


# --------------------------------------------------------------------------- #
# Dataset / training utilities
# --------------------------------------------------------------------------- #


class AccidentDataset(Dataset):
    """
    ``torch.utils.data.Dataset`` wrapping per-branch feature tensors.
    """

    def __init__(
        self,
        features: list[np.ndarray],
        labels: np.ndarray,
    ) -> None:
        """
        Args:
            features: One array per model branch, each shaped ``(N, F_i)``.
            labels: Class labels of shape ``(N,)``.
        """
        self.features = [torch.as_tensor(value) for value in features]
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(
        self,
    ) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            Total number of samples.
        """
        return self.labels.shape[0]

    def __getitem__(
        self,
        index: int,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        Retrieve a single sample by index.

        Args:
            index: Sample index.

        Returns:
            Tuple of ``(branch_feature_list, label)`` where each tensor in the
            list corresponds to one model branch.
        """
        row = [tensor[index] for tensor in self.features]
        return row, self.labels[index]


def _collect_predictions(
    model: AccidentClassifier,
    dataloader: DataLoader,
    device: str,
) -> tuple[list[int], list[float], float]:
    """
    Run inference and return ``(y_true, y_prob1, mean_loss)``.

    ``y_prob1`` stores class-1 posterior probabilities for each sample.

    Args:
        model: Trained classifier to run in evaluation mode.
        dataloader: Loader yielding ``(branch_features, labels)`` batches.
        device: Target device string passed to ``tensor.to``.

    Returns:
        Tuple ``(y_true, y_prob1, mean_loss)`` where ``y_true`` is the list of
        ground-truth labels, ``y_prob1`` is per-sample class-1 probabilities,
        and ``mean_loss`` is the average cross-entropy loss across batches.
    """
    model.eval()
    losses: list[float] = []
    y_true: list[int] = []
    y_prob1: list[float] = []
    with torch.no_grad():
        for batch_features, batch_labels in dataloader:
            feature_tensors = [t.to(device) for t in batch_features]
            labels = batch_labels.to(device)
            logits = model(feature_tensors)
            losses.append(float(nn.functional.cross_entropy(logits, labels).item()))
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = probs
            y_true.extend(labels.detach().cpu().numpy().tolist())
            y_prob1.extend(preds.detach().cpu().numpy().tolist())
    mean_loss = float(np.mean(losses)) if losses else 0.0
    return y_true, y_prob1, mean_loss


def _evaluate(
    model: AccidentClassifier,
    dataloader: DataLoader,
    device: str,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Evaluate ``model`` on ``dataloader`` and return aggregate metrics.

    Args:
        model: Trained classifier to evaluate.
        dataloader: Loader yielding ``(branch_features, labels)`` batches.
        device: Target device string passed to ``tensor.to``.
        threshold: Class-1 probability threshold for binary prediction.

    Returns:
        Dict with keys ``loss``, ``accuracy``, ``recall``, ``precision``,
        ``f1``, and ``confusion_matrix``.
    """
    y_true, y_prob1, mean_loss = _collect_predictions(model, dataloader, device)
    y_prob_arr = np.asarray(y_prob1, dtype=np.float32)
    y_pred = (y_prob_arr >= threshold).astype(np.int64).tolist()
    return {
        "loss": mean_loss,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0.0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0.0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0.0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def _find_best_threshold(
    y_true: list[int],
    y_prob1: list[float],
    metric: str = "f1",
    min_precision: float = 0.0,
    num_candidates: int = 1001,
) -> dict[str, float | list[int] | list[list[int]]]:
    """
    Search a fixed grid of class-1 probability thresholds for the best metric.

    Args:
        y_true: Ground-truth binary labels.
        y_prob1: Predicted class-1 probabilities.
        metric: Metric to maximise; one of ``"accuracy"``, ``"precision"``,
            ``"recall"``, or ``"f1"``.
        min_precision: Minimum precision a candidate threshold must satisfy.
        num_candidates: Number of evenly-spaced thresholds to evaluate.

    Returns:
        Dict containing ``threshold`` and the metric values (``accuracy``,
        ``recall``, ``precision``, ``f1``, ``confusion_matrix``) at the
        best-scoring threshold.

    Raises:
        ValueError: If ``num_candidates < 2`` or ``metric`` is unsupported.
    """
    if not y_prob1:
        return {
            "threshold": 0.5,
            "loss": float("nan"),
            "accuracy": 0.0,
            "recall": 0.0,
            "precision": 0.0,
            "f1": 0.0,
            "confusion_matrix": [[0, 0], [0, 0]],
        }

    y_prob = np.asarray(y_prob1, dtype=np.float32)
    y_true_arr = np.asarray(y_true, dtype=np.int64)

    if num_candidates < 2:
        raise ValueError("num_candidates must be at least 2.")

    thresholds = np.linspace(0.0, 1.0, int(num_candidates))
    best_value = -math.inf
    best_threshold = 0.5
    best_stats: dict[str, float | list[int] | list[list[int]]] = {
        "loss": float("nan"),
        "accuracy": 0.0,
        "recall": 0.0,
        "precision": 0.0,
        "f1": 0.0,
        "confusion_matrix": [[0, 0], [0, 0]],
    }

    for threshold in thresholds:
        pred = (y_prob >= threshold).astype(np.int64)
        acc = float(accuracy_score(y_true_arr, pred))
        recall = float(recall_score(y_true_arr, pred, zero_division=0.0))
        precision = float(precision_score(y_true_arr, pred, zero_division=0.0))
        if precision < min_precision:
            continue
        f1 = float(f1_score(y_true_arr, pred, zero_division=0.0))

        if metric == "accuracy":
            score = acc
        elif metric == "precision":
            score = precision
        elif metric == "recall":
            score = recall
        elif metric == "f1":
            score = f1
        else:
            raise ValueError(f"Unsupported threshold metric: {metric}")

        if score > best_value:
            best_value = score
            best_threshold = float(threshold)
            best_stats = {
                "loss": float("nan"),
                "accuracy": acc,
                "recall": recall,
                "precision": precision,
                "f1": f1,
                "confusion_matrix": confusion_matrix(y_true_arr, pred).tolist(),
            }

    return {
        "threshold": best_threshold,
        **{
            key: float(value)
            for key, value in best_stats.items()
            if key != "confusion_matrix"
        },
        "confusion_matrix": best_stats["confusion_matrix"],
    }


def train_one_epoch(
    model: AccidentClassifier,
    dataloader: DataLoader,
    device: str,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    """
    Run one training epoch and return averaged loss / accuracy / recall.

    Args:
        model: Classifier module.
        dataloader: Loader producing ``(branch_features, labels)`` batches.
        device: Target device string passed to ``tensor.to``.
        optimizer: PyTorch optimiser used for the parameter update.

    Returns:
        Dict with ``loss``, ``accuracy`` and ``recall`` over the epoch.
    """
    model.train()
    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []

    for batch_features, batch_labels in dataloader:
        feature_tensors = [t.to(device) for t in batch_features]
        labels = batch_labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(feature_tensors)
        loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))
        preds = torch.argmax(logits, dim=-1)
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())

    mean_loss = float(np.mean(losses)) if losses else 0.0
    return {
        "loss": mean_loss,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0.0)),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the training entry point.

    Returns:
        :class:`argparse.Namespace` populated with all training hyper-parameters.
    """
    parser = argparse.ArgumentParser(description="Train PyTorch accident classifier.")
    parser.add_argument(
        "--outputs-root",
        default=str(DEFAULT_OUTPUTS_ROOT),
        help="Base folder for node2vec embedding outputs.",
    )
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        help="Base folder for tabular input data.",
    )
    parser.add_argument(
        "--maxspeed-embeddings",
        default=None,
        help="Optional override for outputs/node2vec/edge_maxspeed/best_model_edges.npy.",
    )
    parser.add_argument(
        "--time-embeddings",
        default=None,
        help="Optional override for outputs/node2vec/edge_time/best_model_edges.npy.",
    )
    parser.add_argument(
        "--length-embeddings",
        default=None,
        help="Optional override for outputs/node2vec/edge_length/best_model_edges.npy.",
    )
    parser.add_argument(
        "--lanes-embeddings",
        default=None,
        help="Optional override for outputs/node2vec/edge_lanes/best_model_edges.npy.",
    )
    parser.add_argument(
        "--ref-embeddings",
        default=None,
        help="Optional override for outputs/node2vec/edge_ref/best_model_edges.npy.",
    )
    parser.add_argument(
        "--legacy-feature-dir",
        default=None,
        help="Directory containing legacy Feature-*.csv files.",
    )
    parser.add_argument(
        "--edge-info-csv",
        default=None,
        help="Edge metadata CSV. Defaults to data/final/osm_edge_info_completed.csv.",
    )
    parser.add_argument(
        "--profile",
        default="baseline",
        choices=sorted(ARCHITECTURE_PRESETS.keys()) + ["custom"],
    )
    parser.add_argument(
        "--branch-widths", default=None, help="Comma-separated custom branch widths."
    )
    parser.add_argument("--final-hidden", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--test-random-state", type=int, default=42)
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience."
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="Minimum improvement to reset early-stop counter.",
    )
    parser.add_argument(
        "--early-stop-metric",
        default="loss",
        choices=["loss", "accuracy", "recall", "precision", "f1"],
        help="Validation metric monitored for early stopping.",
    )
    parser.add_argument(
        "--threshold-metric",
        default="f1",
        choices=["accuracy", "precision", "recall", "f1"],
        help="Metric used when searching class-1 threshold.",
    )
    parser.add_argument(
        "--threshold-min-precision",
        type=float,
        default=0.0,
        help="Minimum precision required during threshold search.",
    )
    parser.add_argument(
        "--threshold-candidates",
        type=int,
        default=1001,
        help="Number of candidate thresholds checked for tuning.",
    )
    parser.add_argument("--output-dir", default="outputs/classifier")
    parser.add_argument("--run-name", default="accident_classifier")
    parser.add_argument(
        "--log-file", default=None, help="Optional path to training log output."
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    return parser.parse_args()


def _resolve_branch_widths_for_count(
    profile: str,
    custom_widths: str | None,
    num_branches: int,
) -> list[int]:
    """
    Return per-branch hidden widths sized to ``num_branches``.

    Args:
        profile: Preset name from :data:`ARCHITECTURE_PRESETS` or ``"custom"``.
        custom_widths: Comma-separated widths; required when ``profile`` is
            ``"custom"``.
        num_branches: Number of branches expected by the current feature set.

    Returns:
        A list of branch widths with exactly ``num_branches`` entries.
    """
    if profile == "custom":
        if not custom_widths:
            raise ValueError("--branch-widths is required when --profile is custom.")
        widths = list(parse_branch_widths(custom_widths))
        if len(widths) != num_branches:
            raise ValueError(
                f"Custom branch width count ({len(widths)}) must match selected feature count ({num_branches})."
            )
        return widths

    preset_widths = list(ARCHITECTURE_PRESETS[profile]["branch_widths"])
    if len(preset_widths) == num_branches:
        return preset_widths
    if len(preset_widths) < num_branches:
        return preset_widths + [preset_widths[-1]] * (num_branches - len(preset_widths))
    return preset_widths[:num_branches]


def _build_embedding_paths(
    args: argparse.Namespace,
    outputs_root: Path,
) -> dict[str, Path]:
    """
    Resolve each node2vec embedding path from CLI overrides and defaults.

    Args:
        args: Parsed CLI arguments containing optional per-feature path overrides.
        outputs_root: Root directory used to construct default embedding paths.

    Returns:
        Mapping from embedding key to the resolved ``.npy`` path.
    """
    overrides = {
        "maxspeed": args.maxspeed_embeddings,
        "time": args.time_embeddings,
        "length": args.length_embeddings,
        "lanes": args.lanes_embeddings,
        "ref": args.ref_embeddings,
    }
    return {
        key: _resolve_path(
            overrides[key],
            outputs_root,
            outputs_root / "node2vec" / f"edge_{key}" / "best_model_edges.npy",
        )
        for key in EDGE_N2V_FEATURES
    }


def main() -> None:
    """
    CLI entry point: train a classifier and dump checkpoint artifacts.
    """
    args = parse_args()
    _set_seed(args.seed)

    outputs_root = _resolve_root(args.outputs_root, DEFAULT_OUTPUTS_ROOT)
    data_root = _resolve_root(args.data_root, DEFAULT_DATA_ROOT)
    output_dir = Path(args.output_dir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = (
        Path(args.log_file) if args.log_file is not None else (output_dir / "train.log")
    )
    logger = _make_logger(log_path)
    logger.info("Starting classifier training run.")
    logger.info(f"Arguments: {vars(args)}")

    embeddings = _build_embedding_paths(args, outputs_root)
    _validate_and_log_embeddings(embeddings, logger)

    if args.legacy_feature_dir:
        legacy_dir: Path | None = _resolve_path(
            args.legacy_feature_dir, data_root, Path(args.legacy_feature_dir)
        )
        _ensure_file_exists(legacy_dir, "legacy feature directory")
        if not legacy_dir.is_dir():
            raise ValueError(
                f"legacy feature directory is not a directory: {legacy_dir}"
            )
        edge_info_csv: Path | None = None
    else:
        legacy_dir = None
        edge_info_csv = _resolve_path(
            args.edge_info_csv,
            data_root,
            data_root / "final" / "osm_edge_info_completed.csv",
        )
        _ensure_file_exists(edge_info_csv, "edge info CSV")

    features, labels, category_maps, input_dims, manifest, feature_names = load_dataset(
        embeddings=embeddings,
        legacy_feature_dir=legacy_dir,
        edge_info_csv=edge_info_csv,
        category_maps=None,
        requested_branch_names=list(DEFAULT_BRANCH_NAMES),
    )

    branch_widths = _resolve_branch_widths_for_count(
        args.profile,
        args.branch_widths,
        num_branches=len(feature_names),
    )
    if args.profile == "custom":
        final_hidden = args.final_hidden
        num_classes = 2
    else:
        final_hidden, num_classes = build_branch_dims(args.profile)

    device = args.device
    model = AccidentClassifier(
        branch_input_dims=tuple(input_dims[name] for name in feature_names),
        branch_hidden_dims=branch_widths,
        final_hidden=final_hidden,
        num_classes=num_classes,
        branch_names=feature_names,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    train_idx, test_idx = train_test_split(
        np.arange(len(labels)),
        test_size=args.test_size,
        random_state=args.test_random_state,
        shuffle=True,
    )

    train_loader = DataLoader(
        AccidentDataset([f[train_idx] for f in features], labels[train_idx]),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        AccidentDataset([f[test_idx] for f in features], labels[test_idx]),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "train_recall": [],
        "test_loss": [],
        "test_accuracy": [],
        "test_recall": [],
        "test_precision": [],
        "test_f1": [],
        "test_threshold": [],
    }

    best_threshold = 0.5
    best_epoch = 1
    best_score: float
    if args.early_stop_metric == "loss":
        best_score = float("inf")
        best_is_higher = False
    else:
        best_score = -float("inf")
        best_is_higher = True
    best_state: dict[str, torch.Tensor] = {}
    epochs_without_improvement = 0

    if args.patience < 1:
        raise ValueError("patience must be at least 1.")
    if not (0.0 <= args.threshold_min_precision <= 1.0):
        raise ValueError("threshold-min-precision must be between 0 and 1.")
    if args.threshold_candidates < 2:
        raise ValueError("threshold-candidates must be at least 2.")
    if args.early_stop_metric not in {"loss", "accuracy", "recall", "precision", "f1"}:
        raise ValueError(f"Unsupported early-stop metric: {args.early_stop_metric}")
    if args.threshold_metric not in {"accuracy", "precision", "recall", "f1"}:
        raise ValueError(f"Unsupported threshold metric: {args.threshold_metric}")

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(
            model, train_loader, device=device, optimizer=optimizer
        )

        valid_true, valid_prob1, valid_loss = _collect_predictions(
            model, test_loader, device
        )
        tuned_stats = _find_best_threshold(
            valid_true,
            valid_prob1,
            metric=args.threshold_metric,
            min_precision=args.threshold_min_precision,
            num_candidates=args.threshold_candidates,
        )
        threshold_for_epoch = float(tuned_stats["threshold"])
        test_stats = _evaluate(
            model, test_loader, device=device, threshold=threshold_for_epoch
        )
        test_stats["threshold"] = threshold_for_epoch
        test_stats["test_loss_from_proba"] = valid_loss

        for key in ("loss", "accuracy", "recall"):
            history[f"train_{key}"].append(train_stats[key])
            history[f"test_{key}"].append(test_stats[key])
        history["test_precision"].append(test_stats["precision"])
        history["test_f1"].append(test_stats["f1"])
        history["test_threshold"].append(test_stats["threshold"])

        logger.info(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={train_stats['loss']:.4f} train_acc={train_stats['accuracy']:.4f} "
            f"train_recall={train_stats['recall']:.4f} "
            f"test_loss={test_stats['loss']:.4f} test_acc={test_stats['accuracy']:.4f} "
            f"test_recall={test_stats['recall']:.4f} test_precision={test_stats['precision']:.4f} "
            f"test_f1={test_stats['f1']:.4f} test_threshold={test_stats['threshold']:.4f}"
        )

        metric_value = test_stats.get(args.early_stop_metric)
        if metric_value is None:
            metric_value = test_stats["loss"]

        improved = False
        if best_is_higher:
            if metric_value > (best_score + args.min_delta):
                improved = True
        else:
            if metric_value < (best_score - args.min_delta):
                improved = True

        if improved:
            best_score = float(metric_value)
            best_epoch = epoch
            best_state = {
                k: v.detach().clone().cpu() for k, v in model.state_dict().items()
            }
            best_threshold = threshold_for_epoch
            epochs_without_improvement = 0
            logger.info(
                f"New best epoch={best_epoch} (metric {args.early_stop_metric}={best_score:.6f}, "
                f"threshold={best_threshold:.4f})"
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            logger.info(
                f"Early stopping triggered after {epochs_without_improvement} epochs without "
                f"improvement in {args.early_stop_metric}."
            )
            break

    if not best_state:
        best_state = model.state_dict()

    # Save best checkpoint and evaluation artifacts from the best validation epoch.
    model.load_state_dict(best_state)

    y_true_final, y_prob1_final, final_loss = _collect_predictions(
        model, test_loader, device
    )
    final_stats = _find_best_threshold(
        y_true_final,
        y_prob1_final,
        metric=args.threshold_metric,
        min_precision=args.threshold_min_precision,
        num_candidates=args.threshold_candidates,
    )
    if not math.isnan(final_loss):
        final_stats["loss"] = final_loss

    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "history": history,
            "branch_input_dims": tuple(input_dims[name] for name in feature_names),
            "branch_names": feature_names,
            "branch_hidden_dims": branch_widths,
            "final_hidden": final_hidden,
            "num_classes": num_classes,
            "seed": args.seed,
            "best_epoch": best_epoch,
            "best_threshold": float(final_stats["threshold"]),
            "best_score": best_score,
            "early_stop_metric": args.early_stop_metric,
            "threshold_search": {
                "metric": args.threshold_metric,
                "min_precision": args.threshold_min_precision,
                "num_candidates": args.threshold_candidates,
            },
            "category_maps": category_maps,
            "run_manifest": manifest.__dict__,
            "args": vars(args),
        },
        checkpoint_path,
    )
    last_checkpoint_path = output_dir / "last_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            },
            "history": history,
            "branch_input_dims": tuple(input_dims[name] for name in feature_names),
            "branch_names": feature_names,
            "branch_hidden_dims": branch_widths,
            "final_hidden": final_hidden,
            "num_classes": num_classes,
            "seed": args.seed,
            "best_epoch": best_epoch,
            "best_threshold": float(final_stats["threshold"]),
            "best_score": best_score,
            "early_stop_metric": args.early_stop_metric,
            "threshold_search": {
                "metric": args.threshold_metric,
                "min_precision": args.threshold_min_precision,
                "num_candidates": args.threshold_candidates,
            },
            "category_maps": category_maps,
            "run_manifest": manifest.__dict__,
            "args": vars(args),
        },
        last_checkpoint_path,
    )

    metadata = {
        "test_size": args.test_size,
        "test_random_state": args.test_random_state,
        "input_dims": {name: input_dims[name] for name in feature_names},
        "branch_names": feature_names,
        "branch_widths": branch_widths,
        "best_epoch": best_epoch,
        "best_threshold": float(final_stats["threshold"]),
        "best_score": best_score,
        "early_stop_metric": args.early_stop_metric,
        "threshold_search": {
            "metric": args.threshold_metric,
            "min_precision": args.threshold_min_precision,
            "num_candidates": args.threshold_candidates,
        },
        "final_hidden": final_hidden,
        "category_maps": category_maps,
        "run_manifest": manifest.__dict__,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Reuse the final inference pass to dump a detailed test report.
    y_pred = (
        (np.asarray(y_prob1_final) >= final_stats["threshold"]).astype(int).tolist()
    )
    report_path = output_dir / "eval_on_test.json"
    report_path.write_text(
        json.dumps(
            {
                "best_threshold": final_stats["threshold"],
                "confusion_matrix": final_stats["confusion_matrix"],
                "classification_report": classification_report(
                    y_true_final, y_pred, output_dict=True
                ),
                "history": history,
            },
            indent=2,
        )
    )

    logger.info(f"Saved checkpoint: {checkpoint_path}")
    logger.info(f"Saved metadata: {output_dir / 'metadata.json'}")
    logger.info(f"Saved test report: {report_path}")
    logger.info("Completed classifier training run.")


if __name__ == "__main__":
    main()
