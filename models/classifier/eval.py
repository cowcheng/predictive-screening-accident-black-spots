"""
Evaluate a trained PyTorch accident classifier checkpoint.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
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
from torch.utils.data import DataLoader

from models.classifier.model import AccidentClassifier
from models.classifier.train import (
    DEFAULT_BRANCH_NAMES,
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUTS_ROOT,
    EDGE_N2V_FEATURES,
    LEGACY_BRANCH_NAMES,
    AccidentDataset,
    _ensure_file_exists,
    _resolve_path,
    _resolve_root,
    _validate_and_log_embeddings,
    load_dataset,
)

# --------------------------------------------------------------------------- #
# Checkpoint path resolution
# --------------------------------------------------------------------------- #


def _build_embedding_paths(
    args: argparse.Namespace,
    manifest: dict,
    outputs_root: Path,
) -> dict[str, Path]:
    """
    Resolve each node2vec embedding path from CLI, manifest, then defaults.

    Args:
        args: Parsed CLI namespace containing ``{feature}_embeddings`` overrides.
        manifest: Checkpoint ``run_manifest`` dict; its ``embeddings`` field is
            consulted when the CLI override is absent.
        outputs_root: Base directory used to form default paths.

    Returns:
        Mapping from embedding key (e.g. ``"maxspeed"``) to resolved path.
    """
    cli_overrides = {
        "maxspeed": args.maxspeed_embeddings,
        "time": args.time_embeddings,
        "length": args.length_embeddings,
        "lanes": args.lanes_embeddings,
        "ref": args.ref_embeddings,
    }
    manifest_embeddings = (
        manifest.get("embeddings", {}) if isinstance(manifest, dict) else {}
    )

    resolved: dict[str, Path] = {}
    for key in EDGE_N2V_FEATURES:
        default = outputs_root / "node2vec" / f"edge_{key}" / "best_model_edges.npy"
        override = cli_overrides[key]
        if override is None and key in manifest_embeddings:
            override = str(manifest_embeddings[key])
        resolved[key] = _resolve_path(override, outputs_root, default)
    return resolved


def _resolve_edge_info_path(
    args: argparse.Namespace,
    manifest: dict,
    data_root: Path,
) -> Path:
    """
    Resolve the edge metadata CSV from CLI override, manifest, or default.

    Args:
        args: Parsed CLI namespace.
        manifest: Checkpoint ``run_manifest`` dict; its ``edge_info_csv`` field
            is consulted when the CLI override is absent.
        data_root: Base directory used to form the default path.

    Returns:
        Resolved edge metadata CSV path.
    """
    default = data_root / "final" / "osm_edge_info_completed.csv"
    if args.edge_info_csv is not None:
        return _resolve_path(args.edge_info_csv, data_root, default)
    if isinstance(manifest, dict) and manifest.get("edge_info_csv"):
        return _resolve_path(str(manifest["edge_info_csv"]), data_root, default)
    return default


# --------------------------------------------------------------------------- #
# Checkpoint metadata helpers
# --------------------------------------------------------------------------- #


def _as_dict(
    value: object,
) -> dict:
    """
    Return ``value`` when it is a dict, otherwise an empty dict.

    Args:
        value: Any object to coerce into a dict.

    Returns:
        ``value`` unchanged when it is already a dict; an empty dict otherwise.
    """
    return value if isinstance(value, dict) else {}


def _extract_manifest(
    ckpt: dict,
    metadata: dict,
) -> dict:
    """
    Return the first dict-like manifest from ``ckpt`` or ``metadata``.

    The checkpoint format historically stored the manifest under several keys;
    this helper transparently picks whichever one is present.

    Args:
        ckpt: Raw checkpoint dict loaded from a ``.pt`` file.
        metadata: Auxiliary metadata dict, typically ``ckpt["metadata"]``.

    Returns:
        The first dict-valued manifest found across the candidate keys, or an
        empty dict when none is present.
    """
    for candidate in (
        ckpt.get("run_manifest"),
        ckpt.get("manifest"),
        metadata.get("run_manifest"),
    ):
        if isinstance(candidate, dict):
            return candidate
    return {}


def _resolve_request_branches(
    branch_input_dims: tuple[int, ...],
    manifest: dict,
) -> list[str]:
    """
    Return the branch name list expected by a checkpoint.

    Uses the manifest's ``branch_names`` when present, otherwise infers the
    layout from the number of input dims using :data:`DEFAULT_BRANCH_NAMES`
    or :data:`LEGACY_BRANCH_NAMES`.

    Args:
        branch_input_dims: Branch input widths stored in the checkpoint.
        manifest: Manifest dict extracted from the checkpoint.

    Returns:
        Ordered list of branch names for dataset loading and model construction.
    """
    branch_names = manifest.get("branch_names")
    if isinstance(branch_names, (list, tuple)):
        return list(branch_names)

    if len(branch_input_dims) == len(DEFAULT_BRANCH_NAMES):
        return list(DEFAULT_BRANCH_NAMES)
    if len(branch_input_dims) == len(LEGACY_BRANCH_NAMES):
        return list(LEGACY_BRANCH_NAMES)

    raise ValueError(
        "Cannot infer branch layout from checkpoint. "
        "Use a retrained checkpoint with branch_names metadata or pass matching data loading options."
    )


def _make_logger(
    log_path: Path,
) -> logging.Logger:
    """
    Create a logger that writes INFO logs to both stdout and ``log_path``.

    Args:
        log_path: File path for the log output; opened in write (``"w"``) mode.

    Returns:
        Configured :class:`logging.Logger` named ``"classifier_eval"``.
    """
    logger = logging.getLogger("classifier_eval")
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


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def _evaluate_loader(
    model: AccidentClassifier,
    loader: DataLoader,
    device: str,
    threshold: float = 0.5,
) -> dict:
    """
    Run inference and return detailed metrics plus raw predictions.

    Args:
        model: Trained classifier to evaluate.
        loader: DataLoader producing ``(branch_features, labels)`` batches.
        device: Target device string passed to ``tensor.to``.
        threshold: Minimum class-1 probability required to predict positive.

    Returns:
        Dict with two keys:
          * ``metrics`` — loss, accuracy, recall, precision, F1, confusion
            matrix, and classification report.
          * ``predictions`` — ``y_true``, ``y_pred`` and class-1 probabilities
            ``y_prob1``.
    """
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob1: list[float] = []
    losses: list[float] = []

    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_features, batch_labels in loader:
            feature_tensors = [t.to(device) for t in batch_features]
            labels = batch_labels.to(device)
            logits = model(feature_tensors)
        losses.append(float(criterion(logits, labels).item()))
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.where(probs >= threshold, torch.tensor(1), torch.tensor(0))
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())
        y_prob1.extend(probs.detach().cpu().numpy().tolist())

    metrics = {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0.0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0.0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0.0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0.0
        ),
    }
    return {
        "metrics": metrics,
        "predictions": {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_prob1": y_prob1,
        },
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the evaluation entry point.

    Returns:
        Populated :class:`argparse.Namespace` with all CLI flags resolved.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate accident classifier checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint.pt generated by train.py.",
    )
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
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--log-file", default=None, help="Optional path to evaluation log output."
    )
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--test-size", type=float, default=None)
    parser.add_argument("--test-random-state", type=int, default=None)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override decision threshold for class 1.",
    )
    parser.add_argument(
        "--calibrate-threshold",
        action="store_true",
        help="Search for the best class-1 threshold on this evaluation split.",
    )
    parser.add_argument(
        "--threshold-metric",
        default="f1",
        choices=["accuracy", "precision", "recall", "f1"],
        help="Metric used when --calibrate-threshold is enabled.",
    )
    parser.add_argument(
        "--threshold-min-precision",
        type=float,
        default=0.0,
        help="Minimum precision allowed during threshold search.",
    )
    parser.add_argument(
        "--threshold-candidates",
        type=int,
        default=1001,
        help="Number of candidate thresholds checked when calibrating.",
    )
    return parser.parse_args()


def _find_best_threshold(
    y_true: list[int],
    y_prob1: list[float],
    metric: str = "f1",
    min_precision: float = 0.0,
    num_candidates: int = 1001,
) -> dict[str, float | list[list[int]]]:
    """
    Search a fixed grid of class-1 probabilities for the best threshold.

    Args:
        y_true: Ground-truth binary labels.
        y_prob1: Predicted class-1 probabilities, one per sample.
        metric: Optimisation target; one of ``"accuracy"``, ``"precision"``,
            ``"recall"``, or ``"f1"``.
        min_precision: Candidate thresholds whose precision falls below this
            value are skipped.
        num_candidates: Number of evenly-spaced thresholds evaluated in
            ``[0, 1]``.

    Returns:
        Dict with keys ``"threshold"``, ``"accuracy"``, ``"recall"``,
        ``"precision"``, ``"f1"``, and ``"confusion_matrix"`` for the best
        candidate found.
    """
    if not y_prob1:
        return {
            "threshold": 0.5,
            "accuracy": 0.0,
            "recall": 0.0,
            "precision": 0.0,
            "f1": 0.0,
            "confusion_matrix": [[0, 0], [0, 0]],
        }

    y_prob = np.asarray(y_prob1, dtype=np.float32)
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    thresholds = np.linspace(0.0, 1.0, int(num_candidates))
    best_score = -1.0
    best_threshold = 0.5
    best_metrics = {
        "accuracy": 0.0,
        "recall": 0.0,
        "precision": 0.0,
        "f1": 0.0,
        "confusion_matrix": [[0, 0], [0, 0]],
    }

    for threshold in thresholds:
        pred = (y_prob >= threshold).astype(np.int64)
        accuracy = float(accuracy_score(y_true_arr, pred))
        recall = float(recall_score(y_true_arr, pred, zero_division=0.0))
        precision = float(precision_score(y_true_arr, pred, zero_division=0.0))
        if precision < min_precision:
            continue
        f1 = float(f1_score(y_true_arr, pred, zero_division=0.0))

        if metric == "accuracy":
            score = accuracy
        elif metric == "precision":
            score = precision
        elif metric == "recall":
            score = recall
        elif metric == "f1":
            score = f1
        else:
            raise ValueError(f"Unsupported threshold metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = {
                "accuracy": accuracy,
                "recall": recall,
                "precision": precision,
                "f1": f1,
                "confusion_matrix": confusion_matrix(y_true_arr, pred).tolist(),
            }

    return {"threshold": best_threshold, **best_metrics}


def main() -> None:
    """
    CLI entry point: evaluate a checkpoint on its held-out test split.
    """
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    ckpt = torch.load(checkpoint_path, map_location=args.device)

    if "model_state_dict" not in ckpt:
        raise ValueError("Checkpoint missing model_state_dict.")

    metadata = _as_dict(ckpt.get("metadata"))
    train_args = _as_dict(ckpt.get("args"))
    manifest = _extract_manifest(ckpt, metadata)

    category_maps = ckpt.get("category_maps")
    if not isinstance(category_maps, dict):
        metadata_maps = metadata.get("category_maps")
        category_maps = metadata_maps if isinstance(metadata_maps, dict) else None

    outputs_root = _resolve_root(args.outputs_root, DEFAULT_OUTPUTS_ROOT)
    data_root = _resolve_root(args.data_root, DEFAULT_DATA_ROOT)
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = (
        Path(args.log_file) if args.log_file is not None else (output_dir / "eval.log")
    )
    logger = _make_logger(log_path)
    logger.info("Starting classifier evaluation.")
    logger.info(f"Arguments: {vars(args)}")

    embeddings = _build_embedding_paths(args, manifest, outputs_root)

    if args.legacy_feature_dir:
        legacy_feature_dir: Path | None = _resolve_path(
            args.legacy_feature_dir, data_root, Path(args.legacy_feature_dir)
        )
        _ensure_file_exists(legacy_feature_dir, "legacy feature directory")
        if not legacy_feature_dir.is_dir():
            raise ValueError(
                f"legacy feature directory is not a directory: {legacy_feature_dir}"
            )
        edge_info_csv: Path | None = None
    else:
        legacy_feature_dir = None
        edge_info_csv = _resolve_edge_info_path(args, manifest, data_root)
        _ensure_file_exists(edge_info_csv, "edge info CSV")

    _validate_and_log_embeddings(embeddings, logger)

    branch_input_dims = ckpt.get("branch_input_dims")
    branch_hidden_dims = ckpt.get("branch_hidden_dims")
    if branch_input_dims is None or branch_hidden_dims is None:
        raise ValueError("Checkpoint is missing branch architecture metadata.")
    branch_input_dims = tuple(int(v) for v in branch_input_dims)
    branch_hidden_dims = tuple(int(v) for v in branch_hidden_dims)

    requested_branch_names = _resolve_request_branches(branch_input_dims, manifest)
    if len(requested_branch_names) != len(branch_input_dims):
        raise ValueError(
            "Checkpoint branch metadata is inconsistent with requested branch names."
        )
    if len(requested_branch_names) != len(branch_hidden_dims):
        raise ValueError(
            "Checkpoint branch metadata is inconsistent with branch_hidden_dims."
        )

    features, labels, _, _, _, loaded_branch_names = load_dataset(
        embeddings=embeddings,
        legacy_feature_dir=legacy_feature_dir,
        edge_info_csv=edge_info_csv,
        category_maps=category_maps,
        requested_branch_names=requested_branch_names,
    )
    if loaded_branch_names != requested_branch_names:
        raise ValueError(
            "Checkpoint expects a different branch layout than the requested dataset configuration."
        )

    model = AccidentClassifier(
        branch_input_dims=branch_input_dims,
        branch_hidden_dims=branch_hidden_dims,
        final_hidden=int(ckpt.get("final_hidden", 8)),
        num_classes=int(ckpt.get("num_classes", 2)),
        branch_names=requested_branch_names,
    ).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_size = (
        args.test_size
        if args.test_size is not None
        else float(train_args.get("test_size", 0.3))
    )
    test_random_state = (
        args.test_random_state
        if args.test_random_state is not None
        else int(train_args.get("test_random_state", 42))
    )

    _, test_idx = train_test_split(
        np.arange(len(labels)),
        test_size=test_size,
        random_state=test_random_state,
        shuffle=True,
    )

    test_loader = DataLoader(
        AccidentDataset([f[test_idx] for f in features], labels[test_idx]),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    y_true_for_calibration = np.array([])
    y_prob_for_calibration = np.array([])
    if args.calibrate_threshold:
        model.eval()
        calibrate_true: list[int] = []
        calibrate_prob1: list[float] = []
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                feature_tensors = [t.to(args.device) for t in batch_features]
                labels = batch_labels.to(args.device)
                logits = model(feature_tensors)
                probs = torch.softmax(logits, dim=1)[:, 1]
                calibrate_true.extend(labels.detach().cpu().numpy().tolist())
                calibrate_prob1.extend(probs.detach().cpu().numpy().tolist())
        y_true_for_calibration = np.asarray(calibrate_true)
        y_prob_for_calibration = np.asarray(calibrate_prob1)

    threshold = args.threshold
    threshold_source = "user"
    if threshold is None:
        ckpt_threshold = ckpt.get("best_threshold")
        if ckpt_threshold is None:
            threshold = 0.5
            threshold_source = "default"
        else:
            threshold = float(ckpt_threshold)
            threshold_source = "checkpoint"

    if args.calibrate_threshold:
        threshold_search = _find_best_threshold(
            y_true_for_calibration.tolist(),
            y_prob_for_calibration.tolist(),
            metric=args.threshold_metric,
            min_precision=args.threshold_min_precision,
            num_candidates=args.threshold_candidates,
        )
        threshold = float(threshold_search["threshold"])
        threshold_source = "calibrated"
    else:
        threshold_search = None

    eval_result = _evaluate_loader(
        model, test_loader, device=args.device, threshold=threshold
    )

    report_file = output_dir / "eval_results.json"
    payload = {
        "threshold": float(threshold),
        "threshold_source": threshold_source,
        "metrics": eval_result["metrics"],
    }
    if threshold_search is not None:
        payload["threshold_search"] = threshold_search
    report_file.write_text(json.dumps(payload, indent=2))

    if args.save_predictions:
        pred_file = output_dir / "predictions.json"
        pred_file.write_text(json.dumps(eval_result["predictions"], indent=2))
        logger.info(f"Predictions saved to: {pred_file}")

    metrics = eval_result["metrics"]
    logger.info(f"Threshold source: {threshold_source}")
    logger.info(f"Threshold used: {threshold:.4f}")
    logger.info("Evaluation metrics:")
    logger.info(f"loss: {metrics['loss']:.4f}")
    logger.info(f"accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"recall (class 1): {metrics['recall']:.4f}")
    logger.info(f"precision (class 1): {metrics['precision']:.4f}")
    logger.info(f"f1 (class 1): {metrics['f1']:.4f}")
    logger.info(f"confusion_matrix: {metrics['confusion_matrix']}")
    logger.info(f"report saved to: {report_file}")
    logger.info("Completed classifier evaluation.")


if __name__ == "__main__":
    main()
