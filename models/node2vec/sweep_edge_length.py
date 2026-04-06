"""
Hyperparameter sweep for edge-length node2vec training.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from models.node2vec.train import default_worker_count, train_node2vec


@dataclass(frozen=True)
class EdgeLengthRow:
    """
    A single directed weighted edge parsed from an edge-length list file.

    Attributes:
        start_node: Source node identifier.
        end_node: Destination node identifier.
        edge_length: Physical length of the edge (positive float).
    """

    start_node: str
    end_node: str
    edge_length: float


@dataclass(frozen=True)
class SweepConfig:
    """
    Hyperparameter configuration for a single node2vec training run.

    Attributes:
        name: Human-readable label used to identify this configuration in outputs.
        dimensions: Embedding dimensionality passed to node2vec.
        walk_length: Number of nodes per random walk.
        num_walks: Number of random walks starting from each node.
        window: Word2Vec context window size.
        p: Return parameter controlling the likelihood of revisiting a node.
        q: In-out parameter controlling breadth- vs. depth-first traversal bias.
    """

    name: str
    dimensions: int
    walk_length: int
    num_walks: int
    window: int
    p: float
    q: float


DEFAULT_CONFIGS: tuple[SweepConfig, ...] = (
    SweepConfig("d16_wl20_nw10_win5_p1_q1", 16, 20, 10, 5, 1.0, 1.0),
    SweepConfig("d16_wl40_nw10_win5_p1_q1", 16, 40, 10, 5, 1.0, 1.0),
    SweepConfig("d32_wl20_nw10_win5_p1_q1", 32, 20, 10, 5, 1.0, 1.0),
    SweepConfig("d32_wl40_nw10_win5_p1_q1", 32, 40, 10, 5, 1.0, 1.0),
    SweepConfig("d32_wl40_nw20_win10_p1_q1", 32, 40, 20, 10, 1.0, 1.0),
    SweepConfig("d32_wl40_nw20_win10_p0.5_q2", 32, 40, 20, 10, 0.5, 2.0),
    SweepConfig("d32_wl40_nw20_win10_p2_q0.5", 32, 40, 20, 10, 2.0, 0.5),
    SweepConfig("d64_wl20_nw10_win5_p1_q1", 64, 20, 10, 5, 1.0, 1.0),
    SweepConfig("d64_wl40_nw20_win10_p1_q1", 64, 40, 20, 10, 1.0, 1.0),
)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the edge-length sweep script.

    Returns:
        Parsed argument namespace with fields: input, output_dir, seed,
        test_ratio, negative_samples, workers, min_count, topn.
    """
    parser = argparse.ArgumentParser(
        description="Run an edge-length node2vec hyperparameter sweep."
    )
    parser.add_argument(
        "--input",
        default="data/interim/edge_length.list",
        help="Input edge-length edge list file.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/node2vec/edge_length",
        help="Directory for the selected model and evaluation artifacts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for split generation and training.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Held-out edge ratio for hyperparameter comparison.",
    )
    parser.add_argument(
        "--negative-samples",
        type=int,
        default=50,
        help="Negative targets sampled per held-out edge during evaluation.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_worker_count(),
        help="Node2Vec worker count (defaults to detected CPU cores).",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Word2Vec min_count parameter.",
    )
    parser.add_argument(
        "--topn",
        type=int,
        default=5,
        help="Nearest-neighbour count saved for the final model samples.",
    )
    return parser.parse_args()


def load_edge_length_rows(
    path: str | Path,
) -> list[EdgeLengthRow]:
    """
    Load directed weighted edges from a whitespace-delimited edge-length file.

    Each non-empty, non-comment line must contain at least three columns:
    ``<start_node> <end_node> <edge_length>``. Inline comments beginning with
    ``#`` are stripped before parsing.

    Args:
        path: Path to the edge-length list file.

    Returns:
        List of ``EdgeLengthRow`` objects, one per valid line, preserving
        file order (including duplicates).

    Raises:
        ValueError: If any line has fewer than three columns, if the third
            column cannot be parsed as a float, or if the file contains no
            valid rows.
    """
    rows: list[EdgeLengthRow] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_num, raw_line in enumerate(handle, start=1):
            line = raw_line.split("#", maxsplit=1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(
                    f"Invalid edge-length row at line {line_num}: expected 3 columns, got {len(parts)}."
                )
            try:
                edge_length = float(parts[2])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid edge length at line {line_num}: {parts[2]}"
                ) from exc
            rows.append(
                EdgeLengthRow(
                    start_node=parts[0], end_node=parts[1], edge_length=edge_length
                )
            )
    if not rows:
        raise ValueError(f"Input file is empty: {path}")
    return rows


def unique_rows(
    rows: Iterable[EdgeLengthRow],
) -> list[EdgeLengthRow]:
    """
    Deduplicate edges, keeping the last occurrence of each directed pair.

    Args:
        rows: Iterable of ``EdgeLengthRow`` objects, possibly containing
            duplicate ``(start_node, end_node)`` pairs.

    Returns:
        List of ``EdgeLengthRow`` objects with at most one entry per directed
        node pair. Insertion order of first-seen pairs is preserved.
    """
    deduped: dict[tuple[str, str], EdgeLengthRow] = {}
    for row in rows:
        deduped[(row.start_node, row.end_node)] = row
    return list(deduped.values())


def build_graph_stats(
    rows: Iterable[EdgeLengthRow],
) -> dict[str, float]:
    """
    Compute summary statistics for the edge-length graph.

    Args:
        rows: Iterable of ``EdgeLengthRow`` objects representing the full graph.

    Returns:
        Dictionary with the following keys:

        - ``edge_rows``: Total number of directed edges.
        - ``nodes``: Number of unique nodes.
        - ``min_length`` / ``max_length``: Minimum and maximum edge length.
        - ``mean_length`` / ``median_length``: Average and median edge length.
        - ``p95_length`` / ``p99_length``: 95th and 99th percentile edge lengths.
        - ``avg_out_degree``: Mean number of outgoing edges per source node.
        - ``avg_in_degree``: Mean number of incoming edges per destination node.
    """
    rows = list(rows)
    nodes = set()
    outgoing = Counter()
    incoming = Counter()
    lengths = []
    for row in rows:
        nodes.add(row.start_node)
        nodes.add(row.end_node)
        outgoing[row.start_node] += 1
        incoming[row.end_node] += 1
        lengths.append(row.edge_length)

    sorted_lengths = sorted(lengths)

    def percentile(p: int) -> float:
        index = max(
            0, min(len(sorted_lengths) - 1, int(len(sorted_lengths) * p / 100) - 1)
        )
        return sorted_lengths[index]

    return {
        "edge_rows": len(rows),
        "nodes": len(nodes),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "mean_length": float(np.mean(lengths)),
        "median_length": float(np.median(lengths)),
        "p95_length": percentile(95),
        "p99_length": percentile(99),
        "avg_out_degree": sum(outgoing.values()) / max(len(outgoing), 1),
        "avg_in_degree": sum(incoming.values()) / max(len(incoming), 1),
    }


def split_rows(
    rows: list[EdgeLengthRow],
    *,
    test_ratio: float,
    seed: int,
) -> tuple[list[EdgeLengthRow], list[EdgeLengthRow]]:
    """
    Split edges into train and test sets while preserving node coverage.

    Edges are shuffled deterministically then greedily assigned to the test
    set, skipping any edge whose removal would leave either endpoint isolated
    (degree <= 1 after accounting for already-assigned test edges).

    Args:
        rows: Full list of deduplicated directed edges.
        test_ratio: Target fraction of edges to hold out for evaluation.
        seed: Random seed for reproducible shuffling.

    Returns:
        A two-tuple ``(train_rows, test_rows)`` where ``train_rows`` contains
        all edges not selected for the test set.

    Raises:
        ValueError: If no edges can be safely moved to the test set without
            leaving a node unreachable in the training graph.
    """
    shuffled = list(rows)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    node_degree = Counter()
    for row in shuffled:
        node_degree[row.start_node] += 1
        node_degree[row.end_node] += 1

    target_test_size = max(1, int(len(shuffled) * test_ratio))
    test_rows: list[EdgeLengthRow] = []
    test_keys: set[tuple[str, str]] = set()
    for row in shuffled:
        if len(test_rows) >= target_test_size:
            break
        if node_degree[row.start_node] <= 1 or node_degree[row.end_node] <= 1:
            continue
        test_rows.append(row)
        test_keys.add((row.start_node, row.end_node))
        node_degree[row.start_node] -= 1
        node_degree[row.end_node] -= 1

    if not test_rows:
        raise ValueError(
            "Unable to create a non-empty held-out split while preserving train coverage."
        )

    train_rows = [
        row for row in shuffled if (row.start_node, row.end_node) not in test_keys
    ]
    return train_rows, test_rows


def sample_evaluation_cases(
    test_rows: list[EdgeLengthRow],
    *,
    all_rows: list[EdgeLengthRow],
    negative_samples: int,
    seed: int,
) -> list[dict[str, object]]:
    """
    Build ranked-retrieval evaluation cases from held-out edges.

    For each held-out (positive) edge, ``negative_samples`` distinct nodes are
    drawn uniformly at random from the full node set, excluding the source node
    and any node that is a known positive destination for that source.

    Args:
        test_rows: Held-out edges to use as positive examples.
        all_rows: Complete edge list used to determine the full node set and
            all known positive targets per source (to exclude from negatives).
        negative_samples: Number of negative candidate nodes per positive edge.
        seed: Random seed for reproducible negative sampling.

    Returns:
        List of case dictionaries, one per test edge, each containing:

        - ``source``: Source node identifier.
        - ``positive``: True destination node identifier.
        - ``negatives``: List of ``negative_samples`` sampled negative nodes.
        - ``edge_length``: Physical length of the positive edge.
    """
    rng = random.Random(seed)
    all_nodes = sorted(
        {row.start_node for row in all_rows} | {row.end_node for row in all_rows}
    )
    positive_targets_by_source: dict[str, set[str]] = defaultdict(set)
    for row in all_rows:
        positive_targets_by_source[row.start_node].add(row.end_node)

    cases: list[dict[str, object]] = []
    for row in test_rows:
        forbidden = positive_targets_by_source[row.start_node]
        negatives: list[str] = []
        while len(negatives) < negative_samples:
            candidate = all_nodes[rng.randrange(len(all_nodes))]
            if candidate == row.start_node or candidate in forbidden:
                continue
            negatives.append(candidate)
        cases.append(
            {
                "source": row.start_node,
                "positive": row.end_node,
                "negatives": negatives,
                "edge_length": row.edge_length,
            }
        )
    return cases


def _build_embedding_index(
    model: object,
) -> tuple[dict[str, int], np.ndarray]:
    """
    Extract a token-to-index map and L2-normalised embedding matrix from a model.

    Args:
        model: A trained node2vec / Word2Vec model or its ``KeyedVectors`` object.

    Returns:
        A two-tuple ``(key_to_index, normed_vectors)`` where ``key_to_index``
        maps each node string to its row index in ``normed_vectors``, and
        ``normed_vectors`` is a ``(vocab_size, dim)`` float32 array of
        unit-length embedding vectors.
    """
    keyed_vectors = model.wv if hasattr(model, "wv") else model
    return keyed_vectors.key_to_index, keyed_vectors.get_normed_vectors()


def _score_pair(
    key_to_index: dict[str, int],
    normed_vectors: np.ndarray,
    source: str,
    target: str,
) -> float:
    """
    Compute the cosine similarity between two node embeddings.

    Args:
        key_to_index: Mapping from node string to its row index in
            ``normed_vectors``.
        normed_vectors: L2-normalised embedding matrix of shape
            ``(vocab_size, dim)``.
        source: Source node identifier.
        target: Target node identifier.

    Returns:
        Cosine similarity score in ``[-1, 1]``.
    """
    return float(
        np.dot(
            normed_vectors[key_to_index[source]], normed_vectors[key_to_index[target]]
        )
    )


def _roc_auc(
    scores: list[float],
    labels: list[int],
) -> float:
    """
    Compute the ROC-AUC score using the Wilcoxon rank-sum formula.

    Args:
        scores: Predicted similarity scores, one per sample.
        labels: Binary ground-truth labels (1 = positive, 0 = negative),
            aligned with ``scores``.

    Returns:
        ROC-AUC value in ``[0, 1]``.
    """
    ranked = sorted(zip(scores, labels), key=lambda item: item[0])
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    positive_rank_sum = 0.0
    for rank, (_, label) in enumerate(ranked, start=1):
        if label == 1:
            positive_rank_sum += rank
    return (positive_rank_sum - positive_count * (positive_count + 1) / 2.0) / (
        positive_count * negative_count
    )


def evaluate_model(
    model,
    cases: list[dict[str, object]],
) -> dict[str, float]:
    """
    Evaluate a trained node2vec model on ranked-retrieval cases.

    For each case the positive node is ranked against its sampled negatives by
    cosine similarity. Aggregate retrieval and discrimination metrics are then
    computed over all cases.

    Args:
        model: Trained node2vec / Word2Vec model or its ``KeyedVectors`` object.
        cases: Evaluation cases as returned by ``sample_evaluation_cases``.

    Returns:
        Dictionary of scalar evaluation metrics:

        - ``auc``: ROC-AUC over all positive/negative score pairs.
        - ``mrr``: Mean Reciprocal Rank of the positive node.
        - ``hit_at_1`` / ``hit_at_5`` / ``hit_at_10``: Fraction of cases where
          the positive ranks within the top 1 / 5 / 10 candidates.
        - ``positive_mean_cosine``: Mean cosine similarity for positive pairs.
        - ``negative_mean_cosine``: Mean cosine similarity for negative pairs.
        - ``score_margin``: Difference between positive and negative mean cosines.
        - ``positive_score_length_corr``: Pearson correlation between positive
          cosine scores and their corresponding edge lengths.
    """
    key_to_index, normed_vectors = _build_embedding_index(model)

    positive_scores: list[float] = []
    negative_scores: list[float] = []
    reciprocal_ranks: list[float] = []
    lengths: list[float] = []
    hit_at_1 = 0
    hit_at_5 = 0
    hit_at_10 = 0

    for case in cases:
        source = case["source"]
        positive = case["positive"]
        negatives = case["negatives"]

        positive_score = _score_pair(key_to_index, normed_vectors, source, positive)
        positive_scores.append(positive_score)
        lengths.append(float(case["edge_length"]))
        sampled_scores = [(positive, positive_score)]

        for negative in negatives:
            negative_score = _score_pair(key_to_index, normed_vectors, source, negative)
            negative_scores.append(negative_score)
            sampled_scores.append((negative, negative_score))

        sampled_scores.sort(key=lambda item: item[1], reverse=True)
        positive_rank = next(
            rank
            for rank, (target, _) in enumerate(sampled_scores, start=1)
            if target == positive
        )
        reciprocal_ranks.append(1.0 / positive_rank)
        if positive_rank <= 1:
            hit_at_1 += 1
        if positive_rank <= 5:
            hit_at_5 += 1
        if positive_rank <= 10:
            hit_at_10 += 1

    all_scores = positive_scores + negative_scores
    all_labels = [1] * len(positive_scores) + [0] * len(negative_scores)

    corr = (
        float(np.corrcoef(positive_scores, lengths)[0, 1])
        if len(positive_scores) > 1
        else 0.0
    )
    if np.isnan(corr):
        corr = 0.0

    return {
        "auc": _roc_auc(all_scores, all_labels),
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "hit_at_1": hit_at_1 / len(cases),
        "hit_at_5": hit_at_5 / len(cases),
        "hit_at_10": hit_at_10 / len(cases),
        "positive_mean_cosine": float(np.mean(positive_scores)),
        "negative_mean_cosine": float(np.mean(negative_scores)),
        "score_margin": float(np.mean(positive_scores) - np.mean(negative_scores)),
        "positive_score_length_corr": corr,
    }


def write_rows(
    path: Path,
    rows: Iterable[EdgeLengthRow],
) -> None:
    """
    Write edge-length rows to a whitespace-delimited file.

    Parent directories are created automatically if they do not exist.

    Args:
        path: Destination file path.
        rows: Edge rows to serialise, each written as
            ``<start_node> <end_node> <edge_length>\\n``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f"{row.start_node} {row.end_node} {row.edge_length}\n")


def run_sweep(
    input_path: Path,
    *,
    output_dir: Path,
    seed: int,
    test_ratio: float,
    negative_samples: int,
    workers: int,
    min_count: int,
    topn: int,
) -> None:
    """
    Run the full edge-length node2vec hyperparameter sweep and persist outputs.

    Loads the edge-length graph, creates a train/test split, evaluates every
    configuration in ``DEFAULT_CONFIGS``, selects the best by AUC > MRR >
    Hit@10 > score margin, retrains on the full dataset, and writes all
    artifacts to ``output_dir``.

    Args:
        input_path: Path to the edge-length edge list file.
        output_dir: Directory where all output artifacts are written.
        seed: Random seed for split generation, negative sampling, and training.
        test_ratio: Fraction of edges reserved for evaluation.
        negative_samples: Number of negative nodes sampled per positive edge
            during evaluation.
        workers: Number of worker threads for node2vec training.
        min_count: Word2Vec ``min_count`` parameter (minimum node frequency).
        topn: Number of nearest neighbours to save for each sample node in
            ``best_model_neighbors.json``.
    """
    all_rows = unique_rows(load_edge_length_rows(input_path))
    graph_stats = build_graph_stats(all_rows)
    train_rows, test_rows = split_rows(all_rows, test_ratio=test_ratio, seed=seed)
    cases = sample_evaluation_cases(
        test_rows,
        all_rows=all_rows,
        negative_samples=negative_samples,
        seed=seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        train_split_path = Path(tmpdir) / "edge_length_train_split.list"
        write_rows(train_split_path, train_rows)

        for index, config in enumerate(DEFAULT_CONFIGS, start=1):
            print(f"[{index}/{len(DEFAULT_CONFIGS)}] training {config.name}")
            model, _, _ = train_node2vec(
                train_split_path,
                graph_format="edgelist",
                weight_transform="none",
                dimensions=config.dimensions,
                walk_length=config.walk_length,
                num_walks=config.num_walks,
                window=config.window,
                min_count=min_count,
                workers=workers,
                p=config.p,
                q=config.q,
                seed=seed,
            )
            metrics = evaluate_model(model, cases)
            print(
                "  "
                f"auc={metrics['auc']:.4f} "
                f"mrr={metrics['mrr']:.4f} "
                f"hit@10={metrics['hit_at_10']:.4f} "
                f"margin={metrics['score_margin']:.4f}"
            )
            result = asdict(config)
            result.update(metrics)
            results.append(result)

    results.sort(
        key=lambda row: (
            row["auc"],
            row["mrr"],
            row["hit_at_10"],
            row["score_margin"],
        ),
        reverse=True,
    )
    best_result = results[0]
    best_config = SweepConfig(
        name=str(best_result["name"]),
        dimensions=int(best_result["dimensions"]),
        walk_length=int(best_result["walk_length"]),
        num_walks=int(best_result["num_walks"]),
        window=int(best_result["window"]),
        p=float(best_result["p"]),
        q=float(best_result["q"]),
    )

    best_model_path = output_dir / "best_model.n2v"
    print(f"retraining best configuration on full dataset: {best_config.name}")
    final_model, _, _ = train_node2vec(
        input_path,
        graph_format="edgelist",
        weight_transform="none",
        dimensions=best_config.dimensions,
        walk_length=best_config.walk_length,
        num_walks=best_config.num_walks,
        window=best_config.window,
        min_count=min_count,
        workers=workers,
        p=best_config.p,
        q=best_config.q,
        seed=seed,
        output=best_model_path,
    )
    print(f"saved best model to {best_model_path}")

    sample_nodes = [
        node for node, _ in Counter(row.start_node for row in all_rows).most_common(3)
    ]
    nearest_neighbors = {}
    for node in sample_nodes:
        nearest_neighbors[node] = [
            {"node": other, "score": float(score)}
            for other, score in final_model.wv.similar_by_word(node, topn=topn)
        ]

    sweep_results_path = output_dir / "sweep_results.json"
    best_eval_path = output_dir / "best_eval.json"
    config_path = output_dir / "selected_config.json"
    csv_path = output_dir / "sweep_results.csv"
    neighbors_path = output_dir / "best_model_neighbors.json"
    report_path = output_dir / "report.md"

    sweep_payload = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "seed": seed,
        "test_ratio": test_ratio,
        "negative_samples": negative_samples,
        "workers": workers,
        "min_count": min_count,
        "weight_transform": "none",
        "graph_stats": graph_stats,
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "configs": results,
    }
    sweep_results_path.write_text(json.dumps(sweep_payload, indent=2), encoding="utf-8")
    best_eval_path.write_text(
        json.dumps(
            {
                "selection_metric_priority": [
                    "auc",
                    "mrr",
                    "hit_at_10",
                    "score_margin",
                ],
                "weight_transform": "none",
                "best_result": best_result,
                "graph_stats": graph_stats,
                "train_rows": len(train_rows),
                "test_rows": len(test_rows),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    config_path.write_text(json.dumps(asdict(best_config), indent=2), encoding="utf-8")
    neighbors_path.write_text(json.dumps(nearest_neighbors, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "name",
            "dimensions",
            "walk_length",
            "num_walks",
            "window",
            "p",
            "q",
            "auc",
            "mrr",
            "hit_at_1",
            "hit_at_5",
            "hit_at_10",
            "positive_mean_cosine",
            "negative_mean_cosine",
            "score_margin",
            "positive_score_length_corr",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    report_lines = [
        "# Node2Vec Edge-Length Sweep Report",
        "",
        "## Dataset",
        f"- Input file: `{input_path}`",
        f"- Unique directed edge rows: {graph_stats['edge_rows']}",
        f"- Nodes: {graph_stats['nodes']}",
        f"- Length range: {graph_stats['min_length']:.6f} to {graph_stats['max_length']:.6f}",
        f"- Mean length: {graph_stats['mean_length']:.4f}",
        f"- Median length: {graph_stats['median_length']:.4f}",
        f"- Long-tail summary: p95={graph_stats['p95_length']:.4f}, p99={graph_stats['p99_length']:.4f}",
        "- Training interpretation: each row `<start> <end> <length>` is a directed weighted edge.",
        "- Weight transform: none. Edge lengths are already positive and valid for node2vec.",
        "",
        "## Evaluation Protocol",
        f"- Hold-out split: {len(train_rows)} train edges / {len(test_rows)} test edges",
        f"- Random seed: {seed}",
        f"- Negative samples per positive: {negative_samples}",
        "- Selection priority: AUC, then MRR, then Hit@10, then score margin",
        "",
        "## Ranked Results",
        "",
        "| Rank | Config | AUC | MRR | Hit@1 | Hit@5 | Hit@10 | Margin | Corr |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(results, start=1):
        report_lines.append(
            "| {rank} | {name} | {auc:.4f} | {mrr:.4f} | {hit_at_1:.4f} | {hit_at_5:.4f} | "
            "{hit_at_10:.4f} | {score_margin:.4f} | {positive_score_length_corr:.4f} |".format(
                rank=rank,
                **row,
            )
        )

    report_lines.extend(
        [
            "",
            "## Selected Configuration",
            f"- Config: `{best_config.name}`",
            f"- dimensions={best_config.dimensions}, walk_length={best_config.walk_length}, "
            f"num_walks={best_config.num_walks}, window={best_config.window}, "
            f"p={best_config.p}, q={best_config.q}",
            f"- AUC={best_result['auc']:.4f}, MRR={best_result['mrr']:.4f}, "
            f"Hit@10={best_result['hit_at_10']:.4f}, score_margin={best_result['score_margin']:.4f}",
            f"- Positive cosine vs edge-length correlation: {best_result['positive_score_length_corr']:.4f}",
            "- Final delivery model: retrained on the full input file with the selected configuration.",
            "",
            "## Output Files",
            "- `best_model.n2v`: final model retrained on the full edge-length dataset",
            "- `best_eval.json`: best run metrics and selection metadata",
            "- `selected_config.json`: chosen hyperparameters",
            "- `sweep_results.json` / `sweep_results.csv`: complete comparison across all runs",
            "- `best_model_neighbors.json`: sample nearest neighbours from the final model",
        ]
    )
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def main() -> None:
    """
    Entry point: parse CLI arguments and delegate to ``run_sweep``.
    """
    args = parse_args()
    run_sweep(
        Path(args.input),
        output_dir=Path(args.output_dir),
        seed=args.seed,
        test_ratio=args.test_ratio,
        negative_samples=args.negative_samples,
        workers=args.workers,
        min_count=args.min_count,
        topn=args.topn,
    )


if __name__ == "__main__":
    main()
