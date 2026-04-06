"""
Hyperparameter sweep for edge-ref node2vec training.
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
class EdgeRefRow:
    """
    Represents a single directed edge with a road reference classification weight.

    Attributes:
        start_node: Source node identifier.
        end_node: Destination node identifier.
        ref_value: Raw road reference class value (e.g. 0, 1, 3, 5, 6).
    """

    start_node: str
    end_node: str
    ref_value: float

    @property
    def transformed_weight(
        self,
    ) -> float:
        """
        Absolute ref value offset by a small epsilon to keep zero-ref edges valid.

        Returns:
            float: ``abs(ref_value) + 1e-3``.
        """
        return abs(self.ref_value) + 1e-3


@dataclass(frozen=True)
class SweepConfig:
    """
    Immutable hyperparameter configuration for a single node2vec sweep run.

    Attributes:
        name: Human-readable identifier for the configuration.
        dimensions: Embedding vector dimensionality.
        walk_length: Number of steps per random walk.
        num_walks: Number of random walks per node.
        window: Word2Vec context window size.
        p: Return parameter controlling the likelihood of revisiting a node.
        q: In-out parameter controlling DFS vs BFS walk bias.
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
    Parse command-line arguments for the sweep script.

    Returns:
        argparse.Namespace: Parsed arguments with fields ``input``, ``output_dir``,
            ``seed``, ``test_ratio``, ``negative_samples``, ``workers``,
            ``min_count``, and ``topn``.
    """
    parser = argparse.ArgumentParser(
        description="Run an edge-ref node2vec hyperparameter sweep."
    )
    parser.add_argument(
        "--input",
        default="data/interim/edge_ref.list",
        help="Input edge-ref edge list file.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/node2vec/edge_ref",
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


def load_edge_ref_rows(
    path: str | Path,
) -> list[EdgeRefRow]:
    """
    Load directed weighted edges from a whitespace-delimited edge-ref file.

    Each non-comment line must contain ``<start_node> <end_node> <ref_value>``.
    Lines beginning with or containing ``#`` have the comment portion stripped.

    Args:
        path: Path to the edge-ref ``.list`` file.

    Returns:
        list[EdgeRefRow]: Parsed rows in file order.

    Raises:
        ValueError: If any row has fewer than three columns, if ``ref_value``
            cannot be parsed as a float, or if the file contains no data rows.
    """
    rows: list[EdgeRefRow] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_num, raw_line in enumerate(handle, start=1):
            line = raw_line.split("#", maxsplit=1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(
                    f"Invalid edge-ref row at line {line_num}: expected 3 columns, got {len(parts)}."
                )
            try:
                ref_value = float(parts[2])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid ref value at line {line_num}: {parts[2]}"
                ) from exc
            rows.append(
                EdgeRefRow(start_node=parts[0], end_node=parts[1], ref_value=ref_value)
            )
    if not rows:
        raise ValueError(f"Input file is empty: {path}")
    return rows


def unique_rows(
    rows: Iterable[EdgeRefRow],
) -> list[EdgeRefRow]:
    """
    Deduplicate edges, keeping the last occurrence of each ``(start, end)`` pair.

    Args:
        rows: Iterable of EdgeRefRow objects, potentially containing duplicate
            directed edges.

    Returns:
        list[EdgeRefRow]: Deduplicated rows with insertion-order preserved for
            unique keys (last-write wins on duplicates).
    """
    deduped: dict[tuple[str, str], EdgeRefRow] = {}
    for row in rows:
        deduped[(row.start_node, row.end_node)] = row
    return list(deduped.values())


def build_graph_stats(
    rows: Iterable[EdgeRefRow],
) -> dict[str, float]:
    """
    Compute summary statistics for an edge-ref graph.

    Args:
        rows: Iterable of EdgeRefRow objects representing the full graph.

    Returns:
        dict[str, float]: Mapping of statistic name to value, including node
            and edge counts, raw ref range and mean, transformed weight mean,
            per-value ref distribution counts, and average in/out degrees.
    """
    rows = list(rows)
    nodes = set()
    outgoing = Counter()
    incoming = Counter()
    refs = []
    transformed = []
    for row in rows:
        nodes.add(row.start_node)
        nodes.add(row.end_node)
        outgoing[row.start_node] += 1
        incoming[row.end_node] += 1
        refs.append(row.ref_value)
        transformed.append(row.transformed_weight)

    distribution = Counter(refs)
    return {
        "edge_rows": len(rows),
        "nodes": len(nodes),
        "min_ref": min(refs),
        "max_ref": max(refs),
        "mean_ref": float(np.mean(refs)),
        "mean_transformed_ref": float(np.mean(transformed)),
        "zero_ref_edges": distribution.get(0.0, 0),
        "ref_1": distribution.get(1.0, 0),
        "ref_3": distribution.get(3.0, 0),
        "ref_5": distribution.get(5.0, 0),
        "ref_6": distribution.get(6.0, 0),
        "avg_out_degree": sum(outgoing.values()) / max(len(outgoing), 1),
        "avg_in_degree": sum(incoming.values()) / max(len(incoming), 1),
    }


def split_rows(
    rows: list[EdgeRefRow],
    *,
    test_ratio: float,
    seed: int,
) -> tuple[list[EdgeRefRow], list[EdgeRefRow]]:
    """
    Partition rows into train and held-out test splits.

    Edges are shuffled then greedily assigned to the test set up to
    ``test_ratio`` of the total, skipping any edge whose removal would leave
    either endpoint isolated (degree <= 1).

    Args:
        rows: Full list of deduplicated EdgeRefRow objects.
        test_ratio: Target fraction of edges to hold out for evaluation.
        seed: Random seed for shuffle reproducibility.

    Returns:
        tuple[list[EdgeRefRow], list[EdgeRefRow]]: ``(train_rows, test_rows)``
            where ``train_rows`` contains all edges not in the held-out split.

    Raises:
        ValueError: If no edges can be placed in the test split without
            isolating a node.
    """
    shuffled = list(rows)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    node_degree = Counter()
    for row in shuffled:
        node_degree[row.start_node] += 1
        node_degree[row.end_node] += 1

    target_test_size = max(1, int(len(shuffled) * test_ratio))
    test_rows: list[EdgeRefRow] = []
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
    test_rows: list[EdgeRefRow],
    *,
    all_rows: list[EdgeRefRow],
    negative_samples: int,
    seed: int,
) -> list[dict[str, object]]:
    """
    Build ranked-retrieval evaluation cases for each held-out test edge.

    For each test edge ``(source → positive)`` samples ``negative_samples``
    nodes that are neither the source nor any known positive target of that
    source.

    Args:
        test_rows: Held-out edges to evaluate.
        all_rows: Complete edge list used to identify positive targets and
            the candidate node pool.
        negative_samples: Number of negative targets to sample per test edge.
        seed: Random seed for negative sampling reproducibility.

    Returns:
        list[dict[str, object]]: One dict per test edge containing keys
            ``source``, ``positive``, ``negatives``, ``ref_value``, and
            ``transformed_weight``.
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
                "ref_value": row.ref_value,
                "transformed_weight": row.transformed_weight,
            }
        )
    return cases


def _build_embedding_index(
    model: object,
) -> tuple[dict[str, int], np.ndarray]:
    """
    Extract the key-to-index mapping and L2-normalised vectors from a model.

    Args:
        model: A trained node2vec / Word2Vec model or its ``KeyedVectors``.

    Returns:
        tuple[dict[str, int], np.ndarray]: ``(key_to_index, normed_vectors)``
            where ``normed_vectors[i]`` is the unit-length embedding for the
            node whose vocabulary index is ``i``.
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
        key_to_index: Vocabulary index mapping from :func:`_build_embedding_index`.
        normed_vectors: L2-normalised embedding matrix.
        source: Source node identifier.
        target: Target node identifier.

    Returns:
        float: Dot product of the two unit-length vectors, equivalent to cosine
            similarity.
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
    Compute the ROC-AUC score using the Wilcoxon-Mann-Whitney statistic.

    Args:
        scores: Predicted similarity scores, one per sample.
        labels: Binary ground-truth labels (1 = positive, 0 = negative),
            aligned with ``scores``.

    Returns:
        float: Area under the ROC curve in ``[0, 1]``.
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

    For each case the positive target is ranked among itself and its sampled
    negatives by cosine similarity. Aggregated metrics and a correlation between
    positive scores and transformed ref weights are returned.

    Args:
        model: Trained node2vec / Word2Vec model.
        cases: Evaluation cases produced by :func:`sample_evaluation_cases`.

    Returns:
        dict[str, float]: Metrics including ``auc``, ``mrr``, ``hit_at_1``,
            ``hit_at_5``, ``hit_at_10``, ``positive_mean_cosine``,
            ``negative_mean_cosine``, ``score_margin``, and
            ``positive_score_ref_corr``.
    """
    key_to_index, normed_vectors = _build_embedding_index(model)

    positive_scores: list[float] = []
    negative_scores: list[float] = []
    reciprocal_ranks: list[float] = []
    transformed_refs: list[float] = []
    hit_at_1 = 0
    hit_at_5 = 0
    hit_at_10 = 0

    for case in cases:
        source = case["source"]
        positive = case["positive"]
        negatives = case["negatives"]

        positive_score = _score_pair(key_to_index, normed_vectors, source, positive)
        positive_scores.append(positive_score)
        transformed_refs.append(float(case["transformed_weight"]))
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
        float(np.corrcoef(positive_scores, transformed_refs)[0, 1])
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
        "positive_score_ref_corr": corr,
    }


def write_rows(
    path: Path,
    rows: Iterable[EdgeRefRow],
) -> None:
    """
    Serialize EdgeRefRow objects to a whitespace-delimited edge-list file.

    Creates parent directories as needed.

    Args:
        path: Destination file path.
        rows: Iterable of EdgeRefRow objects to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f"{row.start_node} {row.end_node} {row.ref_value}\n")


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
    Execute the full hyperparameter sweep pipeline.

    Loads and deduplicates edges, creates train/test splits, evaluates all
    configurations in :data:`DEFAULT_CONFIGS`, retrains the best configuration
    on the full dataset, and writes all output artefacts.

    Args:
        input_path: Path to the edge-ref ``.list`` file.
        output_dir: Directory where model and evaluation artefacts are saved.
        seed: Random seed for splitting, negative sampling, and training.
        test_ratio: Fraction of edges held out for evaluation.
        negative_samples: Number of negative targets sampled per test edge.
        workers: Number of worker threads for node2vec training.
        min_count: Word2Vec ``min_count`` parameter.
        topn: Number of nearest neighbours saved for sample nodes.
    """
    all_rows = unique_rows(load_edge_ref_rows(input_path))
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
        train_split_path = Path(tmpdir) / "edge_ref_train_split.list"
        write_rows(train_split_path, train_rows)

        for index, config in enumerate(DEFAULT_CONFIGS, start=1):
            print(f"[{index}/{len(DEFAULT_CONFIGS)}] training {config.name}")
            model, _, _ = train_node2vec(
                train_split_path,
                graph_format="edgelist",
                weight_transform="abs-epsilon",
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
        weight_transform="abs-epsilon",
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
        "weight_transform": "abs-epsilon",
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
                "weight_transform": "abs-epsilon",
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
            "positive_score_ref_corr",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    report_lines = [
        "# Node2Vec Edge-Ref Sweep Report",
        "",
        "## Dataset",
        f"- Input file: `{input_path}`",
        f"- Unique directed edge rows: {graph_stats['edge_rows']}",
        f"- Nodes: {graph_stats['nodes']}",
        f"- Raw ref range: {graph_stats['min_ref']:.1f} to {graph_stats['max_ref']:.1f}",
        f"- Mean raw ref: {graph_stats['mean_ref']:.6f}",
        f"- Mean transformed ref weight: {graph_stats['mean_transformed_ref']:.6f}",
        (
            f"- Ref distribution: 0={int(graph_stats['zero_ref_edges'])}, 1={int(graph_stats['ref_1'])}, "
            f"3={int(graph_stats['ref_3'])}, 5={int(graph_stats['ref_5'])}, 6={int(graph_stats['ref_6'])}"
        ),
        "- Training interpretation: each row `<start> <end> <ref>` is a directed weighted edge.",
        "- Weight transform: `abs(ref) + 0.001` so the dominant `0.0` edges remain near-zero but valid for node2vec.",
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
            "{hit_at_10:.4f} | {score_margin:.4f} | {positive_score_ref_corr:.4f} |".format(
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
            f"- Positive cosine vs transformed ref correlation: {best_result['positive_score_ref_corr']:.4f}",
            "- Final delivery model: retrained on the full input file with the selected configuration.",
            "",
            "## Output Files",
            "- `best_model.n2v`: final model retrained on the full edge-ref dataset",
            "- `best_eval.json`: best run metrics and selection metadata",
            "- `selected_config.json`: chosen hyperparameters",
            "- `sweep_results.json` / `sweep_results.csv`: complete comparison across all runs",
            "- `best_model_neighbors.json`: sample nearest neighbours from the final model",
        ]
    )
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def main() -> None:
    """
    Entry point: parse CLI arguments and delegate to :func:`run_sweep`.
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
