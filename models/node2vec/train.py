"""
Node2Vec training pipeline extracted from notebooks.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Iterable, Optional

import networkx as nx
from node2vec import Node2Vec


def parse_args() -> argparse.ArgumentParser:
    """
    Build training CLI parser.

    Returns:
        argparse.ArgumentParser: parser configured with node2vec training options.
    """
    parser = argparse.ArgumentParser(
        description="Train a Node2Vec model from an edgelist."
    )
    parser.add_argument("--edgelist", required=True, help="Input edge list file.")
    parser.add_argument(
        "--graph-format",
        choices=("auto", "edgelist", "node-angle-transitions"),
        default="auto",
        help=(
            "Input graph format. 'auto' treats 2/3-column rows as a standard edge list and "
            "4-column rows as node-angle transitions: <prev> <mid> <next> <turn_angle>."
        ),
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=4,
        help="Embedding dimensionality.",
    )
    parser.add_argument(
        "--walk-length",
        dest="walk_length",
        type=int,
        default=50,
        help="Length of each random walk.",
    )
    parser.add_argument(
        "--num-walks",
        dest="num_walks",
        type=int,
        default=25,
        help="Number of random walks per node.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="Word2Vec context window.",
    )
    parser.add_argument(
        "--min-count",
        dest="min_count",
        type=int,
        default=1,
        help="Word2Vec min count parameter.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_worker_count(),
        help="Number of workers used by Node2Vec/Word2Vec.",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=1.0,
        help="Node2Vec return hyperparameter.",
    )
    parser.add_argument(
        "--q",
        type=float,
        default=1.0,
        help="Node2Vec in-out hyperparameter.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--undirected",
        action="store_false",
        dest="directed",
        help="Treat the input graph as undirected.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path for the Word2Vec .n2v model.",
    )
    parser.add_argument(
        "--weight-transform",
        choices=("none", "abs", "abs-epsilon"),
        default="none",
        help=(
            "Optional transform applied to weighted edge values before training. "
            "'abs-epsilon' is useful for signed road attributes such as slope."
        ),
    )
    parser.set_defaults(directed=True)
    return parser


def default_worker_count() -> int:
    """
    Return a default safe worker count based on detected CPU availability.

    Returns:
        int: At least 1.
    """
    return max(1, os.cpu_count() or 1)


def _iter_edge_rows(
    path: Path,
) -> Iterable[tuple[int, list[str]]]:
    """
    Yield parsed, non-empty, non-comment edge rows from file.

    Args:
        path: Path to the edge list file.

    Yields:
        tuple[int, list[str]]: A (line_number, parts) pair where parts are
            whitespace-split tokens from the non-comment portion of the line.
    """
    with path.open("r", encoding="utf-8") as handle:
        for line_num, raw_line in enumerate(handle, start=1):
            line = raw_line.split("#", maxsplit=1)[0].strip()
            if not line:
                continue
            parts = line.split()
            yield line_num, parts


def resolve_graph_format(
    path: Path,
    graph_format: str = "auto",
) -> str:
    """
    Resolve the effective graph format for a training input file.

    When graph_format is "auto", the format is inferred from the column count
    of the first data row: 2–3 columns → "edgelist", 4+ columns →
    "node-angle-transitions".

    Args:
        path: Path to the edge list file.
        graph_format: Explicit format override ("edgelist" or
            "node-angle-transitions"), or "auto" to infer from file content.

    Returns:
        str: Resolved format string, either "edgelist" or
            "node-angle-transitions".

    Raises:
        ValueError: If graph_format is "auto" and the first row has fewer than
            2 columns, or if the file is empty or contains only comments.
    """
    if graph_format != "auto":
        return graph_format

    for _, parts in _iter_edge_rows(path):
        if len(parts) in {2, 3}:
            return "edgelist"
        if len(parts) >= 4:
            return "node-angle-transitions"
        raise ValueError(f"Invalid edge list row: {parts!r}")

    raise ValueError(f"Edge list file is empty or contains only comments: {path}")


def _infer_edgelist_format(
    path: Path,
) -> tuple[bool, list[tuple[str, str, float | None]]]:
    """
    Infer if an edge list is weighted and parse each row.

    Args:
        path: Path to the edge list file. Each non-comment row must have
            either 2 columns (unweighted: src dst) or 3+ columns
            (weighted: src dst weight …). Mixing formats across rows raises
            an error.

    Returns:
        tuple[bool, list[tuple[str, str, float | None]]]: A pair of
            (is_weighted, rows). is_weighted is True when the file uses the
            3-column format. Each element of rows is (src, dst, weight) where
            weight is None for unweighted files.

    Raises:
        ValueError: If a row has fewer than 2 columns, a weight value cannot
            be parsed as float, rows mix weighted and unweighted formats, or
            the file is empty / contains only comments.
    """
    rows: list[tuple[str, str, float | None]] = []
    is_weighted: Optional[bool] = None

    for line_num, parts in _iter_edge_rows(path):
        if len(parts) == 2:
            row_is_weighted = False
            rows.append((parts[0], parts[1], None))
        elif len(parts) >= 3:
            row_is_weighted = True
            try:
                rows.append((parts[0], parts[1], float(parts[2])))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid weight value at line {line_num}: {parts[2]}"
                ) from exc
        else:
            raise ValueError(f"Invalid edge list row at line {line_num}: {parts!r}")

        if is_weighted is None:
            is_weighted = row_is_weighted
        elif is_weighted != row_is_weighted:
            raise ValueError(
                "Mixed weighted and unweighted edge rows are not supported. "
                "All rows must use the same format."
            )

    if is_weighted is None:
        raise ValueError(f"Edge list file is empty or contains only comments: {path}")

    return is_weighted, rows


def _turn_angle_to_weight(
    turn_angle: float,
) -> float:
    """
    Map a turn angle in degrees onto a positive transition weight.

    Straighter transitions receive higher weight, while U-turns remain valid with a
    very small positive weight so downstream node2vec training does not reject them.

    Args:
        turn_angle: Turn angle in degrees. The sign is ignored; values outside
            [0, 180] are clamped to 180.

    Returns:
        float: Transition weight in the range (1e-3, 1.001], where values
            closer to 1.001 represent straighter (lower-angle) transitions.
    """
    clamped_angle = min(abs(turn_angle), 180.0)
    straightness = (180.0 - clamped_angle) / 180.0
    return 1e-3 + straightness


def _transform_weight(
    weight: float,
    weight_transform: str,
) -> float:
    """
    Apply an optional transform to a raw edge weight.

    Args:
        weight: Raw edge weight value.
        weight_transform: Transform to apply. One of:
            - "none": return weight unchanged.
            - "abs": return abs(weight).
            - "abs-epsilon": return abs(weight) + 1e-3 (ensures strict positivity
              for signed attributes such as slope).

    Returns:
        float: Transformed weight value.

    Raises:
        ValueError: If weight_transform is not a recognised option.
    """
    if weight_transform == "none":
        return weight
    if weight_transform == "abs":
        return abs(weight)
    if weight_transform == "abs-epsilon":
        return abs(weight) + 1e-3
    raise ValueError(f"Unsupported weight transform: {weight_transform}")


def _build_node_angle_transition_graph(
    path: Path,
    *,
    directed: bool,
) -> tuple[nx.Graph, str]:
    """
    Build a transition graph from a node-angle edge list.

    Each row encodes a road segment transition as
    ``<prev> <mid> <next> <turn_angle>``.  Nodes in the resulting graph
    represent directed road segments ("prev->mid", "mid->next") and edges
    represent valid transitions between consecutive segments, weighted by
    turn angle via :func:`_turn_angle_to_weight`.

    Args:
        path: Path to the node-angle transition file.
        directed: Build a :class:`~networkx.DiGraph` when True, otherwise a
            :class:`~networkx.Graph`.

    Returns:
        tuple[nx.Graph, str]: A (graph, weight_key) pair where weight_key is
            always "weight".

    Raises:
        ValueError: If any row has fewer than 4 columns or the turn angle
            cannot be parsed as a float.
    """
    create_using = nx.DiGraph if directed else nx.Graph
    graph = create_using()

    for line_num, parts in _iter_edge_rows(path):
        if len(parts) < 4:
            raise ValueError(
                "Node-angle transition rows must have at least 4 columns: "
                "<prev> <mid> <next> <turn_angle>."
            )

        start_node, mid_node, end_node, turn_angle = parts[:4]
        try:
            angle_value = float(turn_angle)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid turn angle at line {line_num}: {turn_angle}"
            ) from exc

        source = f"{start_node}->{mid_node}"
        target = f"{mid_node}->{end_node}"
        graph.add_edge(source, target, weight=_turn_angle_to_weight(angle_value))

    return graph, "weight"


_GRAPH_BUILD_CACHE: dict[
    tuple[str, bool, str, str, int, int],
    tuple[nx.Graph, Optional[str]],
] = {}
_GRAPH_BUILD_CACHE_MAX_SIZE = 16


def _build_graph_cached(
    edgelist_path: Path,
    *,
    directed: bool,
    graph_format: str,
    weight_transform: str,
) -> tuple[nx.Graph, Optional[str]]:
    """
    Build and cache a graph from an input path with lightweight validation.

    Cache keys include the normalized path plus file size and mtime to avoid stale
    entries when source files change.

    Args:
        edgelist_path: Path to the edge list file.
        directed: Build a directed graph when True.
        graph_format: Graph format hint for format resolution.
        weight_transform: Weight transform applied to weighted edge values.

    Returns:
        tuple[nx.Graph, Optional[str]]: A (graph, weight_key) pair where graph is
            a networkx Graph or DiGraph instance, and weight_key is "weight" for
            weighted graphs or None for unweighted graphs.
    """
    resolved = edgelist_path.resolve()
    stats = resolved.stat()
    cache_key = (
        str(resolved),
        directed,
        graph_format,
        weight_transform,
        int(stats.st_size),
        stats.st_mtime_ns,
    )

    if cache_key in _GRAPH_BUILD_CACHE:
        graph, weight_key = _GRAPH_BUILD_CACHE[cache_key]
        return graph.copy(), weight_key

    graph, weight_key = _build_graph_uncached(
        resolved,
        directed=directed,
        graph_format=graph_format,
        weight_transform=weight_transform,
    )
    # Keep copy immutable for consumers while serving a copy to each caller.
    _GRAPH_BUILD_CACHE[cache_key] = (graph, weight_key)
    if len(_GRAPH_BUILD_CACHE) > _GRAPH_BUILD_CACHE_MAX_SIZE:
        oldest_key = next(iter(_GRAPH_BUILD_CACHE))
        _GRAPH_BUILD_CACHE.pop(oldest_key, None)

    return graph.copy(), weight_key


def _build_graph_uncached(
    edgelist_file: str | Path,
    *,
    directed: bool = True,
    graph_format: str = "auto",
    weight_transform: str = "none",
) -> tuple[nx.Graph, Optional[str]]:
    """
    Build a networkx graph from an edgelist file.

    Args:
        edgelist_file: Path to an edge list file.
        directed: Build a :class:`~networkx.DiGraph` when True, otherwise a
            :class:`~networkx.Graph`.
        graph_format: Input format hint passed to :func:`resolve_graph_format`.
            "auto" infers the format from file content; "edgelist" and
            "node-angle-transitions" override inference.
        weight_transform: Transform applied to raw edge weights before they are
            stored on the graph. See :func:`_transform_weight` for valid values.
            Only used for "edgelist" format rows that carry a weight column.

    Returns:
        tuple[nx.Graph, Optional[str]]: A (graph, weight_key) pair. weight_key
            is "weight" for weighted graphs and None for unweighted graphs.

    Raises:
        ValueError: If the file is empty, contains invalid rows, or weight
            values cannot be parsed.
    """
    edgelist_path = Path(edgelist_file)
    resolved_graph_format = resolve_graph_format(
        edgelist_path, graph_format=graph_format
    )

    if resolved_graph_format == "node-angle-transitions":
        return _build_node_angle_transition_graph(edgelist_path, directed=directed)

    create_using = nx.DiGraph if directed else nx.Graph
    weighted, rows = _infer_edgelist_format(edgelist_path)
    graph = create_using()

    if weighted:
        graph.add_edges_from(
            (u, v, {"weight": _transform_weight(float(weight), weight_transform)})
            for u, v, weight in rows
        )
        return graph, "weight"

    graph.add_edges_from((u, v) for u, v, _ in rows)
    return graph, None


def build_graph(
    edgelist_file: str | Path,
    *,
    directed: bool = True,
    graph_format: str = "auto",
    weight_transform: str = "none",
) -> tuple[nx.Graph, Optional[str]]:
    """
    Build a networkx graph from an edgelist file.

    This cached entrypoint reuses previously-built graphs when the same path and
    settings are reused across repeated training calls.

    Args:
        edgelist_file: Path to an edge list file.
        directed: Build a :class:`~networkx.DiGraph` when True, otherwise
            a :class:`~networkx.Graph`.
        graph_format: Input format hint passed to :func:`resolve_graph_format`.
            "auto" infers the format from file content; "edgelist" and
            "node-angle-transitions" override inference.
        weight_transform: Transform applied to raw edge weights before they are
            stored on the graph. See :func:`_transform_weight` for valid values.
            Only used for "edgelist" format rows that carry a weight column.
    
    Returns:
        tuple[nx.Graph, Optional[str]]: A (graph, weight_key) pair. weight_key
            is "weight" for weighted graphs and None for unweighted graphs.
    """
    return _build_graph_cached(
        Path(edgelist_file),
        directed=directed,
        graph_format=graph_format,
        weight_transform=weight_transform,
    )


def _validate_positive_weights(
    graph: nx.Graph,
    weight_key: str,
) -> None:
    """
    Validate that every edge weight is a finite, strictly positive number.

    Args:
        graph: NetworkX graph whose edges will be checked.
        weight_key: Edge attribute key that holds the weight value.

    Raises:
        ValueError: If any edge weight is missing, non-numeric, non-finite,
            or not strictly greater than zero.
    """
    for _, _, attrs in graph.edges(data=True):
        weight = attrs.get(weight_key)
        try:
            numeric_weight = float(weight)
        except (TypeError, ValueError) as exc:
            raise ValueError("edge weight must be finite and > 0") from exc

        if not math.isfinite(numeric_weight) or numeric_weight <= 0:
            raise ValueError("edge weight must be finite and > 0")


def train_node2vec(
    edgelist_file: str | Path,
    *,
    graph_format: str = "auto",
    weight_transform: str = "none",
    dimensions: int = 4,
    walk_length: int = 50,
    num_walks: int = 25,
    window: int = 10,
    min_count: int = 1,
    workers: int = default_worker_count(),
    p: float = 1.0,
    q: float = 1.0,
    directed: bool = True,
    seed: int = 42,
    output: Optional[str | Path] = None,
) -> tuple[object, Path, Optional[str]]:
    """
    Train Node2Vec from an edgelist.

    Args:
        edgelist_file: Path to an edge list file accepted by :func:`build_graph`.
        graph_format: Format hint forwarded to :func:`build_graph`.
        weight_transform: Weight transform forwarded to :func:`build_graph`.
        dimensions: Embedding dimensionality passed to Node2Vec.
        walk_length: Number of nodes per random walk.
        num_walks: Number of random walks originating from each node.
        window: Word2Vec skip-gram context window size.
        min_count: Word2Vec minimum token frequency threshold.
        workers: Number of parallel workers for walk generation and Word2Vec.
        p: Node2Vec return hyperparameter (controls likelihood of revisiting a node).
        q: Node2Vec in-out hyperparameter (controls BFS vs. DFS behaviour).
        directed: Treat the graph as directed when True.
        seed: Random seed for reproducibility.
        output: Optional path where the trained Word2Vec model is saved in
            word2vec text format. Parent directories are created as needed.

    Returns:
        tuple[object, Path, Optional[str]]: A (model, edgelist_path, weight_key)
            triple. model is the fitted gensim Word2Vec instance,
            edgelist_path is the resolved input path, and weight_key is
            "weight" for weighted graphs or None for unweighted graphs.

    Raises:
        ValueError: If the graph has fewer than 2 nodes or no edges, or if
            edge weight validation fails.
    """
    edgelist_path = Path(edgelist_file)
    graph, weight_key = build_graph(
        edgelist_path,
        directed=directed,
        graph_format=graph_format,
        weight_transform=weight_transform,
    )

    if weight_key is not None:
        _validate_positive_weights(graph, weight_key)

    if graph.number_of_nodes() < 2:
        raise ValueError(
            f"Graph must contain at least 2 nodes. Found {graph.number_of_nodes()}."
        )
    if graph.number_of_edges() < 1:
        raise ValueError("Graph must contain at least 1 edge.")

    node2vec = Node2Vec(
        graph=graph,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        weight_key=weight_key,
        workers=workers,
        p=p,
        q=q,
        seed=seed,
    )
    model = node2vec.fit(window=window, min_count=min_count)

    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        model.wv.save_word2vec_format(str(output_path))

    return model, edgelist_path, weight_key


def _default_output_path(
    edgelist_file: Path,
    dimensions: int,
    walk_length: int,
    num_walks: int,
) -> Path:
    """
    Build a default model output path derived from the edgelist filename and hyperparameters.

    Args:
        edgelist_file: Path to the input edge list file (only the stem is used).
        dimensions: Embedding dimensionality.
        walk_length: Random walk length.
        num_walks: Number of random walks per node.

    Returns:
        Path: A path of the form ``model/<stem>-<dimensions>_<walk_length>_<num_walks>.n2v``.
    """
    name = f"{edgelist_file.stem}-{dimensions}_{walk_length}_{num_walks}.n2v"
    return Path("model") / name


def main(
    argv: Optional[list[str]] = None,
) -> None:
    """
    CLI entry point for training a Node2Vec model.

    Parses arguments, resolves the output path, calls :func:`train_node2vec`,
    and prints a short summary to stdout.

    Args:
        argv: Argument list to parse. Defaults to ``sys.argv[1:]`` when None.
    """
    parser = parse_args()
    args = parser.parse_args(argv)

    output_path = args.output
    if output_path is None:
        output_path = str(
            _default_output_path(
                Path(args.edgelist),
                args.dimensions,
                args.walk_length,
                args.num_walks,
            )
        )

    model, _, weight_key = train_node2vec(
        args.edgelist,
        graph_format=args.graph_format,
        weight_transform=args.weight_transform,
        dimensions=args.dimensions,
        walk_length=args.walk_length,
        num_walks=args.num_walks,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers,
        p=args.p,
        q=args.q,
        directed=args.directed,
        seed=args.seed,
        output=output_path,
    )

    print(f"Saved model to: {output_path}")
    print(f"Output vectors: {model.wv.vector_size}")
    print(f"Weight key: {weight_key or 'None'}")


if __name__ == "__main__":
    main()
