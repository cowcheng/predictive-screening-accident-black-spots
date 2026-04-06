"""
Node2Vec evaluation pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import gensim
import numpy as np
from matplotlib import pyplot
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE


def parse_args() -> argparse.ArgumentParser:
    """
    Build and return the CLI argument parser for the evaluation pipeline.

    Returns:
        argparse.ArgumentParser: Configured parser with all evaluation flags.
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained Node2Vec model.")
    parser.add_argument("--model", required=True, help="Path to a trained .n2v model.")
    parser.add_argument(
        "--mode",
        choices=(
            "similarity",
            "nearest",
            "edge-embeddings",
            "tsne-nodes",
            "tsne-edges",
        ),
        default="similarity",
        help="Evaluation mode.",
    )
    parser.add_argument(
        "--edgelist",
        help="Path to an edge list file (edge-embeddings / tsne-edges mode).",
    )
    parser.add_argument(
        "--edge-embeddings",
        help="Input/output edge embeddings (.npy). Used by tsne-edges or provided for direct loading.",
    )
    parser.add_argument(
        "--edge-output",
        help="Output path for edge embeddings in edge-embeddings mode.",
    )
    parser.add_argument(
        "--edge-embedding-strategy",
        choices=("concat", "hadamard"),
        default="concat",
        help="How to combine node vectors into edge vectors.",
    )
    parser.add_argument(
        "--w1",
        help="First node ID for similarity mode.",
    )
    parser.add_argument(
        "--w2",
        help="Second node ID for similarity mode.",
    )
    parser.add_argument(
        "--word",
        help="Node ID for nearest-neighbour query.",
    )
    parser.add_argument(
        "--topn",
        type=int,
        default=10,
        help="Top-N results for nearest-neighbour and clustering prints.",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=15.0,
        help="TSNE perplexity.",
    )
    parser.add_argument(
        "--tsne-random-state",
        type=int,
        default=7,
        help="TSNE random seed.",
    )
    parser.add_argument(
        "--dbscan-eps",
        type=float,
        default=5.0,
        help="DBSCAN eps value.",
    )
    parser.add_argument(
        "--dbscan-min-samples",
        type=int,
        default=5,
        help="DBSCAN minimum samples.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="CPU parallelism for TSNE/DBSCAN where supported.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Enable plotting for TSNE modes.",
    )
    parser.add_argument(
        "--plot-output",
        help="If set, save TSNE plot to this path.",
    )
    return parser


def load_model(
    model_file: str | Path,
) -> gensim.models.KeyedVectors:
    """
    Load a Node2Vec model from a word2vec-format file.

    Args:
        model_file: Path to the saved `.n2v` (word2vec-format) model file.

    Returns:
        gensim.models.KeyedVectors: Loaded keyed-vectors object.
    """
    return gensim.models.KeyedVectors.load_word2vec_format(str(model_file))


def load_edgelist(
    edgelist_file: str | Path,
) -> list[tuple[str, str]]:
    """
    Parse a whitespace-delimited edge list file into a list of node-pair tuples.

    Lines starting with ``#`` and blank lines are ignored. Only the first two
    tokens on each line are used; any weight column is discarded.

    Args:
        edgelist_file: Path to the edge list file.

    Returns:
        list[tuple[str, str]]: List of ``(source, target)`` node-ID pairs.
    """
    edges: list[tuple[str, str]] = []
    with Path(edgelist_file).open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            edges.append((parts[0], parts[1]))
    return edges


def build_node_matrix(
    model: gensim.models.KeyedVectors,
) -> np.ndarray:
    """
    Stack all node embeddings from the model into a 2-D matrix.

    Args:
        model: Loaded keyed-vectors object containing node embeddings.

    Returns:
        np.ndarray: Array of shape ``(n_nodes, embedding_dim)``.
    """
    vectors = [model[key] for key in model.key_to_index]
    return np.asarray(vectors)


def build_edge_embeddings(
    model: gensim.models.KeyedVectors,
    edges: Iterable[tuple[str, str]],
    *,
    strategy: str = "concat",
) -> np.ndarray:
    """
    Construct edge-level embeddings by combining source and target node vectors.

    Args:
        model: Loaded keyed-vectors object containing node embeddings.
        edges: Iterable of ``(source, target)`` node-ID pairs.
        strategy: Combination method. ``"concat"`` concatenates the two node
            vectors (output dim = 2 × embedding_dim); ``"hadamard"`` takes
            their element-wise product (output dim = embedding_dim).

    Returns:
        np.ndarray: Array of shape ``(n_edges, output_dim)`` with dtype float.

    Raises:
        KeyError: If any node in an edge is absent from the model vocabulary.
    """
    vectors: list[list[float]] = []
    for source, target in edges:
        if source not in model.key_to_index or target not in model.key_to_index:
            missing = [
                node for node in (source, target) if node not in model.key_to_index
            ]
            raise KeyError(f"Missing node(s) in model vocabulary: {missing}")

        source_vec = model[source]
        target_vec = model[target]
        if strategy == "hadamard":
            vec = source_vec * target_vec
        else:
            vec = np.concatenate([source_vec, target_vec], axis=0)
        vectors.append(vec.tolist())

    return np.asarray(vectors, dtype=float)


def save_edge_embeddings(
    embedding_file: str | Path,
    embeddings: np.ndarray,
) -> None:
    """
    Persist edge embeddings to a NumPy binary file.

    Args:
        embedding_file: Destination path (a ``.npy`` extension is appended by
            ``np.save`` if not already present).
        embeddings: Array to save, typically of shape ``(n_edges, output_dim)``.
    """
    np.save(str(embedding_file), embeddings)


def load_edge_embeddings(
    embedding_file: str | Path,
) -> np.ndarray:
    """
    Load edge embeddings from a NumPy binary file.

    Args:
        embedding_file: Path to a ``.npy`` file produced by
            :func:`save_edge_embeddings`.

    Returns:
        np.ndarray: Loaded embeddings array.
    """
    return np.load(str(embedding_file))


def similarity(
    model: gensim.models.KeyedVectors,
    w1: str,
    w2: str,
) -> float:
    """
    Compute the cosine similarity between two node embeddings.

    Args:
        model: Loaded keyed-vectors object.
        w1: First node ID.
        w2: Second node ID.

    Returns:
        float: Cosine similarity in ``[-1, 1]``.
    """
    return float(model.similarity(w1, w2))


def nearest(
    model: gensim.models.KeyedVectors,
    word: str,
    topn: int = 10,
) -> list[tuple[str, float]]:
    """
    Return the top-N most similar nodes to a given node.

    Args:
        model: Loaded keyed-vectors object.
        word: Query node ID.
        topn: Number of nearest neighbours to return.

    Returns:
        list[tuple[str, float]]: List of ``(node_id, cosine_similarity)`` pairs,
            sorted by descending similarity.
    """
    return model.similar_by_word(word, topn=topn)


def tsne(
    embeddings: np.ndarray,
    *,
    perplexity: float,
    random_state: int,
    n_jobs: int,
) -> np.ndarray:
    """
    Reduce high-dimensional embeddings to 2-D using t-SNE.

    Args:
        embeddings: Input array of shape ``(n_samples, n_features)``.
        perplexity: t-SNE perplexity parameter (roughly the number of effective
            nearest neighbours).
        random_state: Random seed for reproducibility.
        n_jobs: Number of parallel jobs (passed to sklearn where supported).

    Returns:
        np.ndarray: 2-D projection of shape ``(n_samples, 2)``.
    """

    reducer = TSNE(perplexity=perplexity, random_state=random_state, n_jobs=n_jobs)
    return reducer.fit_transform(X=embeddings)


def cluster_dbscan(
    embeddings_2d: np.ndarray,
    *,
    eps: float,
    min_samples: int,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cluster 2-D embeddings with DBSCAN.

    Args:
        embeddings_2d: Array of shape ``(n_samples, 2)`` — typically the output
            of :func:`tsne`.
        eps: Maximum distance between two samples to be considered neighbours.
        min_samples: Minimum number of samples in a neighbourhood for a point
            to be considered a core point.
        n_jobs: Number of parallel jobs for the neighbour search.

    Returns:
        tuple[np.ndarray, np.ndarray]: A pair ``(labels, core_samples_mask)``
            where ``labels`` is an integer array of cluster IDs (``-1`` for
            noise) and ``core_samples_mask`` is a boolean array indicating core
            points.
    """
    clustering = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=n_jobs).fit(
        X=embeddings_2d
    )
    labels = clustering.labels_
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    if clustering.core_sample_indices_.size:
        core_samples_mask[clustering.core_sample_indices_] = True
    return labels, core_samples_mask


def plot_clusters(
    embeddings_2d: np.ndarray,
    labels: np.ndarray,
    core_samples_mask: np.ndarray,
    *,
    output: Optional[str | Path] = None,
    figsize: tuple[float, float] = (100.0, 75.0),
    dpi: int = 300,
    marker_size: int = 5,
) -> None:
    """
    Render a scatter plot of DBSCAN clusters from 2-D embeddings.

    Each cluster is drawn in a distinct colour using the Spectral colour map.
    Only core points are plotted; noise points (label ``-1``) are omitted.

    Args:
        embeddings_2d: 2-D coordinates of shape ``(n_samples, 2)``.
        labels: Cluster label for each sample (``-1`` = noise).
        core_samples_mask: Boolean mask indicating which samples are core points.
        output: If provided, save the figure to this path instead of
            displaying it interactively.
        figsize: Figure width and height in inches.
        dpi: Output resolution in dots per inch.
        marker_size: Marker size used for scatter points.
    """

    pyplot.figure(figsize=figsize, dpi=dpi)

    unique_labels = set(labels.tolist())
    colors = pyplot.cm.Spectral(
        np.linspace(start=0.0, stop=1.0, num=len(unique_labels))
    )
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        points = embeddings_2d[mask & core_samples_mask]
        pyplot.plot(
            points[:, 0],
            points[:, 1],
            "o",
            markerfacecolor=color,
            markeredgecolor="k",
            markersize=marker_size,
        )

    if output:
        pyplot.savefig(str(output), bbox_inches="tight")
    else:
        pyplot.show()


def _default_edge_output_path(
    model_file: str | Path,
) -> Path:
    """
    Derive a default edge-embeddings output path from the model file path.

    Appends ``.npy`` to the model file's existing suffix, e.g.
    ``best_model.n2v`` → ``best_model.n2v.npy``.

    Args:
        model_file: Path to the source model file.

    Returns:
        Path: Derived output path with ``.npy`` appended.
    """
    model_path = Path(model_file)
    return model_path.with_suffix(model_path.suffix + ".npy")


def main(
    argv: Optional[list[str]] = None,
) -> None:
    """
    Entry point for the Node2Vec evaluation pipeline.

    Dispatches to the appropriate evaluation routine based on ``--mode``:

    * ``similarity``      — cosine similarity between two nodes.
    * ``nearest``         — nearest-neighbour query for a single node.
    * ``edge-embeddings`` — compute and save edge embeddings from an edge list.
    * ``tsne-nodes``      — t-SNE + DBSCAN clustering on node embeddings.
    * ``tsne-edges``      — t-SNE + DBSCAN clustering on edge embeddings.

    Args:
        argv: Argument list to parse. Defaults to ``sys.argv[1:]`` when
            ``None``.
    """
    parser = parse_args()
    args = parser.parse_args(argv)
    model = load_model(args.model)

    if args.mode == "similarity":
        if args.w1 is None or args.w2 is None:
            parser.error("--mode similarity requires --w1 and --w2.")
        score = similarity(model, args.w1, args.w2)
        print(f"{args.w1} vs {args.w2}: {score:.6f}")
        return

    if args.mode == "nearest":
        if args.word is None:
            parser.error("--mode nearest requires --word.")
        neighbors = nearest(model, args.word, topn=args.topn)
        print(f"Nearest to {args.word} (top {args.topn}):")
        for word, score in neighbors:
            print(f"{word}\\t{score:.6f}")
        return

    if args.mode == "edge-embeddings":
        if args.edgelist is None:
            parser.error("--mode edge-embeddings requires --edgelist.")
        edges = load_edgelist(args.edgelist)
        embeddings = build_edge_embeddings(
            model,
            edges,
            strategy=args.edge_embedding_strategy,
        )
        output = (
            Path(args.edge_output)
            if args.edge_output
            else _default_edge_output_path(args.model)
        )
        save_edge_embeddings(output, embeddings)
        print(f"Saved edge embeddings to {output}")
        return

    if args.mode == "tsne-nodes":
        node_vectors = build_node_matrix(model)
        node_vectors_2d = tsne(
            node_vectors,
            perplexity=args.perplexity,
            random_state=args.tsne_random_state,
            n_jobs=args.n_jobs,
        )
        labels, core_mask = cluster_dbscan(
            node_vectors_2d,
            eps=args.dbscan_eps,
            min_samples=args.dbscan_min_samples,
            n_jobs=args.n_jobs,
        )
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"Estimated number of clusters: {n_clusters}")
        if args.plot:
            plot_clusters(
                node_vectors_2d,
                labels,
                core_mask,
                output=args.plot_output,
                figsize=(300.0, 200.0),
                dpi=300,
                marker_size=10,
            )
        return

    if args.mode == "tsne-edges":
        if args.edge_embeddings is None:
            if args.edgelist is None:
                parser.error(
                    "--mode tsne-edges requires --edge-embeddings or --edgelist."
                )
            edges = load_edgelist(args.edgelist)
            embeddings = build_edge_embeddings(
                model,
                edges,
                strategy=args.edge_embedding_strategy,
            )
            edge_output = (
                Path(args.edge_output)
                if args.edge_output
                else _default_edge_output_path(args.model)
            )
            save_edge_embeddings(edge_output, embeddings)
            print(f"Saved edge embeddings to {edge_output}")
        else:
            embeddings = load_edge_embeddings(args.edge_embeddings)

        embeddings_2d = tsne(
            embeddings,
            perplexity=args.perplexity,
            random_state=args.tsne_random_state,
            n_jobs=args.n_jobs,
        )
        labels, core_mask = cluster_dbscan(
            embeddings_2d,
            eps=args.dbscan_eps,
            min_samples=args.dbscan_min_samples,
            n_jobs=args.n_jobs,
        )
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"Estimated number of clusters: {n_clusters}")

        if args.plot:
            plot_clusters(
                embeddings_2d,
                labels,
                core_mask,
                output=args.plot_output,
                figsize=(100.0, 75.0),
                dpi=300,
            )
        return


if __name__ == "__main__":
    main()
