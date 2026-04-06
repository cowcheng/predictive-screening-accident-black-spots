"""
Collect OpenStreetMap road network features for a place/query address.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import osmnx as ox
import pandas as pd


def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """
    Parse command-line arguments for OSM data collection.

    Args:
        argv: Explicit argument list. Defaults to ``sys.argv[1:]`` when ``None``.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Build OSM node/edge feature CSVs for a target address."
    )
    parser.add_argument(
        "--address",
        default="Nathan Road",
        help="Address or place to query OSM graph for.",
    )
    parser.add_argument(
        "--network-type",
        default="drive_service",
        help="OSMNX network type: e.g. drive_service, drive, walk.",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Request graph simplification from OSMNx.",
    )
    parser.add_argument(
        "--output",
        default="data/raw/osm_data_raw.csv",
        help="Combined output CSV (nodes + edges).",
    )
    parser.add_argument(
        "--nodes-output",
        default="data/raw/osm_node_info.csv",
        help="Optional dedicated node output CSV.",
    )
    parser.add_argument(
        "--edges-output",
        default="data/raw/osm_edge_info.csv",
        help="Optional dedicated edge output CSV.",
    )
    parser.add_argument(
        "--no-separate",
        action="store_true",
        help="Skip writing separate node/edge files.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=30.0,
        help="HTTP timeout forwarded to OSMnx/requests in seconds.",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=1000.0,
        help="Search radius in meters when using address-based graph lookup.",
    )
    parser.add_argument(
        "--clean-geometry",
        action="store_true",
        help="Convert geometry column to WKT strings.",
    )
    return parser.parse_args(argv)


def _project_root() -> Path:
    """
    Return the absolute path to the project root directory.

    Returns:
        Path two levels above this script file.
    """
    return Path(__file__).resolve().parents[2]


def _normalize_geometry(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert a geometry column to WKT strings for CSV serialization.

    Args:
        df: DataFrame that may contain a ``geometry`` column.

    Returns:
        DataFrame with geometry values cast to strings, or the
        original DataFrame unchanged if no geometry column exists.
    """
    if "geometry" not in df.columns:
        return df
    return df.assign(geometry=df["geometry"].astype(str))


def _collect_graph(
    address: str,
    network_type: str,
    simplify: bool,
    timeout: float,
    distance: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download an OSM road graph and return node and edge DataFrames.

    Attempts ``ox.graph_from_address`` first; falls back to
    ``ox.graph_from_place`` when the address lookup fails.

    Args:
        address: Address or place name to geocode.
        network_type: OSMnx network type (e.g. ``"drive_service"``).
        simplify: Whether to topologically simplify the graph.
        timeout: HTTP request timeout in seconds forwarded to OSMnx.
        distance: Search radius in meters for address-based lookup.

    Returns:
        A ``(node_df, edge_df)`` tuple. ``node_df`` has a ``node_id``
        column (renamed from ``osmid``) and ``edge_df`` has an
        ``edge_osmid`` column. Both include a ``_record_type`` tag.
    """
    ox.settings.requests_timeout = max(1, int(timeout))

    try:
        graph = ox.graph_from_address(
            address,
            dist=distance,
            network_type=network_type,
            simplify=simplify,
        )
    except Exception:
        graph = ox.graph_from_place(
            address,
            network_type=network_type,
            simplify=simplify,
        )

    node_gdf, edge_gdf = ox.graph_to_gdfs(graph)

    node_df = node_gdf.reset_index()
    edge_df = edge_gdf.reset_index()

    node_df = node_df.rename(columns={"osmid": "node_id"})
    edge_df = edge_df.rename(columns={"osmid": "edge_osmid"})

    node_df["_record_type"] = "node"
    edge_df["_record_type"] = "edge"

    return node_df, edge_df


def run_collection(
    args: argparse.Namespace,
) -> int:
    """
    Execute the full collection pipeline and write output CSVs.

    Args:
        args: Parsed CLI arguments from ``parse_args``.

    Returns:
        Exit code (``0`` on success).
    """
    project_root = _project_root()
    output_path = (project_root / args.output).resolve()
    nodes_path = (project_root / args.nodes_output).resolve()
    edges_path = (project_root / args.edges_output).resolve()

    node_df, edge_df = _collect_graph(
        address=args.address,
        network_type=args.network_type,
        simplify=args.simplify,
        timeout=args.request_timeout,
        distance=args.distance,
    )

    if args.clean_geometry:
        node_df = _normalize_geometry(node_df)
        edge_df = _normalize_geometry(edge_df)

    if not args.no_separate:
        nodes_path.parent.mkdir(parents=True, exist_ok=True)
        edges_path.parent.mkdir(parents=True, exist_ok=True)

        node_df.to_csv(nodes_path, index=False)
        edge_df.to_csv(edges_path, index=False)

    all_cols = sorted(set(node_df.columns) | set(edge_df.columns))
    node_aligned = node_df.reindex(columns=all_cols)
    edge_aligned = edge_df.reindex(columns=all_cols)

    combined = pd.concat([node_aligned, edge_aligned], axis=0, ignore_index=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    return 0


def main(
    argv: list[str] | None = None,
) -> int:
    """
    CLI entry point for OSM data collection.

    Args:
        argv: Explicit argument list. Defaults to ``sys.argv[1:]`` when ``None``.

    Returns:
        Exit code (``0`` on success).
    """
    args = parse_args(argv)
    return run_collection(args)


if __name__ == "__main__":
    raise SystemExit(main())
