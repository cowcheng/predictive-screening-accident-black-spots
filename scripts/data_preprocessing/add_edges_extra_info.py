"""
Combine OSM edge records with accident labels, node-level slope, and node-level angle.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd

_PROPERTY_NAME_ALIASES: dict[str, str] = {
    "start_node_id": "start_node",
    "end_node_id": "end_node",
    "road_slope": "slope",
    "road_angle": "angle",
    "turn_angle": "angle",
}


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for the edge enrichment pipeline.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Map accidents to OSM edges and enrich each edge with slope/angle "
            "features when available."
        )
    )

    # Input/output paths
    parser.add_argument(
        "--project-root",
        default=None,
        help="Optional repository root override. Defaults to repo root of this script.",
    )
    parser.add_argument(
        "--edge-input",
        default="data/raw/osm_edge_info.csv",
        help="Input OSM edge list.",
    )
    parser.add_argument(
        "--news-input",
        default="data/raw/news_data_raw.csv",
        help="News table with location/detail columns.",
    )
    parser.add_argument(
        "--edge-output",
        default="data/interim/osm_edge_info_with_accident.csv",
        help="Output edgelist path (augmented with accident/slope/angle).",
    )
    parser.add_argument(
        "--angle-input",
        default="data/interim/node_angle.list",
        help="Node-angle source (comma separated or whitespace list).",
    )
    parser.add_argument(
        "--slope-input",
        default="data/interim/node_slope.list",
        help="Node-slope source (comma separated or whitespace list).",
    )
    parser.add_argument(
        "--road-accident-output",
        default="data/interim/hk_roads_accidents_info.csv",
        help="Output path for the per-road accident count table.",
    )

    # Edge column names
    parser.add_argument(
        "--edge-start-col",
        default="u",
        help="Column in edge-input containing edge start node ids.",
    )
    parser.add_argument(
        "--edge-end-col",
        default="v",
        help="Column in edge-input containing edge end node ids.",
    )
    parser.add_argument(
        "--edge-name-col",
        default="name",
        help="Column in edge-input containing road names.",
    )
    parser.add_argument(
        "--lane-column",
        default="lanes",
        help="Edge column that contains the number of lanes.",
    )

    # News column names
    parser.add_argument(
        "--news-location-col",
        default="location",
        help="News column that contains road/location names.",
    )
    parser.add_argument(
        "--news-detail-col",
        default="detail",
        help="News detail text column.",
    )

    # Output column names
    parser.add_argument(
        "--accident-count-column",
        default="number_of_accident",
        help="Column name for merged accident counts.",
    )
    parser.add_argument(
        "--label-column",
        default="have_accident",
        help="Name of the accident label column.",
    )
    parser.add_argument(
        "--angle-column",
        default="angle",
        help="Column name for merged angle values.",
    )
    parser.add_argument(
        "--slope-column",
        default="slope",
        help="Column name for merged slope values.",
    )
    parser.add_argument(
        "--lanes-diff-column",
        default="lanes_diff",
        help="Name of the output lane-difference column.",
    )

    # News filtering
    parser.add_argument(
        "--keyword",
        default="意外",
        help="Keyword required in news detail.",
    )
    parser.add_argument(
        "--filter-word",
        action="append",
        default=[],
        help="Words/phrases to exclude from detail text. Can be repeated.",
    )
    parser.add_argument(
        "--min-occurrences",
        type=int,
        default=1,
        help=(
            "Minimum mention count per accident bundle required for a road node "
            "to be treated as accident-related."
        ),
    )

    # Behavior toggles
    parser.add_argument(
        "--write-road-accident-output",
        action="store_true",
        help="Write per-road accident summary CSV (Road, NumberOfAccident).",
    )
    parser.add_argument(
        "--skip-angle",
        action="store_true",
        help="Skip angle merge when angle source is unavailable.",
    )
    parser.add_argument(
        "--skip-slope",
        action="store_true",
        help="Skip slope merge when slope source is unavailable.",
    )

    return parser.parse_args()


def _project_root(
    overrides_root: str | None,
) -> Path:
    """
    Resolve the project root directory.

    Args:
        overrides_root: Optional explicit override path.

    Returns:
        Absolute path to the project root.
    """
    if overrides_root:
        return Path(overrides_root).resolve()
    return Path(__file__).resolve().parents[2]


def _read_csv(
    path: Path,
) -> pd.DataFrame:
    """
    Read a CSV file with a clear error if the file does not exist.

    Args:
        path: Absolute path to a CSV file.

    Returns:
        Parsed DataFrame.
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path)


def _normalize_text(
    value: object,
) -> str:
    """
    Apply NFKC normalization, drop full-width spaces, and collapse whitespace.

    Args:
        value: Arbitrary input (string-like or NaN).

    Returns:
        Normalized string, or empty string for NaN.
    """
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\u3000", " ").strip()
    return re.sub(r"\s+", " ", text)


def _normalize_node_id(
    value: object,
) -> str:
    """
    Normalize an OSM node identifier.

    Args:
        value: Raw value (string, int, float, or NaN).

    Returns:
        Normalized id with surrounding quotes and trailing ``.0`` removed.
    """
    text = _normalize_text(value)
    if not text:
        return ""
    text = text.strip("'\"")
    return re.sub(r"\.0+$", "", text)


def _normalize_road_name(
    value: object,
) -> str:
    """
    Normalize a road name for fuzzy matching.

    Removes parenthetical notes, drops trailing English tokens, and collapses
    whitespace.

    Args:
        value: Raw road name.

    Returns:
        Normalized road name (possibly empty).
    """
    text = _normalize_text(value)
    if not text:
        return ""

    text = re.split(r"[（(]", text, maxsplit=1)[0].strip()
    if not text:
        return ""

    kept: list[str] = []
    for token in text.split(" "):
        if not token:
            continue
        if re.search(r"[A-Za-z]", token):
            break
        kept.append(token)

    return " ".join(kept).strip()


def _count_accidents_by_road(
    news_df: pd.DataFrame,
    location_col: str,
    detail_col: str,
    keyword: str,
    filter_words: list[str],
    min_occurrences: int,
) -> dict[str, int]:
    """
    Count news mentions of an accident keyword grouped by normalized road name.

    Mirrors the notebook behavior: keep rows whose detail contains the keyword,
    drop rows containing any filter words, group by normalized location, and
    discard roads with fewer than ``min_occurrences`` mentions.

    Args:
        news_df: News table with location and detail columns.
        location_col: Column containing road/location names.
        detail_col: Column containing news detail text.
        keyword: Keyword required in the detail text.
        filter_words: Phrases that disqualify a row when present.
        min_occurrences: Minimum mention count required to keep a road.

    Returns:
        Mapping from normalized road name to mention count.
    """
    if detail_col not in news_df.columns or location_col not in news_df.columns:
        raise KeyError("Missing required news columns for counting accidents.")

    detail = news_df[detail_col].astype("string").fillna("").map(_normalize_text)
    location = (
        news_df[location_col].astype("string").fillna("").map(_normalize_road_name)
    )

    if detail.empty:
        return {}

    mask = detail.str.contains(keyword, regex=False)
    for word in filter_words:
        masked_word = _normalize_text(word) if word else ""
        if masked_word:
            mask &= ~detail.str.contains(masked_word, regex=False)

    mask &= location.str.len() > 0
    if not mask.any():
        return {}

    counts = location[mask].value_counts().to_dict()
    return {road: count for road, count in counts.items() if count >= min_occurrences}


def _coerce_property_name(
    name: str,
) -> str:
    """
    Map a raw column name to a canonical key used by the property reader.

    Args:
        name: Original column name.

    Returns:
        Canonical (lower-cased) column name.
    """
    lowered = name.strip().lower()
    return _PROPERTY_NAME_ALIASES.get(lowered, lowered)


def _read_node_property_file(
    path: Path,
    kind: str,
    value_column: str,
) -> pd.DataFrame:
    """
    Read a node-property file (CSV or whitespace-separated list) into a tidy frame.

    Args:
        path: Path to the source file.
        kind: Logical kind of property (``"angle"`` or ``"slope"``).
        value_column: Name of the value column expected in the output frame.

    Returns:
        DataFrame with columns ``["start_node", "end_node", value_column]``,
        deduplicated by mean per (start_node, end_node).
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing {kind} source: {path}")

    if path.suffix.lower() == ".csv":
        raw = pd.read_csv(path)
        raw = raw.rename(columns={c: _coerce_property_name(c) for c in raw.columns})
        required = {"start_node", "end_node", value_column}
        if not required.issubset(raw.columns):
            raise ValueError(
                f"{path} does not contain required columns for {kind}: "
                "start_node, end_node, and value column."
            )
        frame = raw[["start_node", "end_node", value_column]].copy()
    else:
        raw = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
        if raw.shape[1] < 3:
            raise ValueError(f"{path} is not a valid {kind} file.")
        # Angle source may carry a mid-node column we want to drop.
        if kind == "angle" and raw.shape[1] >= 4:
            frame = raw.iloc[:, [0, 2, 3]].copy()
        else:
            frame = raw.iloc[:, [0, 1, 2]].copy()
        frame.columns = ["start_node", "end_node", value_column]

    frame["start_node"] = frame["start_node"].map(_normalize_node_id)
    frame["end_node"] = frame["end_node"].map(_normalize_node_id)
    frame = frame[(frame["start_node"] != "") & (frame["end_node"] != "")]
    frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
    frame = frame.dropna(subset=[value_column])
    return frame.groupby(["start_node", "end_node"], as_index=False)[
        value_column
    ].mean()


def _road_counts_to_frame(
    road_counts: dict[str, int],
    count_column: str,
) -> pd.DataFrame:
    """
    Convert a ``{road: count}`` dict into a tabular ``Road``/``count_column`` frame.

    Args:
        road_counts: Mapping of road name to mention count.
        count_column: Name to use for the count column.

    Returns:
        DataFrame with columns ``["Road", count_column]``.
    """
    return pd.DataFrame(
        [
            {"Road": road, count_column: str(count)}
            for road, count in road_counts.items()
        ],
        columns=["Road", count_column],
    )


def _merge_node_property(
    edge_df: pd.DataFrame,
    source_path: Path,
    prop_kind: str,
    value_column: str,
    output_column: str,
    skip_if_missing: bool,
) -> pd.DataFrame:
    """
    Merge a per-node property table onto the edge frame.

    Args:
        edge_df: Edge frame containing ``_start_node_key`` and ``_end_node_key``.
        source_path: Path to the property file.
        prop_kind: Property label used in error messages (e.g. ``"angle"``).
        value_column: Column name expected in the property file.
        output_column: Column name to use after merging.
        skip_if_missing: If True and the source file is absent, fill ``pd.NA``
            instead of raising.

    Returns:
        Edge frame extended with ``output_column``.
    """
    cleanup_columns = {
        output_column,
        f"{output_column}_x",
        f"{output_column}_y",
    }
    edge_df = edge_df.drop(
        columns=[column for column in cleanup_columns if column in edge_df.columns]
    )

    try:
        property_df = _read_node_property_file(
            source_path,
            kind=prop_kind,
            value_column=value_column,
        )
    except FileNotFoundError:
        if not skip_if_missing:
            raise
        edge_df[output_column] = pd.NA
        return edge_df

    merged = edge_df.merge(
        property_df,
        left_on=["_start_node_key", "_end_node_key"],
        right_on=["start_node", "end_node"],
        how="left",
    )
    merged = merged.rename(columns={value_column: output_column})
    return merged.drop(columns=["start_node", "end_node"])


def _compute_lanes_diff(
    edge_df: pd.DataFrame,
    start_key_col: str,
    end_key_col: str,
    lane_col: str,
) -> pd.Series:
    """
    Compute the lane difference between an edge and the next edge sharing its end node.

    For each row, locate the first edge whose start node matches the current
    row's end node and return ``first_lane - current_lane``. Missing values
    yield ``0``.

    Args:
        edge_df: Edge frame containing the lane column and the normalized
            start/end key columns.
        start_key_col: Column name for the normalized start node key.
        end_key_col: Column name for the normalized end node key.
        lane_col: Column containing lane counts.

    Returns:
        Float ``Series`` of lane differences indexed like ``edge_df``.
    """
    if lane_col not in edge_df.columns:
        return pd.Series(0.0, index=edge_df.index)

    lane_values = pd.to_numeric(edge_df[lane_col], errors="coerce")
    lane_lookup = pd.DataFrame(
        {"start_node": edge_df[start_key_col], "lane": lane_values}
    )
    lane_lookup = lane_lookup[
        (lane_lookup["start_node"] != "") & (lane_lookup["lane"].notna())
    ]
    if lane_lookup.empty:
        return pd.Series(0.0, index=edge_df.index)

    first_lane_by_start = lane_lookup.drop_duplicates(subset=["start_node"]).set_index(
        "start_node"
    )["lane"]
    matched_lanes = edge_df[end_key_col].map(first_lane_by_start)
    has_required = matched_lanes.notna() & lane_values.notna()
    return (matched_lanes - lane_values).where(has_required, other=0.0).fillna(0.0)


def run(
    args: argparse.Namespace,
) -> int:
    """
    Execute the enrichment pipeline using the parsed CLI arguments.

    Args:
        args: Namespace returned by :func:`parse_args`.

    Returns:
        Process exit code (``0`` on success).
    """
    root = _project_root(args.project_root)
    edge_input = (root / args.edge_input).resolve()
    news_input = (root / args.news_input).resolve()
    edge_output = (root / args.edge_output).resolve()
    angle_input = (root / args.angle_input).resolve()
    slope_input = (root / args.slope_input).resolve()

    edge_df = _read_csv(edge_input)
    news_df = _read_csv(news_input)

    for col in (args.edge_start_col, args.edge_end_col, args.edge_name_col):
        if col not in edge_df.columns:
            raise KeyError(f"Missing required edge column: {col}")
    for col in (args.news_location_col, args.news_detail_col):
        if col not in news_df.columns:
            raise KeyError(f"Missing required news column: {col}")

    accident_counts = _count_accidents_by_road(
        news_df=news_df,
        location_col=args.news_location_col,
        detail_col=args.news_detail_col,
        keyword=args.keyword,
        filter_words=args.filter_word,
        min_occurrences=args.min_occurrences,
    )

    working = edge_df.copy()
    working["_start_node_key"] = working[args.edge_start_col].map(_normalize_node_id)
    working["_end_node_key"] = working[args.edge_end_col].map(_normalize_node_id)
    working["_road_name_key"] = working[args.edge_name_col].map(_normalize_road_name)

    working[args.accident_count_column] = (
        pd.to_numeric(
            working["_road_name_key"].map(accident_counts).fillna(0),
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )
    working[args.label_column] = (
        working[args.accident_count_column] >= args.min_occurrences
    )
    working[args.lanes_diff_column] = _compute_lanes_diff(
        edge_df=working,
        start_key_col="_start_node_key",
        end_key_col="_end_node_key",
        lane_col=args.lane_column,
    )

    working = _merge_node_property(
        edge_df=working,
        source_path=angle_input,
        prop_kind="angle",
        value_column="angle",
        output_column=args.angle_column,
        skip_if_missing=args.skip_angle,
    )
    working = _merge_node_property(
        edge_df=working,
        source_path=slope_input,
        prop_kind="slope",
        value_column="slope",
        output_column=args.slope_column,
        skip_if_missing=args.skip_slope,
    )

    if args.write_road_accident_output:
        road_accident_df = _road_counts_to_frame(
            road_counts=accident_counts,
            count_column="NumberOfAccident",
        )
        road_accident_output = (root / args.road_accident_output).resolve()
        road_accident_output.parent.mkdir(parents=True, exist_ok=True)
        road_accident_df.to_csv(road_accident_output, index=False)
        print(
            f"Road-level accident counts exported to: {road_accident_output} "
            f"({len(road_accident_df)} roads)"
        )

    legacy_columns = {"key", "angle_x", "angle_y", "slope_x", "slope_y"}
    legacy_to_drop = [column for column in working.columns if column.lower() in legacy_columns]
    if legacy_to_drop:
        working = working.drop(columns=legacy_to_drop)

    working = working.drop(
        columns=["_start_node_key", "_end_node_key", "_road_name_key"]
    )
    edge_output.parent.mkdir(parents=True, exist_ok=True)
    working.to_csv(edge_output, index=False)

    accident_rows = int(working[args.label_column].sum())
    print(f"Road rows marked with accidents: {accident_rows}")
    print(f"Wrote enriched edges to: {edge_output}")
    return 0


def main() -> int:
    """
    Entry point for command-line execution.
    """
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
