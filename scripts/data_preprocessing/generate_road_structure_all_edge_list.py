"""
Generate road-structure .el edge lists from notebook logic.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

NODE2NODE_RE = re.compile(r"^\(\s*([^,]+)\s*,\s*([^,]+)\s*\)$")

KIND_TO_COLS: dict[str, list[str]] = {
    "base": ["start_node", "end_node"],
    "lanes": ["start_node", "end_node", "lanes"],
    "length": ["start_node", "end_node", "length"],
    "ref": ["start_node", "end_node", "ref"],
    "maxspeed": ["start_node", "end_node", "maxspeed"],
    "time": ["start_node", "end_node", "time"],
}

ANGLE_COLS: list[str] = ["start_node", "end_node", "road_angle"]


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the edge list generator.

    Returns:
        argparse.Namespace: Parsed arguments including mode, input paths,
            column names, default fill values, and output configuration.
    """
    parser = argparse.ArgumentParser(
        description="Convert OSM road structure tables into node2vec-style .el edge lists."
    )
    parser.add_argument(
        "--mode",
        choices=("base", "lanes", "length", "ref", "maxspeed", "time", "angle", "all"),
        default="all",
        help="Which edge list to write.",
    )
    parser.add_argument(
        "--edge-input",
        default="data/raw/osm_edge_info.csv",
        help="Input edge table for base/lanes/length/ref/maxspeed lists.",
    )
    parser.add_argument(
        "--angle-input",
        default="data/interim/road_structure_angle_info.csv",
        help="Input table for angle edge list.",
    )
    parser.add_argument(
        "--start-id-col",
        default="u",
        help="Column for source node id.",
    )
    parser.add_argument(
        "--end-id-col",
        default="v",
        help="Column for target node id.",
    )
    parser.add_argument(
        "--node2node-col",
        default="node2node_id",
        help="Fallback tuple column when start/end columns are missing.",
    )
    parser.add_argument(
        "--default-lanes",
        type=float,
        default=1.0,
        help="Fallback lanes value when missing.",
    )
    parser.add_argument(
        "--default-ref",
        type=float,
        default=0.0,
        help="Fallback ref value when missing.",
    )
    parser.add_argument(
        "--default-maxspeed",
        type=float,
        default=50.0,
        help="Fallback maxspeed value when missing.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/interim",
        help="Directory for generated .el files.",
    )
    parser.add_argument(
        "--output",
        default="data/interim/edge_{kind}.list",
        help="Output file when --mode is not 'all'.",
    )
    parser.add_argument(
        "--filename-template",
        default="edge_{kind}.list",
        help="Template used when --mode is 'all'. '{kind}' is replaced by output type.",
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip rows with missing required fields instead of failing.",
    )
    return parser.parse_args()


def _read_csv(
    input_path: str,
) -> pd.DataFrame:
    """
    Read a CSV file into a DataFrame, raising if the path does not exist.

    Args:
        input_path (str): Filesystem path to the CSV file.

    Returns:
        pd.DataFrame: Contents of the CSV file.

    Raises:
        FileNotFoundError: If no file exists at *input_path*.
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path)


def _parse_node_pairs(
    df: pd.DataFrame,
    start_col: str,
    end_col: str,
    node2node_col: str,
) -> pd.DataFrame:
    """
    Extract source/target node id pairs from a DataFrame.

    Prefers explicit *start_col* / *end_col* columns. Falls back to parsing
    ``(u, v)`` tuples from *node2node_col* when those columns are absent.

    Args:
        df (pd.DataFrame): Input edge table.
        start_col (str): Column name for source node ids.
        end_col (str): Column name for target node ids.
        node2node_col (str): Fallback column containing ``"(u, v)"`` tuple strings.

    Returns:
        pd.DataFrame: Two-column DataFrame with ``start_node`` and ``end_node``
            string columns.

    Raises:
        KeyError: If neither the explicit columns nor *node2node_col* are present.
    """
    if start_col in df.columns and end_col in df.columns:
        return pd.DataFrame(
            {
                "start_node": df[start_col].astype("string").str.strip(),
                "end_node": df[end_col].astype("string").str.strip(),
            }
        )
    if node2node_col not in df.columns:
        raise KeyError(
            f"Missing node columns. Provide '{start_col}' and '{end_col}', "
            f"or '{node2node_col}'."
        )
    parsed = df[node2node_col].astype("string").str.extract(NODE2NODE_RE, expand=True)
    return pd.DataFrame(
        {
            "start_node": parsed[0].str.strip(),
            "end_node": parsed[1].str.strip(),
        }
    )


def _prepare_weights(
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """
    Coerce and fill edge weight columns, then derive the travel-time column.

    Numeric coercion is applied to ``lanes``, ``length``, ``ref``, and
    ``maxspeed``. Missing values are filled with the corresponding defaults
    from *args*. A ``time`` column (seconds) is computed as
    ``length / (maxspeed * 0.277777778)``; rows where ``maxspeed <= 0``
    receive ``pd.NA`` for ``time``.

    Args:
        df (pd.DataFrame): Raw edge table containing at least the weight columns.
        args (argparse.Namespace): Parsed CLI arguments supplying
            ``default_lanes``, ``default_ref``, and ``default_maxspeed``.

    Returns:
        pd.DataFrame: Copy of *df* with coerced weight columns and a new
            ``time`` column added.
    """
    prepared = df.copy()
    prepared["lanes"] = pd.to_numeric(prepared["lanes"], errors="coerce").fillna(
        args.default_lanes
    )
    prepared["length"] = pd.to_numeric(prepared["length"], errors="coerce")
    prepared["ref"] = pd.to_numeric(prepared["ref"], errors="coerce").fillna(
        args.default_ref
    )
    prepared["maxspeed"] = pd.to_numeric(prepared["maxspeed"], errors="coerce").fillna(
        args.default_maxspeed
    )
    prepared["time"] = prepared["length"] / (prepared["maxspeed"] * 0.277777778)
    prepared.loc[prepared["maxspeed"] <= 0, "time"] = pd.NA
    return prepared


def _validate_rows(
    df: pd.DataFrame,
    required: list[str],
    skip_invalid: bool,
) -> pd.DataFrame:
    """
    Drop or raise on rows that have null values in required columns.

    Args:
        df (pd.DataFrame): DataFrame to validate.
        required (list[str]): Column names that must be non-null.
        skip_invalid (bool): If ``True``, silently drop bad rows; otherwise
            raise ``ValueError`` when any are found.

    Returns:
        pd.DataFrame: Filtered DataFrame containing only rows where all
            *required* columns are non-null.

    Raises:
        ValueError: If invalid rows are found and *skip_invalid* is ``False``.
    """
    mask = df[required].notna().all(axis=1)
    skipped = int((~mask).sum())
    if skipped and not skip_invalid:
        bad = list(df.loc[~mask].index[:10])
        raise ValueError(
            f"Found {skipped} invalid rows; use --skip-invalid to drop them. "
            f"First invalid row indices: {bad}"
        )
    return df.loc[mask].copy() if skipped else df


def _write_el(
    path: Path,
    df: pd.DataFrame,
    columns: list[str],
) -> int:
    """
    Write selected DataFrame columns to a space-delimited edge list file.

    Parent directories are created automatically if they do not exist.

    Args:
        path (Path): Destination file path.
        df (pd.DataFrame): DataFrame containing the columns to write.
        columns (list[str]): Ordered list of column names to include.

    Returns:
        int: Number of rows written to *path*.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    subset = df[columns]
    with path.open("w") as f:
        for row in subset.itertuples(index=False, name=None):
            f.write(" ".join(map(str, row)) + "\n")
    return len(subset)


def _build_edges(
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """
    Parse node pairs and join prepared weight columns into a single edge DataFrame.

    Args:
        df (pd.DataFrame): Raw edge table read from the edge input CSV.
        args (argparse.Namespace): Parsed CLI arguments used for column names
            and default weight values.

    Returns:
        pd.DataFrame: Merged DataFrame with ``start_node``, ``end_node``, and
            all weight columns (``lanes``, ``length``, ``ref``, ``maxspeed``,
            ``time``), containing only rows with valid node ids.
    """
    nodes = _parse_node_pairs(
        df, args.start_id_col, args.end_id_col, args.node2node_col
    )
    merged = nodes.join(_prepare_weights(df, args), how="inner")
    return _validate_rows(merged, ["start_node", "end_node"], args.skip_invalid)


def _prepare_angle_frame(
    angle_df: pd.DataFrame,
    skip_invalid: bool,
) -> pd.DataFrame:
    """
    Rename angle input columns and validate required fields.

    Args:
        angle_df (pd.DataFrame): Raw angle edge table with ``start_node_id``
            and ``end_node_id`` columns.
        skip_invalid (bool): If ``True``, silently drop rows with missing
            values.

    Returns:
        pd.DataFrame: Frame with ``start_node``, ``end_node``, and
            ``road_angle`` columns, containing only valid rows.
    """
    renamed = angle_df.rename(
        columns={"start_node_id": "start_node", "end_node_id": "end_node"}
    )
    return _validate_rows(renamed[ANGLE_COLS], ANGLE_COLS, skip_invalid)


def _run_mode(
    mode: str,
    args: argparse.Namespace,
    edge_df: pd.DataFrame,
    angle_df: pd.DataFrame | None,
    output_dir: Path,
    output: Path,
    template: str,
) -> list[str]:
    """
    Write one or more edge list files according to the selected mode.

    Args:
        mode (str): One of ``"base"``, ``"lanes"``, ``"length"``, ``"ref"``,
            ``"maxspeed"``, ``"time"``, ``"angle"``, or ``"all"``.
        args (argparse.Namespace): Parsed CLI arguments forwarded to validation
            helpers.
        edge_df (pd.DataFrame): Prepared edge DataFrame from ``_build_edges``.
        angle_df (pd.DataFrame | None): Angle edge DataFrame, or ``None`` if
            the angle input file was not found.
        output_dir (Path): Directory used when *mode* is ``"all"`` to resolve
            output paths via *template*.
        output (Path): Destination path used for single-mode writes; may
            contain a ``{kind}`` placeholder.
        template (str): Filename template applied with ``str.format(kind=...)``
            when *mode* is ``"all"``.

    Returns:
        list[str]: Human-readable summary strings of the form
            ``"<path> (<n> rows)"`` for each file written.

    Raises:
        FileNotFoundError: If *mode* is ``"angle"`` and *angle_df* is ``None``.
    """
    outputs: list[str] = []

    if mode == "angle":
        if angle_df is None:
            raise FileNotFoundError(f"Angle input file not found: {args.angle_input}")
        frame = _prepare_angle_frame(angle_df, args.skip_invalid)
        rows = _write_el(output, frame, ANGLE_COLS)
        outputs.append(f"{output} ({rows} rows)")
        return outputs

    if mode == "all":
        for kind, cols in KIND_TO_COLS.items():
            target = output_dir / template.format(kind=kind)
            frame = _validate_rows(edge_df[cols], cols, args.skip_invalid)
            rows = _write_el(target, frame, cols)
            outputs.append(f"{target} ({rows} rows)")
        if angle_df is not None:
            angle_target = output_dir / template.format(kind="angle")
            frame = _prepare_angle_frame(angle_df, args.skip_invalid)
            rows = _write_el(angle_target, frame, ANGLE_COLS)
            outputs.append(f"{angle_target} ({rows} rows)")
        return outputs

    target = Path(str(output).format(kind=mode))
    cols = KIND_TO_COLS[mode]
    frame = _validate_rows(edge_df[cols], cols, args.skip_invalid)
    rows = _write_el(target, frame, cols)
    outputs.append(f"{target} ({rows} rows)")
    return outputs


def main() -> int:
    """
    Entry point: parse arguments, load inputs, and generate edge list files.

    Returns:
        int: Exit code (``0`` on success).

    Raises:
        FileNotFoundError: If a required input file is missing.
        ValueError: If invalid rows are encountered and ``--skip-invalid`` was
            not passed.
    """
    args = parse_args()
    edges = _build_edges(_read_csv(args.edge_input), args)

    angle_df: pd.DataFrame | None = None
    if args.mode in {"angle", "all"}:
        angle_path = Path(args.angle_input)
        if angle_path.exists():
            angle_df = _read_csv(args.angle_input)
        elif args.mode == "angle":
            raise FileNotFoundError(f"Angle input file not found: {angle_path}")

    outputs = _run_mode(
        args.mode,
        args,
        edges,
        angle_df,
        Path(args.output_dir),
        Path(args.output),
        args.filename_template,
    )

    print("Generated:")
    for item in outputs:
        print(f" - {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
