"""
Generate road-structure node relation artifacts from OSM nodes + edges.
"""

from __future__ import annotations

import argparse
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for road-structure artifact generation.

    Returns:
        argparse.Namespace: Parsed arguments including mode, input/output paths,
            elevation settings, and processing flags.
    """
    parser = argparse.ArgumentParser(
        description="Generate node-based road-structure artifacts (angle/slope)."
    )
    parser.add_argument(
        "--mode",
        choices=("angle", "slope", "all"),
        default="all",
        help="Which node-relational structure(s) to build.",
    )
    parser.add_argument(
        "--node-input",
        default="data/raw/osm_node_info.csv",
        help="OSM node table.",
    )
    parser.add_argument(
        "--edge-input",
        default="data/raw/osm_edge_info.csv",
        help="OSM edge table.",
    )
    parser.add_argument(
        "--start-id-col",
        default="u",
        help="Edge source node column.",
    )
    parser.add_argument(
        "--end-id-col",
        default="v",
        help="Edge target node column.",
    )
    parser.add_argument(
        "--node2node-col",
        default="node2node_id",
        help="Fallback tuple column in format (u, v).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/interim",
        help="Output directory for generated .list files.",
    )
    parser.add_argument(
        "--output",
        default="data/interim/node_{kind}.list",
        help="Output file for single-mode runs. Use '{kind}' to vary by mode.",
    )
    parser.add_argument(
        "--filename-template",
        default="node_{kind}.list",
        help="Template for all-mode outputs where {kind} is angle/slope.",
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip invalid rows instead of failing.",
    )
    parser.add_argument(
        "--slope-zero-fallback",
        action="store_true",
        help=(
            "If elevation lookup fails, generate slope rows with 0.0 rather than failing."
        ),
    )
    parser.add_argument(
        "--elevation-cache-dir",
        default=str(Path("data/interim/elevation_cache")),
        help=(
            "Optional root directory for elevation caches/tiles. "
            "Defaults to a writable project-local cache directory."
        ),
    )
    parser.add_argument(
        "--elevation-product",
        default="SRTM1",
        help="Elevation product passed to elevation.clip().",
    )
    parser.add_argument(
        "--elevation-max-tiles",
        type=int,
        default=9,
        help="Maximum number of tiles elevation.clip() may request in one call.",
    )
    parser.add_argument(
        "--elevation-clip-margin",
        type=float,
        default=0.01,
        help="Extra lon/lat margin for the DEM clipping bounds.",
    )
    parser.add_argument(
        "--slope-min-baseline-m",
        type=float,
        default=30.0,
        help="Minimum horizontal baseline used when estimating per-edge slope from DEM.",
    )
    parser.add_argument(
        "--skip-slope",
        action="store_true",
        help="Skip slope generation even when mode=all.",
    )
    return parser.parse_args()


def _read_csv(
    path: str,
    required_cols: list[str],
) -> pd.DataFrame:
    """
    Read a CSV file and validate that required columns are present.

    Args:
        path: File path to the CSV.
        required_cols: Column names that must exist in the CSV.

    Returns:
        pd.DataFrame: Loaded DataFrame with at least the required columns.

    Raises:
        FileNotFoundError: If the file does not exist at ``path``.
        KeyError: If any column in ``required_cols`` is absent from the CSV.
    """
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    df = pd.read_csv(input_path)
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {input_path}: {missing}")
    return df


def _normalize_ids(
    series: pd.Series,
) -> pd.Series:
    """
    Coerce a Series of node IDs to nullable integer then to string.

    Non-numeric values are converted to ``<NA>`` via ``pd.to_numeric``.

    Args:
        series: Raw node ID values (may be mixed string/numeric).

    Returns:
        pd.Series: String-typed Series with numeric IDs; non-numeric entries
            become ``<NA>``.
    """
    return pd.to_numeric(series, errors="coerce").astype("Int64").astype("string")


def _normalize_node_pairs(
    df: pd.DataFrame,
    start_col: str,
    end_col: str,
    fallback_col: str,
) -> pd.DataFrame:
    """
    Extract and normalise start/end node ID pairs from an edge DataFrame.

    Prefers ``start_col`` / ``end_col`` when both are present; otherwise parses
    ``fallback_col`` as a ``(u, v)`` tuple string.

    Args:
        df: Edge DataFrame containing node pair information.
        start_col: Column name for the source node ID.
        end_col: Column name for the target node ID.
        fallback_col: Column name holding a ``(u, v)`` tuple string used when
            ``start_col`` or ``end_col`` is absent.

    Returns:
        pd.DataFrame: Two-column DataFrame with ``start_node_id`` and
            ``end_node_id``, with NaN rows dropped.

    Raises:
        KeyError: If neither the primary columns nor ``fallback_col`` exist.
    """
    if start_col in df.columns and end_col in df.columns:
        return (
            df[[start_col, end_col]]
            .rename(columns={start_col: "start_node_id", end_col: "end_node_id"})
            .dropna()
            .copy()
        )

    if fallback_col not in df.columns:
        raise KeyError(
            f"Missing node pair columns. Provide '{start_col}' and '{end_col}' or '{fallback_col}'."
        )

    parsed = (
        df[fallback_col]
        .astype("string")
        .str.extract(r"^\(\s*([^,]+)\s*,\s*([^)]+)\)\s*$")
    )
    return (
        pd.DataFrame(
            {
                "start_node_id": parsed[0].str.strip(),
                "end_node_id": parsed[1].str.strip(),
            }
        )
        .dropna()
        .copy()
    )


def _build_adjacency(
    node_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build an undirected adjacency table from directed node pairs.

    Each directed edge ``(u, v)`` is mirrored to ``(v, u)`` and duplicate pairs
    are removed.

    Args:
        node_pairs: DataFrame with ``start_node_id`` and ``end_node_id`` columns.

    Returns:
        pd.DataFrame: Deduplicated two-column DataFrame with ``start_node_id``
            and ``end_node_id`` representing all undirected adjacencies.
    """
    forward = node_pairs[["start_node_id", "end_node_id"]]
    reversed_ = pd.DataFrame(
        {
            "start_node_id": forward["end_node_id"],
            "end_node_id": forward["start_node_id"],
        }
    )
    return (
        pd.concat([forward, reversed_], ignore_index=True)
        .dropna()
        .drop_duplicates(ignore_index=True)
    )


def _parse_node_coords(
    path: str,
) -> pd.DataFrame:
    """
    Load OSM node coordinates and normalise column names.

    Expects columns ``node_id``, ``x`` (longitude), and ``y`` (latitude).

    Args:
        path: Path to the OSM node CSV file.

    Returns:
        pd.DataFrame: Columns ``node_id`` (string), ``lon``, and ``lat``, with
            NaN rows dropped.

    Raises:
        FileNotFoundError: If the node CSV does not exist.
        KeyError: If ``node_id``, ``x``, or ``y`` columns are missing.
    """
    node_df = _read_csv(path, ["node_id", "x", "y"])
    node_df["node_id"] = _normalize_ids(node_df["node_id"])
    return (
        node_df[["node_id", "x", "y"]].dropna().rename(columns={"x": "lon", "y": "lat"})
    )


def _compute_bearing(
    start_lat: pd.Series,
    start_lon: pd.Series,
    end_lat: pd.Series,
    end_lon: pd.Series,
) -> pd.Series:
    """
    Compute the forward bearing (degrees, 0–360) between coordinate pairs.

    Args:
        start_lat: Series of starting latitudes in decimal degrees.
        start_lon: Series of starting longitudes in decimal degrees.
        end_lat: Series of ending latitudes in decimal degrees.
        end_lon: Series of ending longitudes in decimal degrees.

    Returns:
        pd.Series: Bearing values in degrees in the range ``[0, 360)``.
    """
    lon_diff = np.radians(end_lon - start_lon)
    start_lat_rad = np.radians(start_lat)
    end_lat_rad = np.radians(end_lat)
    y = np.sin(lon_diff) * np.cos(end_lat_rad)
    x = np.cos(start_lat_rad) * np.sin(end_lat_rad) - np.sin(start_lat_rad) * np.cos(
        end_lat_rad
    ) * np.cos(lon_diff)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def _attach_coords(
    df: pd.DataFrame,
    node_coords: pd.DataFrame,
    key: str,
    prefix: str,
) -> pd.DataFrame:
    """
    Left-join node coordinates onto ``df`` for a given node-id column.

    Args:
        df: DataFrame containing the node-id column ``key``.
        node_coords: DataFrame with ``node_id``, ``lon``, and ``lat`` columns.
        key: The column name in ``df`` whose values should be looked up.
        prefix: Prefix used for the attached coordinate columns, producing
            ``{prefix}_lat`` and ``{prefix}_lon``.

    Returns:
        pd.DataFrame: ``df`` with two additional columns ``{prefix}_lat`` and
            ``{prefix}_lon``.
    """
    return df.merge(
        node_coords.rename(
            columns={"node_id": key, "lat": f"{prefix}_lat", "lon": f"{prefix}_lon"}
        ),
        on=key,
        how="left",
    )


def _build_angle_rows(
    edge_pairs: pd.DataFrame,
    node_coords: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """
    Compute turn angles for every ``(start → mid → end)`` triplet.

    The turn angle is the signed difference between incoming and outgoing
    bearings, normalised to ``(-180, 180]``.

    Args:
        edge_pairs: DataFrame with ``start_node_id`` and ``end_node_id`` columns
            representing directed road edges.
        node_coords: DataFrame with ``node_id``, ``lon``, and ``lat`` columns.

    Returns:
        tuple[pd.DataFrame, int]: A two-element tuple of:
            - DataFrame with columns ``start_node_id``, ``mid_node_id``,
              ``end_node_id``, and ``turn_angle`` (degrees).
            - Count of triplets dropped because coordinate lookup failed.
    """
    adjacency = _build_adjacency(edge_pairs)
    first_leg = adjacency.rename(columns={"end_node_id": "mid_node_id"})
    second_leg = adjacency.rename(columns={"start_node_id": "mid_node_id"})
    triplets = first_leg.merge(second_leg, on="mid_node_id", how="inner")

    merged = _attach_coords(triplets, node_coords, "start_node_id", "start")
    merged = _attach_coords(merged, node_coords, "mid_node_id", "mid")
    merged = _attach_coords(merged, node_coords, "end_node_id", "end")

    coord_cols = [
        "start_lat",
        "start_lon",
        "mid_lat",
        "mid_lon",
        "end_lat",
        "end_lon",
    ]
    missing = merged[coord_cols].isna().any(axis=1)
    invalid = int(missing.sum())
    merged = merged.loc[~missing].copy()

    bearing_in = _compute_bearing(
        merged["start_lat"], merged["start_lon"], merged["mid_lat"], merged["mid_lon"]
    )
    bearing_out = _compute_bearing(
        merged["mid_lat"], merged["mid_lon"], merged["end_lat"], merged["end_lon"]
    )
    merged["turn_angle"] = ((bearing_out - bearing_in + 180.0) % 360.0) - 180.0
    return (
        merged[["start_node_id", "mid_node_id", "end_node_id", "turn_angle"]],
        invalid,
    )


def _build_slope_resolver(
    allow_fallback: bool,
    all_coords: pd.DataFrame,
    cache_dir: str | None,
    product: str,
    max_download_tiles: int,
    clip_margin: float,
) -> tuple[callable | None, Exception | None]:
    """
    Build a bilinearly-interpolating elevation lookup from a DEM tile.

    Downloads (or reuses a cached) DEM raster covering the bounding box of
    ``all_coords``, reads it into memory, then returns a closure that maps
    ``(lon, lat)`` tuples to interpolated elevation values (metres).

    Args:
        allow_fallback: If ``True``, return ``(None, exc)`` on failure instead
            of re-raising.
        all_coords: DataFrame with ``lon`` and ``lat`` columns used to determine
            the clipping bounds of the DEM download.
        cache_dir: Root directory for elevation tile caches. ``None`` defers to
            the ``elevation`` library default.
        product: DEM product identifier passed to ``elevation.clip()``
            (e.g. ``"SRTM1"``).
        max_download_tiles: Maximum number of tiles ``elevation.clip()`` may
            request in a single call.
        clip_margin: Extra degrees added around the bounding box when clipping
            the DEM.

    Returns:
        tuple[callable | None, Exception | None]:
            - Callable ``(lon_lat) -> float`` on success, or ``None`` on
              failure.
            - ``None`` on success, or the caught ``Exception`` on failure.

    Raises:
        ModuleNotFoundError: If ``elevation`` or ``osgeo`` are not installed
            and ``allow_fallback`` is ``False``.
        RuntimeError: If the DEM raster cannot be opened or clipped and
            ``allow_fallback`` is ``False``.
    """
    try:
        import elevation
        from elevation import clip
        from osgeo import gdal
    except ModuleNotFoundError as exc:
        if allow_fallback:
            return None, exc
        raise

    if cache_dir:
        elevation.CACHE_DIR = str(cache_dir)
    else:
        cache_dir = elevation.CACHE_DIR
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    bounds = (
        float(all_coords["lon"].min() - clip_margin),
        float(all_coords["lat"].min() - clip_margin),
        float(all_coords["lon"].max() + clip_margin),
        float(all_coords["lat"].max() + clip_margin),
    )

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_file:
        dem_path = tmp_file.name

    def _cleanup_dem() -> None:
        try:
            Path(dem_path).unlink()
        except Exception:
            pass

    try:
        clip(
            bounds,
            output=dem_path,
            cache_dir=str(cache_dir),
            product=product,
            max_download_tiles=max_download_tiles,
            margin="0" if clip_margin == 0 else str(clip_margin),
        )
        ds = gdal.Open(dem_path)
        if ds is None:
            raise RuntimeError(f"Failed to open elevation raster: {dem_path}")
        arr = ds.ReadAsArray().astype("float32")
        gt = ds.GetGeoTransform()
        no_data = ds.GetRasterBand(1).GetNoDataValue()
        height, width = arr.shape[:2]
        del ds
    except Exception as exc:
        _cleanup_dem()
        if allow_fallback:
            return None, exc
        raise

    _cleanup_dem()

    def _resolve(
        lon_lat: tuple[float, float],
    ) -> float:
        """
        Return bilinearly-interpolated elevation (metres) for a coordinate.

        Args:
            lon_lat: ``(longitude, latitude)`` pair in decimal degrees.

        Returns:
            float: Interpolated elevation in metres.

        Raises:
            RuntimeError: If the coordinate falls outside the DEM bounds or
                the surrounding pixels contain no-data values.
        """
        lon, lat = lon_lat
        x = (lon - gt[0]) / gt[1]
        y = (lat - gt[3]) / gt[5]
        x0 = math.floor(x)
        y0 = math.floor(y)
        x1 = x0 + 1
        y1 = y0 + 1
        if x0 < 0 or x1 >= width or y0 < 0 or y1 >= height:
            raise RuntimeError(f"Coordinate out of DEM bounds: lon={lon}, lat={lat}")
        q11 = float(arr[y0, x0])
        q21 = float(arr[y0, x1])
        q12 = float(arr[y1, x0])
        q22 = float(arr[y1, x1])
        values = (q11, q21, q12, q22)
        if any(np.isnan(v) for v in values):
            raise RuntimeError(f"No elevation data at lon={lon}, lat={lat}")
        if no_data is not None and any(v == float(no_data) for v in values):
            raise RuntimeError(f"No elevation data at lon={lon}, lat={lat}")
        dx = x - x0
        dy = y - y0
        return (
            q11 * (1.0 - dx) * (1.0 - dy)
            + q21 * dx * (1.0 - dy)
            + q12 * (1.0 - dx) * dy
            + q22 * dx * dy
        )

    return _resolve, None


def _build_slope_rows(
    edge_pairs: pd.DataFrame,
    node_coords: pd.DataFrame,
    allow_fallback: bool,
    cache_dir: str | None,
    product: str,
    max_download_tiles: int,
    clip_margin: float,
    min_baseline_m: float,
) -> tuple[pd.DataFrame, int, int]:
    """
    Compute road slope (degrees) for every edge in the undirected adjacency.

    Slope is ``arctan((end_elev - start_elev) / baseline)`` where ``baseline``
    is the haversine distance between the two nodes, clamped to
    ``min_baseline_m``. Each unique node's elevation is resolved only once,
    and the haversine + slope reduction is fully vectorised.

    Args:
        edge_pairs: DataFrame with ``start_node_id`` and ``end_node_id`` columns.
        node_coords: DataFrame with ``node_id``, ``lon``, and ``lat`` columns.
        allow_fallback: If ``True``, use ``0.0`` for edges where elevation
            lookup fails rather than storing ``NaN``.
        cache_dir: Root directory for elevation tile caches.
        product: DEM product identifier (e.g. ``"SRTM1"``).
        max_download_tiles: Maximum tiles requested in a single DEM clip call.
        clip_margin: Extra degrees added around the coordinate bounding box.
        min_baseline_m: Minimum horizontal distance (metres) used as the
            denominator when computing slope.

    Returns:
        tuple[pd.DataFrame, int, int]: A three-element tuple of:
            - DataFrame with columns ``start_node_id``, ``end_node_id``, and
              ``road_slope`` (degrees, signed).
            - Number of edges skipped due to missing coordinate data.
            - Number of edges where elevation lookup failed.

    Raises:
        RuntimeError: If the DEM resolver cannot be initialised and
            ``allow_fallback`` is ``False``.
    """
    adjacency = _build_adjacency(edge_pairs)
    merged = adjacency.merge(
        node_coords.rename(
            columns={"node_id": "start_node_id", "lat": "start_lat", "lon": "start_lon"}
        ),
        on="start_node_id",
        how="inner",
    ).merge(
        node_coords.rename(
            columns={"node_id": "end_node_id", "lat": "end_lat", "lon": "end_lon"}
        ),
        on="end_node_id",
        how="inner",
    )

    missing = (
        merged[["start_lat", "start_lon", "end_lat", "end_lon"]].isna().any(axis=1)
    )
    skipped = int(missing.sum())
    merged = merged.loc[~missing].copy()
    if merged.empty:
        merged["road_slope"] = []
        return merged[["start_node_id", "end_node_id", "road_slope"]], skipped, 0

    all_coords = pd.concat(
        [
            merged[["start_lon", "start_lat"]].rename(
                columns={"start_lon": "lon", "start_lat": "lat"}
            ),
            merged[["end_lon", "end_lat"]].rename(
                columns={"end_lon": "lon", "end_lat": "lat"}
            ),
        ],
        ignore_index=True,
    )

    resolver, resolver_err = _build_slope_resolver(
        allow_fallback,
        all_coords,
        cache_dir,
        product,
        max_download_tiles,
        clip_margin,
    )
    if resolver is None:
        raise RuntimeError(
            "Failed to initialize elevation data source (network/cache error). "
            "Slope values could not be computed; fix elevation tile access before retrying."
        ) from resolver_err

    unique_nodes = pd.concat(
        [
            merged[["start_node_id", "start_lon", "start_lat"]].rename(
                columns={
                    "start_node_id": "node_id",
                    "start_lon": "lon",
                    "start_lat": "lat",
                }
            ),
            merged[["end_node_id", "end_lon", "end_lat"]].rename(
                columns={
                    "end_node_id": "node_id",
                    "end_lon": "lon",
                    "end_lat": "lat",
                }
            ),
        ],
        ignore_index=True,
    ).drop_duplicates("node_id")

    elev_map: dict = {}
    for node_id, lon, lat in unique_nodes.itertuples(index=False):
        try:
            elev_map[node_id] = resolver((float(lon), float(lat)))
        except Exception:
            elev_map[node_id] = np.nan

    start_elev = merged["start_node_id"].map(elev_map).to_numpy(dtype=float)
    end_elev = merged["end_node_id"].map(elev_map).to_numpy(dtype=float)
    edge_failed = np.isnan(start_elev) | np.isnan(end_elev)
    failed = int(edge_failed.sum())

    lat1 = np.radians(merged["start_lat"].to_numpy(dtype=float))
    lon1 = np.radians(merged["start_lon"].to_numpy(dtype=float))
    lat2 = np.radians(merged["end_lat"].to_numpy(dtype=float))
    lon2 = np.radians(merged["end_lon"].to_numpy(dtype=float))
    hav = (
        np.sin((lat2 - lat1) / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2
    )
    distance_m = 2.0 * 6371.0 * np.arctan2(np.sqrt(hav), np.sqrt(1.0 - hav)) * 1000.0
    baseline = np.maximum(distance_m, min_baseline_m)

    with np.errstate(invalid="ignore"):
        slopes = np.degrees(np.arctan((end_elev - start_elev) / baseline))
    if allow_fallback:
        slopes = np.where(edge_failed, 0.0, slopes)

    merged["road_slope"] = slopes
    return merged[["start_node_id", "end_node_id", "road_slope"]], skipped, failed


def _write_nl(
    path: Path,
    frame: pd.DataFrame,
) -> int:
    """
    Write a DataFrame to a space-separated newline-delimited text file.

    Creates parent directories as needed.

    Args:
        path: Destination file path.
        frame: DataFrame whose rows are written as space-separated values.

    Returns:
        int: Number of rows written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in frame.itertuples(index=False, name=None):
            f.write(" ".join(map(str, row)) + "\n")
    return len(frame)


def _validate_and_report(
    total_invalid: int,
    mode: str,
    skip_invalid: bool,
) -> None:
    """
    Raise an error if invalid rows were found and skipping is not permitted.

    Args:
        total_invalid: Number of invalid/missing rows detected.
        mode: Processing mode label used in the error message (e.g. ``"angle"``).
        skip_invalid: If ``True``, invalid rows are silently tolerated.

    Raises:
        ValueError: If ``total_invalid > 0`` and ``skip_invalid`` is ``False``.
    """
    if total_invalid and not skip_invalid:
        raise ValueError(
            f"Found {total_invalid} invalid rows while building {mode}. Pass --skip-invalid to ignore."
        )


def _resolve_target(
    kind: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> Path:
    """
    Determine the output path for a given artifact ``kind``.

    Uses ``args.output`` when running a single-mode invocation (so the user
    can override the file path directly) and ``args.filename_template`` under
    ``output_dir`` when running in ``all`` mode.

    Args:
        kind: Artifact kind (``"angle"`` or ``"slope"``).
        args: Parsed command-line arguments.
        output_dir: Resolved output directory.

    Returns:
        Path: Destination path for the artifact file.
    """
    if args.mode == kind:
        return Path(args.output.format(kind=kind))
    return output_dir / args.filename_template.format(kind=kind)


def main() -> int:
    """
    Entry point: parse arguments, generate requested road-structure artifacts.

    Orchestrates loading of node/edge CSVs, building angle and/or slope node
    lists according to ``--mode``, and writing ``.list`` output files.

    Returns:
        int: Exit code (``0`` on success).

    Raises:
        FileNotFoundError: If a required input CSV is missing.
        KeyError: If required columns are absent from an input CSV.
        ValueError: If invalid rows are detected and ``--skip-invalid`` is not
            set.
        RuntimeError: If slope computation fails and no fallback is enabled.
    """
    args = parse_args()

    node_df = _parse_node_coords(args.node_input)
    edge_df = _read_csv(args.edge_input, [args.start_id_col, args.end_id_col])
    edge_pairs = _normalize_node_pairs(
        edge_df, args.start_id_col, args.end_id_col, args.node2node_col
    )
    edge_pairs["start_node_id"] = _normalize_ids(edge_pairs["start_node_id"])
    edge_pairs["end_node_id"] = _normalize_ids(edge_pairs["end_node_id"])
    edge_pairs = edge_pairs.dropna()

    output_dir = Path(args.output_dir)
    produced: list[str] = []
    total_invalid = 0

    if args.mode in {"angle", "all"}:
        angle_rows, invalid_angle = _build_angle_rows(edge_pairs, node_df)
        _validate_and_report(invalid_angle, "angle", args.skip_invalid)
        if args.skip_invalid:
            valid_mask = angle_rows.notna().all(axis=1)
            total_invalid += int((~valid_mask).sum())
            angle_rows = angle_rows.loc[valid_mask]
        angle_target = _resolve_target("angle", args, output_dir)
        rows_written = _write_nl(angle_target, angle_rows)
        produced.append(f"{angle_target} ({rows_written} rows)")

    if args.mode in {"slope", "all"} and not args.skip_slope:
        slope_rows, skipped, failed = _build_slope_rows(
            edge_pairs,
            node_df,
            args.slope_zero_fallback,
            args.elevation_cache_dir,
            args.elevation_product,
            args.elevation_max_tiles,
            args.elevation_clip_margin,
            args.slope_min_baseline_m,
        )
        _validate_and_report(skipped, "slope", args.skip_invalid)
        if failed and not args.slope_zero_fallback:
            raise RuntimeError(
                f"Slope computation failed for {failed} edges due to elevation lookup issues."
            )
        if args.skip_invalid:
            valid_mask = slope_rows.notna().all(axis=1)
            total_invalid += int((~valid_mask).sum())
            slope_rows = slope_rows.loc[valid_mask]
        if (
            args.slope_zero_fallback
            and failed > 0
            and failed == len(slope_rows)
            and (
                pd.to_numeric(slope_rows["road_slope"], errors="coerce").fillna(0.0)
                == 0.0
            ).all()
        ):
            raise RuntimeError(
                "Slope data could not be computed from elevation source (all rows are 0.0). "
                "This is not a valid computed slope; check DEM tile access and rerun."
            )
        slope_target = _resolve_target("slope", args, output_dir)
        rows_written = _write_nl(slope_target, slope_rows)
        if args.slope_zero_fallback and slope_rows["road_slope"].isna().all():
            print(
                "Warning: all slope values are missing (DEM not available). "
                "This file contains placeholders, not measured elevation slopes."
            )
        produced.append(f"{slope_target} ({rows_written} rows)")

    if args.mode == "all" and args.skip_slope:
        produced.append("slope skipped (--skip-slope)")

    if total_invalid:
        print(f"Skipped/invalid rows: {total_invalid}")
    print("Generated:")
    for item in produced:
        print(f" - {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
