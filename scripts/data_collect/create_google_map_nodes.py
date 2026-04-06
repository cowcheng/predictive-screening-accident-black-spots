"""
Extract start/end node coordinates from road segment LINESTRING geometries.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """
    Parse command-line arguments for Google Map node extraction.

    Args:
        argv: Explicit argument list. Defaults to ``sys.argv[1:]`` when ``None``.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Build a node-to-node Google Map style CSV from OSM edge geometries."
    )
    parser.add_argument(
        "--input",
        default="data/raw/osm_edge_info.csv",
        help="Input edge table containing geometry column.",
    )
    parser.add_argument(
        "--output",
        default="data/raw/google_map_data_raw.csv",
        help="Output path for parsed start/end coordinates.",
    )
    parser.add_argument(
        "--geometry-col",
        default="geometry",
        help="Column name holding LINESTRING geometry values.",
    )
    parser.add_argument(
        "--start-id-col",
        default="u",
        help="Edge source-node identifier column.",
    )
    parser.add_argument(
        "--end-id-col",
        default="v",
        help="Edge target-node identifier column.",
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip rows with empty/invalid geometry instead of failing.",
    )
    parser.add_argument(
        "--validate-unique",
        action="store_true",
        help="Drop duplicate start-end pairs.",
    )
    return parser.parse_args(argv)


def _parse_linestring(
    value: object,
) -> tuple[float, float, float, float]:
    """
    Parse a WKT LINESTRING into start and end coordinates.

    Args:
        value: Raw geometry value (WKT string, ``None``, or ``float('nan')``).

    Returns:
        A ``(start_lon, start_lat, end_lon, end_lat)`` tuple of floats.

    Raises:
        ValueError: If the geometry is empty, malformed, or unsupported.
    """
    if pd.isna(value):
        raise ValueError("empty geometry")

    text = str(value).strip()
    if not text:
        raise ValueError("empty geometry")

    if not re.match(r"^\s*linestring\s*\(", text, flags=re.IGNORECASE):
        raise ValueError("unsupported geometry format")

    try:
        coord_text = text[text.index("(") + 1 : text.rindex(")")]
    except ValueError as exc:
        raise ValueError("malformed geometry") from exc

    points = [segment.strip() for segment in coord_text.split(",") if segment.strip()]
    if len(points) < 2:
        raise ValueError("insufficient coordinate points")

    def _xy(
        point: str,
    ) -> tuple[float, float]:
        """
        Parse a space-separated coordinate string into (longitude, latitude).

        Args:
            point: A single coordinate string in ``"lon lat"`` format.

        Returns:
            A ``(longitude, latitude)`` tuple of floats.

        Raises:
            ValueError: If the string contains fewer than two whitespace-separated parts.
        """
        parts = point.split()
        if len(parts) < 2:
            raise ValueError("invalid coordinate part")
        return float(parts[0]), float(parts[1])

    start_lon, start_lat = _xy(points[0])
    end_lon, end_lat = _xy(points[-1])
    return start_lon, start_lat, end_lon, end_lat


def run_collection(
    args: argparse.Namespace,
) -> int:
    """
    Read edge geometries from CSV, parse coordinates, and write output.

    Args:
        args: Parsed CLI arguments from ``parse_args``.

    Returns:
        Exit code (``0`` on success).
    """
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    input_df = pd.read_csv(input_path)
    for column in (args.geometry_col, args.start_id_col, args.end_id_col):
        if column not in input_df.columns:
            raise KeyError(f"Missing required column: {column}")

    rows = []
    total_skipped = 0
    for _, row in input_df.iterrows():
        start_node = row[args.start_id_col]
        end_node = row[args.end_id_col]

        try:
            start_lon, start_lat, end_lon, end_lat = _parse_linestring(
                row[args.geometry_col]
            )
        except Exception:
            total_skipped += 1
            if args.skip_invalid:
                continue
            raise

        rows.append(
            {
                "start_node": str(start_node),
                "end_node": str(end_node),
                "edge_id": f"{start_node}->{end_node}",
                "start_long": start_lon,
                "start_lat": start_lat,
                "end_long": end_lon,
                "end_lat": end_lat,
            }
        )

    google_map_df = pd.DataFrame.from_records(rows)

    if args.validate_unique:
        google_map_df = google_map_df.drop_duplicates(
            subset=[
                "start_node",
                "end_node",
                "start_long",
                "start_lat",
                "end_long",
                "end_lat",
            ]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    google_map_df.to_csv(output_path, index=False)

    if total_skipped:
        print(f"Skipped {total_skipped} invalid rows.")
    print(f"Wrote {len(google_map_df)} rows to {output_path}")
    return 0


def main(
    argv: list[str] | None = None,
) -> int:
    """
    CLI entry point for Google Map node extraction.

    Args:
        argv: Explicit argument list. Defaults to ``sys.argv[1:]`` when ``None``.

    Returns:
        Exit code (``0`` on success).
    """
    return run_collection(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
