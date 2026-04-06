"""
Collect Hong Kong road information from overview.hk and save it as CSV.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

DEFAULT_URL = "https://www.overview.hk/street/ssp.php"
DEFAULT_OUTPUT = "data/raw/roads_data_raw.csv"


class RoadDataCollectorError(RuntimeError):
    """
    Raised for expected collection failures.
    """


def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Args:
        argv: Argument list to parse. Defaults to ``sys.argv[1:]`` when *None*.

    Returns:
        Parsed namespace with attributes ``url``, ``output``, ``timeout``,
        ``retries``, ``backoff``, and ``no_dedupe``.
    """
    parser = argparse.ArgumentParser(
        description="Fetch Hong Kong road info from overview.hk and write CSV output."
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="Source URL that returns JSON with a 'data' field.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output CSV path (default: %(default)s)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="Request timeout in seconds.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of attempts on transient request/parse failures.",
    )
    parser.add_argument(
        "--backoff",
        type=float,
        default=0.5,
        help="Base delay for exponential backoff between retries.",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Disable dropping duplicate road rows.",
    )
    return parser.parse_args(argv)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def fetch_payload(
    session: requests.Session,
    url: str,
    timeout: float,
    retries: int,
    backoff: float,
    logger: logging.Logger,
) -> dict[str, Any]:
    """
    Fetch JSON payload from *url* with retry and exponential backoff.

    Args:
        session: Reusable HTTP session.
        url: Endpoint that returns JSON with a ``data`` field.
        timeout: Per-request timeout in seconds.
        retries: Maximum number of attempts.
        backoff: Base delay (seconds) for exponential backoff.
        logger: Logger instance for status messages.

    Returns:
        Parsed JSON dict containing a ``data`` key.
    """
    last_exception: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            logger.debug("Fetching %s (attempt %s/%s)", url, attempt, retries)
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            payload = response.json()

            if not isinstance(payload, dict) or "data" not in payload:
                raise RoadDataCollectorError(
                    "Unexpected JSON payload format: missing 'data'."
                )

            return payload

        except (requests.RequestException, ValueError, RoadDataCollectorError) as exc:
            last_exception = exc
            if attempt == retries:
                break
            delay = backoff * (2 ** (attempt - 1))
            logger.warning(
                "Attempt %s failed (%s). Retrying in %.2fs",
                attempt,
                exc,
                delay,
            )
            time.sleep(delay)

    raise last_exception  # type: ignore[arg-type]


def normalize_data(
    payload: dict[str, Any],
    *,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Convert a raw JSON payload into a tidy DataFrame of road records.

    Args:
        payload: Parsed JSON dict whose ``data`` value is a list of rows.
        logger: Logger instance for status messages.

    Returns:
        DataFrame with columns ``chinese_road_name``, ``chinese_district_name``,
        ``english_road_name``, and ``english_district_name``.
    """
    raw_rows = payload.get("data")
    if not isinstance(raw_rows, list):
        raise RoadDataCollectorError("Payload 'data' must be a list.")

    rows: list[dict[str, str]] = []
    for detail in raw_rows:
        if not isinstance(detail, (list, tuple)) or len(detail) < 4:
            logger.debug("Skipping malformed row: %s", detail)
            continue

        rows.append(
            {
                "chinese_road_name": str(detail[0]),
                "chinese_district_name": str(detail[2]),
                "english_road_name": str(detail[1]),
                "english_district_name": str(detail[3]),
            }
        )

    if not rows:
        raise RoadDataCollectorError("No valid road rows found in payload.")

    return pd.DataFrame(rows)


def save_dataframe(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Write *df* to *output_path* as CSV, creating parent directories as needed.

    Args:
        df: DataFrame to persist.
        output_path: Destination file path; parent directories are created if absent.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def setup_logger() -> logging.Logger:
    """
    Create and configure a module-level logger.

    Returns:
        Logger named ``collect_roads_data`` with a timestamped stream handler
        at ``INFO`` level.
    """
    logger = logging.getLogger("collect_roads_data")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def main(
    argv: list[str] | None = None,
) -> int:
    """
    Entry point for the road data collector.

    Args:
        argv: CLI arguments. Defaults to ``sys.argv[1:]`` when *None*.

    Returns:
        ``0`` on success, ``1`` on failure.
    """
    args = parse_args(argv)
    logger = setup_logger()

    project_root = _project_root()
    output_path = (project_root / args.output).resolve()

    logger.info("Project root: %s", project_root)
    logger.info("Output path: %s", output_path)

    session = requests.Session()
    try:
        payload = fetch_payload(
            session, args.url, args.timeout, args.retries, args.backoff, logger
        )
        df = normalize_data(payload, logger=logger)

        if not args.no_dedupe:
            df = df.drop_duplicates()

        logger.info("Writing %s rows to CSV", len(df))
        save_dataframe(df, output_path)
        logger.info("Done")
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to collect road data: %s", exc)
        return 1
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
