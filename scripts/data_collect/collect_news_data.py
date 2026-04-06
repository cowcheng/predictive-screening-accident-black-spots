"""
Collect RTHK road traffic news and extract entries that reference road names.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import logging
import re
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


class RoadParserError(RuntimeError):
    """
    Raised for expected parsing/collection failures.
    """


class InnerListParser(HTMLParser):
    """
    Extract text from ``<li class="inner">`` entries inside a ``.articles`` div.

    This is a pure-stdlib alternative to BeautifulSoup for environments where
    bs4 is unavailable or explicitly disabled via ``--no-bs4``.
    """

    def __init__(
        self,
    ) -> None:
        """
        Initialise parser state.

        Returns:
            None.
        """
        super().__init__()
        self._in_articles = False
        self._in_item = False
        self._chunks: list[str] = []
        self._articles: list[str] = []

    @staticmethod
    def _has_class(
        attrs: list[tuple[str, str | None]],
        class_name: str,
    ) -> bool:
        """
        Check whether a tag's attribute list contains a given CSS class.

        Args:
            attrs: List of (attribute, value) pairs from the tag.
            class_name: CSS class name to look for.

        Returns:
            ``True`` if ``class_name`` appears in the tag's ``class`` attribute,
            ``False`` otherwise.
        """
        for key, value in attrs:
            if key == "class" and value and class_name in value.split():
                return True
        return False

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        """
        Detect entry into the relevant HTML sections.

        Args:
            tag: The name of the tag (e.g. "div", "li").
            attrs: List of (attribute, value) pairs for the tag.

        Returns:
            None. Updates internal state to track whether we're inside the target.
        """
        if tag == "div" and self._has_class(attrs, "articles"):
            self._in_articles = True
            return

        if self._in_articles and tag == "li" and self._has_class(attrs, "inner"):
            self._in_item = True
            self._chunks = []

    def handle_endtag(
        self,
        tag: str,
    ) -> None:
        """
        Detect exit from relevant HTML sections and finalize article text.

        Args:
            tag: The name of the closing tag (e.g. "div", "li").

        Returns:
            None. If exiting an article item, joins collected text chunks and stores
            the result in the articles list. If exiting the articles div, clears the
            tracking flag.
        """
        if tag == "li" and self._in_item:
            text = "".join(self._chunks).strip()
            if text:
                self._articles.append(text)
            self._in_item = False
            self._chunks = []
            return

        if tag == "div" and self._in_articles:
            self._in_articles = False

    def handle_data(
        self,
        data: str,
    ) -> None:
        """
        Collect text data when inside an article item.

        Args:
            data: A chunk of text data from the HTML content.

        Returns:
            None. Appends data to the current article text if we're inside an item.
        """
        if self._in_item:
            self._chunks.append(data)

    def get_articles(
        self,
    ) -> list[str]:
        """
        Return all extracted article texts.

        Returns:
            A list of strings, each representing an extracted article text.
        """
        return self._articles


@dataclass(frozen=True)
class CollectorConfig:
    """
    Shared configuration for the news collection pipeline.

    Attrs:
        url_template: RTHK URL with ``{date}`` placeholder (YYYYMMDD).
        timeout: HTTP request timeout in seconds.
        retries: Maximum retry attempts per request.
        backoff: Base delay (seconds) for exponential backoff between retries.
        sleep: Optional throttle delay (seconds) between requests.
        user_agent: HTTP User-Agent header value.
        prefer_bs4: Use BeautifulSoup for HTML parsing when True.
        max_workers: Thread pool size (1 disables threading).
    """

    url_template: str
    timeout: float
    retries: int
    backoff: float
    sleep: float
    user_agent: str
    prefer_bs4: bool
    max_workers: int


@dataclass(frozen=True)
class NewsRecord:
    """
    A single traffic news entry matched to a road name.

    Attrs:
        date: Publication date string (e.g. ``2024/01/15``).
        time: Publication time in HKT (e.g. ``16:30``).
        location: Matched road name (Chinese).
        detail: Full article text.
    """

    date: str
    time: str
    location: str
    detail: str


# ── Regex used by parse_article_text ──────────────────────────────────────────
_HKT_RE = re.compile(
    r"^(?P<detail>.+?)\s+(?P<date>\d{4}/\d{2}/\d{2})\s+HKT\s+(?P<time>\d{1,2}:\d{2})"
)


def parse_args(
    argv: Sequence[str],
) -> argparse.Namespace:
    """
    Parse CLI arguments for the news collector.

    Args:
        argv: Sequence of command-line argument strings (excluding the program name).

    Returns:
        Parsed argument namespace with attributes for each CLI option.
    """
    parser = argparse.ArgumentParser(
        description="Fetch RTHK traffic news and extract rows for matching road names."
    )
    parser.add_argument(
        "--source-road-file",
        default="data/raw/roads_data_raw.csv",
        help="Road source CSV (must contain chinese_district_name).",
    )
    parser.add_argument(
        "--fallback-road-file",
        default="data/raw/road_info.csv",
        help="Fallback road source if --source-road-file does not exist.",
    )
    parser.add_argument(
        "--output",
        default="data/raw/news_data_raw.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--start-date",
        default="2010-01-01",
        help="First date to query in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Last date to query in YYYY-MM-DD format. Takes precedence over --days.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=4018,
        help="Fallback: number of days to scan from start-date when --end-date is not set.",
    )
    parser.add_argument(
        "--url",
        default="https://programme.rthk.hk/channel/radio/trafficnews/index.php?d={date}",
        help="RTHK URL template with {date} placeholder as YYYYMMDD.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Request retry attempts per date.",
    )
    parser.add_argument(
        "--backoff",
        type=float,
        default=0.75,
        help="Base backoff delay for retries.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional sleep seconds between requests for throttling.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="Parallel workers (1 disables threading).",
    )
    parser.add_argument(
        "--user-agent",
        default="traffic-news-collector/1.0",
        help="HTTP User-Agent.",
    )
    parser.add_argument(
        "--no-bs4",
        action="store_true",
        help="Force pure-stdlib parser even if BeautifulSoup is installed.",
    )
    parser.add_argument(
        "--disable-dedupe",
        action="store_true",
        help="Keep duplicate rows (including duplicate road matches).",
    )
    return parser.parse_args(list(argv))


def setup_logger() -> logging.Logger:
    """
    Configure and return the module logger.

    Returns:
        A logging.Logger instance configured for console output with timestamps.
    """
    logger = logging.getLogger("collect_news_data")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def _project_root() -> Path:
    """
    Determine the absolute path to the project root directory.

    Returns:
        A Path object pointing to the project root, assumed to be two levels up from this script
    """
    return Path(__file__).resolve().parents[2]


def load_roads(
    project_root: Path,
    source: str,
    fallback: str,
    logger: logging.Logger,
) -> list[str]:
    """
    Load and deduplicate road names from a CSV file.

    Args:
        project_root: Absolute path to the project root directory.
        source: Relative path to the primary road CSV.
        fallback: Relative path to the fallback road CSV.
        logger: Logger instance.

    Returns:
        Sorted list of unique road name strings (longest first).

    Raises:
        FileNotFoundError: Neither source nor fallback CSV exists.
        RoadParserError: CSV lacks the expected column or contains no valid names.
    """
    source_path = project_root / source
    fallback_path = project_root / fallback

    if not source_path.exists():
        if fallback_path.exists():
            logger.warning("Primary road file not found: %s", source_path)
            logger.warning("Using fallback road file: %s", fallback_path)
            source_path = fallback_path
        else:
            raise FileNotFoundError(
                f"Road source not found: {source_path}, fallback missing too: {fallback_path}"
            )

    df = pd.read_csv(source_path)
    if "chinese_district_name" not in df.columns:
        raise RoadParserError(
            f"Expected column 'chinese_district_name' in {source_path.name}"
        )

    roads = (
        df["chinese_district_name"]
        .astype(str)
        .str.strip()
        .loc[lambda s: s.ne("") & s.ne("nan")]
        .unique()
    )

    if len(roads) == 0:
        raise RoadParserError(f"No valid road names found in {source_path.name}")

    return sorted(roads, key=len, reverse=True)


def build_road_pattern(
    roads: Sequence[str],
) -> re.Pattern[str]:
    """
    Compile a regex that matches any of the given road names.

    Args:
        roads: Road name strings to match against.

    Returns:
        Compiled regex with a single capturing group for the matched name.
    """
    escaped = [re.escape(road) for road in roads]
    return re.compile("(" + "|".join(escaped) + ")")


def parse_article_text(
    raw: str,
) -> tuple[str, str, str] | None:
    """
    Extract (detail, date, time) from a raw RTHK article string.

    The expected format ends with ``<detail> YYYY/MM/DD HKT HH:MM``.

    Args:
        raw: Raw article text, possibly with HTML entities and extra whitespace.

    Returns:
        A ``(detail, date, time)`` tuple, or ``None`` if the text cannot be parsed.
    """
    cleaned = html.unescape(re.sub(r"\s+", " ", raw)).strip()
    if not cleaned:
        return None

    # Primary: single-line regex (handles the common non-tab case).
    match = _HKT_RE.match(cleaned)
    if match:
        return match.group("detail").strip(), match.group("date"), match.group("time")

    # Fallback: tab-delimited layout where the last segment holds the timestamp.
    parts = cleaned.split("\t")
    if len(parts) >= 2 and " HKT " in parts[-1]:
        tail = parts[-1].strip()
        date_part, hkt_part = tail.rsplit(" HKT ", 1)
        time_text = hkt_part.strip().split()[0]
        if len(date_part) >= 8 and date_part[0].isdigit() and ":" in time_text:
            return parts[0].strip() or date_part.strip(), date_part.strip(), time_text

    return None


def get_articles(
    html_text: str,
    prefer_bs4: bool,
) -> list[str]:
    """
    Extract article texts from an RTHK traffic news HTML page.

    Args:
        html_text: Raw HTML response body.
        prefer_bs4: Use BeautifulSoup when ``True``, stdlib HTMLParser otherwise.

    Returns:
        List of raw article text strings.
    """
    if prefer_bs4:
        soup = BeautifulSoup(html_text, "html.parser")
        block = soup.find("div", {"class": "articles"})
        if block is None:
            return []
        return [
            node.get_text(" ", strip=True)
            for node in block.find_all("li", {"class": "inner"})
        ]

    parser = InnerListParser()
    parser.feed(html_text)
    return parser.get_articles()


def extract_matching_records(
    articles: Sequence[str],
    road_pattern: re.Pattern[str],
    logger: logging.Logger,
) -> list[NewsRecord]:
    """
    Match parsed articles against known road names.

    Args:
        articles: Raw article text strings from a single page.
        road_pattern: Compiled regex with road name alternatives.
        logger: Logger instance.

    Returns:
        List of ``NewsRecord`` instances — one per (article, road) match.
    """
    records: list[NewsRecord] = []

    for raw in articles:
        parsed = parse_article_text(raw)
        if parsed is None:
            continue

        detail, date, time_text = parsed
        for hit in road_pattern.findall(detail):
            records.append(
                NewsRecord(date=date, time=time_text, location=str(hit), detail=detail)
            )

    if not records:
        logger.debug("No road matches found in article batch")

    return records


def fetch_with_retries(
    session: requests.Session,
    url: str,
    timeout: float,
    retries: int,
    backoff: float,
    logger: logging.Logger,
) -> requests.Response:
    """
    GET *url* with exponential-backoff retries.

    Args:
        session: Shared ``requests.Session``.
        url: Target URL.
        timeout: Per-request timeout in seconds.
        retries: Maximum number of attempts.
        backoff: Base delay multiplied by ``2^(attempt-1)`` between retries.
        logger: Logger instance.

    Returns:
        Successful ``requests.Response``.

    Raises:
        requests.RequestException: All retry attempts exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_exc = exc
            if attempt == retries:
                break
            delay = backoff * (2 ** (attempt - 1))
            logger.warning(
                "Request failed on %s: %s. Retrying in %.2fs", url, exc, delay
            )
            time.sleep(delay)

    assert last_exc is not None
    raise last_exc


def collect_one_day(
    day_offset: int,
    start_date: dt.date,
    config: CollectorConfig,
    road_pattern: re.Pattern[str],
    logger: logging.Logger,
) -> list[NewsRecord]:
    """
    Fetch and parse traffic news for a single calendar day.

    Args:
        day_offset: Number of days after *start_date* to query.
        start_date: Base date for the collection window.
        config: Collector configuration.
        road_pattern: Compiled regex with road name alternatives.
        logger: Logger instance.

    Returns:
        List of matched ``NewsRecord`` instances (may be empty).
    """
    target_date = start_date + dt.timedelta(days=day_offset)
    day_str = target_date.strftime("%Y%m%d")
    url = config.url_template.format(date=day_str)

    session = requests.Session()
    session.headers.update({"User-Agent": config.user_agent})

    try:
        response = fetch_with_retries(
            session,
            url,
            timeout=config.timeout,
            retries=config.retries,
            backoff=config.backoff,
            logger=logger,
        )
    except requests.RequestException as exc:
        logger.error("Skipping %s due request error: %s", day_str, exc)
        return []
    finally:
        session.close()

    articles = get_articles(response.text, prefer_bs4=config.prefer_bs4)
    if not articles:
        logger.debug("No articles found on %s", day_str)
        return []

    records = extract_matching_records(articles, road_pattern, logger)

    if config.sleep > 0:
        time.sleep(config.sleep)

    return records


def collect_news(
    roads: Sequence[str],
    config: CollectorConfig,
    start_date: dt.date,
    days: int,
    logger: logging.Logger,
) -> list[NewsRecord]:
    """
    Collect traffic news across a date range, optionally in parallel.

    Args:
        roads: Road name strings used to build the matching regex.
        config: Collector configuration.
        start_date: First date to query.
        days: Total number of consecutive days to scan.
        logger: Logger instance.

    Returns:
        Aggregated list of ``NewsRecord`` instances.
    """
    road_pattern = build_road_pattern(roads)
    records: list[NewsRecord] = []
    worker_count = max(1, config.max_workers)

    if worker_count == 1:
        for offset in tqdm(range(days), desc="Collecting news", unit="day"):
            records.extend(
                collect_one_day(offset, start_date, config, road_pattern, logger)
            )
        return records

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                collect_one_day, offset, start_date, config, road_pattern, logger
            ): offset
            for offset in range(days)
        }

        for future in tqdm(as_completed(futures), desc="Collecting news", unit="day"):
            records.extend(future.result())

    return records


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """
    Entry point for the RTHK traffic news collector.

    Args:
        argv: Command-line argument strings. Defaults to ``sys.argv[1:]`` when
            ``None``.

    Returns:
        Exit code: ``0`` on success, ``1`` on error.
    """
    args = parse_args(list(argv) if argv is not None else sys.argv[1:])
    logger = setup_logger()
    project_root = _project_root()
    logger.info("Project root: %s", project_root)

    try:
        roads = load_roads(
            project_root=project_root,
            source=args.source_road_file,
            fallback=args.fallback_road_file,
            logger=logger,
        )
    except Exception as exc:
        logger.error("Failed to load road list: %s", exc)
        return 1

    config = CollectorConfig(
        url_template=args.url,
        timeout=args.timeout,
        retries=args.retries,
        backoff=args.backoff,
        sleep=args.sleep,
        user_agent=args.user_agent,
        prefer_bs4=not args.no_bs4,
        max_workers=args.max_workers,
    )

    try:
        start_date = dt.datetime.strptime(args.start_date, "%Y-%m-%d").date()
        if args.end_date:
            end_date = dt.datetime.strptime(args.end_date, "%Y-%m-%d").date()
            if end_date < start_date:
                raise ValueError("end-date must be on or after start-date")
            days = (end_date - start_date).days + 1
        else:
            days = args.days

        records = collect_news(
            roads=roads,
            config=config,
            start_date=start_date,
            days=days,
            logger=logger,
        )

        output_path = (project_root / args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(
            [record.__dict__ for record in records],
            columns=["date", "time", "location", "detail"],
        )
        if not args.disable_dedupe:
            df = df.drop_duplicates()

        df.to_csv(output_path, index=False)
        logger.info("Wrote %s rows to %s", len(df), output_path)
        return 0
    except Exception as exc:
        logger.error("News collection failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
