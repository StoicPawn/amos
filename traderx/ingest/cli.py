"""Command line helpers for spot research tasks."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from traderx.ingest.spot import DownloadRequest, HistoricalDownloader


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download historical data for ad-hoc analysis")
    parser.add_argument("--asset", required=True, help="Symbol to download, e.g. AAPL")
    parser.add_argument("--market", default="SMART", help="Market or exchange identifier")
    parser.add_argument("--timeframe", default="1 day", help="Bar size (e.g. '1 day', '1 hour', '15 min')")
    parser.add_argument("--start", help="Start date (ISO format, UTC assumed)")
    parser.add_argument("--end", help="End date (ISO format, UTC assumed)")
    parser.add_argument("--output", help="Destination CSV file")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    request = DownloadRequest(
        symbol=args.asset,
        market=args.market,
        timeframe=args.timeframe,
        start=args.start,
        end=args.end,
    )

    downloader = HistoricalDownloader(default_market=args.market)
    frame = downloader.download(request)

    if args.output:
        output_path = Path(args.output)
    else:
        idx = frame.index.to_list()
        start_label = idx[0].strftime("%Y%m%d") if idx else ""
        end_label = idx[-1].strftime("%Y%m%d") if idx else ""
        output_name = f"{args.asset}_{args.timeframe.replace(' ', '')}_{start_label}_{end_label}.csv"
        output_path = Path.cwd() / output_name

    downloader.download_to_csv(request, output_path, frame=frame)
    print(f"Saved {len(frame)} bars for {args.asset} to {output_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
