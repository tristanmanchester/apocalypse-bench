#!/usr/bin/env python3
"""Materialize the Markdown Wikipedia dataset as JSONL for wiki-search."""

from __future__ import annotations

import argparse
from datetime import date, datetime
import json
from pathlib import Path
from typing import Any


DATASET = "marin-community/wikipedia-markdown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke runs")
    args = parser.parse_args()

    from datasets import load_dataset

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(DATASET, split="train", streaming=True)
    count = 0
    with out.open("w", encoding="utf-8") as fh:
        for row in dataset:
            fh.write(json.dumps(normalize_row(row), ensure_ascii=False, default=json_default) + "\n")
            count += 1
            if args.limit is not None and count >= args.limit:
                break

    print(json.dumps({"dataset": DATASET, "out": str(out), "rows": count}, indent=2))


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "url": row.get("url"),
        "title": row.get("title"),
        "abstract": row.get("abstract"),
        "date_created": row.get("date_created"),
        "text": row.get("text"),
    }


def json_default(value: Any) -> str:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


if __name__ == "__main__":
    main()
