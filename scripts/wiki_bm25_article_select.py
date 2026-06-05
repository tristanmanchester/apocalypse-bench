#!/usr/bin/env python3
"""Select a compact article-routed Wikipedia chunk set from BM25 category lexicons."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from wiki_bm25_article_probe import aggregate_articles
from wiki_bm25_lexicon_probe import LEXICON, normalize_query, post_json


def is_infobox_hit(hit: dict[str, Any]) -> bool:
    chunk_id = str(hit.get("chunk_id") or "")
    if chunk_id.endswith(":infobox"):
        return True
    heading_path = hit.get("heading_path") or []
    return bool(heading_path and str(heading_path[0]).lower() == "infobox")


def select_articles_and_chunks(
    *,
    url: str,
    chunk_limit: int,
    article_cutoff: int,
    chunks_per_article: int,
    timeout: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    selected_articles: dict[str, dict[str, Any]] = {}
    chunk_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    category_stats: dict[str, Any] = {}

    for category, lexicon in LEXICON.items():
        query = normalize_query(lexicon)
        started = time.time()
        response = post_json(url, {"query": query, "limit": chunk_limit}, timeout)
        hits = [hit for hit in response.get("hits", []) if not is_infobox_hit(hit)]
        ranked_articles = aggregate_articles(hits)
        selected = ranked_articles[:article_cutoff]

        category_stats[category] = {
            "queryTerms": len(query.split()),
            "chunksFetched": len(response.get("hits", [])),
            "chunksAfterInfoboxFilter": len(hits),
            "articlesInChunkPool": len(ranked_articles),
            "selectedArticles": len(selected),
            "latencyMs": response.get("latencyMs"),
            "wallMs": round((time.time() - started) * 1000),
            "topArticles": selected[:20],
        }

        selected_ids = {article["article_id"] for article in selected}
        for rank, article in enumerate(selected, start=1):
            article_id = article["article_id"]
            existing = selected_articles.get(article_id)
            category_entry = {
                "category": category,
                "categoryRank": rank,
                "articleScore": article["articleScore"],
                "maxScore": article["maxScore"],
                "chunkHits": article["chunkHits"],
            }
            if existing is None:
                selected_articles[article_id] = {
                    "article_id": article_id,
                    "title": article.get("title"),
                    "url": article.get("url"),
                    "categories": [category_entry],
                }
            else:
                existing["categories"].append(category_entry)

        per_article_seen: dict[str, set[str]] = defaultdict(set)
        for rank, hit in enumerate(hits, start=1):
            article_id = hit.get("article_id")
            chunk_id = hit.get("chunk_id")
            if article_id not in selected_ids or not chunk_id:
                continue
            if chunk_id in per_article_seen[article_id]:
                continue
            per_article_seen[article_id].add(chunk_id)
            chunk_candidates[article_id].append(
                {
                    "chunk_id": chunk_id,
                    "category": category,
                    "rank": rank,
                    "score": float(hit.get("score") or 0.0),
                    "title": hit.get("title"),
                    "heading_path": hit.get("heading_path") or [],
                }
            )

        print(
            f"{category} chunks={len(response.get('hits', []))} filtered={len(hits)} "
            f"poolArticles={len(ranked_articles)} selected={len(selected)} latencyMs={response.get('latencyMs')}",
            file=sys.stderr,
            flush=True,
        )

    for article in selected_articles.values():
        article["categories"].sort(key=lambda item: (item["categoryRank"], item["category"]))

    selected_chunk_candidates: dict[str, list[dict[str, Any]]] = {}
    for article_id, candidates in chunk_candidates.items():
        candidates.sort(key=lambda item: (-item["score"], item["rank"], item["chunk_id"]))
        selected_chunk_candidates[article_id] = candidates[:chunks_per_article]

    stats = {
        "categoryStats": category_stats,
        "selectedArticleCount": len(selected_articles),
        "selectedChunkCandidateCount": sum(len(items) for items in selected_chunk_candidates.values()),
        "articleCategoryOverlap": overlap_histogram(selected_articles),
    }
    return selected_articles, selected_chunk_candidates, stats


def overlap_histogram(selected_articles: dict[str, dict[str, Any]]) -> dict[str, int]:
    counts: dict[int, int] = defaultdict(int)
    for article in selected_articles.values():
        counts[len(article["categories"])] += 1
    return {str(key): value for key, value in sorted(counts.items())}


def write_selected_outputs(
    *,
    chunks_path: Path,
    out_dir: Path,
    selected_articles: dict[str, dict[str, Any]],
    selected_chunk_candidates: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    articles_path = out_dir / "article_selection.jsonl"
    chunk_ids_path = out_dir / "selected_chunk_ids.tsv"
    chunks_out_path = out_dir / "chunks.jsonl"

    wanted_article_ids = set(selected_articles)
    candidate_by_chunk_id = {
        candidate["chunk_id"]: {**candidate, "article_id": article_id}
        for article_id, candidates in selected_chunk_candidates.items()
        for candidate in candidates
    }
    wanted_chunk_ids = set(candidate_by_chunk_id)

    with articles_path.open("w", encoding="utf-8") as fh:
        for article in sorted(
            selected_articles.values(),
            key=lambda item: (
                item["categories"][0]["categoryRank"],
                item["title"] or "",
                item["article_id"],
            ),
        ):
            fh.write(json.dumps(article, ensure_ascii=False) + "\n")

    with chunk_ids_path.open("w", encoding="utf-8") as fh:
        fh.write("article_id\tchunk_id\tcategory\trank\tscore\theading_path\n")
        for article_id, candidates in sorted(selected_chunk_candidates.items()):
            for candidate in candidates:
                fh.write(
                    "\t".join(
                        [
                            article_id,
                            candidate["chunk_id"],
                            candidate["category"],
                            str(candidate["rank"]),
                            str(candidate["score"]),
                            " / ".join(str(part) for part in candidate.get("heading_path") or []),
                        ]
                    )
                    + "\n"
                )

    written_chunks = 0
    lead_chunks = 0
    bm25_chunks = 0
    article_seen_chunks: dict[str, int] = defaultdict(int)
    seen_chunk_ids: set[str] = set()

    with chunks_path.open(encoding="utf-8") as source, chunks_out_path.open("w", encoding="utf-8") as out:
        for line in source:
            if not line.strip():
                continue
            chunk = json.loads(line)
            article_id = str(chunk.get("article_id") or "")
            chunk_id = str(chunk.get("chunk_id") or "")
            selected_reason = None
            selection: dict[str, Any] = {}

            if article_id in wanted_article_ids and chunk.get("chunk_kind") == "lead":
                selected_reason = "lead"
            elif chunk_id in wanted_chunk_ids:
                selected_reason = "bm25_article_chunk"
                selection = candidate_by_chunk_id[chunk_id]

            if selected_reason is None or chunk_id in seen_chunk_ids:
                continue

            if selected_reason == "lead":
                lead_chunks += 1
            else:
                bm25_chunks += 1
            chunk["_selection"] = {
                "reason": selected_reason,
                "articleCategories": selected_articles[article_id]["categories"],
                **({"bm25": selection} if selection else {}),
            }
            out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            seen_chunk_ids.add(chunk_id)
            article_seen_chunks[article_id] += 1
            written_chunks += 1

    missing_lead_articles = sorted(wanted_article_ids.difference(article_seen_chunks))
    return {
        "article_selection": str(articles_path),
        "selected_chunk_ids": str(chunk_ids_path),
        "chunks": str(chunks_out_path),
        "writtenChunks": written_chunks,
        "leadChunks": lead_chunks,
        "bm25Chunks": bm25_chunks,
        "selectedArticlesWithChunks": len(article_seen_chunks),
        "missingArticles": len(missing_lead_articles),
        "missingArticleSample": missing_lead_articles[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8765/search")
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--chunk-limit", type=int, default=100000)
    parser.add_argument("--article-cutoff", type=int, default=10000)
    parser.add_argument("--chunks-per-article", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()

    started = time.time()
    selected_articles, selected_chunk_candidates, stats = select_articles_and_chunks(
        url=args.url,
        chunk_limit=args.chunk_limit,
        article_cutoff=args.article_cutoff,
        chunks_per_article=args.chunks_per_article,
        timeout=args.timeout,
    )
    output_stats = write_selected_outputs(
        chunks_path=Path(args.chunks),
        out_dir=Path(args.out_dir),
        selected_articles=selected_articles,
        selected_chunk_candidates=selected_chunk_candidates,
    )

    manifest = {
        "manifest_version": 1,
        "selection": "bm25_article_router",
        "lexicon_categories": sorted(LEXICON),
        "chunk_limit_per_category": args.chunk_limit,
        "article_cutoff_per_category": args.article_cutoff,
        "chunks_per_article": args.chunks_per_article,
        "source_chunks": args.chunks,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.time() - started, 1),
        **stats,
        **output_stats,
    }
    manifest_path = Path(args.out_dir) / "selection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
