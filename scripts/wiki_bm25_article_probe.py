#!/usr/bin/env python3
import argparse
import json
import math
import sys
import time
from collections import defaultdict

from wiki_bm25_lexicon_probe import LEXICON, normalize_query, post_json


def article_score(scores):
    if not scores:
        return 0.0
    # Keep the best matching chunk dominant, but reward articles with several
    # independently strong sections so one repeated term does not fully decide.
    return scores[0] + 0.3 * sum(scores[1:3])


def aggregate_articles(hits):
    articles = {}
    for rank, hit in enumerate(hits, start=1):
        article_id = hit.get("article_id")
        if not article_id:
            continue
        score = float(hit.get("score") or 0.0)
        article = articles.get(article_id)
        if article is None:
            article = {
                "article_id": article_id,
                "title": hit.get("title"),
                "url": hit.get("url"),
                "firstRank": rank,
                "chunkHits": 0,
                "scores": [],
            }
            articles[article_id] = article
        article["chunkHits"] += 1
        if len(article["scores"]) < 3:
            article["scores"].append(score)

    ranked = []
    for article in articles.values():
        scores = article.pop("scores")
        article["maxScore"] = scores[0] if scores else 0.0
        article["top3Score"] = sum(scores)
        article["articleScore"] = article_score(scores)
        ranked.append(article)

    ranked.sort(
        key=lambda item: (
            -item["articleScore"],
            item["firstRank"],
            item["title"] or "",
            item["article_id"],
        )
    )
    return ranked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8765/search")
    parser.add_argument("--chunk-limit", type=int, default=100000)
    parser.add_argument("--article-cutoffs", default="1000,5000,10000,25000")
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()

    cutoffs = [int(item) for item in args.article_cutoffs.split(",") if item.strip()]
    max_cutoff = max(cutoffs)
    all_by_cutoff = {cutoff: set() for cutoff in cutoffs}
    article_categories = {cutoff: defaultdict(set) for cutoff in cutoffs}
    output = {
        "url": args.url,
        "chunkLimit": args.chunk_limit,
        "articleCutoffs": cutoffs,
        "categories": {},
        "startedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    for category, lexicon in LEXICON.items():
        query = normalize_query(lexicon)
        started = time.time()
        response = post_json(
            args.url,
            {"query": query, "limit": args.chunk_limit},
            args.timeout,
        )
        hits = response.get("hits", [])
        ranked_articles = aggregate_articles(hits)
        output["categories"][category] = {
            "queryTerms": len(query.split()),
            "chunksFetched": len(hits),
            "articlesInChunkPool": len(ranked_articles),
            "latencyMs": response.get("latencyMs"),
            "wallMs": round((time.time() - started) * 1000),
            "cutoffs": {},
            "topArticles": ranked_articles[:20],
            "chunkHitHistogram": {},
        }

        histogram = defaultdict(int)
        for article in ranked_articles:
            bucket = article["chunkHits"]
            if bucket > 20:
                bucket = 21
            histogram[bucket] += 1
        output["categories"][category]["chunkHitHistogram"] = {
            ("21+" if key == 21 else str(key)): value
            for key, value in sorted(histogram.items(), key=lambda item: (math.inf if item[0] == 21 else item[0]))
        }

        for cutoff in cutoffs:
            selected = ranked_articles[:cutoff]
            for article in selected:
                all_by_cutoff[cutoff].add(article["article_id"])
                article_categories[cutoff][article["article_id"]].add(category)
            output["categories"][category]["cutoffs"][str(cutoff)] = {
                "selectedArticles": len(selected),
                "lowestArticleScore": selected[-1]["articleScore"] if selected else None,
                "lowestFirstRank": selected[-1]["firstRank"] if selected else None,
            }

        print(
            f"{category} chunks={len(hits)} poolArticles={len(ranked_articles)} "
            f"top{max_cutoff}={min(max_cutoff, len(ranked_articles))} latencyMs={response.get('latencyMs')}",
            file=sys.stderr,
            flush=True,
        )

    output["union"] = {}
    for cutoff in cutoffs:
        overlap_counts = defaultdict(int)
        for categories in article_categories[cutoff].values():
            overlap_counts[len(categories)] += 1
        output["union"][str(cutoff)] = {
            "uniqueArticles": len(all_by_cutoff[cutoff]),
            "overlapBuckets": dict(sorted(overlap_counts.items())),
        }

    output["finishedAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
