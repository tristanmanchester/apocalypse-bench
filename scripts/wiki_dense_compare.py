#!/usr/bin/env python3
"""Compare dense wiki embedding collections on the apocalypse question bank."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

import wiki_embed


DEFAULT_MODELS = [
    {
        "name": "arctic_m_v2",
        "collection": "wikipedia_article_router_arctic_m_v2_768",
        "model_id": "Snowflake/snowflake-arctic-embed-m-v2.0",
        "query_prefix": "query: ",
        "query_prompt_name": None,
        "precision": "float16",
        "max_seq_length": 512,
        "truncate_dim": None,
    },
    {
        "name": "embeddinggemma",
        "collection": "wikipedia_article_router_embeddinggemma_768",
        "model_id": "google/embeddinggemma-300m",
        "query_prefix": "",
        "query_prompt_name": "query",
        "precision": "bfloat16",
        "max_seq_length": 512,
        "truncate_dim": None,
    },
]

STOPWORDS = {
    "about",
    "above",
    "after",
    "again",
    "against",
    "along",
    "already",
    "also",
    "although",
    "answer",
    "around",
    "because",
    "before",
    "being",
    "between",
    "cannot",
    "could",
    "during",
    "every",
    "given",
    "having",
    "however",
    "important",
    "instead",
    "known",
    "likely",
    "might",
    "needed",
    "other",
    "people",
    "person",
    "place",
    "possible",
    "question",
    "should",
    "situation",
    "there",
    "these",
    "thing",
    "those",
    "through",
    "under",
    "until",
    "using",
    "where",
    "which",
    "while",
    "would",
}


@dataclass(frozen=True)
class Question:
    qid: str
    category: str
    query: str
    reference_text: str


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--question-bank", action="append", default=None)
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    parser.add_argument("--model-json", action="append", default=[])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit-questions", type=int, default=None)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", default=None)
    args = parser.parse_args()

    model_specs = [json.loads(raw) for raw in args.model_json] if args.model_json else DEFAULT_MODELS
    question_bank_paths = args.question_bank or ["data/question_bank"]
    questions = load_questions([Path(path) for path in question_bank_paths], args.limit_questions)
    chunk_texts = load_chunk_texts(Path(args.chunks))
    results = []
    started_at = time.monotonic()

    for spec in model_specs:
        print(json.dumps({"event": "model_started", "name": spec["name"], "collection": spec["collection"]}), flush=True)
        collection = load_collection_status(args.qdrant_url, spec["collection"])
        model = wiki_embed.load_model(
            spec["model_id"],
            precision=spec.get("precision") or "float32",
            max_seq_length=int(spec.get("max_seq_length") or 512),
            truncate_dim=spec.get("truncate_dim"),
        )
        queries = [spec.get("query_prefix", "") + question.query for question in questions]
        encoded_started = time.monotonic()
        vectors = wiki_embed.encode_texts(
            model,
            queries,
            batch_size=32,
            normalize_embeddings=True,
            prompt_name=spec.get("query_prompt_name"),
        ).tolist()
        encode_elapsed_ms = (time.monotonic() - encoded_started) * 1000
        per_question = []
        latencies = []
        for question, vector in zip(questions, vectors, strict=True):
            search_started = time.monotonic()
            hits = qdrant_search(args.qdrant_url, spec["collection"], vector, args.top_k)
            latency_ms = (time.monotonic() - search_started) * 1000
            latencies.append(latency_ms)
            per_question.append(score_question(question, hits, chunk_texts, args.top_k, latency_ms))
        summary = summarize(spec, collection, questions, per_question, latencies, encode_elapsed_ms)
        results.append({"spec": spec, "collection": collection, "summary": summary, "questions": per_question})
        print(json.dumps({"event": "model_completed", "name": spec["name"], **summary}), flush=True)

    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "qdrant_url": args.qdrant_url,
        "chunks": str(args.chunks),
        "question_count": len(questions),
        "top_k": args.top_k,
        "elapsed_s": round(time.monotonic() - started_at, 3),
        "models": results,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.out_csv:
        write_csv(Path(args.out_csv), results)
    print(json.dumps({"event": "completed", "out_json": args.out_json, "out_csv": args.out_csv}), flush=True)


def load_questions(paths: list[Path], limit: int | None) -> list[Question]:
    rows = []
    for path in expand_paths(paths):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                query = " ".join(
                    flatten_text(part)
                    for part in [row.get("title"), row.get("scenario"), row.get("prompt")]
                    if part
                )
                reference_text = flatten_text(row.get("reference_facts"))
                rows.append(
                    Question(
                        qid=str(row.get("id") or row.get("question_id")),
                        category=str(row.get("category") or path.stem),
                        query=query,
                        reference_text=reference_text,
                    )
                )
                if limit is not None and len(rows) >= limit:
                    return rows
    return rows


def expand_paths(paths: list[Path]) -> list[Path]:
    expanded = []
    for path in paths:
        if path.is_dir():
            expanded.extend(sorted(path.glob("*.jsonl")))
        else:
            expanded.append(path)
    return expanded


def flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(flatten_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(flatten_text(item) for item in value.values())
    return str(value)


def load_chunk_texts(path: Path) -> dict[str, str]:
    chunks = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            chunk_id = row.get("chunk_id")
            if not chunk_id:
                continue
            chunks[str(chunk_id)] = " ".join(
                flatten_text(part)
                for part in [row.get("title"), row.get("heading_path"), row.get("abstract_text"), row.get("text")]
                if part
            ).lower()
    return chunks


def load_collection_status(qdrant_url: str, collection: str) -> dict[str, Any]:
    response = requests.get(f"{qdrant_url.rstrip('/')}/collections/{collection}", timeout=30)
    response.raise_for_status()
    return response.json().get("result", {})


def qdrant_search(qdrant_url: str, collection: str, vector: list[float], top_k: int) -> list[dict[str, Any]]:
    response = requests.post(
        f"{qdrant_url.rstrip('/')}/collections/{collection}/points/search",
        json={"vector": vector, "limit": top_k, "with_payload": True},
        timeout=60,
    )
    response.raise_for_status()
    return response.json().get("result", [])


def score_question(
    question: Question,
    hits: list[dict[str, Any]],
    chunk_texts: dict[str, str],
    top_k: int,
    latency_ms: float,
) -> dict[str, Any]:
    reference_terms = content_terms(question.reference_text)
    query_terms = content_terms(question.query)
    answer_terms = reference_terms - query_terms
    if len(answer_terms) < 5:
        answer_terms = reference_terms
    phrases = reference_phrases(question.reference_text, query_terms)
    hit_texts = []
    top_titles = []
    first_term_hit_rank = None
    first_phrase_hit_rank = None
    for rank, hit in enumerate(hits, start=1):
        payload = hit.get("payload") or {}
        chunk_id = payload.get("chunk_id")
        text = chunk_texts.get(str(chunk_id), "")
        hit_texts.append(text)
        if payload.get("title"):
            top_titles.append(str(payload["title"]))
        if first_term_hit_rank is None and answer_terms and any(term in text for term in answer_terms):
            first_term_hit_rank = rank
        if first_phrase_hit_rank is None and phrases and any(phrase in text for phrase in phrases):
            first_phrase_hit_rank = rank

    joined = "\n".join(hit_texts[:top_k])
    covered_terms = {term for term in answer_terms if term in joined}
    covered_phrases = {phrase for phrase in phrases if phrase in joined}
    return {
        "id": question.qid,
        "category": question.category,
        "term_count": len(answer_terms),
        "phrase_count": len(phrases),
        "term_coverage": safe_ratio(len(covered_terms), len(answer_terms)),
        "phrase_coverage": safe_ratio(len(covered_phrases), len(phrases)),
        "phrase_hit": bool(covered_phrases),
        "first_term_hit_rank": first_term_hit_rank,
        "first_phrase_hit_rank": first_phrase_hit_rank,
        "latency_ms": round(latency_ms, 3),
        "top_titles": top_titles[:5],
    }


def content_terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z][a-z0-9-]{4,}", text.lower())
        if token not in STOPWORDS and not token.isdigit()
    }


def reference_phrases(text: str, query_terms: set[str]) -> set[str]:
    tokens = [
        token
        for token in re.findall(r"[a-z][a-z0-9-]{3,}", text.lower())
        if token not in STOPWORDS
    ]
    phrases = set()
    for length in (2, 3, 4):
        for idx in range(0, max(0, len(tokens) - length + 1)):
            phrase_tokens = tokens[idx : idx + length]
            if all(token in query_terms for token in phrase_tokens):
                continue
            phrase = " ".join(phrase_tokens)
            if 8 <= len(phrase) <= 80:
                phrases.add(phrase)
    return phrases


def safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def summarize(
    spec: dict[str, Any],
    collection: dict[str, Any],
    questions: list[Question],
    per_question: list[dict[str, Any]],
    latencies: list[float],
    encode_elapsed_ms: float,
) -> dict[str, Any]:
    term_coverages = [row["term_coverage"] for row in per_question]
    phrase_coverages = [row["phrase_coverage"] for row in per_question]
    phrase_hits = [row["phrase_hit"] for row in per_question]
    first_term_ranks = [row["first_term_hit_rank"] for row in per_question if row["first_term_hit_rank"] is not None]
    categories = sorted({question.category for question in questions})
    by_category = {}
    for category in categories:
        rows = [row for row in per_question if row["category"] == category]
        by_category[category] = {
            "mean_term_coverage": round(mean(row["term_coverage"] for row in rows), 4),
            "phrase_hit_rate": round(mean(1.0 if row["phrase_hit"] else 0.0 for row in rows), 4),
        }
    return {
        "name": spec["name"],
        "model_id": spec["model_id"],
        "collection": spec["collection"],
        "collection_points": collection.get("points_count"),
        "question_count": len(questions),
        "mean_term_coverage": round(mean(term_coverages), 4),
        "median_term_coverage": round(statistics.median(term_coverages), 4) if term_coverages else 0.0,
        "mean_phrase_coverage": round(mean(phrase_coverages), 4),
        "phrase_hit_rate": round(mean(1.0 if value else 0.0 for value in phrase_hits), 4),
        "mean_first_term_rank": round(mean(first_term_ranks), 3) if first_term_ranks else None,
        "encode_total_ms": round(encode_elapsed_ms, 3),
        "encode_ms_per_query": round(encode_elapsed_ms / max(1, len(questions)), 3),
        "search_latency_p50_ms": round(percentile(latencies, 50), 3),
        "search_latency_p95_ms": round(percentile(latencies, 95), 3),
        "by_category": by_category,
    }


def mean(values: Any) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * (pct / 100.0)
    lower = math.floor(idx)
    upper = math.ceil(idx)
    if lower == upper:
        return ordered[int(idx)]
    return ordered[lower] * (upper - idx) + ordered[upper] * (idx - lower)


def write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "model",
                "id",
                "category",
                "term_coverage",
                "phrase_coverage",
                "phrase_hit",
                "first_term_hit_rank",
                "first_phrase_hit_rank",
                "latency_ms",
                "top_titles",
            ],
        )
        writer.writeheader()
        for result in results:
            model_name = result["spec"]["name"]
            for row in result["questions"]:
                writer.writerow(
                    {
                        "model": model_name,
                        "id": row["id"],
                        "category": row["category"],
                        "term_coverage": row["term_coverage"],
                        "phrase_coverage": row["phrase_coverage"],
                        "phrase_hit": row["phrase_hit"],
                        "first_term_hit_rank": row["first_term_hit_rank"],
                        "first_phrase_hit_rank": row["first_phrase_hit_rank"],
                        "latency_ms": row["latency_ms"],
                        "top_titles": " | ".join(row["top_titles"]),
                    }
                )


if __name__ == "__main__":
    main()
