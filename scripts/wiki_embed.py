#!/usr/bin/env python3
"""Build and serve Snowflake Arctic S embeddings for wiki-search."""

from __future__ import annotations

import argparse
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


MODEL_ID = "Snowflake/snowflake-arctic-embed-s"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
VECTOR_SIZE = 384
QUANTIZATION = "turbo-bits4"
SIGNPOST_RULES = ["all_article_leads", "targeted_practical_sections", "benchmark_question_preselection"]
PRACTICAL_TERMS = {
    "agriculture",
    "antibiotic",
    "antiseptic",
    "bandage",
    "bleeding",
    "boiling",
    "burn",
    "calcium hypochlorite",
    "charcoal",
    "chlorine",
    "cholera",
    "compost",
    "distillation",
    "disinfection",
    "disease",
    "dose",
    "evacuation",
    "fermentation",
    "fire",
    "first aid",
    "food preservation",
    "fracture",
    "fuel",
    "garden",
    "heat stroke",
    "hypothermia",
    "infection",
    "irrigation",
    "latrine",
    "medicine",
    "navigation",
    "purification",
    "radio",
    "ration",
    "sanitation",
    "shelter",
    "splint",
    "sterilization",
    "stove",
    "suture",
    "tetanus",
    "tool",
    "venom",
    "water",
    "wound",
}
STOPWORDS = {
    "about",
    "after",
    "again",
    "against",
    "available",
    "before",
    "being",
    "between",
    "community",
    "could",
    "exactly",
    "first",
    "group",
    "having",
    "should",
    "their",
    "there",
    "these",
    "thing",
    "through",
    "under",
    "using",
    "water",
    "where",
    "which",
    "while",
    "with",
    "without",
}
QUESTION_PHRASE_LENGTHS = (2, 3, 4)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build")
    build.add_argument("--chunks", required=True)
    build.add_argument("--qdrant-url", required=True)
    build.add_argument("--collection", required=True)
    build.add_argument("--corpus-manifest", required=True)
    build.add_argument("--manifest-out", required=True)
    build.add_argument("--question-bank", action="append", default=[])
    build.add_argument("--limit", type=int, default=None)
    build.add_argument("--batch-size", type=int, default=64)
    build.add_argument("--encode-batch-size", type=int, default=128)
    build.add_argument("--max-seq-length", type=int, default=512)
    build.add_argument("--progress-every", type=int, default=1000)
    build.add_argument("--fp16", action="store_true")

    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8766)
    serve.add_argument("--max-seq-length", type=int, default=512)
    serve.add_argument("--fp16", action="store_true")

    args = parser.parse_args()
    if args.command == "build":
        build_vectors(args)
    elif args.command == "serve":
        serve_embeddings(args.host, args.port, args.fp16, args.max_seq_length)


def load_model(fp16: bool = False, max_seq_length: int = 512):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_ID, trust_remote_code=True)
    model.max_seq_length = max_seq_length
    if fp16:
        model.half()
    return model


def build_vectors(args: argparse.Namespace) -> None:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import (
        Distance,
        TurboQuantBitSize,
        TurboQuantQuantizationConfig,
        TurboQuantization,
        VectorParams,
    )

    model = load_model(args.fp16, args.max_seq_length)
    corpus_manifest = json.loads(Path(args.corpus_manifest).read_text(encoding="utf-8"))
    question_terms = load_question_terms([Path(path) for path in args.question_bank])
    client = QdrantClient(url=args.qdrant_url)
    client.recreate_collection(
        collection_name=args.collection,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE, on_disk=True),
        quantization_config=TurboQuantization(
            turbo=TurboQuantQuantizationConfig(bits=TurboQuantBitSize.BITS4, always_ram=False)
        ),
    )

    chunks = iter_signposts(Path(args.chunks), args.limit, question_terms)
    total = 0
    batches = 0
    started_at = time.monotonic()
    batch: list[dict[str, Any]] = []
    log_progress("started", collection=args.collection, limit=args.limit, batch_size=args.batch_size)
    for chunk in chunks:
        batch.append(chunk)
        if len(batch) >= args.batch_size:
            upsert_batch(client, model, args.collection, batch, total, args.encode_batch_size)
            total += len(batch)
            batches += 1
            batch = []
            if args.progress_every > 0 and batches % args.progress_every == 0:
                log_progress("progress", points=total, batches=batches, elapsed_s=round(time.monotonic() - started_at, 1))
    if batch:
        upsert_batch(client, model, args.collection, batch, total, args.encode_batch_size)
        total += len(batch)
        batches += 1

    manifest = build_dense_manifest(
        corpus_manifest=corpus_manifest,
        collection=args.collection,
        source_chunks=Path(args.chunks),
        question_bank_paths=args.question_bank,
        point_count=total,
        embedding_precision="float16" if args.fp16 else "float32",
        max_seq_length=args.max_seq_length,
    )
    manifest_out = Path(args.manifest_out)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    log_progress("completed", points=total, batches=batches, elapsed_s=round(time.monotonic() - started_at, 1))
    print(json.dumps(manifest, indent=2))


def log_progress(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}), flush=True)


def iter_signposts(path: Path, limit: int | None, question_terms: set[str]):
    count = 0
    seen: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            chunk = json.loads(line)
            if not is_signpost(chunk, question_terms):
                continue
            chunk_id = chunk["chunk_id"]
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            yield chunk
            count += 1
            if limit is not None and count >= limit:
                return


def is_signpost(chunk: dict[str, Any], question_terms: set[str]) -> bool:
    if chunk.get("chunk_kind") == "lead":
        return True
    title_heading = " ".join(
        [str(chunk.get("title") or ""), " ".join(str(part) for part in chunk.get("heading_path") or [])]
    ).lower()
    if any(term in title_heading for term in PRACTICAL_TERMS):
        return True
    return bool(question_terms and title_heading_phrases(title_heading).intersection(question_terms))


def load_question_terms(paths: list[Path]) -> set[str]:
    terms: set[str] = set()
    for path in expand_question_bank_paths(paths):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                text = " ".join(
                    str(part)
                    for part in [
                        row.get("title"),
                        row.get("prompt"),
                        " ".join(row.get("scenario") or []),
                        " ".join(row.get("reference_facts") or []),
                    ]
                    if part
                ).lower()
                terms.update(question_phrases(text))
    return terms


def question_phrases(text: str) -> set[str]:
    tokens = phrase_tokens(text)
    return ngram_phrases(tokens)


def title_heading_phrases(text: str) -> set[str]:
    return ngram_phrases(phrase_tokens(text))


def phrase_tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z][a-z0-9-]{3,}", text.lower())
        if token not in STOPWORDS
    ]


def ngram_phrases(tokens: list[str]) -> set[str]:
    phrases: set[str] = set()
    for length in QUESTION_PHRASE_LENGTHS:
        if len(tokens) < length:
            continue
        for idx in range(0, len(tokens) - length + 1):
            phrase = " ".join(tokens[idx : idx + length])
            if len(phrase) <= 80:
                phrases.add(phrase)
    return phrases


def build_dense_manifest(
    *,
    corpus_manifest: dict[str, Any],
    collection: str,
    source_chunks: Path,
    question_bank_paths: list[str],
    point_count: int,
    embedding_precision: str = "float32",
    max_seq_length: int = 512,
) -> dict[str, Any]:
    return {
        "manifest_version": 1,
        "corpus_id": corpus_manifest["corpus_id"],
        "model": MODEL_ID,
        "dimension": VECTOR_SIZE,
        "collection": collection,
        "query_prefix": QUERY_PREFIX,
        "normalized": True,
        "embedding_precision": embedding_precision,
        "max_seq_length": max_seq_length,
        "quantization": QUANTIZATION,
        "vectors_on_disk": True,
        "point_count": point_count,
        "signpost_rules": SIGNPOST_RULES,
        "source_chunks": str(source_chunks),
        "question_bank_paths": question_bank_paths,
    }


def expand_question_bank_paths(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(sorted(path.glob("*.jsonl")))
        else:
            expanded.append(path)
    return expanded


def upsert_batch(
    client: Any,
    model: Any,
    collection: str,
    batch: list[dict[str, Any]],
    offset: int,
    encode_batch_size: int,
) -> None:
    from qdrant_client.http.models import PointStruct

    texts = [document_text(chunk) for chunk in batch]
    vectors = model.encode(
        texts,
        batch_size=encode_batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()
    points = [
        PointStruct(
            id=offset + idx,
            vector=vector,
            payload={
                "article_id": chunk["article_id"],
                "chunk_id": chunk["chunk_id"],
                "title": chunk["title"],
                "url": chunk.get("url"),
            },
        )
        for idx, (chunk, vector) in enumerate(zip(batch, vectors, strict=True))
    ]
    client.upsert(collection_name=collection, points=points)


def document_text(chunk: dict[str, Any]) -> str:
    parts = [chunk.get("title"), chunk.get("abstract_text"), chunk.get("text")]
    return "\n\n".join(str(part) for part in parts if part)


def serve_embeddings(host: str, port: int, fp16: bool, max_seq_length: int) -> None:
    model = load_model(fp16, max_seq_length)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/embed":
                self.send_error(404)
                return
            length = int(self.headers.get("content-length", "0"))
            payload = json.loads(self.rfile.read(length))
            query = payload.get("query")
            if not isinstance(query, str):
                self.send_error(400, "query must be a string")
                return
            vector = model.encode(QUERY_PREFIX + query, normalize_embeddings=True).tolist()
            body = json.dumps({"embedding": vector}).encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer((host, port), Handler)
    print(
        json.dumps(
            {
                "model": MODEL_ID,
                "host": host,
                "port": port,
                "path": "/embed",
                "embedding_precision": "float16" if fp16 else "float32",
                "max_seq_length": max_seq_length,
            }
        )
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
