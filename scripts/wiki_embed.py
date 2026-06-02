#!/usr/bin/env python3
"""Build and serve Snowflake Arctic S embeddings for wiki-search."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


MODEL_ID = "Snowflake/snowflake-arctic-embed-s"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
VECTOR_SIZE = 384


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build")
    build.add_argument("--chunks", required=True)
    build.add_argument("--qdrant-url", required=True)
    build.add_argument("--collection", required=True)
    build.add_argument("--limit", type=int, default=None)
    build.add_argument("--batch-size", type=int, default=64)

    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8766)

    args = parser.parse_args()
    if args.command == "build":
        build_vectors(args)
    elif args.command == "serve":
        serve_embeddings(args.host, args.port)


def load_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_ID, trust_remote_code=True)


def build_vectors(args: argparse.Namespace) -> None:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams

    model = load_model()
    client = QdrantClient(url=args.qdrant_url)
    client.recreate_collection(
        collection_name=args.collection,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE, on_disk=True),
    )

    chunks = iter_signposts(Path(args.chunks), args.limit)
    total = 0
    batch: list[dict[str, Any]] = []
    for chunk in chunks:
        batch.append(chunk)
        if len(batch) >= args.batch_size:
            upsert_batch(client, model, args.collection, batch, total)
            total += len(batch)
            batch = []
    if batch:
        upsert_batch(client, model, args.collection, batch, total)
        total += len(batch)

    print(
        json.dumps(
            {
                "model": MODEL_ID,
                "dimension": VECTOR_SIZE,
                "collection": args.collection,
                "points": total,
            },
            indent=2,
        )
    )


def iter_signposts(path: Path, limit: int | None):
    count = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            chunk = json.loads(line)
            # One-shot full-Wikipedia posture: every lead is embedded; section
            # signposts can be added later by feeding a preselected chunk file.
            if chunk.get("chunk_kind") != "lead":
                continue
            yield chunk
            count += 1
            if limit is not None and count >= limit:
                return


def upsert_batch(client: Any, model: Any, collection: str, batch: list[dict[str, Any]], offset: int) -> None:
    from qdrant_client.http.models import PointStruct

    texts = [document_text(chunk) for chunk in batch]
    vectors = model.encode(texts, normalize_embeddings=True).tolist()
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


def serve_embeddings(host: str, port: int) -> None:
    model = load_model()

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
    print(json.dumps({"model": MODEL_ID, "host": host, "port": port, "path": "/embed"}))
    server.serve_forever()


if __name__ == "__main__":
    main()
