# Wikipedia Retrieval Track

The wiki track keeps the original no-tools benchmark intact and adds local Wikipedia retrieval conditions. Candidate models still run through the configured model router, usually OpenRouter; the local machine owns only corpus storage, search, embeddings, and tool traces.

## Corpus

The primary corpus is `marin-community/wikipedia-markdown` from Hugging Face. Materialize it to JSONL:

```bash
uv run --with datasets scripts/wiki_download.py --out data/wiki/wikipedia-markdown.jsonl
```

For a fast smoke fixture:

```bash
uv run --with datasets scripts/wiki_download.py --out data/wiki/wikipedia-smoke.jsonl --limit 1000
```

## BM25 Index

Build the Rust corpus store and Tantivy BM25 index:

```bash
cargo run -p wiki-search -- ingest --input data/wiki/wikipedia-markdown.jsonl --out data/wiki/full
```

The command prints a manifest. Copy the `corpus_id` value into both `wiki.corpus.manifestId` and `wiki.index.manifestId` in the wiki config. Start the local search service:

```bash
cargo run -p wiki-search -- serve --index data/wiki/full --listen 127.0.0.1:8765
```

Useful fixture commands:

```bash
pnpm wiki:ingest:fixture
pnpm wiki:serve:fixture
```

## Dense Embeddings

Dense search uses only `Snowflake/snowflake-arctic-embed-s` with 384-dimensional vectors. Start Qdrant separately, then build article-lead signpost vectors:

```bash
uv run --with sentence-transformers --with qdrant-client scripts/wiki_embed.py build \
  --chunks data/wiki/full/chunks.jsonl \
  --qdrant-url http://127.0.0.1:6333 \
  --collection wikipedia_arctic_s
```

Start the query embedding endpoint:

```bash
uv run --with sentence-transformers scripts/wiki_embed.py serve --host 127.0.0.1 --port 8766
```

Then start `wiki-search` with dense environment variables:

```bash
WIKI_QDRANT_URL=http://127.0.0.1:6333 \
WIKI_QDRANT_COLLECTION=wikipedia_arctic_s \
WIKI_EMBED_URL=http://127.0.0.1:8766/embed \
cargo run -p wiki-search -- serve --index data/wiki/full --listen 127.0.0.1:8765
```

If these variables are absent, BM25 and literal search still work, and dense/hybrid calls fail clearly.

## Benchmark Config

Use `candidateMode` per model:

- `direct`: current no-tools baseline
- `rag-bm25`: pre-retrieve with BM25 and inject bounded context
- `rag-dense`: pre-retrieve with dense search
- `rag-hybrid`: pre-retrieve with hybrid search
- `agent-bm25`, `agent-dense`, `agent-hybrid`, `agent-rg`, `agent-literal`: bounded tool-loop wiki modes with `wiki_search` and `wiki_read`

The agent modes use the same AI SDK/OpenRouter runner as the rest of the benchmark so routing, retries, cost accounting, and provider metrics stay comparable. They intentionally expose only wiki tools, not live web access.

`apocbench-wiki.yml` shows one model repeated across direct and retrieval conditions. Replace its manifest ids after building your local index, then run a small smoke:

```bash
pnpm dev validate -c apocbench-wiki.yml
pnpm dev run -c apocbench-wiki.yml --limit 2
```

## Reports

Wiki-enabled results persist retrieval traces beside candidate answers. HTML reports show per-question retrieval mode, search/read counts, source titles, and the raw bounded trace so failures can be diagnosed as retrieval misses, source mismatch, or model/tool-use issues.
