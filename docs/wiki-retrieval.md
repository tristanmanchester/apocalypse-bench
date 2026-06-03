# Wikipedia retrieval track

The wiki track keeps the original no-tools benchmark intact and adds local
Wikipedia retrieval conditions. Candidate models still run through the
configured model router, usually OpenRouter; the local machine owns only corpus
storage, search, embeddings, and tool traces.

## Corpus

The primary corpus is `marin-community/wikipedia-markdown` from Hugging Face. Materialize it to JSONL:

```bash
uv run --with datasets scripts/wiki_download.py --out data/wiki/wikipedia-markdown.jsonl
```

For a fast smoke fixture:

```bash
uv run --with datasets scripts/wiki_download.py --out data/wiki/wikipedia-smoke.jsonl --limit 1000
```

## BM25 index

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

## Dense embeddings

Dense search uses a manifest-driven Sentence Transformers embedding pipeline.
The default remains `Snowflake/snowflake-arctic-embed-s`, but the build and
serve commands can use any compatible model, dimension, prompt name, truncation,
and float precision recorded in the dense manifest. The Qdrant collection is
created with on-disk vector storage plus TurboQuant 4-bit quantization so the
signpost index fits the laptop better than raw in-memory float vectors.

Start Qdrant separately, then build dense signposts from all article leads plus
deterministic practical/survival section matches and optional benchmark-question
preselection:

```bash
uv run --with sentence-transformers --with qdrant-client scripts/wiki_embed.py build \
  --precision float16 \
  --max-seq-length 256 \
  --chunks data/wiki/full/chunks.jsonl \
  --corpus-manifest data/wiki/full/manifest.json \
  --qdrant-url http://127.0.0.1:6333 \
  --collection wikipedia_arctic_s \
  --manifest-out data/wiki/full/dense_manifest.json \
  --question-bank data/question_bank
```

The dense manifest records the corpus id, model id, vector dimension, query
prefix, optional prompt names, normalization setting, embedding precision,
quantization, collection name, point count, and signpost selection rules.
`wiki-search` refuses dense search if the manifest does not match the corpus or
collection, if the model/dimension fields are invalid, or if the embedding
service returns a vector with the wrong dimension.

Start the query embedding endpoint:

```bash
uv run --with sentence-transformers scripts/wiki_embed.py serve \
  --precision float16 \
  --max-seq-length 256 \
  --host 127.0.0.1 \
  --port 8766
```

Then start `wiki-search` with dense environment variables:

```bash
WIKI_QDRANT_URL=http://127.0.0.1:6333 \
WIKI_QDRANT_COLLECTION=wikipedia_arctic_s \
WIKI_EMBED_URL=http://127.0.0.1:8766/embed \
WIKI_DENSE_MANIFEST=data/wiki/full/dense_manifest.json \
cargo run -p wiki-search -- serve --index data/wiki/full --listen 127.0.0.1:8765
```

If these variables are absent, BM25 and literal search still work, and dense/hybrid calls fail clearly.

## Benchmark config

Use `candidateMode` per model:

- `direct`: current no-tools baseline
- `rag-bm25`: pre-retrieve with BM25 and inject bounded context
- `rag-dense`: pre-retrieve with dense search
- `rag-hybrid`: pre-retrieve with hybrid search
- `agent-wiki`: production bounded tool-loop mode with all wiki tools exposed
- `agent-bm25`, `agent-bm25-research`, `agent-dense`, `agent-hybrid`, `agent-rg`, `agent-literal`: ablation modes with one mode-specific search tool and `wiki_read`

Agent modes use `@earendil-works/pi-agent-core` with Pi tools backed by the
local wiki service. The harness enforces `wiki.limits.maxToolCalls` and
`wiki.limits.maxTurns`, and the agent receives only wiki tools, not live web
access.

The Pi agent uses a benchmark-owned text tool-call protocol instead of sending
OpenRouter's native `tools` request parameter. Models request tools by emitting
`<tool_call>{"name":"wiki_search","arguments":{"query":"...","topK":5}}</tool_call>`
or `wiki_read` with a `chunkId`; the harness parses that text into Pi tool calls,
executes the wiki tool, and returns the result as text. This keeps agent
conditions comparable for models whose providers do not advertise native
OpenRouter tool support but whose model training still includes tool use.

The neutral XML+JSON form is preferred, but the parser also accepts documented
model-native formats used by the current nine-model matrix:

- Liquid LFM2.5: `<|tool_call_start|>` / `<|tool_call_end|>` with JSON or
  Pythonic calls such as `[wiki_search(query='water purification', topK=5)]`
- Google Gemma 4: `<|tool_call>call:wiki_search{query:<|"|>...<|"|>}<tool_call|>`
- IBM Granite 4.x: `<tool_call>{"name":"...","arguments":{...}}</tool_call>`
- NVIDIA Nemotron 3 Nano: Qwen3 Coder XML, such as
  `<function=wiki_search><parameter=query>...</parameter></function>`
- Microsoft Phi-4 Mini: `functools[...]` JSON calls, plus
  `<|tool_call|>` / `<|/tool_call|>` JSON wrappers when a serving layer emits
  them
- OpenAI gpt-oss: Harmony `to=functions.name ... <|message|>{json}<|call|>`
- Arcee Trinity and generic structured-output paths: raw JSON, OpenAI
  Chat/Responses JSON (`tool_calls`, `function_call`, `function`, `output`
  items), and Anthropic-style `tool_use` content blocks

The production `agent-wiki` mode exposes:

- `wiki_hybrid_search`: default broad discovery tool using BM25 plus dense retrieval
- `wiki_search`: BM25 search for exact terminology, article names, materials, hazards, or symptoms
- `wiki_semantic_search`: dense semantic signpost search for concepts and synonyms
- `wiki_literal_search`: scoped exact phrase search inside a known `articleId` or `chunkId`
- `wiki_read`: bounded source hydration by `chunkId`

The single-tool agent modes are intentionally narrower controls:
`agent-bm25` exposes `wiki_search`, `agent-dense` exposes
`wiki_semantic_search`, `agent-hybrid` exposes `wiki_hybrid_search`, and
literal controls expose `wiki_literal_search`. All agent modes also expose
`wiki_read` for bounded source hydration.

`agent-bm25-research` is a development mode for retrieval prompt/tool-shape
iteration. It exposes `wiki_research`, which accepts up to four BM25 query
variants in one tool call, dedupes the hits, and annotates each hit with the
queries that matched it. Use it to test whether broader lexical query planning
helps before promoting a tool shape to the production `agent-wiki` surface.

`wiki.enabled` is the feature flag. If no wiki mode is configured, the benchmark
uses the current direct no-tools path. If a model uses a wiki `candidateMode`,
`wiki.enabled` must not be `false`.

`apocbench-wiki.yml` repeats the current nine-model OpenRouter matrix across
direct, BM25/dense/hybrid RAG, BM25/dense/hybrid single-tool Pi-agent
conditions, and the production `agent-wiki` condition. It is candidate-only by
default: the runner stores completed answers and skips the configured placeholder
judge. Score completed rows with `scripts/codex_rejudge.ts`.

Replace its manifest ids after building your local index, then run a small smoke:

```bash
pnpm dev validate -c apocbench-wiki.yml
pnpm dev run -c apocbench-wiki.yml --limit 2
```

## Reports

Wiki-enabled results persist retrieval traces beside candidate answers. HTML reports show per-question retrieval mode, search/read counts, source titles, and the raw bounded trace so failures can be diagnosed as retrieval misses, source mismatch, or model/tool-use issues.

## Retrieval-debug harness

The canonical local development harness is `data/question_sets/retrieval-debug-10.json`.
It is not a holdout benchmark. It is a compact, deliberately mixed set of ten
questions used to test retrieval/tool changes with repeated stochastic samples.

Generate a candidate-only config:

```bash
pnpm -s retrieval-debug:config -- \
  --out apocbench-retrieval-debug-10.generated.json \
  --repeats 10 \
  --conditions direct,bm25,hybrid,bm25-research
```

Run it in chunks so progress can be resumed without redoing completed candidate
answers:

```bash
RUN_ID=retrieval-debug-10-gemma31b-$(date +%Y%m%d-%H%M%S) \
CONFIG=apocbench-retrieval-debug-10.generated.json \
bash scripts/run_retrieval_debug_chunks.sh
```

The generated config sets `run.candidateOnly=true`, so the runner stores
candidate answers as `candidate_done` and does not instantiate or call the
configured judge. Score the completed candidate rows with Codex:

```bash
pnpm -s rejudge:codex -- \
  --source-run <candidate-run-id> \
  --out-run <candidate-run-id>-codex-question-b10 \
  --codex-bin /home/tristan/.local/bin/codex \
  --model gpt-5.5 \
  --reasoning low \
  --batch-size 10 \
  --batch-strategy sequential \
  --source-status both
```

Then summarize condition-level medians, means, auto-fails, zero rates, and
paired deltas versus direct:

```bash
pnpm -s retrieval-debug:report -- \
  --run <candidate-run-id>-codex-question-b10 \
  --out logs/<candidate-run-id>-codex-question-b10-report.json
```
