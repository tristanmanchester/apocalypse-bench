# Apocalypse-bench

You’ve got a laptop, a pile of scrap, and exactly zero internet. The question isn’t “can an LLM write code?”—it’s “can it help you do real work without getting anyone hurt?”

`apocalypse-bench` (CLI: `apocbench`) is a TypeScript benchmark runner that:

- Runs a fixed survival/offline-assistant question bank against one or more **candidate models**
- Uses a separate **judge model** to score each answer against a structured rubric (including “auto-fail” conditions)
- Writes local, reproducible artifacts and reports under `runs/<runId>/`

No browsing, no tools, no retrieval—just the model, the prompt, and the consequences.

The optional Wikipedia retrieval track keeps that baseline intact and adds separate wiki-enabled
conditions for local BM25, dense, hybrid, RAG, and agent-style runs. See
[`docs/wiki-retrieval.md`](docs/wiki-retrieval.md) for setup.

You can read my full writeup [here](https://www.crowlabs.tech/blog/apocalypse-bench).

## What’s in this repo

- **Runner CLI**: orchestrates dataset loading, candidate generation, judge scoring, and artifact writing.
- **Dataset**: a JSON (JSONL) question bank in `data/question_bank/` is the single source of truth, loaded directly at runtime.
- **Reports**: generates local HTML and Markdown exports for a run.
- **Dashboard**: a Next.js app in `dashboard/` for exploring runs.
- **Wiki retrieval track**: optional local Markdown Wikipedia indexing through the Rust
  `wiki-search` tool, with retrieval traces persisted per result.

## Quick start

### Prereqs

- Node.js `>= 20`
- `pnpm`

### Install

```bash
pnpm install
```

### Run the CLI (dev)

```bash
pnpm dev run -c apocbench.yml
```

### Build the CLI

```bash
pnpm build
```

After building, the `apocbench` binary points at `dist/cli/index.js`.

## Configure a run

The default config is `apocbench.yml`. It defines:

- Where to read the compiled dataset JSONL
- Which candidate models to run (local via Ollama and/or hosted via OpenRouter)
- Which judge model to use
- Concurrency, budget limits, and output options

Key fields (high level):

- `run.datasetPaths`: JSONL directories (compiled runtime dataset)
- `run.outDir`: where run artifacts go (default `./runs`)
- `candidate`: default generation params
- `judge`: judge routing/model and structured output settings
- `routers`: router endpoints (e.g. Ollama base URL, OpenRouter base URL)
- `models`: the candidate model list

### Providers / routers

- **Ollama (local):** set the `routers.ollama.baseUrl` (default is `http://localhost:11434/api`).
- **OpenAI-compatible local servers:** set `routers.openaiCompatible.baseUrl` to the server's `/v1`
  endpoint and use `router: 'openai-compatible'` on the model. Leave `apiKeyEnv: null` for no-auth
  local servers, or set it to an env var name for authenticated proxies.
- **OpenRouter (hosted):** set `OPENROUTER_API_KEY` in your environment (see `apocbench.yml`).

Example local OpenAI-compatible candidate:

```yaml
routers:
  openaiCompatible:
    baseUrl: 'http://127.0.0.1:1234/v1'
    apiKeyEnv: null
    default:
      temperature: 0.5
      maxTokens: 4000
      timeoutMs: 120000

models:
  - id: 'local-openai'
    router: 'openai-compatible'
    model: 'local-model'
```

For an authenticated OpenAI-compatible server, set `apiKeyEnv` to an environment variable name:

```yaml
routers:
  openaiCompatible:
    baseUrl: 'https://your-proxy.example.com/v1'
    apiKeyEnv: 'LOCAL_OPENAI_API_KEY'
    default:
      temperature: 0.5
      maxTokens: 4000
      timeoutMs: 120000
```

## Dataset (V2)

The question bank is **JSON, and JSON is the single source of truth**: 13 per-category JSONL files
under `data/question_bank/` (`AGR.jsonl`, `CHEM.jsonl`, …), one question object per line. `apocbench`
loads them directly — there is no compile step. See `data/question_bank/info.md` for the schema and
the V2 authoring rules (specific binary rubrics; auto-fails that measure refusal and dangerous error,
never the topic).

Edit the JSONL directly, then:

```bash
pnpm -s test -- test/dataset-validate.test.ts   # enforce the V2 contract
pnpm -s dataset:export                          # refresh the read-only docs/question-bank.md
```

`docs/question-bank.md` is a generated, human-readable copy for browsing (kept in sync by
`test/dataset-export-fresh.test.ts`); do not edit it by hand.

## Outputs

Runs write to `runs/<runId>/`. Expect a mix of:

- Per-question artifacts (candidate answer + judge output)
- Aggregated summaries (JSON)
- Local reports (HTML) and Markdown exports

You can also export Markdown for an existing run ID (see `apocbench --help`).

## Dashboard

There’s a separate Next.js app in `dashboard/`.

```bash
cd dashboard
pnpm install
pnpm dev
```

## Development

Useful commands:

```bash
pnpm lint
pnpm typecheck
pnpm test
```
