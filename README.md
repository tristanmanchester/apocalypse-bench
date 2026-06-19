# Apocalypse Bench

`apocalypse-bench` is a local benchmark runner for testing whether language
models can give useful, safe advice when the internet is gone and the task is
practical.

The default benchmark is a no-tools, no-browsing baseline. A candidate model
receives one survival question and writes an answer from its own knowledge. The
runner then scores that answer against a structured 10-point rubric with a
separate judge model.

The repository also includes an optional offline Wikipedia retrieval track. It
keeps the direct baseline intact, then adds separate wiki-enabled conditions for
local BM25, dense, hybrid, RAG, and bounded agent-style retrieval. See
[docs/wiki-retrieval.md](docs/wiki-retrieval.md) for setup.

## What is included

- `apocbench`, a TypeScript CLI for running candidate models, scoring answers,
  and comparing conditions.
- A 500-question JSONL question bank under `data/question_bank/`.
- SQLite-backed run storage under `runs/apocbench.sqlite`.
- Local HTML and Markdown report generation for completed runs.
- Optional local Wikipedia indexing through the Rust `wiki-search` service.
- Optional asynchronous Codex judging for candidate-only runs.
- A separate dashboard app under `dashboard/` for exploring run outputs.

The full writeup is here:
[Apocalypse Bench](https://www.linkedin.com/pulse/upcoming-apocalypse-which-small-llm-help-us-rebuild-manchester-dq08f).

## Requirements

- Node.js 20 or newer
- pnpm 10 or newer
- An OpenRouter API key for hosted models, unless you only use local routers
- Ollama, LM Studio, vLLM, or another OpenAI-compatible server for local models
- Codex CLI if you use the asynchronous Codex judge workflow

Install dependencies:

```bash
pnpm install
cp .env.example .env
```

Add `OPENROUTER_API_KEY` to `.env` if you use OpenRouter.

## Run a small smoke test

Validate the default config:

```bash
pnpm -s dev validate -c apocbench.yml
```

Run a tiny direct benchmark:

```bash
pnpm -s dev run -c apocbench.yml smoke-run --limit 2
```

Build the CLI:

```bash
pnpm build
```

After building, the package exposes the `apocbench` binary from
`dist/cli/index.js`.

## Run candidates first, judge later

The current research configs usually run candidate generation first and judge
afterward. This makes long model runs easier to resume and keeps expensive
judge calls separate from candidate failures.

The direct-vs-BM25 research config runs all 500 questions across nine model
families and two conditions, for 9,000 candidate answers:

```bash
pnpm -s dev validate -c apocbench-direct-vs-bm25-research.yml --quiet
pnpm -s dev run-and-judge -c apocbench-direct-vs-bm25-research.yml \
  direct-vs-bm25-research-001 --resume
```

To split the same workflow into explicit stages:

```bash
pnpm -s dev run -c apocbench-direct-vs-bm25-research.yml \
  direct-vs-bm25-research-001

pnpm -s dev judge -c apocbench-direct-vs-bm25-research.yml \
  --source-run direct-vs-bm25-research-001 \
  --out-run direct-vs-bm25-research-001-codex-question-paired-b10 \
  --resume

pnpm -s dev compare \
  --run direct-vs-bm25-research-001-codex-question-paired-b10 \
  --baseline-suffix direct \
  --comparison-suffix agent-bm25-research \
  --out logs/direct-vs-bm25-research-001-comparison.json
```

Use `--limit` for smoke tests before launching a full run.

## Configure models

Configs are YAML files. `apocbench.yml` is the smallest starting point. It
defines the dataset, output directory, routers, model list, concurrency, and
judge behavior.

OpenRouter models need `OPENROUTER_API_KEY`:

```yaml
routers:
  openrouter:
    baseUrl: 'https://openrouter.ai/api/v1'
    apiKeyEnv: 'OPENROUTER_API_KEY'
```

Local OpenAI-compatible servers can run without an API key:

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

Ollama uses `http://localhost:11434/api` by default.

## Dataset

The question bank is JSONL and is the source of truth. Each category has one
file under `data/question_bank/`, and each line is one question object.

Useful commands:

```bash
pnpm -s test -- test/dataset-validate.test.ts
pnpm -s dataset:export
```

`docs/question-bank.md` is generated from the JSONL files for browsing. Do not
edit it by hand. See `data/question_bank/info.md` for the schema and authoring
rules.

## Outputs

Runs write local artifacts under `runs/`:

- `runs/apocbench.sqlite`, the shared SQLite database
- per-question candidate answers and judge outputs
- aggregate summaries
- local HTML and Markdown reports

`runs/`, `logs/`, local wiki indexes, and generated experiment configs are
ignored by Git.

## Wikipedia retrieval

The optional retrieval track uses local Wikipedia data and never gives the model
live web access. The main pieces are:

- a Markdown Wikipedia corpus stored under `data/wiki/`
- a Rust BM25 service in `crates/wiki-search`
- optional dense embeddings and hybrid search
- agent modes that expose bounded wiki search and read tools

Start with [docs/wiki-retrieval.md](docs/wiki-retrieval.md). Use
`apocbench-wiki.yml` for a smoke config and
`apocbench-direct-vs-bm25-research.yml` for the direct-vs-BM25 research matrix.

## Development

Common checks:

```bash
pnpm lint
pnpm typecheck
pnpm test
```

The dashboard is a separate app:

```bash
cd dashboard
pnpm install
pnpm dev
```

## License

MIT. See [LICENSE](LICENSE).
