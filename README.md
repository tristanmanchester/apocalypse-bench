# Apocalypse-bench

You’ve got a laptop, a pile of scrap, and exactly zero internet. The question isn’t “can an LLM write code?”—it’s “can it help you do real work without getting anyone hurt?”

`apocalypse-bench` (CLI: `apocbench`) is a TypeScript benchmark runner that:

- Runs a fixed survival/offline-assistant question bank against one or more **candidate models**
- Uses a separate **judge model** to score each answer against a structured rubric (including “auto-fail” conditions)
- Writes local, reproducible artifacts and reports under `runs/<runId>/`

No browsing, no tools, no retrieval—just the model, the prompt, and the consequences.

You can read my full writeup [here](https://www.crowlabs.tech/blog/apocalypse-bench).

## What’s in this repo

- **Runner CLI**: orchestrates dataset loading, candidate generation, judge scoring, and artifact writing.
- **Dataset tooling**: markdown source-of-truth → compiled JSONL used at runtime.
- **Reports**: generates local HTML and Markdown exports for a run.
- **Dashboard**: a Next.js app in `dashboard/` for exploring runs.

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
- **OpenRouter (hosted):** set `OPENROUTER_API_KEY` in your environment (see `apocbench.yml`).

## Dataset workflow

The dataset source of truth lives in `data/question_bank_v8/*.md`.

Compile it to the runtime JSONL (split) format:

```bash
pnpm -s compile:dataset -- --in data/question_bank_v8 --out data/question_bank_v8_jsonl
```

Verify the markdown and JSONL stay in sync:

```bash
pnpm -s test -- test/dataset-sync-split.test.ts
```

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
