# Apocalypse Bench dashboard

A local [Next.js](https://nextjs.org) app for browsing `apocbench` run outputs.
It lists completed runs and, for each one, shows score breakdowns by model,
category, and difficulty, with charts.

The dashboard reads the run artifacts written by the main `apocbench` CLI. By
default it looks for them in `../runs` (the `runs/` directory at the repository
root). Set the `RUNS_DIR` environment variable to read from another location.

## Requirements

- A completed `apocbench` run under `runs/`. See the [root README](../README.md)
  for how to produce one.
- Node.js 20 or newer and pnpm 10 or newer.

## Run the dashboard

From the `dashboard/` directory:

```bash
pnpm install
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000). The home page redirects to
`/runs`, which lists every run found under the runs directory. Select a run to
see its model, category, and difficulty breakdowns.

To read runs from a custom location, set `RUNS_DIR`:

```bash
RUNS_DIR=/path/to/runs pnpm dev
```

## Build for production

```bash
pnpm build
pnpm start
```

## Checks

```bash
pnpm lint
pnpm test
```
