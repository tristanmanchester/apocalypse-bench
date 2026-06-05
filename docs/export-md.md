# Export run to Markdown pack

## Command

From the repo root:

```bash
pnpm -s dev export-md <runId>
```

Example:

```bash
pnpm -s dev export-md default-20251221-220648
```

## Outputs

The exporter writes a deterministic folder (default is `runs/<runId>/markdown/`):

- `RUN.md`
- `metadata.json`
- `results.jsonl`
- `by-domain/`
- `by-model/`
- `cases/` (when `--include-cases=true`)

## Flags

- `--out <dir>`: override output directory (default: `runs/<runId>/markdown`).
- `--mode <by-domain|by-model|both>`: choose output grouping (default: `both`).
- `--include-cases`: include `cases/<caseId>.md` (default: true).
- `--overwrite`: replace existing output directory (default: false).
- `--redact <none|basic>`: only `none` is currently supported.

## Common issues

If you see a `better-sqlite3` native module mismatch, approve and rebuild native deps:

```bash
pnpm approve-builds
pnpm rebuild better-sqlite3
```
