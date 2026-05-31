# Dataset workflow (V2)

## Source of truth

- The question bank is **JSON**. The 13 per-category JSONL files in `data/question_bank/*.jsonl`
  are the single source of truth, edited directly.
- `apocbench` loads them directly at runtime (`run.datasetPaths` in `apocbench.yml`). There is no
  compile step and no separate markdown source.
- Schema and authoring rules: `data/question_bank/info.md`.

## Validate

The V2 contract (id format, canonical categories, exactly 10 rubric items, a refusal auto-fail plus
only technically-wrong/unsafe auto-fails, real scenarios, reference facts, `version: v2`) is enforced
by:

```bash
pnpm -s test -- test/dataset-validate.test.ts
```

## Human-readable export

`docs/question-bank.md` is a generated, read-only rendering of the whole bank for browsing. Refresh it
after editing the JSONL:

```bash
pnpm -s dataset:export
```

A freshness test (`test/dataset-export-fresh.test.ts`) fails if the committed markdown drifts from the
JSONL, so regenerate and commit it together with dataset changes.
