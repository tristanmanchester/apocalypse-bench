import fs from 'node:fs';
import path from 'node:path';
import { loadJsonlMany } from '../core/dataset/loadJsonl';
import type { DatasetLine } from '../core/dataset/schema';

// Usage:
//   pnpm -s dataset:export            # regenerates docs/question-bank.md from data/question_bank/*.jsonl
//   pnpm -s dataset:export -- --in data/question_bank --out docs/question-bank.md
//
// The per-category JSONL files in `data/question_bank/*.jsonl` are the SINGLE source of truth.
// This script renders a human-readable, READ-ONLY markdown copy for browsing. Edit the JSONL,
// then re-run this. `test/dataset-export-fresh.test.ts` fails if the committed markdown is stale.

const CANON_ORDER = [
  'AGR', 'CHEM', 'COMMS', 'ENG', 'ENR', 'ETH', 'PH', 'MAT', 'MEAS', 'MED', 'ORG', 'PED', 'SAFE',
];

const DIFFICULTY_ORDER = ['Easy', 'Medium', 'Hard', 'Very Hard'];

function rubricText(item: DatasetLine['rubric'][number]): string {
  return item.text.replace(/\s+/g, ' ').trim();
}

function renderQuestion(q: DatasetLine): string {
  const lines: string[] = [];
  lines.push(`### ${q.id} — ${q.title ?? ''}`.trimEnd());
  lines.push('');
  lines.push(`- **Difficulty:** ${q.difficulty}`);
  lines.push(`- **Task type:** ${q.task_type ?? 'procedure'}`);
  lines.push('');
  lines.push('**Scenario**');
  for (const s of q.scenario) lines.push(`- ${s}`);
  lines.push('');
  lines.push('**Prompt**');
  lines.push('');
  lines.push('```text');
  lines.push((q.prompt ?? '').trimEnd());
  lines.push('```');
  lines.push('');
  lines.push('**Rubric (10 points)**');
  lines.push('');
  q.rubric.forEach((r, i) => lines.push(`${i + 1}. ${rubricText(r)}`));
  lines.push('');
  lines.push('**Auto-fail (score = 0 if any)**');
  lines.push('');
  for (const a of q.auto_fail) lines.push(`- ${a}`);
  if (q.reference_facts && q.reference_facts.length > 0) {
    lines.push('');
    lines.push('**Reference facts (for judge)**');
    lines.push('');
    for (const f of q.reference_facts) lines.push(`- ${f}`);
  }
  return lines.join('\n');
}

/** Pure renderer: lines -> full markdown document. Used by the export script and the freshness test. */
export function renderQuestionBankMarkdown(lines: DatasetLine[]): string {
  const byArea = new Map<string, DatasetLine[]>();
  for (const q of lines) {
    const area = q.area ?? q.id.split('-')[0]!;
    const arr = byArea.get(area) ?? [];
    arr.push(q);
    byArea.set(area, arr);
  }

  const areas = Array.from(byArea.keys()).sort((a, b) => {
    const ia = CANON_ORDER.indexOf(a);
    const ib = CANON_ORDER.indexOf(b);
    if (ia !== -1 && ib !== -1) return ia - ib;
    if (ia !== -1) return -1;
    if (ib !== -1) return 1;
    return a < b ? -1 : a > b ? 1 : 0;
  });

  const sortQuestions = (arr: DatasetLine[]) =>
    [...arr].sort((x, y) => {
      const dx = DIFFICULTY_ORDER.indexOf(x.difficulty);
      const dy = DIFFICULTY_ORDER.indexOf(y.difficulty);
      if (dx !== dy) return dx - dy;
      return x.id < y.id ? -1 : x.id > y.id ? 1 : 0;
    });

  const out: string[] = [];
  out.push('# Apocalypse-Bench V2 — Question Bank');
  out.push('');
  out.push(
    '> GENERATED FILE — do not edit by hand. The source of truth is `data/question_bank/*.jsonl`.',
  );
  out.push('> Regenerate with `pnpm -s dataset:export`.');
  out.push('');
  out.push(`Total questions: **${lines.length}** across **${areas.length}** categories.`);
  out.push('');

  // Summary table.
  out.push('| Code | Category | Questions |');
  out.push('| --- | --- | --- |');
  for (const area of areas) {
    const arr = byArea.get(area)!;
    out.push(`| ${area} | ${arr[0]!.category} | ${arr.length} |`);
  }
  out.push('');

  for (const area of areas) {
    const arr = sortQuestions(byArea.get(area)!);
    out.push(`## ${area} — ${arr[0]!.category} (${arr.length})`);
    out.push('');
    out.push(arr.map(renderQuestion).join('\n\n---\n\n'));
    out.push('');
  }

  return out.join('\n').replace(/\n+$/g, '\n');
}

function main(): void {
  const args = process.argv.slice(2);
  const inIdx = args.indexOf('--in');
  const outIdx = args.indexOf('--out');
  const inDir = inIdx === -1 ? 'data/question_bank' : args[inIdx + 1]!;
  const outFile = outIdx === -1 ? 'docs/question-bank.md' : args[outIdx + 1]!;

  const { lines } = loadJsonlMany([inDir]);
  const md = renderQuestionBankMarkdown(lines);
  fs.mkdirSync(path.dirname(path.resolve(outFile)), { recursive: true });
  fs.writeFileSync(path.resolve(outFile), md, 'utf8');
  console.error(`Wrote ${lines.length} questions to ${outFile}`);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}
