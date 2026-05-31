import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, test } from 'vitest';
import { loadJsonl, loadJsonlMany } from '../src/core/dataset/loadJsonl';

const DATASET_DIR = 'data/question_bank';

const CANON: Record<string, string> = {
  AGR: 'Agriculture',
  CHEM: 'Chemistry',
  COMMS: 'Communications',
  ENG: 'Engineering',
  ENR: 'Energy',
  ETH: 'Ethics',
  PH: 'Public Health',
  MAT: 'Materials',
  MEAS: 'Measurement',
  MED: 'Medicine',
  ORG: 'Organisation',
  PED: 'Pedagogy',
  SAFE: 'Safety',
};

const DIFFICULTIES = new Set(['Easy', 'Medium', 'Hard', 'Very Hard']);
const TASK_TYPES = new Set([
  'procedure', 'troubleshoot', 'critique', 'compare', 'investigate', 'plan', 'lesson', 'assessment',
]);

const BANNED_RUBRIC = [
  /^directly answers the task\b/i,
  /^addresses the question\b/i,
  /^clear structure\b/i,
  /^well[- ]organised\b/i,
  /^actionable steps\.?$/i,
  /^actionable advice\.?$/i,
  /^uses only (the )?stated materials?\.?$/i,
  /^states assumptions\.?$/i,
  /^viably covers\b/i,
  /^gives an ordered,?\s*(step-by-step )?procedure\.?$/i,
];

const REFUSAL = /refus|non[- ]answer|declin|moral lecture|moralis/i;

function listAreaFiles(): string[] {
  return fs
    .readdirSync(path.resolve(DATASET_DIR))
    .filter(f => f.endsWith('.jsonl'))
    .sort();
}

describe('question bank V2 contract', () => {
  test('every area file is a non-empty canonical set', () => {
    const files = listAreaFiles();
    expect(files.length).toBe(Object.keys(CANON).length);
    for (const f of files) {
      const area = path.basename(f, '.jsonl');
      expect(CANON[area], `unexpected area file ${f}`).toBeDefined();
    }
  });

  test('all questions satisfy the V2 contract', () => {
    const files = listAreaFiles();
    const problems: string[] = [];

    for (const f of files) {
      const area = path.basename(f, '.jsonl');
      const canonCategory = CANON[area]!;
      const { lines } = loadJsonl(path.join(DATASET_DIR, f));
      expect(lines.length, `${f} should be non-empty`).toBeGreaterThan(0);

      for (const q of lines) {
        const tag = q.id;
        if (!/^[A-Z]+-\d{3}$/.test(q.id)) problems.push(`${tag}: bad id format`);
        if (q.id.split('-')[0] !== area) problems.push(`${tag}: id prefix != ${area}`);
        if (q.area !== area) problems.push(`${tag}: area field ${q.area} != ${area}`);
        if (q.category !== canonCategory) problems.push(`${tag}: category ${q.category} != ${canonCategory}`);
        if (!DIFFICULTIES.has(q.difficulty)) problems.push(`${tag}: bad difficulty ${q.difficulty}`);
        if (!q.task_type || !TASK_TYPES.has(q.task_type)) problems.push(`${tag}: bad task_type ${q.task_type}`);
        if (q.rubric.length !== 10) problems.push(`${tag}: rubric has ${q.rubric.length} items (need 10)`);
        for (const r of q.rubric) {
          if (BANNED_RUBRIC.some(re => re.test(r.text.trim()))) {
            problems.push(`${tag}: banned-fluff rubric item: ${r.text.slice(0, 50)}`);
          }
        }
        if (!q.auto_fail.length || (q.auto_fail.length === 1 && q.auto_fail[0] === '(omitted)')) {
          problems.push(`${tag}: empty/omitted auto_fail`);
        } else if (!q.auto_fail.some(a => REFUSAL.test(a))) {
          problems.push(`${tag}: no refusal auto_fail condition`);
        }
        const refSet = new Set(q.reference_facts ?? []);
        for (const a of q.auto_fail) {
          if (refSet.has(a)) problems.push(`${tag}: auto_fail duplicates a reference_fact (leak)`);
          if (a.length > 320) problems.push(`${tag}: absurdly long auto_fail`);
        }
        if (!q.scenario.length || (q.scenario.length === 1 && q.scenario[0] === '(omitted)')) {
          problems.push(`${tag}: scenario omitted/empty`);
        }
        if (!q.prompt.trim()) problems.push(`${tag}: empty prompt`);
        if (!q.reference_facts || q.reference_facts.length === 0) problems.push(`${tag}: no reference_facts`);
        if (q.version !== 'v2') problems.push(`${tag}: version != v2 (${q.version})`);
      }
    }

    expect(problems, problems.slice(0, 50).join('\n')).toEqual([]);
  });

  test('question ids are globally unique and the whole bank loads', () => {
    // loadJsonlMany throws on duplicate ids or area/filename mismatch.
    const { lines } = loadJsonlMany([DATASET_DIR]);
    const ids = new Set(lines.map(l => l.id));
    expect(ids.size).toBe(lines.length);
    expect(lines.length).toBeGreaterThan(380);
  });
});
