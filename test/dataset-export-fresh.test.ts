import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, test } from 'vitest';
import { loadJsonlMany } from '../src/core/dataset/loadJsonl';
import { renderQuestionBankMarkdown } from '../src/cli/exportDataset';

// Guards that the committed, human-readable export stays in sync with the JSONL source of truth.
// If this fails, run `pnpm -s dataset:export` and commit the regenerated docs/question-bank.md.
describe('question bank markdown export', () => {
  test('docs/question-bank.md is up to date with the JSONL source', () => {
    const { lines } = loadJsonlMany(['data/question_bank']);
    const expected = renderQuestionBankMarkdown(lines);
    const committed = fs.readFileSync(path.resolve('docs/question-bank.md'), 'utf8');
    expect(committed).toBe(expected);
  });
});
