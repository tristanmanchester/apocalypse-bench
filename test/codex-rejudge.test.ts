import { describe, expect, test } from 'vitest';

import type { DatasetLine } from '../src/core/dataset/schema';
import type { JudgeOutput } from '../src/core/runner/types';
import {
  buildBatchPrompt,
  buildJudgeBatches,
  outputSchemaForBatch,
  validateBatchOutput,
  type JudgeCase,
  type SourceResultRow,
} from '../scripts/codex_rejudge';

function question(id: string, category: string): DatasetLine {
  return {
    id,
    category,
    title: `${id} title`,
    difficulty: 'Medium',
    scenario: [`Scenario for ${id}`],
    prompt: `What should be done for ${id}?`,
    rubric: Array.from({ length: 10 }, (_, index) => ({
      id: `r${index + 1}`,
      text: `Rubric ${index + 1}`,
      weight: 1,
      maxScore: 1,
    })),
    auto_fail: ['Refusal to answer.'],
    reference_facts: ['Reference fact.'],
  };
}

function sourceRow(questionId: string, modelId: string): SourceResultRow {
  return {
    question_id: questionId,
    model_id: modelId,
    status: 'done',
    candidate_prompt: `Prompt for ${questionId}`,
    candidate_completion: `Answer from ${modelId} for ${questionId}`,
    candidate_metrics_json: null,
    retrieval_trace_json: null,
  };
}

function judgeCase(questionId: string, category: string, modelId: string): JudgeCase {
  return {
    row: sourceRow(questionId, modelId),
    question: question(questionId, category),
  };
}

function key(judgeCase: JudgeCase): string {
  return `${judgeCase.question.category}:${judgeCase.row.question_id}:${judgeCase.row.model_id}`;
}

function outputItem(
  questionId: string,
  modelId: string,
  overrides: Partial<JudgeOutput> = {},
) {
  return {
    question_id: questionId,
    model_id: modelId,
    rubric_scores: Object.fromEntries(
      Array.from({ length: 10 }, (_, index) => [`r${index + 1}`, 1]),
    ),
    auto_fail: false,
    auto_fail_reason: '',
    overall_score: 10,
    notes: 'ok',
    unsafe_flags: [],
    ...overrides,
  };
}

describe('codex rejudge batching', () => {
  test('sequential preserves current ordering and batch behavior', () => {
    const cases = [
      judgeCase('Q3', 'Medicine', 'dense'),
      judgeCase('Q1', 'Engineering', 'direct'),
      judgeCase('Q2', 'Medicine', 'bm25'),
    ];

    const batches = buildJudgeBatches(cases, 2, 'sequential');

    expect(batches.map((batch) => batch.cases.map(key))).toEqual([
      [key(cases[0]), key(cases[1])],
      [key(cases[2])],
    ]);
    expect(batches.map((batch) => batch.category)).toEqual([null, null]);
  });

  test('category never mixes categories and sorts deterministically', () => {
    const cases = [
      judgeCase('MED-002', 'Medicine', 'hybrid'),
      judgeCase('ENG-002', 'Engineering', 'dense'),
      judgeCase('ENG-001', 'Engineering', 'bm25'),
      judgeCase('MED-001', 'Medicine', 'direct'),
      judgeCase('ENG-001', 'Engineering', 'direct'),
    ];

    const batches = buildJudgeBatches(cases, 10, 'category');

    expect(batches).toHaveLength(2);
    for (const batch of batches) {
      expect(new Set(batch.cases.map((item) => item.question.category)).size).toBe(1);
    }
    expect(batches[0].category).toBe('Engineering');
    expect(batches[0].cases.map(key)).toEqual([
      'Engineering:ENG-001:bm25',
      'Engineering:ENG-001:direct',
      'Engineering:ENG-002:dense',
    ]);
    expect(batches[1].category).toBe('Medicine');
  });

  test('category handles category sizes not divisible by batch size', () => {
    const cases = [
      judgeCase('ENG-001', 'Engineering', 'direct'),
      judgeCase('ENG-002', 'Engineering', 'direct'),
      judgeCase('ENG-003', 'Engineering', 'direct'),
      judgeCase('MED-001', 'Medicine', 'direct'),
    ];

    const batches = buildJudgeBatches(cases, 2, 'category');

    expect(batches.map((batch) => [batch.category, batch.cases.length])).toEqual([
      ['Engineering', 2],
      ['Engineering', 1],
      ['Medicine', 1],
    ]);
  });

  test('model groups by model while allowing broad categories in one batch', () => {
    const cases = [
      judgeCase('MAT-040', 'Materials', 'bm25-r02'),
      judgeCase('ENR-005', 'Energy', 'bm25-r01'),
      judgeCase('SAFE-023', 'Safety', 'bm25-r01'),
      judgeCase('AGR-009', 'Agriculture', 'bm25-r02'),
    ];

    const batches = buildJudgeBatches(cases, 10, 'model');

    expect(
      batches.map((batch) => [batch.modelId, batch.category, batch.cases.map(key)]),
    ).toEqual([
      ['bm25-r01', null, ['Energy:ENR-005:bm25-r01', 'Safety:SAFE-023:bm25-r01']],
      ['bm25-r02', null, ['Agriculture:AGR-009:bm25-r02', 'Materials:MAT-040:bm25-r02']],
    ]);
  });

  test('category uses only the supplied cases so upstream model, limit, and resume filters are respected', () => {
    const filteredCases = [
      judgeCase('ENG-002', 'Engineering', 'dense'),
      judgeCase('MED-001', 'Medicine', 'dense'),
    ];

    const batches = buildJudgeBatches(filteredCases, 10, 'category');

    expect(
      batches.flatMap((batch) => batch.cases.map((item) => item.row.model_id)),
    ).toEqual(['dense', 'dense']);
    expect(
      batches.flatMap((batch) => batch.cases.map((item) => item.row.question_id)),
    ).toEqual(['ENG-002', 'MED-001']);
  });

  test('category-model groups by both category and model', () => {
    const cases = [
      judgeCase('ENG-001', 'Engineering', 'direct'),
      judgeCase('ENG-002', 'Engineering', 'dense'),
      judgeCase('ENG-003', 'Engineering', 'direct'),
      judgeCase('MED-001', 'Medicine', 'direct'),
    ];

    const batches = buildJudgeBatches(cases, 10, 'category-model');

    expect(
      batches.map((batch) => [
        batch.category,
        batch.modelId,
        batch.cases.map((item) => item.row.question_id),
      ]),
    ).toEqual([
      ['Engineering', 'dense', ['ENG-002']],
      ['Engineering', 'direct', ['ENG-001', 'ENG-003']],
      ['Medicine', 'direct', ['MED-001']],
    ]);
  });
});

describe('codex rejudge prompt and schema', () => {
  test('category prompt includes calibration and anti-contamination instructions', () => {
    const batch = buildJudgeBatches(
      [
        judgeCase('ENG-001', 'Engineering', 'direct'),
        judgeCase('ENG-002', 'Engineering', 'hybrid'),
      ],
      10,
      'category',
    )[0];

    const prompt = buildBatchPrompt(batch);

    expect(prompt).toContain('category-local for category "Engineering"');
    expect(prompt).toContain(
      'Do not use facts, wording, or missing details from one candidate answer',
    );
    expect(prompt).toContain(
      'A candidate receives credit only for content present in its own answer',
    );
    expect(prompt).toContain('Do not rank answers against each other');
    expect(prompt).toContain(
      'The top-level "results" array must contain one result for every case',
    );
  });

  test('output schema requires exactly one result per input case', () => {
    const schema = outputSchemaForBatch(3) as {
      properties: { results: { minItems: number; maxItems: number } };
    };

    expect(schema.properties.results.minItems).toBe(3);
    expect(schema.properties.results.maxItems).toBe(3);
  });

  test('validation accepts complete output and rejects malformed output', () => {
    const cases = [
      judgeCase('ENG-001', 'Engineering', 'direct'),
      judgeCase('ENG-002', 'Engineering', 'hybrid'),
    ];

    expect(() =>
      validateBatchOutput(cases, {
        results: [outputItem('ENG-001', 'direct'), outputItem('ENG-002', 'hybrid')],
      }),
    ).not.toThrow();

    expect(() =>
      validateBatchOutput(cases, {
        results: [outputItem('ENG-001', 'direct')],
      }),
    ).toThrow(/returned 1 results for 2 cases/);

    expect(() =>
      validateBatchOutput(cases, {
        results: [outputItem('ENG-001', 'direct'), outputItem('ENG-999', 'hybrid')],
      }),
    ).toThrow(/unexpected result/);

    expect(() =>
      validateBatchOutput(cases, {
        results: [
          outputItem('ENG-001', 'direct'),
          outputItem('ENG-002', 'hybrid', {
            rubric_scores: { r1: 1 } as JudgeOutput['rubric_scores'],
          }),
        ],
      }),
    ).toThrow(/missing numeric rubric score/);

    expect(() =>
      validateBatchOutput(cases, {
        results: [
          outputItem('ENG-001', 'direct'),
          outputItem('ENG-002', 'hybrid', { unsafe_flags: 'bad' as unknown as string[] }),
        ],
      }),
    ).toThrow(/non-array unsafe_flags/);
  });
});
