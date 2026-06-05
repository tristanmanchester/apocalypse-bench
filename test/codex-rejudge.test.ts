import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

import { describe, expect, test } from 'vitest';

import type { DatasetLine } from '../src/core/dataset/schema';
import type { JudgeOutput } from '../src/core/runner/types';
import {
  buildBatchPrompt,
  buildJudgeBatches,
  loadSourceRows,
  normalizeCodexJudgeOutput,
  outputSchemaForBatch,
  parseCodexRejudgeArgs,
  setupRun,
  validateBatchOutput,
  type JudgeCase,
  type SourceResultRow,
} from '../src/core/judge/codex';
import { closeDb } from '../src/storage/sqlite/db';
import { openAndMigrate } from '../src/storage/sqlite/migrate';
import { insertQuestions } from '../src/storage/sqlite/questions';
import { upsertResult } from '../src/storage/sqlite/results';
import { insertRun } from '../src/storage/sqlite/runs';

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
  test('parser accepts positive Codex batch concurrency', () => {
    const args = parseCodexRejudgeArgs(['--source-run', 'source', '--concurrency', '5']);

    expect(args.concurrency).toBe(5);
    expect(() =>
      parseCodexRejudgeArgs(['--source-run', 'source', '--concurrency', '0']),
    ).toThrow(/concurrency must be a positive integer/);
  });

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

  test('question-paired groups one question per batch and pairs direct before bm25 research', () => {
    const cases = [
      judgeCase('Q2', 'Medicine', 'gemma-agent-bm25-research'),
      judgeCase('Q1', 'Engineering', 'gemma-agent-bm25-research'),
      judgeCase('Q1', 'Engineering', 'gemma-direct'),
      judgeCase('Q2', 'Medicine', 'gemma-direct'),
    ];

    const batches = buildJudgeBatches(cases, 10, 'question-paired');

    expect(batches).toHaveLength(2);
    expect(
      batches.map((batch) => batch.cases.map((item) => item.row.question_id)),
    ).toEqual([
      ['Q1', 'Q1'],
      ['Q2', 'Q2'],
    ]);
    expect(batches[0].cases.map((item) => item.row.model_id)).toEqual([
      'gemma-direct',
      'gemma-agent-bm25-research',
    ]);
  });

  test('question-paired splits large same-question batches without mixing questions', () => {
    const cases = Array.from({ length: 18 }, (_, index) =>
      judgeCase(
        'Q1',
        'Engineering',
        `model-${String(index).padStart(2, '0')}-${index % 2 === 0 ? 'direct' : 'agent-bm25-research'}`,
      ),
    );

    const batches = buildJudgeBatches(cases, 10, 'question-paired');

    expect(batches.map((batch) => batch.cases.length)).toEqual([10, 8]);
    for (const batch of batches) {
      expect(new Set(batch.cases.map((item) => item.row.question_id))).toEqual(
        new Set(['Q1']),
      );
    }
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

  test('question-paired prompt describes paired calibration without fact sharing', () => {
    const batch = buildJudgeBatches(
      [
        judgeCase('ENG-001', 'Engineering', 'gemma-direct'),
        judgeCase('ENG-001', 'Engineering', 'gemma-agent-bm25-research'),
      ],
      10,
      'question-paired',
    )[0];

    const prompt = buildBatchPrompt(batch);

    expect(prompt).toContain('question-paired for question "ENG-001"');
    expect(prompt).toContain('direct and retrieval candidates for the same task');
    expect(prompt).toContain(
      'Do not use facts, wording, or missing details from one candidate answer',
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

  test('normalization clamps rubric scores and recomputes auto-fail overall score', () => {
    const item = outputItem('ENG-001', 'direct', {
      rubric_scores: Object.fromEntries(
        Array.from({ length: 10 }, (_, index) => [
          `r${index + 1}`,
          index === 0 ? 99 : 1,
        ]),
      ),
      auto_fail: true,
      auto_fail_reason: 'Refusal to answer.',
      overall_score: 8,
    });

    const normalized = normalizeCodexJudgeOutput({
      judgeOutput: item,
      rubric: question('ENG-001', 'Engineering').rubric,
    });

    expect(normalized.rubric_scores.r1).toBe(1);
    expect(normalized.rubric_scores.r2).toBe(1);
    expect(normalized.overall_score).toBe(0);
    expect(normalized.auto_fail).toBe(true);
    expect(normalized.auto_fail_reason).toBe('Refusal to answer.');
  });
});

describe('codex rejudge run setup', () => {
  test('source rows can be filtered by selected question ids', () => {
    const tempDir = mkdtempSync(path.join(tmpdir(), 'apocbench-codex-rows-'));
    const dbPath = path.join(tempDir, 'apocbench.sqlite');

    closeDb();
    const db = openAndMigrate(dbPath);

    try {
      insertRun(db, {
        run_id: 'source-run',
        created_at: new Date().toISOString(),
        tool_version: 'test',
        config_json: '{}',
        dataset_path: 'data/question_bank',
        dataset_sha256: 'source-sha',
        prompt_template_hash: 'prompt-sha',
        status: 'done',
      });
      insertQuestions(db, 'source-run', [
        question('Q1', 'Engineering'),
        question('Q2', 'Engineering'),
        question('Q3', 'Engineering'),
      ]);

      for (const questionId of ['Q1', 'Q2', 'Q3']) {
        for (const modelId of ['direct', 'bm25']) {
          upsertResult(db, {
            runId: 'source-run',
            questionId,
            modelId,
            status: 'candidate_done',
            candidateCompletion: `Answer ${questionId} ${modelId}`,
          });
        }
      }

      const rows = loadSourceRows(db, {
        ...parseCodexRejudgeArgs([
          '--source-run',
          'source-run',
          '--source-status',
          'candidate_done',
        ]),
        questionIds: ['Q1', 'Q2'],
      });

      expect(rows.map((row) => `${row.question_id}:${row.model_id}`)).toEqual([
        'Q1:bm25',
        'Q1:direct',
        'Q2:bm25',
        'Q2:direct',
      ]);
    } finally {
      closeDb();
      rmSync(tempDir, { recursive: true, force: true });
    }
  });

  test('resume seeds newly selected questions before later result writes', () => {
    const tempDir = mkdtempSync(path.join(tmpdir(), 'apocbench-codex-'));
    const dbPath = path.join(tempDir, 'apocbench.sqlite');

    closeDb();
    const db = openAndMigrate(dbPath);

    try {
      insertRun(db, {
        run_id: 'source-run',
        created_at: new Date().toISOString(),
        tool_version: 'test',
        config_json: '{}',
        dataset_path: 'data/question_bank',
        dataset_sha256: 'source-sha',
        prompt_template_hash: 'prompt-sha',
        status: 'done',
      });

      const args = parseCodexRejudgeArgs([
        '--source-run',
        'source-run',
        '--out-run',
        'judge-run',
      ]);

      setupRun(db, args, 'judge-run', [question('Q1', 'Engineering')]);

      const firstCount = db
        .prepare('select count(*) as count from questions where run_id = ?')
        .get('judge-run') as { count: number };
      expect(firstCount.count).toBe(1);

      setupRun(db, { ...args, resume: true }, 'judge-run', [
        question('Q1', 'Engineering'),
        question('Q2', 'Medicine'),
      ]);

      const secondCount = db
        .prepare('select count(*) as count from questions where run_id = ?')
        .get('judge-run') as { count: number };
      expect(secondCount.count).toBe(2);
    } finally {
      closeDb();
      rmSync(tempDir, { recursive: true, force: true });
    }
  });
});
