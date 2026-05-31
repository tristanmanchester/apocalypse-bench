import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';

import type { ApocbenchConfig } from '../src/core/config/schema';
import type { DatasetLine } from '../src/core/dataset/schema';
import type { JudgeOutput } from '../src/core/runner/types';
import { runBenchmark } from '../src/core/runner/orchestrator';
import { generateText, type LanguageModel } from 'ai';
import { judgeWithRubricCompletenessRetry } from '../src/core/runner/judge';

vi.mock('ai', async (importOriginal) => {
  const actual = await importOriginal<typeof import('ai')>();
  return {
    ...actual,
    generateText: vi.fn(),
  };
});

vi.mock('../src/core/runner/judge', async () => {
  const actual = await vi.importActual<typeof import('../src/core/runner/judge')>(
    '../src/core/runner/judge',
  );
  return {
    ...actual,
    judgeWithRubricCompletenessRetry: vi.fn(),
  };
});

vi.mock('../src/core/runner/persistence', () => {
  type ResultRow = {
    runId: string;
    modelId: string;
    questionId: string;
    status: string;
    candidateMetricsJson?: string | null;
    candidatePrompt?: string | null;
    candidateCompletion?: string | null;
    judgeParsedJson?: string | null;
    errorJson?: string | null;
    scoreOverall?: number | null;
    autoFail?: boolean | null;
    autoFailReason?: string | null;
    scoreRubricJson?: string | null;
    judgeRequestJson?: string | null;
    judgeResponseJson?: string | null;
  };

  const results = new Map<string, ResultRow>();
  const questions = new Map<string, DatasetLine>();

  const keyFor = (runId: string, modelId: string, questionId: string) =>
    `${runId}:${modelId}:${questionId}`;

  const db = {
    transaction: (fn: () => void) => fn(),
    prepare: () => ({
      all: (runId: string, ...modelIds: string[]) => {
        const rows: Array<{
          model_id: string;
          question_id: string;
          status: string;
          score_overall: number | null;
          auto_fail: number | null;
          candidate_metrics_json: string | null;
          category: string | null;
          difficulty: string | null;
        }> = [];
        for (const row of results.values()) {
          if (row.runId !== runId) continue;
          if (modelIds.length > 0 && !modelIds.includes(row.modelId)) continue;
          const q = questions.get(`${runId}:${row.questionId}`);
          rows.push({
            model_id: row.modelId,
            question_id: row.questionId,
            status: row.status,
            score_overall: typeof row.scoreOverall === 'number' ? row.scoreOverall : null,
            auto_fail: row.autoFail ? 1 : 0,
            candidate_metrics_json: row.candidateMetricsJson ?? null,
            category: q?.category ?? null,
            difficulty: q?.difficulty ?? null,
          });
        }
        return rows;
      },
    }),
  };

  return {
    openRunnerDb: () => db,
    ensureRunStarted: () => {},
    insertRunQuestions: (_db: unknown, runId: string, qs: DatasetLine[]) => {
      for (const q of qs) {
        questions.set(`${runId}:${q.id}`, q);
      }
    },
    isRunResultDone: (_db: unknown, runId: string, modelId: string, questionId: string) => {
      const row = results.get(keyFor(runId, modelId, questionId));
      return row?.status === 'done' && Boolean(row.judgeParsedJson);
    },
    upsertRunResult: (_db: unknown, params: ResultRow) => {
      const key = keyFor(params.runId, params.modelId, params.questionId);
      const existing = results.get(key);
      if (!existing) {
        results.set(key, params);
        return;
      }
      const merged: ResultRow = { ...existing, ...params };
      const preserve = <K extends keyof ResultRow>(keyName: K) => {
        if (params[keyName] == null) {
          merged[keyName] = existing[keyName];
        }
      };
      preserve('candidateMetricsJson');
      preserve('candidatePrompt');
      preserve('candidateCompletion');
      preserve('judgeParsedJson');
      preserve('judgeRequestJson');
      preserve('judgeResponseJson');
      preserve('scoreOverall');
      preserve('scoreRubricJson');
      preserve('autoFail');
      preserve('autoFailReason');
      preserve('errorJson');
      results.set(key, merged);
    },
    listRunResults: (_db: unknown, runId: string) => {
      const rows: Array<{
        run_id: string;
        model_id: string;
        question_id: string;
        score_overall: number | null;
        score_rubric_json: string | null;
        auto_fail: number | null;
        auto_fail_reason: string | null;
        status: string;
        candidate_metrics_json: string | null;
        candidate_prompt: string | null;
        candidate_completion: string | null;
        judge_parsed_json: string | null;
        error_json: string | null;
        category: string | null;
        difficulty: string | null;
        prompt: string | null;
      }> = [];
      for (const row of results.values()) {
        if (row.runId !== runId) continue;
        const q = questions.get(`${runId}:${row.questionId}`);
        rows.push({
          run_id: row.runId,
          model_id: row.modelId,
          question_id: row.questionId,
          score_overall: typeof row.scoreOverall === 'number' ? row.scoreOverall : null,
          score_rubric_json: row.scoreRubricJson ?? null,
          auto_fail: row.autoFail ? 1 : 0,
          auto_fail_reason: row.autoFailReason ?? null,
          status: row.status,
          candidate_metrics_json: row.candidateMetricsJson ?? null,
          candidate_prompt: row.candidatePrompt ?? null,
          candidate_completion: row.candidateCompletion ?? null,
          judge_parsed_json: row.judgeParsedJson ?? null,
          error_json: row.errorJson ?? null,
          category: q?.category ?? null,
          difficulty: q?.difficulty ?? null,
          prompt: q?.prompt ?? null,
        });
      }
      return rows;
    },
    updateRunStatusForRun: () => {},
    __getStore: () => ({ results, questions }),
  };
});

const generateTextMock = vi.mocked(generateText);
const judgeMock = vi.mocked(judgeWithRubricCompletenessRetry);

const DATASET_ABS = path.resolve(
  process.cwd(),
  'data/question_bank/PH.jsonl',
);

const RUNS_ROOT = path.resolve(process.cwd(), 'runs-test-orchestrator');

function makeConfig(params: {
  outDir: string;
  maxBudgetUsd?: number | null;
  resume?: boolean;
}): ApocbenchConfig {
  return {
    run: {
      name: 'test',
      datasetPaths: ['./data/question_bank'],
      outDir: params.outDir,
      resume: params.resume ?? true,
      concurrency: { candidate: 1, judge: 1 },
      ...(params.maxBudgetUsd != null ? { maxBudgetUsd: params.maxBudgetUsd } : {}),
    },
    judge: {
      router: 'openrouter',
      model: 'judge-model',
      provider: 'openrouter',
      maxTokens: 1000,
      structured: true,
    },
    routers: {
      ollama: {
        baseUrl: 'http://localhost:11434/v1',
        apiKeyEnv: null,
        default: { temperature: 0.2, maxTokens: 50, timeoutMs: 1000 },
      },
      openrouter: {
        baseUrl: 'https://openrouter.ai/api/v1',
        apiKeyEnv: 'OPENROUTER_API_KEY',
        default: { temperature: 0.2, maxTokens: 50, timeoutMs: 1000 },
      },
    },
    models: [{ id: 'm1', router: 'openrouter', model: 'candidate-model' }],
  };
}

function makeQuestions(count: number): DatasetLine[] {
  return Array.from({ length: count }, (_, idx) => ({
    id: `Q${idx + 1}`,
    category: 'cat',
    difficulty: 'Medium',
    scenario: ['a'],
    prompt: 'p',
    rubric: [{ id: 'r1', text: 't', weight: 1, maxScore: 1 }],
    auto_fail: ['no'],
  }));
}

const judgeOutput: JudgeOutput = {
  rubric_scores: { r1: 1 },
  auto_fail: false,
  overall_score: 1,
  notes: 'ok',
};

const fakeModel: LanguageModel = {
  specificationVersion: 'v2',
  provider: 'test',
  modelId: 'test',
  supportedUrls: {},
  doGenerate: async () => {
    throw new Error('not used');
  },
  doStream: async () => {
    throw new Error('not used');
  },
};

beforeEach(() => {
  vi.clearAllMocks();
});

async function getStore() {
  const mod = (await import('../src/core/runner/persistence')) as unknown as {
    __getStore: () => {
      results: Map<
        string,
        {
          runId: string;
          modelId: string;
          questionId: string;
          status?: string;
          candidateMetricsJson?: string | null;
          judgeParsedJson?: string | null;
        }
      >;
      questions: Map<string, DatasetLine>;
    };
  };
  return mod.__getStore();
}

afterEach(async () => {
  const store = await getStore();
  store.results.clear();
  store.questions.clear();
  fs.rmSync(RUNS_ROOT, { recursive: true, force: true });
});

describe('runBenchmark regression coverage', () => {
  test('budget exceeded skips work and emits event once', async () => {
    const outDir = path.join(RUNS_ROOT, 'budget');
    const config = makeConfig({ outDir, maxBudgetUsd: 0.5 });
    const questions = makeQuestions(2);
    const runId = 'budget-test';
    const events: Array<{ type: string }> = [];

    generateTextMock.mockResolvedValueOnce({
      text: 'candidate',
      usage: { prompt_tokens: 1, completion_tokens: 1 } as unknown,
      providerMetadata: { openrouter: { cost: 1 } },
    } as unknown as Awaited<ReturnType<typeof generateText>>);

    judgeMock.mockResolvedValue({
      object: judgeOutput,
      raw: {},
      didRepairRetry: false,
    });

    await runBenchmark({
      config,
      configPath: 'x',
      datasetPath: 'x',
      datasetAbsolutePath: DATASET_ABS,
      questions,
      deps: {
        resolveModel: () => fakeModel,
        resolveJudgeModel: () => fakeModel,
        toolVersion: 'test',
      },
      dryRun: false,
      runIdOverride: runId,
      onEvent: (e) => events.push(e),
    });

    expect(generateTextMock).toHaveBeenCalledTimes(1);
    expect(judgeMock).not.toHaveBeenCalled();
    expect(events.filter((e) => e.type === 'budget_exceeded')).toHaveLength(1);

    const { results } = await getStore();
    const rows = Array.from(results.values())
      .filter((row) => row.runId === runId)
      .sort((a, b) => a.questionId.localeCompare(b.questionId))
      .map((row) => ({ question_id: row.questionId, status: row.status }));

    expect(rows).toEqual([
      { question_id: 'Q1', status: 'skipped' },
      { question_id: 'Q2', status: 'skipped' },
    ]);
  });

  test('resume mode skips completed work', async () => {
    const outDir = path.join(RUNS_ROOT, 'resume');
    const config = makeConfig({ outDir, resume: true });
    const questions = makeQuestions(1);
    const runId = 'resume-test';
    const { results } = await getStore();
    results.set(`${runId}:m1:Q1`, {
      runId,
      modelId: 'm1',
      questionId: 'Q1',
      status: 'done',
      judgeParsedJson: JSON.stringify(judgeOutput),
    });

    await runBenchmark({
      config,
      configPath: 'x',
      datasetPath: 'x',
      datasetAbsolutePath: DATASET_ABS,
      questions,
      deps: {
        resolveModel: () => fakeModel,
        resolveJudgeModel: () => fakeModel,
        toolVersion: 'test',
      },
      dryRun: false,
      runIdOverride: runId,
    });

    expect(generateTextMock).not.toHaveBeenCalled();
    expect(judgeMock).not.toHaveBeenCalled();
  });

  test('candidate metrics store normalized usage', async () => {
    const outDir = path.join(RUNS_ROOT, 'usage');
    const config = makeConfig({ outDir });
    const questions = makeQuestions(1);
    const runId = 'usage-test';

    generateTextMock.mockResolvedValue({
      text: 'candidate',
      usage: { prompt_tokens: 3, completion_tokens: 4 } as unknown,
      providerMetadata: {},
    } as unknown as Awaited<ReturnType<typeof generateText>>);

    judgeMock.mockResolvedValue({
      object: judgeOutput,
      raw: {},
      didRepairRetry: false,
    });

    await runBenchmark({
      config,
      configPath: 'x',
      datasetPath: 'x',
      datasetAbsolutePath: DATASET_ABS,
      questions,
      deps: {
        resolveModel: () => fakeModel,
        resolveJudgeModel: () => fakeModel,
        toolVersion: 'test',
      },
      dryRun: false,
      runIdOverride: runId,
    });

    const { results } = await getStore();
    const row = Array.from(results.values()).find(
      (entry) => entry.runId === runId && entry.questionId === 'Q1',
    );

    expect(row?.candidateMetricsJson).toBeTypeOf('string');
    const parsed = JSON.parse(row?.candidateMetricsJson ?? '{}') as {
      usage?: { promptTokens: number; completionTokens: number; totalTokens: number };
    };
    expect(parsed.usage).toEqual({
      promptTokens: 3,
      completionTokens: 4,
      totalTokens: 7,
    });
  });
});
