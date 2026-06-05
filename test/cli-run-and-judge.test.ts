import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

import { describe, expect, test } from 'vitest';

import {
  countCompleteCandidateRows,
  expectedCandidateCountForRunAndJudge,
  selectedModelIdsForRunAndJudge,
} from '../src/cli/index';
import type { ApocbenchConfig } from '../src/core/config/schema';
import { closeDb } from '../src/storage/sqlite/db';
import { openAndMigrate } from '../src/storage/sqlite/migrate';
import { insertQuestions } from '../src/storage/sqlite/questions';
import { upsertResult } from '../src/storage/sqlite/results';
import { insertRun } from '../src/storage/sqlite/runs';

function configWithModels(modelIds: string[]): ApocbenchConfig {
  return {
    run: {
      name: 'test',
      datasetPaths: ['./data/question_bank'],
      outDir: './runs',
      resume: true,
      candidateOnly: true,
      concurrency: { candidate: 1, judge: 1 },
    },
    candidate: { maxTokens: 1000 },
    judge: { backend: 'codex-cli' },
    routers: {
      ollama: {
        baseUrl: 'http://localhost:11434/api',
        apiKeyEnv: null,
        default: { temperature: 0.2, maxTokens: 1000, timeoutMs: 120000 },
      },
      openrouter: {
        baseUrl: 'https://openrouter.ai/api/v1',
        apiKeyEnv: 'OPENROUTER_API_KEY',
        default: { temperature: 0.2, maxTokens: 1000, timeoutMs: 120000 },
      },
    },
    models: modelIds.map((id) => ({
      id,
      router: 'ollama',
      model: id,
    })),
  };
}

describe('run-and-judge candidate counting', () => {
  test('counts every configured model when no model filter is supplied', () => {
    const config = configWithModels(['direct', 'bm25', 'rerank']);

    expect(expectedCandidateCountForRunAndJudge({ config, questionCount: 7 })).toBe(21);
  });

  test('counts only selected model ids for subset run-and-judge runs', () => {
    const config = configWithModels(['direct', 'bm25', 'rerank']);

    expect(
      selectedModelIdsForRunAndJudge(config, ['rerank', 'missing', 'rerank']),
    ).toEqual(['rerank']);
    expect(
      expectedCandidateCountForRunAndJudge({
        config,
        questionCount: 7,
        requestedModelIds: ['rerank', 'missing', 'rerank'],
      }),
    ).toBe(7);
  });

  test('candidate completion count is limited to selected models and questions', () => {
    const tempDir = mkdtempSync(path.join(tmpdir(), 'apocbench-run-and-judge-'));
    const dbPath = path.join(tempDir, 'apocbench.sqlite');

    closeDb();
    const db = openAndMigrate(dbPath);

    try {
      insertRun(db, {
        run_id: 'candidate-run',
        created_at: new Date().toISOString(),
        tool_version: 'test',
        config_json: '{}',
        dataset_path: 'test',
        dataset_sha256: 'test-sha',
        prompt_template_hash: 'test-prompt',
        status: 'running',
      });
      insertQuestions(
        db,
        'candidate-run',
        ['Q1', 'Q2', 'Q3'].map((id) => ({
          id,
          category: 'Test',
          difficulty: 'Easy',
          scenario: [`Scenario ${id}`],
          prompt: `Prompt ${id}`,
          rubric: [{ id: 'r1', text: 'Answer', weight: 1, maxScore: 1 }],
          auto_fail: ['Refusal to answer.'],
          reference_facts: ['Fact.'],
        })),
      );

      for (const questionId of ['Q1', 'Q2', 'Q3']) {
        for (const modelId of ['direct', 'bm25']) {
          upsertResult(db, {
            runId: 'candidate-run',
            questionId,
            modelId,
            status: 'candidate_done',
            candidateCompletion: `answer ${questionId} ${modelId}`,
          });
        }
      }

      expect(
        countCompleteCandidateRows(db, 'candidate-run', ['bm25'], ['Q1', 'Q2']),
      ).toBe(2);
      expect(countCompleteCandidateRows(db, 'candidate-run', [], ['Q1', 'Q2'])).toBe(0);
      expect(countCompleteCandidateRows(db, 'candidate-run', ['bm25'], [])).toBe(0);
    } finally {
      closeDb();
      rmSync(tempDir, { recursive: true, force: true });
    }
  });
});
