import { describe, expect, test } from 'vitest';

import {
  expectedCandidateCountForRunAndJudge,
  selectedModelIdsForRunAndJudge,
} from '../src/cli/index';
import type { ApocbenchConfig } from '../src/core/config/schema';

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
});
