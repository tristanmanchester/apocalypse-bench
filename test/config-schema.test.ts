import { describe, expect, test } from 'vitest';
import { configSchema } from '../src/core/config/schema';

describe('config schema', () => {
  test('rejects unknown keys', () => {
    const result = configSchema.safeParse({
      run: {
        name: 'x',
        datasetPaths: ['./data/question_bank'],
        outDir: './runs',
        resume: true,
        concurrency: { candidate: 1, judge: 1 },
      },
      judge: {
        router: 'openrouter',
        model: 'google/gemini-3-flash-preview',
        maxTokens: 1000,
        structured: true,
      },
      routers: {
        ollama: {
          baseUrl: 'http://localhost:11434/v1',
          apiKeyEnv: null,
          default: { temperature: 0.2, maxTokens: 800, timeoutMs: 120000 },
        },
        openrouter: {
          baseUrl: 'https://openrouter.ai/api/v1',
          apiKeyEnv: 'OPENROUTER_API_KEY',
          default: { temperature: 0.2, maxTokens: 800, timeoutMs: 120000 },
        },
      },
      models: [{ id: 'm1', router: 'ollama', model: 'llama3.2' }],
      extra: true,
    });

    expect(result.success).toBe(false);
  });
});
