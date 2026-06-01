import { describe, expect, test } from 'vitest';
import { configSchema } from '../src/core/config/schema';

describe('config schema', () => {
  const baseConfig = {
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
  };

  test('rejects unknown keys', () => {
    const result = configSchema.safeParse({
      ...baseConfig,
      extra: true,
    });

    expect(result.success).toBe(false);
  });

  test('accepts openai-compatible candidate router with no auth', () => {
    const result = configSchema.safeParse({
      ...baseConfig,
      routers: {
        ...baseConfig.routers,
        openaiCompatible: {
          baseUrl: 'http://127.0.0.1:1234/v1',
          apiKeyEnv: null,
          headers: { 'X-Local': 'true' },
          queryParams: { profile: 'local' },
          default: { temperature: 0.2, maxTokens: 800, timeoutMs: 120000 },
        },
      },
      models: [{ id: 'local', router: 'openai-compatible', model: 'local-model' }],
    });

    expect(result.success).toBe(true);
    if (!result.success) return;
    expect(result.data.routers.openaiCompatible).toEqual({
      baseUrl: 'http://127.0.0.1:1234/v1',
      apiKeyEnv: null,
      headers: { 'X-Local': 'true' },
      queryParams: { profile: 'local' },
      default: { temperature: 0.2, maxTokens: 800, timeoutMs: 120000 },
    });
  });

  test('requires openai-compatible router config when a model selects it', () => {
    const result = configSchema.safeParse({
      ...baseConfig,
      models: [{ id: 'local', router: 'openai-compatible', model: 'local-model' }],
    });

    expect(result.success).toBe(false);
  });

  test('rejects unknown openai-compatible router keys', () => {
    const result = configSchema.safeParse({
      ...baseConfig,
      routers: {
        ...baseConfig.routers,
        openaiCompatible: {
          baseUrl: 'http://127.0.0.1:1234/v1',
          apiKeyEnv: null,
          default: { temperature: 0.2, maxTokens: 800, timeoutMs: 120000 },
          extra: true,
        },
      },
      models: [{ id: 'local', router: 'openai-compatible', model: 'local-model' }],
    });

    expect(result.success).toBe(false);
  });
});
