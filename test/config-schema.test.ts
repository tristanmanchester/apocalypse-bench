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

  test('accepts run retry policy overrides', () => {
    const result = configSchema.safeParse({
      ...baseConfig,
      run: {
        ...baseConfig.run,
        retry: {
          maxRetries: 6,
          baseMs: 2000,
          maxMs: 60000,
          maxTotalTimeMs: null,
        },
      },
    });

    expect(result.success).toBe(true);
    if (!result.success) return;
    expect(result.data.run.retry).toEqual({
      maxRetries: 6,
      baseMs: 2000,
      maxMs: 60000,
      maxTotalTimeMs: null,
    });
    expect(result.data.judge.backend).toBe('openrouter');
  });

  test('accepts deterministic question shuffle settings', () => {
    const result = configSchema.safeParse({
      ...baseConfig,
      run: {
        ...baseConfig.run,
        questionOrder: 'shuffle',
        questionSeed: 'smoke-v1',
      },
    });

    expect(result.success).toBe(true);
    if (!result.success) return;
    expect(result.data.run.questionOrder).toBe('shuffle');
    expect(result.data.run.questionSeed).toBe('smoke-v1');
  });

  test('rejects invalid question order', () => {
    const result = configSchema.safeParse({
      ...baseConfig,
      run: {
        ...baseConfig.run,
        questionOrder: 'randomish',
      },
    });

    expect(result.success).toBe(false);
  });

  test('accepts codex judge backend with defaults', () => {
    const result = configSchema.safeParse({
      ...baseConfig,
      run: {
        ...baseConfig.run,
        candidateOnly: true,
      },
      judge: {
        backend: 'codex-cli',
      },
    });

    expect(result.success).toBe(true);
    if (!result.success) return;
    expect(result.data.judge).toMatchObject({
      backend: 'codex-cli',
      model: 'gpt-5.5',
      reasoning: 'low',
      codexBin: 'codex',
      batchSize: 10,
      batchStrategy: 'question-paired',
      concurrency: 1,
      sourceStatus: 'both',
      maxRetries: 1,
    });
  });

  test('accepts codex judge concurrency and per-model candidate concurrency', () => {
    const result = configSchema.safeParse({
      ...baseConfig,
      run: {
        ...baseConfig.run,
        candidateOnly: true,
      },
      judge: {
        backend: 'codex-cli',
        concurrency: 5,
      },
      models: [
        {
          id: 'm1',
          router: 'ollama',
          model: 'llama3.2',
          concurrency: 2,
        },
      ],
    });

    expect(result.success).toBe(true);
    if (!result.success) return;
    expect(result.data.judge.backend).toBe('codex-cli');
    if (result.data.judge.backend !== 'codex-cli') return;
    expect(result.data.judge.concurrency).toBe(5);
    expect(result.data.models[0]?.concurrency).toBe(2);
  });

  test('rejects unknown codex judge keys', () => {
    const result = configSchema.safeParse({
      ...baseConfig,
      run: {
        ...baseConfig.run,
        candidateOnly: true,
      },
      judge: {
        backend: 'codex-cli',
        extra: true,
      },
    });

    expect(result.success).toBe(false);
  });

  test('rejects invalid codex batch strategy', () => {
    const result = configSchema.safeParse({
      ...baseConfig,
      run: {
        ...baseConfig.run,
        candidateOnly: true,
      },
      judge: {
        backend: 'codex-cli',
        batchStrategy: 'not-a-strategy',
      },
    });

    expect(result.success).toBe(false);
  });

  test('rejects invalid concurrency values', () => {
    expect(
      configSchema.safeParse({
        ...baseConfig,
        run: {
          ...baseConfig.run,
          candidateOnly: true,
        },
        judge: {
          backend: 'codex-cli',
          concurrency: 0,
        },
      }).success,
    ).toBe(false);

    expect(
      configSchema.safeParse({
        ...baseConfig,
        models: [{ id: 'm1', router: 'ollama', model: 'llama3.2', concurrency: 0 }],
      }).success,
    ).toBe(false);
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
