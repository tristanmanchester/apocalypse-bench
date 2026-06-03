import { describe, expect, test } from 'vitest';
import { configSchema } from '../src/core/config/schema';

describe('wiki config schema', () => {
  const baseConfig = {
    run: {
      name: 'wiki-config',
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

  const wiki = {
    enabled: true,
    service: {
      baseUrl: 'http://127.0.0.1:8765',
      timeoutMs: 5000,
    },
    corpus: {
      manifestId: 'wiki-mini-corpus-v1',
      manifestPath: './data/wiki/manifest.json',
    },
    index: {
      manifestId: 'wiki-mini-index-v1',
      manifestPath: './data/wiki/index-manifest.json',
    },
    limits: {
      searchTopK: 5,
      readMaxChars: 4000,
      contextMaxChars: 12000,
      maxToolCalls: 8,
      maxTurns: 6,
    },
  };

  test('defaults existing model entries to direct mode without wiki config', () => {
    const result = configSchema.safeParse(baseConfig);

    expect(result.success).toBe(true);
    if (!result.success) return;
    expect(result.data.models[0]?.candidateMode).toBe('direct');
    expect(result.data.wiki).toBeUndefined();
  });

  test('rejects wiki candidate modes without wiki config', () => {
    const result = configSchema.safeParse({
      ...baseConfig,
      models: [
        {
          id: 'm1-rag',
          router: 'ollama',
          model: 'llama3.2',
          candidateMode: 'rag-hybrid',
        },
      ],
    });

    expect(result.success).toBe(false);
    if (result.success) return;
    expect(result.error.issues[0]?.path).toEqual(['models', 0, 'candidateMode']);
  });

  test('accepts and preserves wiki config for wiki modes', () => {
    const result = configSchema.safeParse({
      ...baseConfig,
      wiki,
      models: [
        {
          id: 'm1-direct',
          router: 'ollama',
          model: 'llama3.2',
          candidateMode: 'direct',
        },
        {
          id: 'm1-agent',
          router: 'ollama',
          model: 'llama3.2',
          candidateMode: 'agent-bm25-research',
        },
      ],
    });

    expect(result.success).toBe(true);
    if (!result.success) return;
    expect(result.data.wiki).toEqual(wiki);
    expect(result.data.models.map((model) => model.candidateMode)).toEqual([
      'direct',
      'agent-bm25-research',
    ]);
  });

  test('rejects wiki candidate modes when wiki is explicitly disabled', () => {
    const result = configSchema.safeParse({
      ...baseConfig,
      wiki: { ...wiki, enabled: false },
      models: [
        {
          id: 'm1-rag',
          router: 'ollama',
          model: 'llama3.2',
          candidateMode: 'rag-bm25',
        },
      ],
    });

    expect(result.success).toBe(false);
    if (result.success) return;
    expect(result.error.issues[0]?.path).toEqual(['wiki', 'enabled']);
  });
});
