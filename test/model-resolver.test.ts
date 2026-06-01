import { beforeEach, describe, expect, test, vi } from 'vitest';
import type { LanguageModel } from 'ai';

import type { ApocbenchConfig } from '../src/core/config/schema';

const fakeModel = { provider: 'test', modelId: 'model' } as LanguageModel;
const openrouterModelFactory = vi.hoisted(() => vi.fn(() => fakeModel));
const ollamaModelFactory = vi.hoisted(() => vi.fn(() => fakeModel));
const openaiCompatibleModelFactory = vi.hoisted(() => vi.fn(() => fakeModel));
const createOpenRouterClientMock = vi.hoisted(() => vi.fn(() => openrouterModelFactory));
const createOllamaClientMock = vi.hoisted(() => vi.fn(() => ollamaModelFactory));
const createOpenAICompatibleClientMock = vi.hoisted(() =>
  vi.fn(() => openaiCompatibleModelFactory),
);

vi.mock('../src/adapters/openrouter/client', () => ({
  createOpenRouterClient: createOpenRouterClientMock,
}));

vi.mock('../src/adapters/ollama/client', () => ({
  createOllamaClient: createOllamaClientMock,
}));

vi.mock('../src/adapters/openaiCompatible/client', () => ({
  createOpenAICompatibleClient: createOpenAICompatibleClientMock,
}));

function makeConfig(): ApocbenchConfig {
  return {
    run: {
      name: 'test',
      datasetPaths: ['./data/question_bank'],
      outDir: './runs',
      resume: true,
      concurrency: { candidate: 1, judge: 1 },
    },
    judge: {
      router: 'openrouter',
      model: 'judge-model',
      maxTokens: 1000,
      structured: true,
    },
    routers: {
      ollama: {
        baseUrl: 'http://localhost:11434/api',
        apiKeyEnv: null,
        default: { temperature: 0.2, maxTokens: 50, timeoutMs: 1000 },
      },
      openrouter: {
        baseUrl: 'https://openrouter.ai/api/v1',
        apiKeyEnv: 'OPENROUTER_API_KEY',
        headers: { 'X-Title': 'apocbench' },
        default: { temperature: 0.2, maxTokens: 50, timeoutMs: 1000 },
      },
      openaiCompatible: {
        baseUrl: 'http://127.0.0.1:1234/v1',
        apiKeyEnv: null,
        headers: { 'X-Local': 'true' },
        queryParams: { profile: 'bench' },
        default: { temperature: 0.2, maxTokens: 50, timeoutMs: 1000 },
      },
    },
    models: [{ id: 'm1', router: 'openai-compatible', model: 'local-model' }],
  };
}

describe('model resolver', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  test('resolves openai-compatible candidates without OpenRouter env vars', async () => {
    const { resolveCandidateModel } = await import('../src/cli/modelResolver');
    const config = makeConfig();

    const model = resolveCandidateModel(config, config.models[0]!, {});

    expect(model).toBe(fakeModel);
    expect(createOpenAICompatibleClientMock).toHaveBeenCalledWith({
      baseUrl: 'http://127.0.0.1:1234/v1',
      apiKey: undefined,
      headers: { 'X-Local': 'true' },
      queryParams: { profile: 'bench' },
    });
    expect(openaiCompatibleModelFactory).toHaveBeenCalledWith('local-model');
    expect(createOpenRouterClientMock).not.toHaveBeenCalled();
  });

  test('requires configured openai-compatible api key env var', async () => {
    const { resolveCandidateModel } = await import('../src/cli/modelResolver');
    const config = makeConfig();
    config.routers.openaiCompatible!.apiKeyEnv = 'LOCAL_OPENAI_API_KEY';

    expect(() => resolveCandidateModel(config, config.models[0]!, {})).toThrow(
      'missing env var: LOCAL_OPENAI_API_KEY',
    );

    resolveCandidateModel(config, config.models[0]!, {
      LOCAL_OPENAI_API_KEY: 'local-key',
    });

    expect(createOpenAICompatibleClientMock).toHaveBeenLastCalledWith({
      baseUrl: 'http://127.0.0.1:1234/v1',
      apiKey: 'local-key',
      headers: { 'X-Local': 'true' },
      queryParams: { profile: 'bench' },
    });
  });

  test('preserves OpenRouter and Ollama resolution behavior', async () => {
    const { resolveCandidateModel, resolveJudgeModel } = await import('../src/cli/modelResolver');
    const config = makeConfig();
    const env = { OPENROUTER_API_KEY: 'or-key' };

    resolveCandidateModel(
      config,
      { id: 'or', router: 'openrouter', model: 'openrouter-model' },
      env,
    );
    resolveCandidateModel(config, { id: 'ollama', router: 'ollama', model: 'llama3.2' }, {});
    resolveJudgeModel(config, env);

    expect(createOpenRouterClientMock).toHaveBeenCalledWith({
      apiKey: 'or-key',
      baseUrl: 'https://openrouter.ai/api/v1',
      headers: { 'X-Title': 'apocbench' },
    });
    expect(openrouterModelFactory).toHaveBeenCalledWith('openrouter-model', {
      usage: { include: true },
    });
    expect(openrouterModelFactory).toHaveBeenCalledWith('judge-model', {
      usage: { include: true },
    });
    expect(createOllamaClientMock).toHaveBeenCalledWith({
      baseUrl: 'http://localhost:11434/api',
    });
    expect(ollamaModelFactory).toHaveBeenCalledWith('llama3.2');
  });
});
