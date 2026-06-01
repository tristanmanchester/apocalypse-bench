import { beforeEach, describe, expect, test, vi } from 'vitest';

const createOpenAICompatibleMock = vi.hoisted(() => vi.fn(() => vi.fn()));

vi.mock('@ai-sdk/openai-compatible', () => ({
  createOpenAICompatible: createOpenAICompatibleMock,
}));

describe('createOpenAICompatibleClient', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  test('creates a no-auth provider for local endpoints', async () => {
    const { createOpenAICompatibleClient } = await import(
      '../src/adapters/openaiCompatible/client'
    );

    createOpenAICompatibleClient({
      baseUrl: 'http://127.0.0.1:1234/v1',
    });

    expect(createOpenAICompatibleMock).toHaveBeenCalledWith({
      name: 'openai-compatible',
      baseURL: 'http://127.0.0.1:1234/v1',
      includeUsage: true,
    });
  });

  test('passes auth, headers, and query params through to the provider', async () => {
    const { createOpenAICompatibleClient } = await import(
      '../src/adapters/openaiCompatible/client'
    );

    createOpenAICompatibleClient({
      baseUrl: 'http://127.0.0.1:1234/v1/',
      apiKey: 'local-key',
      headers: { 'X-Local': 'true' },
      queryParams: { profile: 'bench' },
    });

    expect(createOpenAICompatibleMock).toHaveBeenCalledWith({
      name: 'openai-compatible',
      baseURL: 'http://127.0.0.1:1234/v1/',
      apiKey: 'local-key',
      headers: { 'X-Local': 'true' },
      queryParams: { profile: 'bench' },
      includeUsage: true,
    });
  });
});
