import { describe, expect, test, vi } from 'vitest';
import {
  WikiClient,
  WikiClientError,
  WikiReadinessError,
  checkWikiReadiness,
} from '../src/core/wiki/client';
import type { WikiConfig } from '../src/core/config/schema';

function jsonResponse(body: unknown, init: ResponseInit = {}): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'content-type': 'application/json' },
    ...init,
  });
}

describe('WikiClient', () => {
  test('calls health with GET and normalizes trailing base URL slashes', async () => {
    const fetchMock = vi.fn(async () =>
      jsonResponse({
        ok: true,
        corpus: { manifestId: 'corpus-v1' },
        index: { manifestId: 'index-v1' },
        capabilities: ['bm25', 'hybrid'],
      }),
    );
    const client = new WikiClient({
      baseUrl: 'http://127.0.0.1:8765/',
      fetch: fetchMock as typeof fetch,
    });

    const health = await client.health();

    expect(health.ok).toBe(true);
    expect(fetchMock).toHaveBeenCalledWith(
      'http://127.0.0.1:8765/health',
      expect.objectContaining({ method: 'GET' }),
    );
  });

  test('posts JSON to search and read endpoints', async () => {
    const fetchMock = vi.fn(async (url: string) => {
      if (url.endsWith('/hybrid_search')) {
        return jsonResponse({
          mode: 'hybrid',
          query: 'water purification',
          hits: [],
          latencyMs: 12,
        });
      }
      return jsonResponse({
        pointer: {
          articleId: 'a1',
          chunkId: 'c1',
          title: 'Water purification',
        },
        text: 'Boiling water...',
        truncated: false,
      });
    });
    const client = new WikiClient({
      baseUrl: 'http://127.0.0.1:8765',
      fetch: fetchMock as typeof fetch,
    });

    await expect(client.hybridSearch({ query: 'water purification', topK: 3 })).resolves.toEqual({
      mode: 'hybrid',
      query: 'water purification',
      hits: [],
      latencyMs: 12,
    });
    await expect(client.read({ chunkId: 'c1', maxChars: 1000 })).resolves.toEqual({
      pointer: {
        articleId: 'a1',
        chunkId: 'c1',
        title: 'Water purification',
      },
      text: 'Boiling water...',
      truncated: false,
    });

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      'http://127.0.0.1:8765/hybrid_search',
      expect.objectContaining({
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ query: 'water purification', limit: 3 }),
      }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      'http://127.0.0.1:8765/read',
      expect.objectContaining({
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ chunkId: 'c1', maxChars: 1000 }),
      }),
    );
  });

  test('forwards scoped literal search pointers', async () => {
    const fetchMock = vi.fn(async () =>
      jsonResponse({
        mode: 'literal',
        query: 'warm gradually',
        hits: [],
        latencyMs: 1,
      }),
    );
    const client = new WikiClient({
      baseUrl: 'http://127.0.0.1:8765',
      fetch: fetchMock as typeof fetch,
    });

    await client.literalSearch({
      query: 'warm gradually',
      topK: 2,
      articleId: 'hypothermia',
      chunkId: 'hypothermia:treatment',
    });

    expect(fetchMock).toHaveBeenCalledWith(
      'http://127.0.0.1:8765/literal_search',
      expect.objectContaining({
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          query: 'warm gradually',
          limit: 2,
          articleId: 'hypothermia',
          chunkId: 'hypothermia:treatment',
        }),
      }),
    );
  });

  test('preserves fused hit source metadata from the Rust service shape', async () => {
    const fetchMock = vi.fn(async () =>
      jsonResponse({
        mode: 'hybrid',
        query: 'water purification',
        hits: [
          {
            mode: 'hybrid',
            score: 0.032,
            bm25_score: 12.5,
            dense_score: 0.88,
            article_id: 'a1',
            chunk_id: 'c1',
            title: 'Water purification',
            heading_path: ['Boiling'],
            url: 'https://en.wikipedia.org/wiki/Water_purification',
            snippet: 'Boiling can disinfect water.',
            sources: ['bm25', 'dense'],
          },
        ],
        latencyMs: 8,
      }),
    );
    const client = new WikiClient({
      baseUrl: 'http://127.0.0.1:8765',
      fetch: fetchMock as typeof fetch,
    });

    await expect(client.hybridSearch({ query: 'water purification', topK: 2 })).resolves.toEqual({
      mode: 'hybrid',
      query: 'water purification',
      hits: [
        {
          pointer: {
            articleId: 'a1',
            chunkId: 'c1',
            title: 'Water purification',
            headingPath: ['Boiling'],
            url: 'https://en.wikipedia.org/wiki/Water_purification',
          },
          mode: 'hybrid',
          score: 0.032,
          bm25Score: 12.5,
          denseScore: 0.88,
          sources: ['bm25', 'dense'],
          snippet: 'Boiling can disinfect water.',
        },
      ],
      latencyMs: 8,
    });
  });

  test('turns non-2xx service responses into WikiClientError', async () => {
    const client = new WikiClient({
      baseUrl: 'http://127.0.0.1:8765',
      fetch: vi.fn(async () => new Response('missing index', { status: 503 })) as typeof fetch,
    });

    await expect(client.search({ query: 'radio' })).rejects.toMatchObject({
      name: 'WikiClientError',
      status: 503,
      message: expect.stringContaining('missing index'),
    });
  });
});

describe('checkWikiReadiness', () => {
  const config: WikiConfig = {
    enabled: true,
    service: { baseUrl: 'http://127.0.0.1:8765' },
    corpus: { manifestId: 'corpus-v1' },
    index: { manifestId: 'index-v1' },
    limits: {
      searchTopK: 5,
      readMaxChars: 4000,
      contextMaxChars: 12000,
    },
  };

  test('returns health and capabilities for a ready wiki service', async () => {
    const readiness = await checkWikiReadiness(
      {
        health: async () => ({
          ok: true,
          corpus: { manifestId: 'corpus-v1' },
          index: { manifestId: 'index-v1' },
          capabilities: ['bm25', 'hybrid'],
        }),
      },
      config,
    );

    expect(readiness.health.ok).toBe(true);
    expect(readiness.capabilities.has('bm25')).toBe(true);
    expect(readiness.capabilities.has('hybrid')).toBe(true);
  });

  test('fails when the service reports the wrong corpus manifest', async () => {
    await expect(
      checkWikiReadiness(
        {
          health: async () => ({
            ok: true,
            corpus: { manifestId: 'other-corpus' },
            index: { manifestId: 'index-v1' },
            capabilities: ['bm25'],
          }),
        },
        config,
      ),
    ).rejects.toBeInstanceOf(WikiReadinessError);
  });

  test('keeps transport errors distinct from readiness errors', async () => {
    await expect(
      checkWikiReadiness(
        {
          health: async () => {
            throw new WikiClientError('wiki service request failed');
          },
        },
        config,
      ),
    ).rejects.toBeInstanceOf(WikiClientError);
  });
});
