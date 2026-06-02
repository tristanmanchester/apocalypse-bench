import { describe, expect, test } from 'vitest';
import type { WikiConfig } from '../src/core/config/schema';
import { createWikiAgentContext } from '../src/core/wiki/agent';

const wiki: WikiConfig = {
  service: { baseUrl: 'http://127.0.0.1:8765' },
  corpus: { manifestId: 'corpus' },
  index: { manifestId: 'index' },
  limits: {
    searchTopK: 3,
    readMaxChars: 500,
    contextMaxChars: 500,
    maxTurns: 4,
  },
};

describe('createWikiAgentContext', () => {
  test('exposes search and read tools that append retrieval trace entries', async () => {
    const ctx = createWikiAgentContext({
      basePrompt: 'answer the question',
      mode: 'agent-bm25',
      wiki,
      client: {
        search: async ({ query }) => ({
          mode: 'bm25',
          query,
          hits: [
            {
              mode: 'bm25',
              pointer: {
                articleId: 'a1',
                chunkId: 'c1',
                title: 'Water purification',
              },
              snippet: 'Boiling water can inactivate pathogens.',
              score: 12,
            },
          ],
        }),
        semanticSearch: async () => ({ mode: 'dense', query: '', hits: [] }),
        hybridSearch: async () => ({ mode: 'hybrid', query: '', hits: [] }),
        literalSearch: async () => ({ mode: 'literal', query: '', hits: [] }),
        read: async () => ({
          pointer: {
            articleId: 'a1',
            chunkId: 'c1',
            title: 'Water purification',
          },
          text: 'Boiling water can inactivate many pathogens.',
          truncated: false,
        }),
      },
    });

    await ctx.tools.wiki_search.execute?.(
      { query: 'water purification', topK: 2 },
      { toolCallId: 'search-1', messages: [] },
    );
    await ctx.tools.wiki_read.execute?.(
      { chunkId: 'c1', maxChars: 200 },
      { toolCallId: 'read-1', messages: [] },
    );

    expect(ctx.prompt).toContain('wiki_search');
    expect(ctx.trace.searches[0]?.hits[0]?.title).toBe('Water purification');
    expect(ctx.trace.reads[0]?.chars).toBeGreaterThan(0);
  });
});
