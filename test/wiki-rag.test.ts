import { describe, expect, test } from 'vitest';
import type { DatasetLine } from '../src/core/dataset/schema';
import { buildWikiGroundedCandidatePrompt } from '../src/core/wiki/rag';
import type { WikiConfig } from '../src/core/config/schema';

const wiki: WikiConfig = {
  service: { baseUrl: 'http://127.0.0.1:8765' },
  corpus: { manifestId: 'corpus' },
  index: { manifestId: 'index' },
  limits: {
    searchTopK: 1,
    readMaxChars: 500,
    contextMaxChars: 500,
  },
};

const question: DatasetLine = {
  id: 'Q1',
  category: 'water',
  difficulty: 'Medium',
  scenario: ['flooded well'],
  prompt: 'How do I make water safer to drink?',
  rubric: [{ id: 'r1', text: 'safe water advice', weight: 1, maxScore: 1 }],
  auto_fail: ['unsafe water advice'],
};

describe('buildWikiGroundedCandidatePrompt', () => {
  test('retrieves bounded context and records trace metadata', async () => {
    const result = await buildWikiGroundedCandidatePrompt({
      question,
      basePrompt: 'base prompt',
      mode: 'rag-bm25',
      wiki,
      client: {
        search: async () => ({
          mode: 'bm25',
          query: 'water',
          hits: [
            {
              mode: 'bm25',
              pointer: {
                articleId: 'a1',
                chunkId: 'c1',
                title: 'Water purification',
                headingPath: ['Boiling'],
                url: 'https://example.test',
              },
              snippet: 'Boiling can disinfect water.',
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
            headingPath: ['Boiling'],
            url: 'https://example.test',
          },
          text: 'Boiling can disinfect water but does not remove chemical contaminants.',
          truncated: false,
        }),
      },
    });

    expect(result.prompt).toContain('base prompt');
    expect(result.prompt).toContain('Water purification');
    expect(result.trace.mode).toBe('rag-bm25');
    expect(result.trace.reads[0]?.title).toBe('Water purification');
  });
});
