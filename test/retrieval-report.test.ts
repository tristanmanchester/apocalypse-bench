import { describe, expect, test } from 'vitest';
import { renderHtmlReport } from '../src/reports/html/renderHtml';
import { renderByModelMd } from '../src/reports/markdown';

describe('retrieval reporting', () => {
  test('renders model-level retrieval summaries and per-question traces', () => {
    const html = renderHtmlReport({
      runId: 'wiki-run',
      summaryJson: {
        createdAt: '2026-06-02T00:00:00.000Z',
        models: [
          {
            modelId: 'model-rag-bm25',
            overallScore: 1,
            overallScoreMean: 1,
            completed: 1,
            failures: 0,
            skipped: 0,
            autoFailRate: 0,
            latencyMs: { medianMs: 100, p90Ms: 100 },
            retrieval: {
              traceCount: 1,
              modes: { 'rag-bm25': 1 },
              toolCallCount: 2,
              searchCount: 1,
              readCount: 2,
              uniqueSourceTitles: ['Water purification'],
              latencyMs: { medianMs: 12, p90Ms: 20 },
            },
          },
        ],
      },
      results: [
        {
          model_id: 'model-rag-bm25',
          question_id: 'WATER-001',
          status: 'done',
          score_overall: 1,
          auto_fail: 0,
          category: 'water',
          difficulty: 'Hard',
          retrieval_trace_json: JSON.stringify({
            mode: 'rag-bm25',
            toolCallCount: 2,
            toolCalls: [
              {
                index: 1,
                toolCallId: 'search-1',
                toolName: 'wiki_search',
                arguments: { query: 'water' },
                status: 'ok',
              },
              {
                index: 2,
                toolCallId: 'read-1',
                toolName: 'wiki_read',
                arguments: { chunkId: 'water:lead' },
                status: 'ok',
              },
            ],
            searches: [{ mode: 'bm25', query: 'water', hits: [], latencyMs: 12 }],
            reads: [{ title: 'Water purification', chars: 100, truncated: false }],
          }),
          candidate_completion: 'Boil water.',
        },
      ],
    });

    expect(html).toContain('wiki reads');
    expect(html).toContain('Wiki retrieval summary');
    expect(html).toContain('toolCallCount');
    expect(html).toContain('rag-bm25: 1');
    expect(html).toContain('Water purification');
  });

  test('renders wiki retrieval traces in markdown exports', () => {
    const md = renderByModelMd({
      model: 'model-rag-bm25',
      cases: [
        {
          caseId: 'WATER-001',
          category: 'water',
          difficulty: 'Hard',
          scenario: ['dirty floodwater'],
          prompt: 'How should I make it safer?',
          rubric: [],
          autoFail: [],
          status: 'done',
          answer: 'Boil it when appropriate.',
          retrievalTrace: {
            mode: 'rag-bm25',
            toolCallCount: 2,
            toolCalls: [
              {
                index: 1,
                toolCallId: 'search-1',
                toolName: 'wiki_search',
                arguments: { query: 'water purification' },
                status: 'ok',
              },
            ],
            searches: [{ mode: 'bm25', query: 'water purification', hits: [] }],
            reads: [{ title: 'Water purification', chars: 100, truncated: false }],
          },
        },
      ],
    });

    expect(md).toContain('#### Wiki retrieval');
    expect(md).toContain('- mode: rag-bm25');
    expect(md).toContain('- tool_calls: 2');
    expect(md).toContain('- sources_read: 1');
    expect(md).toContain('Water purification');
  });
});
