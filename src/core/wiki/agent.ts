import { z } from 'zod';
import type { ToolSet } from 'ai';

import type { CandidateMode, WikiConfig } from '../config/schema';
import type { WikiClient } from './client';
import type { RetrievalTrace } from './rag';
import type { WikiReadResponse, WikiSearchHit, WikiSearchMode, WikiSearchResponse } from './types';

type WikiAgentClient = Pick<
  WikiClient,
  'search' | 'semanticSearch' | 'hybridSearch' | 'literalSearch' | 'read'
>;

export type WikiAgentContext = {
  tools: ToolSet;
  prompt: string;
  trace: RetrievalTrace;
};

export function createWikiAgentContext(params: {
  basePrompt: string;
  mode: CandidateMode;
  wiki: WikiConfig;
  client: WikiAgentClient;
}): WikiAgentContext {
  const { basePrompt, mode, wiki, client } = params;
  const trace: RetrievalTrace = {
    mode,
    queries: [],
    searches: [],
    reads: [],
    contextChars: 0,
    truncated: false,
  };
  const searchMode = searchModeForCandidateMode(mode);

  const tools: ToolSet = {
    wiki_search: {
      description:
        'Search the local offline Wikipedia index. Use specific nouns and article names when possible.',
      inputSchema: z.object({
        query: z.string().min(1),
        topK: z.number().int().positive().max(wiki.limits.searchTopK).optional(),
      }),
      execute: async ({ query, topK }) => {
        const search = await runSearch({
          client,
          mode,
          query,
          topK: topK ?? wiki.limits.searchTopK,
        });
        trace.queries.push(query);
        trace.searches.push(traceSearch(search));
        return {
          mode: search.mode,
          query: search.query,
          hits: search.hits.map(formatHitForTool),
        };
      },
    },
    wiki_read: {
      description:
        'Read a Wikipedia chunk returned by wiki_search. Use this before relying on a search snippet.',
      inputSchema: z.object({
        chunkId: z.string().min(1),
        maxChars: z.number().int().positive().max(wiki.limits.readMaxChars).optional(),
      }),
      execute: async ({ chunkId, maxChars }) => {
        const remaining = Math.max(0, wiki.limits.contextMaxChars - trace.contextChars);
        if (remaining <= 0) {
          trace.truncated = true;
          return {
            error: 'wiki context character budget exhausted',
          };
        }
        const read = await client.read({
          chunkId,
          maxChars: Math.min(maxChars ?? wiki.limits.readMaxChars, remaining),
        });
        trace.contextChars += read.text.length;
        trace.truncated = trace.truncated || read.truncated;
        trace.reads.push(traceRead(read));
        return formatReadForTool(read);
      },
    },
  };

  const prompt = [
    basePrompt,
    '',
    `You have access to offline Wikipedia tools: wiki_search uses ${searchMode} retrieval, and wiki_read opens a returned chunk. Use the tools when factual background would materially improve the answer. Cite useful article titles inline, but keep the final answer practical and conservative for survival use.`,
  ].join('\n');

  return { tools, prompt, trace };
}

function searchModeForCandidateMode(mode: CandidateMode): WikiSearchMode {
  switch (mode) {
    case 'agent-bm25':
      return 'bm25';
    case 'agent-dense':
      return 'dense';
    case 'agent-hybrid':
      return 'hybrid';
    case 'agent-rg':
    case 'agent-literal':
      return 'literal';
    default:
      throw new Error(`candidate mode does not use wiki agent tools: ${mode}`);
  }
}

async function runSearch(params: {
  client: WikiAgentClient;
  mode: CandidateMode;
  query: string;
  topK: number;
}): Promise<WikiSearchResponse> {
  const request = { query: params.query, topK: params.topK };
  switch (params.mode) {
    case 'agent-bm25':
      return params.client.search(request);
    case 'agent-dense':
      return params.client.semanticSearch(request);
    case 'agent-hybrid':
      return params.client.hybridSearch(request);
    case 'agent-rg':
    case 'agent-literal':
      return params.client.literalSearch(request);
    default:
      throw new Error(`candidate mode does not use wiki agent tools: ${params.mode}`);
  }
}

function formatHitForTool(hit: WikiSearchHit) {
  return {
    articleId: hit.pointer.articleId,
    chunkId: hit.pointer.chunkId,
    title: hit.pointer.title,
    headingPath: hit.pointer.headingPath,
    url: hit.pointer.url,
    score: hit.score,
    snippet: hit.snippet,
  };
}

function formatReadForTool(read: WikiReadResponse) {
  return {
    articleId: read.pointer.articleId,
    chunkId: read.pointer.chunkId,
    title: read.pointer.title,
    headingPath: read.pointer.headingPath,
    url: read.pointer.url,
    text: read.text,
    truncated: read.truncated,
  };
}

function traceSearch(search: WikiSearchResponse): RetrievalTrace['searches'][number] {
  return {
    mode: search.mode,
    query: search.query,
    latencyMs: search.latencyMs,
    hits: search.hits.map((hit) => ({
      articleId: hit.pointer.articleId,
      chunkId: hit.pointer.chunkId,
      title: hit.pointer.title,
      headingPath: hit.pointer.headingPath,
      url: hit.pointer.url,
      score: hit.score,
      snippet: hit.snippet,
    })),
  };
}

function traceRead(read: WikiReadResponse): RetrievalTrace['reads'][number] {
  return {
    articleId: read.pointer.articleId,
    chunkId: read.pointer.chunkId,
    title: read.pointer.title,
    headingPath: read.pointer.headingPath,
    url: read.pointer.url,
    chars: read.text.length,
    truncated: read.truncated,
    latencyMs: read.latencyMs,
  };
}
