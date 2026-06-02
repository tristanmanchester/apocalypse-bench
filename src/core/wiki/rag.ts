import type { DatasetLine } from '../dataset/schema';
import type { CandidateMode, WikiConfig } from '../config/schema';
import type { WikiClient } from './client';
import type {
  WikiReadResponse,
  WikiSearchHit,
  WikiSearchMode,
  WikiSearchResponse,
} from './types';

export type RetrievalTrace = {
  mode: CandidateMode;
  queries: string[];
  searches: Array<{
    mode: WikiSearchMode;
    query: string;
    hits: Array<{
      articleId: string;
      chunkId?: string;
      title: string;
      headingPath?: string[];
      url?: string;
      score?: number;
      snippet: string;
    }>;
    latencyMs?: number;
  }>;
  reads: Array<{
    articleId: string;
    chunkId?: string;
    title: string;
    headingPath?: string[];
    url?: string;
    chars: number;
    truncated: boolean;
    latencyMs?: number;
  }>;
  contextChars: number;
  truncated: boolean;
};

export type WikiContextResult = {
  prompt: string;
  trace: RetrievalTrace;
};

type WikiRetriever = Pick<
  WikiClient,
  'search' | 'semanticSearch' | 'hybridSearch' | 'literalSearch' | 'read'
>;

export async function buildWikiGroundedCandidatePrompt(params: {
  question: DatasetLine;
  basePrompt: string;
  mode: CandidateMode;
  wiki: WikiConfig;
  client: WikiRetriever;
}): Promise<WikiContextResult> {
  const { question, basePrompt, mode, wiki, client } = params;
  const query = buildRetrievalQuery(question);
  const search = await runSearch({ client, mode, query, topK: wiki.limits.searchTopK });
  const reads: WikiReadResponse[] = [];
  let contextChars = 0;
  let truncated = false;
  const contextParts: string[] = [];

  for (const hit of search.hits) {
    const chunkId = hit.pointer.chunkId;
    if (!chunkId) continue;
    if (contextChars >= wiki.limits.contextMaxChars) {
      truncated = true;
      break;
    }
    const remaining = wiki.limits.contextMaxChars - contextChars;
    const read = await client.read({
      chunkId,
      maxChars: Math.min(wiki.limits.readMaxChars, remaining),
    });
    reads.push(read);
    contextParts.push(formatSource(read));
    contextChars += read.text.length;
    if (read.truncated) truncated = true;
  }

  const prompt = [
    basePrompt,
    '',
    'Offline Wikipedia context is available below. Use it when it is relevant, but do not overtrust it for safety-critical procedures. If context is irrelevant or incomplete, say so and rely on conservative safe fallback reasoning.',
    '',
    contextParts.length > 0 ? contextParts.join('\n\n') : 'No relevant Wikipedia context was found.',
  ].join('\n');

  return {
    prompt,
    trace: {
      mode,
      queries: [query],
      searches: [traceSearch(search)],
      reads: reads.map(traceRead),
      contextChars,
      truncated,
    },
  };
}

function buildRetrievalQuery(question: DatasetLine): string {
  return [
    question.title,
    ...question.scenario,
    question.prompt,
    question.reference_facts?.join(' '),
  ]
    .filter((part): part is string => typeof part === 'string' && part.trim().length > 0)
    .join('\n');
}

async function runSearch(params: {
  client: WikiRetriever;
  mode: CandidateMode;
  query: string;
  topK: number;
}): Promise<WikiSearchResponse> {
  const request = { query: params.query, topK: params.topK };
  switch (params.mode) {
    case 'rag-bm25':
    case 'agent-bm25':
      return params.client.search(request);
    case 'rag-dense':
    case 'agent-dense':
      return params.client.semanticSearch(request);
    case 'agent-rg':
    case 'agent-literal':
      return params.client.literalSearch(request);
    case 'rag-hybrid':
    case 'agent-hybrid':
      return params.client.hybridSearch(request);
    case 'direct':
      throw new Error('direct mode does not use wiki search');
  }
}

function formatSource(read: WikiReadResponse): string {
  const heading =
    read.pointer.headingPath && read.pointer.headingPath.length > 0
      ? ` > ${read.pointer.headingPath.join(' > ')}`
      : '';
  return [
    `[Wikipedia: ${read.pointer.title}${heading}]`,
    read.pointer.url ? `URL: ${read.pointer.url}` : null,
    read.text,
  ]
    .filter((part): part is string => Boolean(part))
    .join('\n');
}

function traceSearch(search: WikiSearchResponse): RetrievalTrace['searches'][number] {
  return {
    mode: search.mode,
    query: search.query,
    latencyMs: search.latencyMs,
    hits: search.hits.map(traceHit),
  };
}

function traceHit(hit: WikiSearchHit): RetrievalTrace['searches'][number]['hits'][number] {
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
