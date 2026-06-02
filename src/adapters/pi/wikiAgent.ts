import {
  Agent,
  type AgentEvent,
  type AgentLoopTurnUpdate,
  type AgentTool,
} from '@earendil-works/pi-agent-core';
import {
  registerBuiltInApiProviders,
  Type,
  type AssistantMessage,
  type Model,
  type OpenRouterRouting,
  type Usage,
} from '@earendil-works/pi-ai';

import type { ApocbenchConfig, CandidateMode } from '../../core/config/schema';
import { toOpenRouterProviderParam } from '../../core/config/schema';
import { CANDIDATE_SYSTEM_PROMPT } from '../../core/prompts/systemPrompts';
import type { WikiClient } from '../../core/wiki/client';
import type { RetrievalTrace } from '../../core/wiki/rag';
import type { WikiReadResponse, WikiSearchMode, WikiSearchResponse } from '../../core/wiki/types';
import { redactSecrets } from '../../utils/redaction';
import { createTextToolProtocolStreamFn } from './textToolProtocol';

registerBuiltInApiProviders();

type ModelEntry = ApocbenchConfig['models'][number];

type WikiAgentClient = Pick<
  WikiClient,
  'search' | 'semanticSearch' | 'hybridSearch' | 'literalSearch' | 'read'
>;

type WikiToolSearchRequest = {
  query: string;
  topK: number;
  articleId?: string;
  chunkId?: string;
};

type SearchToolSpec = {
  name: string;
  label: string;
  description: string;
  execute: (
    client: WikiAgentClient,
    request: WikiToolSearchRequest,
  ) => Promise<WikiSearchResponse>;
};

export type PiWikiAgentResult = {
  completion: string;
  retrievalTrace: RetrievalTrace;
  usage?: Usage;
  generationId?: string;
  costUsd?: number;
};

export async function runPiWikiAgent(params: {
  config: ApocbenchConfig;
  modelEntry: ModelEntry;
  basePrompt: string;
  mode: CandidateMode;
  wikiClient: WikiAgentClient;
  signal?: AbortSignal;
}): Promise<PiWikiAgentResult> {
  const { config, modelEntry, basePrompt, mode, wikiClient } = params;
  if (!config.wiki) throw new Error(`missing wiki config for candidateMode: ${mode}`);
  if (modelEntry.router !== 'openrouter') {
    throw new Error(`Pi wiki agent requires an OpenRouter model, got router: ${modelEntry.router}`);
  }

  const trace = createTrace(mode);
  const finalMessages: AgentEvent[] = [];
  let toolCalls = 0;
  let lastAssistant: AssistantMessage | undefined;

  const agentRef: { current?: Agent } = {};
  const prepareNextTurn = (): AgentLoopTurnUpdate | undefined => {
    const currentAgent = agentRef.current;
    if (!currentAgent) return undefined;
    const maxTurns = config.wiki?.limits.maxTurns;
    if (maxTurns != null && countAssistantTurns(currentAgent.state.messages) >= maxTurns) {
      return {
        context: {
          systemPrompt: currentAgent.state.systemPrompt,
          messages: [
            ...currentAgent.state.messages,
            {
              role: 'user',
              content:
                'Stop using tools now and provide the final answer for judging. If the available Wikipedia context is incomplete, say so briefly and answer conservatively.',
              timestamp: Date.now(),
            },
          ],
          tools: [],
        },
      };
    }
    return undefined;
  };

  const agent = new Agent({
    initialState: {
      systemPrompt: buildPiAgentSystemPrompt(mode),
      model: toPiOpenRouterModel(config, modelEntry),
      thinkingLevel: 'off',
      tools: createWikiTools({
        mode,
        wiki: config.wiki,
        client: wikiClient,
        trace,
      }),
      messages: [],
    },
    getApiKey: (provider) => {
      if (provider !== 'openrouter') return undefined;
      const envName = config.routers.openrouter.apiKeyEnv;
      return process.env[envName];
    },
    beforeToolCall: async () => {
      toolCalls += 1;
      const maxToolCalls = config.wiki?.limits.maxToolCalls;
      if (maxToolCalls != null && toolCalls > maxToolCalls) {
        return {
          block: true,
          reason: `wiki tool budget exceeded: maxToolCalls=${maxToolCalls}`,
        };
      }
      return undefined;
    },
    prepareNextTurn,
    streamFn: createTextToolProtocolStreamFn(),
    toolExecution: 'sequential',
  });
  agentRef.current = agent;

  agent.subscribe((event: AgentEvent) => {
    finalMessages.push(event);
    if (event.type === 'message_end' && event.message.role === 'assistant') {
      lastAssistant = event.message;
    }
  });

  await agent.prompt(basePrompt);
  await agent.waitForIdle();

  const assistant = lastAssistant ?? findLastAssistantMessage(agent.state.messages);
  if (!assistant) {
    const redacted = redactSecrets(finalMessages);
    throw new Error(`Pi agent did not produce an assistant message: ${JSON.stringify(redacted)}`);
  }

  if (assistant.stopReason === 'error' || assistant.stopReason === 'aborted') {
    throw new Error(assistant.errorMessage ?? `Pi agent stopped with ${assistant.stopReason}`);
  }

  const completion = assistant.content
    .filter((content) => content.type === 'text')
    .map((content) => content.text)
    .join('\n')
    .trim();

  if (!completion) {
    throw new Error('Pi agent produced an empty final answer');
  }

  return {
    completion,
    retrievalTrace: trace,
    usage: assistant.usage,
    generationId: assistant.responseId,
    costUsd: assistant.usage?.cost.total,
  };
}

function buildPiAgentSystemPrompt(mode: CandidateMode): string {
  const toolGuidance =
    mode === 'agent-wiki'
      ? 'You may use offline Wikipedia tools. Use wiki_hybrid_search as the default broad search because it combines BM25 and dense retrieval. Use wiki_search for exact terminology, article names, materials, hazards, or symptoms. Use wiki_semantic_search for concepts and synonyms when wording may differ. Use wiki_literal_search only to find an exact phrase inside a known articleId or chunkId from a prior result. Read chunks with wiki_read before relying on snippets, and treat Wikipedia as useful but fallible source material.'
      : `You may use offline Wikipedia tools. The search tool uses ${searchModeForCandidateMode(mode)} retrieval over a local Wikipedia index. Search when factual background would materially improve the answer, read chunks before relying on snippets, and treat Wikipedia as useful but fallible source material.`;
  return [
    CANDIDATE_SYSTEM_PROMPT,
    '',
    toolGuidance,
    'Use only the provided wiki tools. When you have enough context, stop using tools and provide the final answer directly for judging. Keep the answer practical, conservative, and safety-aware.',
  ].join('\n');
}

function toPiOpenRouterModel(
  config: ApocbenchConfig,
  modelEntry: ModelEntry,
): Model<'openai-completions'> {
  const routing = modelEntry.routing
    ? toOpenRouterProviderParam(modelEntry.routing)
    : modelEntry.provider
      ? { order: [modelEntry.provider], allow_fallbacks: false }
      : undefined;
  const maxTokens =
    config.candidate?.maxTokens ??
    modelEntry.params?.maxTokens ??
    config.routers.openrouter.default.maxTokens ??
    4096;

  return {
    id: modelEntry.model,
    name: modelEntry.model,
    api: 'openai-completions',
    provider: 'openrouter',
    baseUrl: config.routers.openrouter.baseUrl,
    reasoning: false,
    input: ['text'],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 128000,
    maxTokens,
    compat: {
      thinkingFormat: 'openrouter',
      openRouterRouting: routing as OpenRouterRouting | undefined,
    },
  };
}

function createTrace(mode: CandidateMode): RetrievalTrace {
  return {
    mode,
    queries: [],
    searches: [],
    reads: [],
    contextChars: 0,
    truncated: false,
  };
}

function createWikiTools(params: {
  mode: CandidateMode;
  wiki: NonNullable<ApocbenchConfig['wiki']>;
  client: WikiAgentClient;
  trace: RetrievalTrace;
}): AgentTool[] {
  const { mode, wiki, client, trace } = params;
  const searchTools = searchToolsForCandidateMode(mode).map((searchTool): AgentTool => ({
    name: searchTool.name,
    label: searchTool.label,
    description: searchTool.description,
    parameters: Type.Object({
      query: Type.String({ minLength: 1 }),
      topK: Type.Optional(Type.Number({ minimum: 1, maximum: wiki.limits.searchTopK })),
      articleId: Type.Optional(Type.String({ minLength: 1 })),
      chunkId: Type.Optional(Type.String({ minLength: 1 })),
    }),
    executionMode: 'sequential',
    execute: async (_toolCallId, input) => {
      const args = input as {
        query: string;
        topK?: number;
        articleId?: string;
        chunkId?: string;
      };
      const query = args.query;
      const topK =
        typeof args.topK === 'number'
          ? Math.min(Math.max(Math.floor(args.topK), 1), wiki.limits.searchTopK)
          : wiki.limits.searchTopK;
      const search = await searchTool.execute(client, {
        query,
        topK,
        articleId: args.articleId,
        chunkId: args.chunkId,
      });
      trace.queries.push(query);
      trace.searches.push(traceSearch(search));
      const formatted = formatSearchForTool(search);
      return {
        content: [{ type: 'text', text: JSON.stringify(formatted) }],
        details: formatted,
      };
    },
  }));

  return [
    ...searchTools,
    {
      name: 'wiki_read',
      label: 'Wikipedia read',
      description:
        'Read a bounded Wikipedia chunk returned by a wiki search tool. Use chunkId from a search hit.',
      parameters: Type.Object({
        chunkId: Type.String({ minLength: 1 }),
        maxChars: Type.Optional(Type.Number({ minimum: 1, maximum: wiki.limits.readMaxChars })),
      }),
      executionMode: 'sequential',
      execute: async (_toolCallId, input) => {
        const args = input as { chunkId: string; maxChars?: number };
        const remaining = Math.max(0, wiki.limits.contextMaxChars - trace.contextChars);
        if (remaining <= 0) {
          trace.truncated = true;
          return {
            content: [{ type: 'text', text: 'wiki context character budget exhausted' }],
            details: { error: 'context_budget_exhausted' },
          };
        }
        const requested =
          typeof args.maxChars === 'number'
            ? Math.min(Math.max(Math.floor(args.maxChars), 1), wiki.limits.readMaxChars)
            : wiki.limits.readMaxChars;
        const read = await client.read({
          chunkId: args.chunkId,
          maxChars: Math.min(requested, remaining),
        });
        trace.contextChars += read.text.length;
        trace.truncated = trace.truncated || read.truncated;
        trace.reads.push(traceRead(read));
        const formatted = formatReadForTool(read);
        return {
          content: [{ type: 'text', text: JSON.stringify(formatted) }],
          details: formatted,
        };
      },
    },
  ];
}

const BM25_SEARCH_TOOL: SearchToolSpec = {
  name: 'wiki_search',
  label: 'Wikipedia BM25 search',
  description:
    'Search the local offline Wikipedia BM25 index. Use concise specific queries with article names, procedures, materials, hazards, or symptoms.',
  execute: (client, request) => client.search(request),
};

const SEMANTIC_SEARCH_TOOL: SearchToolSpec = {
  name: 'wiki_semantic_search',
  label: 'Wikipedia semantic search',
  description:
    'Search local offline Wikipedia dense signposts semantically. Use this for concepts, synonyms, and related background when exact words may differ.',
  execute: (client, request) => client.semanticSearch(request),
};

const HYBRID_SEARCH_TOOL: SearchToolSpec = {
  name: 'wiki_hybrid_search',
  label: 'Wikipedia hybrid search',
  description:
    'Search local offline Wikipedia with deterministic BM25 plus dense hybrid retrieval. Use this as the default broad discovery tool when both exact terms and semantic matches matter.',
  execute: (client, request) => client.hybridSearch(request),
};

const LITERAL_SEARCH_TOOL: SearchToolSpec = {
  name: 'wiki_literal_search',
  label: 'Wikipedia literal search',
  description:
    'Find an exact phrase inside a known Wikipedia articleId or chunkId from a prior search/read result. Use BM25 or hybrid first for broad discovery.',
  execute: (client, request) => client.literalSearch(request),
};

function searchToolsForCandidateMode(mode: CandidateMode): SearchToolSpec[] {
  switch (mode) {
    case 'agent-bm25':
      return [BM25_SEARCH_TOOL];
    case 'agent-dense':
      return [SEMANTIC_SEARCH_TOOL];
    case 'agent-hybrid':
      return [HYBRID_SEARCH_TOOL];
    case 'agent-wiki':
      return [
        HYBRID_SEARCH_TOOL,
        BM25_SEARCH_TOOL,
        SEMANTIC_SEARCH_TOOL,
        LITERAL_SEARCH_TOOL,
      ];
    case 'agent-rg':
    case 'agent-literal':
      return [LITERAL_SEARCH_TOOL];
    default:
      throw new Error(`candidate mode does not expose Pi wiki tools: ${mode}`);
  }
}

function searchModeForCandidateMode(mode: CandidateMode): WikiSearchMode {
  switch (mode) {
    case 'agent-bm25':
      return 'bm25';
    case 'agent-dense':
      return 'dense';
    case 'agent-hybrid':
    case 'agent-wiki':
      return 'hybrid';
    case 'agent-rg':
    case 'agent-literal':
      return 'literal';
    default:
      throw new Error(`candidate mode does not use Pi wiki tools: ${mode}`);
  }
}

function formatSearchForTool(search: WikiSearchResponse) {
  return {
    mode: search.mode,
    query: search.query,
    hits: search.hits.map((hit) => ({
      articleId: hit.pointer.articleId,
      chunkId: hit.pointer.chunkId,
      title: hit.pointer.title,
      headingPath: hit.pointer.headingPath,
      url: hit.pointer.url,
      score: hit.score,
      bm25Score: hit.bm25Score,
      denseScore: hit.denseScore,
      sources: hit.sources,
      snippet: hit.snippet,
    })),
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
      bm25Score: hit.bm25Score,
      denseScore: hit.denseScore,
      sources: hit.sources,
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

function findLastAssistantMessage(messages: unknown[]): AssistantMessage | undefined {
  for (let idx = messages.length - 1; idx >= 0; idx -= 1) {
    const message = messages[idx] as Partial<AssistantMessage> | undefined;
    if (message?.role === 'assistant') return message as AssistantMessage;
  }
  return undefined;
}

function countAssistantTurns(messages: unknown[]): number {
  return messages.filter((message) => {
    return (message as { role?: unknown } | null | undefined)?.role === 'assistant';
  }).length;
}
