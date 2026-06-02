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
import type {
  WikiReadResponse,
  WikiSearchMode,
  WikiSearchResponse,
} from '../../core/wiki/types';
import { redactSecrets } from '../../utils/redaction';
import {
  createTextToolProtocolStreamFn,
  finalAnswerFromToolCall,
  parseTextToolCall,
} from './textToolProtocol';

registerBuiltInApiProviders();

const AGENT_WIKI_MIN_SEARCH_CALLS = 1;
const AGENT_WIKI_RECOMMENDED_SEARCH_CALLS = 2;

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
    beforeToolCall: async (context) => {
      const traceEntry = recordToolCallStart(trace, context);
      toolCalls += 1;
      const maxToolCalls = config.wiki?.limits.maxToolCalls;
      if (maxToolCalls != null && toolCalls > maxToolCalls) {
        if (traceEntry) {
          finishToolCallTrace(traceEntry, {
            status: 'blocked',
            error: `wiki tool budget exceeded: maxToolCalls=${maxToolCalls}`,
          });
        }
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

  const abortAgent = () => agent.abort();
  params.signal?.addEventListener('abort', abortAgent, { once: true });

  agent.subscribe((event: AgentEvent) => {
    finalMessages.push(event);
    if (event.type === 'message_end' && event.message.role === 'assistant') {
      lastAssistant = event.message;
    }
  });

  try {
    if (params.signal?.aborted) agent.abort();
    await agent.prompt(basePrompt);
    await agent.waitForIdle();
    lastAssistant = await repairAgentOutputIfNeeded({
      agent,
      mode,
      trace,
      lastAssistant,
      basePrompt,
    });
  } finally {
    params.signal?.removeEventListener('abort', abortAgent);
  }

  const assistant = lastAssistant ?? findLastAssistantMessage(agent.state.messages);
  if (!assistant) {
    const redacted = redactSecrets(finalMessages);
    throw new Error(`Pi agent did not produce an assistant message: ${JSON.stringify(redacted)}`);
  }

  if (assistant.stopReason === 'error' || assistant.stopReason === 'aborted') {
    throw new Error(assistant.errorMessage ?? `Pi agent stopped with ${assistant.stopReason}`);
  }

  const completion = normalizeFinalAnswerText(assistantText(assistant));

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
      ? [
          'You must use the local offline Wikipedia tools before answering.',
          '',
          'Workflow for every question:',
          '- Use at least one Wikipedia search before answering. Choose the first search by the prompt: use wiki_search for exact named devices, chemicals, diseases, materials, article titles, or technical terms; use wiki_hybrid_search for broad factual discovery when both exact terms and semantic matches matter.',
          '- Build search queries from distinctive nouns in the user prompt: materials, tools, chemicals, measurements, symptoms, named designs, failed methods, and desired outcome. Do not search only the failed method.',
          '- For troubleshooting questions, include the unusual available resources in the query. Example: if a low-carbon steel chisel failed to harden and the prompt mentions charcoal, leather, and bones, search for those materials plus hardening/case-hardening.',
          `- Aim for ${AGENT_WIKI_RECOMMENDED_SEARCH_CALLS} searches when useful. Run a second search with different wording when the first results are thin, ambiguous, or only adjacent: use wiki_search for exact names, chemicals, diseases, materials, article titles, or technical terms, or wiki_semantic_search for concepts and synonyms when wording may differ.`,
          '- Compare the returned titles, snippets, sources, articleIds, and chunkIds. Do not assume the first result is best.',
          '- Prefer article lead/overview chunks over narrow subtype chunks when the title is relevant but the heading path is adjacent or unrelated.',
          '- Read a chunk with wiki_read when search snippets are insufficient or when you rely on a specific factual claim.',
          '- If a read chunk is adjacent, vague, or not directly useful, run one refined search and read a better chunk.',
          '- Use wiki_literal_search only after search/read, with an exact copied phrase and a known chunkId or articleId.',
          '- When you have enough context, provide the final answer. You may use final_answer for deterministic submission, but plain text is also accepted.',
          '- Do not describe your search process, tool calls, retrieval, Wikipedia, or this harness in the final answer.',
          '',
          'Search selection:',
          '- Use wiki_search first for exact named devices, chemicals, diseases, materials, article titles, or technical terms.',
          '- Use wiki_hybrid_search for broad search because it combines BM25 and dense retrieval.',
          '- Use wiki_semantic_search when wording is vague, conceptual, or likely to differ from the article text.',
          '- Use wiki_literal_search only after search/read, with an exact copied phrase and a known chunkId or articleId.',
          '',
          'Do not merely summarize search snippets. Turn the retrieved facts into an executable answer for the original scenario, with concrete checks or verification steps when relevant.',
          'Treat snippets as leads, not evidence. Treat Wikipedia as useful but incomplete source material. Do not skip tools just because you already know the topic.',
        ].join('\n')
      : `You may use offline Wikipedia tools. The search tool uses ${searchModeForCandidateMode(mode)} retrieval over a local Wikipedia index. Search when factual background would materially improve the answer, read chunks before relying on snippets, and treat Wikipedia as useful but fallible source material.`;
  return [
    CANDIDATE_SYSTEM_PROMPT,
    '',
    toolGuidance,
    mode === 'agent-wiki'
      ? 'Use only the provided wiki tools and final_answer. Keep the final answer practical, conservative, and safety-aware. If final_answer is awkward for your format, answer normally in plain text after using the wiki tools.'
      : 'Use only the provided wiki tools. When you have enough context, stop using tools and provide the final answer directly for judging. Keep the answer practical, conservative, and safety-aware.',
  ].join('\n');
}

async function repairAgentOutputIfNeeded(params: {
  agent: Agent;
  mode: CandidateMode;
  trace: RetrievalTrace;
  lastAssistant?: AssistantMessage;
  basePrompt: string;
}): Promise<AssistantMessage | undefined> {
  const { agent, mode, trace, basePrompt } = params;
  let lastAssistant = params.lastAssistant ?? findLastAssistantMessage(agent.state.messages);

  for (let attempt = 0; attempt < 3; attempt += 1) {
    const completion = lastAssistant ? normalizeFinalAnswerText(assistantText(lastAssistant)) : '';
    const issue = agentRepairIssue({ mode, trace, completion });
    if (!issue) return lastAssistant;

    if (issue.kind === 'final') {
      agent.state.tools = [];
    }

    await agent.prompt(repairPrompt(issue, basePrompt));
    await agent.waitForIdle();
    lastAssistant = findLastAssistantMessage(agent.state.messages) ?? lastAssistant;
  }

  return lastAssistant;
}

type AgentRepairIssue =
  | { kind: 'retrieval'; searchCount: number }
  | { kind: 'final' };

function agentRepairIssue(params: {
  mode: CandidateMode;
  trace: RetrievalTrace;
  completion: string;
}): AgentRepairIssue | undefined {
  if (params.mode === 'agent-wiki') {
    const searchCount = params.trace.searches.length;
    if (searchCount < AGENT_WIKI_MIN_SEARCH_CALLS) {
      return { kind: 'retrieval', searchCount };
    }
  }
  if (
    params.completion.length === 0 ||
    isMalformedTerminalOutput(params.completion) ||
    isMetaNonAnswerOutput(params.completion)
  ) {
    return { kind: 'final' };
  }
  return undefined;
}

function repairPrompt(issue: AgentRepairIssue, basePrompt: string): string {
  if (issue.kind === 'retrieval') {
    return [
      'The harness cannot accept your answer yet because the required Wikipedia retrieval is incomplete.',
      `Recorded so far: ${issue.searchCount} search call(s).`,
      'Before answering, use at least one relevant Wikipedia search tool call.',
      'Build the query from distinctive nouns in the original scenario.',
      'Then answer the original scenario directly. Read a chunk if a search snippet is not enough to support a factual claim.',
    ].join('\n');
  }

  return [
    'Your previous message did not contain a usable final answer for the benchmark judge.',
    'It either described tool use/retrieval, emitted only tool syntax, or failed to answer the original scenario.',
    'Original scenario:',
    basePrompt,
    '',
    'Now provide the complete final answer in plain text.',
    'Do not call any tools. Do not mention tools, search, retrieval, Wikipedia, the harness, or judging.',
  ].join('\n');
}

function normalizeFinalAnswerText(raw: string): string {
  const trimmed = raw.trim();
  if (trimmed.length === 0) return '';

  const parsed = parseTextToolCall(trimmed);
  if (parsed) {
    const finalAnswer = finalAnswerFromToolCall(parsed);
    if (finalAnswer != null) return finalAnswer.trim();
    return '';
  }

  const embeddedAnswer = extractJsonStringField(trimmed, 'answer');
  if (embeddedAnswer != null) return embeddedAnswer.trim();

  const finalTag = /<final_answer>\s*([\s\S]*?)\s*<\/final_answer>/i.exec(trimmed);
  if (finalTag?.[1]) return cleanTerminalTags(finalTag[1]);

  const cleaned = cleanTerminalTags(trimmed);
  return isMalformedTerminalOutput(cleaned) ? '' : cleaned;
}

function cleanTerminalTags(input: string): string {
  return stripNativeChannelMarkup(input)
    .replace(/<\/?final_answer>/gi, '')
    .replace(/<\/tool_call>/gi, '')
    .replace(/<\|end\|>/gi, '')
    .trim();
}

function stripNativeChannelMarkup(input: string): string {
  let output = input.trim();
  output = output.replace(/^<\|channel>\s*(?:thought|analysis|final|commentary)\s*<channel\|>\s*/i, '');
  output = output.replace(/<\|channel>\s*(?:thought|analysis|final|commentary)\s*<channel\|>/gi, '');
  output = output.replace(/^<\|(?:thought|analysis|final|commentary)\|>\s*/i, '');
  output = output.replace(/<\|(?:thought|analysis|final|commentary)\|>/gi, '');
  return output;
}

function isMalformedTerminalOutput(text: string): boolean {
  const trimmed = text.trim();
  if (trimmed.length === 0) return true;
  if (/^<\/?tool_call>\s*$/i.test(trimmed)) return true;
  if (/^<\|end\|>\s*$/i.test(trimmed)) return true;
  if (/^<\/tool_call>\s*<\|end\|>\s*$/i.test(trimmed)) return true;
  if (/<tool_call>[\s\S]*"name"\s*:\s*"final_answer"/i.test(trimmed)) return true;
  if (/<tool_call>|<\/tool_call>|<\|tool_call/i.test(trimmed)) return true;
  return false;
}

function isMetaNonAnswerOutput(text: string): boolean {
  const trimmed = text.trim();
  if (trimmed.length === 0) return true;
  const lower = trimmed.toLowerCase();
  const answerWords = lower.split(/\s+/).filter(Boolean).length;
  const hasExplicitToolProcess =
    /\b(?:search|look up|call|read|tool call|tool_call|wiki_(?:search|read|hybrid_search|semantic_search|literal_search)|retrieval process|search process|wikipedia tools?|local offline wikipedia|harness)\b/.test(
      lower,
    );
  const openingMetaPatterns = [
    /^okay,\s+(?:let(?:'s| me)|i need|we need)\b/,
    /^alright,\s+(?:let(?:'s| me)|i need|we need)\b/,
    /^the user (?:wants|is asking|asked|needs)\b/,
    /^i (?:need|should|will) (?:address|include|provide|make sure|structure|craft|answer)\b/,
    /^first,\s+i (?:need|should|will)\b/,
  ];
  if (
    openingMetaPatterns.some((pattern) => pattern.test(lower)) &&
    (answerWords < 120 || hasExplicitToolProcess)
  ) {
    return true;
  }

  const toolProcessPatterns = [
    /\bi (?:need|will|would|should|can|cannot|can't|must) (?:to )?(?:search|look up|call|read)\b/,
    /\bi (?:need|will|would|should|can|cannot|can't|must) (?:to )?use (?:the )?(?:tool|tools|wiki|wikipedia)\b/,
    /\bi(?:'ll| will) (?:search|look up|use|call|read)\b/,
    /\b(?:tool call|tool_call|wiki_(?:search|read|hybrid_search|semantic_search|literal_search))\b/,
    /\b(?:retrieval process|search process|wikipedia tools?|local offline wikipedia|harness)\b/,
    /\b(?:i need more information|i don't have enough information|unable to access|can't access)\b/,
  ];
  if (toolProcessPatterns.some((pattern) => pattern.test(lower))) return true;

  const genericPlanningPatterns = [
    /\b(?:i|we) (?:need|should|will) (?:to )?(?:address|include|provide|make sure|structure|craft|answer)\b/,
    /\b(?:i|we) (?:will|should|need to) (?:produce|give|supply|draft|craft) (?:the )?final answer\b/,
  ];
  if (genericPlanningPatterns.some((pattern) => pattern.test(lower)) && answerWords < 120) {
    return true;
  }

  if (answerWords < 25 && /\b(?:search|read|tool|retriev|wikipedia)\b/.test(lower)) return true;
  return false;
}

function extractJsonStringField(input: string, field: string): string | undefined {
  const match = new RegExp(`"${field}"\\s*:\\s*"((?:\\\\.|[^"\\\\])*)"`).exec(input);
  if (!match?.[1]) return undefined;
  try {
    return JSON.parse(`"${match[1]}"`) as string;
  } catch {
    return match[1].replace(/\\n/g, '\n').replace(/\\"/g, '"');
  }
}

function assistantText(message: AssistantMessage): string {
  const text = message.content
    .filter((content) => content.type === 'text')
    .map((content) => content.text)
    .join('\n')
    .trim();
  if (text.length > 0) return text;

  return message.content
    .filter((content) => content.type === 'thinking')
    .map((content) => content.thinking)
    .join('\n');
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
    toolCallCount: 0,
    toolCalls: [],
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
    execute: async (toolCallId, input) => {
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
      const request = {
        query,
        topK,
        articleId: args.articleId,
        chunkId: args.chunkId,
      };
      const toolTrace = ensureToolCallTrace(trace, {
        toolCallId,
        toolName: searchTool.name,
        args: request,
      });
      try {
        const search = await searchTool.execute(client, request);
        trace.queries.push(query);
        trace.searches.push(traceSearch(search));
        const formatted = formatSearchForTool(search);
        const result = {
          content: [{ type: 'text' as const, text: JSON.stringify(formatted) }],
          details: formatted,
        };
        finishToolCallTrace(toolTrace, {
          status: 'ok',
          result: summarizeSearchToolResult(result.content, search),
        });
        return result;
      } catch (error) {
        finishToolCallTrace(toolTrace, {
          status: 'error',
          error: error instanceof Error ? error.message : String(error),
        });
        throw error;
      }
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
      execute: async (toolCallId, input) => {
        const args = input as { chunkId: string; maxChars?: number };
        const requested =
          typeof args.maxChars === 'number'
            ? Math.min(Math.max(Math.floor(args.maxChars), 1), wiki.limits.readMaxChars)
            : wiki.limits.readMaxChars;
        const toolTrace = ensureToolCallTrace(trace, {
          toolCallId,
          toolName: 'wiki_read',
          args: { chunkId: args.chunkId, maxChars: requested },
        });
        const remaining = Math.max(0, wiki.limits.contextMaxChars - trace.contextChars);
        if (remaining <= 0) {
          trace.truncated = true;
          finishToolCallTrace(toolTrace, {
            status: 'error',
            error: 'context_budget_exhausted',
            result: {
              contentText: 'wiki context character budget exhausted',
              contentTextChars: 'wiki context character budget exhausted'.length,
              error: 'context_budget_exhausted',
            },
          });
          return {
            content: [{ type: 'text', text: 'wiki context character budget exhausted' }],
            details: { error: 'context_budget_exhausted' },
          };
        }
        try {
          const read = await client.read({
            chunkId: args.chunkId,
            maxChars: Math.min(requested, remaining),
          });
          trace.contextChars += read.text.length;
          trace.truncated = trace.truncated || read.truncated;
          trace.reads.push(traceRead(read));
          const formatted = formatReadForTool(read);
          const result = {
            content: [{ type: 'text' as const, text: JSON.stringify(formatted) }],
            details: formatted,
          };
          finishToolCallTrace(toolTrace, {
            status: 'ok',
            result: summarizeReadToolResult(result.content, read),
          });
          return result;
        } catch (error) {
          finishToolCallTrace(toolTrace, {
            status: 'error',
            error: error instanceof Error ? error.message : String(error),
          });
          throw error;
        }
      },
    },
  ];
}

function recordToolCallStart(
  trace: RetrievalTrace,
  context: unknown,
): RetrievalTrace['toolCalls'][number] | undefined {
  if (!isRecord(context)) return undefined;
  const toolCall = context.toolCall;
  if (!isRecord(toolCall)) return undefined;
  const toolCallId =
    typeof toolCall.id === 'string' && toolCall.id.length > 0
      ? toolCall.id
      : `tool-${trace.toolCalls.length + 1}`;
  const toolName =
    typeof toolCall.name === 'string' && toolCall.name.length > 0
      ? toolCall.name
      : 'unknown_tool';
  return startToolCallTrace(trace, {
    toolCallId,
    toolName,
    args: context.args ?? toolCall.arguments ?? {},
  });
}

function ensureToolCallTrace(
  trace: RetrievalTrace,
  params: { toolCallId: string; toolName: string; args: Record<string, unknown> },
): RetrievalTrace['toolCalls'][number] {
  const existing = trace.toolCalls.find(
    (call) => call.toolCallId === params.toolCallId && call.status === 'pending',
  );
  if (existing) {
    existing.toolName = params.toolName;
    existing.arguments = cloneRecord(params.args);
    return existing;
  }
  return startToolCallTrace(trace, params);
}

function startToolCallTrace(
  trace: RetrievalTrace,
  params: { toolCallId: string; toolName: string; args: unknown },
): RetrievalTrace['toolCalls'][number] {
  const entry: RetrievalTrace['toolCalls'][number] = {
    index: trace.toolCalls.length + 1,
    toolCallId: params.toolCallId,
    toolName: params.toolName,
    arguments: cloneRecord(params.args),
    status: 'pending',
    startedAtMs: Date.now(),
  };
  trace.toolCalls.push(entry);
  trace.toolCallCount = trace.toolCalls.length;
  return entry;
}

function finishToolCallTrace(
  entry: RetrievalTrace['toolCalls'][number],
  update: Pick<RetrievalTrace['toolCalls'][number], 'status'> &
    Partial<Pick<RetrievalTrace['toolCalls'][number], 'result' | 'error'>>,
): void {
  const completedAtMs = Date.now();
  entry.status = update.status;
  entry.completedAtMs = completedAtMs;
  entry.latencyMs = Math.max(0, completedAtMs - entry.startedAtMs);
  if (update.result) entry.result = update.result;
  if (update.error) entry.error = update.error;
}

function summarizeSearchToolResult(
  content: Array<{ type: 'text'; text: string }>,
  search: WikiSearchResponse,
): NonNullable<RetrievalTrace['toolCalls'][number]['result']> {
  const contentText = contentTextForTrace(content);
  return {
    contentText,
    contentTextChars: contentText.length,
    search: {
      mode: search.mode,
      query: search.query,
      hitCount: search.hits.length,
      topHits: search.hits.slice(0, 5).map((hit) => ({
        articleId: hit.pointer.articleId,
        chunkId: hit.pointer.chunkId,
        title: hit.pointer.title,
        score: hit.score,
        bm25Score: hit.bm25Score,
        denseScore: hit.denseScore,
        sources: hit.sources,
      })),
    },
  };
}

function summarizeReadToolResult(
  content: Array<{ type: 'text'; text: string }>,
  read: WikiReadResponse,
): NonNullable<RetrievalTrace['toolCalls'][number]['result']> {
  const contentText = contentTextForTrace(content);
  return {
    contentText,
    contentTextChars: contentText.length,
    read: {
      articleId: read.pointer.articleId,
      chunkId: read.pointer.chunkId,
      title: read.pointer.title,
      chars: read.text.length,
      truncated: read.truncated,
    },
  };
}

function contentTextForTrace(content: Array<{ type: 'text'; text: string }>): string {
  return content.map((item) => item.text).join('\n');
}

function cloneRecord(value: unknown): Record<string, unknown> {
  if (!isRecord(value)) return {};
  return JSON.parse(JSON.stringify(value)) as Record<string, unknown>;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
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
