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
import { rerankWithQmd } from '../../core/wiki/qmdReranker';
import { redactSecrets } from '../../utils/redaction';
import {
  createTextToolProtocolStreamFn,
  finalAnswerFromToolCall,
  parseTextToolCall,
} from './textToolProtocol';

registerBuiltInApiProviders();

const AGENT_WIKI_MIN_SEARCH_CALLS = 1;
const AGENT_WIKI_MIN_READ_CALLS = 1;
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

type ResearchSearchHit = ReturnType<typeof formatSearchForTool>['hits'][number] & {
  queries: string[];
  firstRank: number;
  rerankScore?: number;
};

type RerankedResearchHit = ResearchSearchHit & {
  rerankScore: number;
};

type SmartResearchRead = ReturnType<typeof formatReadForTool> & {
  selectionRank: number;
  matchedQueries: string[];
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

  const trace = createTrace(mode);
  const finalMessages: AgentEvent[] = [];
  let toolCalls = 0;
  let lastAssistant: AssistantMessage | undefined;

  const agentRef: { current?: Agent } = {};
  const prepareNextTurn = (): AgentLoopTurnUpdate | undefined => {
    const currentAgent = agentRef.current;
    if (!currentAgent) return undefined;
    const maxTurns = config.wiki?.limits.maxTurns;
    if (
      maxTurns != null &&
      countAssistantTurns(currentAgent.state.messages) >= maxTurns
    ) {
      if (isReadRequiredMode(mode) && trace.reads.length < AGENT_WIKI_MIN_READ_CALLS) {
        return {
          context: {
            systemPrompt: currentAgent.state.systemPrompt,
            messages: [
              ...currentAgent.state.messages,
              {
                role: 'user',
                content:
                  'Before answering, you still must call wiki_read on the most relevant chunkId returned by your wiki_research result. After reading, answer directly.',
                timestamp: Date.now(),
              },
            ],
            tools: currentAgent.state.tools,
          },
        };
      }
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
      model: toPiModel(config, modelEntry),
      thinkingLevel: toPiThinkingLevel(config),
      tools: createWikiTools({
        mode,
        wiki: config.wiki,
        client: wikiClient,
        trace,
        basePrompt,
      }),
      messages: [],
    },
    getApiKey: (provider) => {
      if (provider === 'openrouter') {
        const envName = config.routers.openrouter.apiKeyEnv;
        return process.env[envName];
      }
      if (provider === 'openai-compatible') {
        const envName = config.routers.openaiCompatible?.apiKeyEnv;
        return envName ? process.env[envName] : undefined;
      }
      if (provider === 'ollama') return 'ollama';
      return undefined;
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
    finalMessages.push(redactThinkingFromAgentEvent(event));
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
    throw new Error(
      `Pi agent did not produce an assistant message: ${JSON.stringify(redacted)}`,
    );
  }

  if (assistant.stopReason === 'error' || assistant.stopReason === 'aborted') {
    throw new Error(
      assistant.errorMessage ?? `Pi agent stopped with ${assistant.stopReason}`,
    );
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

function isBm25ResearchMode(mode: CandidateMode): boolean {
  return (
    mode === 'agent-bm25-research' ||
    mode === 'agent-bm25-research-v2' ||
    mode === 'agent-bm25-research-read-required'
  );
}

function isBm25RerankResearchMode(mode: CandidateMode): boolean {
  return mode === 'agent-bm25-rerank-research';
}

function isReadRequiredMode(mode: CandidateMode): boolean {
  return mode === 'agent-bm25-research-read-required';
}

function isSmartResearchMode(mode: CandidateMode): boolean {
  return (
    mode === 'agent-bm25-research-smart-read' ||
    mode === 'agent-hybrid-research-smart-read'
  );
}

function smartResearchSearchMode(mode: CandidateMode): WikiSearchMode {
  if (mode === 'agent-hybrid-research-smart-read') return 'hybrid';
  return 'bm25';
}

function buildPiAgentSystemPrompt(mode: CandidateMode): string {
  const toolGuidance = (() => {
    if (mode === 'agent-wiki') {
      return [
        'You must use the local offline Wikipedia tools before answering.',
        '',
        'Workflow for every question:',
        '- Use at least one Wikipedia search before answering. Choose the first search by the prompt: use wiki_search for exact named devices, chemicals, diseases, materials, article titles, or technical terms; use wiki_hybrid_search for broad factual discovery when both exact terms and semantic matches matter.',
        '- Build search queries from distinctive nouns in the user prompt: materials, tools, chemicals, measurements, symptoms, named designs, failed methods, and desired outcome. Do not search only the failed method.',
        '- For troubleshooting questions, include unusual available resources, symptoms, measurements, failed attempts, and desired outcome in the query.',
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
      ].join('\n');
    }

    if (mode === 'agent-bm25') {
      return [
        'You must use the local offline Wikipedia BM25 tools before answering.',
        '',
        'Workflow for every question:',
        '- Use wiki_search at least once before answering. It is a lexical BM25 search over local offline Wikipedia.',
        '- Build concise queries from distinctive words in the original prompt: materials, tools, chemicals, measurements, symptoms, named designs, failed methods, and desired outcome.',
        '- Compare returned titles, snippets, articleIds, and chunkIds. Do not assume the first result is best.',
        '- Use wiki_read for a relevant chunk before relying on a specific factual claim. Treat snippets as leads, not evidence.',
        '- If the first search is thin or only adjacent, run one refined BM25 search with different exact terms.',
        '- When you have enough context, provide the final answer. You may use final_answer for deterministic submission, but plain text is also accepted.',
        '- Do not describe your search process, tool calls, retrieval, Wikipedia, or this harness in the final answer.',
        '',
        'Do not merely summarize search snippets. Turn the retrieved facts into an executable answer for the original scenario, with concrete checks or verification steps when relevant.',
        'Treat Wikipedia as useful but incomplete source material. If retrieved context is irrelevant or incomplete, answer conservatively and state uncertainty.',
      ].join('\n');
    }

    if (isBm25ResearchMode(mode) || isBm25RerankResearchMode(mode)) {
      const requiresRead = isReadRequiredMode(mode);
      const toolName = isBm25RerankResearchMode(mode)
        ? 'wiki_rerank_research'
        : 'wiki_research';
      const toolDescription = isBm25RerankResearchMode(mode)
        ? 'It runs broad lexical BM25 searches, then uses a local semantic reranker to reorder the candidate snippets by relevance to the original task.'
        : 'It accepts multiple exact BM25-style queries and returns deduped Wikipedia candidates with the queries that matched each hit.';
      return [
        requiresRead
          ? 'You must use the local offline Wikipedia research tools and read at least one returned chunk before answering.'
          : 'You must use the local offline Wikipedia research tools before answering.',
        '',
        'Important retrieval behavior:',
        '- BM25 is lexical search, not semantic search. A high BM25 score means strong word overlap; it does not prove the page is about the scenario.',
        '- Titles, headings, and snippets are relevance checks. If they point to a song, band, place, name, list, or adjacent topic while the scenario needs a material, hazard, process, organism, or measurement, treat that hit as irrelevant and reformulate.',
        '- Good BM25 queries are short and noun-heavy: include the material, organism, symptom, device, chemical, procedure, hazard, measurement, or failure mode. Avoid filler instructions from the prompt unless those words are truly the subject.',
        '- Use several query shapes instead of one long sentence: literal scenario terms, technical synonyms, and the underlying rule/procedure/hazard.',
        '',
        'Workflow for every question:',
        `- Use ${toolName} first. ${toolDescription}`,
        '- Provide 2-4 query variants when possible: one literal query from the scenario, one query with the key material/tool/symptom names, and one query for the underlying rule, hazard, ratio, or procedure.',
        '- Compare titles, headings, matched queries, snippets, articleIds, and chunkIds. Do not assume the first result is best.',
        isBm25RerankResearchMode(mode)
          ? '- Treat rerankScore as a relevance prior, not proof. Still reject candidates whose title/heading/snippet do not match the original task.'
          : '- Treat BM25 score as a lexical match signal, not proof. Prefer the candidate whose title/heading/snippet actually match the original task.',
        requiresRead
          ? '- You are not allowed to give the final answer until you have called wiki_read on at least one relevant chunkId returned by wiki_research.'
          : '- Use wiki_read before relying on a concrete factual claim. Snippets are leads, not evidence.',
        '- Snippets are leads, not evidence. Pick the most relevant chunkId from the search results and read it before turning facts into advice.',
        '- For safety, medicine, chemistry, mechanics, ratios, load-bearing, fire, poison, pressure, electricity, birth, or directional procedures, read at least one relevant chunk and verify surprising directionality or hazards with a second search or read when possible.',
        '- If retrieved context is irrelevant, adjacent, or seems to contradict the scenario, ignore it and answer conservatively from first principles rather than forcing it into the answer.',
        '- When you have enough context, provide the final answer. You may use final_answer for deterministic submission, but plain text is also accepted.',
        '- Do not describe your search process, tool calls, retrieval, Wikipedia, or this harness in the final answer.',
        '',
        'Do not merely summarize search snippets. Turn retrieved facts into an executable answer for the original scenario, with concrete checks, verification steps, and safe stop conditions when relevant.',
      ].join('\n');
    }

    if (isSmartResearchMode(mode)) {
      const searchLabel =
        mode === 'agent-hybrid-research-smart-read'
          ? 'hybrid BM25+dense'
          : 'BM25';
      return [
        `You must use the local offline Wikipedia smart research tool before answering. It runs ${searchLabel} searches, ranks candidates, and reads the best chunks for you.`,
        '',
        'Workflow for every question:',
        '- Use wiki_smart_research first. It accepts multiple query variants and returns both ranked search candidates and read chunks.',
        '- Provide 2-4 query variants when possible: one literal query from the scenario, one query with key material/tool/symptom names, and one query for the underlying rule, hazard, ratio, or procedure.',
        '- Set readCount to 2 unless the question is simple and one specific article is obviously enough. The tool reads the highest-ranked distinct chunks automatically.',
        '- Treat the returned read chunks as evidence only when they directly match the scenario. If a chunk is adjacent or irrelevant, ignore it rather than forcing it into the answer.',
        '- Use the read chunks to verify concrete factual claims, directions, hazards, measurements, materials, symptoms, and stop-work triggers.',
        '- Do not merely summarize the read chunks. Turn the useful facts into an executable answer for the original scenario.',
        '- If retrieved context is incomplete, answer conservatively from first principles and state uncertainty briefly.',
        '- When you have enough context, provide the final answer. You may use final_answer for deterministic submission, but plain text is also accepted.',
        '- Do not describe your search process, tool calls, retrieval, Wikipedia, or this harness in the final answer.',
      ].join('\n');
    }

    return `You may use offline Wikipedia tools. The search tool uses ${searchModeForCandidateMode(mode)} retrieval over a local Wikipedia index. Search when factual background would materially improve the answer, read chunks before relying on snippets, and treat Wikipedia as useful but fallible source material.`;
  })();
  return [
    CANDIDATE_SYSTEM_PROMPT,
    '',
    toolGuidance,
    mode === 'agent-wiki' ||
    mode === 'agent-bm25' ||
    isBm25ResearchMode(mode) ||
    isBm25RerankResearchMode(mode) ||
    isSmartResearchMode(mode)
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
  let lastAssistant =
    params.lastAssistant ?? findLastAssistantMessage(agent.state.messages);

  for (let attempt = 0; attempt < 3; attempt += 1) {
    const completion = lastAssistant
      ? normalizeFinalAnswerText(assistantText(lastAssistant))
      : '';
    const issue = agentRepairIssue({ mode, trace, completion });
    if (!issue) return lastAssistant;

    trace.repairAttemptCount += 1;
    trace.repairReasons.push(repairIssueReason(issue));

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
  | { kind: 'read'; readCount: number }
  | { kind: 'final' };

function agentRepairIssue(params: {
  mode: CandidateMode;
  trace: RetrievalTrace;
  completion: string;
}): AgentRepairIssue | undefined {
  if (
    params.mode === 'agent-wiki' ||
    params.mode === 'agent-bm25' ||
    isBm25ResearchMode(params.mode) ||
    isBm25RerankResearchMode(params.mode) ||
    isSmartResearchMode(params.mode)
  ) {
    const searchCount = params.trace.searches.length;
    if (searchCount < AGENT_WIKI_MIN_SEARCH_CALLS) {
      return { kind: 'retrieval', searchCount };
    }
  }
  if (isReadRequiredMode(params.mode)) {
    const readCount = params.trace.reads.length;
    if (readCount < AGENT_WIKI_MIN_READ_CALLS) {
      return { kind: 'read', readCount };
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

function repairIssueReason(issue: AgentRepairIssue): string {
  if (issue.kind === 'retrieval') {
    return `retrieval_required_searches_missing:${issue.searchCount}`;
  }
  if (issue.kind === 'read') {
    return `retrieval_required_reads_missing:${issue.readCount}`;
  }
  return 'missing_or_malformed_final_answer';
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
  if (issue.kind === 'read') {
    return [
      'The harness cannot accept your answer yet because this mode requires reading a Wikipedia chunk before answering.',
      `Recorded so far: ${issue.readCount} read call(s).`,
      'Use wiki_read on the most relevant chunkId returned by your wiki_research result.',
      'If the first result was irrelevant, run one refined wiki_research query and then read the best chunkId from that result.',
      'After reading, answer the original scenario directly. Do not mention tools, search, retrieval, Wikipedia, the harness, or judging.',
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

  const wholeJsonAnswer = extractWholeJsonAnswer(trimmed);
  if (wholeJsonAnswer != null) return wholeJsonAnswer.trim();

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
  output = output.replace(
    /^<\|channel>\s*(?:thought|analysis|final|commentary)\s*<channel\|>\s*/i,
    '',
  );
  output = output.replace(
    /<\|channel>\s*(?:thought|analysis|final|commentary)\s*<channel\|>/gi,
    '',
  );
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
  if (
    genericPlanningPatterns.some((pattern) => pattern.test(lower)) &&
    answerWords < 120
  ) {
    return true;
  }

  if (answerWords < 25 && /\b(?:search|read|tool|retriev|wikipedia)\b/.test(lower))
    return true;
  return false;
}

function extractWholeJsonAnswer(input: string): string | undefined {
  if (!input.startsWith('{') || !input.endsWith('}')) return undefined;
  try {
    const parsed = JSON.parse(input) as unknown;
    if (!isRecord(parsed) || typeof parsed.answer !== 'string') return undefined;
    return parsed.answer;
  } catch {
    return undefined;
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

function toPiModel(
  config: ApocbenchConfig,
  modelEntry: ModelEntry,
): Model<'openai-completions'> {
  if (modelEntry.router === 'ollama') return toPiOllamaModel(config, modelEntry);
  if (modelEntry.router === 'openai-compatible') {
    return toPiOpenAiCompatibleModel(config, modelEntry);
  }
  return toPiOpenRouterModel(config, modelEntry);
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
    reasoning: Boolean(config.candidate?.reasoning?.enabled),
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

function toPiThinkingLevel(
  config: ApocbenchConfig,
): 'off' | 'minimal' | 'low' | 'medium' | 'high' | 'xhigh' {
  const reasoning = config.candidate?.reasoning;
  if (!reasoning?.enabled) return 'off';
  return reasoning.effort ?? 'high';
}

function redactThinkingFromAgentEvent(event: AgentEvent): AgentEvent {
  if (!('message' in event) || !event.message || event.message.role !== 'assistant') {
    return event;
  }
  return {
    ...event,
    message: {
      ...event.message,
      content: event.message.content.map((block) => {
        if (block.type !== 'thinking') return block;
        return {
          ...block,
          thinking: '[redacted candidate reasoning]',
        };
      }),
    },
  };
}

function toPiOllamaModel(
  config: ApocbenchConfig,
  modelEntry: ModelEntry,
): Model<'openai-completions'> {
  const maxTokens =
    config.candidate?.maxTokens ??
    modelEntry.params?.maxTokens ??
    config.routers.ollama.default.maxTokens ??
    4096;

  return {
    id: modelEntry.model,
    name: modelEntry.model,
    api: 'openai-completions',
    provider: 'ollama',
    baseUrl: ollamaOpenAiBaseUrl(config.routers.ollama.baseUrl),
    reasoning: false,
    input: ['text'],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 32768,
    maxTokens,
    compat: {
      supportsDeveloperRole: false,
      supportsReasoningEffort: false,
    },
  };
}

function toPiOpenAiCompatibleModel(
  config: ApocbenchConfig,
  modelEntry: ModelEntry,
): Model<'openai-completions'> {
  const routerConfig = config.routers.openaiCompatible;
  if (!routerConfig) throw new Error('missing router config: routers.openaiCompatible');
  const maxTokens =
    config.candidate?.maxTokens ??
    modelEntry.params?.maxTokens ??
    routerConfig.default.maxTokens ??
    4096;

  return {
    id: modelEntry.model,
    name: modelEntry.model,
    api: 'openai-completions',
    provider: 'openai-compatible',
    baseUrl: routerConfig.baseUrl,
    reasoning: false,
    input: ['text'],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 32768,
    maxTokens,
    compat: {
      supportsDeveloperRole: false,
      supportsReasoningEffort: false,
    },
  };
}

function ollamaOpenAiBaseUrl(baseUrl: string): string {
  const trimmed = baseUrl.replace(/\/+$/, '');
  if (trimmed.endsWith('/v1')) return trimmed;
  if (trimmed.endsWith('/api')) return trimmed.slice(0, -'/api'.length) + '/v1';
  return `${trimmed}/v1`;
}

function createTrace(mode: CandidateMode): RetrievalTrace {
  return {
    mode,
    queries: [],
    toolCallCount: 0,
    repairAttemptCount: 0,
    repairReasons: [],
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
  basePrompt: string;
}): AgentTool[] {
  const { mode, wiki, client, trace, basePrompt } = params;
  const researchTools =
    isBm25ResearchMode(mode) ? [createResearchTool({ wiki, client, trace })] : [];
  const rerankResearchTools = isBm25RerankResearchMode(mode)
    ? [createRerankResearchTool({ wiki, client, trace, basePrompt })]
    : [];
  const smartResearchTools = isSmartResearchMode(mode)
    ? [
        createSmartResearchTool({
          mode: smartResearchSearchMode(mode),
          wiki,
          client,
          trace,
        }),
      ]
    : [];
  const searchTools = searchToolsForCandidateMode(mode).map(
    (searchTool): AgentTool => ({
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
    }),
  );

  return [
    ...researchTools,
    ...rerankResearchTools,
    ...smartResearchTools,
    ...searchTools,
    {
      name: 'wiki_read',
      label: 'Wikipedia read',
      description:
        'Read a bounded Wikipedia chunk returned by a wiki search tool. Use chunkId from a search hit.',
      parameters: Type.Object({
        chunkId: Type.String({ minLength: 1 }),
        maxChars: Type.Optional(
          Type.Number({ minimum: 1, maximum: wiki.limits.readMaxChars }),
        ),
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

function createSmartResearchTool(params: {
  mode: WikiSearchMode;
  wiki: NonNullable<ApocbenchConfig['wiki']>;
  client: WikiAgentClient;
  trace: RetrievalTrace;
}): AgentTool {
  const { mode, wiki, client, trace } = params;
  return {
    name: 'wiki_smart_research',
    label: `Wikipedia ${mode} smart research`,
    description:
      'Run 1-4 local Wikipedia searches, rank and dedupe the results, automatically read the best distinct chunks, and return both candidates and read evidence. Use this before answering.',
    parameters: Type.Object({
      query: Type.String({ minLength: 1 }),
      query2: Type.Optional(Type.String({ minLength: 1 })),
      query3: Type.Optional(Type.String({ minLength: 1 })),
      query4: Type.Optional(Type.String({ minLength: 1 })),
      topK: Type.Optional(Type.Number({ minimum: 1, maximum: wiki.limits.searchTopK })),
      readCount: Type.Optional(Type.Number({ minimum: 1, maximum: 2 })),
    }),
    executionMode: 'sequential',
    execute: async (toolCallId, input) => {
      const args = input as {
        query: string;
        query2?: string;
        query3?: string;
        query4?: string;
        topK?: number;
        readCount?: number;
      };
      const topK =
        typeof args.topK === 'number'
          ? Math.min(Math.max(Math.floor(args.topK), 1), wiki.limits.searchTopK)
          : wiki.limits.searchTopK;
      const readCount =
        typeof args.readCount === 'number'
          ? Math.min(Math.max(Math.floor(args.readCount), 1), 2)
          : 2;
      const queries = [args.query, args.query2, args.query3, args.query4]
        .filter((query): query is string => typeof query === 'string')
        .map((query) => query.trim())
        .filter((query, index, all) => query.length > 0 && all.indexOf(query) === index)
        .slice(0, 4);
      const toolTrace = ensureToolCallTrace(trace, {
        toolCallId,
        toolName: 'wiki_smart_research',
        args: { ...args, topK, readCount, mode },
      });
      try {
        const searches = [];
        for (const query of queries) {
          const search = await runSmartResearchSearch({ client, mode, query, topK });
          searches.push(search);
          trace.queries.push(query);
          trace.searches.push(traceSearch(search));
        }

        const formattedResearch = formatResearchForTool(searches, `${mode}-smart-read`);
        const selectedHits = selectSmartReadHits(formattedResearch.hits, readCount);
        const reads: SmartResearchRead[] = [];
        for (const hit of selectedHits) {
          if (!hit.chunkId) continue;
          const remaining = Math.max(0, wiki.limits.contextMaxChars - trace.contextChars);
          if (remaining <= 0) {
            trace.truncated = true;
            break;
          }
          const read = await client.read({
            chunkId: hit.chunkId,
            maxChars: Math.min(wiki.limits.readMaxChars, remaining),
          });
          trace.contextChars += read.text.length;
          trace.truncated = trace.truncated || read.truncated;
          trace.reads.push(traceRead(read));
          reads.push({
            ...formatReadForTool(read),
            selectionRank: hit.firstRank,
            matchedQueries: hit.queries,
          });
        }

        const formatted = {
          ...formattedResearch,
          readCount: reads.length,
          reads,
        };
        const result = {
          content: [{ type: 'text' as const, text: JSON.stringify(formatted) }],
          details: formatted,
        };
        finishToolCallTrace(toolTrace, {
          status: 'ok',
          result: {
            contentText: result.content[0].text,
            contentTextChars: result.content[0].text.length,
            search: {
              mode: `${mode}-smart-read`,
              query: queries.join(' | '),
              hitCount: formattedResearch.hits.length,
              topHits: formattedResearch.hits.slice(0, 5).map((hit) => ({
                articleId: hit.articleId,
                chunkId: hit.chunkId,
                title: hit.title,
                score: hit.score,
                bm25Score: hit.bm25Score,
                denseScore: hit.denseScore,
                sources: hit.sources,
              })),
            },
            read:
              reads.length > 0
                ? {
                    articleId: reads[0].articleId,
                    chunkId: reads[0].chunkId,
                    title: reads[0].title,
                    chars: reads.reduce((total, read) => total + read.text.length, 0),
                    truncated: reads.some((read) => read.truncated),
                  }
                : undefined,
          },
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
  };
}

function createResearchTool(params: {
  wiki: NonNullable<ApocbenchConfig['wiki']>;
  client: WikiAgentClient;
  trace: RetrievalTrace;
}): AgentTool {
  const { wiki, client, trace } = params;
  return {
    name: 'wiki_research',
    label: 'Wikipedia BM25 research',
    description:
      'Run 1-4 lexical BM25 searches over local offline Wikipedia, merge and dedupe the hits, and return candidates annotated by which query matched. Use this before wiki_read for robust factual discovery.',
    parameters: Type.Object({
      query: Type.String({ minLength: 1 }),
      query2: Type.Optional(Type.String({ minLength: 1 })),
      query3: Type.Optional(Type.String({ minLength: 1 })),
      query4: Type.Optional(Type.String({ minLength: 1 })),
      topK: Type.Optional(Type.Number({ minimum: 1, maximum: wiki.limits.searchTopK })),
    }),
    executionMode: 'sequential',
    execute: async (toolCallId, input) => {
      const args = input as {
        query: string;
        query2?: string;
        query3?: string;
        query4?: string;
        topK?: number;
      };
      const topK =
        typeof args.topK === 'number'
          ? Math.min(Math.max(Math.floor(args.topK), 1), wiki.limits.searchTopK)
          : wiki.limits.searchTopK;
      const queries = [args.query, args.query2, args.query3, args.query4]
        .filter((query): query is string => typeof query === 'string')
        .map((query) => query.trim())
        .filter((query, index, all) => query.length > 0 && all.indexOf(query) === index)
        .slice(0, 4);
      const toolTrace = ensureToolCallTrace(trace, {
        toolCallId,
        toolName: 'wiki_research',
        args: { ...args, topK },
      });
      try {
        const searches = [];
        for (const query of queries) {
          const search = await client.search({ query, topK });
          searches.push(search);
          trace.queries.push(query);
          trace.searches.push(traceSearch(search));
        }
        const formatted = formatResearchForTool(searches);
        const result = {
          content: [{ type: 'text' as const, text: JSON.stringify(formatted) }],
          details: formatted,
        };
        finishToolCallTrace(toolTrace, {
          status: 'ok',
          result: {
            contentText: result.content[0].text,
            contentTextChars: result.content[0].text.length,
            search: {
              mode: 'bm25-research',
              query: queries.join(' | '),
              hitCount: formatted.hits.length,
              topHits: formatted.hits.slice(0, 5).map((hit) => ({
                articleId: hit.articleId,
                chunkId: hit.chunkId,
                title: hit.title,
                score: hit.score,
                bm25Score: hit.bm25Score,
                sources: hit.sources,
              })),
            },
          },
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
  };
}

function createRerankResearchTool(params: {
  wiki: NonNullable<ApocbenchConfig['wiki']>;
  client: WikiAgentClient;
  trace: RetrievalTrace;
  basePrompt: string;
}): AgentTool {
  const { wiki, client, trace, basePrompt } = params;
  return {
    name: 'wiki_rerank_research',
    label: 'Wikipedia BM25 + QMD rerank research',
    description:
      'Run 1-4 broad lexical BM25 searches over local offline Wikipedia, merge the candidate pool, rerank candidates with a local Qwen3 reranker, and return the most relevant hits. Use this before wiki_read for robust factual discovery.',
    parameters: Type.Object({
      query: Type.String({ minLength: 1 }),
      query2: Type.Optional(Type.String({ minLength: 1 })),
      query3: Type.Optional(Type.String({ minLength: 1 })),
      query4: Type.Optional(Type.String({ minLength: 1 })),
      topK: Type.Optional(Type.Number({ minimum: 1, maximum: wiki.limits.searchTopK })),
    }),
    executionMode: 'sequential',
    execute: async (toolCallId, input) => {
      const args = input as {
        query: string;
        query2?: string;
        query3?: string;
        query4?: string;
        topK?: number;
      };
      const topK =
        typeof args.topK === 'number'
          ? Math.min(Math.max(Math.floor(args.topK), 1), wiki.limits.searchTopK)
          : wiki.limits.searchTopK;
      const queries = [args.query, args.query2, args.query3, args.query4]
        .filter((query): query is string => typeof query === 'string')
        .map((query) => query.trim())
        .filter((query, index, all) => query.length > 0 && all.indexOf(query) === index)
        .slice(0, 4);
      const toolTrace = ensureToolCallTrace(trace, {
        toolCallId,
        toolName: 'wiki_rerank_research',
        args: { ...args, topK, candidateTopK: rerankCandidateTopK(topK) },
      });
      try {
        const searches = [];
        for (const query of queries) {
          const search = await client.search({
            query,
            topK: rerankCandidateTopK(topK),
          });
          searches.push(search);
          trace.queries.push(query);
          trace.searches.push(traceSearch(search));
        }

        const candidates = formatResearchForTool(
          searches,
          'bm25-rerank-candidates',
          rerankCandidateTopK(topK),
        );
        const reranked = await rerankWithQmd(
          buildRerankQuery(basePrompt, queries),
          candidates.hits.map((hit, index) => ({
            id: rerankDocumentId(hit, index),
            title: hit.title,
            text: formatRerankDocumentText(hit),
          })),
        );
        const hitsById = new Map(
          candidates.hits.map((hit, index) => [rerankDocumentId(hit, index), hit]),
        );
        const hits: RerankedResearchHit[] = reranked
          .map((ranked) => {
            const hit = hitsById.get(ranked.id);
            if (!hit) return undefined;
            return {
              ...hit,
              rerankScore: roundRerankScore(ranked.score),
            };
          })
          .filter((hit): hit is RerankedResearchHit => hit != null)
          .slice(0, 10);
        const formatted = {
          mode: 'bm25-rerank-research',
          queries,
          candidateCount: candidates.hits.length,
          hits,
        };
        const result = {
          content: [{ type: 'text' as const, text: JSON.stringify(formatted) }],
          details: formatted,
        };
        finishToolCallTrace(toolTrace, {
          status: 'ok',
          result: {
            contentText: result.content[0].text,
            contentTextChars: result.content[0].text.length,
            search: {
              mode: 'bm25-rerank-research',
              query: queries.join(' | '),
              hitCount: formatted.hits.length,
              topHits: formatted.hits.slice(0, 5).map((hit) => ({
                articleId: hit.articleId,
                chunkId: hit.chunkId,
                title: hit.title,
                score: hit.score,
                bm25Score: hit.bm25Score,
                sources: hit.sources,
                rerankScore: hit.rerankScore,
              })),
            },
          },
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
  };
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
    case 'agent-bm25-research':
    case 'agent-bm25-research-v2':
    case 'agent-bm25-rerank-research':
    case 'agent-bm25-research-read-required':
    case 'agent-bm25-research-smart-read':
    case 'agent-hybrid-research-smart-read':
      return [];
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
    case 'agent-bm25-research':
    case 'agent-bm25-research-v2':
    case 'agent-bm25-rerank-research':
    case 'agent-bm25-research-read-required':
    case 'agent-bm25-research-smart-read':
      return 'bm25';
    case 'agent-dense':
      return 'dense';
    case 'agent-hybrid':
    case 'agent-hybrid-research-smart-read':
    case 'agent-wiki':
      return 'hybrid';
    case 'agent-rg':
    case 'agent-literal':
      return 'literal';
    default:
      throw new Error(`candidate mode does not use Pi wiki tools: ${mode}`);
  }
}

function runSmartResearchSearch(params: {
  client: WikiAgentClient;
  mode: WikiSearchMode;
  query: string;
  topK: number;
}): Promise<WikiSearchResponse> {
  const request = { query: params.query, topK: params.topK };
  if (params.mode === 'hybrid') return params.client.hybridSearch(request);
  return params.client.search(request);
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

function rerankCandidateTopK(finalTopK: number): number {
  return Math.max(50, finalTopK * 10);
}

function buildRerankQuery(basePrompt: string, queries: string[]): string {
  return [
    'Find Wikipedia evidence that is directly relevant to answering this apocalypse-bench survival task.',
    '',
    'Original task:',
    basePrompt,
    '',
    'Search queries:',
    ...queries.map((query) => `- ${query}`),
    '',
    'Prefer pages whose title, heading, and snippet match the material, process, hazard, organism, symptom, tool, or measurement in the task. Ignore accidental lexical matches.',
  ].join('\n');
}

function rerankDocumentId(hit: ResearchSearchHit, index: number): string {
  return hit.chunkId ?? `${hit.articleId}:${index}`;
}

function formatRerankDocumentText(hit: ResearchSearchHit): string {
  return [
    `Title: ${hit.title}`,
    hit.headingPath && hit.headingPath.length > 0
      ? `Heading: ${hit.headingPath.join(' > ')}`
      : null,
    `Matched queries: ${hit.queries.join(' | ')}`,
    hit.snippet ? `Snippet: ${hit.snippet}` : null,
  ]
    .filter((line): line is string => line != null && line.length > 0)
    .join('\n');
}

function roundRerankScore(score: number): number {
  return Math.round(score * 10000) / 10000;
}

function formatResearchForTool(
  searches: WikiSearchResponse[],
  mode = 'bm25-research',
  maxHits = 10,
) {
  const byKey = new Map<string, ResearchSearchHit>();
  for (const search of searches) {
    const formatted = formatSearchForTool(search);
    formatted.hits.forEach((hit, index) => {
      const key = hit.chunkId ?? `${hit.articleId}:${hit.title}`;
      const existing = byKey.get(key);
      if (existing) {
        existing.queries.push(search.query);
        existing.firstRank = Math.min(existing.firstRank, index + 1);
        existing.score = Math.max(existing.score ?? 0, hit.score ?? 0);
        existing.bm25Score = Math.max(existing.bm25Score ?? 0, hit.bm25Score ?? 0);
        existing.sources = Array.from(
          new Set([...(existing.sources ?? []), ...(hit.sources ?? [])]),
        );
        return;
      }
      byKey.set(key, {
        ...hit,
        queries: [search.query],
        firstRank: index + 1,
      });
    });
  }

  const hits = Array.from(byKey.values())
    .sort(
      (left, right) =>
        right.queries.length - left.queries.length ||
        left.firstRank - right.firstRank ||
        (right.score ?? 0) - (left.score ?? 0) ||
        left.title.localeCompare(right.title),
    )
    .slice(0, maxHits);

  return {
    mode,
    queries: searches.map((search) => search.query),
    hits,
  };
}

function selectSmartReadHits(
  hits: ResearchSearchHit[],
  readCount: number,
): ResearchSearchHit[] {
  const selected: ResearchSearchHit[] = [];
  const seenArticles = new Set<string>();
  for (const hit of hits) {
    if (!hit.chunkId) continue;
    if (seenArticles.has(hit.articleId)) continue;
    selected.push(hit);
    seenArticles.add(hit.articleId);
    if (selected.length >= readCount) return selected;
  }
  for (const hit of hits) {
    if (!hit.chunkId || selected.some((selectedHit) => selectedHit.chunkId === hit.chunkId))
      continue;
    selected.push(hit);
    if (selected.length >= readCount) break;
  }
  return selected;
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
