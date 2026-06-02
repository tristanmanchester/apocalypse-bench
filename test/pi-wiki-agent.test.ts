import { beforeEach, describe, expect, test, vi } from 'vitest';
import type { ApocbenchConfig } from '../src/core/config/schema';

type MockTool = {
  name: string;
  execute: (toolCallId: string, input: Record<string, unknown>) => Promise<unknown>;
};

type MockAgentOptions = {
  initialState: {
    systemPrompt: string;
    model: { id: string; provider: string; compat?: unknown };
    tools: MockTool[];
    messages: unknown[];
  };
  beforeToolCall: (context: unknown) => Promise<unknown>;
};

type MockAgentState = MockAgentOptions['initialState'] & {
  isStreaming: boolean;
  pendingToolCalls: Set<string>;
};

const agentMock = vi.hoisted(() => ({
  options: undefined as MockAgentOptions | undefined,
  responseText: 'Use boiled water and verify disinfection.',
  thinkingText: undefined as string | undefined,
  promptCalls: 0,
}));

vi.mock('@earendil-works/pi-agent-core', () => ({
  Agent: class MockAgent {
    state: MockAgentState;

    constructor(options: MockAgentOptions) {
      agentMock.options = options;
      this.state = {
        ...options.initialState,
        messages: [],
        isStreaming: false,
        pendingToolCalls: new Set(),
      };
    }

    subscribe() {
      return () => undefined;
    }

    async prompt() {
      agentMock.promptCalls += 1;
      const content = [];
      if (agentMock.thinkingText) {
        content.push({ type: 'thinking', thinking: agentMock.thinkingText });
      }
      content.push({ type: 'text', text: agentMock.responseText });
      this.state.messages.push({
        role: 'assistant',
        content,
        api: 'openai-completions',
        provider: 'openrouter',
        model: this.state.model.id,
        responseId: 'gen-1',
        stopReason: 'stop',
        usage: {
          input: 10,
          output: 20,
          cacheRead: 0,
          cacheWrite: 0,
          totalTokens: 30,
          cost: {
            input: 0,
            output: 0,
            cacheRead: 0,
            cacheWrite: 0,
            total: 0.001,
          },
        },
        timestamp: Date.now(),
      });
    }

    async waitForIdle() {
      return undefined;
    }
  },
}));

const config: ApocbenchConfig = {
  run: {
    name: 'wiki-agent-test',
    datasetPath: './data/question_bank/sample.jsonl',
    outDir: './runs',
    resume: false,
    concurrency: { candidate: 1, judge: 1 },
  },
  candidate: { maxTokens: 512 },
  judge: {
    router: 'openrouter',
    model: 'openai/gpt-4o-mini',
    maxTokens: 512,
    structured: true,
  },
  routers: {
    openrouter: {
      apiKeyEnv: 'OPENROUTER_API_KEY',
      baseUrl: 'https://openrouter.ai/api/v1',
      default: { maxTokens: 512 },
    },
    ollama: {
      baseUrl: 'http://127.0.0.1:11434/api',
      default: {},
    },
  },
  wiki: {
    service: { baseUrl: 'http://127.0.0.1:8765' },
    corpus: { manifestId: 'corpus' },
    index: { manifestId: 'index' },
    limits: {
      searchTopK: 3,
      readMaxChars: 500,
      contextMaxChars: 500,
      maxToolCalls: 1,
      maxTurns: 4,
    },
  },
  models: [
    {
      id: 'liquid-agent',
      router: 'openrouter',
      model: 'liquid/lfm2-8b-a1b',
      provider: 'liquid',
      candidateMode: 'agent-bm25',
    },
  ],
};

describe('runPiWikiAgent', () => {
  beforeEach(() => {
    agentMock.options = undefined;
    agentMock.responseText = 'Use boiled water and verify disinfection.';
    agentMock.thinkingText = undefined;
    agentMock.promptCalls = 0;
  });

  test('creates a Pi OpenRouter agent with bounded wiki tools and trace recording', async () => {
    const { runPiWikiAgent } = await import('../src/adapters/pi/wikiAgent');
    const result = await runPiWikiAgent({
      config,
      modelEntry: config.models[0]!,
      basePrompt: 'How do I make river water safer?',
      mode: 'agent-bm25',
      wikiClient: {
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

    expect(result.completion).toContain('boiled water');
    expect(result.costUsd).toBe(0.001);
    expect(agentMock.options?.initialState.model.provider).toBe('openrouter');
    expect(
      (
        agentMock.options?.initialState.model.compat as
          | { openRouterRouting?: unknown }
          | undefined
      )?.openRouterRouting,
    ).toEqual({
      order: ['liquid'],
      allow_fallbacks: false,
    });
    expect(agentMock.options?.initialState.tools.map((tool) => tool.name)).toEqual([
      'wiki_search',
      'wiki_read',
    ]);

    const searchTool = agentMock.options?.initialState.tools[0];
    expect(searchTool).toBeDefined();
    if (!searchTool) return;
    await searchTool.execute('search-1', { query: 'water purification', topK: 2 });
    expect(result.retrievalTrace.searches[0]?.hits[0]?.title).toBe('Water purification');

    const readTool = agentMock.options?.initialState.tools[1];
    expect(readTool).toBeDefined();
    if (!readTool) return;
    await readTool.execute('read-1', { chunkId: 'c1', maxChars: 100 });
    expect(result.retrievalTrace.reads[0]).toMatchObject({
      title: 'Water purification',
      chunkId: 'c1',
      chars: 44,
      truncated: false,
    });
    expect(result.retrievalTrace.toolCallCount).toBe(2);
    expect(result.retrievalTrace.toolCalls[0]).toMatchObject({
      index: 1,
      toolCallId: 'search-1',
      toolName: 'wiki_search',
      arguments: {
        query: 'water purification',
        topK: 2,
      },
      status: 'ok',
      result: {
        search: {
          mode: 'bm25',
          query: 'water purification',
          hitCount: 1,
        },
      },
    });
    expect(result.retrievalTrace.toolCalls[0]?.result?.contentText).toContain(
      'Water purification',
    );
    expect(result.retrievalTrace.toolCalls[1]).toMatchObject({
      index: 2,
      toolCallId: 'read-1',
      toolName: 'wiki_read',
      arguments: {
        chunkId: 'c1',
        maxChars: 100,
      },
      status: 'ok',
      result: {
        read: {
          articleId: 'a1',
          chunkId: 'c1',
          title: 'Water purification',
          chars: 44,
          truncated: false,
        },
      },
    });
    expect(result.retrievalTrace.toolCalls[1]?.result?.contentText).toContain(
      'Boiling water',
    );

    await expect(agentMock.options?.beforeToolCall({})).resolves.toBeUndefined();
    await expect(agentMock.options?.beforeToolCall({})).resolves.toEqual({
      block: true,
      reason: 'wiki tool budget exceeded: maxToolCalls=1',
    });
  });

  test('strips native channel wrappers from final answer text', async () => {
    agentMock.responseText =
      '<|channel>thought\n<channel|>Use dry gypsum as a desiccant and measure the timed discharge rate.';

    const { runPiWikiAgent } = await import('../src/adapters/pi/wikiAgent');
    const result = await runPiWikiAgent({
      config,
      modelEntry: config.models[0]!,
      basePrompt: 'How do I build a meter?',
      mode: 'agent-bm25',
      wikiClient: {
        search: async ({ query }) => ({ mode: 'bm25', query, hits: [] }),
        semanticSearch: async () => ({ mode: 'dense', query: '', hits: [] }),
        hybridSearch: async () => ({ mode: 'hybrid', query: '', hits: [] }),
        literalSearch: async () => ({ mode: 'literal', query: '', hits: [] }),
        read: async () => ({
          pointer: { articleId: 'a1', chunkId: 'c1', title: 'Kearny fallout meter' },
          text: 'A dry chamber is required.',
          truncated: false,
        }),
      },
    });

    expect(result.completion).toBe(
      'Use dry gypsum as a desiccant and measure the timed discharge rate.',
    );
  });

  test('does not repair long actionable answers just because they include planning preamble', async () => {
    agentMock.responseText = [
      'We need to produce the final answer now. The practical procedure is as follows.',
      'Use clean containers, keep raw and treated materials separated, mark the working area, and assign one person to read back the steps before anyone starts.',
      'First prepare the materials and check that the measuring device is dry, clean, and intact.',
      'Then demonstrate the process slowly, have each trainee repeat it, and correct mistakes immediately.',
      'Use fixed visual marks rather than memory.',
      'Record the target value on the container, test the result before use, and stop if the test is outside the marked range.',
      'Before release, run a final check: confirm the container label, confirm the tool used for raw material never touches treated material, confirm the timing step was completed, and confirm the measured result is in range.',
      'If any check fails, discard or reprocess the batch rather than guessing.',
    ].join(' ');

    const { runPiWikiAgent } = await import('../src/adapters/pi/wikiAgent');
    const result = await runPiWikiAgent({
      config,
      modelEntry: config.models[0]!,
      basePrompt: 'How do I run a practical safety training?',
      mode: 'agent-bm25',
      wikiClient: wikiClientStub(),
    });

    expect(agentMock.promptCalls).toBe(1);
    expect(result.completion).toContain('The practical procedure is as follows');
  });

  test('exposes only the mode-specific search tool plus read', async () => {
    const { runPiWikiAgent } = await import('../src/adapters/pi/wikiAgent');
    const cases = [
      ['agent-bm25', 'wiki_search'],
      ['agent-dense', 'wiki_semantic_search'],
      ['agent-hybrid', 'wiki_hybrid_search'],
      ['agent-literal', 'wiki_literal_search'],
      ['agent-rg', 'wiki_literal_search'],
    ] as const;

    for (const [candidateMode, searchToolName] of cases) {
      agentMock.options = undefined;
      await runPiWikiAgent({
        config: {
          ...config,
          models: [
            {
              ...config.models[0]!,
              candidateMode,
            },
          ],
        },
        modelEntry: {
          ...config.models[0]!,
          candidateMode,
        },
        basePrompt: 'How do I make river water safer?',
        mode: candidateMode,
        wikiClient: wikiClientStub(),
      });

      const options = agentMock.options as MockAgentOptions | undefined;
      expect(options?.initialState.tools.map((tool: MockTool) => tool.name)).toEqual([
        searchToolName,
        'wiki_read',
      ]);
    }
  });

  test('exposes all production wiki tools for agent-wiki', async () => {
    const { runPiWikiAgent } = await import('../src/adapters/pi/wikiAgent');
    const calls = {
      search: vi.fn(async ({ query }: { query: string }) => ({
        mode: 'bm25' as const,
        query,
        hits: [],
      })),
      semanticSearch: vi.fn(async ({ query }: { query: string }) => ({
        mode: 'dense' as const,
        query,
        hits: [],
      })),
      hybridSearch: vi.fn(async ({ query }: { query: string }) => ({
        mode: 'hybrid' as const,
        query,
        hits: [],
      })),
      literalSearch: vi.fn(async ({ query }: { query: string }) => ({
        mode: 'literal' as const,
        query,
        hits: [],
      })),
      read: vi.fn(async () => ({
        pointer: {
          articleId: 'a1',
          chunkId: 'c1',
          title: 'Water purification',
        },
        text: 'Boiling water can inactivate many pathogens.',
        truncated: false,
      })),
    };

    const result = await runPiWikiAgent({
      config: {
        ...config,
        models: [
          {
            ...config.models[0]!,
            candidateMode: 'agent-wiki',
          },
        ],
      },
      modelEntry: {
        ...config.models[0]!,
        candidateMode: 'agent-wiki',
      },
      basePrompt: 'How do I make river water safer?',
      mode: 'agent-wiki',
      wikiClient: calls,
    });

    const options = agentMock.options as MockAgentOptions | undefined;
    expect(options?.initialState.systemPrompt).toContain(
      'You must use the local offline Wikipedia tools before answering.',
    );
    expect(options?.initialState.systemPrompt).toContain(
      'Use at least one Wikipedia search before answering',
    );
    expect(options?.initialState.systemPrompt).toContain(
      'Build search queries from distinctive nouns in the user prompt',
    );
    expect(options?.initialState.systemPrompt).toContain(
      'Read a chunk with wiki_read when search snippets are insufficient',
    );
    expect(options?.initialState.systemPrompt).toContain(
      'with concrete checks or verification steps when relevant',
    );
    expect(options?.initialState.systemPrompt).toContain(
      'You may use final_answer for deterministic submission',
    );
    expect(options?.initialState.tools.map((tool: MockTool) => tool.name)).toEqual([
      'wiki_hybrid_search',
      'wiki_search',
      'wiki_semantic_search',
      'wiki_literal_search',
      'wiki_read',
    ]);

    await options?.initialState.tools[0]?.execute('hybrid-1', {
      query: 'water treatment',
      topK: 2,
    });
    await options?.initialState.tools[1]?.execute('bm25-1', {
      query: 'boiling water',
      topK: 2,
    });
    await options?.initialState.tools[2]?.execute('dense-1', {
      query: 'making contaminated water potable',
      topK: 2,
    });
    await options?.initialState.tools[3]?.execute('literal-1', {
      query: 'inactivate pathogens',
      chunkId: 'c1',
    });

    expect(calls.hybridSearch).toHaveBeenCalledWith({
      query: 'water treatment',
      topK: 2,
      articleId: undefined,
      chunkId: undefined,
    });
    expect(calls.search).toHaveBeenCalledWith({
      query: 'boiling water',
      topK: 2,
      articleId: undefined,
      chunkId: undefined,
    });
    expect(calls.semanticSearch).toHaveBeenCalledWith({
      query: 'making contaminated water potable',
      topK: 2,
      articleId: undefined,
      chunkId: undefined,
    });
    expect(calls.literalSearch).toHaveBeenCalledWith({
      query: 'inactivate pathogens',
      topK: 3,
      articleId: undefined,
      chunkId: 'c1',
    });
    expect(result.retrievalTrace.searches.map((search) => search.mode)).toEqual([
      'hybrid',
      'bm25',
      'dense',
      'literal',
    ]);
    expect(result.retrievalTrace.toolCallCount).toBe(4);
    expect(result.retrievalTrace.toolCalls.map((call) => call.toolName)).toEqual([
      'wiki_hybrid_search',
      'wiki_search',
      'wiki_semantic_search',
      'wiki_literal_search',
    ]);
    expect(result.retrievalTrace.toolCalls.map((call) => call.status)).toEqual([
      'ok',
      'ok',
      'ok',
      'ok',
    ]);
    expect(result.retrievalTrace.toolCalls[3]?.arguments).toEqual({
      query: 'inactivate pathogens',
      topK: 3,
      chunkId: 'c1',
    });
  });

  test('prefers final text over thinking content when extracting the answer', async () => {
    agentMock.thinkingText =
      'Okay, I need to search and plan before answering the user scenario.';
    agentMock.responseText = 'Use dry gypsum as a desiccant and record leaf collapse over time.';

    const { runPiWikiAgent } = await import('../src/adapters/pi/wikiAgent');
    const result = await runPiWikiAgent({
      config,
      modelEntry: config.models[0]!,
      basePrompt: 'How do I build a meter?',
      mode: 'agent-bm25',
      wikiClient: wikiClientStub(),
    });

    expect(result.completion).toBe(
      'Use dry gypsum as a desiccant and record leaf collapse over time.',
    );
  });
});

function wikiClientStub() {
  return {
    search: async ({ query }: { query: string }) => ({
      mode: 'bm25' as const,
      query,
      hits: [],
    }),
    semanticSearch: async ({ query }: { query: string }) => ({
      mode: 'dense' as const,
      query,
      hits: [],
    }),
    hybridSearch: async ({ query }: { query: string }) => ({
      mode: 'hybrid' as const,
      query,
      hits: [],
    }),
    literalSearch: async ({ query }: { query: string }) => ({
      mode: 'literal' as const,
      query,
      hits: [],
    }),
    read: async () => ({
      pointer: {
        articleId: 'a1',
        chunkId: 'c1',
        title: 'Water purification',
      },
      text: 'Boiling water can inactivate many pathogens.',
      truncated: false,
    }),
  };
}
