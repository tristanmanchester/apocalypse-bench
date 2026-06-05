import { describe, expect, test } from 'vitest';
import type { AssistantMessage, Context, Tool } from '@earendil-works/pi-ai';
import {
  finalAnswerFromToolCall,
  normalizeParsedToolCall,
  parseTextToolCall,
  toTextToolProtocolContext,
} from '../src/adapters/pi/textToolProtocol';

const assistantToolCall: AssistantMessage = {
  role: 'assistant',
  content: [
    {
      type: 'toolCall',
      id: 'call-1',
      name: 'wiki_search',
      arguments: { query: 'boiling water disinfection', topK: 5 },
    },
  ],
  api: 'openai-completions',
  provider: 'openrouter',
  model: 'liquid/lfm-2.5-1.2b-thinking:free',
  usage: {
    input: 1,
    output: 1,
    cacheRead: 0,
    cacheWrite: 0,
    totalTokens: 2,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
  },
  stopReason: 'toolUse',
  timestamp: 1,
};

describe('Pi text tool protocol', () => {
  test('parses model-emitted tool call blocks', () => {
    expect(
      parseTextToolCall(
        '<tool_call>{"name":"wiki_search","arguments":{"query":"water filter","topK":3}}</tool_call>',
      ),
    ).toEqual({
      name: 'wiki_search',
      arguments: { query: 'water filter', topK: 3 },
    });
    expect(
      parseTextToolCall(
        '<|tool_call_start|>{"name":"wiki_read","arguments":{"chunkId":"c1"}}<|tool_call_end|>',
      ),
    ).toEqual({
      name: 'wiki_read',
      arguments: { chunkId: 'c1' },
    });
    expect(
      parseTextToolCall(
        "<|tool_call_start|>[wiki_search(query='water purification boiling pathogens', topK=5)]<|tool_call_end|>",
      ),
    ).toEqual({
      name: 'wiki_search',
      arguments: { query: 'water purification boiling pathogens', topK: 5 },
    });
    expect(
      parseTextToolCall(
        "<|tool_call_start|>[wiki_search(query='water purification boiling pathogens', topK=5),wiki_read(chunkId='chunk_789')]<|tool_call_end|>",
      ),
    ).toEqual({
      name: 'wiki_search',
      arguments: { query: 'water purification boiling pathogens', topK: 5 },
    });
    expect(
      parseTextToolCall(
        '<|tool_call>call:wiki_search{query:<|"|>water purification<|"|>,topK:5}<tool_call|><|tool_response>',
      ),
    ).toEqual({
      name: 'wiki_search',
      arguments: { query: 'water purification', topK: 5 },
    });
    expect(
      parseTextToolCall(
        '<tool_call><function=wiki_search><parameter=query>water purification</parameter><parameter=topK>5</parameter></function></tool_call>',
      ),
    ).toEqual({
      name: 'wiki_search',
      arguments: { query: 'water purification', topK: 5 },
    });
    expect(
      parseTextToolCall(
        '<|tool_call|>{"name":"wiki_read","arguments":{"chunkId":"c2","maxChars":1200}}<|/tool_call|>',
      ),
    ).toEqual({
      name: 'wiki_read',
      arguments: { chunkId: 'c2', maxChars: 1200 },
    });
    expect(
      parseTextToolCall(
        '<tool_call>[{"name":"wiki_search","arguments":{"query":"field surgery antisepsis","topK":4}}]</tool_call>',
      ),
    ).toEqual({
      name: 'wiki_search',
      arguments: { query: 'field surgery antisepsis', topK: 4 },
    });
    expect(
      parseTextToolCall(
        '<|channel|>commentary to=functions.wiki_search <|constrain|>json<|message|>{"query":"water purification","topK":5}<|call|>',
      ),
    ).toEqual({
      name: 'wiki_search',
      arguments: { query: 'water purification', topK: 5 },
    });
    expect(
      parseTextToolCall(
        '{"tool_calls":[{"function":{"name":"wiki_search","arguments":"{\\"query\\":\\"water purification\\",\\"topK\\":5}"}}]}',
      ),
    ).toEqual({
      name: 'wiki_search',
      arguments: { query: 'water purification', topK: 5 },
    });
    expect(
      parseTextToolCall(
        '{"function_call":{"name":"wiki_search","arguments":"{\\"query\\":\\"water purification\\",\\"topK\\":5}"}}',
      ),
    ).toEqual({
      name: 'wiki_search',
      arguments: { query: 'water purification', topK: 5 },
    });
    expect(
      parseTextToolCall(
        '{"output":[{"type":"function_call","name":"wiki_search","arguments":"{\\"query\\":\\"water purification\\",\\"topK\\":5}"}]}',
      ),
    ).toEqual({
      name: 'wiki_search',
      arguments: { query: 'water purification', topK: 5 },
    });
    expect(
      parseTextToolCall(
        '{"content":[{"type":"tool_use","name":"wiki_search","input":{"query":"water purification","topK":5}}]}',
      ),
    ).toEqual({
      name: 'wiki_search',
      arguments: { query: 'water purification', topK: 5 },
    });
    expect(
      parseTextToolCall(
        'functools[{"name":"wiki_search","arguments":{"query":"water purification","topK":5}}]',
      ),
    ).toEqual({
      name: 'wiki_search',
      arguments: { query: 'water purification', topK: 5 },
    });
  });

  test('covers documented tool-call formats for the nine wiki models', () => {
    const modelFormatFixtures: Array<[string, string]> = [
      [
        'liquid/lfm-2.5-1.2b-thinking:free',
        '<|tool_call_start|>[wiki_search(query="water purification", topK=5)]<|tool_call_end|>',
      ],
      [
        'google/gemma-4-26b-a4b-it',
        '<|tool_call>call:wiki_search{query:<|"|>water purification<|"|>,topK:5}<tool_call|>',
      ],
      [
        'google/gemma-4-31b-it',
        '<|tool_call>call:wiki_search{query:<|"|>water purification<|"|>,topK:5}<tool_call|>',
      ],
      [
        'ibm-granite/granite-4.1-8b',
        '<tool_call>{"name":"wiki_search","arguments":{"query":"water purification","topK":5}}</tool_call>',
      ],
      [
        'nvidia/nemotron-3-nano-30b-a3b',
        '<tool_call><function=wiki_search><parameter=query>water purification</parameter><parameter=topK>5</parameter></function></tool_call>',
      ],
      [
        'arcee-ai/trinity-mini',
        '{"content":[{"type":"tool_use","name":"wiki_search","input":{"query":"water purification","topK":5}}]}',
      ],
      [
        'ibm-granite/granite-4.0-h-micro',
        '<tool_call>{"name":"wiki_search","arguments":{"query":"water purification","topK":5}}</tool_call>',
      ],
      [
        'microsoft/phi-4-mini-instruct',
        'functools[{"name":"wiki_search","arguments":{"query":"water purification","topK":5}}]',
      ],
      [
        'openai/gpt-oss-20b',
        '<|channel|>commentary to=functions.wiki_search <|constrain|>json<|message|>{"query":"water purification","topK":5}<|call|>',
      ],
    ];

    expect(modelFormatFixtures).toHaveLength(9);
    for (const [model, text] of modelFormatFixtures) {
      expect(parseTextToolCall(text), model).toEqual({
        name: 'wiki_search',
        arguments: { query: 'water purification', topK: 5 },
      });
    }
  });

  test('parses and normalizes self-healed tool-call formats observed in runs', () => {
    expect(
      parseTextToolCall(
        'We need to produce tool calls. {"name":"wiki_hybrid_search","arguments":{"query":"potato propagation seed potatoes cutting into pieces sprouting","topK":5}}',
      ),
    ).toEqual({
      name: 'wiki_hybrid_search',
      arguments: {
        query: 'potato propagation seed potatoes cutting into pieces sprouting',
        topK: 5,
      },
    });

    const pseudoApiCall = parseTextToolCall(
      '__API Call Begin__ `search ( q = "sustainable agriculture practices" )` __Awaiting API Call Result(s)__',
    );
    expect(pseudoApiCall).toEqual({
      name: 'search',
      arguments: { q: 'sustainable agriculture practices' },
    });

    const tools = [
      {
        name: 'wiki_hybrid_search',
        description: 'Hybrid search.',
        parameters: { type: 'object', properties: {} },
      },
      {
        name: 'wiki_read',
        description: 'Read.',
        parameters: { type: 'object', properties: {} },
      },
    ] satisfies Tool[];
    expect(pseudoApiCall ? normalizeParsedToolCall(pseudoApiCall, tools) : undefined).toEqual({
      name: 'wiki_hybrid_search',
      arguments: { query: 'sustainable agriculture practices' },
    });

    const argumentOnlyRead = parseTextToolCall('{"chunkId":"633593:heat-treatment","maxChars":4000}');
    expect(argumentOnlyRead).toEqual({
      name: 'wiki_read',
      arguments: { chunkId: '633593:heat-treatment', maxChars: 4000 },
    });

    expect(
      parseTextToolCall(
        'We need to read the relevant chunk now.{"chunkId":"1557416:lead","maxChars":4000}',
      ),
    ).toEqual({
      name: 'wiki_read',
      arguments: { chunkId: '1557416:lead', maxChars: 4000 },
    });

    const argumentOnlySearch = parseTextToolCall(
      '{"query":"case hardening steel charcoal bone","topK":5}',
    );
    expect(argumentOnlySearch ? normalizeParsedToolCall(argumentOnlySearch, tools) : undefined)
      .toEqual({
        name: 'wiki_hybrid_search',
        arguments: { query: 'case hardening steel charcoal bone', topK: 5 },
      });

    const embeddedArgumentOnlySearch = parseTextToolCall(
      `We must call wiki_hybrid_search to fetch information. Let's do:{"query":"Kearny fallout meter electroscope","topK":5}`,
    );
    expect(
      embeddedArgumentOnlySearch
        ? normalizeParsedToolCall(embeddedArgumentOnlySearch, tools)
        : undefined,
    ).toEqual({
      name: 'wiki_hybrid_search',
      arguments: { query: 'Kearny fallout meter electroscope', topK: 5 },
    });
  });

  test('ignores malformed tool call blocks', () => {
    expect(parseTextToolCall('<tool_call>{"name":"wiki_search"</tool_call>')).toBeUndefined();
    expect(
      parseTextToolCall('<tool_call>{"name":"wiki_search","arguments":[]}</tool_call>'),
    ).toBeUndefined();
  });

  test('recognizes virtual final_answer calls as terminal answer text', () => {
    const call = parseTextToolCall(
      '<tool_call>{"name":"final_answer","arguments":{"answer":"Boil water and let it cool before drinking."}}</tool_call>',
    );
    expect(call).toEqual({
      name: 'final_answer',
      arguments: { answer: 'Boil water and let it cool before drinking.' },
    });
    expect(call ? finalAnswerFromToolCall(call) : undefined).toBe(
      'Boil water and let it cool before drinking.',
    );
    expect(
      finalAnswerFromToolCall({
        name: 'final_answer',
        arguments: { response: 'Use dry browns to fix a wet compost pile.' },
      }),
    ).toBe('Use dry browns to fix a wet compost pile.');
  });

  test('converts Pi tool turns to plain text and strips native tool declarations', () => {
    const context: Context = {
      systemPrompt: 'Answer survival questions.',
      tools: [
        {
          name: 'wiki_search',
          description: 'Search offline Wikipedia.',
          parameters: { type: 'object', properties: {} },
        },
      ],
      messages: [
        { role: 'user', content: 'How do I disinfect water?', timestamp: 0 },
        assistantToolCall,
        {
          role: 'toolResult',
          toolCallId: 'call-1',
          toolName: 'wiki_search',
          content: [{ type: 'text', text: '{"hits":[{"title":"Water purification"}]}' }],
          isError: false,
          timestamp: 2,
        },
      ],
    };

    const converted = toTextToolProtocolContext(context);
    expect(converted.tools).toBeUndefined();
    expect(converted.systemPrompt).toContain('Wikipedia tool protocol');
    expect(converted.systemPrompt).toContain('"name": "final_answer"');
    expect(converted.systemPrompt).toContain(
      '<tool_call>{"name":"final_answer","arguments":{"answer":"complete final answer text"}}</tool_call>',
    );
    expect(converted.messages[1]).toMatchObject({
      role: 'assistant',
      content: [
        {
          type: 'text',
          text: '<tool_call>{"name":"wiki_search","arguments":{"query":"boiling water disinfection","topK":5}}</tool_call>',
        },
      ],
      stopReason: 'stop',
    });
    expect(converted.messages[2]).toMatchObject({
      role: 'user',
      content: expect.stringContaining('<tool_result>'),
    });
  });

  test('truncates tool results in the model-facing text transcript', () => {
    const context: Context = {
      systemPrompt: 'Answer survival questions.',
      tools: [
        {
          name: 'wiki_read',
          description: 'Read offline Wikipedia.',
          parameters: { type: 'object', properties: {} },
        },
      ],
      messages: [
        {
          role: 'toolResult',
          toolCallId: 'read-1',
          toolName: 'wiki_read',
          content: [{ type: 'text', text: 'x'.repeat(3500) }],
          isError: false,
          timestamp: 2,
        },
      ],
    };

    const converted = toTextToolProtocolContext(context);
    const content = converted.messages[0]?.content;
    expect(typeof content).toBe('string');
    expect(content).toContain('[tool result truncated for context: 500 chars omitted]');
    expect(content).not.toContain('x'.repeat(3500));
  });
});
