import {
  createAssistantMessageEventStream,
  streamSimple,
  type Api,
  type AssistantMessage,
  type Context,
  type Message,
  type Model,
  type SimpleStreamOptions,
  type StreamFunction,
  type Tool,
  type ToolCall,
  type ToolResultMessage,
  type Usage,
} from '@earendil-works/pi-ai';

const TOOL_CALL_PATTERNS = [
  /<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/i,
  /<\|tool_call_start\|>\s*([\s\S]*?)\s*<\|tool_call_end\|>/i,
  /<\|tool_call\|>\s*([\s\S]*?)\s*<\|\/tool_call\|>/i,
];
const GEMMA_TOOL_CALL_PATTERN =
  /<\|tool_call>call:([A-Za-z_][A-Za-z0-9_]*)\{([\s\S]*?)\}<tool_call\|>/i;
const HARMONY_TOOL_CALL_PATTERN =
  /to=(?:functions\.)?([A-Za-z_][A-Za-z0-9_]*)[\s\S]*?<\|message\|>\s*([\s\S]*?)\s*<\|call\|>/i;
const QWEN_FUNCTION_PATTERN =
  /<function=([A-Za-z_][A-Za-z0-9_]*)>\s*([\s\S]*?)\s*<\/function>/i;
const QWEN_PARAMETER_PATTERN =
  /<parameter=([A-Za-z_][A-Za-z0-9_]*)>\s*([\s\S]*?)\s*<\/parameter>/gi;
const PHI4_MINI_FUNCTIONS_PATTERN = /functools\[(.*?)\]/is;
const FINAL_ANSWER_TOOL_NAME = 'final_answer';
const MAX_TOOL_RESULT_CONTEXT_CHARS = 3000;

const EMPTY_USAGE: Usage = {
  input: 0,
  output: 0,
  cacheRead: 0,
  cacheWrite: 0,
  totalTokens: 0,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

export type ParsedTextToolCall = {
  name: string;
  arguments: Record<string, unknown>;
};

export function createTextToolProtocolStreamFn(): StreamFunction {
  return (model: Model<Api>, context: Context, options?: SimpleStreamOptions) => {
    const stream = createAssistantMessageEventStream();

    void (async () => {
      try {
        const textContext = toTextToolProtocolContext(context);
        const inner = streamSimple(model, textContext, options);
        let finalMessage: AssistantMessage | undefined;

        for await (const event of inner) {
          if (event.type === 'error') {
            stream.push(event);
            return;
          }
          if (event.type === 'done') {
            finalMessage = event.message;
          }
        }

        const assistant = finalMessage ?? (await inner.result());
        const parsed = parseTextToolCall(assistantText(assistant));
        if (!parsed) {
          stream.push({
            type: 'done',
            reason: assistant.stopReason === 'length' ? 'length' : 'stop',
            message: assistant,
          });
          return;
        }

        const normalized = normalizeParsedToolCall(parsed, context.tools ?? []);
        const finalAnswer = finalAnswerFromToolCall(normalized);
        if (finalAnswer != null) {
          stream.push({
            type: 'done',
            reason: 'stop',
            message: {
              ...assistant,
              content: [{ type: 'text', text: finalAnswer }],
              stopReason: 'stop',
            },
          });
          return;
        }

        const toolMessage = toToolCallMessage(assistant, normalized);
        stream.push({
          type: 'done',
          reason: 'toolUse',
          message: toolMessage,
        });
      } catch (error) {
        const message = createErrorMessage(model, error);
        stream.push({
          type: 'error',
          reason: 'error',
          error: message,
        });
      }
    })();

    return stream;
  };
}

export function toTextToolProtocolContext(context: Context): Context {
  return {
    systemPrompt: appendTextToolProtocol(context.systemPrompt, context.tools ?? []),
    messages: context.messages.map(toTextProtocolMessage),
  };
}

export function parseTextToolCall(text: string): ParsedTextToolCall | undefined {
  const gemmaCall = parseGemmaToolCall(text);
  if (gemmaCall) return gemmaCall;

  const harmonyCall = parseHarmonyToolCall(text);
  if (harmonyCall) return harmonyCall;

  const payload = extractToolCallPayload(text);
  const candidate = payload ?? text.trim();

  const jsonCall = parseJsonToolCall(candidate);
  if (jsonCall) return jsonCall;

  const argumentOnlyJsonCall = parseArgumentOnlyJsonToolCall(candidate);
  if (argumentOnlyJsonCall) return argumentOnlyJsonCall;

  const embeddedJsonCall = parseEmbeddedJsonToolCall(candidate);
  if (embeddedJsonCall) return embeddedJsonCall;

  const embeddedArgumentOnlyJsonCall = parseEmbeddedArgumentOnlyJsonToolCall(candidate);
  if (embeddedArgumentOnlyJsonCall) return embeddedArgumentOnlyJsonCall;

  const qwenCall = parseQwenXmlToolCall(candidate);
  if (qwenCall) return qwenCall;

  const phi4MiniCall = parsePhi4MiniToolCall(candidate);
  if (phi4MiniCall) return phi4MiniCall;

  const pseudoApiCall = parsePseudoApiToolCall(candidate);
  if (pseudoApiCall) return pseudoApiCall;

  return parsePythonicToolCall(candidate);
}

export function finalAnswerFromToolCall(call: ParsedTextToolCall): string | undefined {
  if (call.name !== FINAL_ANSWER_TOOL_NAME) return undefined;
  const value =
    call.arguments.answer ??
    call.arguments.final_answer ??
    call.arguments.response ??
    call.arguments.text;
  return typeof value === 'string' && value.trim().length > 0 ? value.trim() : '';
}

function parseJsonToolCall(payload: string): ParsedTextToolCall | undefined {
  let parsed: unknown;
  try {
    parsed = JSON.parse(payload);
  } catch {
    return undefined;
  }

  if (Array.isArray(parsed)) {
    for (const item of parsed) {
      if (!item || typeof item !== 'object' || Array.isArray(item)) continue;
      const call = parseJsonToolCallRecord(item as Record<string, unknown>);
      if (call) return call;
    }
    return undefined;
  }

  if (!parsed || typeof parsed !== 'object') return undefined;
  if (Array.isArray(parsed)) return undefined;
  return parseJsonToolCallRecord(parsed as Record<string, unknown>);
}

function parseArgumentOnlyJsonToolCall(payload: string): ParsedTextToolCall | undefined {
  let parsed: unknown;
  try {
    parsed = JSON.parse(payload.trim());
  } catch {
    return undefined;
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return undefined;
  const record = parsed as Record<string, unknown>;

  if (typeof record.chunkId === 'string' && record.chunkId.trim().length > 0) {
    return {
      name: 'wiki_read',
      arguments: normalizeArgumentOnlyRecord(record),
    };
  }

  if (typeof record.query === 'string' && record.query.trim().length > 0) {
    return {
      name: 'search',
      arguments: normalizeArgumentOnlyRecord(record),
    };
  }

  return undefined;
}

function normalizeArgumentOnlyRecord(record: Record<string, unknown>): Record<string, unknown> {
  const args = { ...record };
  if (typeof args.max_chars === 'number' && args.maxChars == null) {
    args.maxChars = args.max_chars;
    delete args.max_chars;
  }
  return args;
}

function parseEmbeddedJsonToolCall(payload: string): ParsedTextToolCall | undefined {
  if (!/"(?:name|tool|function_call|tool_calls)"\s*:/.test(payload)) return undefined;
  for (let idx = 0; idx < payload.length; idx += 1) {
    if (payload[idx] !== '{') continue;
    const end = findMatchingBrace(payload, idx);
    if (end < 0) continue;
    const candidate = payload.slice(idx, end + 1);
    if (!/"(?:arguments|parameters|args|input|function|tool_calls)"\s*:/.test(candidate)) {
      continue;
    }
    const call = parseJsonToolCall(candidate);
    if (call) return call;
  }
  return undefined;
}

function parseEmbeddedArgumentOnlyJsonToolCall(payload: string): ParsedTextToolCall | undefined {
  if (!/"(?:chunkId|query)"\s*:/.test(payload)) return undefined;
  for (let idx = 0; idx < payload.length; idx += 1) {
    if (payload[idx] !== '{') continue;
    const end = findMatchingBrace(payload, idx);
    if (end < 0) continue;
    const call = parseArgumentOnlyJsonToolCall(payload.slice(idx, end + 1));
    if (call) return call;
  }
  return undefined;
}

function parseJsonToolCallRecord(
  record: Record<string, unknown>,
): ParsedTextToolCall | undefined {
  const openAiCall = parseOpenAiJsonToolCall(record);
  if (openAiCall) return openAiCall;

  const nestedCall = parseNestedJsonToolCall(record);
  if (nestedCall) return nestedCall;

  const name =
    typeof record.name === 'string'
      ? record.name
      : typeof record.tool === 'string'
        ? record.tool
        : record.function_name;
  if (typeof name !== 'string' || name.trim().length === 0) return undefined;
  const args = record.arguments ?? record.parameters ?? record.args ?? record.input ?? {};
  const normalizedArgs = normalizeJsonArguments(args);
  if (!normalizedArgs) return undefined;

  return {
    name: name.trim(),
    arguments: normalizedArgs,
  };
}

function parseOpenAiJsonToolCall(
  record: Record<string, unknown>,
): ParsedTextToolCall | undefined {
  const toolCalls = record.tool_calls;
  if (Array.isArray(toolCalls) && toolCalls.length > 0) {
    const first = toolCalls[0] as Record<string, unknown> | undefined;
    const fn = first?.function as Record<string, unknown> | undefined;
    if (!fn) return undefined;
    return parseFunctionJsonRecord(fn);
  }

  const fn = record.function;
  if (fn && typeof fn === 'object' && !Array.isArray(fn)) {
    return parseFunctionJsonRecord(fn as Record<string, unknown>);
  }

  const functionCall = record.function_call;
  if (functionCall && typeof functionCall === 'object' && !Array.isArray(functionCall)) {
    return parseFunctionJsonRecord(functionCall as Record<string, unknown>);
  }

  return undefined;
}

function parseNestedJsonToolCall(
  record: Record<string, unknown>,
): ParsedTextToolCall | undefined {
  for (const key of ['output', 'content']) {
    const value = record[key];
    if (!Array.isArray(value)) continue;
    const call = parseJsonToolCall(JSON.stringify(value));
    if (call) return call;
  }
  return undefined;
}

function parseFunctionJsonRecord(
  record: Record<string, unknown>,
): ParsedTextToolCall | undefined {
  if (typeof record.name !== 'string' || record.name.trim().length === 0) return undefined;
  const args = normalizeJsonArguments(record.arguments ?? record.parameters ?? {});
  if (!args) return undefined;
  return {
    name: record.name.trim(),
    arguments: args,
  };
}

function normalizeJsonArguments(input: unknown): Record<string, unknown> | undefined {
  if (typeof input === 'string') {
    try {
      const parsed = JSON.parse(input) as unknown;
      return normalizeJsonArguments(parsed);
    } catch {
      return undefined;
    }
  }
  if (!input || typeof input !== 'object' || Array.isArray(input)) return undefined;
  return input as Record<string, unknown>;
}

function parseGemmaToolCall(text: string): ParsedTextToolCall | undefined {
  const match = GEMMA_TOOL_CALL_PATTERN.exec(text);
  if (!match) return undefined;
  return {
    name: match[1],
    arguments: parseGemmaArguments(match[2] ?? ''),
  };
}

function parseGemmaArguments(input: string): Record<string, unknown> {
  const args: Record<string, unknown> = {};
  for (const part of splitTopLevelArguments(input, ',')) {
    const separator = findUnquotedColon(part);
    if (separator <= 0) continue;
    const key = part.slice(0, separator).trim();
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(key)) continue;
    args[key] = parseGemmaValue(part.slice(separator + 1).trim());
  }
  return args;
}

function parseGemmaValue(input: string): unknown {
  const quoted = /^<\|"\|>([\s\S]*?)<\|"\|>$/.exec(input);
  if (quoted) return quoted[1] ?? '';
  return parsePythonicValue(input);
}

function parseHarmonyToolCall(text: string): ParsedTextToolCall | undefined {
  const match = HARMONY_TOOL_CALL_PATTERN.exec(text);
  if (!match) return undefined;
  const args = normalizeJsonArgumentsFromText(match[2] ?? '');
  if (!args) return undefined;
  return {
    name: match[1],
    arguments: args,
  };
}

function parseQwenXmlToolCall(text: string): ParsedTextToolCall | undefined {
  const match = QWEN_FUNCTION_PATTERN.exec(text);
  if (!match) return undefined;
  const args: Record<string, unknown> = {};
  const body = match[2] ?? '';
  for (const parameterMatch of body.matchAll(QWEN_PARAMETER_PATTERN)) {
    const key = parameterMatch[1];
    const rawValue = parameterMatch[2]?.trim() ?? '';
    args[key] = normalizeQwenParameterValue(rawValue);
  }
  return {
    name: match[1],
    arguments: args,
  };
}

function parsePhi4MiniToolCall(text: string): ParsedTextToolCall | undefined {
  const match = PHI4_MINI_FUNCTIONS_PATTERN.exec(text);
  if (!match) return undefined;
  return parseJsonToolCall(`[${match[1]}]`);
}

function parsePseudoApiToolCall(text: string): ParsedTextToolCall | undefined {
  const match =
    /(?:__API Call Begin__|API Call Begin|Awaiting API Call Result\(s\))[\s\S]*?`?\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(([\s\S]*?)\)\s*`?/i.exec(
      text,
    );
  if (!match) return undefined;
  return parsePythonicToolCall(`${match[1]}(${match[2] ?? ''})`);
}

function normalizeJsonArgumentsFromText(input: string): Record<string, unknown> | undefined {
  return normalizeJsonArguments(input.trim());
}

function normalizeQwenParameterValue(input: string): unknown {
  const jsonValue = tryParseJson(input);
  if (jsonValue !== undefined) return jsonValue;
  return parsePythonicValue(input) ?? input;
}

function tryParseJson(input: string): unknown {
  if (!/^(\[|\{|"|-?\d|true\b|false\b|null\b)/.test(input)) return undefined;
  try {
    return JSON.parse(input) as unknown;
  } catch {
    return undefined;
  }
}

function parsePythonicToolCall(payload: string): ParsedTextToolCall | undefined {
  const trimmed = payload.trim().replace(/^\[\s*/, '');
  const match = /^([A-Za-z_][A-Za-z0-9_]*)\s*\(/.exec(trimmed);
  if (!match) return undefined;

  const name = match[1];
  const openParenIndex = trimmed.indexOf('(', name.length);
  const closeParenIndex = findMatchingParen(trimmed, openParenIndex);
  if (closeParenIndex < 0) return undefined;

  const argText = trimmed.slice(openParenIndex + 1, closeParenIndex).trim();
  const args: Record<string, unknown> = {};
  if (argText.length === 0) return { name, arguments: args };

  for (const part of splitPythonicArguments(argText)) {
    const separator = findUnquotedEquals(part);
    if (separator <= 0) return undefined;
    const key = part.slice(0, separator).trim();
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(key)) return undefined;
    const value = parsePythonicValue(part.slice(separator + 1).trim());
    if (value === undefined) return undefined;
    args[key] = value;
  }

  return { name, arguments: args };
}

function findMatchingParen(input: string, openParenIndex: number): number {
  let quote: '"' | "'" | undefined;
  let escaped = false;
  let depth = 0;

  for (let idx = openParenIndex; idx < input.length; idx += 1) {
    const char = input[idx];
    if (escaped) {
      escaped = false;
      continue;
    }
    if (char === '\\' && quote) {
      escaped = true;
      continue;
    }
    if ((char === '"' || char === "'") && !quote) {
      quote = char;
      continue;
    }
    if (char === quote) {
      quote = undefined;
      continue;
    }
    if (quote) continue;
    if (char === '(') depth += 1;
    if (char === ')') {
      depth -= 1;
      if (depth === 0) return idx;
    }
  }

  return -1;
}

function findMatchingBrace(input: string, openBraceIndex: number): number {
  let quote: '"' | undefined;
  let escaped = false;
  let depth = 0;

  for (let idx = openBraceIndex; idx < input.length; idx += 1) {
    const char = input[idx];
    if (escaped) {
      escaped = false;
      continue;
    }
    if (char === '\\' && quote) {
      escaped = true;
      continue;
    }
    if (char === '"' && !quote) {
      quote = char;
      continue;
    }
    if (char === quote) {
      quote = undefined;
      continue;
    }
    if (quote) continue;
    if (char === '{') depth += 1;
    if (char === '}') {
      depth -= 1;
      if (depth === 0) return idx;
    }
  }

  return -1;
}

function splitPythonicArguments(input: string): string[] {
  return splitTopLevelArguments(input, ',');
}

function splitTopLevelArguments(input: string, delimiter: string): string[] {
  const parts: string[] = [];
  let current = '';
  let quote: '"' | "'" | undefined;
  let escaped = false;

  for (const char of input) {
    if (escaped) {
      current += char;
      escaped = false;
      continue;
    }
    if (char === '\\' && quote) {
      current += char;
      escaped = true;
      continue;
    }
    if ((char === '"' || char === "'") && !quote) {
      quote = char;
      current += char;
      continue;
    }
    if (char === quote) {
      quote = undefined;
      current += char;
      continue;
    }
    if (char === delimiter && !quote) {
      parts.push(current.trim());
      current = '';
      continue;
    }
    current += char;
  }

  if (quote) return [];
  if (current.trim().length > 0) parts.push(current.trim());
  return parts;
}

function findUnquotedEquals(input: string): number {
  return findUnquotedCharacter(input, '=');
}

function findUnquotedColon(input: string): number {
  return findUnquotedCharacter(input, ':');
}

function findUnquotedCharacter(input: string, needle: string): number {
  let quote: '"' | "'" | undefined;
  let escaped = false;
  for (let idx = 0; idx < input.length; idx += 1) {
    const char = input[idx];
    if (escaped) {
      escaped = false;
      continue;
    }
    if (char === '\\' && quote) {
      escaped = true;
      continue;
    }
    if ((char === '"' || char === "'") && !quote) {
      quote = char;
      continue;
    }
    if (char === quote) {
      quote = undefined;
      continue;
    }
    if (char === needle && !quote) return idx;
  }
  return -1;
}

function parsePythonicValue(input: string): unknown {
  if (input.length === 0) return undefined;
  if (
    (input.startsWith("'") && input.endsWith("'")) ||
    (input.startsWith('"') && input.endsWith('"'))
  ) {
    return input
      .slice(1, -1)
      .replace(/\\\\/g, '\\')
      .replace(/\\'/g, "'")
      .replace(/\\"/g, '"')
      .replace(/\\n/g, '\n')
      .replace(/\\t/g, '\t');
  }
  if (/^-?\d+(\.\d+)?$/.test(input)) return Number(input);
  if (input === 'True' || input === 'true') return true;
  if (input === 'False' || input === 'false') return false;
  if (input === 'None' || input === 'null') return null;
  return undefined;
}

function appendTextToolProtocol(systemPrompt: string | undefined, tools: Tool[]): string {
  const searchToolName = tools.find((tool) => tool.name !== 'wiki_read')?.name ?? 'wiki_search';
  const toolDefinitions = JSON.stringify(
    [
      ...tools.map((tool) => ({
        name: tool.name,
        description: tool.description,
        parameters: tool.parameters,
      })),
      {
        name: FINAL_ANSWER_TOOL_NAME,
        description:
          'Optionally submit the final benchmark answer after completing the required Wikipedia searches and reads. Plain text final answers are also accepted.',
        parameters: {
          type: 'object',
          properties: {
            answer: {
              type: 'string',
              description: 'The complete final answer for the benchmark judge.',
            },
          },
          required: ['answer'],
        },
      },
    ],
    null,
    2,
  );

  return [
    systemPrompt,
    '',
    'Wikipedia tool protocol:',
    'Tool definitions:',
    toolDefinitions,
    'When you need a tool, output exactly one tool call and no other text in this form:',
    `<tool_call>{"name":"${searchToolName}","arguments":{"query":"example query","topK":5}}</tool_call>`,
    'or:',
    '<tool_call>{"name":"wiki_read","arguments":{"chunkId":"chunk-id-from-search","maxChars":4000}}</tool_call>',
    'When you are ready to answer, plain text is accepted. For deterministic extraction, you may instead output exactly one final_answer call and no other text:',
    '<tool_call>{"name":"final_answer","arguments":{"answer":"complete final answer text"}}</tool_call>',
    'The neutral JSON form above is preferred. If your chat template is trained to use a native tool-call format, the harness also accepts documented Liquid, Gemma, Qwen/Nemotron, Phi funtools, Harmony, OpenAI Responses/Chat, and Anthropic tool_use text calls.',
    'After a wiki tool result is returned, continue reasoning from the result. Use final_answer only for the completed final answer; if that format is awkward, answer normally in plain text.',
  ]
    .filter((part) => part != null && String(part).trim().length > 0)
    .join('\n');
}

export function normalizeParsedToolCall(
  call: ParsedTextToolCall,
  tools: Tool[],
): ParsedTextToolCall {
  if (call.name === FINAL_ANSWER_TOOL_NAME) return call;
  const availableToolNames = tools.map((tool) => tool.name);
  const searchToolName =
    availableToolNames.find((name) => name !== 'wiki_read' && name.includes('search')) ??
    'wiki_search';
  let name = call.name;
  if (['search', 'wiki_search_tool', 'wikipedia_search'].includes(name)) {
    name = searchToolName;
  } else if (['read', 'wiki_read_tool', 'wikipedia_read'].includes(name)) {
    name = 'wiki_read';
  }

  const args = { ...call.arguments };
  if (args.query == null && typeof args.q === 'string') {
    args.query = args.q;
    delete args.q;
  }
  if (args.topK == null && typeof args.k === 'number') args.topK = args.k;
  if (args.topK == null && typeof args.limit === 'number') args.topK = args.limit;
  if (args.maxChars == null && typeof args.max_chars === 'number') {
    args.maxChars = args.max_chars;
    delete args.max_chars;
  }

  return { name, arguments: args };
}

function extractToolCallPayload(text: string): string | undefined {
  for (const pattern of TOOL_CALL_PATTERNS) {
    const match = pattern.exec(text);
    if (match?.[1]) return match[1].trim();
  }
  return undefined;
}

function toTextProtocolMessage(message: Message): Message {
  if (message.role === 'user') {
    return {
      ...message,
      content: stringifyUserContent(message.content),
    };
  }

  if (message.role === 'assistant') {
    return {
      ...message,
      content: [{ type: 'text', text: stringifyAssistantContent(message) }],
      stopReason: message.stopReason === 'toolUse' ? 'stop' : message.stopReason,
    };
  }

  return toolResultToUserMessage(message);
}

function toolResultToUserMessage(message: ToolResultMessage): Message {
  const payload = truncateToolResultForContext(stringifyTextContent(message.content));
  const status = message.isError ? 'error' : 'ok';
  return {
    role: 'user',
    content: [
      `Tool result (${status}) for ${message.toolName}, id ${message.toolCallId}:`,
      '<tool_result>',
      payload,
      '</tool_result>',
    ].join('\n'),
    timestamp: message.timestamp,
  };
}

function truncateToolResultForContext(payload: string): string {
  if (payload.length <= MAX_TOOL_RESULT_CONTEXT_CHARS) return payload;
  return [
    payload.slice(0, MAX_TOOL_RESULT_CONTEXT_CHARS),
    `[tool result truncated for context: ${payload.length - MAX_TOOL_RESULT_CONTEXT_CHARS} chars omitted]`,
  ].join('\n');
}

function stringifyUserContent(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return String(content);
  return stringifyTextContent(content);
}

function stringifyAssistantContent(message: AssistantMessage): string {
  const parts = message.content.map((content) => {
    switch (content.type) {
      case 'text':
        return content.text;
      case 'thinking':
        return content.thinking;
      case 'toolCall':
        return `<tool_call>${JSON.stringify({
          name: content.name,
          arguments: content.arguments,
        })}</tool_call>`;
    }
  });
  return parts.filter((part) => part.trim().length > 0).join('\n');
}

function stringifyTextContent(content: unknown[]): string {
  return content
    .map((item) => {
      if (!item || typeof item !== 'object') return '';
      const record = item as Record<string, unknown>;
      if (record.type === 'text' && typeof record.text === 'string') return record.text;
      if (record.type === 'image') return '[image omitted from text tool protocol]';
      return '';
    })
    .filter((part) => part.trim().length > 0)
    .join('\n');
}

function assistantText(message: AssistantMessage): string {
  return message.content
    .map((content) => {
      if (content.type === 'text') return content.text;
      if (content.type === 'thinking') return content.thinking;
      return '';
    })
    .join('\n');
}

function toToolCallMessage(
  message: AssistantMessage,
  parsed: ParsedTextToolCall,
): AssistantMessage {
  const toolCall: ToolCall = {
    type: 'toolCall',
    id: `text-tool-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    name: parsed.name,
    arguments: parsed.arguments,
  };
  return {
    ...message,
    content: [toolCall],
    stopReason: 'toolUse',
  };
}

function createErrorMessage(model: Model<Api>, error: unknown): AssistantMessage {
  return {
    role: 'assistant',
    content: [],
    api: model.api,
    provider: model.provider,
    model: model.id,
    usage: EMPTY_USAGE,
    stopReason: 'error',
    errorMessage: error instanceof Error ? error.message : String(error),
    timestamp: Date.now(),
  };
}
