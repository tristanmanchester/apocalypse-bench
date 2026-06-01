import { generateObject, NoObjectGeneratedError, type LanguageModel } from 'ai';
import type { z } from 'zod';
import type { ProviderOptions } from '@ai-sdk/provider-utils';
import { buildJudgeOutputSchemaWithRubricIds, type JudgeOutput } from './types';
import { computeBackoffMs, sleep } from '../../utils/backoff';
import { redactSecrets } from '../../utils/redaction';
import {
  classifyRetryError,
  computeRetryDelayMs,
  DEFAULT_RETRY_POLICY,
  shouldRetryWithinBudget,
} from './retryPolicy';

const _deps = {
  generateObject,
};

export type JudgeRequest = {
  model: LanguageModel;
  prompt?: string;
  messages?: Array<{ role: 'system' | 'user'; content: string }>;
  maxTokens: number;
  timeoutMs?: number | null;
  temperature: number | null | undefined;
  providerOptions?: ProviderOptions;
  /** Required rubric IDs - used to build explicit schema keys */
  rubricIds: string[];
};

export type JudgeRetryOptions = {
  maxRetries: number;
  baseMs: number;
  maxMs: number;
  maxTotalTimeMs?: number | null;
};

export type JudgeRetryEvent = {
  attempt: number;
  maxRetries: number;
  delayMs: number;
  reason: string;
  statusCode?: number;
};

type AbortableCall<T> = {
  call: () => Promise<T>;
  abort: () => void;
};

const REPAIR_PROMPT_MAX_CHARS = 6000;

function debugLogJudgeParsedObject(payload: unknown): void {
  if (!process.env.DEBUG_JUDGE) return;
  console.error('[DEBUG_JUDGE] Parsed object:', JSON.stringify(payload, null, 2));
}

function redactText(text: string): string {
  const redacted = redactSecrets(text);
  return typeof redacted === 'string' ? redacted : String(redacted);
}

function truncateBase(base: string, maxChars: number): string {
  if (base.length <= maxChars) return base;
  if (maxChars <= 3) return base.slice(0, maxChars);
  return `${base.slice(0, maxChars - 3)}...`;
}

function buildRepairPrompt(
  req: Pick<JudgeRequest, 'prompt' | 'messages'>,
  instruction: string,
): string {
  const base = req.messages
    ? req.messages.map((m) => redactText(m.content)).join('\n')
    : redactText(req.prompt as string);
  const suffix = `\n\n${instruction}`;
  if (suffix.length >= REPAIR_PROMPT_MAX_CHARS) {
    return suffix.slice(0, REPAIR_PROMPT_MAX_CHARS);
  }
  const maxBaseChars = REPAIR_PROMPT_MAX_CHARS - suffix.length;
  const clippedBase = truncateBase(base, maxBaseChars);
  return `${clippedBase}${suffix}`;
}

function createAbortableCall<T>(call: (signal: AbortSignal) => Promise<T>): AbortableCall<T> {
  const controller = new AbortController();
  return {
    call: () => call(controller.signal),
    abort: () => controller.abort(new Error('aborted')),
  };
}

async function judgeOnceWithRetry(
  req: JudgeRequest,
  retry: JudgeRetryOptions,
  onRetry?: (event: JudgeRetryEvent) => void,
): Promise<{ object: JudgeOutput; raw: unknown }> {
  const startedAtMs = Date.now();
  for (let attempt = 0; attempt <= retry.maxRetries; attempt++) {
    try {
      return await judgeOnce(req);
    } catch (err) {
      const retryDecision = classifyRetryError(err);
      if (!retryDecision.retryable || attempt === retry.maxRetries) throw err;
      const delayMs = computeRetryDelayMs({
        attempt,
        policy: retry,
        retryAfterMs: retryDecision.retryAfterMs,
      });
      if (
        !shouldRetryWithinBudget({
          startedAtMs,
          nowMs: Date.now(),
          delayMs,
          policy: retry,
        })
      ) {
        throw err;
      }
      onRetry?.({
        attempt: attempt + 1,
        maxRetries: retry.maxRetries,
        delayMs,
        reason: retryDecision.reason,
        statusCode: retryDecision.statusCode,
      });
      await sleep(delayMs);
    }
  }
  throw new Error('unreachable');
}

export async function judgeOnce(req: JudgeRequest): Promise<{
  object: JudgeOutput;
  raw: unknown;
}> {
  // Build schema with explicit rubric ID keys to ensure the model knows which keys are required
  const schema = buildJudgeOutputSchemaWithRubricIds(req.rubricIds);

  if (!req.messages && typeof req.prompt !== 'string') {
    throw new Error('judge request must include either prompt or messages');
  }

  const timeoutMs = req.timeoutMs ?? null;
  if (timeoutMs != null && Number.isFinite(timeoutMs) && timeoutMs > 0) {
    const abortable = createAbortableCall((abortSignal) =>
      _deps.generateObject({
        model: req.model,
        schema: schema as z.ZodTypeAny,
        ...(req.messages ? { messages: req.messages } : { prompt: req.prompt as string }),
        maxOutputTokens: req.maxTokens,
        temperature: req.temperature ?? undefined,
        providerOptions: req.providerOptions,
        abortSignal,
      }),
    );
    const timeoutId = setTimeout(() => abortable.abort(), timeoutMs);
    try {
      const result = await abortable.call();
      return {
        object: result.object as JudgeOutput,
        raw: {
          response: result.response,
          providerMetadata: result.providerMetadata,
          usage: result.usage,
          finishReason: result.finishReason,
          warnings: result.warnings,
          request: result.request,
        },
      };
    } finally {
      clearTimeout(timeoutId);
    }
  }

  const result = await _deps.generateObject({
    model: req.model,
    schema: schema as z.ZodTypeAny,
    ...(req.messages ? { messages: req.messages } : { prompt: req.prompt as string }),
    maxOutputTokens: req.maxTokens,
    temperature: req.temperature ?? undefined,
    providerOptions: req.providerOptions,
  });

  // Debug: log parsed object to see what the model is returning
  debugLogJudgeParsedObject(result.object);

  return {
    object: result.object as JudgeOutput,
    raw: {
      response: result.response,
      providerMetadata: result.providerMetadata,
      usage: result.usage,
      finishReason: result.finishReason,
      warnings: result.warnings,
      request: result.request,
    },
  };
}

export function __setJudgeDepsForTest(next: Partial<typeof _deps>): void {
  Object.assign(_deps, next);
}

export async function judgeWithRepairRetry(
  req: JudgeRequest,
  opts?: {
    retry?: Partial<JudgeRetryOptions>;
    onRetry?: (event: JudgeRetryEvent) => void;
  },
): Promise<{
  object: JudgeOutput;
  raw: unknown;
  didRepairRetry: boolean;
}> {
  const retry: JudgeRetryOptions = {
    ...DEFAULT_RETRY_POLICY,
    ...opts?.retry,
  };

  try {
    const { object, raw } = await judgeOnceWithRetry(req, retry, opts?.onRetry);
    return { object, raw, didRepairRetry: false };
  } catch (err) {
    if (classifyRetryError(err).retryable) throw err;
    if (!NoObjectGeneratedError.isInstance(err)) throw err;

    const retryPrompt = buildRepairPrompt(
      req,
      'IMPORTANT: Repair your output to be strictly valid JSON matching the schema. Output JSON only.',
    );

    const backoff = computeBackoffMs(0, { retries: 1, baseMs: 800, maxMs: 4000 });
    await sleep(backoff);
    const { object, raw } = await judgeOnceWithRetry(
      { ...req, prompt: retryPrompt, messages: undefined },
      retry,
      opts?.onRetry,
    );
    return { object, raw, didRepairRetry: true };
  }
}

function missingRubricIds(params: {
  judgeOutput: JudgeOutput;
  rubric: Array<{ id: string }>;
}): string[] {
  const scored = params.judgeOutput.rubric_scores ?? {};
  const missing: string[] = [];
  for (const item of params.rubric) {
    if (typeof scored[item.id] !== 'number') missing.push(item.id);
  }
  return missing;
}

export async function judgeWithRubricCompletenessRetry(
  req: JudgeRequest,
  params: { rubric: Array<{ id: string }> },
  opts?: { retry?: Partial<JudgeRetryOptions>; onRetry?: (event: JudgeRetryEvent) => void },
): Promise<{
  object: JudgeOutput;
  raw: unknown;
  didRepairRetry: boolean;
}> {
  const first = await judgeWithRepairRetry(req, opts);
  const missing = missingRubricIds({ judgeOutput: first.object, rubric: params.rubric });
  if (missing.length === 0) return first;

  const rubricIds = params.rubric.map((r) => r.id).join(', ');
  const retryPrompt = buildRepairPrompt(
    req,
    `IMPORTANT: Your prior output omitted rubric_scores for required ids. ` +
      `Output JSON only. rubric_scores must include ALL ids: ${rubricIds}. ` +
      `Do not omit any id.`,
  );

  const second = await judgeWithRepairRetry(
    { ...req, prompt: retryPrompt, messages: undefined },
    opts,
  );
  const stillMissing = missingRubricIds({
    judgeOutput: second.object,
    rubric: params.rubric,
  });
  if (stillMissing.length > 0) {
    throw new Error(
      `judge output missing rubric_scores for ids: ${stillMissing.join(', ')}`,
    );
  }

  return { ...second, didRepairRetry: true };
}

export function computeOverallScore(params: {
  judgeOutput: JudgeOutput;
  rubric: Array<{ id: string; weight: number; maxScore: number }>;
}): {
  overallScore: number;
  rubricScores: Record<string, number>;
} {
  const { judgeOutput, rubric } = params;
  const rubricScores: Record<string, number> = {};
  let sum = 0;

  for (const item of rubric) {
    const raw = judgeOutput.rubric_scores[item.id] ?? 0;
    const bounded = Math.max(0, Math.min(item.maxScore, raw));
    rubricScores[item.id] = bounded;
    sum += bounded * item.weight;
  }

  return { overallScore: judgeOutput.auto_fail ? 0 : sum, rubricScores };
}
