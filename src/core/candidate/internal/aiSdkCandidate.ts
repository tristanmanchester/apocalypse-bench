import { generateText } from 'ai';
import type { LanguageModel } from 'ai';
import type { ProviderOptions } from '@ai-sdk/provider-utils';

import { runPiWikiAgent } from '../../../adapters/pi/wikiAgent';
import type { ApocbenchConfig } from '../../config/schema';
import {
  isAgentCandidateMode,
  isWikiCandidateMode,
  toOpenRouterProviderParam,
} from '../../config/schema';
import type { DatasetLine } from '../../dataset/schema';
import { buildCandidatePrompt } from '../../prompts/candidatePrompt';
import { CANDIDATE_SYSTEM_PROMPT } from '../../prompts/systemPrompts';
import type { WikiClient } from '../../wiki/client';
import { buildWikiGroundedCandidatePrompt } from '../../wiki/rag';
import { sleep } from '../../../utils/backoff';
import {
  normalizeOpenRouterUsageFromProviderMetadata,
  normalizeUsage,
} from '../../runner/openrouterUsage';
import {
  classifyRetryError,
  computeRetryDelayMs,
  shouldRetryWithinBudget,
  type RetryPolicy,
} from '../../runner/retryPolicy';
import { extractOpenRouterCost } from '../../runner/budget';
import type { CandidateExecutionResult } from '../types';

type ModelEntry = ApocbenchConfig['models'][number];
type GenerateTextArgs = Parameters<typeof generateText>[0];
type TextMessages = Array<
  | { role: 'system'; content: string }
  | { role: 'user'; content: string }
>;

type CandidateRouterDefaults = {
  temperature?: number | null;
  maxTokens?: number;
  timeoutMs?: number;
};

type AbortableCall<T> = {
  call: () => Promise<T>;
  abort: () => void;
};

function createAbortableCall<T>(call: (signal: AbortSignal) => Promise<T>): AbortableCall<T> {
  const controller = new AbortController();
  return {
    call: () => call(controller.signal),
    abort: () => controller.abort(new Error('aborted')),
  };
}

export async function executeAiSdkCandidate(params: {
  config: ApocbenchConfig;
  modelEntry: ModelEntry;
  model: LanguageModel;
  question: DatasetLine;
  wikiClient?: WikiClient;
  retryPolicy: RetryPolicy;
  onRetry?: (event: {
    attempt: number;
    maxRetries: number;
    delayMs: number;
    reason: string;
    statusCode?: number;
  }) => void;
}): Promise<CandidateExecutionResult> {
  const { config, modelEntry, model, question, wikiClient, retryPolicy, onRetry } = params;
  const candidateMode = modelEntry.candidateMode ?? 'direct';
  const basePrompt = buildCandidatePrompt(question);
  let effectivePrompt = basePrompt;
  let retrievalTrace: CandidateExecutionResult['retrievalTrace'];
  const startedAtMs = Date.now();

  if (isWikiCandidateMode(candidateMode)) {
    if (!config.wiki || !wikiClient) {
      throw new Error(`missing wiki config/client for candidateMode: ${candidateMode}`);
    }

    if (isAgentCandidateMode(candidateMode)) {
      const agentResult = await runPiWikiAgent({
        config,
        modelEntry,
        mode: candidateMode,
        basePrompt,
        wikiClient,
      });
      return {
        prompt: basePrompt,
        completion: agentResult.completion,
        retrievalTrace: agentResult.retrievalTrace,
        generationId: agentResult.generationId,
        metrics: {
          latencyMs: Date.now() - startedAtMs,
          usage: agentResult.usage,
          costUsd: agentResult.costUsd,
        },
      };
    } else {
      const wikiContext = await buildWikiGroundedCandidatePrompt({
        question,
        basePrompt,
        mode: candidateMode,
        wiki: config.wiki,
        client: wikiClient,
      });
      effectivePrompt = wikiContext.prompt;
      retrievalTrace = wikiContext.trace;
    }
  }

  const routerDefaults = getCandidateRouterDefaults(config, modelEntry);
  const result = await generateTextWithRetry({
    timeoutMs:
      modelEntry.params?.timeoutMs ??
      routerDefaults.timeoutMs ??
      null,
    retryPolicy,
    call: {
      model,
      messages: [
        { role: 'system', content: CANDIDATE_SYSTEM_PROMPT },
        { role: 'user', content: effectivePrompt },
      ] as TextMessages,
      temperature:
        modelEntry.params?.temperature ??
        routerDefaults.temperature ??
        undefined,
      maxOutputTokens: getCandidateMaxOutputTokens(config, modelEntry),
      providerOptions: buildCandidateProviderOptions(config, modelEntry),
    },
    onRetry,
  });

  const candidateCostUsd = extractOpenRouterCost(result);
  const candidateOr = normalizeOpenRouterUsageFromProviderMetadata(
    (result as { providerMetadata?: unknown } | null | undefined)?.providerMetadata,
  );
  const costUsd = candidateOr.costUsd ?? candidateCostUsd ?? undefined;

  return {
    prompt: effectivePrompt,
    completion: result.text,
    retrievalTrace,
    generationId: extractOpenRouterGenerationId(result) ?? undefined,
    metrics: {
      latencyMs: Date.now() - startedAtMs,
      usage: candidateOr.usage ?? normalizeUsage(result.usage),
      costUsd,
    },
  };
}

async function generateTextWithRetry(params: {
  call: Omit<GenerateTextArgs, 'abortSignal'>;
  timeoutMs?: number | null;
  retryPolicy: RetryPolicy;
  onRetry?: (event: {
    attempt: number;
    maxRetries: number;
    delayMs: number;
    reason: string;
    statusCode?: number;
  }) => void;
}): Promise<Awaited<ReturnType<typeof generateText>>> {
  const { call, timeoutMs, retryPolicy, onRetry } = params;
  const startedAtMs = Date.now();
  for (let attempt = 0; attempt <= retryPolicy.maxRetries; attempt++) {
    try {
      if (timeoutMs != null && Number.isFinite(timeoutMs) && timeoutMs > 0) {
        const abortable = createAbortableCall((abortSignal) => {
          return generateText({
            ...(call as GenerateTextArgs),
            abortSignal,
          });
        });
        const timeoutId = setTimeout(() => abortable.abort(), timeoutMs);
        try {
          return await abortable.call();
        } finally {
          clearTimeout(timeoutId);
        }
      }

      return await generateText(call as GenerateTextArgs);
    } catch (err) {
      const retryDecision = classifyRetryError(err);
      if (!retryDecision.retryable || attempt === retryPolicy.maxRetries) throw err;
      const delayMs = computeRetryDelayMs({
        attempt,
        policy: retryPolicy,
        retryAfterMs: retryDecision.retryAfterMs,
      });
      if (
        !shouldRetryWithinBudget({
          startedAtMs,
          nowMs: Date.now(),
          delayMs,
          policy: retryPolicy,
        })
      ) {
        throw err;
      }
      onRetry?.({
        attempt: attempt + 1,
        maxRetries: retryPolicy.maxRetries,
        delayMs,
        reason: retryDecision.reason,
        statusCode: retryDecision.statusCode,
      });
      await sleep(delayMs);
    }
  }
  throw new Error('unreachable');
}

function buildCandidateProviderOptions(
  config: ApocbenchConfig,
  modelEntry: ModelEntry,
): ProviderOptions | undefined {
  return (modelEntry.router === 'openrouter'
    ? {
        openrouter: {
          ...(modelEntry.routing
            ? { provider: toOpenRouterProviderParam(modelEntry.routing) }
            : modelEntry.provider
              ? {
                  provider: {
                    order: [modelEntry.provider],
                    allow_fallbacks: false,
                  },
                }
              : {}),
        },
      }
    : modelEntry.router === 'ollama'
      ? {
          ollama: {
            options: {
              num_predict:
                config.candidate?.maxTokens ??
                modelEntry.params?.maxTokens ??
                config.routers[modelEntry.router].default.maxTokens ??
                undefined,
            },
          },
        }
      : undefined) as ProviderOptions | undefined;
}

function getCandidateRouterDefaults(
  config: ApocbenchConfig,
  modelEntry: ModelEntry,
): CandidateRouterDefaults {
  if (modelEntry.router === 'openai-compatible') {
    const routerConfig = config.routers.openaiCompatible;
    if (!routerConfig) {
      throw new Error('missing router config: routers.openaiCompatible');
    }
    return routerConfig.default;
  }

  return config.routers[modelEntry.router].default;
}

function getCandidateMaxOutputTokens(
  config: ApocbenchConfig,
  modelEntry: ModelEntry,
): number | undefined {
  if (modelEntry.router === 'ollama') return undefined;
  const defaults = getCandidateRouterDefaults(config, modelEntry);
  return (
    config.candidate?.maxTokens ??
    modelEntry.params?.maxTokens ??
    defaults.maxTokens ??
    undefined
  );
}

function extractOpenRouterGenerationId(result: unknown): string | null {
  const responseId = (result as { response?: { id?: unknown } } | null | undefined)
    ?.response?.id;
  if (typeof responseId === 'string' && responseId.length > 0) return responseId;

  const topLevelId = (result as { id?: unknown } | null | undefined)?.id;
  if (typeof topLevelId === 'string' && topLevelId.length > 0) return topLevelId;

  const providerMetadataId = (
    result as
      | { providerMetadata?: { openrouter?: { id?: unknown } } }
      | null
      | undefined
  )?.providerMetadata?.openrouter?.id;
  if (typeof providerMetadataId === 'string' && providerMetadataId.length > 0)
    return providerMetadataId;

  return null;
}
