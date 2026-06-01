import fs from 'node:fs';
import path from 'node:path';
import PQueue from 'p-queue';
import { generateText } from 'ai';
import type { LanguageModel } from 'ai';
import type { ProviderOptions } from '@ai-sdk/provider-utils';

import type { ApocbenchConfig } from '../config/schema';
import type { DatasetLine } from '../dataset/schema';
import { buildCandidatePrompt } from '../prompts/candidatePrompt';
import { buildJudgePrompt } from '../prompts/judgePrompt';
import { CANDIDATE_SYSTEM_PROMPT, JUDGE_SYSTEM_PROMPT } from '../prompts/systemPrompts';
import { aggregateModel } from '../scoring/aggregate';
import { computeOverallScore, judgeWithRubricCompletenessRetry } from './judge';
import { makeRunId, promptTemplateHash } from './runId';
import type { CandidateMetrics, JudgeOutput } from './types';
import { sha256FileHex } from '../../utils/hash';
import { redactSecrets } from '../../utils/redaction';
import { sleep } from '../../utils/backoff';
import { writeJson } from '../../reports/json/exports';
import { renderHtmlReport } from '../../reports/html/renderHtml';
import {
  ensureRunStarted,
  insertRunQuestions,
  isRunResultDone,
  listRunResults,
  openRunnerDb,
  type RunnerDb,
  updateRunStatusForRun,
  upsertRunResult,
} from './persistence';
import {
  normalizeOpenRouterUsageFromProviderMetadata,
  normalizeUsage,
  type NormalizedUsage,
} from './openrouterUsage';
import { toOpenRouterProviderParam } from '../config/schema';
import {
  extractOpenRouterCost,
  initBudgetState,
  isBudgetExceeded,
  recordSpend,
  type BudgetState,
} from './budget';
import { maybeEmitOpenRouterGenerationMetrics } from './generationMetrics';
import {
  classifyRetryError,
  computeRetryDelayMs,
  DEFAULT_RETRY_POLICY,
  shouldRetryWithinBudget,
  type RetryPolicy,
} from './retryPolicy';

export type RunnerEvent =
  | { type: 'run_started'; runId: string; startedAtMs: number }
  | { type: 'question_started'; runId: string; modelId: string; questionId: string }
  | {
      type: 'generation_metrics';
      runId: string;
      modelId: string;
      questionId: string;
      generationId?: string;
      tps?: number;
      generationTimeMs?: number;
      tokens?: { prompt: number; completion: number; total: number };
    }
  | {
      type: 'question_completed';
      runId: string;
      modelId: string;
      questionId: string;
      overallScore: number;
      latencyMs?: number;
      usage?: unknown;
      costUsd?: number;
    }
  | {
      type: 'question_failed';
      runId: string;
      modelId: string;
      questionId: string;
      stage: 'candidate' | 'judge';
      message: string;
      latencyMs?: number;
      usage?: unknown;
      costUsd?: number;
    }
  | {
      type: 'request_retry';
      runId: string;
      modelId: string;
      questionId: string;
      stage: 'candidate' | 'judge';
      attempt: number;
      maxRetries: number;
      delayMs: number;
      reason: string;
      statusCode?: number;
    }
  | { type: 'budget_exceeded'; runId: string; maxBudgetUsd: number }
  | { type: 'budget_spent'; runId: string; spentUsd: number; source: 'candidate' | 'judge' }
  | { type: 'run_completed'; runId: string };

export type RunnerDeps = {
  resolveModel: (
    modelEntry: ApocbenchConfig['models'][number],
    config: ApocbenchConfig,
  ) => LanguageModel;
  resolveJudgeModel: (config: ApocbenchConfig) => LanguageModel;
  toolVersion: string;
};

type AbortableCall<T> = {
  call: () => Promise<T>;
  signal: AbortSignal;
  abort: () => void;
};

function createAbortableCall<T>(call: (signal: AbortSignal) => Promise<T>): AbortableCall<T> {
  const controller = new AbortController();
  return {
    call: () => call(controller.signal),
    signal: controller.signal,
    abort: () => controller.abort(new Error('aborted')),
  };
}

export type RunResult = {
  runId: string;
  outDir: string;
  summaryPath: string;
  reportPath: string;
};

type TextMessages = Array<
  | { role: 'system'; content: string }
  | { role: 'user'; content: string }
>;

type RunContext = {
  config: ApocbenchConfig;
  deps: RunnerDeps;
  db: RunnerDb;
  runId: string;
  onEvent?: (e: RunnerEvent) => void;
  budgetState: BudgetState;
  retryPolicy: RetryPolicy;
  judgeModel: LanguageModel;
  resumeMode: boolean;
};

type ModelEntry = ApocbenchConfig['models'][number];
type CandidateRouterDefaults = {
  temperature?: number | null;
  maxTokens?: number;
  timeoutMs?: number;
};

type GenerateTextArgs = Parameters<typeof generateText>[0];

function selectQuestions(params: {
  allQuestions: DatasetLine[];
  config: ApocbenchConfig;
  limitOverride?: number | null;
  categoriesOverride?: string[] | null;
}): DatasetLine[] {
  const { allQuestions, config, limitOverride, categoriesOverride } = params;
  const questionLimit = limitOverride ?? config.run.questionLimit ?? null;
  const categories = categoriesOverride ?? config.run.categories ?? null;

  let questions = allQuestions;
  if (categories && categories.length > 0) {
    const allowed = new Set(categories);
    questions = questions.filter((q) => allowed.has(q.category));
  }
  if (questionLimit != null) {
    questions = questions.slice(0, questionLimit);
  }
  return questions;
}

function resolveModels(params: {
  config: ApocbenchConfig;
  selectedModelIds?: string[];
}): ModelEntry[] {
  const { config, selectedModelIds } = params;
  return config.models.filter((m) =>
    selectedModelIds && selectedModelIds.length > 0 ? selectedModelIds.includes(m.id) : true,
  );
}

function createQueues(params: { config: ApocbenchConfig; models: ModelEntry[] }): {
  judgeQueue: PQueue;
  perModelQueue: Map<string, PQueue>;
} {
  const { config, models } = params;
  const judgeQueue = new PQueue({ concurrency: config.run.concurrency.judge });
  const perModelQueue = new Map<string, PQueue>();

  for (const model of models) {
    perModelQueue.set(
      model.id,
      new PQueue({ concurrency: config.run.concurrency.candidate }),
    );
  }

  return { judgeQueue, perModelQueue };
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
      await sleep(
        delayMs,
      );
    }
  }
  throw new Error('unreachable');
}

function redactReason(reason: string): string {
  const redacted = redactSecrets(reason);
  return typeof redacted === 'string' ? redacted : String(redacted);
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

function buildJudgeProviderOptions(config: ApocbenchConfig): ProviderOptions {
  return {
    openrouter: {
      ...(config.judge.reasoning ? { reasoning: { enabled: true } } : {}),
      ...(config.judge.routing
        ? { provider: toOpenRouterProviderParam(config.judge.routing) }
        : config.judge.provider
          ? {
              provider: {
                order: [config.judge.provider],
                allow_fallbacks: false,
              },
            }
          : {}),
    },
  } as ProviderOptions;
}

async function handleJudgeQuestion(params: {
  ctx: RunContext;
  modelEntry: ModelEntry;
  question: DatasetLine;
  candidateText: string;
  lastCandidateLatencyMs: number | undefined;
  lastCandidateUsage: NormalizedUsage | null | undefined;
  lastCandidateCostUsd: number | undefined;
}): Promise<void> {
  const {
    ctx,
    modelEntry,
    question,
    candidateText,
    lastCandidateLatencyMs,
    lastCandidateUsage,
    lastCandidateCostUsd,
  } = params;
  const { config, db, runId, onEvent, budgetState, judgeModel, retryPolicy } = ctx;
  const budgetCheck = () => isBudgetExceeded({ state: budgetState, runId, onEvent });

  if (budgetCheck()) {
    upsertRunResult(db, {
      runId,
      modelId: modelEntry.id,
      questionId: question.id,
      status: 'skipped',
      errorJson: JSON.stringify({ reason: 'budget_exceeded' }),
    });
    return;
  }

  const judgePrompt = buildJudgePrompt({
    question,
    candidateAnswer: candidateText,
  });
  const rubricIds = question.rubric.map((r) => r.id);
  try {
    const { object, raw } = await judgeWithRubricCompletenessRetry(
      {
        model: judgeModel,
        messages: [
          { role: 'system', content: JUDGE_SYSTEM_PROMPT },
          { role: 'user', content: judgePrompt },
        ] as TextMessages,
        maxTokens: config.judge.maxTokens,
        timeoutMs: config.routers.openrouter.default.timeoutMs ?? null,
        temperature: config.judge.temperature,
        providerOptions: buildJudgeProviderOptions(config),
        rubricIds,
      },
      { rubric: question.rubric.map((r) => ({ id: r.id })) },
      {
        retry: retryPolicy,
        onRetry: (retry) =>
          onEvent?.({
            type: 'request_retry',
            runId,
            modelId: modelEntry.id,
            questionId: question.id,
            stage: 'judge',
            attempt: retry.attempt,
            maxRetries: retry.maxRetries,
            delayMs: retry.delayMs,
            reason: redactReason(retry.reason),
            ...(retry.statusCode != null ? { statusCode: retry.statusCode } : {}),
          }),
      },
    );

    const computed = computeOverallScore({
      judgeOutput: object,
      rubric: question.rubric.map((r) => ({
        id: r.id,
        weight: r.weight,
        maxScore: r.maxScore,
      })),
    });

    const judgeParsed: JudgeOutput = {
      ...object,
      overall_score: computed.overallScore,
    };

    const judgeCostUsd = extractOpenRouterCost(raw);
    if (judgeCostUsd != null) {
      recordSpend({
        state: budgetState,
        runId,
        costUsd: judgeCostUsd,
        source: 'judge',
        onEvent,
      });
    }

    const redactedRequest = redactSecrets({
      model: config.judge.model,
      provider: config.judge.provider,
    });

    db.transaction(() => {
      upsertRunResult(db, {
        runId,
        modelId: modelEntry.id,
        questionId: question.id,
        judgeRequestJson: JSON.stringify(redactedRequest),
        judgeResponseJson: JSON.stringify(raw),
        judgeParsedJson: JSON.stringify(judgeParsed),
        scoreOverall: computed.overallScore,
        scoreRubricJson: JSON.stringify(computed.rubricScores),
        autoFail: judgeParsed.auto_fail,
        autoFailReason: judgeParsed.auto_fail_reason,
        status: 'done',
      });
    })();

    onEvent?.({
      type: 'question_completed',
      runId,
      modelId: modelEntry.id,
      questionId: question.id,
      overallScore: computed.overallScore,
      latencyMs: lastCandidateLatencyMs,
      usage: lastCandidateUsage,
      costUsd: lastCandidateCostUsd,
    });

    // Fire-and-forget: don't block judge slot for metrics fetch
    // Extract generation ID immediately to avoid retaining the full response
    const judgeGenerationId = extractOpenRouterGenerationId(raw);
    if (judgeGenerationId) {
      void maybeEmitOpenRouterGenerationMetrics({
        config,
        runId,
        modelId: modelEntry.id,
        questionId: question.id,
        generationId: judgeGenerationId,
        onEvent,
      });
    }
  } catch (err) {
    const message = (err as Error).message;
    upsertRunResult(db, {
      runId,
      modelId: modelEntry.id,
      questionId: question.id,
      status: 'judge_failed',
      errorJson: JSON.stringify({ message }),
    });
    onEvent?.({
      type: 'question_failed',
      runId,
      modelId: modelEntry.id,
      questionId: question.id,
      stage: 'judge',
      message,
      latencyMs: lastCandidateLatencyMs,
      usage: lastCandidateUsage,
      costUsd: lastCandidateCostUsd,
    });

    // On failure path, no generation ID to fetch metrics for
    // (removed fire-and-forget call that passed null result)
  }
}

async function handleCandidateQuestion(params: {
  ctx: RunContext;
  modelEntry: ModelEntry;
  candidateModel: LanguageModel;
  question: DatasetLine;
  judgeQueue: PQueue;
}): Promise<void> {
  const { ctx, modelEntry, candidateModel, question, judgeQueue } = params;
  const { config, db, runId, onEvent, budgetState, resumeMode, retryPolicy } = ctx;
  const budgetCheck = () => isBudgetExceeded({ state: budgetState, runId, onEvent });

  if (resumeMode && isRunResultDone(db, runId, modelEntry.id, question.id)) {
    return;
  }

  if (budgetCheck()) {
    upsertRunResult(db, {
      runId,
      modelId: modelEntry.id,
      questionId: question.id,
      status: 'skipped',
      errorJson: JSON.stringify({ reason: 'budget_exceeded' }),
    });
    return;
  }

  onEvent?.({
    type: 'question_started',
    runId,
    modelId: modelEntry.id,
    questionId: question.id,
  });

  const candidatePrompt = buildCandidatePrompt(question);
  const candidateStart = Date.now();

  // These variables capture only primitive/normalized data to avoid retaining
  // references to the large AI SDK response objects in closures.
  let lastCandidateLatencyMs: number | undefined;
  let lastCandidateUsage: NormalizedUsage | null | undefined;
  let lastCandidateCostUsd: number | undefined;

  try {
    const routerDefaults = getCandidateRouterDefaults(config, modelEntry);
    const candidateResult = await generateTextWithRetry({
      timeoutMs:
        modelEntry.params?.timeoutMs ??
        routerDefaults.timeoutMs ??
        null,
      retryPolicy,
      call: {
        model: candidateModel,
        messages: [
          { role: 'system', content: CANDIDATE_SYSTEM_PROMPT },
          { role: 'user', content: candidatePrompt },
        ] as TextMessages,
        temperature:
          modelEntry.params?.temperature ??
          routerDefaults.temperature ??
          undefined,
        maxOutputTokens: getCandidateMaxOutputTokens(config, modelEntry),
        providerOptions: buildCandidateProviderOptions(config, modelEntry),
      },
      onRetry: (retry) =>
        onEvent?.({
          type: 'request_retry',
          runId,
          modelId: modelEntry.id,
          questionId: question.id,
          stage: 'candidate',
          attempt: retry.attempt,
          maxRetries: retry.maxRetries,
          delayMs: retry.delayMs,
          reason: redactReason(retry.reason),
          ...(retry.statusCode != null ? { statusCode: retry.statusCode } : {}),
        }),
    });

    const candidateText = candidateResult.text;
    const candidateCostUsd = extractOpenRouterCost(candidateResult);
    if (candidateCostUsd != null) {
      recordSpend({
        state: budgetState,
        runId,
        costUsd: candidateCostUsd,
        source: 'candidate',
        onEvent,
      });
    }

    lastCandidateLatencyMs = Date.now() - candidateStart;
    // Extract and normalize usage immediately to break reference to candidateResult.
    // This is critical: the AI SDK response object can be large and retaining it
    // in closures (like the judge queue task) causes memory accumulation.
    const candidateOr = normalizeOpenRouterUsageFromProviderMetadata(
      (candidateResult as { providerMetadata?: unknown } | null | undefined)
        ?.providerMetadata,
    );
    // Prefer OpenRouter normalized usage, fall back to normalizing standard usage
    lastCandidateUsage = candidateOr.usage ?? normalizeUsage(candidateResult.usage);
    lastCandidateCostUsd = candidateOr.costUsd ?? candidateCostUsd ?? undefined;

    const candidateGenerationId = extractOpenRouterGenerationId(candidateResult);
    if (candidateGenerationId) {
      onEvent?.({
        type: 'generation_metrics',
        runId,
        modelId: modelEntry.id,
        questionId: question.id,
        generationId: candidateGenerationId,
      });
    }

    // Fire-and-forget: don't block candidate slot for metrics fetch
    // Note: Only pass the generationId, not the full result, to avoid retaining
    // the large AI SDK response object in the closure until the async completes.
    if (candidateGenerationId) {
      void maybeEmitOpenRouterGenerationMetrics({
        config,
        runId,
        modelId: modelEntry.id,
        questionId: question.id,
        generationId: candidateGenerationId,
        onEvent,
      });
    }

    const metrics: CandidateMetrics = {
      latencyMs: lastCandidateLatencyMs,
      usage: lastCandidateUsage,
      costUsd: lastCandidateCostUsd,
    };

    upsertRunResult(db, {
      runId,
      modelId: modelEntry.id,
      questionId: question.id,
      candidatePrompt,
      candidateCompletion: candidateText,
      candidateMetricsJson: JSON.stringify(metrics),
      status: 'candidate_done',
    });

    if (budgetCheck()) {
      upsertRunResult(db, {
        runId,
        modelId: modelEntry.id,
        questionId: question.id,
        status: 'skipped',
        errorJson: JSON.stringify({ reason: 'budget_exceeded' }),
      });
      return;
    }

    // Don't await - let the candidate slot free up immediately
    // The judgeQueue.onIdle() at the end ensures all judge tasks complete
    void judgeQueue.add(async () => {
      await handleJudgeQuestion({
        ctx,
        modelEntry,
        question,
        candidateText,
        lastCandidateLatencyMs,
        lastCandidateUsage,
        lastCandidateCostUsd,
      });
    });
  } catch (err) {
    const message = (err as Error).message;
    const candidateLatencyMs = Date.now() - candidateStart;
    upsertRunResult(db, {
      runId,
      modelId: modelEntry.id,
      questionId: question.id,
      candidatePrompt,
      status: 'candidate_failed',
      errorJson: JSON.stringify({ message }),
    });
    onEvent?.({
      type: 'question_failed',
      runId,
      modelId: modelEntry.id,
      questionId: question.id,
      stage: 'candidate',
      message,
      latencyMs: candidateLatencyMs,
    });
  }
}

function scheduleCandidateTasks(params: {
  ctx: RunContext;
  models: ModelEntry[];
  questions: DatasetLine[];
  perModelQueue: Map<string, PQueue>;
  judgeQueue: PQueue;
}): Array<Promise<void>> {
  const { ctx, models, questions, perModelQueue, judgeQueue } = params;
  const tasks: Array<Promise<void>> = [];

  for (const modelEntry of models) {
    const candidateModel = ctx.deps.resolveModel(modelEntry, ctx.config);
    const modelQueue = perModelQueue.get(modelEntry.id)!;

    for (const question of questions) {
      tasks.push(
        modelQueue.add(async () => {
          await handleCandidateQuestion({
            ctx,
            modelEntry,
            candidateModel,
            question,
            judgeQueue,
          });
        }),
      );
    }
  }

  return tasks;
}

function computeModelSummaries(params: {
  db: RunnerDb;
  runId: string;
  modelIds: string[];
}): Array<ReturnType<typeof aggregateModel>> {
  const { db, runId, modelIds } = params;
  if (modelIds.length === 0) return [];
  const placeholders = modelIds.map(() => '?').join(', ');
  const rows = db
    .prepare(
      `
              SELECT
                mr.model_id,
                mr.question_id,
                mr.status,
                mr.score_overall,
                mr.auto_fail,
                mr.candidate_metrics_json,
                q.category,
                q.difficulty
              FROM model_results mr
              JOIN questions q
                ON q.run_id = mr.run_id AND q.question_id = mr.question_id
              WHERE mr.run_id = ? AND mr.model_id IN (${placeholders})
            `,
    )
    .all(runId, ...modelIds) as Array<{
    model_id: string;
    question_id: string;
    status: string;
    score_overall: number | null;
    auto_fail: number | null;
    candidate_metrics_json: string | null;
    category: string | null;
    difficulty: string | null;
  }>;

  type KnownStatus =
    | 'done'
    | 'candidate_done'
    | 'candidate_failed'
    | 'judge_failed'
    | 'skipped';
  const toKnownStatus = (s: string): KnownStatus =>
    s === 'done' ||
    s === 'candidate_done' ||
    s === 'candidate_failed' ||
    s === 'judge_failed' ||
    s === 'skipped'
      ? s
      : 'candidate_failed';

  const parseLatencyMs = (candidateMetricsJson: string | null): number | null => {
    if (!candidateMetricsJson) return null;
    try {
      const parsed = JSON.parse(candidateMetricsJson) as {
        latencyMs?: unknown;
      } | null;
      const latencyMs = parsed?.latencyMs;
      return typeof latencyMs === 'number' && Number.isFinite(latencyMs)
        ? latencyMs
        : null;
    } catch {
      return null;
    }
  };

  const perModel = new Map<string, Parameters<typeof aggregateModel>[0]['questionScores']>();
  for (const row of rows) {
    const list = perModel.get(row.model_id) ?? [];
    list.push({
      questionId: row.question_id,
      category: row.category ?? 'unknown',
      difficulty: row.difficulty ?? 'unknown',
      status: toKnownStatus(row.status),
      overallScore: typeof row.score_overall === 'number' ? row.score_overall : 0,
      autoFail: row.auto_fail === 1,
      latencyMs: parseLatencyMs(row.candidate_metrics_json),
    });
    perModel.set(row.model_id, list);
  }

  return modelIds.map((modelId) =>
    aggregateModel({ modelId, questionScores: perModel.get(modelId) ?? [] }),
  );
}

export async function runBenchmark(params: {
  config: ApocbenchConfig;
  configPath: string;
  datasetPath: string;
  datasetAbsolutePath: string;
  questions: DatasetLine[];
  deps: RunnerDeps;
  dryRun: boolean;
  runIdOverride?: string;
  selectedModelIds?: string[];
  limitOverride?: number | null;
  categoriesOverride?: string[] | null;
  forceResume?: boolean;
  onEvent?: (e: RunnerEvent) => void;
}): Promise<RunResult | null> {
  const {
    config,
    datasetAbsolutePath,
    deps,
    dryRun,
    selectedModelIds,
    onEvent,
  } = params;

  const resumeMode = config.run.resume || params.forceResume === true;
  const questions = selectQuestions({
    allQuestions: params.questions,
    config,
    limitOverride: params.limitOverride,
    categoriesOverride: params.categoriesOverride,
  });

  const datasetSha = sha256FileHex(datasetAbsolutePath);
  const templateHash = promptTemplateHash(buildCandidatePrompt, buildJudgePrompt);

  if (dryRun) return null;

  const runId = params.runIdOverride ?? makeRunId(config.run.name);
  const runOutDir = path.resolve(process.cwd(), config.run.outDir, runId);

  onEvent?.({ type: 'run_started', runId, startedAtMs: Date.now() });

  const db = openRunnerDb({ outDir: config.run.outDir });
  ensureRunStarted({
    db,
    runId,
    toolVersion: deps.toolVersion,
    config,
    datasetPath: datasetAbsolutePath,
    datasetSha256: datasetSha,
    promptTemplateHash: templateHash,
  });
  insertRunQuestions(db, runId, questions);

  const judgeModel = deps.resolveJudgeModel(config);
  const models = resolveModels({ config, selectedModelIds });
  const { judgeQueue, perModelQueue } = createQueues({ config, models });

  const maxBudgetUsd = config.run.maxBudgetUsd ?? null;
  const budgetState = initBudgetState(maxBudgetUsd);
  const retryPolicy = { ...DEFAULT_RETRY_POLICY, ...config.run.retry };
  const ctx: RunContext = {
    config,
    deps,
    db,
    runId,
    onEvent,
    budgetState,
    retryPolicy,
    judgeModel,
    resumeMode,
  };

  const tasks = scheduleCandidateTasks({
    ctx,
    models,
    questions,
    perModelQueue,
    judgeQueue,
  });

  await Promise.all(tasks);
  await judgeQueue.onIdle();

  const modelIds = models.map((m) => m.id);
  const summaries = computeModelSummaries({ db, runId, modelIds });

  const summary = {
    runId,
    createdAt: new Date().toISOString(),
    datasetPath: datasetAbsolutePath,
    datasetSha256: datasetSha,
    promptTemplateHash: templateHash,
    judge: { model: config.judge.model, provider: config.judge.provider },
    models: summaries,
  };

  const summaryPath = path.join(runOutDir, 'summary.json');
  writeJson(summaryPath, summary);

  const reportPath = path.join(runOutDir, 'report.html');
  const results = listRunResults(db, runId);
  const html = renderHtmlReport({ runId, summaryJson: summary, results });
  fs.mkdirSync(runOutDir, { recursive: true });
  fs.writeFileSync(reportPath, html);

  updateRunStatusForRun(db, runId, 'completed');
  onEvent?.({ type: 'run_completed', runId });
  return { runId, outDir: runOutDir, summaryPath, reportPath };
}
