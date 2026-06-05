import fs from 'node:fs';
import path from 'node:path';
import PQueue from 'p-queue';
import type { LanguageModel } from 'ai';
import type { ProviderOptions } from '@ai-sdk/provider-utils';

import type {
  ApocbenchConfig,
  CandidateMode,
  OpenRouterJudgeConfig,
} from '../config/schema';
import {
  isCodexJudgeConfig,
  isOpenRouterJudgeConfig,
  isWikiCandidateMode,
  toOpenRouterProviderParam,
} from '../config/schema';
import type { DatasetLine } from '../dataset/schema';
import { buildCandidatePrompt } from '../prompts/candidatePrompt';
import { buildJudgePrompt } from '../prompts/judgePrompt';
import { JUDGE_SYSTEM_PROMPT } from '../prompts/systemPrompts';
import {
  checkWikiReadiness,
  createWikiClientFromConfig,
  type WikiClient,
} from '../wiki/client';
import type { WikiReadiness, WikiSearchMode } from '../wiki/types';
import { executeCandidate, type CandidateMetrics } from '../candidate';
import { aggregateModel } from '../scoring/aggregate';
import { computeOverallScore, judgeWithRubricCompletenessRetry } from './judge';
import { makeRunId, promptTemplateHash } from './runId';
import type { JudgeOutput } from './types';
import { sha256FileHex, sha256Hex } from '../../utils/hash';
import { redactSecrets } from '../../utils/redaction';
import { writeJson } from '../../reports/json/exports';
import { renderHtmlReport } from '../../reports/html/renderHtml';
import {
  ensureRunStarted,
  insertRunQuestions,
  isRunCandidateDone,
  isRunResultDone,
  listRunResults,
  openRunnerDb,
  type RunnerDb,
  updateRunStatusForRun,
  upsertRunResult,
} from './persistence';
import {
  extractOpenRouterCost,
  initBudgetState,
  isBudgetExceeded,
  recordSpend,
  type BudgetState,
} from './budget';
import { maybeEmitOpenRouterGenerationMetrics } from './generationMetrics';
import { DEFAULT_RETRY_POLICY, type RetryPolicy } from './retryPolicy';

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
  | {
      type: 'budget_spent';
      runId: string;
      spentUsd: number;
      source: 'candidate' | 'judge';
    }
  | { type: 'run_completed'; runId: string };

export type RunnerDeps = {
  resolveModel: (
    modelEntry: ApocbenchConfig['models'][number],
    config: ApocbenchConfig,
  ) => LanguageModel;
  resolveJudgeModel: (config: ApocbenchConfig) => LanguageModel;
  toolVersion: string;
};

export type RunResult = {
  runId: string;
  outDir: string;
  summaryPath: string;
  reportPath: string;
};

type TextMessages = Array<
  { role: 'system'; content: string } | { role: 'user'; content: string }
>;

type RunContext = {
  config: ApocbenchConfig;
  deps: RunnerDeps;
  db: RunnerDb;
  runId: string;
  onEvent?: (e: RunnerEvent) => void;
  budgetState: BudgetState;
  retryPolicy: RetryPolicy;
  judgeModel: LanguageModel | null;
  resumeMode: boolean;
  wikiClient?: WikiClient;
};

type ModelEntry = ApocbenchConfig['models'][number];

function deterministicQuestionShuffle(
  questions: DatasetLine[],
  seed: string,
): DatasetLine[] {
  return [...questions].sort((left, right) => {
    const leftKey = sha256Hex(`${seed}\u0000${left.id}`);
    const rightKey = sha256Hex(`${seed}\u0000${right.id}`);
    return leftKey.localeCompare(rightKey) || left.id.localeCompare(right.id);
  });
}

export function selectQuestions(params: {
  allQuestions: DatasetLine[];
  config: ApocbenchConfig;
  limitOverride?: number | null;
  categoriesOverride?: string[] | null;
  questionIdsOverride?: string[] | null;
}): DatasetLine[] {
  const { allQuestions, config, limitOverride, categoriesOverride, questionIdsOverride } =
    params;
  const categories = categoriesOverride ?? config.run.categories ?? null;
  const questionIds = questionIdsOverride ?? config.run.questionIds ?? null;
  const questionLimit =
    limitOverride ??
    (questionIdsOverride && questionIdsOverride.length > 0
      ? null
      : (config.run.questionLimit ?? null));

  let questions = allQuestions;
  if (categories && categories.length > 0) {
    const allowed = new Set(categories);
    questions = questions.filter((q) => allowed.has(q.category));
  }
  if (questionIds && questionIds.length > 0) {
    const allowed = new Set(questionIds);
    questions = questions.filter((q) => allowed.has(q.id));
  }
  if ((config.run.questionOrder ?? 'sequential') === 'shuffle') {
    questions = deterministicQuestionShuffle(
      questions,
      config.run.questionSeed ?? config.run.name,
    );
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
    selectedModelIds && selectedModelIds.length > 0
      ? selectedModelIds.includes(m.id)
      : true,
  );
}

export function createQueues(params: { config: ApocbenchConfig; models: ModelEntry[] }): {
  judgeQueue: PQueue;
  perModelQueue: Map<string, PQueue>;
} {
  const { config, models } = params;
  const judgeQueue = new PQueue({ concurrency: config.run.concurrency.judge });
  const perModelQueue = new Map<string, PQueue>();

  for (const model of models) {
    perModelQueue.set(
      model.id,
      new PQueue({ concurrency: model.concurrency ?? config.run.concurrency.candidate }),
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
    result as { providerMetadata?: { openrouter?: { id?: unknown } } } | null | undefined
  )?.providerMetadata?.openrouter?.id;
  if (typeof providerMetadataId === 'string' && providerMetadataId.length > 0)
    return providerMetadataId;

  return null;
}

function redactReason(reason: string): string {
  const redacted = redactSecrets(reason);
  return typeof redacted === 'string' ? redacted : String(redacted);
}

function requireOpenRouterJudgeConfig(config: ApocbenchConfig): OpenRouterJudgeConfig {
  if (!isOpenRouterJudgeConfig(config.judge)) {
    throw new Error('Inline judging requires judge.backend=openrouter');
  }
  return config.judge;
}

function buildJudgeProviderOptions(config: ApocbenchConfig): ProviderOptions {
  const judgeConfig = requireOpenRouterJudgeConfig(config);
  return {
    openrouter: {
      ...(judgeConfig.reasoning ? { reasoning: { enabled: true } } : {}),
      ...(judgeConfig.routing
        ? { provider: toOpenRouterProviderParam(judgeConfig.routing) }
        : judgeConfig.provider
          ? {
              provider: {
                order: [judgeConfig.provider],
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
  lastCandidateUsage: CandidateMetrics['usage'] | undefined;
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
  const { config, db, runId, onEvent, budgetState, retryPolicy } = ctx;
  const judgeModel = ctx.judgeModel;
  if (!judgeModel) {
    throw new Error('judge model is unavailable in candidate-only mode');
  }
  const judgeConfig = requireOpenRouterJudgeConfig(config);
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
        maxTokens: judgeConfig.maxTokens,
        timeoutMs: config.routers.openrouter.default.timeoutMs ?? null,
        temperature: judgeConfig.temperature,
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
      model: judgeConfig.model,
      provider: judgeConfig.provider,
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

function nonAnswerAutoFailReason(candidateText: string): string | null {
  const trimmed = candidateText.trim();
  if (trimmed.length === 0) return 'candidate produced an empty final answer';

  const normalized = trimmed.replace(/\s+/g, ' ');
  const lower = normalized.toLowerCase();
  const wordCount = normalized.split(/\s+/).filter(Boolean).length;
  const toolIntent =
    /\b(?:tool calls?|tool_call|invoke search|produce the tool calls?|api call|awaiting api call|wiki_(?:search|read|hybrid_search|semantic_search|literal_search))\b/i.test(
      normalized,
    ) ||
    /\b(?:i|we) (?:need|will|should|must|would|can) (?:to )?(?:search|look up|call|read)\b/i.test(
      normalized,
    );

  if (toolIntent && wordCount < 40) {
    return 'candidate described tool/search intent instead of providing a final answer';
  }

  if (
    /^<tool_call>[\s\S]*<\/tool_call>$/i.test(trimmed) ||
    /^```(?:json)?\s*\{[\s\S]*"name"\s*:/i.test(trimmed) ||
    /^\{[\s\S]*"name"\s*:[\s\S]*"arguments"\s*:/i.test(trimmed)
  ) {
    return 'candidate output only tool-call syntax instead of a final answer';
  }

  if (
    wordCount < 12 &&
    /\b(?:cannot|can't|unable|sorry|apologize|insufficient information|need more information)\b/.test(
      lower,
    )
  ) {
    return 'candidate produced a refusal or non-answer';
  }

  return null;
}

function zeroRubricScores(question: DatasetLine): Record<string, number> {
  return Object.fromEntries(question.rubric.map((rubric) => [rubric.id, 0]));
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
  const candidateOnly = config.run.candidateOnly === true;

  if (
    resumeMode &&
    (candidateOnly
      ? isRunCandidateDone(db, runId, modelEntry.id, question.id)
      : isRunResultDone(db, runId, modelEntry.id, question.id))
  ) {
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

  const candidateStart = Date.now();
  let candidatePrompt: string | undefined;
  let retrievalTraceJson: string | undefined;

  try {
    const candidateResult = await executeCandidate({
      config,
      modelEntry,
      model: candidateModel,
      question,
      wikiClient: ctx.wikiClient,
      retryPolicy,
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

    candidatePrompt = candidateResult.prompt;
    retrievalTraceJson = candidateResult.retrievalTrace
      ? JSON.stringify(candidateResult.retrievalTrace)
      : undefined;
    const candidateText = candidateResult.completion;
    const candidateMetrics = candidateResult.metrics;
    if (candidateMetrics.costUsd != null) {
      recordSpend({
        state: budgetState,
        runId,
        costUsd: candidateMetrics.costUsd,
        source: 'candidate',
        onEvent,
      });
    }

    const candidateGenerationId = candidateResult.generationId;
    if (candidateGenerationId) {
      onEvent?.({
        type: 'generation_metrics',
        runId,
        modelId: modelEntry.id,
        questionId: question.id,
        generationId: candidateGenerationId,
      });
    }

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

    upsertRunResult(db, {
      runId,
      modelId: modelEntry.id,
      questionId: question.id,
      candidatePrompt,
      retrievalTraceJson,
      candidateCompletion: candidateText,
      candidateMetricsJson: JSON.stringify(candidateMetrics),
      status: 'candidate_done',
    });

    if (candidateOnly) {
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

    const nonAnswerReason = nonAnswerAutoFailReason(candidateText);
    if (nonAnswerReason) {
      const judgeParsed: JudgeOutput = {
        rubric_scores: zeroRubricScores(question),
        auto_fail: true,
        auto_fail_reason: nonAnswerReason,
        overall_score: 0,
        notes: nonAnswerReason,
      };
      upsertRunResult(db, {
        runId,
        modelId: modelEntry.id,
        questionId: question.id,
        judgeRequestJson: JSON.stringify({ skipped: 'non_answer_auto_fail' }),
        judgeResponseJson: JSON.stringify({ skipped: 'non_answer_auto_fail' }),
        judgeParsedJson: JSON.stringify(judgeParsed),
        scoreOverall: 0,
        scoreRubricJson: JSON.stringify(judgeParsed.rubric_scores),
        autoFail: true,
        autoFailReason: nonAnswerReason,
        status: 'done',
      });
      onEvent?.({
        type: 'question_completed',
        runId,
        modelId: modelEntry.id,
        questionId: question.id,
        overallScore: 0,
        latencyMs: candidateMetrics.latencyMs,
        usage: candidateMetrics.usage,
        costUsd: candidateMetrics.costUsd,
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
        lastCandidateLatencyMs: candidateMetrics.latencyMs,
        lastCandidateUsage: candidateMetrics.usage,
        lastCandidateCostUsd: candidateMetrics.costUsd,
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
      retrievalTraceJson,
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
                mr.retrieval_trace_json,
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
    retrieval_trace_json: string | null;
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

  const perModel = new Map<
    string,
    Parameters<typeof aggregateModel>[0]['questionScores']
  >();
  const retrievalByModel = new Map<string, string[]>();
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
    if (row.retrieval_trace_json) {
      const traces = retrievalByModel.get(row.model_id) ?? [];
      traces.push(row.retrieval_trace_json);
      retrievalByModel.set(row.model_id, traces);
    }
  }

  return modelIds.map((modelId) => {
    const summary = aggregateModel({
      modelId,
      questionScores: perModel.get(modelId) ?? [],
    });
    const retrieval = summarizeRetrievalTraces(retrievalByModel.get(modelId) ?? []);
    return retrieval.traceCount > 0 ? { ...summary, retrieval } : summary;
  });
}

function requiredWikiCapabilitiesForMode(mode: CandidateMode): WikiSearchMode[] {
  switch (mode) {
    case 'rag-bm25':
    case 'agent-bm25':
    case 'agent-bm25-research':
    case 'agent-bm25-research-v2':
    case 'agent-bm25-rerank-research':
    case 'agent-bm25-research-read-required':
    case 'agent-bm25-research-smart-read':
      return ['bm25'];
    case 'rag-dense':
    case 'agent-dense':
      return ['dense'];
    case 'rag-hybrid':
    case 'agent-hybrid':
    case 'agent-hybrid-research-smart-read':
      return ['hybrid'];
    case 'agent-wiki':
      return ['bm25', 'dense', 'hybrid', 'literal'];
    case 'agent-rg':
    case 'agent-literal':
      return ['literal'];
    case 'direct':
      return [];
  }
}

function assertWikiCapabilities(params: {
  readiness: WikiReadiness;
  models: ModelEntry[];
}): void {
  const { readiness, models } = params;
  for (const model of models) {
    const mode = model.candidateMode ?? 'direct';
    const capabilities = requiredWikiCapabilitiesForMode(mode);
    const missing = capabilities.filter(
      (capability) => !readiness.capabilities.has(capability),
    );
    if (missing.length === 0) continue;
    if (missing.length === 1) {
      throw new Error(
        `wiki service is missing required capability '${missing[0]}' for model '${model.id}' candidateMode '${mode}'`,
      );
    }
    throw new Error(
      `wiki service is missing required capabilities '${missing.join(', ')}' for model '${model.id}' candidateMode '${mode}'`,
    );
  }
}

function summarizeRetrievalTraces(rawTraces: string[]): {
  traceCount: number;
  modes: Record<string, number>;
  toolCallCount: number;
  searchCount: number;
  readCount: number;
  uniqueSourceTitles: string[];
  latencyMs: {
    medianMs: number | null;
    meanMs: number | null;
    p90Ms: number | null;
    minMs: number | null;
    maxMs: number | null;
  };
} {
  const modes: Record<string, number> = {};
  const titles = new Set<string>();
  const latencies: number[] = [];
  let traceCount = 0;
  let toolCallCount = 0;
  let searchCount = 0;
  let readCount = 0;

  for (const raw of rawTraces) {
    const trace = safeParseObject(raw);
    if (!trace) continue;
    traceCount += 1;
    if (typeof trace.mode === 'string') {
      modes[trace.mode] = (modes[trace.mode] ?? 0) + 1;
    }
    const searches = Array.isArray(trace.searches) ? trace.searches : [];
    const reads = Array.isArray(trace.reads) ? trace.reads : [];
    const toolCalls = Array.isArray(trace.toolCalls) ? trace.toolCalls : [];
    toolCallCount +=
      typeof trace.toolCallCount === 'number' && Number.isFinite(trace.toolCallCount)
        ? trace.toolCallCount
        : toolCalls.length;
    searchCount += searches.length;
    readCount += reads.length;
    for (const search of searches) {
      if (!isRecord(search)) continue;
      pushLatency(latencies, search.latencyMs);
    }
    for (const read of reads) {
      if (!isRecord(read)) continue;
      pushLatency(latencies, read.latencyMs);
      if (typeof read.title === 'string' && read.title.length > 0) {
        titles.add(read.title);
      }
    }
  }

  return {
    traceCount,
    modes,
    toolCallCount,
    searchCount,
    readCount,
    uniqueSourceTitles: Array.from(titles).sort(),
    latencyMs: latencyStats(latencies),
  };
}

function safeParseObject(raw: string): Record<string, unknown> | null {
  try {
    const parsed = JSON.parse(raw);
    return isRecord(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function pushLatency(values: number[], value: unknown): void {
  if (typeof value === 'number' && Number.isFinite(value)) {
    values.push(value);
  }
}

function latencyStats(values: number[]): {
  medianMs: number | null;
  meanMs: number | null;
  p90Ms: number | null;
  minMs: number | null;
  maxMs: number | null;
} {
  if (values.length === 0) {
    return { medianMs: null, meanMs: null, p90Ms: null, minMs: null, maxMs: null };
  }
  const sorted = [...values].sort((a, b) => a - b);
  const percentile = (p: number) =>
    sorted[Math.min(sorted.length - 1, Math.floor(p * (sorted.length - 1)))]!;
  return {
    medianMs: percentile(0.5),
    meanMs: values.reduce((sum, value) => sum + value, 0) / values.length,
    p90Ms: percentile(0.9),
    minMs: sorted[0]!,
    maxMs: sorted[sorted.length - 1]!,
  };
}

export async function runBenchmark(params: {
  config: ApocbenchConfig;
  configPath: string;
  datasetPath: string;
  datasetAbsolutePath: string;
  datasetMetadataPath?: string;
  datasetSha256?: string;
  questions: DatasetLine[];
  deps: RunnerDeps;
  dryRun: boolean;
  runIdOverride?: string;
  selectedModelIds?: string[];
  limitOverride?: number | null;
  categoriesOverride?: string[] | null;
  questionIdsOverride?: string[] | null;
  forceResume?: boolean;
  onEvent?: (e: RunnerEvent) => void;
}): Promise<RunResult | null> {
  const { config, datasetAbsolutePath, deps, dryRun, selectedModelIds, onEvent } = params;

  const resumeMode = config.run.resume || params.forceResume === true;
  const questions = selectQuestions({
    allQuestions: params.questions,
    config,
    limitOverride: params.limitOverride,
    categoriesOverride: params.categoriesOverride,
    questionIdsOverride: params.questionIdsOverride,
  });

  const datasetMetadataPath = params.datasetMetadataPath ?? datasetAbsolutePath;
  const datasetSha = params.datasetSha256 ?? sha256FileHex(datasetAbsolutePath);
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
    datasetPath: datasetMetadataPath,
    datasetSha256: datasetSha,
    promptTemplateHash: templateHash,
  });
  insertRunQuestions(db, runId, questions);

  const candidateOnly = config.run.candidateOnly === true;
  if (!candidateOnly && isCodexJudgeConfig(config.judge)) {
    throw new Error(
      'judge.backend=codex-cli requires run.candidateOnly=true and a separate Codex judge stage',
    );
  }
  const judgeModel = candidateOnly ? null : deps.resolveJudgeModel(config);
  const models = resolveModels({ config, selectedModelIds });
  const needsWiki = models.some((model) =>
    isWikiCandidateMode(model.candidateMode ?? 'direct'),
  );
  const wikiClient =
    needsWiki && config.wiki ? createWikiClientFromConfig(config.wiki) : undefined;
  if (needsWiki) {
    if (!config.wiki || !wikiClient) {
      throw new Error('missing wiki config for wiki-enabled candidate modes');
    }
    const readiness = await checkWikiReadiness(wikiClient, config.wiki);
    assertWikiCapabilities({ readiness, models });
  }
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
    wikiClient,
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
  const judgeSummary = (() => {
    if (candidateOnly) return { skipped: true, reason: 'candidateOnly' };
    if (isOpenRouterJudgeConfig(config.judge)) {
      return {
        backend: 'openrouter',
        model: config.judge.model,
        provider: config.judge.provider,
      };
    }
    return {
      backend: config.judge.backend,
      model: config.judge.model,
      reasoning: config.judge.reasoning,
    };
  })();

  const summary = {
    runId,
    createdAt: new Date().toISOString(),
    datasetPath: datasetMetadataPath,
    datasetSha256: datasetSha,
    promptTemplateHash: templateHash,
    judge: judgeSummary,
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
