import type { LanguageModel } from 'ai';

import type { ApocbenchConfig } from '../config/schema';
import type { DatasetLine } from '../dataset/schema';
import type { WikiClient } from '../wiki/client';
import type { RetryPolicy } from '../runner/retryPolicy';
import { executeAiSdkCandidate } from './internal/aiSdkCandidate';
import type { CandidateExecutionResult } from './types';

type ModelEntry = ApocbenchConfig['models'][number];

export async function executeCandidate(params: {
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
  return executeAiSdkCandidate(params);
}

export type { CandidateExecutionResult, CandidateMetrics } from './types';
