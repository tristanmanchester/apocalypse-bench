import type { RetrievalTrace } from '../wiki/rag';

export type CandidateMetrics = {
  latencyMs: number;
  usage?: unknown;
  costUsd?: number;
};

export type CandidateExecutionResult = {
  prompt: string;
  completion: string;
  metrics: CandidateMetrics;
  retrievalTrace?: RetrievalTrace;
  generationId?: string;
};
