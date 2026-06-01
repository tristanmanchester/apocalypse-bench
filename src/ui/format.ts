import type { RunnerEvent } from '../core/runner/orchestrator';
import { normalizeUsage, type NormalizedUsage } from '../core/runner/openrouterUsage';

export function formatNumber(value: number, decimals: number): string {
  if (!Number.isFinite(value)) return '—';
  return value.toFixed(decimals);
}

export function formatUsd(value: number | null | undefined): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—';
  return `$${formatNumber(value, 4)}`;
}

export function formatDuration(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return '—';
  const secondsTotal = Math.floor(ms / 1000);
  const hours = Math.floor(secondsTotal / 3600);
  const minutes = Math.floor((secondsTotal % 3600) / 60);
  const seconds = secondsTotal % 60;

  if (hours > 0)
    return `${hours}h ${String(minutes).padStart(2, '0')}m ${String(seconds).padStart(2, '0')}s`;
  if (minutes > 0) return `${minutes}m ${String(seconds).padStart(2, '0')}s`;
  return `${seconds}s`;
}

export { normalizeUsage, type NormalizedUsage };

export function safeJson(value: unknown, maxLen = 240): string {
  try {
    const s = JSON.stringify(value);
    if (typeof s !== 'string') return '…';
    return s.length > maxLen ? `${s.slice(0, maxLen)}…` : s;
  } catch {
    return '…';
  }
}

export function formatEventSummary(e: RunnerEvent): string {
  switch (e.type) {
    case 'run_started':
      return `run_started ${e.runId}`;
    case 'run_completed':
      return `run_completed ${e.runId}`;
    case 'question_started':
      return `question_started ${e.modelId} ${e.questionId}`;
    case 'question_completed':
      return `question_completed ${e.modelId} ${e.questionId} score=${formatNumber(e.overallScore, 2)}`;
    case 'question_failed':
      return `question_failed ${e.modelId} ${e.questionId} stage=${e.stage} ${e.message}`;
    case 'request_retry':
      return `request_retry ${e.modelId} ${e.questionId} stage=${e.stage} attempt=${e.attempt}/${e.maxRetries} delay=${formatDuration(e.delayMs)} ${e.reason}`;
    case 'budget_spent':
      return `budget_spent ${formatUsd(e.spentUsd)}`;
    case 'budget_exceeded':
      return `budget_exceeded max=${formatUsd(e.maxBudgetUsd)}`;
    case 'generation_metrics': {
      const tpsStr = typeof e.tps === 'number' ? formatNumber(e.tps, 1) : '—';
      return `generation_metrics ${e.modelId} ${e.questionId} tps=${tpsStr}`;
    }
    default:
      // Avoid JSON.stringify which can be expensive and retain references
      return `unknown_event`;
  }
}
