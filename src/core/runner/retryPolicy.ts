export type RetryPolicy = {
  maxRetries: number;
  baseMs: number;
  maxMs: number;
  maxTotalTimeMs?: number | null;
};

export const DEFAULT_RETRY_POLICY: RetryPolicy = {
  maxRetries: 6,
  baseMs: 2000,
  maxMs: 60000,
  maxTotalTimeMs: null,
};

export type RetryDecision = {
  retryable: boolean;
  reason: string;
  statusCode?: number;
  retryAfterMs?: number;
};

const RETRYABLE_STATUS_CODES = new Set([408, 409, 425, 429]);
const NON_RETRYABLE_STATUS_CODES = new Set([400, 401, 403, 404, 422]);

const RETRYABLE_TEXT =
  /\b(429|5\d\d|ECONNRESET|ETIMEDOUT|ENOTFOUND|EAI_AGAIN)\b|timeout|aborted|temporarily rate-limited upstream|rate limited|try again|overloaded/i;

const NON_RETRYABLE_TEXT =
  /\b(400|401|403|404|422)\b|missing authentication|authentication header|unauthorized|forbidden|invalid model|invalid provider|unsupported parameter|schema|config/i;

export function isRetryableError(err: unknown): boolean {
  return classifyRetryError(err).retryable;
}

export function classifyRetryError(err: unknown): RetryDecision {
  const facts = collectErrorFacts(err);
  const retryAfterMs = firstDefined(facts.retryAfterMs);
  const statusCode = firstDefined(facts.statusCodes);

  const nonRetryableStatus = facts.statusCodes.find((code) =>
    NON_RETRYABLE_STATUS_CODES.has(code),
  );
  if (nonRetryableStatus != null) {
    return {
      retryable: false,
      reason: statusReason(nonRetryableStatus),
      statusCode: nonRetryableStatus,
      retryAfterMs,
    };
  }

  const retryableStatus = facts.statusCodes.find(
    (code) => RETRYABLE_STATUS_CODES.has(code) || (code >= 500 && code <= 599),
  );
  if (retryableStatus != null) {
    return {
      retryable: true,
      reason: statusReason(retryableStatus),
      statusCode: retryableStatus,
      retryAfterMs,
    };
  }

  const text = facts.texts.join('\n');
  if (NON_RETRYABLE_TEXT.test(text)) {
    return {
      retryable: false,
      reason: firstUsefulText(facts.texts) ?? 'non-retryable error',
      statusCode,
      retryAfterMs,
    };
  }

  if (RETRYABLE_TEXT.test(text)) {
    return {
      retryable: true,
      reason: firstUsefulText(facts.texts) ?? 'retryable error',
      statusCode,
      retryAfterMs,
    };
  }

  return {
    retryable: false,
    reason: firstUsefulText(facts.texts) ?? 'non-retryable error',
    statusCode,
    retryAfterMs,
  };
}

export function computeRetryDelayMs(params: {
  attempt: number;
  policy: RetryPolicy;
  retryAfterMs?: number;
}): number {
  const exp = Math.min(params.policy.maxMs, params.policy.baseMs * 2 ** params.attempt);
  const jitter = exp * (0.2 * (Math.random() - 0.5) * 2);
  const backoffMs = Math.max(0, Math.round(exp + jitter));
  return Math.max(backoffMs, params.retryAfterMs ?? 0);
}

export function shouldRetryWithinBudget(params: {
  startedAtMs: number;
  nowMs: number;
  delayMs: number;
  policy: RetryPolicy;
}): boolean {
  const maxTotalTimeMs = params.policy.maxTotalTimeMs ?? null;
  if (maxTotalTimeMs == null) return true;
  return params.nowMs + params.delayMs - params.startedAtMs <= maxTotalTimeMs;
}

type ErrorFacts = {
  texts: string[];
  statusCodes: number[];
  retryAfterMs: number[];
};

function collectErrorFacts(err: unknown): ErrorFacts {
  const facts: ErrorFacts = { texts: [], statusCodes: [], retryAfterMs: [] };
  collectUnknown(err, facts, new WeakSet<object>(), 0);
  if (facts.texts.length === 0) facts.texts.push(String(err));
  return facts;
}

function collectUnknown(
  value: unknown,
  facts: ErrorFacts,
  seen: WeakSet<object>,
  depth: number,
): void {
  if (value == null || depth > 6) return;

  if (typeof value === 'string') {
    collectText(value, facts);
    return;
  }

  if (typeof value === 'number') {
    collectStatusCode(value, facts);
    return;
  }

  if (typeof value !== 'object') {
    collectText(String(value), facts);
    return;
  }

  if (seen.has(value)) return;
  seen.add(value);

  if (value instanceof Error) {
    collectText(value.name, facts);
    collectText(value.message, facts);
  }

  if (isHeadersLike(value)) {
    collectRetryAfter(value.get('retry-after'), facts);
    collectRetryAfter(value.get('Retry-After'), facts);
  }

  const record = value as Record<string, unknown>;
  for (const key of [
    'message',
    'name',
    'code',
    'status',
    'statusCode',
    'raw',
    'body',
    'responseBody',
  ]) {
    const item = record[key];
    if (typeof item === 'number') collectStatusCode(item, facts);
    if (typeof item === 'string') collectText(item, facts);
  }

  collectRetryAfter(record['retry-after'] ?? record['Retry-After'], facts);

  for (const key of ['headers', 'responseHeaders']) {
    const headers = record[key];
    if (headers && typeof headers === 'object') {
      if (isHeadersLike(headers)) {
        collectRetryAfter(headers.get('retry-after'), facts);
        collectRetryAfter(headers.get('Retry-After'), facts);
      } else {
        const h = headers as Record<string, unknown>;
        collectRetryAfter(h['retry-after'] ?? h['Retry-After'], facts);
      }
    }
  }

  for (const key of ['cause', 'response', 'metadata', 'data', 'error']) {
    collectUnknown(record[key], facts, seen, depth + 1);
  }
}

function collectText(text: string, facts: ErrorFacts): void {
  const trimmed = text.trim();
  if (!trimmed) return;
  facts.texts.push(trimmed);
  const statusMatch = trimmed.match(/\b([45]\d\d)\b/);
  if (statusMatch) collectStatusCode(Number(statusMatch[1]), facts);
}

function collectStatusCode(value: number, facts: ErrorFacts): void {
  if (!Number.isInteger(value) || value < 100 || value > 599) return;
  facts.statusCodes.push(value);
}

function collectRetryAfter(value: unknown, facts: ErrorFacts): void {
  const parsed = parseRetryAfterMs(value);
  if (parsed != null) facts.retryAfterMs.push(parsed);
}

export function parseRetryAfterMs(value: unknown, nowMs = Date.now()): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return Math.max(0, Math.round(value * 1000));
  }
  if (typeof value !== 'string') return null;

  const trimmed = value.trim();
  if (!trimmed) return null;

  const seconds = Number(trimmed);
  if (Number.isFinite(seconds)) return Math.max(0, Math.round(seconds * 1000));

  const dateMs = Date.parse(trimmed);
  if (Number.isNaN(dateMs)) return null;
  return Math.max(0, dateMs - nowMs);
}

function isHeadersLike(value: object): value is { get(name: string): string | null } {
  return typeof (value as { get?: unknown }).get === 'function';
}

function firstDefined(values: number[]): number | undefined {
  return values.find((value) => value != null);
}

function firstUsefulText(texts: string[]): string | undefined {
  return texts.find((text) => text.length > 0);
}

function statusReason(statusCode: number): string {
  if (statusCode === 429) return 'HTTP 429 rate limit';
  if (statusCode >= 500) return `HTTP ${statusCode} provider/server error`;
  return `HTTP ${statusCode}`;
}
