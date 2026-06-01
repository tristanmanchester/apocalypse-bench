import { describe, expect, test, vi } from 'vitest';
import {
  classifyRetryError,
  computeRetryDelayMs,
  DEFAULT_RETRY_POLICY,
  isRetryableError,
  parseRetryAfterMs,
  shouldRetryWithinBudget,
} from '../src/core/runner/retryPolicy';

describe('retry policy', () => {
  test('classifies retryable OpenRouter and network errors', () => {
    expect(isRetryableError(new Error('429 Provider returned error'))).toBe(true);
    expect(isRetryableError({ cause: { statusCode: 429 } })).toBe(true);
    expect(
      isRetryableError({
        metadata: {
          raw: 'google/gemma-4-26b-a4b-it:free is temporarily rate-limited upstream',
        },
      }),
    ).toBe(true);
    expect(isRetryableError(new Error('500 upstream overloaded'))).toBe(true);
    expect(isRetryableError(Object.assign(new Error('socket reset'), { code: 'ECONNRESET' }))).toBe(
      true,
    );
    expect(isRetryableError(Object.assign(new Error('timed out'), { code: 'ETIMEDOUT' }))).toBe(
      true,
    );
    expect(isRetryableError(Object.assign(new Error('dns'), { code: 'ENOTFOUND' }))).toBe(true);
    expect(isRetryableError(new Error('aborted'))).toBe(true);
  });

  test('classifies permanent request and auth errors as non-retryable', () => {
    expect(isRetryableError(new Error('401 Missing Authentication header'))).toBe(false);
    expect(isRetryableError(new Error('400 unsupported parameter: temperature'))).toBe(false);
    expect(isRetryableError({ response: { status: 422 }, message: 'schema validation failed' })).toBe(
      false,
    );
  });

  test('extracts retry status and retry-after hints', () => {
    const decision = classifyRetryError({
      response: {
        status: 429,
        headers: { 'retry-after': '30' },
      },
      message: 'Provider returned error',
    });

    expect(decision).toMatchObject({
      retryable: true,
      statusCode: 429,
      retryAfterMs: 30000,
    });
  });

  test('parses numeric and HTTP-date Retry-After values', () => {
    expect(parseRetryAfterMs('30', 1_000)).toBe(30000);
    expect(parseRetryAfterMs(2, 1_000)).toBe(2000);
    expect(parseRetryAfterMs('Thu, 01 Jan 1970 00:00:05 GMT', 1_000)).toBe(4000);
  });

  test('computes jittered exponential delay and honors longer Retry-After', () => {
    const random = vi.spyOn(Math, 'random').mockReturnValue(0.5);
    try {
      expect(
        computeRetryDelayMs({
          attempt: 2,
          policy: DEFAULT_RETRY_POLICY,
        }),
      ).toBe(8000);
      expect(
        computeRetryDelayMs({
          attempt: 0,
          policy: DEFAULT_RETRY_POLICY,
          retryAfterMs: 30000,
        }),
      ).toBe(30000);
    } finally {
      random.mockRestore();
    }
  });

  test('enforces max total retry time budget', () => {
    expect(
      shouldRetryWithinBudget({
        startedAtMs: 0,
        nowMs: 1000,
        delayMs: 2000,
        policy: { maxRetries: 3, baseMs: 100, maxMs: 1000, maxTotalTimeMs: 4000 },
      }),
    ).toBe(true);

    expect(
      shouldRetryWithinBudget({
        startedAtMs: 0,
        nowMs: 3000,
        delayMs: 2000,
        policy: { maxRetries: 3, baseMs: 100, maxMs: 1000, maxTotalTimeMs: 4000 },
      }),
    ).toBe(false);
  });
});
