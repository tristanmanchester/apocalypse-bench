import { afterEach, describe, expect, test, vi } from 'vitest';
import { generateObject } from 'ai';
import {
  __setJudgeDepsForTest,
  judgeWithRepairRetry,
} from '../src/core/runner/judge';

const output = {
  rubric_scores: { r1: 1 },
  auto_fail: false,
  auto_fail_reason: '',
  overall_score: 1,
  notes: 'ok',
};

describe('judge retry handling', () => {
  afterEach(() => {
    __setJudgeDepsForTest({ generateObject });
  });

  test('judge succeeds after retryable provider failure', async () => {
    const generateObject = vi
      .fn()
      .mockRejectedValueOnce(Object.assign(new Error('Provider returned error'), { statusCode: 429 }))
      .mockResolvedValueOnce({
        object: output,
        response: {},
        providerMetadata: {},
        usage: {},
        finishReason: 'stop',
        warnings: [],
        request: {},
      });
    __setJudgeDepsForTest({ generateObject });
    const retryEvents: Array<{ attempt: number; statusCode?: number }> = [];

    const result = await judgeWithRepairRetry(
      {
        model: {} as never,
        prompt: 'score this',
        maxTokens: 100,
        temperature: null,
        rubricIds: ['r1'],
      },
      {
        retry: { maxRetries: 2, baseMs: 1, maxMs: 1 },
        onRetry: (event) => retryEvents.push(event),
      },
    );

    expect(result.object).toEqual(output);
    expect(generateObject).toHaveBeenCalledTimes(2);
    expect(retryEvents).toEqual([
      expect.objectContaining({ attempt: 1, statusCode: 429 }),
    ]);
  });
});
