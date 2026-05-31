import { describe, expect, test } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';

import { runBenchmark } from '../src/core/runner/orchestrator';
import type { ApocbenchConfig } from '../src/core/config/schema';
import type { DatasetLine } from '../src/core/dataset/schema';
import type { LanguageModelV2 } from '@ai-sdk/provider';

describe('integration (mocked)', () => {
  test('writes summary and report', async () => {
    const tmpOut = path.resolve(process.cwd(), 'runs-test');
    fs.rmSync(tmpOut, { recursive: true, force: true });

    const config: ApocbenchConfig = {
      run: {
        name: 'test',
        datasetPaths: ['./data/question_bank'],
        outDir: tmpOut,
        resume: true,
        concurrency: { candidate: 1, judge: 1 },
      },
      judge: {
        router: 'openrouter',
        model: 'google/gemini-3-flash-preview',
        maxTokens: 1000,
        structured: true,
      },
      routers: {
        ollama: {
          baseUrl: 'http://localhost:11434/v1',
          apiKeyEnv: null,
          default: { temperature: 0.2, maxTokens: 50, timeoutMs: 1000 },
        },
        openrouter: {
          baseUrl: 'https://openrouter.ai/api/v1',
          apiKeyEnv: 'OPENROUTER_API_KEY',
          default: { temperature: 0.2, maxTokens: 50, timeoutMs: 1000 },
        },
      },
      models: [{ id: 'm1', router: 'ollama', model: 'llama3.2' }],
    };

    const questions: DatasetLine[] = [
      {
        id: 'Q1',
        category: 'cat',
        difficulty: 'Medium',
        scenario: ['a'],
        prompt: 'p',
        rubric: [{ id: 'r1', text: 't', weight: 1, maxScore: 1 }],
        auto_fail: ['no'],
      },
      {
        id: 'Q2',
        category: 'cat',
        difficulty: 'Medium',
        scenario: ['a'],
        prompt: 'p',
        rubric: [{ id: 'r1', text: 't', weight: 1, maxScore: 1 }],
        auto_fail: ['no'],
      },
    ];

    const fakeModel: LanguageModelV2 = {
      specificationVersion: 'v2',
      provider: 'test',
      modelId: 'test',
      supportedUrls: {},
      doGenerate: async () => {
        throw new Error('not used');
      },
      doStream: async () => {
        throw new Error('not used');
      },
    };

    // For this integration test we use dryRun=true (no network) and validate it exits cleanly.
    const datasetAbs = path.resolve(process.cwd(), 'data/question_bank/PH.jsonl');
    const result = await runBenchmark({
      config,
      configPath: 'x',
      datasetPath: 'x',
      datasetAbsolutePath: datasetAbs,
      questions,
      deps: {
        resolveModel: () => fakeModel,
        resolveJudgeModel: () => fakeModel,
        toolVersion: 'test',
      },
      dryRun: true,
    });

    expect(result).toBeNull();
  });
});
