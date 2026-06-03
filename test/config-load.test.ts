import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, expect, test } from 'vitest';
import { loadConfig } from '../src/core/config/loadConfig';

describe('loadConfig', () => {
  test('expands YAML merge keys before schema validation', () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'apocbench-config-'));
    const configPath = path.join(dir, 'apocbench.yml');
    fs.writeFileSync(
      configPath,
      `
run:
  name: merge-config
  datasetPaths: ['./data/question_bank']
  outDir: './runs'
  resume: true
  concurrency:
    candidate: 1
    judge: 1
candidate:
  maxTokens: 100
judge:
  router: openrouter
  model: example/judge
  maxTokens: 100
  structured: true
routers:
  ollama:
    baseUrl: http://127.0.0.1:11434/api
    default: {}
  openrouter:
    baseUrl: https://openrouter.ai/api/v1
    apiKeyEnv: OPENROUTER_API_KEY
    default: {}
models:
  - &base_model
    id: model-direct
    router: openrouter
    model: example/model
    candidateMode: direct
  - <<: *base_model
    id: model-direct-copy
`,
    );

    const { config } = loadConfig(configPath);

    expect(config.models).toHaveLength(2);
    expect(config.models[1]).toMatchObject({
      id: 'model-direct-copy',
      router: 'openrouter',
      model: 'example/model',
      candidateMode: 'direct',
    });
  });
});
