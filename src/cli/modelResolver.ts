import type { LanguageModel } from 'ai';

import { createOpenAICompatibleClient } from '../adapters/openaiCompatible/client';
import { createOpenRouterClient } from '../adapters/openrouter/client';
import { createOllamaClient } from '../adapters/ollama/client';
import { isOpenRouterJudgeConfig, type ApocbenchConfig } from '../core/config/schema';

type Env = NodeJS.ProcessEnv;
type CandidateModelEntry = ApocbenchConfig['models'][number];

export class MissingEnvVarError extends Error {
  constructor(readonly envVar: string) {
    super(`missing env var: ${envVar}`);
  }
}

function requireEnv(env: Env, envVar: string): string {
  const value = env[envVar];
  if (!value) throw new MissingEnvVarError(envVar);
  return value;
}

export function resolveCandidateModel(
  config: ApocbenchConfig,
  modelEntry: CandidateModelEntry,
  env: Env = process.env,
): LanguageModel {
  if (modelEntry.router === 'openrouter') {
    const apiKey = requireEnv(env, config.routers.openrouter.apiKeyEnv);
    const openrouter = createOpenRouterClient({
      apiKey,
      baseUrl: config.routers.openrouter.baseUrl,
      headers: config.routers.openrouter.headers,
    });
    return openrouter(modelEntry.model, { usage: { include: true } });
  }

  if (modelEntry.router === 'openai-compatible') {
    const routerConfig = config.routers.openaiCompatible;
    if (!routerConfig) {
      throw new Error('missing router config: routers.openaiCompatible');
    }
    const apiKey = routerConfig.apiKeyEnv
      ? requireEnv(env, routerConfig.apiKeyEnv)
      : undefined;
    const provider = createOpenAICompatibleClient({
      baseUrl: routerConfig.baseUrl,
      apiKey,
      headers: routerConfig.headers,
      queryParams: routerConfig.queryParams,
    });
    return provider(modelEntry.model);
  }

  const ollama = createOllamaClient({ baseUrl: config.routers.ollama.baseUrl });
  return ollama(modelEntry.model);
}

export function resolveJudgeModel(
  config: ApocbenchConfig,
  env: Env = process.env,
): LanguageModel {
  if (!isOpenRouterJudgeConfig(config.judge)) {
    throw new Error('Codex judge configs are scored with the batched judge command');
  }
  const apiKey = requireEnv(env, config.routers.openrouter.apiKeyEnv);
  const openrouter = createOpenRouterClient({
    apiKey,
    baseUrl: config.routers.openrouter.baseUrl,
    headers: config.routers.openrouter.headers,
  });
  return openrouter(config.judge.model, { usage: { include: true } });
}
