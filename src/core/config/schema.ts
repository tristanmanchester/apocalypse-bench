import { z } from 'zod';

const requestDefaultsSchema = z
  .object({
    temperature: z.number().nullable().optional(),
    maxTokens: z.number().int().positive().optional(),
    timeoutMs: z.number().int().positive().optional(),
  })
  .strict();

const candidateReasoningSchema = z
  .object({
    enabled: z.boolean().default(true),
    effort: z.enum(['minimal', 'low', 'medium', 'high', 'xhigh']).optional(),
    maxTokens: z.number().int().positive().optional(),
    exclude: z.boolean().default(true),
  })
  .strict()
  .superRefine((reasoning, ctx) => {
    if (!reasoning.enabled) return;
    if (!reasoning.effort && !reasoning.maxTokens) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: 'candidate.reasoning requires effort or maxTokens when enabled.',
        path: ['effort'],
      });
    }
    if (reasoning.effort && reasoning.maxTokens) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: 'Provide only one of candidate.reasoning.effort or maxTokens.',
        path: ['maxTokens'],
      });
    }
  });

const retryPolicySchema = z
  .object({
    maxRetries: z.number().int().nonnegative().optional(),
    baseMs: z.number().int().positive().optional(),
    maxMs: z.number().int().positive().optional(),
    maxTotalTimeMs: z.number().int().positive().nullable().optional(),
  })
  .strict();

const openAiCompatibleRouterSchema = z
  .object({
    baseUrl: z.string().min(1),
    apiKeyEnv: z.string().min(1).nullable().optional(),
    headers: z.record(z.string(), z.string()).optional(),
    queryParams: z.record(z.string(), z.string()).optional(),
    default: requestDefaultsSchema,
  })
  .strict();

const openRouterRoutingSchema = z
  .object({
    requireParameters: z.boolean().optional(),
    allowFallbacks: z.boolean().optional(),
    order: z.array(z.string()).optional(),
    only: z.array(z.string()).optional(),
    ignore: z.array(z.string()).optional(),
    sort: z.enum(['price', 'throughput', 'latency']).optional(),
    dataCollection: z.enum(['allow', 'deny']).optional(),
    zdr: z.boolean().optional(),
    maxPrice: z
      .object({
        prompt: z.number().optional(),
        completion: z.number().optional(),
        request: z.number().optional(),
        image: z.number().optional(),
      })
      .strict()
      .optional(),
    quantizations: z.array(z.string()).optional(),
  })
  .strict();

const codexBatchStrategySchema = z.enum([
  'sequential',
  'model',
  'category',
  'category-model',
  'question-paired',
]);

const codexSourceStatusSchema = z.enum(['done', 'candidate_done', 'both']);

const openRouterJudgeSchema = z
  .object({
    backend: z.literal('openrouter').optional(),
    router: z.literal('openrouter'),
    model: z.string().min(1),
    provider: z.string().min(1).optional(),
    temperature: z.number().nullable().optional(),
    maxTokens: z.number().int().positive(),
    structured: z.boolean(),
    reasoning: z.boolean().optional(),
    routing: openRouterRoutingSchema.optional(),
  })
  .strict()
  .transform((judge) => ({ ...judge, backend: judge.backend ?? 'openrouter' }) as const);

const codexJudgeSchema = z
  .object({
    backend: z.literal('codex-cli'),
    model: z.string().min(1).default('gpt-5.5'),
    reasoning: z.string().min(1).default('low'),
    codexBin: z.string().min(1).default('codex'),
    batchSize: z.number().int().positive().default(10),
    batchStrategy: codexBatchStrategySchema.default('question-paired'),
    concurrency: z.number().int().positive().default(1),
    sourceStatus: codexSourceStatusSchema.default('both'),
    maxRetries: z.number().int().nonnegative().default(1),
    tmpDir: z.string().min(1).default('logs/codex-rejudge'),
    disableFeatures: z
      .array(z.string().min(1))
      .default([
        'plugins',
        'apps',
        'memories',
        'tool_suggest',
        'skill_mcp_dependency_install',
      ]),
  })
  .strict();

const judgeSchema = z.union([openRouterJudgeSchema, codexJudgeSchema]);

const candidateModeSchema = z
  .enum([
    'direct',
    'rag-bm25',
    'rag-dense',
    'rag-hybrid',
    'agent-bm25',
    'agent-bm25-research',
    'agent-bm25-research-v2',
    'agent-bm25-rerank-research',
    'agent-bm25-research-read-required',
    'agent-bm25-research-smart-read',
    'agent-hybrid-research-smart-read',
    'agent-dense',
    'agent-hybrid',
    'agent-wiki',
    'agent-rg',
    'agent-literal',
  ])
  .default('direct');

export function isWikiCandidateMode(mode: z.infer<typeof candidateModeSchema>): boolean {
  return mode !== 'direct';
}

export function isAgentCandidateMode(mode: z.infer<typeof candidateModeSchema>): boolean {
  return mode.startsWith('agent-');
}

const wikiConfigSchema = z
  .object({
    enabled: z.boolean().optional(),
    service: z
      .object({
        baseUrl: z.string().min(1),
        timeoutMs: z.number().int().positive().optional(),
      })
      .strict(),
    corpus: z
      .object({
        manifestId: z.string().min(1),
        manifestPath: z.string().min(1).optional(),
      })
      .strict(),
    index: z
      .object({
        manifestId: z.string().min(1),
        manifestPath: z.string().min(1).optional(),
      })
      .strict(),
    limits: z
      .object({
        searchTopK: z.number().int().positive(),
        readMaxChars: z.number().int().positive(),
        contextMaxChars: z.number().int().positive(),
        maxToolCalls: z.number().int().positive().optional(),
        maxTurns: z.number().int().positive().optional(),
      })
      .strict(),
  })
  .strict();

export function toOpenRouterProviderParam(
  routing?: z.infer<typeof openRouterRoutingSchema>,
): Record<string, unknown> | undefined {
  if (!routing) return undefined;

  const provider: Record<string, unknown> = {};

  if (routing.requireParameters != null)
    provider.require_parameters = routing.requireParameters;
  if (routing.allowFallbacks != null) provider.allow_fallbacks = routing.allowFallbacks;
  if (routing.order) provider.order = routing.order;
  if (routing.only) provider.only = routing.only;
  if (routing.ignore) provider.ignore = routing.ignore;
  if (routing.sort) provider.sort = routing.sort;
  if (routing.dataCollection) provider.data_collection = routing.dataCollection;
  if (routing.zdr != null) provider.zdr = routing.zdr;
  if (routing.maxPrice) provider.max_price = routing.maxPrice;
  if (routing.quantizations) provider.quantizations = routing.quantizations;

  return Object.keys(provider).length > 0 ? provider : undefined;
}

export const configSchema = z
  .object({
    run: z
      .object({
        name: z.string().min(1),
        datasetPath: z.string().min(1).optional(),
        datasetPaths: z.array(z.string().min(1)).min(1).optional(),
        outDir: z.string().min(1),
        resume: z.boolean(),
        questionLimit: z.number().int().positive().nullable().optional(),
        questionOrder: z.enum(['sequential', 'shuffle']).default('sequential'),
        questionSeed: z.string().min(1).optional(),
        categories: z.array(z.string().min(1)).nullable().optional(),
        questionIds: z.array(z.string().min(1)).nullable().optional(),
        candidateOnly: z.boolean().optional(),
        maxBudgetUsd: z.number().positive().nullable().optional(),
        retry: retryPolicySchema.optional(),
        concurrency: z
          .object({
            candidate: z.number().int().positive(),
            judge: z.number().int().positive(),
          })
          .strict(),
      })
      .strict()
      .superRefine((run, ctx) => {
        if (run.datasetPath && run.datasetPaths) {
          ctx.addIssue({
            code: z.ZodIssueCode.custom,
            message:
              'Provide exactly one of run.datasetPath (string) or run.datasetPaths (string[]).',
            path: ['datasetPaths'],
          });
        }

        if (!run.datasetPath && !run.datasetPaths) {
          ctx.addIssue({
            code: z.ZodIssueCode.custom,
            message: 'Missing run.datasetPath or run.datasetPaths.',
            path: ['datasetPath'],
          });
        }
      }),

    candidate: z
      .object({
        maxTokens: z.number().int().positive().optional(),
        reasoning: candidateReasoningSchema.optional(),
      })
      .strict()
      .optional(),

    judge: judgeSchema,

    routers: z
      .object({
        ollama: z
          .object({
            baseUrl: z.string().min(1),
            apiKeyEnv: z.string().min(1).nullable().optional(),
            default: requestDefaultsSchema,
          })
          .strict(),
        openrouter: z
          .object({
            baseUrl: z.string().min(1),
            apiKeyEnv: z.string().min(1),
            headers: z.record(z.string(), z.string()).optional(),
            default: requestDefaultsSchema,
          })
          .strict(),
        openaiCompatible: openAiCompatibleRouterSchema.optional(),
      })
      .strict(),

    wiki: wikiConfigSchema.optional(),

    models: z
      .array(
        z
          .object({
            id: z.string().min(1),
            router: z.enum(['ollama', 'openrouter', 'openai-compatible']),
            model: z.string().min(1),
            provider: z.string().min(1).optional(),
            candidateMode: candidateModeSchema,
            params: requestDefaultsSchema.optional(),
            promptFormat: z.string().min(1).optional(),
            routing: openRouterRoutingSchema.optional(),
            concurrency: z.number().int().positive().optional(),
          })
          .strict(),
      )
      .min(1),
  })
  .superRefine((config, ctx) => {
    if (
      config.models.some((model) => model.router === 'openai-compatible') &&
      !config.routers.openaiCompatible
    ) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message:
          'Missing routers.openaiCompatible for models using router=openai-compatible.',
        path: ['routers', 'openaiCompatible'],
      });
    }

    const wikiModelIndex = config.models.findIndex((model) =>
      isWikiCandidateMode(model.candidateMode),
    );
    if (wikiModelIndex !== -1 && !config.wiki) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: 'Missing wiki config for models using wiki candidateMode.',
        path: ['models', wikiModelIndex, 'candidateMode'],
      });
    }
    if (wikiModelIndex !== -1 && config.wiki?.enabled === false) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: 'wiki.enabled must be true for models using wiki candidateMode.',
        path: ['wiki', 'enabled'],
      });
    }
  })
  .strict();

export type ApocbenchConfig = z.input<typeof configSchema>;
export type ParsedApocbenchConfig = z.infer<typeof configSchema>;
export type CandidateMode = z.infer<typeof candidateModeSchema>;
export type WikiConfig = z.infer<typeof wikiConfigSchema>;
export type CodexJudgeConfig = z.infer<typeof codexJudgeSchema>;
export type OpenRouterJudgeConfig = z.infer<typeof openRouterJudgeSchema>;

export function isCodexJudgeConfig(
  judge: ApocbenchConfig['judge'],
): judge is CodexJudgeConfig {
  return judge.backend === 'codex-cli';
}

export function isOpenRouterJudgeConfig(
  judge: ApocbenchConfig['judge'],
): judge is OpenRouterJudgeConfig {
  return (
    judge.backend === 'openrouter' || ('router' in judge && judge.router === 'openrouter')
  );
}
