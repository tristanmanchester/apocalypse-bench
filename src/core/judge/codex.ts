import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import type Database from 'better-sqlite3';
import PQueue from 'p-queue';

import type { DatasetLine } from '../dataset/schema';
import { loadJsonlMany } from '../dataset/loadJsonl';
import { buildJudgePrompt } from '../prompts/judgePrompt';
import { computeOverallScore } from '../runner/judge';
import type { JudgeOutput } from '../runner/types';
import { openAndMigrate } from '../../storage/sqlite/migrate';
import { insertQuestions } from '../../storage/sqlite/questions';
import { getRun, insertRun, updateRunStatus } from '../../storage/sqlite/runs';
import { upsertResult } from '../../storage/sqlite/results';
import { sha256Hex } from '../../utils/hash';

export type BatchStrategy =
  | 'sequential'
  | 'model'
  | 'category'
  | 'category-model'
  | 'question-paired';

export type CodexRejudgeArgs = {
  db: string;
  sourceRun: string;
  outRun?: string;
  dataset: string;
  codexBin: string;
  model: string;
  reasoning: string;
  disableFeatures: string[];
  batchSize: number;
  batchStrategy: BatchStrategy;
  concurrency: number;
  maxRetries: number;
  tmpDir: string;
  limit?: number;
  models?: string[];
  sourceStatus: 'done' | 'candidate_done' | 'both';
  resume: boolean;
};

export const DEFAULT_CODEX_DISABLE_FEATURES = [
  'plugins',
  'apps',
  'memories',
  'tool_suggest',
  'skill_mcp_dependency_install',
];

export type SourceResultRow = {
  question_id: string;
  model_id: string;
  status: string;
  candidate_prompt: string | null;
  candidate_completion: string | null;
  candidate_metrics_json: string | null;
  retrieval_trace_json: string | null;
};

export type JudgeCase = {
  row: SourceResultRow;
  question: DatasetLine;
};

export type JudgeBatch = {
  index: number;
  category: string | null;
  modelId: string | null;
  cases: JudgeCase[];
};

type CodexJudgeItem = JudgeOutput & {
  model_id: string;
  question_id: string;
};

export type CodexBatchOutput = {
  results: CodexJudgeItem[];
};

function usage(): never {
  console.error(`Usage:
  pnpm -s rejudge:codex -- --source-run <run-id> [options]

Options:
  --db <path>             SQLite DB path (default: runs/apocbench.sqlite)
  --out-run <run-id>      Destination run ID (default: <source>-codex-gpt55-low-<timestamp>)
  --dataset <path>        Question bank file or directory (default: data/question_bank)
  --codex-bin <path>      Codex CLI binary (default: codex)
  --model <model>         Codex model (default: gpt-5.5)
  --reasoning <level>     Codex reasoning effort (default: low)
  --disable-feature <f>   Disable a Codex feature for exec (repeatable; default: plugins, apps, memories, tool_suggest, skill_mcp_dependency_install)
  --no-default-disable-features
                          Do not apply the default feature disables
  --batch-size <n>        Cases per Codex call (default: 5)
  --batch-strategy <s>    sequential | model | category | category-model | question-paired (default: sequential)
  --concurrency <n>       Concurrent Codex batch processes (default: 1)
  --max-retries <n>       Retries per batch after parse/validation failure (default: 1)
  --tmp-dir <path>        Working/log directory (default: logs/codex-rejudge)
  --limit <n>             Limit selected result rows, for smoke tests
  --models <a,b,c>        Restrict source rows to model IDs
  --source-status <s>     Source row status: done | candidate_done | both (default: done)
  --resume                Resume an existing destination run`);
  process.exit(2);
}

export function parseCodexRejudgeArgs(argv: string[]): CodexRejudgeArgs {
  const args: CodexRejudgeArgs = {
    db: 'runs/apocbench.sqlite',
    dataset: 'data/question_bank',
    codexBin: 'codex',
    model: 'gpt-5.5',
    reasoning: 'low',
    disableFeatures: [...DEFAULT_CODEX_DISABLE_FEATURES],
    batchSize: 5,
    batchStrategy: 'sequential',
    concurrency: 1,
    maxRetries: 1,
    tmpDir: 'logs/codex-rejudge',
    sourceStatus: 'done',
    resume: false,
    sourceRun: '',
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = () => {
      const value = argv[i + 1];
      if (!value || value.startsWith('--')) usage();
      i += 1;
      return value;
    };

    switch (arg) {
      case '--':
        break;
      case '--db':
        args.db = next();
        break;
      case '--source-run':
        args.sourceRun = next();
        break;
      case '--out-run':
        args.outRun = next();
        break;
      case '--dataset':
        args.dataset = next();
        break;
      case '--codex-bin':
        args.codexBin = next();
        break;
      case '--model':
        args.model = next();
        break;
      case '--reasoning':
        args.reasoning = next();
        break;
      case '--disable-feature':
        args.disableFeatures.push(next());
        break;
      case '--no-default-disable-features':
        args.disableFeatures = [];
        break;
      case '--batch-size':
        args.batchSize = Number.parseInt(next(), 10);
        break;
      case '--batch-strategy': {
        const value = next();
        const normalized = value === 'category-balanced' ? 'category' : value;
        if (
          normalized !== 'sequential' &&
          normalized !== 'model' &&
          normalized !== 'category' &&
          normalized !== 'category-model' &&
          normalized !== 'question-paired'
        ) {
          throw new Error(
            '--batch-strategy must be sequential, model, category, category-model, or question-paired',
          );
        }
        args.batchStrategy = normalized;
        break;
      }
      case '--concurrency':
        args.concurrency = Number.parseInt(next(), 10);
        break;
      case '--max-retries':
        args.maxRetries = Number.parseInt(next(), 10);
        break;
      case '--tmp-dir':
        args.tmpDir = next();
        break;
      case '--limit':
        args.limit = Number.parseInt(next(), 10);
        break;
      case '--models':
        args.models = next()
          .split(',')
          .map((model) => model.trim())
          .filter(Boolean);
        break;
      case '--source-status': {
        const value = next();
        if (value !== 'done' && value !== 'candidate_done' && value !== 'both') {
          throw new Error('--source-status must be done, candidate_done, or both');
        }
        args.sourceStatus = value;
        break;
      }
      case '--resume':
        args.resume = true;
        break;
      case '--help':
      case '-h':
        usage();
        break;
      default:
        console.error(`Unknown argument: ${arg}`);
        usage();
    }
  }

  if (!args.sourceRun) usage();
  if (!Number.isInteger(args.batchSize) || args.batchSize < 1) {
    throw new Error('--batch-size must be a positive integer');
  }
  if (!Number.isInteger(args.concurrency) || args.concurrency < 1) {
    throw new Error('--concurrency must be a positive integer');
  }
  if (!Number.isInteger(args.maxRetries) || args.maxRetries < 0) {
    throw new Error('--max-retries must be a non-negative integer');
  }
  if (args.limit !== undefined && (!Number.isInteger(args.limit) || args.limit < 1)) {
    throw new Error('--limit must be a positive integer');
  }

  return args;
}

function timestamp(): string {
  return new Date().toISOString().replace(/[-:]/g, '').replace(/\..+$/, 'Z');
}

function makeOutputRunId(sourceRun: string, model: string, reasoning: string): string {
  const safeModel = model
    .replace(/[^a-zA-Z0-9]+/g, '-')
    .replace(/^-|-$/g, '')
    .toLowerCase();
  return `${sourceRun}-codex-${safeModel}-${reasoning}-${timestamp()}`;
}

function loadQuestions(datasetPath: string): Map<string, DatasetLine> {
  const { lines: questions } = loadJsonlMany([datasetPath]);
  return new Map(questions.map((question) => [question.id, question]));
}

function loadSourceRows(
  db: Database.Database,
  args: CodexRejudgeArgs,
): SourceResultRow[] {
  let sql = `
    select question_id, model_id, status, candidate_prompt, candidate_completion,
           candidate_metrics_json, retrieval_trace_json
    from model_results
    where run_id = ?
  `;
  const params: unknown[] = [args.sourceRun];

  if (args.sourceStatus === 'both') {
    sql += ` and status in ('done', 'candidate_done')`;
  } else {
    sql += ` and status = ?`;
    params.push(args.sourceStatus);
  }

  if (args.models && args.models.length > 0) {
    sql += ` and model_id in (${args.models.map(() => '?').join(',')})`;
    params.push(...args.models);
  }

  sql +=
    ' and candidate_completion is not null and length(trim(candidate_completion)) > 0';
  sql += ' order by question_id, model_id';
  if (args.limit !== undefined) {
    sql += ' limit ?';
    params.push(args.limit);
  }

  return db.prepare(sql).all(...params) as SourceResultRow[];
}

function existingDoneKeys(db: Database.Database, runId: string): Set<string> {
  const rows = db
    .prepare(
      "select question_id, model_id from model_results where run_id = ? and status = 'done'",
    )
    .all(runId) as Array<{ question_id: string; model_id: string }>;
  return new Set(rows.map((row) => `${row.question_id}\u0000${row.model_id}`));
}

export function setupRun(
  db: Database.Database,
  args: CodexRejudgeArgs,
  outRun: string,
  selectedQuestions: DatasetLine[],
): void {
  const existing = getRun(db, outRun);
  if (existing && !args.resume) {
    throw new Error(
      `Destination run already exists: ${outRun}. Pass --resume to continue it.`,
    );
  }

  if (!existing) {
    const source = getRun(db, args.sourceRun);
    const selectedIds = selectedQuestions.map((question) => question.id);
    const config = {
      mode: 'codex-rejudge',
      sourceRun: args.sourceRun,
      judge: {
        backend: 'codex-cli',
        model: args.model,
        reasoning: args.reasoning,
        disableFeatures: args.disableFeatures,
        batchSize: args.batchSize,
        batchStrategy: args.batchStrategy,
        concurrency: args.concurrency,
        maxRetries: args.maxRetries,
      },
      filters: {
        models: args.models ?? null,
        limit: args.limit ?? null,
        sourceStatus: args.sourceStatus,
      },
      sourceConfig: source?.config_json ? JSON.parse(source.config_json) : null,
    };

    insertRun(db, {
      run_id: outRun,
      created_at: new Date().toISOString(),
      tool_version: 'codex-cli-rejudge',
      config_json: JSON.stringify(config),
      dataset_path: args.dataset,
      dataset_sha256:
        source?.dataset_sha256 ??
        sha256Hex(JSON.stringify({ dataset: args.dataset, selectedIds })),
      prompt_template_hash: sha256Hex(
        JSON.stringify({ judgePrompt: 'buildJudgePrompt', model: args.model }),
      ),
      status: 'running',
    });
    insertQuestions(db, outRun, selectedQuestions);
    return;
  }

  insertQuestions(db, outRun, selectedQuestions);
  updateRunStatus(db, outRun, 'running');
}

export function outputSchemaForBatch(batchSize: number): Record<string, unknown> {
  const rubricScoreSchema = {
    type: 'object',
    additionalProperties: false,
    required: ['r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9', 'r10'],
    properties: Object.fromEntries(
      Array.from({ length: 10 }, (_, index) => [`r${index + 1}`, { type: 'number' }]),
    ),
  };

  return {
    type: 'object',
    additionalProperties: false,
    required: ['results'],
    properties: {
      results: {
        type: 'array',
        minItems: batchSize,
        maxItems: batchSize,
        items: {
          type: 'object',
          additionalProperties: false,
          required: [
            'model_id',
            'question_id',
            'rubric_scores',
            'auto_fail',
            'auto_fail_reason',
            'overall_score',
            'notes',
            'unsafe_flags',
          ],
          properties: {
            model_id: { type: 'string' },
            question_id: { type: 'string' },
            rubric_scores: {
              ...rubricScoreSchema,
            },
            auto_fail: { type: 'boolean' },
            auto_fail_reason: { type: 'string' },
            overall_score: { type: 'number' },
            notes: { type: 'string' },
            unsafe_flags: {
              type: 'array',
              items: { type: 'string' },
            },
          },
        },
      },
    },
  };
}

function compareStrings(left: string, right: string): number {
  return left.localeCompare(right, 'en');
}

function compareCasesForCategory(left: JudgeCase, right: JudgeCase): number {
  return (
    compareStrings(left.question.category, right.question.category) ||
    compareStrings(left.row.question_id, right.row.question_id) ||
    compareStrings(left.row.model_id, right.row.model_id)
  );
}

function compareCasesForCategoryModel(left: JudgeCase, right: JudgeCase): number {
  return (
    compareStrings(left.question.category, right.question.category) ||
    compareStrings(left.row.model_id, right.row.model_id) ||
    compareStrings(left.row.question_id, right.row.question_id)
  );
}

function compareCasesForModel(left: JudgeCase, right: JudgeCase): number {
  return (
    compareStrings(left.row.model_id, right.row.model_id) ||
    compareStrings(left.row.question_id, right.row.question_id)
  );
}

function modeRank(modelId: string): number {
  if (modelId.endsWith('-direct') || modelId.includes('-direct-')) return 0;
  if (
    modelId.endsWith('-agent-bm25-research-v2') ||
    modelId.includes('-agent-bm25-research-v2-')
  ) {
    return 2;
  }
  if (
    modelId.endsWith('-agent-bm25-rerank-research') ||
    modelId.includes('-agent-bm25-rerank-research-')
  ) {
    return 3;
  }
  if (
    modelId.endsWith('-agent-bm25-research-smart-read') ||
    modelId.includes('-agent-bm25-research-smart-read-')
  ) {
    return 4;
  }
  if (
    modelId.endsWith('-agent-hybrid-research-smart-read') ||
    modelId.includes('-agent-hybrid-research-smart-read-')
  ) {
    return 5;
  }
  if (
    modelId.endsWith('-agent-bm25-research') ||
    modelId.endsWith('-agent-bm25-research-read-required') ||
    modelId.includes('-agent-bm25-research-') ||
    modelId.includes('-agent-bm25-research-read-required-')
  ) {
    return 1;
  }
  return 2;
}

function baseModelFamily(modelId: string): string {
  return modelId
    .replace(/-agent-hybrid-research-smart-read(?=$|-)/, '')
    .replace(/-agent-bm25-research-smart-read(?=$|-)/, '')
    .replace(/-agent-bm25-research-read-required(?=$|-)/, '')
    .replace(/-agent-bm25-rerank-research(?=$|-)/, '')
    .replace(/-agent-bm25-research-v2(?=$|-)/, '')
    .replace(/-agent-bm25-research(?=$|-)/, '')
    .replace(/-direct(?=$|-)/, '');
}

function compareCasesForQuestionPaired(left: JudgeCase, right: JudgeCase): number {
  return (
    compareStrings(left.row.question_id, right.row.question_id) ||
    compareStrings(
      baseModelFamily(left.row.model_id),
      baseModelFamily(right.row.model_id),
    ) ||
    modeRank(left.row.model_id) - modeRank(right.row.model_id) ||
    compareStrings(left.row.model_id, right.row.model_id)
  );
}

function makeBatch(
  index: number,
  cases: JudgeCase[],
  strategy: BatchStrategy,
): JudgeBatch {
  const categories = new Set(cases.map((judgeCase) => judgeCase.question.category));
  const modelIds = new Set(cases.map((judgeCase) => judgeCase.row.model_id));
  return {
    index,
    category:
      strategy === 'sequential' || strategy === 'model' || categories.size !== 1
        ? null
        : (cases[0]?.question.category ?? null),
    modelId:
      (strategy === 'category-model' || strategy === 'model') && modelIds.size === 1
        ? (cases[0]?.row.model_id ?? null)
        : null,
    cases,
  };
}

export function buildJudgeBatches(
  cases: JudgeCase[],
  batchSize: number,
  strategy: BatchStrategy = 'sequential',
): JudgeBatch[] {
  if (!Number.isInteger(batchSize) || batchSize < 1) {
    throw new Error('batchSize must be a positive integer');
  }

  if (strategy === 'sequential') {
    const batches: JudgeBatch[] = [];
    for (let offset = 0; offset < cases.length; offset += batchSize) {
      batches.push(
        makeBatch(batches.length + 1, cases.slice(offset, offset + batchSize), strategy),
      );
    }
    return batches;
  }

  const sorted = [...cases].sort(
    strategy === 'category'
      ? compareCasesForCategory
      : strategy === 'model'
        ? compareCasesForModel
        : strategy === 'question-paired'
          ? compareCasesForQuestionPaired
          : compareCasesForCategoryModel,
  );
  const groups = new Map<string, JudgeCase[]>();
  for (const judgeCase of sorted) {
    const key =
      strategy === 'category'
        ? judgeCase.question.category
        : strategy === 'model'
          ? judgeCase.row.model_id
          : strategy === 'question-paired'
            ? judgeCase.row.question_id
            : `${judgeCase.question.category}\u0000${judgeCase.row.model_id}`;
    const group = groups.get(key);
    if (group) {
      group.push(judgeCase);
    } else {
      groups.set(key, [judgeCase]);
    }
  }

  const batches: JudgeBatch[] = [];
  for (const groupCases of groups.values()) {
    for (let offset = 0; offset < groupCases.length; offset += batchSize) {
      batches.push(
        makeBatch(
          batches.length + 1,
          groupCases.slice(offset, offset + batchSize),
          strategy,
        ),
      );
    }
  }
  return batches;
}

export function buildBatchPrompt(batch: JudgeBatch): string {
  const questionIds = new Set(batch.cases.map((judgeCase) => judgeCase.row.question_id));
  const singleQuestionId =
    questionIds.size === 1 ? (batch.cases[0]?.row.question_id ?? null) : null;
  const calibrationText = singleQuestionId
    ? `This batch is question-paired for question "${singleQuestionId}". You may use the shared question only to keep scoring calibrated and consistent across direct and retrieval candidates for the same task.`
    : batch.category
      ? `This batch is category-local for category "${batch.category}". You may use the shared category only to keep scoring calibrated and consistent across similar tasks.`
      : 'This batch may contain unrelated categories. Use the surrounding cases only to keep rubric interpretation consistent.';
  const caseText = batch.cases
    .map((judgeCase, index) => {
      const requiredRubricIds = judgeCase.question.rubric
        .map((item) => item.id)
        .join(', ');
      const singlePrompt = buildJudgePrompt({
        question: judgeCase.question,
        candidateAnswer: judgeCase.row.candidate_completion ?? '',
      });
      return `<case index="${index + 1}" question_id="${judgeCase.row.question_id}" model_id="${judgeCase.row.model_id}">
Required output identifiers:
- question_id: ${judgeCase.row.question_id}
- model_id: ${judgeCase.row.model_id}
- required rubric IDs: ${requiredRubricIds}

${singlePrompt}
</case>`;
    })
    .join('\n\n');

  return `You are rejudging apocalypse-bench candidate answers.

${calibrationText}

Judge each case independently using only that case's content. Do not inspect files, do not browse, do not call tools, and do not use outside context except ordinary language understanding. The benchmark question, rubric, auto-fails, reference facts, and candidate answer are all included inside each case.

Do not use facts, wording, or missing details from one candidate answer to fill gaps in another candidate answer. A candidate receives credit only for content present in its own answer. Do not rank answers against each other or assign relative leaderboard scores; every score must reflect absolute compliance with that case's rubric and auto-fail rules.

Return exactly one JSON object matching the provided schema. The top-level "results" array must contain one result for every case. Each result must repeat the exact "question_id" and "model_id". Include every required rubric ID in "rubric_scores". If there are no unsafe flags, return an empty "unsafe_flags" array.

${caseText}`;
}

function runCodex(
  args: CodexRejudgeArgs,
  outRun: string,
  batch: JudgeBatch,
  prompt: string,
): Promise<CodexBatchOutput> {
  return new Promise((resolve, reject) => {
    const runDir = path.resolve(args.tmpDir, outRun);
    mkdirSync(runDir, { recursive: true });
    const batchIndex = batch.index;
    const schemaPath = path.join(
      runDir,
      `batch-${String(batchIndex).padStart(4, '0')}.schema.json`,
    );
    const outputPath = path.join(
      runDir,
      `batch-${String(batchIndex).padStart(4, '0')}.output.json`,
    );
    const logPath = path.join(runDir, `batch-${String(batchIndex).padStart(4, '0')}.log`);
    writeFileSync(
      schemaPath,
      JSON.stringify(outputSchemaForBatch(batch.cases.length), null, 2),
    );

    const child = spawn(
      args.codexBin,
      [
        'exec',
        '--ephemeral',
        '--skip-git-repo-check',
        '--sandbox',
        'read-only',
        ...args.disableFeatures.flatMap((feature) => ['--disable', feature]),
        '-c',
        'approval_policy="never"',
        '-c',
        `model_reasoning_effort="${args.reasoning}"`,
        '-c',
        'model_verbosity="low"',
        '--model',
        args.model,
        '--output-schema',
        schemaPath,
        '-o',
        outputPath,
        '-',
      ],
      {
        cwd: process.cwd(),
        env: process.env,
        stdio: ['pipe', 'pipe', 'pipe'],
      },
    );

    let stdout = '';
    let stderr = '';
    child.stdout.setEncoding('utf8');
    child.stderr.setEncoding('utf8');
    child.stdout.on('data', (chunk: string) => {
      stdout += chunk;
    });
    child.stderr.on('data', (chunk: string) => {
      stderr += chunk;
    });
    child.on('error', reject);
    child.on('close', (code) => {
      writeFileSync(
        logPath,
        JSON.stringify(
          {
            batchStrategy: args.batchStrategy,
            batchCategory: batch.category,
            batchModelId: batch.modelId,
            batchSize: batch.cases.length,
            batchIndex,
            batchQuestionIds: batch.cases.map((judgeCase) => judgeCase.row.question_id),
            batchModelIds: batch.cases.map((judgeCase) => judgeCase.row.model_id),
            code,
            cases: batch.cases.map((judgeCase) => ({
              questionId: judgeCase.row.question_id,
              modelId: judgeCase.row.model_id,
              category: judgeCase.question.category,
            })),
            stdout,
            stderr,
          },
          null,
          2,
        ),
      );

      if (code !== 0) {
        reject(
          new Error(
            `codex exec failed for batch ${batchIndex} with exit code ${code}. See ${logPath}`,
          ),
        );
        return;
      }
      if (!existsSync(outputPath)) {
        reject(
          new Error(
            `codex exec did not create output file for batch ${batchIndex}. See ${logPath}`,
          ),
        );
        return;
      }

      try {
        resolve(JSON.parse(readFileSync(outputPath, 'utf8')) as CodexBatchOutput);
      } catch (error) {
        reject(
          new Error(
            `Failed to parse Codex output for batch ${batchIndex}: ${String(error)}. See ${outputPath}`,
          ),
        );
      }
    });
    child.stdin.end(prompt);
  });
}

export function validateBatchOutput(
  batch: JudgeCase[],
  output: CodexBatchOutput,
): Map<string, CodexJudgeItem> {
  if (!output || !Array.isArray(output.results)) {
    throw new Error('Codex output missing results array');
  }
  if (output.results.length !== batch.length) {
    throw new Error(
      `Codex output returned ${output.results.length} results for ${batch.length} cases`,
    );
  }

  const expectedKeys = new Set(
    batch.map(
      (judgeCase) => `${judgeCase.row.question_id}\u0000${judgeCase.row.model_id}`,
    ),
  );
  const byKey = new Map<string, CodexJudgeItem>();
  for (const item of output.results) {
    if (typeof item.question_id !== 'string' || typeof item.model_id !== 'string') {
      throw new Error('Codex output has malformed question_id or model_id');
    }
    const key = `${item.question_id}\u0000${item.model_id}`;
    if (!expectedKeys.has(key)) {
      throw new Error(
        `Codex output returned unexpected result for ${item.question_id} / ${item.model_id}`,
      );
    }
    if (byKey.has(key)) {
      throw new Error(
        `Codex output returned duplicate result for ${item.question_id} / ${item.model_id}`,
      );
    }
    byKey.set(key, item);
  }

  for (const judgeCase of batch) {
    const key = `${judgeCase.row.question_id}\u0000${judgeCase.row.model_id}`;
    const item = byKey.get(key);
    if (!item) {
      throw new Error(
        `Codex output missing result for ${judgeCase.row.question_id} / ${judgeCase.row.model_id}`,
      );
    }
    for (const rubricItem of judgeCase.question.rubric) {
      const value = item.rubric_scores[rubricItem.id];
      if (typeof value !== 'number' || Number.isNaN(value)) {
        throw new Error(
          `Codex output missing numeric rubric score ${rubricItem.id} for ${judgeCase.row.question_id} / ${judgeCase.row.model_id}`,
        );
      }
    }
    if (typeof item.auto_fail !== 'boolean') {
      throw new Error(
        `Codex output has non-boolean auto_fail for ${judgeCase.row.question_id} / ${judgeCase.row.model_id}`,
      );
    }
    if (typeof item.overall_score !== 'number' || Number.isNaN(item.overall_score)) {
      throw new Error(
        `Codex output has non-numeric overall_score for ${judgeCase.row.question_id} / ${judgeCase.row.model_id}`,
      );
    }
    if (!Array.isArray(item.unsafe_flags)) {
      throw new Error(
        `Codex output has non-array unsafe_flags for ${judgeCase.row.question_id} / ${judgeCase.row.model_id}`,
      );
    }
  }

  return byKey;
}

export function normalizeCodexJudgeOutput(params: {
  judgeOutput: JudgeOutput;
  rubric: DatasetLine['rubric'];
}): JudgeOutput {
  const computed = computeOverallScore({
    judgeOutput: params.judgeOutput,
    rubric: params.rubric.map((item) => ({
      id: item.id,
      weight: item.weight,
      maxScore: item.maxScore,
    })),
  });

  return {
    ...params.judgeOutput,
    rubric_scores: computed.rubricScores,
    overall_score: computed.overallScore,
  };
}

async function judgeBatchWithRetry(
  args: CodexRejudgeArgs,
  outRun: string,
  batch: JudgeBatch,
): Promise<Map<string, CodexJudgeItem>> {
  const prompt = buildBatchPrompt(batch);
  let lastError: unknown;

  for (let attempt = 0; attempt <= args.maxRetries; attempt += 1) {
    try {
      const output = await runCodex(args, outRun, batch, prompt);
      return validateBatchOutput(batch.cases, output);
    } catch (error) {
      lastError = error;
      console.error(
        `[codex-rejudge] batch ${batch.index} attempt ${attempt + 1}/${args.maxRetries + 1} failed: ${String(error)}`,
      );
    }
  }

  throw lastError instanceof Error ? lastError : new Error(String(lastError));
}

function storeDone(
  db: Database.Database,
  outRun: string,
  args: CodexRejudgeArgs,
  batch: JudgeBatch,
  judgeCase: JudgeCase,
  item: CodexJudgeItem,
): void {
  const judgeParsed = normalizeCodexJudgeOutput({
    judgeOutput: item,
    rubric: judgeCase.question.rubric,
  });

  upsertResult(db, {
    runId: outRun,
    questionId: judgeCase.row.question_id,
    modelId: judgeCase.row.model_id,
    status: 'done',
    candidatePrompt: judgeCase.row.candidate_prompt ?? undefined,
    candidateCompletion: judgeCase.row.candidate_completion ?? undefined,
    candidateMetricsJson: judgeCase.row.candidate_metrics_json ?? undefined,
    retrievalTraceJson: judgeCase.row.retrieval_trace_json ?? undefined,
    judgeRequestJson: JSON.stringify({
      backend: 'codex-cli',
      model: args.model,
      reasoning: args.reasoning,
      batchStrategy: args.batchStrategy,
      batchCategory: batch.category,
      batchModelId: batch.modelId,
      batchSize: batch.cases.length,
      batchIndex: batch.index,
      batchQuestionIds: batch.cases.map((batchCase) => batchCase.row.question_id),
      batchModelIds: batch.cases.map((batchCase) => batchCase.row.model_id),
      questionId: judgeCase.row.question_id,
      modelId: judgeCase.row.model_id,
    }),
    judgeResponseJson: JSON.stringify(item),
    judgeParsedJson: JSON.stringify(judgeParsed),
    scoreOverall: judgeParsed.overall_score,
    scoreRubricJson: JSON.stringify(judgeParsed.rubric_scores),
    autoFail: judgeParsed.auto_fail,
    autoFailReason: judgeParsed.auto_fail_reason,
  });
}

function storeJudgeFailure(
  db: Database.Database,
  outRun: string,
  judgeCase: JudgeCase,
  error: unknown,
): void {
  upsertResult(db, {
    runId: outRun,
    questionId: judgeCase.row.question_id,
    modelId: judgeCase.row.model_id,
    status: 'judge_failed',
    candidatePrompt: judgeCase.row.candidate_prompt ?? undefined,
    candidateCompletion: judgeCase.row.candidate_completion ?? undefined,
    candidateMetricsJson: judgeCase.row.candidate_metrics_json ?? undefined,
    retrievalTraceJson: judgeCase.row.retrieval_trace_json ?? undefined,
    errorJson: JSON.stringify({ error: String(error) }),
  });
}

type ScoreRow = {
  question_id: string;
  model_id: string;
  score_overall: number | null;
  auto_fail: number | null;
};

function pairedDeltaSummary(
  rows: ScoreRow[],
  baselineModel: string,
  comparisonModel: string,
): Record<string, unknown> {
  const baseline = new Map<string, number>();
  for (const row of rows) {
    if (row.model_id === baselineModel && typeof row.score_overall === 'number') {
      baseline.set(row.question_id, row.score_overall);
    }
  }

  const deltas: number[] = [];
  for (const row of rows) {
    if (row.model_id !== comparisonModel || typeof row.score_overall !== 'number')
      continue;
    const baseScore = baseline.get(row.question_id);
    if (baseScore === undefined) continue;
    deltas.push(row.score_overall - baseScore);
  }

  const meanDelta =
    deltas.length === 0
      ? null
      : deltas.reduce((total, delta) => total + delta, 0) / deltas.length;
  return {
    baselineModel,
    comparisonModel,
    pairedCount: deltas.length,
    meanDelta,
    wins: deltas.filter((delta) => delta > 0).length,
    losses: deltas.filter((delta) => delta < 0).length,
    ties: deltas.filter((delta) => delta === 0).length,
  };
}

function summarizeRun(db: Database.Database, runId: string): Record<string, unknown> {
  const statusCounts = db
    .prepare(
      'select status, count(*) as count from model_results where run_id = ? group by status order by status',
    )
    .all(runId) as Array<{ status: string; count: number }>;
  const modelScores = db
    .prepare(
      `select model_id as modelId,
              count(*) as n,
              sum(case when status = 'done' then 1 else 0 end) as done,
              sum(case when status != 'done' then 1 else 0 end) as failures,
              avg(case when status = 'done' then score_overall else null end) as avgScore,
              sum(case when status = 'done' and auto_fail then 1 else 0 end) as autoFails
       from model_results
       where run_id = ?
       group by model_id
       order by model_id`,
    )
    .all(runId) as Array<{
    modelId: string;
    n: number;
    done: number;
    failures: number;
    avgScore: number | null;
    autoFails: number | null;
  }>;
  const scoreRows = db
    .prepare(
      `select question_id, model_id, score_overall, auto_fail
       from model_results
       where run_id = ? and status = 'done'
       order by question_id, model_id`,
    )
    .all(runId) as ScoreRow[];
  const modelIds = [...new Set(scoreRows.map((row) => row.model_id))].sort(
    compareStrings,
  );
  const directModel =
    modelIds.find((modelId) => modelId.endsWith('-direct')) ??
    modelIds.find((modelId) => modelId.includes('direct'));
  const pairedDeltasVsDirect = directModel
    ? modelIds
        .filter((modelId) => modelId !== directModel)
        .map((modelId) => pairedDeltaSummary(scoreRows, directModel, modelId))
    : [];
  const retrievalModels = modelIds.filter((modelId) =>
    /agent-(bm25|dense|hybrid)/.test(modelId),
  );
  const pairedDeltasBetweenRetrieval = [];
  for (let i = 0; i < retrievalModels.length; i += 1) {
    for (let j = i + 1; j < retrievalModels.length; j += 1) {
      pairedDeltasBetweenRetrieval.push(
        pairedDeltaSummary(scoreRows, retrievalModels[i], retrievalModels[j]),
      );
    }
  }
  const judgeBatchFailureCount =
    statusCounts.find((row) => row.status === 'judge_failed')?.count ?? 0;

  return {
    runId,
    statusCounts,
    modelScores,
    pairedDeltasVsDirect,
    pairedDeltasBetweenRetrieval,
    judgeBatchFailureCount,
  };
}

export async function runCodexRejudge(
  args: CodexRejudgeArgs,
): Promise<Record<string, unknown>> {
  const outRun =
    args.outRun ?? makeOutputRunId(args.sourceRun, args.model, args.reasoning);
  const db = openAndMigrate(args.db);
  const questionsById = loadQuestions(args.dataset);
  const sourceRows = loadSourceRows(db, args);

  if (sourceRows.length === 0) {
    throw new Error(`No candidate-complete source rows found for ${args.sourceRun}`);
  }

  const selectedQuestions = Array.from(
    new Map(
      sourceRows.map((row) => {
        const question = questionsById.get(row.question_id);
        if (!question)
          throw new Error(`Question ${row.question_id} not found in ${args.dataset}`);
        return [question.id, question] as const;
      }),
    ).values(),
  );

  setupRun(db, args, outRun, selectedQuestions);

  const doneKeys = args.resume ? existingDoneKeys(db, outRun) : new Set<string>();
  const cases = sourceRows
    .filter((row) => !doneKeys.has(`${row.question_id}\u0000${row.model_id}`))
    .map((row) => {
      const question = questionsById.get(row.question_id);
      if (!question)
        throw new Error(`Question ${row.question_id} not found in ${args.dataset}`);
      return { row, question };
    });

  console.log(
    `[codex-rejudge] source=${args.sourceRun} out=${outRun} selected=${sourceRows.length} pending=${cases.length} batchSize=${args.batchSize} batchStrategy=${args.batchStrategy} concurrency=${args.concurrency}`,
  );

  const batches = buildJudgeBatches(cases, args.batchSize, args.batchStrategy);
  let completed = sourceRows.length - cases.length;
  const queue = new PQueue({ concurrency: args.concurrency });
  await Promise.all(
    batches.map((batch) =>
      queue.add(async () => {
        try {
          const outputByKey = await judgeBatchWithRetry(args, outRun, batch);
          const tx = db.transaction(() => {
            for (const judgeCase of batch.cases) {
              const item = outputByKey.get(
                `${judgeCase.row.question_id}\u0000${judgeCase.row.model_id}`,
              );
              if (!item)
                throw new Error('validated output map unexpectedly missing item');
              storeDone(db, outRun, args, batch, judgeCase, item);
            }
          });
          tx();
          completed += batch.cases.length;
          console.log(
            `[codex-rejudge] batch ${batch.index} done category=${batch.category ?? 'mixed'} (${completed}/${sourceRows.length})`,
          );
        } catch (error) {
          const tx = db.transaction(() => {
            for (const judgeCase of batch.cases)
              storeJudgeFailure(db, outRun, judgeCase, error);
          });
          tx();
          completed += batch.cases.length;
          console.error(
            `[codex-rejudge] batch ${batch.index} failed permanently: ${String(error)} (${completed}/${sourceRows.length})`,
          );
        }
      }),
    ),
  );

  const summary = summarizeRun(db, outRun);
  updateRunStatus(db, outRun, 'completed');
  console.log(JSON.stringify(summary, null, 2));
  return summary;
}

export async function main(argv = process.argv.slice(2)): Promise<void> {
  const args = parseCodexRejudgeArgs(argv);
  await runCodexRejudge(args);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}
