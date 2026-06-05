#!/usr/bin/env node
import 'dotenv/config';
import {
  buildApplication,
  buildCommand,
  buildRouteMap,
  numberParser,
  run,
} from '@stricli/core';

import {
  MissingEnvVarError,
  resolveCandidateModel,
  resolveJudgeModel as resolveConfiguredJudgeModel,
} from './modelResolver';
import { loadConfig } from '../core/config/loadConfig';
import { isCodexJudgeConfig, type ApocbenchConfig } from '../core/config/schema';
import { expandDatasetPaths, loadJsonl, loadJsonlMany } from '../core/dataset/loadJsonl';
import {
  runCodexRejudge,
  type BatchStrategy,
  type CodexRejudgeArgs,
} from '../core/judge/codex';
import type { RunnerEvent } from '../core/runner/orchestrator';
import { runBenchmark, selectQuestions } from '../core/runner/orchestrator';
import { sanitizeEvent } from '../core/runner/sanitizeEvent';
import { diffSummaries, type RunSummary } from '../core/scoring/diff';
import { renderHtmlReport } from '../reports/html/renderHtml';
import {
  normalizeNewlines,
  renderByDomainMd,
  renderRunIndexMd,
  slugify,
  type DomainRenderCase,
  type DomainRenderCaseResult,
  renderByModelMd,
  type ModelRenderCaseResult,
  renderCaseMd,
  type CaseRenderResult,
  type ExportModelResult,
  type ExportRunMetadata,
} from '../reports/markdown';
import { openAndMigrate } from '../storage/sqlite/migrate';
import { listRunModelResults } from '../storage/sqlite/queries';
import { getRun } from '../storage/sqlite/runs';
import { App } from '../ui/App';

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { render } from 'ink';

type CliContext = {
  process: NodeJS.Process;
};

function readToolVersion(): string {
  let dir = path.dirname(fileURLToPath(import.meta.url));
  for (;;) {
    const pkgPath = path.join(dir, 'package.json');
    if (fs.existsSync(pkgPath)) {
      try {
        const parsed = JSON.parse(fs.readFileSync(pkgPath, 'utf8')) as {
          version?: unknown;
        } | null;
        if (parsed && typeof parsed.version === 'string' && parsed.version.length > 0) {
          return parsed.version;
        }
      } catch {
        // ignore and fall through
      }
    }

    const parent = path.dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }

  return 'unknown';
}

const TOOL_VERSION = readToolVersion();

function die(msg: string): never {
  console.error(msg);
  process.exit(1);
}

type RunFlags = {
  config: string;
  dryRun?: boolean;
  quiet?: boolean;
  json?: boolean;
  limit?: number;
  categories?: readonly string[];
  questions?: readonly string[];
  models?: readonly string[];
};

async function runCommand(
  this: CliContext,
  flags: RunFlags,
  runId?: string,
  forceResume = false,
): Promise<void | Error> {
  const { config: loadedConfig } = loadConfig(flags.config);
  const config: ApocbenchConfig = forceResume
    ? { ...loadedConfig, run: { ...loadedConfig.run, resume: true } }
    : loadedConfig;

  const dataset = config.run.datasetPaths
    ? loadJsonlMany(config.run.datasetPaths)
    : loadJsonl(config.run.datasetPath!);
  const modelCount = config.models.length;
  const selectedQuestionsCount = selectQuestions({
    allQuestions: dataset.lines,
    config,
    limitOverride: typeof flags.limit === 'number' ? flags.limit : null,
    categoriesOverride: flags.categories ? Array.from(flags.categories) : null,
    questionIdsOverride: flags.questions ? Array.from(flags.questions) : null,
  }).length;
  const questionsPerModel = selectedQuestionsCount;
  const totalQuestions = selectedQuestionsCount * modelCount;

  // Keep a bounded event buffer so long runs don't exhaust the JS heap.
  // The UI only needs ~50 recent events for display (logs panel shows 16, plus some buffer).
  // Previously 2000 caused memory issues with React/Ink rendering overhead.
  const events: RunnerEvent[] = [];
  const EVENTS_LIMIT = 100;

  const resolveModel = (m: ApocbenchConfig['models'][number]) => {
    try {
      return resolveCandidateModel(config, m, process.env);
    } catch (err) {
      if (err instanceof MissingEnvVarError) die(err.message);
      throw err;
    }
  };

  const resolveJudgeModel = () => {
    try {
      return resolveConfiguredJudgeModel(config, process.env);
    } catch (err) {
      if (err instanceof MissingEnvVarError) die(err.message);
      throw err;
    }
  };

  const subscribers = new Set<(e: RunnerEvent) => void>();

  const runPromise = runBenchmark({
    config,
    configPath: flags.config,
    datasetPath: config.run.datasetPaths
      ? config.run.datasetPaths.join(',')
      : config.run.datasetPath!,
    datasetAbsolutePath:
      'absolutePath' in dataset
        ? dataset.absolutePath
        : path.resolve(process.cwd(), expandDatasetPaths(config.run.datasetPaths!)[0]!),
    questions: dataset.lines,
    deps: { resolveModel, resolveJudgeModel, toolVersion: TOOL_VERSION },
    dryRun: flags.dryRun ?? false,
    runIdOverride: runId,
    forceResume: forceResume ? true : undefined,
    selectedModelIds: flags.models ? Array.from(flags.models) : undefined,
    limitOverride: typeof flags.limit === 'number' ? flags.limit : null,
    categoriesOverride: flags.categories ? Array.from(flags.categories) : null,
    questionIdsOverride: flags.questions ? Array.from(flags.questions) : null,
    onEvent: (e) => {
      const sanitized = sanitizeEvent(e);
      events.push(sanitized);
      if (events.length > EVENTS_LIMIT) events.splice(0, events.length - EVENTS_LIMIT);
      for (const s of subscribers) s(sanitized);
      if (flags.json) process.stdout.write(JSON.stringify(sanitized) + '\n');
    },
  });

  if (flags.quiet || flags.json) {
    const r = await runPromise;
    if (r && flags.json) process.stdout.write(JSON.stringify(r) + '\n');
    return;
  }

  // debug output intentionally removed

  render(
    <App
      runPromise={runPromise}
      getInitialEvents={() => events.slice()}
      subscribeToEvents={(onEvent) => {
        subscribers.add(onEvent);
        return () => {
          subscribers.delete(onEvent);
        };
      }}
      totalQuestions={totalQuestions}
      questionsPerModel={questionsPerModel}
      modelCount={modelCount}
    />,
  );
}

type ValidateFlags = {
  config: string;
  quiet?: boolean;
};

async function validateCommand(this: CliContext, flags: ValidateFlags): Promise<void> {
  const { config } = loadConfig(flags.config);
  const dataset = config.run.datasetPaths
    ? loadJsonlMany(config.run.datasetPaths)
    : loadJsonl(config.run.datasetPath!);
  if (!flags.quiet)
    console.log(`config ok; dataset ok (${dataset.lines.length} questions)`);
}

type ReportFlags = {
  outDir?: string;
};

async function reportCommand(
  this: CliContext,
  flags: ReportFlags,
  runId: string,
): Promise<void> {
  const outDir = flags.outDir ?? './runs';
  const db = openAndMigrate(path.resolve(process.cwd(), outDir, 'apocbench.sqlite'));
  const rows = listRunModelResults(db, runId);
  const byModel = new Map<string, { overallScore: number; autoFailCount: number }>();
  for (const r of rows) {
    const model = byModel.get(r.model_id) ?? { overallScore: 0, autoFailCount: 0 };
    if (r.status === 'done' && typeof r.score_overall === 'number')
      model.overallScore += r.score_overall;
    if (r.auto_fail === 1) model.autoFailCount += 1;
    byModel.set(r.model_id, model);
  }
  const summary = {
    runId,
    models: Array.from(byModel.entries()).map(([modelId, m]) => ({
      modelId,
      overallScore: m.overallScore,
      autoFailCount: m.autoFailCount,
    })),
  } satisfies RunSummary;

  const runDir = path.resolve(process.cwd(), outDir, runId);
  fs.mkdirSync(runDir, { recursive: true });
  fs.writeFileSync(path.join(runDir, 'summary.json'), JSON.stringify(summary, null, 2));
  fs.writeFileSync(
    path.join(runDir, 'report.html'),
    renderHtmlReport({ runId, summaryJson: summary, results: rows }),
  );
  console.log(JSON.stringify(summary, null, 2));
}

type ExportMdFlags = {
  out?: string;
  mode?: string;
  includeCases?: boolean;
  overwrite?: boolean;
  redact?: string;
};

async function exportMdCommand(
  this: CliContext,
  flags: ExportMdFlags,
  runId: string,
): Promise<void> {
  const outDir = flags.out ?? path.join('runs', runId, 'markdown');
  const mode = flags.mode ?? 'both';
  const includeCases = flags.includeCases ?? true;
  const overwrite = flags.overwrite ?? false;
  const redact = flags.redact ?? 'none';
  if (redact !== 'none') {
    die(`unsupported redact mode (MVP supports only "none"): ${redact}`);
  }

  const db = openAndMigrate(path.resolve(process.cwd(), 'runs', 'apocbench.sqlite'));
  const run = getRun(db, runId);
  if (!run) die(`run not found: ${runId}`);
  const results = listRunModelResults(db, runId);
  let config: unknown = null;
  try {
    config = JSON.parse(run.config_json);
  } catch {
    config = null;
  }

  const resolvedOutDir = path.resolve(process.cwd(), outDir);
  if (fs.existsSync(resolvedOutDir)) {
    if (!overwrite) {
      die(`output already exists (use --overwrite to replace): ${resolvedOutDir}`);
    }
    fs.rmSync(resolvedOutDir, { recursive: true, force: true });
  }
  fs.mkdirSync(resolvedOutDir, { recursive: true });

  const byDomainDir = path.join(resolvedOutDir, 'by-domain');
  const byModelDir = path.join(resolvedOutDir, 'by-model');
  fs.mkdirSync(byDomainDir, { recursive: true });
  fs.mkdirSync(byModelDir, { recursive: true });
  if (includeCases) {
    fs.mkdirSync(path.join(resolvedOutDir, 'cases'), { recursive: true });
  }

  const metadata: ExportRunMetadata = {
    runId: run.run_id,
    createdAt: run.created_at,
    toolVersion: run.tool_version,
    status: run.status,
    config,
    datasetPath: run.dataset_path,
    datasetSha256: run.dataset_sha256,
    promptTemplateHash: run.prompt_template_hash,
  };

  const configId = (() => {
    if (!config || typeof config !== 'object') return null;
    const runConfig = (config as Record<string, unknown>).run;
    if (!runConfig || typeof runConfig !== 'object') return null;
    const name = (runConfig as Record<string, unknown>).name;
    return typeof name === 'string' ? name : null;
  })();

  const byDomain = Array.from(
    new Set(
      results
        .map((row) => row.category)
        .filter(
          (value): value is string => typeof value === 'string' && value.length > 0,
        ),
    ),
  );
  const byModel = Array.from(
    new Set(
      results
        .map((row) => row.model_id)
        .filter(
          (value): value is string => typeof value === 'string' && value.length > 0,
        ),
    ),
  );
  const cases = includeCases
    ? Array.from(
        new Set(
          results
            .map((row) => row.question_id)
            .filter(
              (value): value is string => typeof value === 'string' && value.length > 0,
            ),
        ),
      )
    : [];
  const allModels = Array.from(
    new Set(
      results
        .map((row) => row.model_id)
        .filter(
          (value): value is string => typeof value === 'string' && value.length > 0,
        ),
    ),
  );

  const normalizeText = (value: string | null) =>
    typeof value === 'string' ? normalizeNewlines(value) : null;

  const normalizeJson = (value: string | null): Record<string, unknown> | null => {
    if (typeof value !== 'string') return null;
    try {
      return JSON.parse(normalizeNewlines(value)) as Record<string, unknown>;
    } catch {
      return null;
    }
  };

  const resultsRecords: ExportModelResult[] = results.map((row) => ({
    runId: row.run_id,
    modelId: row.model_id,
    caseId: row.question_id,
    status: row.status,
    prompt: normalizeText(row.prompt),
    answer: normalizeText(row.candidate_completion),
    candidatePrompt: normalizeText(row.candidate_prompt),
    candidateMetrics: normalizeJson(row.candidate_metrics_json),
    retrievalTrace: normalizeJson(row.retrieval_trace_json),
    scoreOverall: row.score_overall,
    scoreRubric: normalizeJson(row.score_rubric_json),
    autoFail: typeof row.auto_fail === 'number' ? row.auto_fail === 1 : null,
    autoFailReason: row.auto_fail_reason,
    judgeParsed: normalizeJson(row.judge_parsed_json),
    judgeRaw: normalizeText(row.judge_response_json),
    error: normalizeJson(row.error_json),
  }));

  const resultsJsonl = resultsRecords.map((record) => JSON.stringify(record)).join('\n');

  fs.writeFileSync(
    path.join(resolvedOutDir, 'RUN.md'),
    renderRunIndexMd({
      frontMatter: {
        run_id: metadata.runId,
        created_utc: metadata.createdAt,
        dataset: metadata.datasetPath ?? null,
        config_id: configId,
        git_commit: metadata.gitCommit ?? null,
        schema_version: null,
      },
      byDomain,
      byModel,
      cases,
    }),
  );
  fs.writeFileSync(
    path.join(resolvedOutDir, 'metadata.json'),
    JSON.stringify(metadata, null, 2),
  );
  fs.writeFileSync(path.join(resolvedOutDir, 'results.jsonl'), resultsJsonl);

  const parseJsonArray = (value: string | null): string[] => {
    if (typeof value !== 'string') return [];
    try {
      const parsed = JSON.parse(normalizeNewlines(value));
      return Array.isArray(parsed) ? parsed.filter((v) => typeof v === 'string') : [];
    } catch {
      return [];
    }
  };

  const parseRubric = (value: string | null): DomainRenderCase['rubric'] => {
    if (typeof value !== 'string') return [];
    try {
      const parsed = JSON.parse(normalizeNewlines(value));
      if (!Array.isArray(parsed)) return [];
      return parsed
        .map((item) => {
          if (!item || typeof item !== 'object') return null;
          const record = item as Record<string, unknown>;
          if (typeof record.id !== 'string' || typeof record.text !== 'string')
            return null;
          const rubricItem = {
            id: record.id,
            text: record.text,
          } as DomainRenderCase['rubric'][number];
          if (typeof record.weight === 'number') rubricItem.weight = record.weight;
          if (typeof record.maxScore === 'number') rubricItem.maxScore = record.maxScore;
          return rubricItem;
        })
        .filter((item): item is DomainRenderCase['rubric'][number] => item !== null);
    } catch {
      return [];
    }
  };

  if (mode === 'by-domain' || mode === 'both') {
    const domainMap = new Map<string, Map<string, DomainRenderCase>>();
    for (const row of results) {
      const domain = row.category ?? 'unknown';
      const caseId = row.question_id ?? 'unknown';
      let caseMap = domainMap.get(domain);
      if (!caseMap) {
        caseMap = new Map<string, DomainRenderCase>();
        domainMap.set(domain, caseMap);
      }

      let caseEntry = caseMap.get(caseId);
      if (!caseEntry) {
        caseEntry = {
          caseId,
          category: domain,
          difficulty: row.difficulty ?? 'unknown',
          scenario: parseJsonArray(row.scenario),
          prompt: normalizeNewlines(row.prompt ?? ''),
          rubric: parseRubric(row.rubric_json),
          autoFail: parseJsonArray(row.auto_fail_json),
          results: [],
        };
        caseMap.set(caseId, caseEntry);
      }

      const resultEntry: DomainRenderCaseResult = {
        modelId: row.model_id,
        status: row.status,
        answer: normalizeText(row.candidate_completion),
        retrievalTrace: normalizeJson(row.retrieval_trace_json),
        scoreOverall: row.score_overall,
        autoFail: typeof row.auto_fail === 'number' ? row.auto_fail === 1 : null,
        autoFailReason: row.auto_fail_reason,
        judgeParsed: normalizeJson(row.judge_parsed_json),
        judgeRaw: normalizeText(row.judge_response_json),
        error: normalizeJson(row.error_json),
      };
      caseEntry.results.push(resultEntry);
    }

    for (const [, caseMap] of domainMap.entries()) {
      for (const caseEntry of caseMap.values()) {
        const present = new Set(caseEntry.results.map((result) => result.modelId));
        for (const modelId of allModels) {
          if (present.has(modelId)) continue;
          caseEntry.results.push({
            modelId,
            status: 'MISSING',
            answer: null,
            retrievalTrace: null,
            scoreOverall: null,
            autoFail: null,
            autoFailReason: null,
            judgeParsed: null,
            judgeRaw: null,
            error: null,
          });
        }
      }
    }

    for (const [domain, caseMap] of domainMap.entries()) {
      const caseList = Array.from(caseMap.values());
      const modelCount = new Set(
        caseList.flatMap((caseItem) => caseItem.results.map((result) => result.modelId)),
      ).size;

      const content = renderByDomainMd({
        frontMatter: {
          run_id: metadata.runId,
          domain,
          case_count: caseList.length,
          model_count: modelCount,
        },
        domain,
        cases: caseList,
      });
      fs.writeFileSync(path.join(byDomainDir, `${slugify(domain)}.md`), content);
    }
  }

  if (mode === 'by-model' || mode === 'both') {
    const caseMap = new Map<
      string,
      {
        category: string;
        difficulty: string;
        scenario: string[];
        prompt: string;
        rubric: DomainRenderCase['rubric'];
        autoFail: string[];
      }
    >();
    for (const row of results) {
      const caseId = row.question_id ?? 'unknown';
      if (caseMap.has(caseId)) continue;
      caseMap.set(caseId, {
        category: row.category ?? 'unknown',
        difficulty: row.difficulty ?? 'unknown',
        scenario: parseJsonArray(row.scenario),
        prompt: normalizeNewlines(row.prompt ?? ''),
        rubric: parseRubric(row.rubric_json),
        autoFail: parseJsonArray(row.auto_fail_json),
      });
    }

    const resultsByModel = new Map<string, ModelRenderCaseResult[]>();
    for (const row of results) {
      const modelId = row.model_id ?? 'unknown';
      const caseId = row.question_id ?? 'unknown';
      const base = caseMap.get(caseId) ?? {
        category: row.category ?? 'unknown',
        difficulty: row.difficulty ?? 'unknown',
        scenario: parseJsonArray(row.scenario),
        prompt: normalizeNewlines(row.prompt ?? ''),
        rubric: parseRubric(row.rubric_json),
        autoFail: parseJsonArray(row.auto_fail_json),
      };
      const entry: ModelRenderCaseResult = {
        caseId,
        category: base.category,
        difficulty: base.difficulty,
        scenario: base.scenario,
        prompt: base.prompt,
        rubric: base.rubric,
        autoFail: base.autoFail,
        status: row.status ?? 'unknown',
        answer: normalizeText(row.candidate_completion),
        retrievalTrace: normalizeJson(row.retrieval_trace_json),
        scoreOverall: row.score_overall,
        autoFailFlag: typeof row.auto_fail === 'number' ? row.auto_fail === 1 : null,
        autoFailReason: row.auto_fail_reason,
        judgeParsed: normalizeJson(row.judge_parsed_json),
        judgeRaw: normalizeText(row.judge_response_json),
        error: normalizeJson(row.error_json),
      };
      const list = resultsByModel.get(modelId) ?? [];
      list.push(entry);
      resultsByModel.set(modelId, list);
    }

    const allCaseIds = Array.from(caseMap.keys());
    for (const [, caseList] of resultsByModel.entries()) {
      const seen = new Set(caseList.map((entry) => entry.caseId));
      for (const caseId of allCaseIds) {
        if (seen.has(caseId)) continue;
        const base = caseMap.get(caseId);
        if (!base) continue;
        caseList.push({
          caseId,
          category: base.category,
          difficulty: base.difficulty,
          scenario: base.scenario,
          prompt: base.prompt,
          rubric: base.rubric,
          autoFail: base.autoFail,
          status: 'MISSING',
          answer: null,
          retrievalTrace: null,
          scoreOverall: null,
          autoFailFlag: null,
          autoFailReason: null,
          judgeParsed: null,
          judgeRaw: null,
          error: null,
        });
      }
    }

    const baseSlugCounts = new Map<string, number>();
    const modelSlugMap = new Map<string, string>();
    const sortedModels = Array.from(resultsByModel.keys()).sort((a, b) =>
      a.localeCompare(b),
    );
    for (const modelId of sortedModels) {
      const baseSlug = slugify(modelId);
      const count = (baseSlugCounts.get(baseSlug) ?? 0) + 1;
      baseSlugCounts.set(baseSlug, count);
      if (count > 1) {
        const suffix = Math.abs(hashString(modelId)).toString(36).slice(0, 6);
        modelSlugMap.set(modelId, `${baseSlug}-${suffix}`);
      } else {
        modelSlugMap.set(modelId, baseSlug);
      }
    }

    for (const modelId of sortedModels) {
      const caseList = resultsByModel.get(modelId) ?? [];
      const content = renderByModelMd({
        frontMatter: {
          run_id: metadata.runId,
          model: modelId,
          case_count: caseList.length,
        },
        model: modelId,
        cases: caseList,
      });
      const slug = modelSlugMap.get(modelId) ?? slugify(modelId);
      fs.writeFileSync(path.join(byModelDir, `${slug}.md`), content);
    }
  }

  if (includeCases) {
    const caseMap = new Map<
      string,
      {
        category: string;
        difficulty: string;
        scenario: string[];
        prompt: string;
        rubric: DomainRenderCase['rubric'];
        autoFail: string[];
      }
    >();
    for (const row of results) {
      const caseId = row.question_id ?? 'unknown';
      if (caseMap.has(caseId)) continue;
      caseMap.set(caseId, {
        category: row.category ?? 'unknown',
        difficulty: row.difficulty ?? 'unknown',
        scenario: parseJsonArray(row.scenario),
        prompt: normalizeNewlines(row.prompt ?? ''),
        rubric: parseRubric(row.rubric_json),
        autoFail: parseJsonArray(row.auto_fail_json),
      });
    }

    const resultsByCase = new Map<string, CaseRenderResult[]>();
    for (const row of results) {
      const caseId = row.question_id ?? 'unknown';
      const entry: CaseRenderResult = {
        modelId: row.model_id ?? 'unknown',
        status: row.status ?? 'unknown',
        answer: normalizeText(row.candidate_completion),
        retrievalTrace: normalizeJson(row.retrieval_trace_json),
        scoreOverall: row.score_overall,
        autoFail: typeof row.auto_fail === 'number' ? row.auto_fail === 1 : null,
        autoFailReason: row.auto_fail_reason,
        judgeParsed: normalizeJson(row.judge_parsed_json),
        judgeRaw: normalizeText(row.judge_response_json),
        error: normalizeJson(row.error_json),
      };
      const list = resultsByCase.get(caseId) ?? [];
      list.push(entry);
      resultsByCase.set(caseId, list);
    }

    for (const [caseId, caseInfo] of caseMap.entries()) {
      const caseResults = resultsByCase.get(caseId) ?? [];
      const present = new Set(caseResults.map((result) => result.modelId));
      for (const modelId of allModels) {
        if (present.has(modelId)) continue;
        caseResults.push({
          modelId,
          status: 'MISSING',
          answer: null,
          retrievalTrace: null,
          scoreOverall: null,
          autoFail: null,
          autoFailReason: null,
          judgeParsed: null,
          judgeRaw: null,
          error: null,
        });
      }
      const content = renderCaseMd({
        frontMatter: {
          run_id: metadata.runId,
          case_id: caseId,
          domain: caseInfo.category,
          difficulty: caseInfo.difficulty,
        },
        caseId,
        domain: caseInfo.category,
        difficulty: caseInfo.difficulty,
        scenario: caseInfo.scenario,
        prompt: caseInfo.prompt,
        rubric: caseInfo.rubric,
        autoFail: caseInfo.autoFail,
        results: caseResults,
      });
      fs.writeFileSync(path.join(resolvedOutDir, 'cases', `${caseId}.md`), content);
    }
  }

  void { mode, redact };
}

function hashString(value: string): number {
  let hash = 0;
  for (let i = 0; i < value.length; i += 1) {
    hash = (hash * 31 + value.charCodeAt(i)) | 0;
  }
  return hash;
}

type DiffFlags = {
  outDir?: string;
};

type JudgeFlags = {
  config: string;
  sourceRun?: string;
  'source-run'?: string;
  outRun?: string;
  'out-run'?: string;
  resume?: boolean;
  limit?: number;
  models?: readonly string[];
  json?: boolean;
};

type RunAndJudgeFlags = RunFlags & {
  outRun?: string;
  resume?: boolean;
  compareOut?: string;
};

type CompareFlags = {
  run?: string;
  baselineRun?: string;
  'baseline-run'?: string;
  comparisonRun?: string;
  'comparison-run'?: string;
  outDir?: string;
  'out-dir'?: string;
  baselineSuffix?: string;
  'baseline-suffix'?: string;
  comparisonSuffix?: string;
  'comparison-suffix'?: string;
  out?: string;
};

function datasetPathForCodexJudge(config: ApocbenchConfig): string {
  if (config.run.datasetPaths) {
    if (config.run.datasetPaths.length !== 1) {
      throw new Error('Codex judge currently requires exactly one datasetPaths entry');
    }
    return config.run.datasetPaths[0]!;
  }
  return config.run.datasetPath!;
}

function codexArgsFromConfig(params: {
  config: ApocbenchConfig;
  sourceRun: string;
  outRun?: string;
  resume?: boolean;
  limit?: number;
  models?: readonly string[];
}): CodexRejudgeArgs {
  const { config } = params;
  if (!isCodexJudgeConfig(config.judge)) {
    throw new Error('dev judge requires judge.backend=codex-cli');
  }
  return {
    db: path.join(config.run.outDir, 'apocbench.sqlite'),
    sourceRun: params.sourceRun,
    outRun: params.outRun,
    dataset: datasetPathForCodexJudge(config),
    codexBin: config.judge.codexBin,
    model: config.judge.model,
    reasoning: config.judge.reasoning,
    disableFeatures: config.judge.disableFeatures,
    batchSize: config.judge.batchSize,
    batchStrategy: config.judge.batchStrategy as BatchStrategy,
    concurrency: config.judge.concurrency,
    maxRetries: config.judge.maxRetries,
    tmpDir: config.judge.tmpDir,
    limit: params.limit,
    models: params.models ? Array.from(params.models) : undefined,
    sourceStatus: config.judge.sourceStatus,
    resume: params.resume ?? false,
  };
}

async function judgeCommand(this: CliContext, flags: JudgeFlags): Promise<void> {
  const { config } = loadConfig(flags.config);
  const sourceRun = flags.sourceRun ?? flags['source-run'];
  if (!sourceRun) die('missing source run id (pass --sourceRun or --source-run)');
  const summary = await runCodexRejudge(
    codexArgsFromConfig({
      config,
      sourceRun,
      outRun: flags.outRun ?? flags['out-run'],
      resume: flags.resume,
      limit: flags.limit,
      models: flags.models,
    }),
  );
  if (flags.json) console.log(JSON.stringify(summary));
}

function countCompleteCandidateRows(
  db: ReturnType<typeof openAndMigrate>,
  runId: string,
  modelIds?: readonly string[],
): number {
  const params: unknown[] = [runId];
  let modelFilter = '';
  if (modelIds && modelIds.length > 0) {
    modelFilter = `and model_id in (${modelIds.map(() => '?').join(',')})`;
    params.push(...modelIds);
  }

  const row = db
    .prepare(
      `select count(*) as count
       from model_results
       where run_id = ?
         ${modelFilter}
         and status in ('candidate_done', 'done')
         and candidate_completion is not null
         and length(trim(candidate_completion)) > 0`,
    )
    .get(...params) as { count: number };
  return row.count;
}

export function selectedModelIdsForRunAndJudge(
  config: ApocbenchConfig,
  requestedModelIds?: readonly string[],
): string[] {
  if (!requestedModelIds || requestedModelIds.length === 0) {
    return config.models.map((model) => model.id);
  }
  const requested = new Set(requestedModelIds);
  return config.models
    .filter((model) => requested.has(model.id))
    .map((model) => model.id);
}

export function expectedCandidateCountForRunAndJudge(params: {
  config: ApocbenchConfig;
  questionCount: number;
  requestedModelIds?: readonly string[];
}): number {
  return (
    params.questionCount *
    selectedModelIdsForRunAndJudge(params.config, params.requestedModelIds).length
  );
}

function stripModelSuffix(modelId: string, suffix: string): string | null {
  const token = `-${suffix}`;
  return modelId.endsWith(token) ? modelId.slice(0, -token.length) : null;
}

function mean(values: number[]): number | null {
  return values.length === 0
    ? null
    : values.reduce((total, value) => total + value, 0) / values.length;
}

function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((left, right) => left - right);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[mid]! : (sorted[mid - 1]! + sorted[mid]!) / 2;
}

function summarizeConditionRows(rows: ReturnType<typeof listRunModelResults>) {
  const done = rows.filter(
    (row) => row.status === 'done' && typeof row.score_overall === 'number',
  );
  const scores = done.map((row) => row.score_overall as number);
  const autoFails = done.filter((row) => row.auto_fail === 1).length;
  const zeros = done.filter((row) => row.score_overall === 0).length;
  return {
    rows: rows.length,
    done: done.length,
    failures: rows.length - done.length,
    meanScore: mean(scores),
    medianScore: median(scores),
    autoFails,
    autoFailRate: done.length === 0 ? null : autoFails / done.length,
    zeros,
    zeroRate: done.length === 0 ? null : zeros / done.length,
  };
}

function pairedDeltasForRows(params: {
  rows: ReturnType<typeof listRunModelResults>;
  baselineSuffix: string;
  comparisonSuffix: string;
  modelBase?: string;
}) {
  const baseline = new Map<string, number>();
  const comparisonRows = [];
  for (const row of params.rows) {
    const baselineBase = stripModelSuffix(row.model_id, params.baselineSuffix);
    if (
      baselineBase &&
      (!params.modelBase || baselineBase === params.modelBase) &&
      row.status === 'done' &&
      typeof row.score_overall === 'number'
    ) {
      baseline.set(`${baselineBase}\u0000${row.question_id}`, row.score_overall);
    }
    const comparisonBase = stripModelSuffix(row.model_id, params.comparisonSuffix);
    if (
      comparisonBase &&
      (!params.modelBase || comparisonBase === params.modelBase) &&
      row.status === 'done' &&
      typeof row.score_overall === 'number'
    ) {
      comparisonRows.push({ ...row, modelBase: comparisonBase });
    }
  }
  const deltas = comparisonRows
    .map((row) => {
      const baselineScore = baseline.get(`${row.modelBase}\u0000${row.question_id}`);
      return baselineScore == null ? null : (row.score_overall as number) - baselineScore;
    })
    .filter((value): value is number => value != null);
  return {
    pairedCount: deltas.length,
    meanDelta: mean(deltas),
    medianDelta: median(deltas),
    wins: deltas.filter((delta) => delta > 0).length,
    losses: deltas.filter((delta) => delta < 0).length,
    ties: deltas.filter((delta) => delta === 0).length,
  };
}

function buildPairedComparisonReport(params: {
  runId: string;
  rows: ReturnType<typeof listRunModelResults>;
  baselineSuffix: string;
  comparisonSuffix: string;
  baselineRunId?: string;
  comparisonRunId?: string;
}) {
  const baselineRows = params.rows.filter((row) =>
    Boolean(stripModelSuffix(row.model_id, params.baselineSuffix)),
  );
  const comparisonRows = params.rows.filter((row) =>
    Boolean(stripModelSuffix(row.model_id, params.comparisonSuffix)),
  );
  const modelBases = Array.from(
    new Set(
      params.rows
        .flatMap((row) => [
          stripModelSuffix(row.model_id, params.baselineSuffix),
          stripModelSuffix(row.model_id, params.comparisonSuffix),
        ])
        .filter((value): value is string => Boolean(value)),
    ),
  ).sort((left, right) => left.localeCompare(right));
  return {
    runId: params.runId,
    sourceRuns: {
      baseline: params.baselineRunId ?? params.runId,
      comparison: params.comparisonRunId ?? params.runId,
    },
    baselineSuffix: params.baselineSuffix,
    comparisonSuffix: params.comparisonSuffix,
    overall: {
      baseline: summarizeConditionRows(baselineRows),
      comparison: summarizeConditionRows(comparisonRows),
      paired: pairedDeltasForRows(params),
    },
    models: modelBases.map((modelBase) => ({
      modelBase,
      baseline: summarizeConditionRows(
        baselineRows.filter(
          (row) => stripModelSuffix(row.model_id, params.baselineSuffix) === modelBase,
        ),
      ),
      comparison: summarizeConditionRows(
        comparisonRows.filter(
          (row) => stripModelSuffix(row.model_id, params.comparisonSuffix) === modelBase,
        ),
      ),
      paired: pairedDeltasForRows({ ...params, modelBase }),
    })),
  };
}

async function compareCommand(
  this: CliContext,
  flags: CompareFlags,
  runId?: string,
): Promise<void> {
  const resolvedRunId = flags.run ?? runId;
  const baselineRunId = flags.baselineRun ?? flags['baseline-run'] ?? resolvedRunId;
  const comparisonRunId = flags.comparisonRun ?? flags['comparison-run'] ?? resolvedRunId;
  if (!baselineRunId || !comparisonRunId) {
    die(
      'missing scored run id (pass positional runId/--run, or both --baseline-run and --comparison-run)',
    );
  }
  const outDir = flags.outDir ?? flags['out-dir'] ?? './runs';
  const db = openAndMigrate(path.resolve(process.cwd(), outDir, 'apocbench.sqlite'));
  const rows =
    baselineRunId === comparisonRunId
      ? listRunModelResults(db, baselineRunId)
      : [
          ...listRunModelResults(db, baselineRunId),
          ...listRunModelResults(db, comparisonRunId),
        ];
  const report = buildPairedComparisonReport({
    runId:
      resolvedRunId ??
      (baselineRunId === comparisonRunId
        ? baselineRunId
        : `${baselineRunId}..${comparisonRunId}`),
    rows,
    baselineSuffix: flags.baselineSuffix ?? flags['baseline-suffix'] ?? 'direct',
    comparisonSuffix:
      flags.comparisonSuffix ?? flags['comparison-suffix'] ?? 'agent-bm25-research',
    baselineRunId,
    comparisonRunId,
  });
  if (flags.out) {
    fs.mkdirSync(path.dirname(path.resolve(process.cwd(), flags.out)), {
      recursive: true,
    });
    fs.writeFileSync(
      path.resolve(process.cwd(), flags.out),
      JSON.stringify(report, null, 2),
    );
  }
  console.log(JSON.stringify(report, null, 2));
}

async function runAndJudgeCommand(
  this: CliContext,
  flags: RunAndJudgeFlags,
  runId: string,
): Promise<void | Error> {
  const { config } = loadConfig(flags.config);
  if (!isCodexJudgeConfig(config.judge)) {
    throw new Error('run-and-judge requires judge.backend=codex-cli');
  }
  if (config.run.candidateOnly !== true) {
    throw new Error('run-and-judge requires run.candidateOnly=true');
  }
  const candidateConfig: ApocbenchConfig = {
    ...config,
    run: { ...config.run, candidateOnly: true },
  };
  await runCommand.call(
    this,
    { ...flags, quiet: true, json: false, config: flags.config },
    runId,
    flags.resume === true,
  );
  if (flags.dryRun) {
    const dataset = candidateConfig.run.datasetPaths
      ? loadJsonlMany(candidateConfig.run.datasetPaths)
      : loadJsonl(candidateConfig.run.datasetPath!);
    const selectedQuestionCount = selectQuestions({
      allQuestions: dataset.lines,
      config: candidateConfig,
      limitOverride: typeof flags.limit === 'number' ? flags.limit : null,
      categoriesOverride: flags.categories ? Array.from(flags.categories) : null,
      questionIdsOverride: flags.questions ? Array.from(flags.questions) : null,
    }).length;
    const expectedCandidates = expectedCandidateCountForRunAndJudge({
      config: candidateConfig,
      questionCount: selectedQuestionCount,
      requestedModelIds: flags.models,
    });
    const result = {
      candidateRunId: runId,
      dryRun: true,
      expectedCandidates,
      judgeBackend: config.judge.backend,
      judgeModel: config.judge.model,
      batchStrategy: config.judge.batchStrategy,
      batchSize: config.judge.batchSize,
      judgeConcurrency: config.judge.concurrency,
    };
    if (flags.json) {
      console.log(JSON.stringify(result));
    } else {
      console.log(JSON.stringify(result, null, 2));
    }
    return;
  }
  const dataset = candidateConfig.run.datasetPaths
    ? loadJsonlMany(candidateConfig.run.datasetPaths)
    : loadJsonl(candidateConfig.run.datasetPath!);
  const selectedQuestionCount = selectQuestions({
    allQuestions: dataset.lines,
    config: candidateConfig,
    limitOverride: typeof flags.limit === 'number' ? flags.limit : null,
    categoriesOverride: flags.categories ? Array.from(flags.categories) : null,
    questionIdsOverride: flags.questions ? Array.from(flags.questions) : null,
  }).length;
  const selectedModelIds = selectedModelIdsForRunAndJudge(candidateConfig, flags.models);
  const expectedCandidates = expectedCandidateCountForRunAndJudge({
    config: candidateConfig,
    questionCount: selectedQuestionCount,
    requestedModelIds: flags.models,
  });
  const db = openAndMigrate(
    path.resolve(process.cwd(), candidateConfig.run.outDir, 'apocbench.sqlite'),
  );
  const completeCandidates = countCompleteCandidateRows(db, runId, selectedModelIds);
  if (completeCandidates !== expectedCandidates) {
    throw new Error(
      `candidate run incomplete: ${completeCandidates}/${expectedCandidates} candidate rows complete`,
    );
  }

  const outRun =
    flags.outRun ?? `${runId}-codex-question-paired-b${config.judge.batchSize}`;
  const judgeSummary = await runCodexRejudge(
    codexArgsFromConfig({
      config: candidateConfig,
      sourceRun: runId,
      outRun,
      resume: flags.resume ?? true,
      models: flags.models,
    }),
  );
  const compareOut = flags.compareOut ?? path.join('logs', `${outRun}-comparison.json`);
  const compareRows = listRunModelResults(db, outRun);
  const comparison = buildPairedComparisonReport({
    runId: outRun,
    rows: compareRows,
    baselineSuffix: 'direct',
    comparisonSuffix: 'agent-bm25-research',
  });
  fs.mkdirSync(path.dirname(path.resolve(process.cwd(), compareOut)), {
    recursive: true,
  });
  fs.writeFileSync(
    path.resolve(process.cwd(), compareOut),
    JSON.stringify(comparison, null, 2),
  );
  const result = {
    candidateRunId: runId,
    judgeRunId: outRun,
    summary: judgeSummary,
    reportPath: path.resolve(process.cwd(), compareOut),
  };
  if (flags.json) {
    console.log(JSON.stringify(result));
  } else {
    console.log(JSON.stringify(result, null, 2));
  }
}

async function diffCommand(
  this: CliContext,
  flags: DiffFlags,
  runId1: string,
  runId2: string,
): Promise<void> {
  const outDir = flags.outDir ?? './runs';
  const db = openAndMigrate(path.resolve(process.cwd(), outDir, 'apocbench.sqlite'));

  const toSummary = (runId: string): RunSummary => {
    const rows = listRunModelResults(db, runId);
    const byModel = new Map<string, { overallScore: number; autoFailCount: number }>();
    for (const r of rows) {
      const model = byModel.get(r.model_id) ?? { overallScore: 0, autoFailCount: 0 };
      if (r.status === 'done' && typeof r.score_overall === 'number')
        model.overallScore += r.score_overall;
      if (r.auto_fail === 1) model.autoFailCount += 1;
      byModel.set(r.model_id, model);
    }
    return {
      runId,
      models: Array.from(byModel.entries()).map(([modelId, m]) => ({
        modelId,
        overallScore: m.overallScore,
        autoFailCount: m.autoFailCount,
      })),
    };
  };

  console.log(
    JSON.stringify(diffSummaries(toSummary(runId1), toSummary(runId2)), null, 2),
  );
}

type ResumeFlags = RunFlags;

async function resumeCommand(
  this: CliContext,
  flags: ResumeFlags,
  runId: string,
): Promise<void | Error> {
  console.log(`resuming run ${runId}`);
  return runCommand.call(this, flags, runId, true);
}

const root = buildRouteMap({
  routes: {
    run: buildCommand<RunFlags, [runId?: string], CliContext>({
      docs: {
        brief: 'Run benchmark',
        customUsage: [
          'run -c apocbench.yml [--dry-run] [--json] [--quiet] [--limit N] [--categories a,b] [--questions Q1,Q2]',
          'run <runId> -c apocbench.yml  # resume by runId',
        ],
      },
      parameters: {
        flags: {
          config: { kind: 'parsed', brief: 'Path to apocbench.yml', parse: (s) => s },
          dryRun: {
            kind: 'boolean',
            brief: 'Validate only (no API calls)',
            optional: true,
          },
          quiet: { kind: 'boolean', brief: 'Suppress TUI output', optional: true },
          json: { kind: 'boolean', brief: 'Emit JSONL events', optional: true },
          limit: {
            kind: 'parsed',
            brief: 'Limit questions',
            optional: true,
            parse: numberParser,
          },
          categories: {
            kind: 'parsed',
            brief: 'Comma-separated categories',
            optional: true,
            variadic: ',',
            parse: (s) => s,
          },
          questions: {
            kind: 'parsed',
            brief: 'Comma-separated question ids to run',
            optional: true,
            variadic: ',',
            parse: (s) => s,
          },
          models: {
            kind: 'parsed',
            brief: 'Comma-separated model ids to run (matches config models[].id)',
            optional: true,
            variadic: ',',
            parse: (s) => s,
          },
        },
        aliases: {
          c: 'config',
        },
        positional: {
          kind: 'tuple',
          parameters: [
            {
              brief: 'Optional run id to resume',
              optional: true,
              parse: (s) => s,
              placeholder: 'runId',
            },
          ],
        },
      },
      func: runCommand,
    }),
    validate: buildCommand<ValidateFlags, [], CliContext>({
      docs: { brief: 'Validate config and dataset' },
      parameters: {
        flags: {
          config: { kind: 'parsed', brief: 'Path to apocbench.yml', parse: (s) => s },
          quiet: { kind: 'boolean', brief: 'Suppress output', optional: true },
        },
        aliases: { c: 'config' },
      },
      func: validateCommand,
    }),
    report: buildCommand<ReportFlags, [runId: string], CliContext>({
      docs: { brief: 'Generate run summary and HTML report' },
      parameters: {
        flags: {
          outDir: {
            kind: 'parsed',
            brief: 'Output directory (defaults to ./runs)',
            optional: true,
            parse: (s) => s,
          },
        },
        positional: {
          kind: 'tuple',
          parameters: [
            {
              brief: 'Run id',
              parse: (s) => s,
              placeholder: 'runId',
            },
          ],
        },
      },
      func: reportCommand,
    }),
    judge: buildCommand<JudgeFlags, [], CliContext>({
      docs: { brief: 'Score completed candidate rows with configured Codex judge' },
      parameters: {
        flags: {
          config: { kind: 'parsed', brief: 'Path to apocbench.yml', parse: (s) => s },
          sourceRun: {
            kind: 'parsed',
            brief: 'Candidate source run id',
            optional: true,
            parse: (s) => s,
          },
          'source-run': {
            kind: 'parsed',
            brief: 'Candidate source run id',
            optional: true,
            parse: (s) => s,
          },
          outRun: {
            kind: 'parsed',
            brief: 'Destination scored run id',
            optional: true,
            parse: (s) => s,
          },
          'out-run': {
            kind: 'parsed',
            brief: 'Destination scored run id',
            optional: true,
            parse: (s) => s,
          },
          resume: { kind: 'boolean', brief: 'Resume existing judge run', optional: true },
          json: { kind: 'boolean', brief: 'Emit compact JSON summary', optional: true },
          limit: {
            kind: 'parsed',
            brief: 'Limit selected source rows',
            optional: true,
            parse: numberParser,
          },
          models: {
            kind: 'parsed',
            brief: 'Comma-separated source model ids',
            optional: true,
            variadic: ',',
            parse: (s) => s,
          },
        },
        aliases: {
          c: 'config',
        },
      },
      func: judgeCommand,
    }),
    'run-and-judge': buildCommand<RunAndJudgeFlags, [runId: string], CliContext>({
      docs: { brief: 'Run candidate-only benchmark, Codex-judge it, and compare' },
      parameters: {
        flags: {
          config: { kind: 'parsed', brief: 'Path to apocbench.yml', parse: (s) => s },
          dryRun: {
            kind: 'boolean',
            brief: 'Validate only (no API calls)',
            optional: true,
          },
          quiet: { kind: 'boolean', brief: 'Suppress TUI output', optional: true },
          json: { kind: 'boolean', brief: 'Emit JSON output', optional: true },
          resume: {
            kind: 'boolean',
            brief: 'Resume candidate and judge runs',
            optional: true,
          },
          outRun: {
            kind: 'parsed',
            brief: 'Destination scored run id',
            optional: true,
            parse: (s) => s,
          },
          compareOut: {
            kind: 'parsed',
            brief: 'Comparison JSON output path',
            optional: true,
            parse: (s) => s,
          },
          limit: {
            kind: 'parsed',
            brief: 'Limit questions',
            optional: true,
            parse: numberParser,
          },
          categories: {
            kind: 'parsed',
            brief: 'Comma-separated categories',
            optional: true,
            variadic: ',',
            parse: (s) => s,
          },
          questions: {
            kind: 'parsed',
            brief: 'Comma-separated question ids',
            optional: true,
            variadic: ',',
            parse: (s) => s,
          },
          models: {
            kind: 'parsed',
            brief: 'Comma-separated model ids',
            optional: true,
            variadic: ',',
            parse: (s) => s,
          },
        },
        aliases: { c: 'config' },
        positional: {
          kind: 'tuple',
          parameters: [
            { brief: 'Candidate run id', parse: (s) => s, placeholder: 'runId' },
          ],
        },
      },
      func: runAndJudgeCommand,
    }),
    exportMd: buildCommand<ExportMdFlags, [runId: string], CliContext>({
      docs: { brief: 'Export run to LLM-friendly Markdown pack' },
      parameters: {
        flags: {
          out: {
            kind: 'parsed',
            brief: 'Output directory (defaults to runs/<runId>/markdown)',
            optional: true,
            parse: (s) => s,
          },
          mode: {
            kind: 'parsed',
            brief: 'Output mode: by-domain, by-model, or both',
            optional: true,
            parse: (s) => s,
          },
          includeCases: {
            kind: 'boolean',
            brief: 'Include per-case Markdown files',
            optional: true,
          },
          overwrite: {
            kind: 'boolean',
            brief: 'Overwrite existing output',
            optional: true,
          },
          redact: {
            kind: 'parsed',
            brief: 'Redaction mode (none or basic)',
            optional: true,
            parse: (s) => s,
          },
        },
        positional: {
          kind: 'tuple',
          parameters: [{ brief: 'Run id', parse: (s) => s, placeholder: 'runId' }],
        },
      },
      func: exportMdCommand,
    }),
    diff: buildCommand<DiffFlags, [runId1: string, runId2: string], CliContext>({
      docs: { brief: 'Diff two runs' },
      parameters: {
        flags: {
          outDir: {
            kind: 'parsed',
            brief: 'Output directory (defaults to ./runs)',
            optional: true,
            parse: (s) => s,
          },
        },
        positional: {
          kind: 'tuple',
          parameters: [
            { brief: 'Run id 1', parse: (s) => s, placeholder: 'runId1' },
            { brief: 'Run id 2', parse: (s) => s, placeholder: 'runId2' },
          ],
        },
      },
      func: diffCommand,
    }),
    compare: buildCommand<CompareFlags, [runId?: string], CliContext>({
      docs: { brief: 'Generate paired direct-vs-retrieval comparison report' },
      parameters: {
        flags: {
          run: {
            kind: 'parsed',
            brief: 'Scored run id',
            optional: true,
            parse: (s) => s,
          },
          baselineRun: {
            kind: 'parsed',
            brief: 'Scored run id containing baseline rows',
            optional: true,
            parse: (s) => s,
          },
          'baseline-run': {
            kind: 'parsed',
            brief: 'Scored run id containing baseline rows',
            optional: true,
            parse: (s) => s,
          },
          comparisonRun: {
            kind: 'parsed',
            brief: 'Scored run id containing comparison rows',
            optional: true,
            parse: (s) => s,
          },
          'comparison-run': {
            kind: 'parsed',
            brief: 'Scored run id containing comparison rows',
            optional: true,
            parse: (s) => s,
          },
          outDir: {
            kind: 'parsed',
            brief: 'Output directory containing apocbench.sqlite',
            optional: true,
            parse: (s) => s,
          },
          'out-dir': {
            kind: 'parsed',
            brief: 'Output directory containing apocbench.sqlite',
            optional: true,
            parse: (s) => s,
          },
          baselineSuffix: {
            kind: 'parsed',
            brief: 'Baseline model id suffix',
            optional: true,
            parse: (s) => s,
          },
          'baseline-suffix': {
            kind: 'parsed',
            brief: 'Baseline model id suffix',
            optional: true,
            parse: (s) => s,
          },
          comparisonSuffix: {
            kind: 'parsed',
            brief: 'Comparison model id suffix',
            optional: true,
            parse: (s) => s,
          },
          'comparison-suffix': {
            kind: 'parsed',
            brief: 'Comparison model id suffix',
            optional: true,
            parse: (s) => s,
          },
          out: {
            kind: 'parsed',
            brief: 'Write JSON report to path',
            optional: true,
            parse: (s) => s,
          },
        },
        positional: {
          kind: 'tuple',
          parameters: [
            {
              brief: 'Scored run id',
              optional: true,
              parse: (s) => s,
              placeholder: 'runId',
            },
          ],
        },
      },
      func: compareCommand,
    }),
    resume: buildCommand<ResumeFlags, [runId: string], CliContext>({
      docs: { brief: 'Resume a run by id (alias of run)' },
      parameters: {
        flags: {
          config: { kind: 'parsed', brief: 'Path to apocbench.yml', parse: (s) => s },
          dryRun: {
            kind: 'boolean',
            brief: 'Validate only (no API calls)',
            optional: true,
          },
          quiet: { kind: 'boolean', brief: 'Suppress TUI output', optional: true },
          json: { kind: 'boolean', brief: 'Emit JSONL events', optional: true },
          limit: {
            kind: 'parsed',
            brief: 'Limit questions',
            optional: true,
            parse: numberParser,
          },
          categories: {
            kind: 'parsed',
            brief: 'Comma-separated categories',
            optional: true,
            variadic: ',',
            parse: (s) => s,
          },
          questions: {
            kind: 'parsed',
            brief: 'Comma-separated question ids to run',
            optional: true,
            variadic: ',',
            parse: (s) => s,
          },
          models: {
            kind: 'parsed',
            brief: 'Comma-separated model ids to run (matches config models[].id)',
            optional: true,
            variadic: ',',
            parse: (s) => s,
          },
        },
        aliases: { c: 'config' },
        positional: {
          kind: 'tuple',
          parameters: [{ brief: 'Run id', parse: (s) => s, placeholder: 'runId' }],
        },
      },
      func: resumeCommand,
    }),
  },
  docs: { brief: 'apocbench: offline survival/apocalypse LLM benchmark runner' },
});

const app = buildApplication<CliContext>(root, {
  name: 'apocbench',
  scanner: { caseStyle: 'allow-kebab-for-camel', allowArgumentEscapeSequence: true },
});

await run(app, process.argv.slice(2), { process });
