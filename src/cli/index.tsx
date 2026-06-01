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
import type { ApocbenchConfig } from '../core/config/schema';
import { expandDatasetPaths, loadJsonl, loadJsonlMany } from '../core/dataset/loadJsonl';
import type { RunnerEvent } from '../core/runner/orchestrator';
import { runBenchmark } from '../core/runner/orchestrator';
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
import { getTotalQuestions, getQuestionsPerModel } from '../ui/uiStats';

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
  const questionsPerModel = getQuestionsPerModel(config, dataset.lines.length);
  const totalQuestions = getTotalQuestions(config, dataset.lines.length, modelCount);

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
    datasetPath: config.run.datasetPaths ? config.run.datasetPaths.join(',') : config.run.datasetPath!,
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

  const db = openAndMigrate(
    path.resolve(process.cwd(), 'runs', 'apocbench.sqlite'),
  );
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
        .filter((value): value is string => typeof value === 'string' && value.length > 0),
    ),
  );
  const byModel = Array.from(
    new Set(
      results
        .map((row) => row.model_id)
        .filter((value): value is string => typeof value === 'string' && value.length > 0),
    ),
  );
  const cases = includeCases
    ? Array.from(
        new Set(
          results
            .map((row) => row.question_id)
            .filter((value): value is string => typeof value === 'string' && value.length > 0),
        ),
      )
    : [];
  const allModels = Array.from(
    new Set(
      results
        .map((row) => row.model_id)
        .filter((value): value is string => typeof value === 'string' && value.length > 0),
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
    scoreOverall: row.score_overall,
    scoreRubric: normalizeJson(row.score_rubric_json),
    autoFail: typeof row.auto_fail === 'number' ? row.auto_fail === 1 : null,
    autoFailReason: row.auto_fail_reason,
    judgeParsed: normalizeJson(row.judge_parsed_json),
    judgeRaw: normalizeText(row.judge_response_json),
    error: normalizeJson(row.error_json),
  }));

  const resultsJsonl = resultsRecords
    .map((record) => JSON.stringify(record))
    .join('\n');

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
          if (typeof record.id !== 'string' || typeof record.text !== 'string') return null;
          const rubricItem = {
            id: record.id,
            text: record.text,
          } as DomainRenderCase['rubric'][number];
          if (typeof record.weight === 'number') rubricItem.weight = record.weight;
          if (typeof record.maxScore === 'number') rubricItem.maxScore = record.maxScore;
          return rubricItem;
        })
        .filter(
          (item): item is DomainRenderCase['rubric'][number] => item !== null,
        );
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
        caseList.flatMap((caseItem) =>
          caseItem.results.map((result) => result.modelId),
        ),
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
          'run -c apocbench.yml [--dry-run] [--json] [--quiet] [--limit N] [--categories a,b]',
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
          overwrite: { kind: 'boolean', brief: 'Overwrite existing output', optional: true },
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
