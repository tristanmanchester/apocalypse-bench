import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";

type SummaryJson = {
  runId: string;
  createdAt: string;
  models: Array<{
    modelId: string;
    totalQuestions: number;
    completed: number;
    failures: number;
    skipped: number;
    autoFailCount: number;
    autoFailRate: number;
    overallScore: number;
    overallScoreMean: number;
    rubricScoreSum: number;
    latencyMs: {
      medianMs: number;
      meanMs: number;
      p90Ms: number;
      minMs: number;
      maxMs: number;
    };
    categoryBreakdown: Array<{
      category: string;
      totalQuestions: number;
      completed: number;
      failures: number;
      skipped: number;
      autoFailCount: number;
      autoFailRate: number;
      overallScore: number;
      overallScoreMean: number;
      rubricScoreSum: number;
      latencyMs: {
        medianMs: number;
        meanMs: number;
        p90Ms: number;
        minMs: number;
        maxMs: number;
      };
    }>;
    difficultyBreakdown: Array<{
      difficulty: string;
      totalQuestions: number;
      completed: number;
      failures: number;
      skipped: number;
      autoFailCount: number;
      autoFailRate: number;
      overallScore: number;
      overallScoreMean: number;
      rubricScoreSum: number;
      latencyMs: {
        medianMs: number;
        meanMs: number;
        p90Ms: number;
        minMs: number;
        maxMs: number;
      };
    }>;
  }>;
};

type ResultsRow = {
  runId: string;
  modelId: string;
  caseId: string;
  status: string;
  scoreOverall?: number;
  autoFail?: boolean;
  autoFailReason?: string;
  scoreRubric?: Record<string, number>;
  candidateMetrics?: {
    latencyMs?: number;
    usage?: {
      promptTokens?: number;
      completionTokens?: number;
      totalTokens?: number;
    };
    costUsd?: number;
  };
};

function getArg(flag: string): string | undefined {
  const idx = process.argv.indexOf(flag);
  if (idx === -1) return undefined;
  return process.argv[idx + 1];
}

function requiredArg(flag: string): string {
  const value = getArg(flag);
  if (!value) {
    throw new Error(`Missing required arg ${flag}`);
  }
  return value;
}

function stableId(value: string): string {
  return createHash("sha256").update(value).digest("hex").slice(0, 12);
}

function csvEscape(value: unknown): string {
  if (value === null || value === undefined) return "";
  const s = String(value);
  if (/[\n\r",]/.test(s)) return `"${s.replaceAll("\"", '""')}"`;
  return s;
}

function writeCsv(outPath: string, headers: string[], rows: Array<Record<string, unknown>>): void {
  mkdirSync(dirname(outPath), { recursive: true });
  const lines: string[] = [];
  lines.push(headers.join(","));
  for (const row of rows) {
    lines.push(headers.map((h) => csvEscape(row[h])).join(","));
  }
  writeFileSync(outPath, `${lines.join("\n")}\n`, "utf8");
}

function readSummary(summaryPath: string): SummaryJson {
  return JSON.parse(readFileSync(summaryPath, "utf8")) as SummaryJson;
}

function parseJsonl(filePath: string): unknown[] {
  const content = readFileSync(filePath, "utf8");
  const lines = content.split(/\r?\n/).filter((l) => l.trim().length > 0);
  return lines.map((l) => JSON.parse(l));
}

function normalizeModelLabel(modelId: string): string {
  return modelId.replaceAll("/", "-");
}

function main(): void {
  const runDir = resolve(requiredArg("--run"));
  const outDir = resolve(getArg("--out") ?? join(runDir, "plot_csv"));

  const summaryPath = join(runDir, "summary.json");
  const resultsJsonlPath = join(runDir, "markdown", "results.jsonl");

  const summary = readSummary(summaryPath);
  const runId = summary.runId;

  const results: ResultsRow[] = parseJsonl(resultsJsonlPath) as ResultsRow[];

  // 1) Overall ranking bar chart
  writeCsv(
    join(outDir, "overall_ranking.csv"),
    ["runId", "modelId", "overallScoreMean", "autoFailRate", "autoFailCount", "totalQuestions", "latencyMedianMs", "latencyP90Ms"],
    summary.models.map((m) => ({
      runId,
      modelId: m.modelId,
      overallScoreMean: m.overallScoreMean,
      autoFailRate: m.autoFailRate,
      autoFailCount: m.autoFailCount,
      totalQuestions: m.totalQuestions,
      latencyMedianMs: m.latencyMs.medianMs,
      latencyP90Ms: m.latencyMs.p90Ms,
    })),
  );

  // 2) Auto-fail rate bar chart (separate file for convenience)
  writeCsv(
    join(outDir, "autofail_rates.csv"),
    ["runId", "modelId", "autoFailRate", "autoFailCount", "totalQuestions"],
    summary.models.map((m) => ({
      runId,
      modelId: m.modelId,
      autoFailRate: m.autoFailRate,
      autoFailCount: m.autoFailCount,
      totalQuestions: m.totalQuestions,
    })),
  );

  // 3) Score vs difficulty
  const scoreByDifficultyRows: Array<Record<string, unknown>> = [];
  for (const m of summary.models) {
    for (const d of m.difficultyBreakdown) {
      scoreByDifficultyRows.push({
        runId,
        modelId: m.modelId,
        difficulty: d.difficulty,
        overallScoreMean: d.overallScoreMean,
        autoFailRate: d.autoFailRate,
        autoFailCount: d.autoFailCount,
        totalQuestions: d.totalQuestions,
      });
    }
  }
  writeCsv(
    join(outDir, "difficulty_scores.csv"),
    ["runId", "modelId", "difficulty", "overallScoreMean", "autoFailRate", "autoFailCount", "totalQuestions"],
    scoreByDifficultyRows,
  );

  // 4) Category heatmaps (score + auto-fail)
  const categoryScoreRows: Array<Record<string, unknown>> = [];
  const categoryAutoFailRows: Array<Record<string, unknown>> = [];
  for (const m of summary.models) {
    for (const c of m.categoryBreakdown) {
      categoryScoreRows.push({
        runId,
        modelId: m.modelId,
        category: c.category,
        overallScoreMean: c.overallScoreMean,
        totalQuestions: c.totalQuestions,
      });
      categoryAutoFailRows.push({
        runId,
        modelId: m.modelId,
        category: c.category,
        autoFailRate: c.autoFailRate,
        autoFailCount: c.autoFailCount,
        totalQuestions: c.totalQuestions,
      });
    }
  }
  writeCsv(
    join(outDir, "category_scores.csv"),
    ["runId", "modelId", "category", "overallScoreMean", "totalQuestions"],
    categoryScoreRows,
  );
  writeCsv(
    join(outDir, "category_autofails.csv"),
    ["runId", "modelId", "category", "autoFailRate", "autoFailCount", "totalQuestions"],
    categoryAutoFailRows,
  );

  // 5) Risk vs competence scatter
  writeCsv(
    join(outDir, "risk_vs_competence.csv"),
    [
      "runId",
      "modelId",
      "overallScoreMean",
      "autoFailRate",
      "latencyMedianMs",
      "latencyP90Ms",
      "totalQuestions",
    ],
    summary.models.map((m) => ({
      runId,
      modelId: m.modelId,
      overallScoreMean: m.overallScoreMean,
      autoFailRate: m.autoFailRate,
      latencyMedianMs: m.latencyMs.medianMs,
      latencyP90Ms: m.latencyMs.p90Ms,
      totalQuestions: m.totalQuestions,
    })),
  );

  // 6) Per-question distribution (scores/latency/tokens)
  writeCsv(
    join(outDir, "per_question_scores.csv"),
    [
      "runId",
      "modelId",
      "caseId",
      "status",
      "scoreOverall",
      "autoFail",
      "autoFailReason",
      "latencyMs",
      "promptTokens",
      "completionTokens",
      "totalTokens",
      "costUsd",
    ],
    results.map((r) => ({
      runId: r.runId ?? runId,
      modelId: r.modelId,
      caseId: r.caseId,
      status: r.status,
      scoreOverall: r.scoreOverall ?? "",
      autoFail: r.autoFail ?? "",
      autoFailReason: r.autoFailReason ?? "",
      latencyMs: r.candidateMetrics?.latencyMs ?? "",
      promptTokens: r.candidateMetrics?.usage?.promptTokens ?? "",
      completionTokens: r.candidateMetrics?.usage?.completionTokens ?? "",
      totalTokens: r.candidateMetrics?.usage?.totalTokens ?? "",
      costUsd: r.candidateMetrics?.costUsd ?? "",
    })),
  );

  // 7) Auto-fail reasons breakdown
  const reasons = new Map<string, number>();
  for (const r of results) {
    if (!r.autoFail) continue;
    const key = `${r.modelId}||${(r.autoFailReason ?? "").trim()}`;
    reasons.set(key, (reasons.get(key) ?? 0) + 1);
  }
  const autoFailReasonRows: Array<Record<string, unknown>> = [];
  for (const [key, count] of reasons.entries()) {
    const [modelId, reason] = key.split("||");
    autoFailReasonRows.push({
      runId,
      modelId,
      autoFailReason: reason,
      count,
    });
  }
  writeCsv(
    join(outDir, "autofail_reasons.csv"),
    ["runId", "modelId", "autoFailReason", "count"],
    autoFailReasonRows,
  );

  // 8) Rubric item means (per model) + long-form (model, rubricItem)
  const rubricSums = new Map<string, Record<string, number>>();
  const rubricCounts = new Map<string, Record<string, number>>();

  for (const r of results) {
    const per = r.scoreRubric ?? {};
    if (!rubricSums.has(r.modelId)) rubricSums.set(r.modelId, {});
    if (!rubricCounts.has(r.modelId)) rubricCounts.set(r.modelId, {});
    const sums = rubricSums.get(r.modelId)!;
    const counts = rubricCounts.get(r.modelId)!;
    for (const [k, v] of Object.entries(per)) {
      if (typeof v !== "number") continue;
      sums[k] = (sums[k] ?? 0) + v;
      counts[k] = (counts[k] ?? 0) + 1;
    }
  }

  const rubricMeansLong: Array<Record<string, unknown>> = [];
  for (const [modelId, sums] of rubricSums.entries()) {
    const counts = rubricCounts.get(modelId) ?? {};
    for (const rubricItem of Object.keys(sums).sort()) {
      const denom = counts[rubricItem] ?? 0;
      rubricMeansLong.push({
        runId,
        modelId,
        rubricItem,
        meanScore: denom ? sums[rubricItem] / denom : "",
        n: denom,
      });
    }
  }
  writeCsv(
    join(outDir, "rubric_means_long.csv"),
    ["runId", "modelId", "rubricItem", "meanScore", "n"],
    rubricMeansLong,
  );

  // 9) Latency profile
  writeCsv(
    join(outDir, "latency_profile.csv"),
    ["runId", "modelId", "medianMs", "p90Ms", "meanMs", "minMs", "maxMs"],
    summary.models.map((m) => ({
      runId,
      modelId: m.modelId,
      medianMs: m.latencyMs.medianMs,
      p90Ms: m.latencyMs.p90Ms,
      meanMs: m.latencyMs.meanMs,
      minMs: m.latencyMs.minMs,
      maxMs: m.latencyMs.maxMs,
    })),
  );

  // Manifest: helps you see what got generated
  const generatedFiles = [
    "overall_ranking.csv",
    "autofail_rates.csv",
    "difficulty_scores.csv",
    "category_scores.csv",
    "category_autofails.csv",
    "risk_vs_competence.csv",
    "per_question_scores.csv",
    "autofail_reasons.csv",
    "rubric_means_long.csv",
    "latency_profile.csv",
  ];

  const manifestRows = generatedFiles.map((f) => {
    const p = join(outDir, f);
    return {
      runId,
      file: f,
      path: relative(process.cwd(), p),
      id: stableId(`${runId}:${f}`),
    };
  });
  writeCsv(join(outDir, "_manifest.csv"), ["runId", "file", "path", "id"], manifestRows);

  // eslint-disable-next-line no-console
  console.log(`Wrote ${generatedFiles.length + 1} CSV files to ${relative(process.cwd(), outDir)}`);
  for (const f of ["_manifest.csv", ...generatedFiles]) {
    // eslint-disable-next-line no-console
    console.log(`- ${join(relative(process.cwd(), outDir), f)}`);
  }
}

main();
