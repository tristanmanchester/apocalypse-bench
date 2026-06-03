import fs from 'node:fs';

import Database from 'better-sqlite3';

type Args = {
  db: string;
  run: string;
  questionSetPath: string;
  baseline: string;
  modelPrefix: string;
  out?: string;
};

type Row = {
  question_id: string;
  model_id: string;
  score_overall: number | null;
  auto_fail: number | null;
  auto_fail_reason: string | null;
  status: string;
};

type ParsedModelId = {
  condition: string;
  repeat: string;
};

function usage(): never {
  console.error(`Usage:
  pnpm -s retrieval-debug:report -- --run <run-id> [options]

Options:
  --db <path>             SQLite DB path (default: runs/apocbench.sqlite)
  --question-set <path>   Question set JSON (default: data/question_sets/retrieval-debug-10.json)
  --baseline <condition>  Baseline condition key (default: direct)
  --model-prefix <prefix> Model id prefix before -<condition>-rNN (default: gemma31b)
  --out <path>            Optional JSON output path`);
  process.exit(2);
}

function parseArgs(argv: string[]): Args {
  const args: Args = {
    db: 'runs/apocbench.sqlite',
    questionSetPath: 'data/question_sets/retrieval-debug-10.json',
    baseline: 'direct',
    modelPrefix: 'gemma31b',
    run: '',
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const next = () => {
      const value = argv[index + 1];
      if (!value || value.startsWith('--')) usage();
      index += 1;
      return value;
    };
    switch (arg) {
      case '--':
        break;
      case '--db':
        args.db = next();
        break;
      case '--run':
        args.run = next();
        break;
      case '--question-set':
        args.questionSetPath = next();
        break;
      case '--baseline':
        args.baseline = next();
        break;
      case '--model-prefix':
        args.modelPrefix = next();
        break;
      case '--out':
        args.out = next();
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

  if (!args.run) usage();
  return args;
}

function escapeRegex(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function parseModelId(modelId: string, modelPrefix: string): ParsedModelId | null {
  const match = new RegExp(`^${escapeRegex(modelPrefix)}-(.+)-r(\\d+)$`).exec(modelId);
  if (!match) return null;
  return {
    condition: match[1]!,
    repeat: `r${match[2]!}`,
  };
}

function mean(values: number[]): number | null {
  return values.length === 0
    ? null
    : values.reduce((total, value) => total + value, 0) / values.length;
}

function quantile(values: number[], q: number): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((left, right) => left - right);
  const index = (sorted.length - 1) * q;
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  if (lower === upper) return sorted[lower]!;
  const weight = index - lower;
  return sorted[lower]! * (1 - weight) + sorted[upper]! * weight;
}

function summarizeValues(rows: Array<Row & ParsedModelId>) {
  const done = rows.filter(
    (row) => row.status === 'done' && typeof row.score_overall === 'number',
  );
  const scores = done.map((row) => row.score_overall as number);
  const q1 = quantile(scores, 0.25);
  const q3 = quantile(scores, 0.75);
  return {
    n: rows.length,
    done: done.length,
    failures: rows.length - done.length,
    meanScore: mean(scores),
    medianScore: quantile(scores, 0.5),
    minScore: scores.length > 0 ? Math.min(...scores) : null,
    maxScore: scores.length > 0 ? Math.max(...scores) : null,
    iqr: q1 == null || q3 == null ? null : q3 - q1,
    autoFails: done.filter((row) => row.auto_fail === 1).length,
    autoFailRate:
      done.length === 0
        ? null
        : done.filter((row) => row.auto_fail === 1).length / done.length,
    zeros: done.filter((row) => row.score_overall === 0).length,
    zeroRate:
      done.length === 0
        ? null
        : done.filter((row) => row.score_overall === 0).length / done.length,
  };
}

function main(): void {
  const args = parseArgs(process.argv.slice(2));
  const questionSet = JSON.parse(fs.readFileSync(args.questionSetPath, 'utf8')) as {
    id?: string;
    questionIds?: string[];
  };
  const selectedQuestionIds = new Set(questionSet.questionIds ?? []);
  const db = new Database(args.db, { readonly: true });
  const rawRows = db
    .prepare(
      `select question_id, model_id, score_overall, auto_fail, auto_fail_reason, status
       from model_results
       where run_id = ?
       order by question_id, model_id`,
    )
    .all(args.run) as Row[];

  const rows = rawRows
    .filter(
      (row) => selectedQuestionIds.size === 0 || selectedQuestionIds.has(row.question_id),
    )
    .map((row) => {
      const parsed = parseModelId(row.model_id, args.modelPrefix);
      return parsed ? { ...row, ...parsed } : null;
    })
    .filter((row): row is Row & ParsedModelId => row != null);

  const byCondition = new Map<string, Array<Row & ParsedModelId>>();
  const byQuestionCondition = new Map<string, Array<Row & ParsedModelId>>();
  for (const row of rows) {
    const conditionRows = byCondition.get(row.condition) ?? [];
    conditionRows.push(row);
    byCondition.set(row.condition, conditionRows);

    const key = `${row.question_id}\u0000${row.condition}`;
    const questionRows = byQuestionCondition.get(key) ?? [];
    questionRows.push(row);
    byQuestionCondition.set(key, questionRows);
  }

  const baseline = new Map<string, number>();
  for (const row of rows) {
    if (
      row.condition === args.baseline &&
      row.status === 'done' &&
      typeof row.score_overall === 'number'
    ) {
      baseline.set(`${row.question_id}\u0000${row.repeat}`, row.score_overall);
    }
  }

  const conditions = Array.from(byCondition.keys()).sort();
  const conditionSummaries = conditions.map((condition) => {
    const conditionRows = byCondition.get(condition)!;
    const deltas = conditionRows
      .filter(
        (row) =>
          row.condition !== args.baseline &&
          row.status === 'done' &&
          typeof row.score_overall === 'number',
      )
      .map((row) => {
        const baselineScore = baseline.get(`${row.question_id}\u0000${row.repeat}`);
        return baselineScore == null
          ? null
          : (row.score_overall as number) - baselineScore;
      })
      .filter((delta): delta is number => delta != null);
    const baselineMean = mean(Array.from(baseline.values()));
    const meanDelta = mean(deltas);
    return {
      condition,
      ...summarizeValues(conditionRows),
      pairedVsBaseline:
        condition === args.baseline
          ? null
          : {
              baseline: args.baseline,
              pairedCount: deltas.length,
              meanDelta,
              medianDelta: quantile(deltas, 0.5),
              meanLiftPercent:
                baselineMean == null || baselineMean === 0 || meanDelta == null
                  ? null
                  : (meanDelta / baselineMean) * 100,
              wins: deltas.filter((delta) => delta > 0).length,
              losses: deltas.filter((delta) => delta < 0).length,
              ties: deltas.filter((delta) => delta === 0).length,
            },
    };
  });

  const questionSummaries = Array.from(selectedQuestionIds).map((questionId) => {
    const perCondition = Object.fromEntries(
      conditions.map((condition) => [
        condition,
        summarizeValues(byQuestionCondition.get(`${questionId}\u0000${condition}`) ?? []),
      ]),
    );
    return { questionId, conditions: perCondition };
  });

  const report = {
    run: args.run,
    questionSet: questionSet.id ?? args.questionSetPath,
    baseline: args.baseline,
    modelPrefix: args.modelPrefix,
    conditions: conditionSummaries,
    questions: questionSummaries,
  };

  if (args.out) {
    fs.writeFileSync(args.out, `${JSON.stringify(report, null, 2)}\n`);
  }
  console.log(JSON.stringify(report, null, 2));
}

main();
