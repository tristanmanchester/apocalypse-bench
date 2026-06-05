import fs from 'node:fs';
import path from 'node:path';

type QuestionSet = {
  id: string;
  questionIds: string[];
};

type Condition = {
  key: string;
  candidateMode: string;
};

type Args = {
  questionSetPath: string;
  out: string;
  repeats: number;
  conditions: string[];
  runName: string;
  modelId: string;
  manifestId: string;
  wikiBaseUrl: string;
  candidateConcurrency: number;
  judgeConcurrency: number;
  temperature: number | null;
  maxTokens: number;
  timeoutMs: number;
};

const DEFAULT_CONDITIONS: Condition[] = [
  { key: 'direct', candidateMode: 'direct' },
  { key: 'bm25', candidateMode: 'agent-bm25' },
  { key: 'hybrid', candidateMode: 'agent-hybrid' },
  { key: 'bm25-research', candidateMode: 'agent-bm25-research' },
];

const DEFAULT_FULL_WIKI_MANIFEST_ID =
  '03873037ca5d577c9142d313ea437762828033188d2f06bbb58e5bb3704a1789';

const CONDITION_BY_KEY = new Map(
  DEFAULT_CONDITIONS.map((condition) => [condition.key, condition]),
);

function usage(): never {
  console.error(`Usage:
  pnpm -s retrieval-debug:config -- [options]

Options:
  --question-set <path>   Question set JSON (default: data/question_sets/retrieval-debug-10.json)
  --out <path>            Output config path (default: apocbench-retrieval-debug-10.generated.json)
  --repeats <n>           Repeat count per condition (default: 10)
  --conditions <a,b,c>    Conditions: direct,bm25,hybrid,bm25-research
  --run-name <name>       Config run.name (default: retrieval-debug-10-gemma31b)
  --model <id>            Candidate OpenRouter model (default: google/gemma-4-31b-it)
  --manifest-id <id>      Wiki corpus/index manifest id (default: current full wiki id)
  --wiki-url <url>        Wiki search service base URL (default: http://127.0.0.1:8765)
  --candidate-concurrency <n>
                          Candidate concurrency per condition chunk (default: 1)
  --judge-concurrency <n> Stored for schema compatibility only; candidateOnly skips judge (default: 1)
  --temperature <n|null>  Candidate temperature (default: 0.5)
  --max-tokens <n>        Candidate max tokens (default: 4000)
  --timeout-ms <n>        Candidate timeout in ms (default: 180000)`);
  process.exit(2);
}

function parseArgs(argv: string[]): Args {
  const args: Args = {
    questionSetPath: 'data/question_sets/retrieval-debug-10.json',
    out: 'apocbench-retrieval-debug-10.generated.json',
    repeats: 10,
    conditions: DEFAULT_CONDITIONS.map((condition) => condition.key),
    runName: 'retrieval-debug-10-gemma31b',
    modelId: 'google/gemma-4-31b-it',
    manifestId: DEFAULT_FULL_WIKI_MANIFEST_ID,
    wikiBaseUrl: 'http://127.0.0.1:8765',
    candidateConcurrency: 1,
    judgeConcurrency: 1,
    temperature: 0.5,
    maxTokens: 4000,
    timeoutMs: 180000,
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
      case '--question-set':
        args.questionSetPath = next();
        break;
      case '--out':
        args.out = next();
        break;
      case '--repeats':
        args.repeats = Number.parseInt(next(), 10);
        break;
      case '--conditions':
        args.conditions = next()
          .split(',')
          .map((condition) => condition.trim())
          .filter(Boolean);
        break;
      case '--run-name':
        args.runName = next();
        break;
      case '--model':
        args.modelId = next();
        break;
      case '--manifest-id':
        args.manifestId = next();
        break;
      case '--wiki-url':
        args.wikiBaseUrl = next();
        break;
      case '--candidate-concurrency':
        args.candidateConcurrency = Number.parseInt(next(), 10);
        break;
      case '--judge-concurrency':
        args.judgeConcurrency = Number.parseInt(next(), 10);
        break;
      case '--temperature': {
        const value = next();
        args.temperature = value === 'null' ? null : Number.parseFloat(value);
        break;
      }
      case '--max-tokens':
        args.maxTokens = Number.parseInt(next(), 10);
        break;
      case '--timeout-ms':
        args.timeoutMs = Number.parseInt(next(), 10);
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

  if (!Number.isInteger(args.repeats) || args.repeats < 1) {
    throw new Error('--repeats must be a positive integer');
  }
  if (!Number.isInteger(args.candidateConcurrency) || args.candidateConcurrency < 1) {
    throw new Error('--candidate-concurrency must be a positive integer');
  }
  if (!Number.isInteger(args.judgeConcurrency) || args.judgeConcurrency < 1) {
    throw new Error('--judge-concurrency must be a positive integer');
  }
  if (args.temperature !== null && !Number.isFinite(args.temperature)) {
    throw new Error('--temperature must be a number or null');
  }
  if (!Number.isInteger(args.maxTokens) || args.maxTokens < 1) {
    throw new Error('--max-tokens must be a positive integer');
  }
  if (!Number.isInteger(args.timeoutMs) || args.timeoutMs < 1) {
    throw new Error('--timeout-ms must be a positive integer');
  }
  for (const condition of args.conditions) {
    if (!CONDITION_BY_KEY.has(condition)) {
      throw new Error(
        `Unknown condition '${condition}'. Known: ${Array.from(CONDITION_BY_KEY.keys()).join(', ')}`,
      );
    }
  }
  return args;
}

function readQuestionSet(filePath: string): QuestionSet {
  const parsed = JSON.parse(fs.readFileSync(filePath, 'utf8')) as Partial<QuestionSet>;
  if (
    !parsed.id ||
    !Array.isArray(parsed.questionIds) ||
    parsed.questionIds.length === 0
  ) {
    throw new Error(`Invalid question set: ${filePath}`);
  }
  return { id: parsed.id, questionIds: parsed.questionIds };
}

function repeatLabel(index: number): string {
  return `r${String(index).padStart(2, '0')}`;
}

function buildConfig(args: Args, questionSet: QuestionSet) {
  const models = [];
  for (const conditionKey of args.conditions) {
    const condition = CONDITION_BY_KEY.get(conditionKey)!;
    for (let repeat = 1; repeat <= args.repeats; repeat += 1) {
      models.push({
        id: `gemma31b-${condition.key}-${repeatLabel(repeat)}`,
        router: 'openrouter',
        model: args.modelId,
        candidateMode: condition.candidateMode,
        params: {
          temperature: args.temperature,
          maxTokens: args.maxTokens,
          timeoutMs: args.timeoutMs,
        },
      });
    }
  }

  return {
    run: {
      name: args.runName,
      datasetPaths: ['./data/question_bank'],
      outDir: './runs',
      resume: false,
      candidateOnly: true,
      questionLimit: null,
      categories: null,
      questionIds: questionSet.questionIds,
      maxBudgetUsd: null,
      retry: {
        maxRetries: 3,
        baseMs: 2000,
        maxMs: 30000,
        maxTotalTimeMs: 300000,
      },
      concurrency: {
        candidate: args.candidateConcurrency,
        judge: args.judgeConcurrency,
      },
    },
    candidate: {
      maxTokens: args.maxTokens,
    },
    judge: {
      router: 'openrouter',
      model: 'openai/gpt-4.1-nano',
      temperature: 0,
      maxTokens: 4096,
      structured: true,
      reasoning: false,
    },
    routers: {
      ollama: {
        baseUrl: 'http://localhost:11434/api',
        apiKeyEnv: null,
        default: {
          temperature: 0.5,
          maxTokens: 4000,
          timeoutMs: 180000,
        },
      },
      openrouter: {
        baseUrl: 'https://openrouter.ai/api/v1',
        apiKeyEnv: 'OPENROUTER_API_KEY',
        headers: {
          'HTTP-Referer': 'https://github.com/tristanmanchester/apocalypse-bench',
          'X-Title': 'apocalypse-bench',
        },
        default: {
          temperature: args.temperature,
          maxTokens: args.maxTokens,
          timeoutMs: args.timeoutMs,
        },
      },
    },
    wiki: {
      enabled: true,
      service: {
        baseUrl: args.wikiBaseUrl,
        timeoutMs: 10000,
      },
      corpus: {
        manifestId: args.manifestId,
        manifestPath: './data/wiki/full/manifest.json',
      },
      index: {
        manifestId: args.manifestId,
        manifestPath: './data/wiki/full/manifest.json',
      },
      limits: {
        searchTopK: 5,
        readMaxChars: 4000,
        contextMaxChars: 12000,
        maxToolCalls: 10,
        maxTurns: 7,
      },
    },
    models,
  };
}

function main(): void {
  const args = parseArgs(process.argv.slice(2));
  const questionSet = readQuestionSet(args.questionSetPath);
  const config = buildConfig(args, questionSet);
  fs.mkdirSync(path.dirname(path.resolve(args.out)), { recursive: true });
  fs.writeFileSync(args.out, `${JSON.stringify(config, null, 2)}\n`);
  console.log(
    `wrote ${args.out}: ${questionSet.questionIds.length} questions x ${args.conditions.length} conditions x ${args.repeats} repeats = ${questionSet.questionIds.length * args.conditions.length * args.repeats} candidates`,
  );
}

main();
