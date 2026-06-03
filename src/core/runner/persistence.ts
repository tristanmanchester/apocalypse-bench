import path from 'node:path';
import type { DbHandle } from '../../storage/sqlite/db';
import type { ApocbenchConfig } from '../config/schema';
import type { DatasetLine } from '../dataset/schema';
import { openAndMigrate } from '../../storage/sqlite/migrate';
import { insertRun, updateRunStatus } from '../../storage/sqlite/runs';
import { insertQuestions } from '../../storage/sqlite/questions';
import {
  isCandidateDone,
  isResultDone,
  upsertResult,
  type UpsertResultParams,
} from '../../storage/sqlite/results';
import { listRunModelResults } from '../../storage/sqlite/queries';

export type RunnerDb = DbHandle;

export function openRunnerDb(params: { outDir: string }): RunnerDb {
  return openAndMigrate(path.resolve(process.cwd(), params.outDir, 'apocbench.sqlite'));
}

export function ensureRunStarted(params: {
  db: RunnerDb;
  runId: string;
  toolVersion: string;
  config: ApocbenchConfig;
  datasetPath: string;
  datasetSha256: string;
  promptTemplateHash: string;
}): void {
  const {
    db,
    runId,
    toolVersion,
    config,
    datasetPath,
    datasetSha256,
    promptTemplateHash,
  } = params;
  const existingRun = db
    .prepare('SELECT run_id FROM runs WHERE run_id = ?')
    .get(runId) as { run_id: string } | undefined;
  if (!existingRun) {
    insertRun(db, {
      run_id: runId,
      created_at: new Date().toISOString(),
      tool_version: toolVersion,
      config_json: JSON.stringify(config),
      dataset_path: datasetPath,
      dataset_sha256: datasetSha256,
      prompt_template_hash: promptTemplateHash,
      status: 'running',
    });
  } else {
    updateRunStatus(db, runId, 'running');
  }
}

export function insertRunQuestions(
  db: RunnerDb,
  runId: string,
  questions: DatasetLine[],
): void {
  insertQuestions(db, runId, questions);
}

export function isRunResultDone(
  db: RunnerDb,
  runId: string,
  modelId: string,
  questionId: string,
): boolean {
  return isResultDone(db, runId, modelId, questionId);
}

export function isRunCandidateDone(
  db: RunnerDb,
  runId: string,
  modelId: string,
  questionId: string,
): boolean {
  return isCandidateDone(db, runId, modelId, questionId);
}

export function upsertRunResult(db: RunnerDb, params: UpsertResultParams): void {
  upsertResult(db, params);
}

export function listRunResults(db: RunnerDb, runId: string) {
  return listRunModelResults(db, runId);
}

export function updateRunStatusForRun(db: RunnerDb, runId: string, status: string): void {
  updateRunStatus(db, runId, status);
}
