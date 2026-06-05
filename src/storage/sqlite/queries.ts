import type { DbHandle } from './db';

export type ModelResultRow = {
  run_id: string;
  model_id: string;
  question_id: string;
  score_overall: number | null;
  score_rubric_json: string | null;
  auto_fail: number | null;
  auto_fail_reason: string | null;
  status: string;
  candidate_metrics_json: string | null;
  candidate_prompt: string | null;
  candidate_completion: string | null;
  retrieval_trace_json: string | null;
  judge_response_json: string | null;
  judge_parsed_json: string | null;
  error_json: string | null;
  category: string | null;
  difficulty: string | null;
  prompt: string | null;
  scenario: string | null;
  rubric_json: string | null;
  auto_fail_json: string | null;
};

export function listRunModelResults(db: DbHandle, runId: string): ModelResultRow[] {
  return db
    .prepare(
      `
      SELECT
        mr.run_id,
        mr.model_id,
        mr.question_id,
        mr.score_overall,
        mr.score_rubric_json,
        mr.auto_fail,
        mr.auto_fail_reason,
        mr.status,
        mr.candidate_metrics_json,
        mr.candidate_prompt,
        mr.candidate_completion,
        mr.retrieval_trace_json,
        mr.judge_response_json,
        mr.judge_parsed_json,
        mr.error_json,
        q.category,
        q.difficulty,
        q.prompt,
        q.scenario,
        q.rubric_json,
        q.auto_fail_json
      FROM model_results mr
      LEFT JOIN questions q
        ON q.run_id = mr.run_id AND q.question_id = mr.question_id
      WHERE mr.run_id = ?
    `,
    )
    .all(runId) as ModelResultRow[];
}
