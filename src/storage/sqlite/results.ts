import type { DbHandle } from './db';

export type ResultStatus = 'done' | 'candidate_done' | 'candidate_failed' | 'judge_failed' | 'skipped';

export type UpsertResultParams = {
  runId: string;
  modelId: string;
  questionId: string;
  candidatePrompt?: string;
  candidateCompletion?: string;
  candidateMetricsJson?: string;
  retrievalTraceJson?: string;
  judgeRequestJson?: string;
  judgeResponseJson?: string;
  judgeParsedJson?: string;
  scoreOverall?: number;
  scoreRubricJson?: string;
  autoFail?: boolean;
  autoFailReason?: string;
  status: ResultStatus;
  errorJson?: string;
};

export function isResultDone(
  db: DbHandle,
  runId: string,
  modelId: string,
  questionId: string,
): boolean {
  const row = db
    .prepare(
      `
      SELECT status, judge_parsed_json
      FROM model_results
      WHERE run_id = ? AND model_id = ? AND question_id = ?
    `,
    )
    .get(runId, modelId, questionId) as { status?: string; judge_parsed_json?: string } | undefined;

  return row?.status === 'done' && Boolean(row.judge_parsed_json);
}

export function upsertResult(db: DbHandle, p: UpsertResultParams): void {
  db.prepare(
    `
    INSERT INTO model_results (
      run_id, model_id, question_id,
      candidate_prompt, candidate_completion, candidate_metrics_json,
      retrieval_trace_json,
      judge_request_json, judge_response_json, judge_parsed_json,
      score_overall, score_rubric_json,
      auto_fail, auto_fail_reason,
      status, error_json
    ) VALUES (
      @run_id, @model_id, @question_id,
      @candidate_prompt, @candidate_completion, @candidate_metrics_json,
      @retrieval_trace_json,
      @judge_request_json, @judge_response_json, @judge_parsed_json,
      @score_overall, @score_rubric_json,
      @auto_fail, @auto_fail_reason,
      @status, @error_json
    )
    ON CONFLICT(run_id, model_id, question_id) DO UPDATE SET
      candidate_prompt = COALESCE(excluded.candidate_prompt, model_results.candidate_prompt),
      candidate_completion = COALESCE(excluded.candidate_completion, model_results.candidate_completion),
      candidate_metrics_json = COALESCE(excluded.candidate_metrics_json, model_results.candidate_metrics_json),
      retrieval_trace_json = COALESCE(excluded.retrieval_trace_json, model_results.retrieval_trace_json),
      judge_request_json = COALESCE(excluded.judge_request_json, model_results.judge_request_json),
      judge_response_json = COALESCE(excluded.judge_response_json, model_results.judge_response_json),
      judge_parsed_json = COALESCE(excluded.judge_parsed_json, model_results.judge_parsed_json),
      score_overall = COALESCE(excluded.score_overall, model_results.score_overall),
      score_rubric_json = COALESCE(excluded.score_rubric_json, model_results.score_rubric_json),
      auto_fail = COALESCE(excluded.auto_fail, model_results.auto_fail),
      auto_fail_reason = COALESCE(excluded.auto_fail_reason, model_results.auto_fail_reason),
      status = excluded.status,
      error_json = COALESCE(excluded.error_json, model_results.error_json)
  `,
  ).run({
    run_id: p.runId,
    model_id: p.modelId,
    question_id: p.questionId,
    candidate_prompt: p.candidatePrompt ?? null,
    candidate_completion: p.candidateCompletion ?? null,
    candidate_metrics_json: p.candidateMetricsJson ?? null,
    retrieval_trace_json: p.retrievalTraceJson ?? null,
    judge_request_json: p.judgeRequestJson ?? null,
    judge_response_json: p.judgeResponseJson ?? null,
    judge_parsed_json: p.judgeParsedJson ?? null,
    score_overall: p.scoreOverall ?? null,
    score_rubric_json: p.scoreRubricJson ?? null,
    auto_fail: p.autoFail == null ? null : p.autoFail ? 1 : 0,
    auto_fail_reason: p.autoFailReason ?? null,
    status: p.status,
    error_json: p.errorJson ?? null,
  });
}
