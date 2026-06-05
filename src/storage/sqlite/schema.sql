PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  tool_version TEXT NOT NULL,
  config_json TEXT NOT NULL,
  dataset_path TEXT NOT NULL,
  dataset_sha256 TEXT NOT NULL,
  prompt_template_hash TEXT NOT NULL,
  status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS questions (
  run_id TEXT NOT NULL,
  question_id TEXT NOT NULL,
  category TEXT,
  difficulty TEXT,
  scenario TEXT,
  prompt TEXT,
  rubric_json TEXT,
  auto_fail_json TEXT,
  PRIMARY KEY (run_id, question_id),
  FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS model_results (
  run_id TEXT NOT NULL,
  model_id TEXT NOT NULL,
  question_id TEXT NOT NULL,
  candidate_prompt TEXT,
  candidate_completion TEXT,
  candidate_metrics_json TEXT,
  retrieval_trace_json TEXT,
  judge_request_json TEXT,
  judge_response_json TEXT,
  judge_parsed_json TEXT,
  score_overall REAL,
  score_rubric_json TEXT,
  auto_fail INTEGER,
  auto_fail_reason TEXT,
  status TEXT NOT NULL,
  error_json TEXT,
  PRIMARY KEY (run_id, model_id, question_id),
  FOREIGN KEY (run_id, question_id) REFERENCES questions(run_id, question_id) ON DELETE CASCADE
);
