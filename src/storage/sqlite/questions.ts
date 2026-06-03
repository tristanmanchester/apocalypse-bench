import type { DbHandle } from './db';
import type { DatasetLine } from '../../core/dataset/schema';

export function insertQuestions(
  db: DbHandle,
  runId: string,
  questions: DatasetLine[],
): void {
  const stmt = db.prepare(
    `
    INSERT INTO questions (
      run_id, question_id, category, difficulty, scenario, prompt, rubric_json, auto_fail_json
    ) VALUES (
      @run_id, @question_id, @category, @difficulty, @scenario, @prompt, @rubric_json, @auto_fail_json
    )
    ON CONFLICT(run_id, question_id) DO UPDATE SET
      category = excluded.category,
      difficulty = excluded.difficulty,
      scenario = excluded.scenario,
      prompt = excluded.prompt,
      rubric_json = excluded.rubric_json,
      auto_fail_json = excluded.auto_fail_json
  `,
  );

  const insertMany = db.transaction((rows: DatasetLine[]) => {
    for (const q of rows) {
      stmt.run({
        run_id: runId,
        question_id: q.id,
        category: q.category,
        difficulty: q.difficulty,
        scenario: JSON.stringify(q.scenario),
        prompt: q.prompt,
        rubric_json: JSON.stringify(q.rubric),
        auto_fail_json: JSON.stringify(q.auto_fail),
      });
    }
  });

  insertMany(questions);
}
