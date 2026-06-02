import { z } from 'zod';

export const judgeOutputSchema = z
  .object({
    rubric_scores: z.record(z.string(), z.number()),
    auto_fail: z.boolean(),
    auto_fail_reason: z.string().optional(),
    overall_score: z.number(),
    notes: z.string(),
    unsafe_flags: z.array(z.string()).optional(),
  })
  .strict();

export type JudgeOutput = z.infer<typeof judgeOutputSchema>;

/**
 * Builds a Zod schema for judge output with explicit rubric ID keys.
 * This ensures the JSON Schema sent to the model explicitly requires
 * each rubric ID as a property, rather than using additionalProperties.
 */
export function buildJudgeOutputSchemaWithRubricIds(rubricIds: string[]): z.ZodTypeAny {
  // Build an object type with explicit keys for each rubric ID
  const rubricScoresShape: Record<string, z.ZodNumber> = {};
  for (const id of rubricIds) {
    rubricScoresShape[id] = z.number().describe(`Score for rubric item ${id}`);
  }

  return z
    .object({
      rubric_scores: z.object(rubricScoresShape).strict(),
      auto_fail: z.boolean(),
      auto_fail_reason: z.string().optional(),
      overall_score: z.number(),
      notes: z.string(),
      unsafe_flags: z.array(z.string()).optional(),
    })
    .strict();
}

export type AggregateLatencyStats = {
  medianMs: number | null;
  meanMs: number | null;
  p90Ms: number | null;
  minMs: number | null;
  maxMs: number | null;
};

export type AggregateGroupSummary = {
  totalQuestions: number;
  completed: number;
  failures: number;
  skipped: number;
  autoFailCount: number;
  autoFailRate: number | null;
  overallScore: number;
  overallScoreMean: number | null;
  rubricScoreSum: number;
  latencyMs: AggregateLatencyStats;
};

export type CategoryBreakdownSummary = AggregateGroupSummary & { category: string };
export type DifficultyBreakdownSummary = AggregateGroupSummary & { difficulty: string };

export type ModelRunSummary = {
  modelId: string;
  totalQuestions: number;
  completed: number;
  failures: number;
  skipped: number;
  autoFailCount: number;
  autoFailRate: number | null;
  overallScore: number;
  overallScoreMean: number | null;
  rubricScoreSum: number;
  latencyMs: AggregateLatencyStats;
  categoryBreakdown: CategoryBreakdownSummary[];
  difficultyBreakdown: DifficultyBreakdownSummary[];
};
