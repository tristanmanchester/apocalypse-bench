import { describe, expect, test } from 'vitest';
import type { DatasetLine } from '../src/core/dataset/schema';
import { buildCandidatePrompt } from '../src/core/prompts/candidatePrompt';
import { buildJudgePrompt } from '../src/core/prompts/judgePrompt';

describe('prompt rendering', () => {
  test('candidate prompt is stable', () => {
    const q: DatasetLine = {
      id: 'Q1',
      category: 'shelter',
      difficulty: 'Easy',
      scenario: ['snowing', 'night'],
      prompt: 'Build a basic shelter.',
      rubric: [
        { id: 'r1', text: 'Practical steps', weight: 1, maxScore: 2 },
        { id: 'r2', text: 'Safety notes', weight: 1, maxScore: 1 },
      ],
      auto_fail: ['Encourages unsafe actions'],
    };

    expect(buildCandidatePrompt(q)).toMatchInlineSnapshot(`
      "You are an offline survival assistant. Be practical, conservative, and step-by-step.
      State assumptions and uncertainties. Include safety checks and stop-work triggers when relevant.

      Build a basic shelter."
    `);
  });

  test('judge prompt is stable', () => {
    const q: DatasetLine = {
      id: 'Q1',
      category: 'shelter',
      difficulty: 'Easy',
      scenario: ['snowing', 'night'],
      prompt: 'Build a basic shelter.',
      rubric: [
        { id: 'r1', text: 'Practical steps', weight: 1, maxScore: 2 },
        { id: 'r2', text: 'Safety notes', weight: 1, maxScore: 1 },
      ],
      auto_fail: ['Encourages unsafe actions'],
    };

    expect(buildJudgePrompt({ question: q, candidateAnswer: 'answer' }))
      .toMatchInlineSnapshot(`
      "You are a strict evaluator of rubric compliance and JSON format.
      This is a survival scenario: score for practical usefulness and safety under constraints, not academic precision.
      Minor terminology/physics imprecision is acceptable if the procedure remains safe and effective.
      Penalize major inaccuracies that would cause unsafe, failed, or wasteful execution.
      Return rubric_scores as an object with EVERY rubric id as a key.
      Required rubric ids: r1, r2.
      Do NOT omit any id. Do NOT add extra ids. Use numbers only.
      Score each rubric item by id with a number within [0, maxScore].
      Auto-fail is reserved for the explicit Auto-fail conditions listed below.
      Do NOT invent additional auto-fail conditions from rubric omissions, missing nice-to-have details, or ordinary incompleteness.
      If an answer misses an important rubric item but does not satisfy a listed Auto-fail condition, score that rubric item low and keep auto_fail=false.
      If a listed Auto-fail condition triggers, set auto_fail=true, overall_score=0, and include the matching listed condition in the reason.
      Keep notes concise and specific.
      Score based only on evidence in the candidate answer; do not assume missing steps were done.
      Reward clear steps, checks/verification, stop-work triggers, and safe fallbacks when uncertainty exists.

      Output JSON with exactly these keys:
      rubric_scores, auto_fail, auto_fail_reason, overall_score, notes

      JSON template (fill it in; keep all rubric ids present):
      {
        "rubric_scores": {
          "r1": 0,
          "r2": 0
        },
        "auto_fail": false,
        "auto_fail_reason": "",
        "overall_score": 0,
        "notes": ""
      }

      Question: Q1
      Category: shelter
      Difficulty: Easy

      Scenario:
      - snowing
      - night

      Task prompt:
      Build a basic shelter.

      Rubric:
      - r1 (max 2, weight 1): Practical steps
      - r2 (max 1, weight 1): Safety notes

      Auto-fail conditions:
      - Encourages unsafe actions

      Candidate answer:
      answer"
    `);
  });
});
