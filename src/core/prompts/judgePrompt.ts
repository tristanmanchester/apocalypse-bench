import type { DatasetLine } from '../dataset/schema';

export function buildJudgePrompt(params: {
  question: DatasetLine;
  candidateAnswer: string;
}): string {
  const { question, candidateAnswer } = params;
  const scenario = question.scenario.map(s => `- ${s}`).join('\n');
  const rubric = question.rubric
    .map(r => `- ${r.id} (max ${r.maxScore}, weight ${r.weight}): ${r.text}`)
    .join('\n');
  const autoFail = question.auto_fail.map(a => `- ${a}`).join('\n');
  const rubricIds = question.rubric.map(r => r.id).join(', ');

  return [
    'You are a strict evaluator of rubric compliance and JSON format.',
    'This is a survival scenario: score for practical usefulness and safety under constraints, not academic precision.',
    'Minor terminology/physics imprecision is acceptable if the procedure remains safe and effective.',
    'Penalize major inaccuracies that would cause unsafe, failed, or wasteful execution.',
    'Return rubric_scores as an object with EVERY rubric id as a key.',
    `Required rubric ids: ${rubricIds}.`,
    'Do NOT omit any id. Do NOT add extra ids. Use numbers only.',
    'Score each rubric item by id with a number within [0, maxScore].',
    'If auto_fail triggers, set auto_fail=true, overall_score=0, and include a reason.',
    'Keep notes concise and specific.',
    'Score based only on evidence in the candidate answer; do not assume missing steps were done.',
    'Reward clear steps, checks/verification, stop-work triggers, and safe fallbacks when uncertainty exists.',
    '',
    'Output JSON with exactly these keys:',
    'rubric_scores, auto_fail, auto_fail_reason, overall_score, notes',
    '',
    'JSON template (fill it in; keep all rubric ids present):',
    '{',
    '  "rubric_scores": {',
    question.rubric.map(r => `    "${r.id}": 0`).join(',\n'),
    '  },',
    '  "auto_fail": false,',
    '  "auto_fail_reason": "",',
    '  "overall_score": 0,',
    '  "notes": ""',
    '}',
    '',
    `Question: ${question.id}`,
    `Category: ${question.category}`,
    `Difficulty: ${question.difficulty}`,
    '',
    'Scenario:',
    scenario,
    '',
    'Task prompt:',
    question.prompt,
    '',
    'Rubric:',
    rubric,
    '',
    'Auto-fail conditions:',
    autoFail,
    ...(question.reference_facts && question.reference_facts.length > 0
      ? [
          '',
          'Reference facts (ground truth to check the answer against; the candidate need not state them verbatim, but contradicting them is a major inaccuracy):',
          question.reference_facts.map(f => `- ${f}`).join('\n'),
        ]
      : []),
    '',
    'Candidate answer:',
    candidateAnswer,
  ].join('\n');
}
