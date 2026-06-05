# apocalypse-bench: benchmark overview

## What this benchmark is

`apocalypse-bench` (CLI: `apocbench`) is a TypeScript benchmark for evaluating
how useful LLMs are as an **offline survival/apocalypse assistant**.

It runs a fixed question bank (`data/question_bank/*.jsonl`) across one or more
**candidate models** (local via Ollama, hosted via OpenRouter, or any supported
OpenAI-compatible router), then uses a separate **judge model** to score each
answer against a structured rubric.

## What it measures (and what it doesn’t)

### What it measures

- **Practical usefulness under constraints**: Can a model produce a step-by-step
  plan using only the stated resources?
- **Correctness on hard, knowledge-dense tasks**: many questions are
  human-expert difficulty and turn on exact quantities, temperatures, doses,
  formulas, and procedures.
- **Refusal vs. usefulness**: the bank deliberately includes
  hazardous-sounding but legitimate survival tasks, such as making black powder
  to clear stumps or distilling ether for a field amputation. A model that
  refuses a legitimate question is less useful and scores 0 on it.
- **Safety awareness**: does it flag the real hazards and give concrete
  mitigations and stop-work triggers?
- **Actionability and concision**: answers are expected to be structured and
  scannable, not essays.

### What it does not measure

- Internet retrieval or browsing. The default direct track assumes no internet
  and no tools.
- Live web tool use. The optional Wikipedia retrieval track uses only a local,
  offline corpus and is reported as a separate condition.
- Long-term memory or personalization.
- A public leaderboard (reports are local artifacts).

## How a run works

1. **Load config** from `apocbench.yml` (or JSON), validating against a strict schema.
2. **Load dataset** directly from JSONL in `data/question_bank/*.jsonl` (the single source of truth).
3. For each candidate model and each question:
   - Generate a candidate answer with an offline survival assistant system
     prompt. The direct candidate sees only the question's `prompt`.
   - Send the question, rubric, and candidate answer to the judge model.
   - Parse judge output (structured JSON), compute a per-question score, and persist artifacts.
4. **Aggregate** scores by category/difficulty and produce reports (HTML + Markdown) under `runs/<runId>/`.

## Scoring model

Each question has:

- A **rubric of exactly 10 criteria**, each scored 0/1 with weight 1, for a
  0-10 scale per question. Every item is a specific, binary, load-bearing check
  such as a named fact, quantity, step, landmark, ratio, or temperature.
- **`reference_facts`**: the precise correct values the judge checks against.
- **`auto_fail` conditions** that force the score to 0. Every question includes
  a refusal condition, so refusing scores 0. The others fire only on advice that
  is technically wrong or physically unsafe, never on the subject matter.

The judge returns per-rubric scores, an auto-fail flag and reason, and notes.
The runner computes the overall score: 0 if auto-fail, otherwise the weighted
sum.

## Dataset structure (V2)

The V2 bank is **500 questions** across 13 categories, stored as one JSONL file
per category. Each line follows:

`id, area, category, title, difficulty, task_type, scenario[], prompt, rubric[10], auto_fail[], reference_facts[], version`

Difficulty is one of `Easy | Medium | Hard | Very Hard`. The full enumerated
bank is browsable in `docs/question-bank.md`, a generated read-only export.
Schema and authoring rules live in `data/question_bank/info.md`.

| Code    | Category       | Focus                                                                                    |
| ------- | -------------- | ---------------------------------------------------------------------------------------- |
| `AGR`   | Agriculture    | soil, rotations, seed-saving, pests, irrigation, livestock, food production              |
| `CHEM`  | Chemistry      | acids, distillation, soap, tanning, reagents, energetics, extraction                     |
| `COMMS` | Communications | radio, antennas, signalling, field telephony, HF propagation                             |
| `ENG`   | Engineering    | structures, rigging, pumps, kilns, machines (civil + mechanical), with load/stress calcs |
| `ENR`   | Energy         | heat, fuel, combustion, micro-generation, batteries, wiring                              |
| `ETH`   | Ethics         | allocation, consent, justice, conscription, disclosure, intergenerational tradeoffs      |
| `PH`    | Public Health  | water, sanitation, food safety, outbreak control, disease transmission                   |
| `MAT`   | Materials      | metalwork, ceramics, glass, mortar, adhesives, textiles, fabrication                     |
| `MEAS`  | Measurement    | standards, surveying, navigation, timekeeping, calibration, experimental method          |
| `MED`   | Medicine       | first aid, trauma, obstetrics, field procedures, dosing logic                            |
| `ORG`   | Organisation   | coordination, logistics, registries, ICS, decision processes, crisis ops                 |
| `PED`   | Pedagogy       | teaching and competency assessment in high-risk skills                                   |
| `SAFE`  | Safety         | risk assessment, SOPs, CO/fume/fire/confined-space hazards, rescue                       |

## Where outputs go

Runs write to `runs/<runId>/` and typically include:

- `summary.json` (aggregates by model/category/difficulty)
- An HTML report, such as `report.html`
- Per-model exports (including Markdown summaries under `runs/<runId>/markdown/`)
