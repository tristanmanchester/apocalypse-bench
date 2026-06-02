export function renderHtmlReport(params: {
  runId: string;
  summaryJson: unknown;
  results?: unknown;
}): string {
  const summaryJson = params.summaryJson as Record<string, unknown> | null;
  const summaryPretty = JSON.stringify(params.summaryJson, null, 2);

  const results = Array.isArray(params.results)
    ? (params.results as Record<string, unknown>[])
    : [];
  const hasAnyResultDetails = results.some(
    (r) =>
      typeof r.candidate_prompt === 'string' ||
      typeof r.prompt === 'string' ||
      typeof r.candidate_completion === 'string',
  );

  const models = Array.isArray(summaryJson?.models) ? summaryJson?.models : [];
  const sortedModels = [...models].sort(
    (a, b) => number(b?.overallScore) - number(a?.overallScore),
  );

  return `<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>apocbench - ${escapeHtml(params.runId)}</title>
    <style>
      :root { color-scheme: light dark; }
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; line-height: 1.35; }
      code, pre, table { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
      h1 { margin: 0 0 8px; }
      h2 { margin: 24px 0 8px; }
      .muted { opacity: 0.75; }
      .card { border: 1px solid rgba(127,127,127,0.25); border-radius: 10px; padding: 12px 14px; margin: 12px 0; }
      .grid { display: grid; grid-template-columns: max-content 1fr; gap: 6px 12px; }
      table { width: 100%; border-collapse: collapse; }
      th, td { padding: 6px 10px; border-bottom: 1px solid rgba(127,127,127,0.25); vertical-align: top; }
      th { text-align: left; font-weight: 600; }
      td.num, th.num { text-align: right; }
      tr:hover td { background: rgba(127,127,127,0.08); }
      details { margin: 12px 0; }
      summary { cursor: pointer; font-weight: 600; }
      pre { background: rgba(127,127,127,0.12); padding: 12px; border-radius: 10px; overflow: auto; }
    </style>
  </head>
  <body>
    <h1>apocbench</h1>
    <p>runId: <code>${escapeHtml(params.runId)}</code></p>

    <div class="card">
      <div class="grid">
        ${row('createdAt', string(summaryJson?.createdAt))}
        ${row('datasetPath', string(summaryJson?.datasetPath))}
        ${row('datasetSha256', string(summaryJson?.datasetSha256))}
        ${row('promptTemplateHash', string((summaryJson as Record<string, unknown>)?.promptTemplateHash))}
        ${row(
          'judge',
          (() => {
            const judge =
              summaryJson && typeof summaryJson === 'object'
                ? (summaryJson as Record<string, unknown>).judge
                : null;

            if (!judge || typeof judge !== 'object') return null;
            const judgeObj = judge as Record<string, unknown>;
            const model = string(judgeObj.model);
            if (!model) return null;
            const provider = string(judgeObj.provider);
            return provider ? `${model} (${provider})` : model;
          })(),
        )}
      </div>
      <div class="muted" style="margin-top: 8px">
        Note: prompts and completions may contain sensitive data.
      </div>
    </div>

    ${
      sortedModels.length > 0
        ? `
    <h2>Models</h2>
    <table>
      <thead>
        <tr>
          <th>modelId</th>
          <th class="num" title="Sum of all scores for this model">score</th>
          <th class="num" title="Mean score per completed question">avg</th>
          <th class="num" title="Questions completed (candidate + judge succeeded)">done</th>
          <th class="num" title="Execution errors (candidate or judge crashed)">errors</th>
          <th class="num" title="Questions not attempted">skipped</th>
          <th class="num" title="% of completed questions flagged by judge (e.g. refusal, unsafe content) — scored as 0">auto-fail %</th>
          <th class="num" title="Median latency">p50 ms</th>
          <th class="num" title="90th percentile latency">p90 ms</th>
          <th class="num" title="Executed or blocked wiki tool calls recorded in retrieval traces">wiki tool calls</th>
          <th class="num" title="Wiki source reads recorded in retrieval traces">wiki reads</th>
        </tr>
      </thead>
      <tbody>
        ${sortedModels.map((m) => renderModelRow(m)).join('\n')}
      </tbody>
    </table>
    <p class="muted" style="font-size: 0.9em; margin-top: 8px;">
      <strong>errors</strong> = execution failures (API/parsing errors). 
      <strong>auto-fail %</strong> = % of completed questions where the judge flagged the response (e.g. refusal, unsafe content); these receive a score of 0 but are counted as "done".
    </p>

    <h2>Breakdowns</h2>
    ${sortedModels.map((m) => renderModelDetails(m)).join('\n')}
    `
        : `<p class="muted">No model summaries found.</p>`
    }

    ${
      results.length > 0 && !hasAnyResultDetails
        ? `<div class="card"><div class="muted">Per-question details are missing for this run.</div></div>`
        : ''
    }

    ${renderResultsSection(results)}

    <h2>Raw summary JSON</h2>
    <pre>${escapeHtml(summaryPretty)}</pre>
  </body>
</html>`;
}

function renderResultsSection(results: Record<string, unknown>[]): string {
  if (results.length === 0) return '';

  const withDetails = results.filter(
    (r) =>
      typeof r.candidate_prompt === 'string' ||
      typeof r.prompt === 'string' ||
      typeof r.candidate_completion === 'string' ||
      typeof r.judge_parsed_json === 'string',
  );
  if (withDetails.length === 0) {
    return `
      <h2>Results</h2>
      <div class="card">
        <div class="muted">
          No per-question prompt/completion/judge details were found in the run database.
        </div>
      </div>
    `.trim();
  }

  const byModel = new Map<string, Record<string, unknown>[]>();
  for (const r of results) {
    const modelId = String(r.model_id ?? 'unknown');
    const list = byModel.get(modelId) ?? [];
    list.push(r);
    byModel.set(modelId, list);
  }

  const sortedModels = Array.from(byModel.entries()).sort((a, b) =>
    a[0].localeCompare(b[0]),
  );

  return `
    <h2>Results</h2>
    ${sortedModels
      .map(([modelId, rows]) => {
        const sorted = [...rows].sort((a, b) =>
          String(a.question_id).localeCompare(String(b.question_id)),
        );
        return `
          <details>
            <summary>${escapeHtml(modelId)} (${sorted.length})</summary>
            ${sorted
              .map((r) => {
                const qid = String(r.question_id ?? 'unknown');
                const status = String(r.status ?? 'unknown');
                const overall = fmtNum(r.score_overall, 2);
                const autoFail =
                  r.auto_fail === 1 ? 'true' : r.auto_fail === 0 ? 'false' : '—';
                const category = r.category ? String(r.category) : 'unknown';
                const difficulty = r.difficulty ? String(r.difficulty) : 'unknown';

                const prompt =
                  typeof r.candidate_prompt === 'string'
                    ? r.candidate_prompt
                    : typeof r.prompt === 'string'
                      ? r.prompt
                      : null;
                const candidate =
                  typeof r.candidate_completion === 'string'
                    ? r.candidate_completion
                    : null;
                const retrievalTrace =
                  typeof r.retrieval_trace_json === 'string'
                    ? safeJsonParse(r.retrieval_trace_json)
                    : null;

                const judgeParsed =
                  typeof r.judge_parsed_json === 'string'
                    ? safeJsonParse(r.judge_parsed_json)
                    : null;
                const judgeNotes =
                  isObjectRecord(judgeParsed) && typeof judgeParsed.notes === 'string'
                    ? judgeParsed.notes
                    : null;
                const rubricScores =
                  isObjectRecord(judgeParsed) && isObjectRecord(judgeParsed.rubric_scores)
                    ? judgeParsed.rubric_scores
                    : null;

                const scoreRubric =
                  typeof r.score_rubric_json === 'string'
                    ? safeJsonParse(r.score_rubric_json)
                    : null;

                const errorJson = typeof r.error_json === 'string' ? r.error_json : null;

                return `
                  <details class="card">
                    <summary>${escapeHtml(qid)} — ${escapeHtml(category)} / ${escapeHtml(difficulty)} — status=${escapeHtml(
                      status,
                    )} score=${escapeHtml(overall)} auto_fail=${escapeHtml(autoFail)}</summary>
                    ${prompt ? `<h3>Prompt</h3><pre>${escapeHtml(prompt)}</pre>` : ''}
                    ${retrievalTrace ? renderRetrievalTrace(retrievalTrace) : ''}
                    ${candidate ? `<h3>Candidate</h3><pre>${escapeHtml(candidate)}</pre>` : ''}
                    ${judgeNotes ? `<h3>Judge notes</h3><pre>${escapeHtml(judgeNotes)}</pre>` : ''}
                    ${rubricScores ? `<h3>Judge rubric_scores</h3><pre>${escapeHtml(JSON.stringify(rubricScores, null, 2))}</pre>` : ''}
                    ${scoreRubric ? `<h3>Computed rubric scores</h3><pre>${escapeHtml(JSON.stringify(scoreRubric, null, 2))}</pre>` : ''}
                    ${errorJson ? `<h3>Error</h3><pre>${escapeHtml(errorJson)}</pre>` : ''}
                  </details>
                `.trim();
              })
              .join('\n')}
          </details>
        `.trim();
      })
      .join('\n')}
  `.trim();
}

function renderRetrievalTrace(trace: unknown): string {
  if (!isObjectRecord(trace)) return '';
  const mode = typeof trace.mode === 'string' ? trace.mode : 'unknown';
  const searches = Array.isArray(trace.searches) ? trace.searches : [];
  const reads = Array.isArray(trace.reads) ? trace.reads : [];
  const toolCalls = Array.isArray(trace.toolCalls) ? trace.toolCalls : [];
  const toolCallCount =
    typeof trace.toolCallCount === 'number' && Number.isFinite(trace.toolCallCount)
      ? trace.toolCallCount
      : toolCalls.length;
  const titles = reads
    .map((read) =>
      isObjectRecord(read) && typeof read.title === 'string' ? read.title : null,
    )
    .filter((title): title is string => Boolean(title));
  return `
    <h3>Wiki retrieval</h3>
    <div class="grid">
      ${row('mode', mode)}
      ${row('tool calls', String(toolCallCount))}
      ${row('searches', String(searches.length))}
      ${row('sources read', String(reads.length))}
      ${titles.length > 0 ? row('source titles', titles.join(', ')) : ''}
    </div>
    <pre>${escapeHtml(JSON.stringify(trace, null, 2))}</pre>
  `;
}

function safeJsonParse(raw: string): unknown {
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function isObjectRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

function row(label: string, value: string | null): string {
  if (!value) return '';
  return `<div class="muted">${escapeHtml(label)}</div><div><code>${value}</code></div>`;
}

function number(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

function string(v: unknown): string | null {
  return typeof v === 'string' && v.length > 0 ? escapeHtml(v) : null;
}

function fmtNum(v: unknown, digits = 2): string {
  const n = typeof v === 'number' && Number.isFinite(v) ? v : null;
  return n == null ? '—' : n.toFixed(digits);
}

function fmtInt(v: unknown): string {
  const n = typeof v === 'number' && Number.isFinite(v) ? v : null;
  return n == null ? '—' : String(Math.trunc(n));
}

function fmtPct(v: unknown): string {
  const n = typeof v === 'number' && Number.isFinite(v) ? v : null;
  return n == null ? '—' : `${(n * 100).toFixed(1)}%`;
}

function fmtMs(v: unknown): string {
  const n = typeof v === 'number' && Number.isFinite(v) ? v : null;
  return n == null ? '—' : `${Math.round(n)}ms`;
}

function renderModelRow(m: Record<string, unknown>): string {
  const latency =
    typeof m.latencyMs === 'object' && m.latencyMs !== null
      ? (m.latencyMs as Record<string, unknown>)
      : {};
  const retrieval = isObjectRecord(m.retrieval) ? m.retrieval : null;
  return `
    <tr>
      <td>${escapeHtml(String(m?.modelId ?? 'unknown'))}</td>
      <td class="num">${fmtNum(m?.overallScore, 2)}</td>
      <td class="num">${fmtNum(m?.overallScoreMean, 4)}</td>
      <td class="num">${fmtInt(m?.completed)}</td>
      <td class="num">${fmtInt(m?.failures)}</td>
      <td class="num">${fmtInt(m?.skipped)}</td>
      <td class="num">${fmtPct(m?.autoFailRate)}</td>
      <td class="num">${fmtMs(latency.medianMs)}</td>
      <td class="num">${fmtMs(latency.p90Ms)}</td>
      <td class="num">${fmtInt(retrieval?.toolCallCount)}</td>
      <td class="num">${fmtInt(retrieval?.readCount)}</td>
    </tr>
  `.trim();
}

function renderBreakdownTable(params: {
  kind: 'category' | 'difficulty';
  rows: Record<string, unknown>[];
}): string {
  const key = params.kind === 'category' ? 'category' : 'difficulty';
  const sorted = [...params.rows].sort(
    (a, b) =>
      number(b?.overallScore) - number(a?.overallScore) ||
      String(a?.[key]).localeCompare(String(b?.[key])),
  );

  return `
    <table>
      <thead>
        <tr>
          <th>${key}</th>
          <th class="num" title="Sum of all scores">score</th>
          <th class="num" title="Mean score per completed question">avg</th>
          <th class="num" title="Questions completed">done</th>
          <th class="num" title="Execution errors">errors</th>
          <th class="num" title="Questions not attempted">skipped</th>
          <th class="num" title="% flagged by judge">auto-fail %</th>
          <th class="num" title="Median latency">p50 ms</th>
          <th class="num" title="90th percentile latency">p90 ms</th>
        </tr>
      </thead>
      <tbody>
        ${sorted
          .map((r) => {
            const latency =
              typeof r.latencyMs === 'object' && r.latencyMs !== null
                ? (r.latencyMs as Record<string, unknown>)
                : {};
            return `
              <tr>
                <td>${escapeHtml(String(r?.[key] ?? 'unknown'))}</td>
                <td class="num">${fmtNum(r?.overallScore, 2)}</td>
                <td class="num">${fmtNum(r?.overallScoreMean, 4)}</td>
                <td class="num">${fmtInt(r?.completed)}</td>
                <td class="num">${fmtInt(r?.failures)}</td>
                <td class="num">${fmtInt(r?.skipped)}</td>
                <td class="num">${fmtPct(r?.autoFailRate)}</td>
                <td class="num">${fmtMs(latency.medianMs)}</td>
                <td class="num">${fmtMs(latency.p90Ms)}</td>
              </tr>
            `.trim();
          })
          .join('\n')}
      </tbody>
    </table>
  `.trim();
}

function renderModelDetails(m: Record<string, unknown>): string {
  const categories = Array.isArray(m?.categoryBreakdown) ? m.categoryBreakdown : [];
  const difficulties = Array.isArray(m?.difficultyBreakdown) ? m.difficultyBreakdown : [];
  const retrieval = isObjectRecord(m?.retrieval) ? m.retrieval : null;

  return `
    <details>
      <summary>${escapeHtml(String(m?.modelId ?? 'unknown'))}</summary>
      <div class="card">
        <div class="grid">
          ${row('totalQuestions', fmtInt(m?.totalQuestions))}
          ${row('completed', fmtInt(m?.completed))}
          ${row('failures', fmtInt(m?.failures))}
          ${row('skipped', fmtInt(m?.skipped))}
          ${row('overallScore', fmtNum(m?.overallScore, 2))}
          ${row('overallScoreMean', fmtNum(m?.overallScoreMean, 4))}
          ${row('autoFailRate', fmtPct(m?.autoFailRate))}
        </div>
      </div>
      ${retrieval ? renderRetrievalSummary(retrieval) : ''}
      ${
        categories.length > 0
          ? `<h3>By category</h3>${renderBreakdownTable({ kind: 'category', rows: categories })}`
          : `<p class="muted">No category breakdown available.</p>`
      }
      ${
        difficulties.length > 0
          ? `<h3>By difficulty</h3>${renderBreakdownTable({ kind: 'difficulty', rows: difficulties })}`
          : `<p class="muted">No difficulty breakdown available.</p>`
      }
    </details>
  `.trim();
}

function renderRetrievalSummary(retrieval: Record<string, unknown>): string {
  const latency = isObjectRecord(retrieval.latencyMs) ? retrieval.latencyMs : {};
  const titles = Array.isArray(retrieval.uniqueSourceTitles)
    ? retrieval.uniqueSourceTitles.map((title) => String(title)).slice(0, 12)
    : [];
  return `
    <h3>Wiki retrieval summary</h3>
    <div class="card">
      <div class="grid">
        ${row('traceCount', fmtInt(retrieval.traceCount))}
        ${row('modes', formatModes(retrieval.modes))}
        ${row('toolCallCount', fmtInt(retrieval.toolCallCount))}
        ${row('searchCount', fmtInt(retrieval.searchCount))}
        ${row('readCount', fmtInt(retrieval.readCount))}
        ${row('retrieval p50', fmtMs(latency.medianMs))}
        ${row('retrieval p90', fmtMs(latency.p90Ms))}
        ${titles.length > 0 ? row('source titles', titles.join(', ')) : ''}
      </div>
    </div>
  `.trim();
}

function formatModes(value: unknown): string | null {
  if (!isObjectRecord(value)) return null;
  const entries = Object.entries(value).sort((a, b) => a[0].localeCompare(b[0]));
  if (entries.length === 0) return null;
  return entries.map(([mode, count]) => `${mode}: ${fmtInt(count)}`).join(', ');
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
