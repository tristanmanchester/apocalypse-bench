import YAML from 'yaml';

type FrontMatter = Record<string, unknown>;

export type ExportRunMetadata = {
  runId: string;
  createdAt: string;
  completedAt?: string | null;
  durationMs?: number | null;
  toolVersion?: string | null;
  status?: string | null;
  config?: unknown;
  datasetPath?: string | null;
  datasetSha256?: string | null;
  promptTemplateHash?: string | null;
  gitCommit?: string | null;
  gitDirty?: boolean | null;
};

export type ExportRubricItem = {
  id: string;
  text: string;
  weight?: number;
  maxScore?: number;
};

export type ExportCase = {
  caseId: string;
  category: string;
  difficulty: string;
  scenario: string[];
  prompt: string;
  rubric: ExportRubricItem[];
  autoFail: string[];
  referenceFacts?: string[] | null;
};

export type ExportModelResult = {
  runId: string;
  modelId: string;
  caseId: string;
  status: string;
  prompt?: string | null;
  answer?: string | null;
  candidatePrompt?: string | null;
  candidateMetrics?: Record<string, unknown> | null;
  retrievalTrace?: Record<string, unknown> | null;
  scoreOverall?: number | null;
  scoreRubric?: Record<string, unknown> | null;
  autoFail?: boolean | null;
  autoFailReason?: string | null;
  judgeParsed?: Record<string, unknown> | null;
  judgeRaw?: string | null;
  error?: Record<string, unknown> | null;
};

export type ExportRecord = {
  run: ExportRunMetadata;
  cases: ExportCase[];
  results: ExportModelResult[];
};

type RenderParams = {
  frontMatter?: FrontMatter;
  body?: string;
};

export type RunIndexParams = {
  frontMatter?: FrontMatter;
  byDomain?: string[];
  byModel?: string[];
  cases?: string[];
};

export type DomainRenderCaseResult = {
  modelId: string;
  status: string;
  answer?: string | null;
  scoreOverall?: number | null;
  autoFail?: boolean | null;
  autoFailReason?: string | null;
  judgeParsed?: Record<string, unknown> | null;
  judgeRaw?: string | null;
  error?: Record<string, unknown> | null;
};

export type DomainRenderCase = {
  caseId: string;
  category: string;
  difficulty: string;
  scenario: string[];
  prompt: string;
  rubric: ExportRubricItem[];
  autoFail: string[];
  results: DomainRenderCaseResult[];
};

export type DomainRenderParams = {
  frontMatter?: FrontMatter;
  domain: string;
  cases: DomainRenderCase[];
};

export type ModelRenderCaseResult = {
  caseId: string;
  category: string;
  difficulty: string;
  scenario: string[];
  prompt: string;
  rubric: ExportRubricItem[];
  autoFail: string[];
  status: string;
  answer?: string | null;
  scoreOverall?: number | null;
  autoFailFlag?: boolean | null;
  autoFailReason?: string | null;
  judgeParsed?: Record<string, unknown> | null;
  judgeRaw?: string | null;
  error?: Record<string, unknown> | null;
};

export type ModelRenderParams = {
  frontMatter?: FrontMatter;
  model: string;
  cases: ModelRenderCaseResult[];
};

export type CaseRenderResult = {
  modelId: string;
  status: string;
  answer?: string | null;
  scoreOverall?: number | null;
  autoFail?: boolean | null;
  autoFailReason?: string | null;
  judgeParsed?: Record<string, unknown> | null;
  judgeRaw?: string | null;
  error?: Record<string, unknown> | null;
};

export type CaseRenderParams = {
  frontMatter?: FrontMatter;
  caseId: string;
  domain: string;
  difficulty: string;
  scenario: string[];
  prompt: string;
  rubric: ExportRubricItem[];
  autoFail: string[];
  results: CaseRenderResult[];
};

export function renderRunIndexMd(params: RunIndexParams = {}): string {
  const lines: string[] = ['# Run Index'];

  const byDomain = (params.byDomain ?? [])
    .filter((value) => value.length > 0)
    .sort((a, b) => a.localeCompare(b));
  if (byDomain.length > 0) {
    lines.push('', '## By Domain');
    for (const domain of byDomain) {
      const slug = slugify(domain);
      lines.push(`- [${domain}](by-domain/${slug}.md)`);
    }
  }

  const byModel = (params.byModel ?? [])
    .filter((value) => value.length > 0)
    .sort((a, b) => a.localeCompare(b));
  if (byModel.length > 0) {
    lines.push('', '## By Model');
    for (const model of byModel) {
      const slug = slugify(model);
      lines.push(`- [${model}](by-model/${slug}.md)`);
    }
  }

  const cases = (params.cases ?? [])
    .filter((value) => value.length > 0)
    .sort((a, b) => a.localeCompare(b));
  if (cases.length > 0) {
    lines.push('', '## Cases');
    for (const caseId of cases) {
      lines.push(`- [${caseId}](cases/${caseId}.md)`);
    }
  }

  return renderWithFrontMatter({ frontMatter: params.frontMatter, body: lines.join('\n') });
}

export function renderByDomainMd(params: DomainRenderParams): string {
  const lines: string[] = [`# Domain: ${params.domain}`];
  const sortedCases = [...params.cases].sort((a, b) => a.caseId.localeCompare(b.caseId));

  sortedCases.forEach((caseItem, index) => {
    if (index > 0) lines.push('', '---');
    const sortedResults = [...caseItem.results].sort((a, b) =>
      a.modelId.localeCompare(b.modelId),
    );
    lines.push(
      '',
      `## Case: ${caseItem.caseId}`,
      '## Scenario',
      renderTextBlock(caseItem.scenario.join('\n')),
      '## Prompt',
      renderTextBlock(caseItem.prompt),
      '## Rubric',
    );

    if (caseItem.rubric.length === 0) {
      lines.push('MISSING');
    } else {
      for (const item of caseItem.rubric) {
        const weight =
          typeof item.weight === 'number' ? ` weight=${item.weight}` : '';
        const maxScore =
          typeof item.maxScore === 'number' ? ` maxScore=${item.maxScore}` : '';
        lines.push(`- [${item.id}] ${item.text}${weight}${maxScore}`);
      }
    }

    if (caseItem.autoFail.length > 0) {
      lines.push('', `Auto-fail: ${caseItem.autoFail.join('; ')}`);
    }

    lines.push('## Results');
    if (sortedResults.length === 0) {
      lines.push('MISSING');
    } else {
      for (const result of sortedResults) {
        lines.push(`### ${result.modelId}`);
        lines.push(`- status: ${result.status}`);
        if (typeof result.scoreOverall === 'number')
          lines.push(`- score_overall: ${result.scoreOverall}`);
        if (typeof result.autoFail === 'boolean')
          lines.push(`- auto_fail: ${result.autoFail}`);
        if (result.autoFailReason)
          lines.push(`- auto_fail_reason: ${result.autoFailReason}`);
        lines.push('', renderTextBlock(result.answer ?? ''));
      }
    }

    lines.push('## Judge');
    if (sortedResults.length === 0) {
      lines.push('MISSING');
    } else {
      for (const result of sortedResults) {
        lines.push(`### ${result.modelId}`);
        const judgeJson = result.judgeParsed
          ? JSON.stringify(result.judgeParsed, null, 2)
          : result.judgeRaw ?? '';
        lines.push(renderJsonBlock(judgeJson));
        if (result.error) {
          lines.push('', 'Error:', renderJsonBlock(JSON.stringify(result.error, null, 2)));
        }
      }
    }
  });

  return renderWithFrontMatter({ frontMatter: params.frontMatter, body: lines.join('\n') });
}

export function renderByModelMd(params: ModelRenderParams): string {
  const lines: string[] = [`# Model: ${params.model}`];
  const sortedCases = [...params.cases].sort((a, b) => a.caseId.localeCompare(b.caseId));

  sortedCases.forEach((caseItem, index) => {
    if (index > 0) lines.push('', '---');
    lines.push(
      '',
      `## Case: ${caseItem.caseId}`,
      `- domain: ${caseItem.category}`,
      `- difficulty: ${caseItem.difficulty}`,
      '## Scenario',
      renderTextBlock(caseItem.scenario.join('\n')),
      '## Prompt',
      renderTextBlock(caseItem.prompt),
      '## Rubric',
    );

    if (caseItem.rubric.length === 0) {
      lines.push('MISSING');
    } else {
      for (const item of caseItem.rubric) {
        const weight =
          typeof item.weight === 'number' ? ` weight=${item.weight}` : '';
        const maxScore =
          typeof item.maxScore === 'number' ? ` maxScore=${item.maxScore}` : '';
        lines.push(`- [${item.id}] ${item.text}${weight}${maxScore}`);
      }
    }

    if (caseItem.autoFail.length > 0) {
      lines.push('', `Auto-fail: ${caseItem.autoFail.join('; ')}`);
    }

    lines.push('## Results');
    lines.push(`- status: ${caseItem.status}`);
    if (typeof caseItem.scoreOverall === 'number')
      lines.push(`- score_overall: ${caseItem.scoreOverall}`);
    if (typeof caseItem.autoFailFlag === 'boolean')
      lines.push(`- auto_fail: ${caseItem.autoFailFlag}`);
    if (caseItem.autoFailReason) lines.push(`- auto_fail_reason: ${caseItem.autoFailReason}`);
    lines.push('', renderTextBlock(caseItem.answer ?? ''));

    lines.push('## Judge');
    const judgeJson = caseItem.judgeParsed
      ? JSON.stringify(caseItem.judgeParsed, null, 2)
      : caseItem.judgeRaw ?? '';
    lines.push(renderJsonBlock(judgeJson));
    if (caseItem.error) {
      lines.push('', 'Error:', renderJsonBlock(JSON.stringify(caseItem.error, null, 2)));
    }
  });

  return renderWithFrontMatter({ frontMatter: params.frontMatter, body: lines.join('\n') });
}

export function renderCaseMd(params: CaseRenderParams): string {
  const lines: string[] = [`# Case: ${params.caseId}`];
  lines.push(`- domain: ${params.domain}`);
  lines.push(`- difficulty: ${params.difficulty}`);

  lines.push('', '## Scenario', renderTextBlock(params.scenario.join('\n')));
  lines.push('## Prompt', renderTextBlock(params.prompt));
  lines.push('## Rubric');
  if (params.rubric.length === 0) {
    lines.push('MISSING');
  } else {
    for (const item of params.rubric) {
      const weight =
        typeof item.weight === 'number' ? ` weight=${item.weight}` : '';
      const maxScore =
        typeof item.maxScore === 'number' ? ` maxScore=${item.maxScore}` : '';
      lines.push(`- [${item.id}] ${item.text}${weight}${maxScore}`);
    }
  }
  if (params.autoFail.length > 0) {
    lines.push('', `Auto-fail: ${params.autoFail.join('; ')}`);
  }

  const sortedResults = [...params.results].sort((a, b) =>
    a.modelId.localeCompare(b.modelId),
  );
  lines.push('', '## Results');
  if (sortedResults.length === 0) {
    lines.push('MISSING');
  } else {
    for (const result of sortedResults) {
      lines.push(`### ${result.modelId}`);
      lines.push(`- status: ${result.status}`);
      if (typeof result.scoreOverall === 'number')
        lines.push(`- score_overall: ${result.scoreOverall}`);
      if (typeof result.autoFail === 'boolean')
        lines.push(`- auto_fail: ${result.autoFail}`);
      if (result.autoFailReason) lines.push(`- auto_fail_reason: ${result.autoFailReason}`);
      lines.push('', renderTextBlock(result.answer ?? ''));
      const judgeJson = result.judgeParsed
        ? JSON.stringify(result.judgeParsed, null, 2)
        : result.judgeRaw ?? '';
      lines.push('## Judge', renderJsonBlock(judgeJson));
      if (result.error) {
        lines.push('', 'Error:', renderJsonBlock(JSON.stringify(result.error, null, 2)));
      }
    }
  }

  return renderWithFrontMatter({ frontMatter: params.frontMatter, body: lines.join('\n') });
}

export function slugify(value: string): string {
  const normalized = value
    .normalize('NFKD')
    .replace(/[^\x20-\x7E]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
  return normalized || 'unknown';
}

export function writeFrontMatter(frontMatter: FrontMatter | undefined): string {
  if (!frontMatter || Object.keys(frontMatter).length === 0) return '';
  const yaml = YAML.stringify(frontMatter).trimEnd();
  return `---\n${normalizeNewlines(yaml)}\n---\n`;
}

export function normalizeNewlines(value: string): string {
  return value.replace(/\r\n?/g, '\n');
}

function renderTextBlock(value: string): string {
  const normalized = normalizeNewlines(value).trim();
  if (!normalized) return 'MISSING';
  return `\`\`\`text\n${normalized}\n\`\`\``;
}

function renderJsonBlock(value: string): string {
  const normalized = normalizeNewlines(value).trim();
  if (!normalized) return 'MISSING';
  return `\`\`\`json\n${normalized}\n\`\`\``;
}

function renderWithFrontMatter(params: RenderParams): string {
  const frontMatter = writeFrontMatter(params.frontMatter);
  const body = params.body ? normalizeNewlines(params.body) : '';
  return `${frontMatter}${body}`;
}
