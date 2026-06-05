import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

export type QmdRerankDocument = {
  id: string;
  text: string;
  title?: string;
};

export type QmdRerankResult = QmdRerankDocument & {
  score: number;
  originalIndex: number;
};

type QmdLlamaCpp = {
  rerank: (
    query: string,
    documents: Array<{ file: string; text: string; title?: string }>,
    options?: { model?: string },
  ) => Promise<{
    results: Array<{ file: string; score: number; index: number }>;
    model: string;
  }>;
};

type QmdLlmModule = {
  DEFAULT_RERANK_MODEL_URI: string;
  LlamaCpp: new (config?: {
    rerankModel?: string;
    inactivityTimeoutMs?: number;
  }) => QmdLlamaCpp;
};

let modulePromise: Promise<QmdLlmModule> | undefined;
let reranker: QmdLlamaCpp | undefined;

export async function rerankWithQmd(
  query: string,
  documents: QmdRerankDocument[],
): Promise<QmdRerankResult[]> {
  if (documents.length <= 1) {
    return documents.map((document, index) => ({
      ...document,
      score: documents.length === 1 ? 1 : 0,
      originalIndex: index,
    }));
  }

  const qmd = await loadQmdLlmModule();
  reranker ??= new qmd.LlamaCpp({
    rerankModel: process.env.APOCBENCH_QMD_RERANK_MODEL ?? qmd.DEFAULT_RERANK_MODEL_URI,
    inactivityTimeoutMs: 10 * 60 * 1000,
  });

  const result = await reranker.rerank(
    query,
    documents.map((document) => ({
      file: document.id,
      text: document.text,
      title: document.title,
    })),
  );

  const byId = new Map(documents.map((document, index) => [document.id, { document, index }]));
  return result.results
    .map((ranked) => {
      const source = byId.get(ranked.file);
      if (!source) return undefined;
      return {
        ...source.document,
        score: ranked.score,
        originalIndex: source.index,
      };
    })
    .filter((ranked): ranked is QmdRerankResult => ranked != null);
}

async function loadQmdLlmModule(): Promise<QmdLlmModule> {
  modulePromise ??= import(resolveQmdLlmModuleUrl()).then((module) => {
    const maybeModule = module as Partial<QmdLlmModule>;
    if (!maybeModule.LlamaCpp || !maybeModule.DEFAULT_RERANK_MODEL_URI) {
      throw new Error('QMD reranker module did not expose LlamaCpp rerank support');
    }
    return {
      LlamaCpp: maybeModule.LlamaCpp,
      DEFAULT_RERANK_MODEL_URI: maybeModule.DEFAULT_RERANK_MODEL_URI,
    };
  });
  return modulePromise;
}

function resolveQmdLlmModuleUrl(): string {
  const directPath = path.resolve(
    process.cwd(),
    'node_modules',
    '@tobilu',
    'qmd',
    'dist',
    'llm.js',
  );
  if (fs.existsSync(directPath)) return pathToFileURL(directPath).href;

  const pnpmRoot = path.resolve(process.cwd(), 'node_modules', '.pnpm');
  if (fs.existsSync(pnpmRoot)) {
    const packageDir = fs
      .readdirSync(pnpmRoot)
      .find((entry) => entry.startsWith('@tobilu+qmd@'));
    if (packageDir) {
      return pathToFileURL(
        path.join(pnpmRoot, packageDir, 'node_modules', '@tobilu', 'qmd', 'dist', 'llm.js'),
      ).href;
    }
  }

  throw new Error('Unable to resolve @tobilu/qmd/dist/llm.js from node_modules');
}
