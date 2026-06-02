import type { WikiConfig } from '../config/schema';
import type {
  WikiHealthResponse,
  WikiReadRequest,
  WikiReadResponse,
  WikiReadiness,
  WikiSearchMode,
  WikiSearchRequest,
  WikiSearchResponse,
} from './types';

type Fetch = typeof fetch;

export type WikiClientOptions = {
  baseUrl: string;
  timeoutMs?: number;
  fetch?: Fetch;
};

export class WikiClientError extends Error {
  readonly status?: number;

  constructor(message: string, opts: { status?: number; cause?: unknown } = {}) {
    super(message, opts.cause ? { cause: opts.cause } : undefined);
    this.name = 'WikiClientError';
    this.status = opts.status;
  }
}

export class WikiReadinessError extends WikiClientError {
  constructor(message: string) {
    super(message);
    this.name = 'WikiReadinessError';
  }
}

export class WikiClient {
  private readonly baseUrl: string;
  private readonly timeoutMs?: number;
  private readonly fetchImpl: Fetch;

  constructor(opts: WikiClientOptions) {
    this.baseUrl = opts.baseUrl.replace(/\/+$/, '');
    this.timeoutMs = opts.timeoutMs;
    this.fetchImpl = opts.fetch ?? fetch;
  }

  async health(): Promise<WikiHealthResponse> {
    return this.request<WikiHealthResponse>('/health', { method: 'GET' });
  }

  async search(request: WikiSearchRequest): Promise<WikiSearchResponse> {
    return this.postSearch('/search', request);
  }

  async semanticSearch(request: WikiSearchRequest): Promise<WikiSearchResponse> {
    return this.postSearch('/semantic_search', request);
  }

  async hybridSearch(request: WikiSearchRequest): Promise<WikiSearchResponse> {
    return this.postSearch('/hybrid_search', request);
  }

  async literalSearch(request: WikiSearchRequest): Promise<WikiSearchResponse> {
    return this.postSearch('/literal_search', request);
  }

  async read(request: WikiReadRequest): Promise<WikiReadResponse> {
    const raw = await this.post<WikiReadRequest, RawWikiReadResponse>('/read', request);
    return normalizeReadResponse(raw);
  }

  private async postSearch(
    path: string,
    request: WikiSearchRequest,
  ): Promise<WikiSearchResponse> {
    const raw = await this.post<{ query: string; limit?: number }, RawWikiSearchResponse>(path, {
      query: request.query,
      limit: request.topK,
    });
    return normalizeSearchResponse(raw);
  }

  private async post<TRequest, TResponse>(path: string, body: TRequest): Promise<TResponse> {
    return this.request<TResponse>(path, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    });
  }

  private async request<T>(path: string, init: RequestInit): Promise<T> {
    const controller = new AbortController();
    const timeoutId =
      this.timeoutMs != null
        ? setTimeout(() => controller.abort(new Error('wiki request timed out')), this.timeoutMs)
        : undefined;

    try {
      const response = await this.fetchImpl(`${this.baseUrl}${path}`, {
        ...init,
        signal: controller.signal,
      });

      if (!response.ok) {
        const message = await response.text().catch(() => response.statusText);
        throw new WikiClientError(
          `wiki service request failed: ${response.status} ${message || response.statusText}`,
          { status: response.status },
        );
      }

      return (await response.json()) as T;
    } catch (err) {
      if (err instanceof WikiClientError) throw err;
      throw new WikiClientError('wiki service request failed', { cause: err });
    } finally {
      if (timeoutId != null) clearTimeout(timeoutId);
    }
  }
}

type RawWikiSearchResponse = WikiSearchResponse | {
  mode: WikiSearchResponse['mode'];
  query: string;
  hits: Array<{
    mode: WikiSearchResponse['mode'];
    score?: number | null;
    article_id?: string;
    chunk_id?: string;
    title?: string;
    heading_path?: string[];
    url?: string;
    snippet?: string;
  }>;
  latencyMs?: number;
};

type RawWikiReadResponse = WikiReadResponse | {
  article_id?: string;
  chunk_id?: string;
  title?: string;
  heading_path?: string[];
  url?: string;
  text: string;
  truncated: boolean;
  latencyMs?: number;
};

function normalizeSearchResponse(raw: RawWikiSearchResponse): WikiSearchResponse {
  return {
    mode: raw.mode,
    query: raw.query,
    latencyMs: raw.latencyMs,
    hits: raw.hits.map((hit) => {
      if ('pointer' in hit) return hit;
      return {
        pointer: {
          articleId: hit.article_id ?? '',
          chunkId: hit.chunk_id,
          title: hit.title ?? '',
          url: hit.url,
          headingPath: hit.heading_path,
        },
        mode: hit.mode,
        score: hit.score ?? undefined,
        snippet: hit.snippet ?? '',
      };
    }),
  };
}

function normalizeReadResponse(raw: RawWikiReadResponse): WikiReadResponse {
  if ('pointer' in raw) return raw;
  return {
    pointer: {
      articleId: raw.article_id ?? '',
      chunkId: raw.chunk_id,
      title: raw.title ?? '',
      url: raw.url,
      headingPath: raw.heading_path,
    },
    text: raw.text,
    truncated: raw.truncated,
    latencyMs: raw.latencyMs,
  };
}

export function createWikiClientFromConfig(config: WikiConfig): WikiClient {
  return new WikiClient({
    baseUrl: config.service.baseUrl,
    timeoutMs: config.service.timeoutMs,
  });
}

export async function checkWikiReadiness(
  client: Pick<WikiClient, 'health'>,
  config: WikiConfig,
): Promise<WikiReadiness> {
  const health = await client.health();
  if (!health.ok) {
    throw new WikiReadinessError('wiki service is not healthy');
  }
  if (health.corpus.manifestId !== config.corpus.manifestId) {
    throw new WikiReadinessError(
      `wiki corpus manifest mismatch: expected ${config.corpus.manifestId}, got ${health.corpus.manifestId}`,
    );
  }
  if (health.index.manifestId !== config.index.manifestId) {
    throw new WikiReadinessError(
      `wiki index manifest mismatch: expected ${config.index.manifestId}, got ${health.index.manifestId}`,
    );
  }

  return {
    health,
    capabilities: new Set<WikiSearchMode>(health.capabilities),
  };
}
