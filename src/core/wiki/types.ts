export type WikiSearchMode = 'bm25' | 'dense' | 'hybrid' | 'literal';

export type WikiSourcePointer = {
  articleId: string;
  chunkId?: string;
  url?: string;
  title: string;
  headingPath?: string[];
};

export type WikiSearchHit = {
  pointer: WikiSourcePointer;
  mode: WikiSearchMode;
  score?: number;
  bm25Score?: number;
  denseScore?: number;
  snippet: string;
};

export type WikiSearchRequest = {
  query: string;
  topK?: number;
};

export type WikiSearchResponse = {
  mode: WikiSearchMode;
  query: string;
  hits: WikiSearchHit[];
  latencyMs?: number;
};

export type WikiReadRequest = {
  articleId?: string;
  chunkId?: string;
  maxChars?: number;
};

export type WikiReadResponse = {
  pointer: WikiSourcePointer;
  text: string;
  truncated: boolean;
  latencyMs?: number;
};

export type WikiIndexHealth = {
  manifestId: string;
  status: 'ready' | 'unavailable';
};

export type WikiHealthResponse = {
  ok: boolean;
  corpus: {
    manifestId: string;
  };
  index: {
    manifestId: string;
  };
  capabilities: WikiSearchMode[];
  indexes?: {
    bm25?: WikiIndexHealth;
    dense?: WikiIndexHealth;
  };
};

export type WikiReadiness = {
  health: WikiHealthResponse;
  capabilities: Set<WikiSearchMode>;
};
