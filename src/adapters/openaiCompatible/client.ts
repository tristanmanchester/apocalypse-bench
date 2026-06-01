import { createOpenAICompatible } from '@ai-sdk/openai-compatible';

export type OpenAICompatibleClientOptions = {
  baseUrl: string;
  apiKey?: string;
  headers?: Record<string, string>;
  queryParams?: Record<string, string>;
};

export function createOpenAICompatibleClient(opts: OpenAICompatibleClientOptions) {
  return createOpenAICompatible({
    name: 'openai-compatible',
    baseURL: opts.baseUrl,
    ...(opts.apiKey ? { apiKey: opts.apiKey } : {}),
    ...(opts.headers ? { headers: opts.headers } : {}),
    ...(opts.queryParams ? { queryParams: opts.queryParams } : {}),
    includeUsage: true,
  });
}
