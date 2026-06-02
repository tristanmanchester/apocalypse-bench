import fs from 'node:fs';
import path from 'node:path';
import YAML from 'yaml';
import { z } from 'zod';
import { configSchema, type ApocbenchConfig } from './schema';

export type LoadConfigResult = {
  config: ApocbenchConfig;
  raw: unknown;
  absolutePath: string;
};

function parseConfigFile(contents: string, absolutePath: string): unknown {
  const ext = path.extname(absolutePath).toLowerCase();
  if (ext === '.json') return JSON.parse(contents);
  return YAML.parse(contents, { strict: true, merge: true });
}

export function loadConfig(configPath: string): LoadConfigResult {
  const absolutePath = path.resolve(process.cwd(), configPath);
  const contents = fs.readFileSync(absolutePath, 'utf8');
  const raw = parseConfigFile(contents, absolutePath);
  const parsed = configSchema.safeParse(raw);
  if (!parsed.success) {
    throw new z.ZodError(parsed.error.issues);
  }

  return { config: parsed.data, raw, absolutePath };
}
