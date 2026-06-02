import { afterEach, describe, expect, test, vi } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';

import { migrate, schemaPathForModule } from './migrate';
import type { DbHandle } from './db';

describe('sqlite migration path handling', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  test('builds schema path from a Windows drive-letter module path', () => {
    const schemaPath = schemaPathForModule(
      'D:\\repo\\dist\\storage\\sqlite\\migrate.js',
      path.win32,
    );

    expect(schemaPath).toBe('D:\\repo\\dist\\storage\\sqlite\\schema.sql');
    expect(schemaPath).not.toMatch(/^\/[A-Za-z]:/);
  });

  test('preserves spaces in Windows module paths', () => {
    const schemaPath = schemaPathForModule(
      'D:\\repo with spaces\\dist\\storage\\sqlite\\migrate.js',
      path.win32,
    );

    expect(schemaPath).toBe('D:\\repo with spaces\\dist\\storage\\sqlite\\schema.sql');
  });

  test('builds schema path from a POSIX module path', () => {
    const schemaPath = schemaPathForModule(
      '/repo/dist/storage/sqlite/migrate.js',
      path.posix,
    );

    expect(schemaPath).toBe('/repo/dist/storage/sqlite/schema.sql');
  });

  test('migrate reads the schema adjacent to the runtime module path', () => {
    const exec = vi.fn();
    const prepare = vi.fn(() => ({ all: () => [{ name: 'retrieval_trace_json' }] }));
    const db = { exec, prepare } as unknown as DbHandle;
    const readFileSync = vi.spyOn(fs, 'readFileSync').mockReturnValue('create table x(id);');

    migrate(db);

    expect(readFileSync).toHaveBeenCalledTimes(1);
    const [schemaPath, encoding] = readFileSync.mock.calls[0]!;
    expect(String(schemaPath).replaceAll(path.sep, '/')).toMatch(
      /src\/storage\/sqlite\/schema\.sql$/,
    );
    expect(encoding).toBe('utf8');
    expect(exec).toHaveBeenCalledWith('create table x(id);');
    expect(prepare).toHaveBeenCalledWith('PRAGMA table_info(model_results)');
  });
});
