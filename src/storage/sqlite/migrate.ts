import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { openDb, type DbHandle } from './db';

type PathApi = Pick<typeof path, 'dirname' | 'join'>;

export function schemaPathForModule(modulePath: string, pathApi: PathApi = path): string {
  return pathApi.join(pathApi.dirname(modulePath), 'schema.sql');
}

export function migrate(db: DbHandle): void {
  const schemaPath = schemaPathForModule(fileURLToPath(import.meta.url));
  const sql = fs.readFileSync(schemaPath, 'utf8');
  db.exec(sql);
  ensureColumn(db, 'model_results', 'retrieval_trace_json', 'TEXT');
}

export function openAndMigrate(sharedDbPath: string): DbHandle {
  const db = openDb(sharedDbPath);
  migrate(db);
  return db;
}

function ensureColumn(db: DbHandle, table: string, column: string, type: string): void {
  const rows = db.prepare(`PRAGMA table_info(${table})`).all() as Array<{ name: string }>;
  if (rows.some((row) => row.name === column)) return;
  db.exec(`ALTER TABLE ${table} ADD COLUMN ${column} ${type}`);
}
