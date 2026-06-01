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
}

export function openAndMigrate(sharedDbPath: string): DbHandle {
  const db = openDb(sharedDbPath);
  migrate(db);
  return db;
}
