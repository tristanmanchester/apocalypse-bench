---
title: "fix: Preserve Windows SQLite schema paths"
type: fix
status: completed
date: 2026-06-01
---

# fix: Preserve Windows SQLite schema paths

## Summary

Fix the Windows startup failure reported in issue #3 by replacing URL pathname-based module path handling with platform-correct file URL conversion for the bundled SQLite schema. Keep the change narrow: schema discovery should work on Windows, POSIX, paths with spaces, and UNC-style paths without changing user-facing config path semantics.

---

## Problem Frame

Issue #3 reports that a path is being rewritten into a doubled drive-letter form like `/d:/d:/path` on Windows. The concrete local culprit is `src/storage/sqlite/migrate.ts`, which derives `schema.sql` with `new URL(import.meta.url).pathname`. Node's URL documentation explicitly calls `.pathname` incorrect for Windows file URLs and recommends `fileURLToPath()` for platform-specific filesystem paths.

The rest of the repo generally uses `path.resolve()` for user config paths, which is less likely to create the reported drive-letter duplication. This plan therefore targets the module-relative schema path first and asks implementation to audit nearby file URL conversion, not to redesign all path handling.

---

## Requirements

**Windows path correctness**

- R1. SQLite migration must locate `schema.sql` on Windows when the built module is loaded from a drive-letter path such as `D:\...\dist\storage\sqlite\migrate.js`.
- R2. SQLite migration must not construct intermediate paths that begin with URL pathname artifacts such as `/D:/...` or `/d:/d:/...`.
- R3. Schema discovery must preserve paths containing spaces or percent-encoded characters after file URL conversion.

**Compatibility**

- R4. POSIX schema discovery must continue to locate the same `schema.sql` path as before.
- R5. User-supplied config paths such as `run.outDir`, `run.datasetPath`, and `run.datasetPaths` must keep their current behavior unless a direct audit finds the same URL-pathname bug.

**Verification**

- R6. The regression must be covered by a unit-level test that can run on non-Windows CI while still validating Windows path composition through path-library injection or equivalent pure helper coverage.

---

## Key Technical Decisions

- KTD1. Use `fileURLToPath(import.meta.url)` at the module boundary: this is the Node-supported way to turn an ES module file URL into a filesystem path, including Windows drive letters, UNC paths, spaces, and percent-decoding.
- KTD2. Extract a small schema path helper for testability: `migrate()` can stay simple while a helper accepts an already-converted module file path and a path implementation, letting tests exercise `path.win32` behavior on any host.
- KTD3. Keep config path resolution out of scope unless the audit finds URL-derived paths there: config paths already flow through `path.resolve(process.cwd(), value)` in `src/core/config/loadConfig.ts`, `src/core/dataset/loadJsonl.ts`, and `src/storage/sqlite/db.ts`; broad normalization changes would risk changing valid user workflows.

---

## Implementation Units

### U1. Replace SQLite schema URL pathname conversion

- **Goal:** Make `migrate()` derive `schema.sql` from a real filesystem module path instead of a URL pathname.
- **Requirements:** R1, R2, R3, R4
- **Dependencies:** None
- **Files:**
  - `src/storage/sqlite/migrate.ts`
  - `src/storage/sqlite/migrate.test.ts`
- **Approach:** Import `fileURLToPath` from `node:url`, convert `import.meta.url` once, and join `schema.sql` from the module directory. Factor the path composition into a small exported or locally testable helper that accepts the converted module file path; this keeps Windows expectations testable without requiring the test process to run on Windows.
- **Patterns to follow:** `src/cli/index.tsx` already uses `fileURLToPath(import.meta.url)` in `readToolVersion()` before walking parent directories.
- **Test scenarios:**
  - Given a Windows module path like `D:\repo\dist\storage\sqlite\migrate.js`, when the helper builds the schema path with `path.win32`, then the result is `D:\repo\dist\storage\sqlite\schema.sql` and never starts with `/D:/`.
  - Given a Windows module path under a directory with spaces, when the helper builds the schema path with `path.win32`, then the spaces remain literal filesystem characters.
  - Given a POSIX module path like `/repo/dist/storage/sqlite/migrate.js`, when the helper builds the schema path with `path.posix`, then the result is `/repo/dist/storage/sqlite/schema.sql`.
  - Given the real test runtime module URL, when `migrate()` is invoked against a mocked database and mocked schema read, then it reads the schema path adjacent to `migrate.ts` or its built equivalent rather than from a URL pathname.
- **Verification:** A reviewer can see `migrate.ts` no longer uses `.pathname`, and the new tests prove both Windows and POSIX composition.

### U2. Audit adjacent filesystem path derivation

- **Goal:** Confirm the issue's path symptom is not repeated in other runtime path derivation that would affect Windows runs.
- **Requirements:** R5
- **Dependencies:** U1
- **Files:**
  - `src/core/config/loadConfig.ts`
  - `src/core/dataset/loadJsonl.ts`
  - `src/storage/sqlite/db.ts`
  - `src/cli/index.tsx`
- **Approach:** Search for `new URL(...).pathname`, `.pathname` on file URLs, and any hand-rolled drive-letter manipulation. Leave path.resolve-based user config handling unchanged unless a direct URL-to-path bug is found. If another file URL path exists, apply the same `fileURLToPath()` pattern and add a focused regression test near the affected module.
- **Patterns to follow:** Keep file-path behavior local to the module that owns the path; do not introduce a global path normalization layer for a one-site bug.
- **Test scenarios:**
  - Test expectation: none unless the audit finds another executable path conversion bug. This unit is an audit guard to prevent an incomplete fix, not a behavioral change by itself.
- **Verification:** The implementation notes or PR description identify the searched pattern and either list no additional findings or point to the additional file-specific test added.

---

## Scope Boundaries

- Do not change how users specify `run.outDir`, `run.datasetPath`, or `run.datasetPaths` in config.
- Do not add Windows-specific path string rewrites; rely on Node and `node:path` platform behavior.
- Do not refactor SQLite migration or database lifecycle beyond the schema path fix.

### Deferred to Follow-Up Work

- Add a broader Windows CLI smoke test once the project has a Windows CI job or an established cross-platform smoke harness.

---

## Risks & Dependencies

- The main risk is overfitting the test to the helper instead of the runtime bug. Keep the helper small and make the runtime path visibly pass through `fileURLToPath(import.meta.url)`.
- If the published build changes the relative location of `schema.sql`, this fix must preserve the existing assumption that the schema file is bundled next to the compiled migration module.

---

## Sources & Research

- GitHub issue #3: `doesn't work on windows`, reporting `/d:/d:/path`.
- Node.js URL documentation: `fileURLToPath()` returns platform-specific filesystem paths and documents `.pathname` as incorrect for Windows drive paths and UNC paths.
- Local code: `src/storage/sqlite/migrate.ts` currently uses `new URL(import.meta.url).pathname`; `src/cli/index.tsx` already uses `fileURLToPath(import.meta.url)` successfully.
