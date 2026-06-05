import { pathToFileURL } from 'node:url';

import { main } from '../src/core/judge/codex';

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}
