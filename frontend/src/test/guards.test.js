import { readdirSync, readFileSync } from 'node:fs';
import { join, resolve, sep } from 'node:path';

import { describe, expect, it } from 'vitest';

const SRC_DIR = resolve(process.cwd(), 'src');

function sourceFiles(directory) {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const fullPath = join(directory, entry.name);
    return entry.isDirectory() ? sourceFiles(fullPath) : [fullPath];
  });
}

const productionFiles = sourceFiles(SRC_DIR)
  .filter((file) => !file.includes(`${sep}test${sep}`))
  .map((file) => ({ file, text: readFileSync(file, 'utf8') }));

describe('frontend source hygiene', () => {
  it('never references backend secret names', () => {
    const offenders = productionFiles
      .filter(({ text }) =>
        /LLM_API_KEY|OPENAI_API_KEY|API_SECRET|sk-[A-Za-z0-9]/.test(text),
      )
      .map(({ file }) => file);

    expect(offenders).toEqual([]);
  });

  it('keeps the backend URL inside the API module', () => {
    const offenders = productionFiles
      .filter(
        ({ file, text }) =>
          !file.endsWith(`${sep}client.js`) &&
          /localhost:3000|127\.0\.0\.1:3000/.test(text),
      )
      .map(({ file }) => file);

    expect(offenders).toEqual([]);
  });

  it('has an env-configured API module', () => {
    const client = productionFiles.find(({ file }) =>
      file.endsWith(`${sep}api${sep}client.js`),
    );

    expect(client).toBeDefined();
    expect(client.text).toContain('VITE_API_BASE_URL');
  });
});
