import { expect, type APIRequestContext, type Locator } from "@playwright/test";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));

// The reference solutions the library's own tests use: what a learner would type.
export function solution(name: string): string {
  return readFileSync(resolve(here, "../../tests/solutions", name), "utf8");
}

// Edit a workspace file the way an editor would: through the workspace API.
export async function saveFile(request: APIRequestContext, path: string, content: string): Promise<void> {
  const current = await request.get(`/api/files/${path}`);
  const etag = current.ok() ? ((await current.json()) as { etag: string }).etag : undefined;
  const response = await request.put(`/api/files/${path}`, {
    data: { content },
    headers: etag ? { "If-Match": etag } : {},
  });
  expect(response.ok(), await response.text()).toBe(true);
}

// Counts the clicks a flow takes, so a test can say "at most N".
export class Clicks {
  n = 0;
  async on(target: Locator): Promise<void> {
    this.n += 1;
    await target.click();
  }
}
