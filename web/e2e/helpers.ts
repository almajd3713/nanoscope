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

// Run a lesson's check and wait for its result, the way the Run the check button does.
export async function checkLesson(request: APIRequestContext, lesson: string): Promise<{ passed: boolean }> {
  const queued = await request.post(`/api/curricula/${lesson}/check`, { data: { variant: "cpu" } });
  expect(queued.ok(), await queued.text()).toBe(true);
  const { id } = (await queued.json()) as { id: number };
  let job: { state: string; result: { passed: boolean } | null; error: string | null } | null = null;
  await expect
    .poll(
      async () => {
        job = (await (await request.get(`/api/jobs/${id}`)).json()) as typeof job;
        return job?.state;
      },
      { timeout: 120_000 },
    )
    .toMatch(/done|failed/);
  expect(job!.state, job!.error ?? "").toBe("done");
  return { passed: job!.result!.passed };
}

// A workspace file's text.
export async function readFile(request: APIRequestContext, path: string): Promise<string> {
  return ((await (await request.get(`/api/files/${path}`)).json()) as { content: string }).content;
}
