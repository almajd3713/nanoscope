// Seeding and browser state shared by readme-shots.ts and demo-video.ts. They talk to a running
// `nanoscope serve` on loopback (no token) at $SHOTS_URL, default http://127.0.0.1:18780.
import { request, type APIRequestContext, type APIResponse, type BrowserContext, type Page } from "@playwright/test";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export const BASE_URL: string = process.env["SHOTS_URL"] ?? "http://127.0.0.1:18780";
export const REPO: string = resolve(dirname(fileURLToPath(import.meta.url)), "../..");

export const LESSON_04 = "foundations/04-multi-head";
export const LESSON_07 = "modern-block/07-assemble";
export const FILE_04 = "lessons/foundations/04-multi-head/starter.py";
export const FILE_07 = "lessons/modern-block/07-assemble/starter.py";
export const COMPARE_PATH =
  "/compare?runs=baselines/tinystories-5min/modern,baselines/tinystories-5min/bigram,baselines/tinystories-5min/gpt2";
export const STUDY_PATH = "/studies/m1-ablation";

export type Level = "Learn" | "Tinker" | "Research" | "Extend";

async function ensureOk(response: APIResponse, method: string, path: string): Promise<void> {
  if (!response.ok()) throw new Error(`${method} ${path}: ${response.status()} ${await response.text()}`);
}

export async function connect(): Promise<APIRequestContext> {
  const api = await request.newContext({ baseURL: BASE_URL });
  const unreachable = new Error(`cannot reach ${BASE_URL}: start nanoscope serve first (see the comment at the top of this file)`);
  try {
    const response = await api.get("/api/presets");
    if (!response.ok()) throw unreachable;
  } catch {
    await api.dispose();
    throw unreachable;
  }
  return api;
}

export async function setPolicy(api: APIRequestContext, policy: "open" | "guided"): Promise<void> {
  const path = "/api/learn/policy";
  await ensureOk(await api.post(path, { data: { policy } }), "POST", path);
}

export async function startLesson(api: APIRequestContext, lesson: string): Promise<void> {
  const path = `/api/curricula/${lesson}/start`;
  await ensureOk(await api.post(path, { data: {} }), "POST", path);
}

export async function readFile(api: APIRequestContext, path: string): Promise<string> {
  const url = `/api/files/${path}`;
  const response = await api.get(url);
  await ensureOk(response, "GET", url);
  return ((await response.json()) as { content: string }).content;
}

export async function saveFile(api: APIRequestContext, path: string, content: string): Promise<void> {
  const url = `/api/files/${path}`;
  const current = await api.get(url);
  const etag = current.ok() ? ((await current.json()) as { etag: string }).etag : undefined;
  const response = await api.put(url, { data: { content }, headers: etag ? { "If-Match": etag } : {} });
  await ensureOk(response, "PUT", url);
}

export function shipped(lesson: string): string {
  return readFileSync(resolve(REPO, "nanoscope", "curricula", lesson, "starter.py"), "utf8");
}

export function solution(lesson: string): string {
  return readFileSync(resolve(REPO, "tests", "solutions", `${lesson}.py`), "utf8");
}

export function withRmsNormImport(starter: string): string {
  const imports = "GELUMLP, LayerNorm,";
  if (!starter.includes(imports)) {
    throw new Error('the lesson 07 starter no longer imports "GELUMLP, LayerNorm,"; update withRmsNormImport');
  }
  return starter.replace(imports, "GELUMLP, LayerNorm, RMSNorm,");
}

export async function prepareLesson04(api: APIRequestContext): Promise<void> {
  await startLesson(api, LESSON_04);
  await saveFile(api, FILE_04, shipped(LESSON_04));
}

export async function prepareLesson07(api: APIRequestContext): Promise<void> {
  await startLesson(api, LESSON_07);
  await saveFile(api, FILE_07, withRmsNormImport(shipped(LESSON_07)));
}

export async function runCheck(api: APIRequestContext, lesson: string): Promise<boolean> {
  const queued = `/api/curricula/${lesson}/check`;
  const posted = await api.post(queued, { data: { variant: "cpu" } });
  await ensureOk(posted, "POST", queued);
  const { id } = (await posted.json()) as { id: number };
  const deadline = Date.now() + 120_000;
  while (Date.now() < deadline) {
    const path = `/api/jobs/${id}`;
    const response = await api.get(path);
    await ensureOk(response, "GET", path);
    const job = (await response.json()) as { state: string; result: { passed: boolean } | null; error: string | null };
    if (job.state === "done") return Boolean(job.result?.passed);
    if (job.state === "failed") throw new Error(`check job ${id} for ${lesson} failed: ${job.error}`);
    await new Promise((done) => setTimeout(done, 500));
  }
  throw new Error(`check job ${id} for ${lesson} did not finish in 120 s`);
}

export async function doneRuns(api: APIRequestContext, prefix: string): Promise<number> {
  const response = await api.get("/api/runs", { params: { prefix, state: "done" } });
  await ensureOk(response, "GET", "/api/runs");
  return ((await response.json()) as unknown[]).length;
}

export async function setLevel(context: BrowserContext, level: Level): Promise<void> {
  await context.addInitScript((value: string) => {
    try {
      window.localStorage.setItem("nanoscope.level", value);
      window.localStorage.setItem("nanoscope.unlockOffered", "1");
    } catch {
      // storage blocked: the level stays the default
    }
  }, level);
}

export async function settle(page: Page): Promise<void> {
  await page.evaluate(async () => {
    await document.fonts.ready;
  });
  await page.waitForTimeout(500);
}
