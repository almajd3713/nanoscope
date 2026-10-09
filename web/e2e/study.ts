import { expect, type APIRequestContext } from "@playwright/test";
import { execFileSync } from "node:child_process";
import { chmodSync, existsSync } from "node:fs";
import { inflateRawSync } from "node:zlib";

// The workspace the stack bind-mounts (stack.sh); the studies' spec files and the repository live there.
export const WORKSPACE = process.env["E2E_WORKSPACE"] ?? process.env["NANOSCOPE_WORKSPACE"] ?? "/tmp/nanoscope-e2e-workspace";

export function git(...args: string[]): string {
  return execFileSync("git", ["-C", WORKSPACE, ...args], { encoding: "utf8" }).trim();
}

// A repository in the workspace with an identity. Host and container (another uid) both write into
// it, so git shares it world-writable: whoever creates a file or folder in .git, the other can use it.
export function initRepo(): void {
  if (!existsSync(`${WORKSPACE}/.git`)) git("init", "-q", "-b", "main");
  git("config", "user.email", "e2e@example.com");
  git("config", "user.name", "e2e");
  git("config", "commit.gpgsign", "false");
  git("config", "core.sharedRepository", "0666");
  share();
  chmodSync(WORKSPACE, 0o777);
}

// Let the container's user write what this one made (COMMIT_EDITMSG, the index, new folders).
export function share(): void {
  try {
    execFileSync("chmod", ["-R", "a+rwX", `${WORKSPACE}/.git`], { stdio: "ignore" });
  } catch {
    /* what the container made belongs to its uid; it was created shareable */
  }
}

// Commit whatever other specs left behind, so a record study starts from a clean tree.
export function tidy(): void {
  git("add", "-A");
  if (git("status", "--porcelain") !== "") git("commit", "-q", "-m", "e2e: tidy");
  share();
}

// Two tiny variants, three seeds: 10 steps a run, so a whole study takes a couple of minutes on a CPU.
export function studyToml(name: string, mode: "explore" | "record" = "explore"): string {
  return `name = "${name}"
preset = "tinystories-5min"
baseline = "small"
seeds = [0, 1, 2]
mode = "${mode}"
tolerance = 0.02

[budget]
tokens = 20480

[[variants]]
name = "small"
model = "nanoscope.models.bigram:Bigram"
[variants.kwargs]
d_model = 8

[[variants]]
name = "wide"
model = "nanoscope.models.bigram:Bigram"
[variants.kwargs]
d_model = 32
`;
}

export async function saveStudy(request: APIRequestContext, toml: string): Promise<void> {
  const saved = await request.post("/api/studies", { data: { toml, overwrite: true } });
  expect(saved.ok(), await saved.text()).toBe(true);
}

// Follow a queued job to its result.
export async function follow<T>(request: APIRequestContext, id: number, timeout = 120_000): Promise<T> {
  let job: { state: string; result: T | null; error: string | null } | null = null;
  await expect
    .poll(
      async () => {
        job = (await (await request.get(`/api/jobs/${id}`)).json()) as typeof job;
        return job?.state;
      },
      { timeout },
    )
    .toMatch(/done|failed|cancelled/);
  expect(job!.state, job!.error ?? "").toBe("done");
  return job!.result as T;
}

export async function waitStudyDone(request: APIRequestContext, name: string, runs: number): Promise<void> {
  await expect
    .poll(
      async () => {
        const all = (await (await request.get("/api/studies")).json()) as { name: string; runs_by_state: Record<string, number> }[];
        return all.find((s) => s.name === name)?.runs_by_state["done"] ?? 0;
      },
      { timeout: 420_000, intervals: [3000] },
    )
    .toBe(runs);
}

// The files of a zip, as text (the bundle): central directory, then each entry (stored or deflated).
export function readZip(buffer: Buffer): Record<string, string> {
  const end = buffer.lastIndexOf(Buffer.from([0x50, 0x4b, 0x05, 0x06]));
  const count = buffer.readUInt16LE(end + 10);
  let offset = buffer.readUInt32LE(end + 16);
  const files: Record<string, string> = {};
  for (let i = 0; i < count; i++) {
    const method = buffer.readUInt16LE(offset + 10);
    const size = buffer.readUInt32LE(offset + 20);
    const nameLength = buffer.readUInt16LE(offset + 28);
    const extra = buffer.readUInt16LE(offset + 30);
    const comment = buffer.readUInt16LE(offset + 32);
    const local = buffer.readUInt32LE(offset + 42);
    const name = buffer.subarray(offset + 46, offset + 46 + nameLength).toString("utf8");
    const dataStart = local + 30 + buffer.readUInt16LE(local + 26) + buffer.readUInt16LE(local + 28);
    const raw = buffer.subarray(dataStart, dataStart + size);
    files[name] = (method === 8 ? inflateRawSync(raw) : raw).toString("utf8");
    offset += 46 + nameLength + extra + comment;
  }
  return files;
}
