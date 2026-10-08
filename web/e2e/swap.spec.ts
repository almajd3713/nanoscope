import { expect, test, type APIRequestContext } from "@playwright/test";
import { saveFile } from "./helpers";

const FILE = "lessons/modern-block/07-assemble/starter.py";

async function readFile(request: APIRequestContext): Promise<string> {
  return ((await (await request.get(`/api/files/${FILE}`)).json()) as { content: string }).content;
}

// The learner has built RMSNorm and imports it; the file otherwise is the lesson's starter.
async function prepare(request: APIRequestContext): Promise<string> {
  await request.post("/api/learn/policy", { data: { policy: "open" } });
  const started = await request.post("/api/curricula/modern-block/07-assemble/start", { data: {} });
  expect(started.ok(), await started.text()).toBe(true);
  const starter = await readFile(request);
  const ready = starter.includes("RMSNorm,") || starter.includes("RMSNorm\n")
    ? starter
    : starter.replace("GELUMLP, LayerNorm,", "GELUMLP, LayerNorm, RMSNorm,");
  await saveFile(request, FILE, ready);
  return ready;
}

// The numbers `git diff --numstat` would print for the file: lines added and lines removed.
function numstat(before: string, after: string): [number, number] {
  const a = before.split("\n");
  const b = after.split("\n");
  let removed = 0;
  let added = 0;
  const count = new Map<string, number>();
  for (const line of a) count.set(line, (count.get(line) ?? 0) + 1);
  for (const line of b) {
    const n = count.get(line) ?? 0;
    if (n > 0) count.set(line, n - 1);
    else added += 1;
  }
  for (const n of count.values()) removed += n;
  return [added, removed];
}

// Plan done-when: swapping LayerNorm for RMSNorm in the inspector changes exactly one line
// (`git diff --numstat` shows 1/1), and doing it by drag and drop gives the identical file.
test("swapping a norm in the inspector changes one line, and drag and drop gives the same file", async ({ page, request }) => {
  const before = await prepare(request);
  await page.goto(`/model/${FILE}`);
  await page.getByLabel("LayerNorm in norm", { exact: true }).click();
  await page.getByLabel("Swap for").selectOption("RMSNorm");
  await expect.poll(() => readFile(request)).toContain("norm=RMSNorm(),");
  const viaInspector = await readFile(request);
  expect(numstat(before, viaInspector)).toEqual([1, 1]);
  expect(viaInspector.replace("norm=RMSNorm(),", "norm=LayerNorm(),")).toBe(before); // nothing else moved

  // the same swap by drag and drop, from the original file
  await saveFile(request, FILE, before);
  await page.reload();
  await expect(page.getByLabel("LayerNorm in norm", { exact: true })).toBeVisible();
  await page.getByRole("navigation", { name: "Block palette" }).getByText("RMSNorm", { exact: true }).dragTo(page.getByLabel("LayerNorm in norm", { exact: true }));
  await expect.poll(() => readFile(request)).toContain("norm=RMSNorm(),");
  expect(await readFile(request)).toBe(viaInspector);
});
