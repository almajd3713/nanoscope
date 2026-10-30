// Records a short demo, $DEMO_OUT/nanoscope-demo.webm (default ~/nanoscope-demo/out): lesson 04's
// check fails, the fix passes and unlocks Attention, a block swap on the model page, then compare.
// Needs the server of readme-shots.ts. Convert: ffmpeg -y -i nanoscope-demo.webm -c:v libx264
// -pix_fmt yuv420p -crf 20 -movflags +faststart nanoscope-demo.mp4
import { chromium, expect } from "@playwright/test";
import { mkdirSync, rmSync } from "node:fs";
import { homedir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  BASE_URL,
  COMPARE_PATH,
  FILE_04,
  FILE_07,
  LESSON_04,
  connect,
  prepareLesson04,
  prepareLesson07,
  readFile,
  saveFile,
  setLevel,
  setPolicy,
  settle,
  solution,
} from "./demo-api";

const DEMO_OUT = process.env["DEMO_OUT"] ?? join(homedir(), "nanoscope-demo", "out");
const raw = join(DEMO_OUT, "raw");
const output = join(DEMO_OUT, "nanoscope-demo.webm");
const viewport = { width: 1280, height: 800 };
const timeout = 60_000;

async function main(): Promise<void> {
  const api = await connect();
  try {
    await setPolicy(api, "open");
    await prepareLesson07(api);
    await prepareLesson04(api);
    await setPolicy(api, "guided");
    mkdirSync(raw, { recursive: true });
    const browser = await chromium.launch();
    try {
      const context = await browser.newContext({
        baseURL: BASE_URL,
        viewport,
        deviceScaleFactor: 1,
        colorScheme: "light",
        recordVideo: { dir: raw, size: viewport },
      });
      await setLevel(context, "Tinker");
      const page = await context.newPage();

      await page.goto(`/model/${FILE_04}`);
      const check = page.getByRole("button", { name: "Run the check" }).first();
      await expect(check).toBeVisible({ timeout });
      await expect(page.locator(".monaco-editor").first()).toBeVisible({ timeout });
      await settle(page);
      await page.waitForTimeout(3000);

      await check.click();
      await expect(page.getByText("has no parameter named 'q.weight'").first()).toBeVisible({ timeout });
      await page.waitForTimeout(5000);

      await saveFile(api, FILE_04, solution(LESSON_04));
      await expect(page.locator(".monaco-editor .view-lines").first()).toContainText("nn.Linear", { timeout });
      await page.waitForTimeout(4000);

      await check.click();
      await expect(page.getByText("Check passed. Attention is unlocked.")).toBeVisible({ timeout });
      await page.waitForTimeout(5000);

      await setPolicy(api, "open");
      await page.goto(`/model/${FILE_07}`);
      const layer = page.getByLabel("LayerNorm in norm", { exact: true });
      await expect(layer).toBeVisible({ timeout });
      await settle(page);
      await page.waitForTimeout(3000);
      await page.getByRole("navigation", { name: "Block palette" }).getByText("RMSNorm", { exact: true }).dragTo(layer);
      await expect.poll(() => readFile(api, FILE_07), { timeout: 30_000 }).toContain("norm=RMSNorm(),");
      await expect(page.getByLabel("RMSNorm in norm", { exact: true })).toBeVisible({ timeout });
      await page.waitForTimeout(5000);

      await page.evaluate(() => {
        try {
          window.localStorage.setItem("nanoscope.level", "Research");
        } catch {
          /* storage blocked: the forest plot still shows */
        }
      });
      await page.goto(COMPARE_PATH);
      await expect(page.getByRole("img", { name: /95% CI/ })).toBeVisible({ timeout });
      await settle(page);
      await page.waitForTimeout(9000);

      const video = page.video();
      await context.close();
      await video!.saveAs(output);
      rmSync(raw, { recursive: true, force: true });
      console.log(`wrote ${output}`);
    } finally {
      await browser.close();
    }
  } finally {
    await api.dispose();
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error: unknown) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
