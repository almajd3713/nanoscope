// Writes the README screenshots, docs/images/<screen>-<light|dark>.png, from real data.
// 1. Train the M1 study into a demo home (docs/dev/plans/2026-10-30-presentable-show-it.md, Step 2.1).
// 2. make web, then: NANOSCOPE_HOME=$HOME/nanoscope-demo/home NANOSCOPE_WORKSPACE=$HOME/nanoscope-demo/ws
//    NANOSCOPE_DATA_DIR=$HOME/.nanoscope/data uv run nanoscope serve --port 18780 --worker cpu
// 3. pnpm -C web exec tsx scripts/readme-shots.ts
import { chromium, expect } from "@playwright/test";
import { mkdirSync } from "node:fs";
import { join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  BASE_URL,
  COMPARE_PATH,
  FILE_07,
  LESSON_04,
  LESSON_07,
  REPO,
  STUDY_PATH,
  connect,
  doneRuns,
  prepareLesson04,
  prepareLesson07,
  readFile,
  runCheck,
  saveFile,
  setPolicy,
  settle,
  shipped,
  setLevel,
  withRmsNormImport,
  type Level,
} from "./demo-api";

const viewport = { width: 1280, height: 800 };
const themes = ["light", "dark"] as const;
const shots: { shot: string; level: Level }[] = [
  { shot: "lesson-check", level: "Learn" },
  { shot: "model-swap", level: "Tinker" },
  { shot: "compare", level: "Research" },
  { shot: "study", level: "Research" },
];
const timeout = 30_000;

async function main(): Promise<void> {
  const out = join(REPO, "docs", "images");
  const api = await connect();
  try {
    await setPolicy(api, "open");
    const n = await doneRuns(api, "studies/m1-ablation/");
    if (n !== 24) throw new Error(`the m1-ablation study has ${n} of 24 runs done; finish Step 2.1 first`);
    await prepareLesson04(api);
    await prepareLesson07(api);
    if (await runCheck(api, LESSON_04)) {
      throw new Error("lesson 04's starter passed its check; the lesson-check screenshot needs a failing check");
    }
    mkdirSync(out, { recursive: true });
    const browser = await chromium.launch();
    try {
      for (const theme of themes) {
        for (const { shot, level } of shots) {
          const context = await browser.newContext({ baseURL: BASE_URL, viewport, deviceScaleFactor: 2, colorScheme: theme });
          await setLevel(context, level);
          const page = await context.newPage();
          if (shot === "lesson-check") {
            await page.goto(`/learn/${LESSON_04}`);
            const reason = page.getByText("has no parameter named 'q.weight'").first();
            await expect(reason).toBeVisible({ timeout });
            await reason.scrollIntoViewIfNeeded();
          } else if (shot === "model-swap") {
            await saveFile(api, FILE_07, withRmsNormImport(shipped(LESSON_07)));
            await page.goto(`/model/${FILE_07}`);
            const layer = page.getByLabel("LayerNorm in norm", { exact: true });
            await expect(layer).toBeVisible({ timeout });
            await page.getByRole("navigation", { name: "Block palette" }).getByText("RMSNorm", { exact: true }).dragTo(layer);
            await expect.poll(() => readFile(api, FILE_07), { timeout }).toContain("norm=RMSNorm(),");
            const swapped = page.getByLabel("RMSNorm in norm", { exact: true });
            await expect(swapped).toBeVisible({ timeout });
            await swapped.click();
            await expect(page.locator(".monaco-editor").first()).toBeVisible({ timeout });
          } else if (shot === "compare") {
            await page.goto(COMPARE_PATH);
            await expect(page.getByRole("img", { name: /95% CI/ })).toBeVisible({ timeout });
          } else {
            await page.goto(STUDY_PATH);
            await expect(page.getByRole("heading", { name: "Variants × seeds" })).toBeVisible({ timeout });
          }
          await settle(page);
          const path = join(out, `${shot}-${theme}.png`);
          await page.screenshot({ path });
          console.log(`wrote ${relative(REPO, path)}`);
          await context.close();
        }
      }
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
