import { expect, test } from "@playwright/test";
import { unlinkSync, writeFileSync, mkdirSync } from "node:fs";
import { git, initRepo, share, tidy, saveStudy, studyToml, WORKSPACE } from "./study";

const STAMP = Date.now().toString(36);
const NAME = `e2e-dirty-${STAMP}`;
const NOTE = `notes/ideas-${STAMP}.md`;

// Plan done-when: record mode is refused on a dirty tree, and the page lists the files.
test("record mode is refused on a dirty tree, with the list of files", async ({ page, request }) => {
  initRepo();
  tidy();
  await saveStudy(request, studyToml(NAME, "record"));
  git("add", `studies/${NAME}.toml`);
  git("commit", "-q", "-m", "spec");
  share();
  mkdirSync(`${WORKSPACE}/notes`, { recursive: true });
  writeFileSync(`${WORKSPACE}/${NOTE}`, "an idea\n");

  await page.goto(`/studies/${NAME}/edit`);
  const status = page.getByRole("region", { name: "Git status" });
  await expect(status).toContainText("1 uncommitted change");
  await expect(status).toContainText(NOTE);
  await expect(page.getByRole("button", { name: "Train 6 runs" })).toBeDisabled();

  // the server says the same, with the files, if asked anyway
  const refused = await request.post(`/api/studies/${NAME}/run`);
  expect(refused.status()).toBe(422);
  const problem = (await refused.json()) as { detail: string; changed: string[] };
  expect(problem.detail).toContain("clean git tree");
  expect(problem.changed).toEqual([NOTE]);

  // once the tree is clean the spec is committed and the page lets it train
  unlinkSync(`${WORKSPACE}/${NOTE}`);
  await expect(status).toContainText("clean", { timeout: 20_000 });
  await expect(page.getByRole("button", { name: "Train 6 runs" })).toBeEnabled();
});
