import { expect, test } from "@playwright/test";
import { git, initRepo, readZip, tidy, saveStudy, studyToml, waitStudyDone } from "./study";

const NAME = `e2e-prereg-${Date.now().toString(36)}`;

// The dialog commits the spec in a worker; the hash it shows is the commit, and every run of the
// study records it.
test("the preregistration dialog commits, and the hash appears in study.json", async ({ page, request }) => {
  initRepo();
  tidy();
  await saveStudy(request, studyToml(NAME, "record"));

  await page.goto(`/studies/${NAME}/edit`);
  await expect(page.getByRole("region", { name: "Git status" })).toContainText("uncommitted");
  await expect(page.getByRole("button", { name: "Train 6 runs" })).toBeDisabled();
  await page.getByRole("button", { name: "Commit preregistration…" }).click();

  const dialog = page.getByRole("alertdialog");
  // the preview is a worker job: it waits for a free slot behind any run in progress
  await expect(dialog.getByText("exactly what the commit contains")).toBeVisible({ timeout: 120_000 });
  await expect(dialog.getByLabel("Diff")).toContainText(`name = "${NAME}"`);
  await expect(dialog.getByText(`Preregister study ${NAME}`)).toBeVisible();
  await dialog.getByRole("button", { name: "Commit preregistration" }).click();

  const done = dialog.getByRole("status");
  await expect(done).toContainText("Preregistration committed", { timeout: 120_000 });
  const hash = (await done.locator("code").innerText()).trim();
  expect(hash).toMatch(/^[0-9a-f]{40}$/);
  expect(git("rev-parse", "HEAD")).toBe(hash);
  expect(git("log", "-1", "--format=%B")).toContain("Committed-via: nanoscope");
  await dialog.getByRole("button", { name: "Close" }).click();

  // now the tree is clean and the spec committed: record may train
  const train = page.getByRole("button", { name: "Train 6 runs" });
  await expect(train).toBeEnabled({ timeout: 20_000 });
  await train.click();
  await expect(page).toHaveURL(new RegExp(`/studies/${NAME}$`));
  await waitStudyDone(request, NAME, 6);

  const bundle = await request.get(`/api/studies/${NAME}/bundle.zip`);
  expect(bundle.ok()).toBe(true);
  const manifest = JSON.parse(readZip(await bundle.body())["study.json"]!) as Record<string, string>;
  expect(manifest["preregistration_commit"]).toBe(hash);
  expect(manifest["committed_via"]).toBe("nanoscope");
});
