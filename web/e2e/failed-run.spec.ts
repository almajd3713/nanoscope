import { expect, test } from "@playwright/test";

// Plan done-when: a failed run shows its error from status.json. Lesson 2's starter raises
// NotImplementedError, so training it fails with a message the page must show word for word.
test("a failed run shows its error from status.json", async ({ page }) => {
  await page.goto("/learn/foundations/02-mlp");
  const start = page.getByRole("button", { name: "Start lesson" });
  if (await start.isVisible()) await start.click();
  await expect(page.getByRole("button", { name: "Train" })).toBeVisible(); // the model page, or the lesson page of a started lesson
  await page.getByRole("button", { name: "Train" }).click();
  await page.waitForURL(/\/runs\//);
  const error = page.getByRole("alert");
  await expect(error).toContainText("NotImplementedError", { timeout: 240_000 });
  await expect(error).toContainText("embed, window, hidden layer, project");
  await expect(page.getByText("failed", { exact: true }).first()).toBeVisible();
  await expect(page.getByRole("region", { name: "Traceback" })).toContainText("starter.py");
});
