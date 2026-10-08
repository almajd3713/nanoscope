import { expect, test } from "@playwright/test";
import { Clicks, saveFile, solution } from "./helpers";

// Plan done-when: Foundations lesson 1 goes start -> train -> check passed in at most 6 clicks,
// counting the first-run question. Typing the solution is the learner's own work, not a click.
test("lesson 1: start, train, check passed in at most 6 clicks", async ({ page, request }) => {
  const clicks = new Clicks();

  await page.goto("/learn");
  await page.waitForURL(/\/welcome/); // nobody has chosen guided or open yet
  await clicks.on(page.getByRole("button", { name: "Continue to lessons" }));
  await clicks.on(page.getByRole("link", { name: /A bigram language model/ }));
  await clicks.on(page.getByRole("button", { name: "Start lesson" }));
  await expect(page.getByRole("button", { name: "Train" })).toBeVisible(); // the model page, or the lesson page of a started lesson

  await saveFile(request, "lessons/foundations/01-bigram/starter.py", solution("foundations/01-bigram.py"));

  await clicks.on(page.getByRole("button", { name: "Train" }));
  await page.waitForURL(/\/runs\//);
  await expect(page.getByText("done", { exact: true }).first()).toBeVisible({ timeout: 300_000 });

  await page.goBack(); // the browser's back button is a click too
  clicks.n += 1;
  await clicks.on(page.getByRole("button", { name: "Run the check" }));
  const result = page.getByRole("region", { name: "Check result" });
  await expect(result.getByText("passed", { exact: true }).first()).toBeVisible({ timeout: 300_000 });

  expect(clicks.n).toBeLessThanOrEqual(6);
});
