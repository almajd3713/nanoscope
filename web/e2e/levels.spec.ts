import { expect, test } from "@playwright/test";

// Plan done-when: Learn and Tinker submit identical POST /api/runs bodies when nothing was
// touched. The request is intercepted, so nothing trains.
test("Learn and Tinker send the same run request when nothing was touched", async ({ page }) => {
  const bodies: unknown[] = [];
  await page.route("**/api/runs", async (route) => {
    if (route.request().method() !== "POST") return route.fallback();
    bodies.push(route.request().postDataJSON());
    await route.fulfill({
      status: 202,
      contentType: "application/json",
      body: JSON.stringify({ ref: "tinystories-5min/mlp/seed-0", state: "queued" }),
    });
  });
  const setLevel = (level: string) => page.evaluate((l) => localStorage.setItem("nanoscope.level", l), level);

  // Learn: the lesson's Train button, for lesson 2's model
  await page.goto("/learn/foundations/02-mlp");
  await setLevel("Learn");
  await page.reload();
  const start = page.getByRole("button", { name: "Start lesson" });
  if (await start.isVisible()) await start.click();
  await expect(page.getByRole("button", { name: "Train" })).toBeVisible(); // the model page, or the lesson page of a started lesson
  await page.getByRole("button", { name: "Train" }).click();
  await page.waitForURL(/\/runs\//);

  // Tinker: the run form for the same model, nothing edited
  await setLevel("Tinker");
  await page.goto("/runs/new");
  await page.getByRole("combobox", { name: "Model" }).selectOption({ label: "lessons/foundations/02-mlp/starter.py:MyMLP" });
  await page.getByRole("button", { name: "Train" }).click();
  await page.waitForURL(/\/runs\//);

  expect(bodies).toHaveLength(2);
  expect(bodies[1]).toEqual(bodies[0]);
});
