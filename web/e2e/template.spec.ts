import { expect, test } from "@playwright/test";
import { readFile } from "./helpers";

const FILE = "lessons/foundations/03-attention-head/starter.py";

// The graph route of lesson F03: the same OneHead, built by filling the eight slots of the
// attention template instead of writing forward(). The lesson page starts it from the template.
const FILLS: [slot: string, block: string][] = [
  ["q", "Linear"], ["k", "Linear"], ["v", "Linear"], ["scores", "ScaledDotScores"],
  ["mask", "CausalMask"], ["normalize", "Softmax"], ["mix", "WeightedSum"], ["out", "Linear"],
];

// The whole palette and the template on screen at once: a drag from a list that is still
// scrolling lands on the neighbouring item.
test.use({ viewport: { width: 1280, height: 1500 } });

// Plan done-when: lesson F03 completed through the template canvas, and its `equivalent` check
// passes against the naive reference.
test("lesson F03 through the template canvas passes its equivalent check", async ({ page, request }) => {
  await request.post("/api/learn/policy", { data: { policy: "open" } });
  await page.goto("/learn/foundations/03-attention-head");
  await page.getByRole("button", { name: "Build it from blocks" }).click();
  await page.waitForURL(`**/model/${FILE}`);
  await expect(page.getByRole("region", { name: "AttentionTemplate template" })).toBeVisible();
  const palette = page.getByRole("navigation", { name: "Block palette" });
  for (const [slot, block] of FILLS) {
    await palette.getByRole("listitem").filter({ hasText: new RegExp(`^${block}$`) }).dragTo(page.getByRole("group", { name: `${slot} slot`, exact: true }));
    await expect(page.getByRole("group", { name: `${slot} slot`, exact: true })).toContainText(`${block}(…)`);
  }
  const source = await readFile(request, FILE);
  expect(source).toContain("mask=CausalMask()");
  expect(source).not.toContain("=None");

  await page.getByRole("button", { name: "Run the check" }).click();
  const result = page.getByRole("region", { name: "Check result" });
  await expect(result.getByText("passed", { exact: true }).first()).toBeVisible({ timeout: 120_000 });
  await expect(result.getByText("same-as-reference")).toBeVisible();
});
