import { expect, test } from "@playwright/test";

// Plan done-when: Unlock all on the Components page sets policy open in learn/unlocks.json.
test("Unlock all sets the policy to open", async ({ page, request }) => {
  await page.goto("/components");
  await page.getByRole("button", { name: "Unlock all" }).click();
  const dialog = page.getByRole("alertdialog");
  await expect(dialog).toContainText("Unlock every block?");
  await dialog.getByRole("button", { name: "Unlock all" }).click();
  await expect(page.getByRole("button", { name: "Unlock all" })).toHaveCount(0);

  const view = (await (await request.get("/api/learn/unlocks")).json()) as { policy: string; lockable: Record<string, { state: string }> };
  expect(view.policy).toBe("open");
  expect(Object.values(view.lockable).every((e) => e.state !== "locked")).toBe(true);
});
