import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import first from "../../test/fixtures/unlocks-first-run.json";
import { mockApi } from "../../test/fetch";
import { resetLevelCache } from "./level";
import { Providers } from "./providers";
import { Shell } from "./Shell";
import { resetOffer } from "./unlockOffer";

// Attention earned, RMSNorm skipped, the rest locked: the library's own shape
const lockable: Record<string, { lesson: string; state: string; reason: string | null }> = Object.fromEntries(
  Object.entries(first.lockable).map(([id, e]) => [id, { lesson: e.lesson, state: "locked", reason: null }]),
);
lockable["block:Attention"] = { ...lockable["block:Attention"]!, state: "earned" };
lockable["block:RMSNorm"] = { ...lockable["block:RMSNorm"]!, state: "skipped", reason: "done it before" };
const guided = { ...first, first_run: false, policy: "guided", lockable, unlocks: {} };

function renderShell(unlocks: object = guided, extra: Parameters<typeof mockApi>[0] = {}) {
  const seen = mockApi({
    "GET /api/jobs": { body: [] },
    "GET /api/workers": { body: [] },
    "GET /api/learn/unlocks": { body: unlocks },
    ...extra,
  });
  render(
    <Providers>
      <MemoryRouter initialEntries={["/runs"]}>
        <Routes>
          <Route element={<Shell />}>
            <Route path="/runs" element={<p>runs page</p>} />
          </Route>
        </Routes>
      </MemoryRouter>
    </Providers>,
  );
  return seen;
}

const choose = (level: string) => userEvent.click(screen.getByRole("radio", { name: level }));

beforeEach(() => {
  localStorage.clear();
  act(() => {
    resetLevelCache();
    resetOffer();
  });
  vi.stubGlobal("matchMedia", () => ({ matches: false, addEventListener() {}, removeEventListener() {} }));
});
afterEach(() => vi.unstubAllGlobals());

describe("Unlock all offer on switching level", () => {
  it("is made when the level becomes Research, says what stays recorded, and keeps the locks by default", async () => {
    const seen = renderShell();
    await choose("Research");
    const dialog = await screen.findByRole("alertdialog");
    expect(dialog.textContent).toContain("You switched to Research. Unlock every block too?");
    expect(dialog.textContent).toMatch(/Lessons still lock \d+ blocks and features/);
    expect(dialog.textContent).toMatch(/the 1 you earned and the 1 you skipped stay recorded/);
    await userEvent.click(screen.getByRole("button", { name: "Keep the locks" }));
    expect(screen.queryByRole("alertdialog")).toBeNull();
    expect(seen.some((s) => s.method === "POST")).toBe(false); // nothing changed
  });

  it("is made once only: not again on the next switch, nor after a reload", async () => {
    renderShell();
    await choose("Research");
    await userEvent.click(await screen.findByRole("button", { name: "Keep the locks" }));
    await choose("Tinker");
    await choose("Extend");
    expect(screen.queryByRole("alertdialog")).toBeNull();
    expect(localStorage.getItem("nanoscope.unlockOffered")).toBe("1");
  });

  it("is not made for Learn or Tinker, nor when the policy is already open", async () => {
    renderShell();
    await choose("Tinker");
    expect(screen.queryByRole("alertdialog")).toBeNull();
    await choose("Learn");
    expect(screen.queryByRole("alertdialog")).toBeNull();
  });

  it("unlocks everything when accepted", async () => {
    const seen = renderShell(guided, { "POST /api/learn/unlock": { body: { policy: "open" } } });
    await choose("Extend");
    await screen.findByRole("alertdialog");
    await userEvent.click(screen.getByRole("button", { name: "Unlock all" }));
    await waitFor(() => expect(seen.find((s) => s.method === "POST")?.body).toEqual({ all: true }));
    await waitFor(() => expect(screen.queryByRole("alertdialog")).toBeNull());
  });

  it("makes no offer when the policy is open", async () => {
    renderShell({ ...guided, policy: "open" });
    await choose("Research");
    await new Promise((r) => setTimeout(r, 50));
    expect(screen.queryByRole("alertdialog")).toBeNull();
  });
});
