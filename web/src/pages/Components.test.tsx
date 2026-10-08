import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import first from "../../test/fixtures/unlocks-first-run.json";
import { mockApi } from "../../test/fetch";
import { Providers } from "../app/providers";
import { resetSettingsCache } from "../app/settings";
import { Components } from "./Components";

// the library's own shape: Attention earned, RMSNorm skipped with a reason, the rest locked
const lockable: Record<string, { lesson: string; state: string; reason: string | null }> = Object.fromEntries(
  Object.entries(first.lockable).map(([id, e]) => [id, { lesson: e.lesson, state: "locked", reason: null }]),
);
lockable["block:Attention"] = { ...lockable["block:Attention"]!, state: "earned" };
lockable["block:RMSNorm"] = { ...lockable["block:RMSNorm"]!, state: "skipped", reason: "I have implemented RMSNorm before, in a course" };
const guided = {
  ...first,
  first_run: false,
  policy: "guided",
  lockable,
  unlocks: {
    "block:Attention": { how: "earned", lesson: "foundations/04-multi-head", at: "2026-10-07T17:07:00+00:00", evidence: "learn/checks/c.json", reason: null },
    "block:RMSNorm": { how: "skipped", lesson: "modern-block/01-rmsnorm", at: "2026-10-07T17:12:00+00:00", evidence: null, reason: "I have implemented RMSNorm before, in a course" },
  },
};

function renderPage(routes: Parameters<typeof mockApi>[0] = {}) {
  const seen = mockApi({ "GET /api/learn/unlocks": { body: guided }, ...routes });
  render(
    <Providers>
      <MemoryRouter initialEntries={["/components"]}>
        <Routes>
          <Route path="/components" element={<Components />} />
          <Route path="/learn/*" element={<p>lesson page</p>} />
        </Routes>
      </MemoryRouter>
    </Providers>,
  );
  return seen;
}

beforeEach(() => {
  localStorage.clear();
  act(() => resetSettingsCache());
});
afterEach(() => vi.unstubAllGlobals());

describe("Components", () => {
  it("lists every block and feature with its state, lesson and how it was unlocked", async () => {
    renderPage();
    const table = await screen.findByRole("table");
    const attention = within(table).getByText("block:Attention").closest("tr")!;
    expect(within(attention).getByText("earned")).toBeTruthy();
    expect(within(attention).getByText("foundations/04-multi-head")).toBeTruthy();
    expect(within(attention).getByText(/check passed/)).toBeTruthy();
    expect(within(attention).getByRole("link", { name: "evidence" }).getAttribute("href")).toBe("/learn/foundations/04-multi-head");
    const rms = within(table).getByText("block:RMSNorm").closest("tr")!;
    expect(within(rms).getByText("skipped")).toBeTruthy();
    expect(within(rms).getByText(/I have implemented RMSNorm before, in a course/)).toBeTruthy();
    const locked = within(table).getByText("block:Block").closest("tr")!;
    expect(within(locked).getByText("locked")).toBeTruthy();
    expect(within(locked).getByRole("button", { name: "Unlock one…" })).toBeTruthy();
    expect(within(attention).queryByRole("button", { name: "Unlock one…" })).toBeNull();
    const counts = screen.getAllByText(/^\d+$/, { selector: "strong" }).map((e) => `${e.textContent} ${e.parentElement!.textContent!.replace(/^\d+ /, "")}`);
    expect(counts).toEqual(["1 earned", "1 skipped", `${Object.keys(lockable).length - 2} locked`]);
  });

  it("asks before unlocking everything, and unlocking changes nothing until confirmed", async () => {
    const seen = renderPage({ "POST /api/learn/unlock": { body: { ...guided, policy: "open" } } });
    await screen.findByRole("table");
    await userEvent.click(screen.getByRole("button", { name: "Unlock all" }));
    const dialog = await screen.findByRole("alertdialog");
    expect(within(dialog).getByText("Unlock every block?")).toBeTruthy();
    expect(within(dialog).getByText(/2 unlocks already recorded stay as they are/)).toBeTruthy();
    // Cancel has the focus, so a stray Enter does not unlock anything
    expect(document.activeElement).toBe(within(dialog).getByRole("button", { name: "Cancel" }));
    await userEvent.click(within(dialog).getByRole("button", { name: "Cancel" }));
    expect(seen.some((r) => r.method === "POST")).toBe(false);
    await userEvent.click(screen.getByRole("button", { name: "Unlock all" }));
    await userEvent.click(within(await screen.findByRole("alertdialog")).getByRole("button", { name: "Unlock all" }));
    await waitFor(() => expect(seen.some((r) => r.method === "POST")).toBe(true));
    expect(seen.find((r) => r.method === "POST")?.body).toEqual({ all: true });
  });

  it("unlocks one block with a reason, and will not without one", async () => {
    const seen = renderPage({ "POST /api/learn/unlock": { body: guided } });
    await screen.findByRole("table");
    const row = screen.getByText("block:Block").closest("tr")!;
    await userEvent.click(within(row).getByRole("button", { name: "Unlock one…" }));
    const dialog = await screen.findByRole("dialog");
    const unlock = within(dialog).getByRole("button", { name: "Unlock" }) as HTMLButtonElement;
    expect(unlock.disabled).toBe(true);
    await userEvent.type(within(dialog).getByLabelText("Reason"), "I know this");
    await userEvent.click(unlock);
    await waitFor(() => expect(seen.some((r) => r.method === "POST")).toBe(true));
    expect(seen.find((r) => r.method === "POST")?.body).toEqual({ id: "block:Block", reason: "I know this", all: false });
  });

  it("shows the library's refusal in the dialog, word for word", async () => {
    renderPage({
      "POST /api/learn/unlock": {
        status: 422,
        problem: true,
        body: { title: "Cannot be done as asked", status: 422, detail: "'Nope' is not locked by any lesson (lockable: Attention, Block)" },
      },
    });
    await screen.findByRole("table");
    await userEvent.click(within(screen.getByText("block:Block").closest("tr")!).getByRole("button", { name: "Unlock one…" }));
    const dialog = await screen.findByRole("dialog");
    await userEvent.type(within(dialog).getByLabelText("Reason"), "x");
    await userEvent.click(within(dialog).getByRole("button", { name: "Unlock" }));
    expect((await within(dialog).findByRole("alert")).textContent).toContain("is not locked by any lesson");
  });

  it("offers no Unlock all when nothing is locked", async () => {
    renderPage({ "GET /api/learn/unlocks": { body: { ...guided, lockable: Object.fromEntries(Object.entries(lockable).map(([id, e]) => [id, { ...e, state: "open" }])) } } });
    await screen.findByRole("table");
    expect(screen.queryByRole("button", { name: "Unlock all" })).toBeNull();
  });
});
