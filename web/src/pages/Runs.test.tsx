import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import baselines from "../../test/fixtures/runs-baselines.json";
import runs from "../../test/fixtures/runs.json";
import { installEventSource, lastSource } from "../../test/eventsource";
import { mockApi } from "../../test/fetch";
import { resetLevelCache, setLevel } from "../app/level";
import { Providers } from "../app/providers";
import { resetSettingsCache } from "../app/settings";
import { Runs } from "./Runs";

function renderRuns(path = "/runs") {
  const seen = mockApi({
    "GET /api/runs": { body: runs },
  });
  render(
    <Providers>
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route path="/runs" element={<Runs />} />
          <Route path="/compare" element={<p>compare page</p>} />
        </Routes>
      </MemoryRouter>
    </Providers>,
  );
  return seen;
}

beforeEach(() => {
  localStorage.clear();
  act(() => {
    resetSettingsCache();
    resetLevelCache();
  });
  installEventSource();
});
afterEach(() => vi.unstubAllGlobals());

describe("Runs", () => {
  it("lists the runs newest first, with state, model, step and val_bpb as the API gives them", async () => {
    renderRuns();
    const table = await screen.findByRole("table");
    const rows = within(table).getAllByRole("row").slice(1);
    expect(rows).toHaveLength(runs.length);
    const refs = rows.map((r) => within(r).getByRole("link").textContent);
    expect(refs).toEqual([...runs].reverse().map((r) => r.ref));
    const failed = rows.find((r) => within(r).queryByText("failed"))!;
    expect(within(failed).getByText("MyBigram")).toBeTruthy();
    expect(within(failed).getByText("0 / 500")).toBeTruthy();
  });

  it("filters by state", async () => {
    renderRuns();
    await screen.findByRole("table");
    await userEvent.click(screen.getByRole("button", { name: "failed" }));
    const rows = within(screen.getByRole("table")).getAllByRole("row").slice(1);
    expect(rows).toHaveLength(runs.filter((r) => r.state === "failed").length);
    expect(screen.getByRole("button", { name: "failed" }).getAttribute("aria-pressed")).toBe("true");
  });

  it("asks the API for the prefix you type", async () => {
    const seen = renderRuns("/runs?prefix=tinystories-5min/mybigram");
    await screen.findByRole("table");
    expect(seen.some((r) => r.path === "/api/runs?prefix=tinystories-5min%2Fmybigram")).toBe(true);
    expect((screen.getByLabelText("Filter by ref prefix") as HTMLInputElement).value).toBe("tinystories-5min/mybigram");
  });

  it("updates a run in place from the live events, and says when the stream is paused", async () => {
    renderRuns();
    await screen.findByRole("table");
    const ref = runs[0]!.ref;
    act(() => lastSource().open());
    act(() => lastSource().emit("state", { ref, state: "running", step: 40, max_steps: 500 }));
    act(() => lastSource().emit("eval", { ref, step: 50, val_loss: 3, val_bpb: 1.234 }, "50"));
    const row = screen.getByRole("link", { name: ref }).closest("tr")!;
    expect(within(row).getByText("running")).toBeTruthy();
    expect(within(row).getByText("40 / 500")).toBeTruthy();
    expect(within(row).getByText("1.234")).toBeTruthy();
    act(() => lastSource().fail());
    expect(await screen.findByText("Live updates paused. Reconnecting…")).toBeTruthy();
  });

  it("offers Compare selected only from Tinker up, and only for two or more runs", async () => {
    renderRuns();
    await screen.findByRole("table");
    expect(screen.queryByRole("link", { name: "Compare selected" })).toBeNull();
    expect(screen.queryByRole("link", { name: /New run/ })).toBeNull();
    act(() => setLevel("Tinker"));
    const link = screen.getByRole("link", { name: "Compare selected" });
    expect(link.getAttribute("aria-disabled")).toBe("true");
    await userEvent.click(screen.getByRole("checkbox", { name: `Select ${runs[0]!.ref}` }));
    await userEvent.click(screen.getByRole("checkbox", { name: `Select ${runs[1]!.ref}` }));
    expect(link.getAttribute("aria-disabled")).toBe("false");
    expect(link.getAttribute("href")).toBe(`/compare?runs=${runs[0]!.ref},${runs[1]!.ref}`);
    expect(screen.getByRole("link", { name: /New run/ }).getAttribute("href")).toBe("/runs/new");
  });

  it("adds the shipped baselines when asked", async () => {
    // the prefix decides what the list returns
    vi.stubGlobal(
      "fetch",
      vi.fn(async (request: Request) => {
        const url = new URL(request.url);
        const prefix = url.searchParams.get("prefix");
        return new Response(JSON.stringify(prefix === "baselines" ? baselines : runs), {
          headers: { "content-type": "application/json" },
        });
      }),
    );
    render(
      <Providers>
        <MemoryRouter initialEntries={["/runs"]}>
          <Routes>
            <Route path="/runs" element={<Runs />} />
          </Routes>
        </MemoryRouter>
      </Providers>,
    );
    await screen.findByRole("table");
    expect(screen.queryByText(baselines[0]!.ref)).toBeNull();
    await userEvent.click(screen.getByRole("checkbox", { name: "Show shipped baselines" }));
    await waitFor(() => expect(screen.getByText(baselines[0]!.ref)).toBeTruthy());
  });

  it("shows the library's error", async () => {
    mockApi({
      "GET /api/runs": { status: 422, problem: true, body: { title: "Cannot be done as asked", status: 422, detail: "bad prefix '..'" } },
    });
    render(
      <Providers>
        <MemoryRouter initialEntries={["/runs"]}>
          <Routes>
            <Route path="/runs" element={<Runs />} />
          </Routes>
        </MemoryRouter>
      </Providers>,
    );
    expect((await screen.findByRole("alert")).textContent).toContain("bad prefix '..'");
  });
});
