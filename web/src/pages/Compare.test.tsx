import { act, render, screen, within } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import compare from "../../test/fixtures/compare.json";
import { mockApi } from "../../test/fetch";
import { Providers } from "../app/providers";
import { resetSettingsCache } from "../app/settings";
import { Compare } from "./Compare";

function renderCompare(path: string, routes: Parameters<typeof mockApi>[0] = {}) {
  const seen = mockApi({ "POST /api/compare": { body: compare }, ...routes });
  render(
    <Providers>
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route path="/compare" element={<Compare />} />
          <Route path="/runs" element={<p>runs list</p>} />
        </Routes>
      </MemoryRouter>
    </Providers>,
  );
  return seen;
}

const URL_ = "/compare?runs=bigram,modern,gpt2&preset=tinystories-5min";

beforeEach(() => {
  localStorage.clear();
  act(() => resetSettingsCache());
});
afterEach(() => vi.unstubAllGlobals());

describe("Compare", () => {
  it("asks the API for the runs in the URL, with the preset", async () => {
    const seen = renderCompare(URL_);
    await screen.findByRole("heading", { name: "Compare" });
    expect(seen.find((r) => r.path === "/api/compare")?.body).toEqual({
      sets: ["bigram", "modern", "gpt2"], baseline: null, preset: "tinystories-5min", metric: "val_bpb",
    });
  });

  it("shows the library's title, verdict sentences and table cells verbatim", async () => {
    renderCompare(URL_);
    await screen.findByRole("heading", { name: "Compare" });
    expect(screen.getByText(compare.title)).toBeTruthy();
    const verdicts = screen.getByRole("region", { name: "Verdicts" });
    for (const row of compare.rows) {
      if (row.text.statement) expect(within(verdicts).getByText(row.text.statement)).toBeTruthy();
    }
    const table = screen.getByRole("table");
    for (const row of compare.rows) {
      const tr = within(table).getByText(row.label).closest("tr")!;
      expect(within(tr).getByText(row.text.value)).toBeTruthy();
      expect(within(tr).getByText(row.text.delta)).toBeTruthy();
      expect(within(tr).getByText(row.verdict)).toBeTruthy();
    }
    expect(within(table).getByText(compare.params_header)).toBeTruthy();
    expect(within(table).getByText(`Δ vs ${compare.baseline}, 95% CI`)).toBeTruthy();
  });

  it("states the verdict as a word with a glyph, and the baseline as the baseline", async () => {
    renderCompare(URL_);
    await screen.findByRole("heading", { name: "Compare" });
    const table = screen.getByRole("table");
    const modern = within(table).getByText("Modern").closest("tr")!;
    expect(within(modern).getByText("better")).toBeTruthy();
    const base = within(table).getByText("GPT2").closest("tr")!;
    expect(within(base).getByText("baseline")).toBeTruthy();
    expect(within(base).getByText("(baseline)")).toBeTruthy();
  });

  it("draws the forest plot and every seed's curve, and prints the precision plan line", async () => {
    renderCompare(URL_);
    await screen.findByRole("heading", { name: "Compare" });
    expect(screen.getByRole("img").getAttribute("aria-label")).toContain("Modern");
    expect(screen.getByRole("region", { name: "Curves" })).toBeTruthy();
    expect(screen.getByText(compare.precision_plan.text!)).toBeTruthy();
  });

  it("shows the library's notes as warnings, word for word", async () => {
    renderCompare(URL_, {
      "POST /api/compare": { body: { ...compare, notes: ["runs trained on different numbers of tokens; this compares the training budgets as much as the models"] } },
    });
    expect(await screen.findByText(/runs trained on different numbers of tokens/)).toBeTruthy();
  });

  it("says why there is no interval when seeds are too few", async () => {
    const few = {
      ...compare,
      precision_plan: { note: "a confidence interval needs at least 3 seeds" },
      rows: compare.rows.map((r) =>
        r.delta
          ? { ...r, verdict: "no CI", delta: { ...r.delta, ci95_low: null, ci95_high: null, n: 1 }, text: { ...r.text, verdict: "no CI: need 3+ seeds each", statement: `${r.label} vs GPT2: +0.100 bpb, need 3+ seeds each for an interval` } }
          : r,
      ),
    };
    renderCompare(URL_, { "POST /api/compare": { body: few } });
    expect(await screen.findByText("a confidence interval needs at least 3 seeds")).toBeTruthy();
    expect(screen.getAllByText(/need 3\+ seeds each/).length).toBeGreaterThan(0);
  });

  it("asks for runs when there are fewer than two", async () => {
    renderCompare("/compare?runs=bigram");
    expect(await screen.findByText("Pick at least two runs")).toBeTruthy();
    expect(screen.getByRole("link", { name: "Choose runs" }).getAttribute("href")).toBe("/runs");
  });

  it("shows the library's error word for word", async () => {
    renderCompare(URL_, {
      "POST /api/compare": {
        status: 422,
        problem: true,
        body: { title: "Cannot be done as asked", status: 422, detail: "these runs were evaluated on different text: [('a', 1), ('b', 2)]" },
      },
    });
    expect((await screen.findByRole("alert")).textContent).toContain("evaluated on different text");
  });
});
