import { act, render, screen, within } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import curricula from "../../test/fixtures/curricula.json";
import unlocksFirst from "../../test/fixtures/unlocks-first-run.json";
import { mockApi } from "../../test/fetch";
import { Providers } from "../app/providers";
import { resetSettingsCache } from "../app/settings";
import { Lessons } from "./Lessons";

function renderPage() {
  return render(
    <Providers>
      <MemoryRouter initialEntries={["/learn"]}>
        <Routes>
          <Route path="/learn" element={<Lessons />} />
          <Route path="/welcome" element={<p>welcome page</p>} />
        </Routes>
      </MemoryRouter>
    </Providers>,
  );
}

const guided = { ...unlocksFirst, first_run: false, policy: "guided" };

beforeEach(() => {
  localStorage.clear();
  act(() => resetSettingsCache());
});
afterEach(() => vi.unstubAllGlobals());

describe("Lessons", () => {
  it("sends a first-time visitor to the welcome page", async () => {
    mockApi({ "GET /api/learn/unlocks": { body: unlocksFirst }, "GET /api/curricula": { body: curricula } });
    renderPage();
    expect(await screen.findByText("welcome page")).toBeTruthy();
  });

  it("lists each path with its lessons, counts, estimates and states as the API gives them", async () => {
    mockApi({ "GET /api/learn/unlocks": { body: guided }, "GET /api/curricula": { body: curricula } });
    renderPage();
    const foundations = await screen.findByRole("region", { name: "Foundations" });
    expect(within(foundations).getAllByRole("link")).toHaveLength(6);
    expect(within(foundations).getByText(/level 0 · 0 of 6 passed/)).toBeTruthy();
    expect(within(foundations).getByText("cpu 14 min")).toBeTruthy();
    const multiHead = within(foundations).getByRole("link", { name: /Multi-head attention/ });
    expect(within(multiHead).getByText("block:Attention")).toBeTruthy();
    expect(within(multiHead).getByText("cpu 0.5 min")).toBeTruthy();
    expect(within(multiHead).getByText("locked")).toBeTruthy();
    expect(within(multiHead).getByText("needs foundations/03-attention-head")).toBeTruthy();
    expect(multiHead.getAttribute("href")).toBe("/learn/foundations/04-multi-head");
    // a lesson with no unmet prerequisite shows its state word verbatim
    const first = within(foundations).getByRole("link", { name: /A bigram language model/ });
    expect(within(first).getByText("not-started")).toBeTruthy();
  });

  it("shows both compute variants where a lesson has them", async () => {
    mockApi({ "GET /api/learn/unlocks": { body: guided }, "GET /api/curricula": { body: curricula } });
    renderPage();
    const modern = await screen.findByRole("region", { name: "The modern block" });
    const assemble = within(modern).getByRole("link", { name: /Assemble the modern block/ });
    expect(within(assemble).getByText("cpu 15 min / gpu 10 min")).toBeTruthy();
  });

  it("says the policy, and the library's error when the request fails", async () => {
    mockApi({
      "GET /api/learn/unlocks": { body: guided },
      "GET /api/curricula": {
        status: 422,
        problem: true,
        body: { title: "Cannot be done as asked", status: 422, detail: "lesson foundations/02: bad toml" },
      },
    });
    renderPage();
    expect((await screen.findByRole("alert")).textContent).toContain("lesson foundations/02: bad toml");
    expect(screen.getByText("guided")).toBeTruthy();
  });
});
