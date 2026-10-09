import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import broken from "../../test/fixtures/authoring-broken.json";
import ready from "../../test/fixtures/authoring-ready.json";
import unchecked from "../../test/fixtures/authoring-unchecked.json";
import { mockApi } from "../../test/fetch";
import { Providers } from "../app/providers";
import { resetSettingsCache, setShowCommands } from "../app/settings";
import { Authoring } from "./Authoring";

const FOLDER = "curricula/my-course/01-thing";

function open(doc: unknown, routes: Parameters<typeof mockApi>[0] = {}, folder = FOLDER) {
  const seen = mockApi({ [`GET /api/authoring/${folder}`]: { body: doc }, ...routes });
  render(
    <Providers>
      <MemoryRouter initialEntries={[`/authoring/${folder}`]}>
        <Routes>
          <Route path="/authoring/*" element={<Authoring />} />
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
    setShowCommands(true);
  });
});
afterEach(() => vi.unstubAllGlobals());

describe("Authoring", () => {
  it("shows the check's own words and a row per check with starter and solution outcomes", async () => {
    open(ready);
    await screen.findByRole("heading", { name: "My layer" });
    const result = screen.getByRole("region", { name: "Check result" });
    expect(within(result).getByText("ready")).toBeTruthy();
    expect(within(result).getByText(ready.result.summary)).toBeTruthy();
    expect(within(result).getByText(ready.result.checks[0]!.starter.reason)).toBeTruthy();
    expect(within(result).getByText(ready.result.checks[0]!.solution.reason)).toBeTruthy();
    expect(within(result).getByText(/the files have not changed since/)).toBeTruthy();
    expect(screen.getByRole("button", { name: "Check the lesson again" })).toBeTruthy();
    expect(screen.getByText(`nanoscope learn author-check ${FOLDER}`)).toBeTruthy();
    // a preview: Start and the check are off
    expect((screen.getByRole("button", { name: "Start" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("says the files changed when they did", async () => {
    open({ ...ready, result: { ...ready.result, fresh: false } });
    expect(await screen.findByText(/the files changed since: check again/)).toBeTruthy();
  });

  it("lists every loader problem when the folder does not load", async () => {
    open(broken, {}, "curricula/other/01-thing");
    await screen.findByText("does not load");
    for (const p of broken.problems) {
      expect(screen.getByText(p.where)).toBeTruthy();
      expect(screen.getByText(p.message)).toBeTruthy();
    }
    expect(screen.getByText(/The curriculum loader found 3 problems/)).toBeTruthy();
    expect(screen.getByText("No preview and no check runs yet")).toBeTruthy();
    expect(screen.queryByRole("region", { name: "As a learner sees it" })).toBeNull();
  });

  it("asks for a first check, queues the job and reads the folder again when it is done", async () => {
    const seen = open(unchecked, {
      [`POST /api/authoring/${FOLDER}/check`]: { status: 202, body: { id: 3, kind: "author-check", state: "queued" } },
      "GET /api/jobs/3": { body: { id: 3, kind: "author-check", state: "done", result: ready.result, error: null } },
    });
    await screen.findByText("Not checked yet");
    await userEvent.click(screen.getByRole("button", { name: "Check the lesson" }));
    await waitFor(() => expect(seen.some((r) => r.method === "POST" && r.path === `/api/authoring/${FOLDER}/check`)).toBe(true));
    await waitFor(() => expect(seen.filter((r) => r.path === `/api/authoring/${FOLDER}`).length).toBeGreaterThan(1));
    expect(seen.find((r) => r.method === "POST")?.body).toEqual({ variant: "cpu" });
  });
});
