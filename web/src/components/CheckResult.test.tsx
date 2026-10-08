import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import curricula from "../../test/fixtures/curricula.json";
import lesson04 from "../../test/fixtures/lesson-04.json";
import { mockApi } from "../../test/fetch";
import { Providers } from "../app/providers";
import { resetSettingsCache } from "../app/settings";
import { Lesson } from "../pages/Lesson";
import { CheckResult } from "./CheckResult";

const REASON = "output differs from the reference: max abs diff 0.176 on inputs of shape [[2, 6, 16]] (tolerance 1e-05)";

describe("CheckResult", () => {
  it("shows each check's id, kind and the library's reason word for word", () => {
    render(
      <CheckResult
        status="failed"
        title="Multi-head attention"
        checks={[
          { id: "built", kind: "defines", passed: true, reason: "MultiHead(d_model=16, context_length=8) builds, 1,024 parameters" },
          { id: "same-as-reference", kind: "equivalent", passed: false, reason: REASON },
          { id: "built-by-hand", kind: "forbid", passed: null, reason: "not run yet" },
        ]}
      >
        Read the reasons.
      </CheckResult>,
    );
    expect(screen.getByText(REASON)).toBeTruthy();
    expect(screen.getByText("failed")).toBeTruthy();
    expect(screen.getByLabelText("passed")).toBeTruthy();
    expect(screen.getByLabelText("not run yet")).toBeTruthy();
    expect(screen.getByText("equivalent")).toBeTruthy();
  });
});

function job(state: string, extra: object = {}) {
  return {
    id: 7, kind: "check", lane: "interactive", state, ref: null, owner: "local", device: null,
    worker_id: null, attempts: 1, payload: { lesson: "foundations/04-multi-head", variant: "cpu" },
    result: null, error: null, created_at: 1, started_at: null, finished_at: null, ...extra,
  };
}

function renderLesson(routes: Parameters<typeof mockApi>[0]) {
  const seen = mockApi({
    "GET /api/curricula": { body: curricula },
    "GET /api/curricula/foundations/04-multi-head": { body: { ...lesson04, state: "started" } },
    "GET /api/files/lessons/foundations/04-multi-head/starter.py": {
      body: { path: "lessons/foundations/04-multi-head/starter.py", content: "x = 1\n", etag: "e" },
    },
    ...routes,
  });
  render(
    <Providers>
      <MemoryRouter initialEntries={["/learn/foundations/04-multi-head"]}>
        <Routes>
          <Route path="/learn/:path/:lesson" element={<Lesson />} />
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

describe("Lesson check", () => {
  it("enqueues the check, shows it waiting, then each verdict and what passing unlocked", async () => {
    let jobs: unknown[] = [];
    const seen = renderLesson({
      "GET /api/jobs": { body: () => jobs },
      "POST /api/curricula/foundations/04-multi-head/check": {
        status: 202,
        body: () => {
          jobs = [job("queued")];
          return job("queued");
        },
      },
    });
    await userEvent.click(await screen.findByRole("button", { name: "Run the check" }));
    expect(await screen.findByText("Waiting for a worker to start the check")).toBeTruthy();
    expect(seen.find((r) => r.method === "POST" && r.path.endsWith("/check"))?.body).toEqual({ variant: "cpu" });
    jobs = [
      job("done", {
        result: {
          passed: true, check_id: "c1", lesson: "foundations/04-multi-head",
          checks: [
            { id: "built", passed: true, reason: "MultiHead builds" },
            { id: "same-as-reference", passed: true, reason: "matches the reference" },
            { id: "built-by-hand", passed: true, reason: "no shortcuts" },
          ],
        },
      }),
    ];
    // the page polls while the job is queued
    const panel = await screen.findByRole("region", { name: "Check result" }, { timeout: 4000 });
    await waitFor(() => expect(within(panel).getByText("matches the reference")).toBeTruthy());
    expect(within(panel).getByText("Check passed. Attention is unlocked.")).toBeTruthy();
  });

  it("shows the last result after a reload, with the library's reasons", async () => {
    renderLesson({
      "GET /api/jobs": {
        body: [
          job("done", {
            result: {
              passed: false, check_id: "c1", lesson: "foundations/04-multi-head",
              checks: [{ id: "same-as-reference", passed: false, reason: REASON }],
            },
          }),
        ],
      },
    });
    const panel = await screen.findByRole("region", { name: "Check result" });
    expect(within(panel).getByText(REASON)).toBeTruthy();
    expect(within(panel).getByText("Read the reasons, edit your file, and run the check again.")).toBeTruthy();
  });

  it("says so when the job failed instead of hiding it", async () => {
    renderLesson({ "GET /api/jobs": { body: [job("failed", { error: "worker crashed: out of memory" })] } });
    expect((await screen.findByRole("alert")).textContent).toContain("worker crashed: out of memory");
  });
});
