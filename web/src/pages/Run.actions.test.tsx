import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import failed from "../../test/fixtures/run-failed.json";
import { installEventSource } from "../../test/eventsource";
import { mockApi } from "../../test/fetch";
import { resetLevelCache, setLevel } from "../app/level";
import { Providers } from "../app/providers";
import { resetSettingsCache } from "../app/settings";
import { Run } from "./Run";

const REF = "tinystories-5min/mybigram/seed-0";

const withState = (state: string) => ({
  ...failed,
  status: { ...failed.status, state, error: null, step: 200, max_steps: 500 },
});

function renderRun(state: string, routes: Parameters<typeof mockApi>[0] = {}) {
  const seen = mockApi({ [`GET /api/runs/${REF}`]: { body: withState(state) }, ...routes });
  render(
    <Providers>
      <MemoryRouter initialEntries={[`/runs/${REF}`]}>
        <Routes>
          <Route path="/runs/*" element={<Run />} />
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

describe("Run actions", () => {
  it("stops a running run", async () => {
    const seen = renderRun("running", {
      [`POST /api/runs/${REF}/stop`]: { body: { ref: REF, stopping: [REF], cancelled_jobs: [] } },
    });
    await userEvent.click(await screen.findByRole("button", { name: "Stop run" }));
    expect(seen.some((r) => r.method === "POST" && r.path === `/api/runs/${REF}/stop`)).toBe(true);
    expect(screen.queryByRole("button", { name: "Resume" })).toBeNull();
  });

  it("resumes a stopped run, and shows the library's refusal word for word", async () => {
    renderRun("stopped", {
      [`POST /api/runs/${REF}/resume`]: {
        status: 409,
        problem: true,
        body: { title: "Conflict", status: 409, detail: `${REF} was trained from a class a worker cannot import again; its source is saved next to the run` },
      },
    });
    expect(screen.queryByRole("button", { name: "Stop run" })).toBeNull();
    await userEvent.click(await screen.findByRole("button", { name: "Resume" }));
    expect((await screen.findByRole("alert")).textContent).toContain("a worker cannot import again");
  });

  it("offers Duplicate and change one thing from Tinker up, linking to the run form with this run", async () => {
    renderRun("done");
    await screen.findByRole("heading", { name: REF });
    expect(screen.queryByRole("link", { name: /Duplicate and change one thing/ })).toBeNull();
    act(() => setLevel("Tinker"));
    const link = screen.getByRole("link", { name: /Duplicate and change one thing/ });
    expect(link.getAttribute("href")).toBe(`/runs/new?from=${encodeURIComponent(REF)}`);
  });

  it("generates text from a finished run with the API's defaults", async () => {
    const seen = renderRun("done", {
      [`POST /api/runs/${REF}/generate`]: { body: { ref: REF, text: "Once upon a time there was a cat.", job_id: 5 } },
    });
    await userEvent.type(await screen.findByLabelText("Prompt"), "Once");
    await userEvent.click(screen.getByRole("button", { name: "Generate" }));
    expect(await screen.findByText("Once upon a time there was a cat.")).toBeTruthy();
    expect(seen.find((r) => r.path.endsWith("/generate"))?.body).toEqual({
      prompt: "Once", max_new_tokens: 200, temperature: 0.8, seed: 42, timeout: 30,
    });
  });

  it("says when no worker is there to generate, in the library's words", async () => {
    renderRun("done", {
      [`POST /api/runs/${REF}/generate`]: {
        status: 503,
        problem: true,
        body: { title: "No worker available", status: 503, detail: "no worker picked the job up: start one with `nanoscope worker` (or `nanoscope serve --worker cpu`)" },
      },
    });
    await userEvent.click(await screen.findByRole("button", { name: "Generate" }));
    expect((await screen.findByRole("alert")).textContent).toContain("no worker picked the job up");
  });

  it("offers no generate box while a run is still running", async () => {
    renderRun("running");
    await screen.findByRole("heading", { name: REF });
    expect(screen.queryByRole("button", { name: "Generate" })).toBeNull();
  });
});
