import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import failed from "../../test/fixtures/run-failed.json";
import { installEventSource, lastSource, MockEventSource } from "../../test/eventsource";
import { mockApi } from "../../test/fetch";
import { Providers } from "../app/providers";
import { resetSettingsCache } from "../app/settings";
import { Run } from "./Run";

const REF = "tinystories-5min/mybigram/seed-0";

function renderRun(detail: unknown) {
  mockApi({ [`GET /api/runs/${REF}`]: { body: detail } });
  return render(
    <Providers>
      <MemoryRouter initialEntries={[`/runs/${REF}`]}>
        <Routes>
          <Route path="/runs/*" element={<Run />} />
        </Routes>
      </MemoryRouter>
    </Providers>,
  );
}

const running = {
  ...failed,
  status: { ...failed.status, state: "running", error: null, step: 300, max_steps: 500 },
  summary: { ...failed.summary, final_step: 300, final_val_bpb: 1.214 },
  baseline: {
    ref: "baselines/tinystories-5min/bigram", metric: "val_bpb", n_seeds: 3, values: [1.2, 1.21, 1.22],
    interval: [1.19, 1.24], value: 1.214, inside: true,
  },
};

beforeEach(() => {
  localStorage.clear();
  act(() => resetSettingsCache());
  installEventSource();
});
afterEach(() => vi.unstubAllGlobals());

describe("Run header", () => {
  it("shows a failed run's error and traceback tail from status.json, word for word", async () => {
    renderRun(failed);
    expect(await screen.findByRole("heading", { name: REF })).toBeTruthy();
    expect(screen.getByText("failed")).toBeTruthy();
    const alert = screen.getByRole("alert");
    expect(within(alert).getByText("NotImplementedError")).toBeTruthy();
    expect(within(alert).getByText("embed idx, then project the embeddings to scores")).toBeTruthy();
    const trace = screen.getByRole("region", { name: "Traceback" });
    expect(trace.textContent).toContain('raise NotImplementedError("embed idx, then project the embeddings to scores")');
    // a finished run opens no live stream
    expect(MockEventSource.all).toHaveLength(0);
  });

  it("copies the traceback from the click handler", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    vi.stubGlobal("navigator", { clipboard: { writeText } });
    renderRun(failed);
    await userEvent.click(await screen.findByRole("button", { name: "Copy" }));
    expect(writeText).toHaveBeenCalledWith(failed.status.error.traceback.join("\n"));
  });

  it("follows a running run from its events: state, step, ETA, device and the validation readout", async () => {
    renderRun(running);
    await screen.findByRole("heading", { name: REF });
    expect(screen.getByText("running")).toBeTruthy();
    expect(screen.getByText("step 300")).toBeTruthy();
    expect(screen.getByText("step 300 of 500")).toBeTruthy();
    expect(screen.getByText("Validation bpb, step 300 (lower is better)")).toBeTruthy();
    expect(screen.getByText("1.214")).toBeTruthy();
    expect(screen.getByText("inside the shipped range for one new run: 1.190 – 1.240")).toBeTruthy();
    expect(lastSource().url).toBe(`/api/runs/${REF}/events`);
    act(() => lastSource().open());
    act(() => lastSource().emit("step", { step: 350, loss: 2.1, tokens_per_sec: 40000, elapsed: 0.2 }, "350"));
    expect(screen.getByText("step 350")).toBeTruthy();
    expect(screen.getByText("step 350 of 500")).toBeTruthy();
    expect(screen.getByText("40k")).toBeTruthy();
    expect(screen.getByText("0:30 left")).toBeTruthy(); // 150 steps at 0.2 s
    act(() => lastSource().emit("eval", { step: 350, val_loss: 3, val_bpb: 1.2 }, "350"));
    expect(screen.getByText("Validation bpb, step 350 (lower is better)")).toBeTruthy();
    expect(screen.getByText("1.200")).toBeTruthy();
  });

  it("says live updates are paused when the stream drops, and keeps the last values", async () => {
    renderRun(running);
    await screen.findByRole("heading", { name: REF });
    act(() => lastSource().open());
    act(() => lastSource().fail());
    expect(await screen.findByText("Live updates paused. Reconnecting…")).toBeTruthy();
    expect(screen.getByText("step 300 of 500")).toBeTruthy();
    expect(screen.getByText("1.214")).toBeTruthy();
  });

  it("stops listening once the run is done", async () => {
    renderRun(running);
    await screen.findByRole("heading", { name: REF });
    const source = lastSource();
    act(() => source.open());
    mockApi({ [`GET /api/runs/${REF}`]: { body: { ...running, status: { ...running.status, state: "done", step: 500 } } } });
    act(() => source.emit("state", { state: "done", step: 500, max_steps: 500, error: null }));
    await waitFor(() => expect(source.closed).toBe(true));
    expect(screen.getByText("done")).toBeTruthy();
  });

  it("opens no stream for a run with no status.json, such as a shipped baseline", async () => {
    renderRun({ ...failed, status: null, summary: { ...failed.summary, final_step: 500, final_val_bpb: 1.315 } });
    await screen.findByRole("heading", { name: REF });
    expect(MockEventSource.all).toHaveLength(0);
    expect(screen.getByText("1.315")).toBeTruthy();
  });

  it("shows the library's error when the run does not exist", async () => {
    mockApi({
      [`GET /api/runs/${REF}`]: {
        status: 404,
        problem: true,
        body: { title: "Not found", status: 404, detail: `no run at ref '${REF}'` },
      },
    });
    render(
      <Providers>
        <MemoryRouter initialEntries={[`/runs/${REF}`]}>
          <Routes>
            <Route path="/runs/*" element={<Run />} />
          </Routes>
        </MemoryRouter>
      </Providers>,
    );
    expect((await screen.findByRole("alert")).textContent).toContain(`no run at ref '${REF}'`);
  });
});
