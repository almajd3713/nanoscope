import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import jobs from "../../test/fixtures/jobs.json";
import { mockApi } from "../../test/fetch";
import { resetLevelCache, setLevel } from "../app/level";
import { Providers } from "../app/providers";
import { QueuePanel } from "./QueuePanel";

// worker.v1 documents, as the workers write them
const worker = (device: string, running: number[], age = 2) => ({
  schema: 1, nanoscope: "0.4.0", worker_id: `almajd3713-1-${device}`, device, slots: 1, pid: 1, host: "h",
  jobs: running, started_at: "2026-10-08T12:00:00+00:00", heartbeat_at: "2026-10-08T12:00:02+00:00", age,
});

function renderPanel(routes: Parameters<typeof mockApi>[0] = {}) {
  const seen = mockApi({ "GET /api/jobs": { body: jobs }, "GET /api/workers": { body: [worker("cpu", [])] }, ...routes });
  render(
    <Providers>
      <MemoryRouter>
        <QueuePanel />
      </MemoryRouter>
    </Providers>,
  );
  return seen;
}

beforeEach(() => {
  localStorage.clear();
  act(() => resetLevelCache());
});
afterEach(() => vi.unstubAllGlobals());

describe("QueuePanel", () => {
  it("is one line until opened: the counts and the worker", async () => {
    renderPanel();
    expect(await screen.findByText("2 queued")).toBeTruthy();
    expect(screen.getByText("· cpu idle")).toBeTruthy();
    expect(screen.queryByRole("list")).toBeNull();
    expect(screen.getByRole("button", { name: "Queue" }).getAttribute("aria-expanded")).toBe("false");
  });

  it("opens into the jobs, active first, with a cancel for queued ones only", async () => {
    renderPanel();
    await screen.findByText("2 queued");
    await userEvent.click(screen.getByRole("button", { name: "Queue" }));
    const items = within(screen.getByRole("list")).getAllByRole("listitem");
    expect(items).toHaveLength(jobs.length);
    expect(within(items[0]!).getByText("queued")).toBeTruthy();
    const failed = items.find((i) => within(i).queryByText("failed"))!;
    expect(within(failed).queryByRole("button")).toBeNull();
    expect(screen.getAllByRole("button", { name: /Cancel job/ })).toHaveLength(2);
    expect(screen.getByText("Cancelling a run saves a checkpoint first; train it again and it resumes.")).toBeTruthy();
  });

  it("cancels a job without asking", async () => {
    const seen = renderPanel({ "POST /api/jobs/3/cancel": { body: { ...jobs[2], state: "cancelled" } } });
    await screen.findByText("2 queued");
    await userEvent.click(screen.getByRole("button", { name: "Queue" }));
    await userEvent.click(screen.getByRole("button", { name: "Cancel job 3" }));
    await waitFor(() => expect(seen.some((r) => r.method === "POST" && r.path === "/api/jobs/3/cancel")).toBe(true));
  });

  it("says when no worker is running, and shows each worker from Research up", async () => {
    renderPanel({ "GET /api/workers": { body: [] } });
    expect(await screen.findByText("· no worker")).toBeTruthy();
  });

  it("lists workers with their device and slots at Research", async () => {
    renderPanel({ "GET /api/workers": { body: [worker("cpu", [4]), worker("cuda:0", [])] } });
    await screen.findByText("2 queued");
    act(() => setLevel("Research"));
    expect(await screen.findByText("almajd3713-1-cpu · cpu · 1 slot · busy")).toBeTruthy();
    expect(screen.getByText("almajd3713-1-cuda:0 · cuda:0 · 1 slot · idle")).toBeTruthy();
    expect(screen.getByRole("link", { name: "Hardware" }).getAttribute("href")).toBe("/hardware");
  });

  it("ignores a worker whose heartbeat stopped", async () => {
    renderPanel({ "GET /api/workers": { body: [worker("cpu", [], 600)] } });
    expect(await screen.findByText("· no worker")).toBeTruthy();
  });

  it("shows the library's error", async () => {
    renderPanel({
      "GET /api/jobs": { status: 422, problem: true, body: { title: "Cannot be done as asked", status: 422, detail: "queue.db is locked" } },
    });
    await userEvent.click(await screen.findByRole("button", { name: "Queue" }));
    expect((await screen.findByRole("alert")).textContent).toContain("queue.db is locked");
  });
});
