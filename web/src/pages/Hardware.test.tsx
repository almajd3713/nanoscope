import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import models from "../../test/fixtures/models.json";
import presets from "../../test/fixtures/presets.json";
import { mockApi } from "../../test/fetch";
import { resetLevelCache } from "../app/level";
import { Providers } from "../app/providers";
import { Hardware } from "./Hardware";

const GB = 1024 ** 3;
const hardware = {
  devices: [
    { name: "cpu", kind: "cpu", memory_total: null, memory_free: null, label: null },
    { name: "cuda:0", kind: "cuda", memory_total: 8.6 * GB, memory_free: 6.3 * GB, label: "NVIDIA GeForce RTX 4070 Laptop GPU" },
  ],
  cpu_count: 16, torch: "2.14.0+cu130", workers: [],
};
const worker = (device: string, slots: number, jobs: number[]) => ({
  worker_id: `host-1-${device}`, device, slots, jobs, started_at: "2026-10-08T21:02:00+00:00", heartbeat_at: "x", age: 1, secrets: { HF_TOKEN: true, WANDB_API_KEY: false }, pid: 1, host: "h", nanoscope: "0.5", schema: 1,
});
const job = (id: number, state: string, ref: string | null, over: object = {}) => ({
  id, kind: "run", lane: "batch", state, ref, owner: "local", device: state === "running" ? "cuda:0" : null, worker_id: null, attempts: 0, payload: {}, result: null, error: null, created_at: id, ...over,
});
const jobs = [
  job(1, "running", "studies/m1/gpt2/seed-0"),
  job(2, "running", "studies/m1/modern/seed-0"),
  job(3, "queued", "studies/m1/no-rope/seed-0"),
  job(4, "done", null, { kind: "bench", payload: { model: "Bigram" } }),
];
const benchRow = { schema: 1, at: "2026-10-08T20:58:00+00:00", model: "GPT2", device: "cuda:0", step_ms: 29.7, tokens_per_sec: 68844, tflops: 0.65, verdict: "CPU-bound: the GPU is busy 31% of each step." };

beforeEach(() => {
  localStorage.clear();
  resetLevelCache();
});
afterEach(() => vi.unstubAllGlobals());

function open(routes: Parameters<typeof mockApi>[0] = {}) {
  const seen = mockApi({
    "GET /api/hardware": { body: hardware },
    "GET /api/workers": { body: [worker("cuda:0", 2, [1, 2]), worker("cpu", 1, [])] },
    "GET /api/jobs": { body: jobs },
    "GET /api/hardware/bench": { body: [benchRow] },
    "GET /api/models": { body: models },
    "GET /api/presets": { body: presets },
    ...routes,
  });
  render(
    <Providers>
      <MemoryRouter>
        <Hardware />
      </MemoryRouter>
    </Providers>,
  );
  return seen;
}

describe("Hardware page", () => {
  it("shows devices with the GPU's name, workers with their secrets by name, and what is busy", async () => {
    open();
    const devices = await screen.findByRole("region", { name: "Devices" });
    expect(within(devices).getByText("NVIDIA GeForce RTX 4070 Laptop GPU")).toBeTruthy();
    expect(within(devices).getByText("6.3 of 8.6 GB")).toBeTruthy();
    expect(within(devices).getByText("1 worker · 2 slots · all busy")).toBeTruthy();
    expect(within(devices).getByText("1 worker · 1 slot · idle")).toBeTruthy();
    const workers = screen.getByRole("region", { name: "Workers" });
    expect(within(workers).getAllByText("HF_TOKEN").length).toBe(2); // one per worker
    expect(within(workers).queryByText("WANDB_API_KEY")).toBeNull(); // a secret the worker lacks is not listed
    expect(within(workers).getByText("1, 2")).toBeTruthy();
  });

  it("lists jobs with a filter, cancels one, and stops a whole study", async () => {
    const seen = open({ "POST /api/jobs/3/cancel": { body: job(3, "cancelled", "studies/m1/no-rope/seed-0") }, "POST /api/studies/m1/stop": { body: { study: "m1", stopping: [] } } });
    const list = await screen.findByRole("region", { name: "Jobs" });
    expect(within(list).getByText("2 running")).toBeTruthy();
    await userEvent.click(within(list).getByRole("radio", { name: "Queued" }));
    expect(within(list).queryByText("studies/m1/gpt2/seed-0")).toBeNull();
    await userEvent.click(within(list).getByRole("button", { name: "Cancel job 3" }));
    await waitFor(() => expect(seen.some((s) => s.path === "/api/jobs/3/cancel")).toBe(true));
    await userEvent.click(within(list).getByRole("button", { name: "Stop study m1" }));
    await waitFor(() => expect(seen.some((s) => s.path === "/api/studies/m1/stop")).toBe(true));
  });

  it("runs a bench on the chosen device, warns when the device is training, and lists the history", async () => {
    const seen = open({ "POST /api/bench": { status: 202, body: job(9, "queued", null, { kind: "bench" }) } });
    const bench = await screen.findByRole("region", { name: "Bench" });
    expect(within(bench).getByText(/CPU-bound: the GPU is busy 31%/)).toBeTruthy();
    expect(within(bench).getByText("68,844")).toBeTruthy();
    expect(within(bench).getByText(/cuda:0 is training 2 runs; a bench there now measures a shared device/)).toBeTruthy();
    await userEvent.click(within(bench).getByRole("button", { name: "Run bench" }));
    await waitFor(() => expect(seen.some((s) => s.path === "/api/bench")).toBe(true));
    expect(seen.find((s) => s.path === "/api/bench")!.body).toMatchObject({ device: "cuda:0", steps: 60, preset: "tinystories-5min" });
  });

  it("says plainly when no worker runs", async () => {
    open({ "GET /api/workers": { body: [] }, "GET /api/jobs": { body: [] }, "GET /api/hardware/bench": { body: [] } });
    expect((await screen.findAllByText(/No worker is running/)).length).toBe(2); // the workers panel, and the bench waiting
    expect(screen.getByText(/No bench results yet/)).toBeTruthy();
  });
});
