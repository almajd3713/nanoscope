import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import models from "../../test/fixtures/models.json";
import benchHistory from "../../test/fixtures/bench-history.json";
import liveHardware from "../../test/fixtures/hardware-live.json";
import liveJobs from "../../test/fixtures/jobs-live.json";
import liveWorkers from "../../test/fixtures/workers-live.json";
import presets from "../../test/fixtures/presets.json";
import { mockApi } from "../../test/fetch";
import { resetLevelCache, setLevel } from "../app/level";
import { Providers } from "../app/providers";
import { Hardware } from "./Hardware";

// Real API output: a live queue with a CPU worker, a GPU worker holding two runs of a study, two more
// queued, and one saved bench (see the Python test that writes web/test/fixtures/*-live.json).
const hardware = liveHardware;
const jobs = liveJobs;
const benchRow = benchHistory[0]!;
const job = (id: number) => jobs.find((j) => j.id === id)!;

beforeEach(() => {
  localStorage.clear();
  resetLevelCache();
  setLevel("Research");
});
afterEach(() => vi.unstubAllGlobals());

function open(routes: Parameters<typeof mockApi>[0] = {}) {
  const seen = mockApi({
    "GET /api/hardware": { body: hardware },
    "GET /api/workers": { body: liveWorkers },
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
  it("keeps the bench history but hides the bench controls below Research", async () => {
    setLevel("Tinker");
    open();
    expect(await screen.findByText("Bench")).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Run bench" })).toBeNull();
    expect(screen.getByText(Math.round(benchRow.tokens_per_sec).toLocaleString("en-US"))).toBeTruthy();
  });

  it("shows devices with the GPU's name, workers with their secrets by name, and what is busy", async () => {
    open();
    const devices = await screen.findByRole("region", { name: "Devices" });
    expect(within(devices).getByText("NVIDIA GeForce RTX 4070 Laptop GPU")).toBeTruthy();
    expect(within(devices).getByText("6.9 of 8.0 GB")).toBeTruthy();
    expect(within(devices).getByText("1 worker · 2 slots · all busy")).toBeTruthy();
    expect(within(devices).getByText("1 worker · 1 slot · idle")).toBeTruthy();
    const workers = screen.getByRole("region", { name: "Workers" });
    expect(within(workers).getAllByText("HF_TOKEN").length).toBe(2); // one per worker
    expect(within(workers).queryByText("WANDB_API_KEY")).toBeNull(); // a secret the worker lacks is not listed
    expect(within(workers).getByText("4, 5")).toBeTruthy();
  });

  it("lists jobs with a filter, cancels one, and stops a whole study", async () => {
    const seen = open({ "POST /api/jobs/6/cancel": { body: { ...job(6), state: "cancelled" } }, "POST /api/studies/late/stop": { body: { study: "late", stopping: [] } } });
    const list = await screen.findByRole("region", { name: "Jobs" });
    expect(within(list).getByText("2 running")).toBeTruthy();
    await userEvent.click(within(list).getByRole("radio", { name: "Queued" }));
    expect(within(list).queryByText("studies/late/small/seed-0")).toBeNull();
    await userEvent.click(within(list).getByRole("button", { name: "Cancel job 6" }));
    await waitFor(() => expect(seen.some((s) => s.path === "/api/jobs/6/cancel")).toBe(true));
    await userEvent.click(within(list).getByRole("button", { name: "Stop study late" }));
    await waitFor(() => expect(seen.some((s) => s.path === "/api/studies/late/stop")).toBe(true));
  });

  it("runs a bench on the chosen device, warns when the device is training, and lists the history", async () => {
    const seen = open({ "POST /api/bench": { status: 202, body: { ...job(8), id: 9, state: "queued" } } });
    const bench = await screen.findByRole("region", { name: "Bench" });
    expect(within(bench).getByText(/CPU run: nothing to compare against/)).toBeTruthy();
    expect(within(bench).getByText(Math.round(benchRow.tokens_per_sec).toLocaleString("en-US"))).toBeTruthy();
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
