import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import checkpoints from "../../test/fixtures/checkpoints.json";
import latest from "../../test/fixtures/inspect-latest.json";
import step5 from "../../test/fixtures/inspect-5.json";
import runDetail from "../../test/fixtures/run-gpt2.json";
import { mockApi } from "../../test/fetch";
import { Providers } from "../app/providers";
import { resetSettingsCache, setShowCommands } from "../app/settings";
import { Inspect } from "./Inspect";

const REF = latest.ref;
const job = (id: number, state: string, result: unknown = null, error: string | null = null) => ({
  id, kind: "inspect", state, lane: "interactive", owner: "local", payload: {}, result, error, created_at: 1,
});

// The worker's answer depends on the step asked for: the newest when none is given.
const jobFor = (sent: unknown) => {
  const step = (sent as { step?: number }).step;
  return job(step === undefined ? 1 : step, "queued");
};

function open(routes: Parameters<typeof mockApi>[0] = {}) {
  const seen = mockApi({
    [`GET /api/runs/${REF}`]: { body: runDetail },
    [`GET /api/runs/${REF}/checkpoints`]: { body: checkpoints },
    [`POST /api/runs/${REF}/inspect`]: { status: 202, body: jobFor },
    "GET /api/jobs/1": { body: job(1, "done", latest) },
    "GET /api/jobs/5": { body: job(5, "done", step5) },
    "GET /api/jobs/10": { body: job(10, "done", { ...latest, step: 10 }) },
    ...routes,
  });
  render(
    <Providers>
      <MemoryRouter initialEntries={[`/inspect/${REF}`]}>
        <Routes>
          <Route path="/inspect/*" element={<Inspect />} />
        </Routes>
      </MemoryRouter>
    </Providers>,
  );
  return seen;
}

const posts = (seen: ReturnType<typeof mockApi>) => seen.filter((r) => r.method === "POST").map((r) => r.body);

beforeEach(() => {
  localStorage.clear();
  act(() => {
    resetSettingsCache();
    setShowCommands(true);
  });
});
afterEach(() => vi.unstubAllGlobals());

describe("Inspect", () => {
  it("runs the default prompt at the latest checkpoint and draws a map per head and the lens", async () => {
    const seen = open();
    await screen.findByRole("heading", { name: "Attention" });
    expect(posts(seen)).toEqual([{ prompt: "Once upon a time", top_k: 5 }]);
    for (const module of ["blocks.0.attn", "blocks.1.attn"]) {
      for (const head of [0, 1]) expect(screen.getByRole("button", { name: `${module} head ${head}` })).toBeTruthy();
    }
    expect(screen.getByRole("table", { name: "Logit lens by layer" })).toBeTruthy();
    expect(screen.getByRole("columnheader", { name: "blocks.1 = output" })).toBeTruthy();
    // the first lens row: the layers' guesses come from the report
    const first = latest.lens.layers[0]!.positions[0]!.top[0]!;
    expect(screen.getAllByText(JSON.stringify(first.text)).length).toBeGreaterThan(0);
    expect(screen.getByText("Selected")).toBeTruthy();
  });

  it("selects a head and a row", async () => {
    open();
    await screen.findByRole("heading", { name: "Attention" });
    await userEvent.click(screen.getByRole("button", { name: "blocks.0.attn head 1" }));
    expect(screen.getByRole("button", { name: "blocks.0.attn head 1" }).getAttribute("aria-pressed")).toBe("true");
    expect(screen.getByRole("button", { name: "blocks.1.attn head 0" }).getAttribute("aria-pressed")).toBe("false");
    const token = JSON.stringify(latest.tokens[1]!.text);
    const label = screen.getAllByRole("button", { name: token }).find((b) => b.getAttribute("aria-pressed") !== null)!;
    await userEvent.click(label);
    const selected = screen.getByRole("region", { name: "Selected" });
    expect(within(selected).getByText("blocks.0.attn")).toBeTruthy();
    expect(within(selected).getAllByRole("row").length).toBe(3); // the header and the row's 2 weights
  });

  it("changing the checkpoint asks again with the same prompt", async () => {
    const seen = open();
    await screen.findByRole("heading", { name: "Attention" });
    await userEvent.click(screen.getByRole("radio", { name: "5" }));
    await waitFor(() => expect(posts(seen)).toContainEqual({ prompt: "Once upon a time", top_k: 5, step: 5 }));
    await waitFor(() => expect(screen.getByRole("radio", { name: "5" }).getAttribute("aria-checked")).toBe("true"));
  });

  it("shows the job's refusal word for word and keeps the previous result", async () => {
    const refusal = "ValueError: the prompt is 66 tokens; inspect takes at most 64 (attention maps grow with the square of the length)";
    open();
    await screen.findByRole("heading", { name: "Attention" });
    // the next job fails
    mockApi({
      [`GET /api/runs/${REF}`]: { body: runDetail },
      [`GET /api/runs/${REF}/checkpoints`]: { body: checkpoints },
      [`POST /api/runs/${REF}/inspect`]: { status: 202, body: job(7, "queued") },
      "GET /api/jobs/7": { body: job(7, "failed", null, refusal) },
    });
    await userEvent.clear(screen.getByLabelText("Prompt"));
    await userEvent.type(screen.getByLabelText("Prompt"), "a long one");
    await userEvent.click(screen.getByRole("button", { name: "Inspect" }));
    expect(await screen.findByText("The inspect job failed")).toBeTruthy();
    expect(screen.getByText(refusal)).toBeTruthy();
    expect(screen.getByRole("heading", { name: "Logit lens" })).toBeTruthy();
    expect(screen.getByText(/still show the previous result/)).toBeTruthy();
  });

  it("across steps runs one job per archived step", async () => {
    const seen = open();
    await screen.findByRole("heading", { name: "Attention" });
    await userEvent.click(screen.getByRole("radio", { name: "Across steps" }));
    expect(await screen.findByRole("heading", { name: /Across steps/ })).toBeTruthy();
    await waitFor(() => expect(screen.getByText(/2 of 2 jobs done/)).toBeTruthy());
    expect(posts(seen)).toContainEqual({ prompt: "Once upon a time", top_k: 5, step: 5 });
    expect(posts(seen)).toContainEqual({ prompt: "Once upon a time", top_k: 5, step: 10 });
  });

  it("says what a single kept checkpoint means and how to keep more", async () => {
    open({ [`GET /api/runs/${REF}/checkpoints`]: { body: [checkpoints.at(-1)] } });
    await screen.findByRole("heading", { name: "Attention" });
    expect(screen.getByText(/kept only its latest checkpoint, step 20/)).toBeTruthy();
    expect(screen.queryByRole("radio", { name: "Across steps" })).toBeNull();
  });
});
