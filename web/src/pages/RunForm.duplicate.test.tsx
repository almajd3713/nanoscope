import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import models from "../../test/fixtures/models.json";
import presets from "../../test/fixtures/presets.json";
import gpt2 from "../../test/fixtures/run-gpt2.json";
import { mockApi } from "../../test/fetch";
import { Providers } from "../app/providers";
import { resetSettingsCache, setShowCommands } from "../app/settings";
import { RunForm } from "./RunForm";

const REF = "tinystories-5min/gpt2/seed-0";

function entry(seed: number) {
  return { ref: `tinystories-5min/gpt2/seed-${seed}`, state: "done", step: 500, max_steps: 500, val_bpb: 1.3, updated: null, stale: false, error: null };
}

function renderDuplicate() {
  // the original was trained with a non-default context length and d_model of 128
  const original = {
    ...gpt2,
    config: { ...gpt2.config, model: { ...gpt2.config.model, kwargs: { ...gpt2.config.model.kwargs, n_layers: 6 } } },
  };
  const seen = mockApi({
    "GET /api/models": { body: models },
    "GET /api/presets": { body: presets },
    [`GET /api/runs/${REF}`]: { body: original },
    "GET /api/runs": { body: [entry(0), entry(1), entry(2)] },
    "POST /api/validate/run": { body: (sent: unknown) => ({ ok: true, problems: [], refs: [`echo ${JSON.stringify((sent as { kwargs: object }).kwargs)}`] }) },
  });
  render(
    <Providers>
      <MemoryRouter initialEntries={[`/runs/new?from=${encodeURIComponent(REF)}`]}>
        <Routes>
          <Route path="/runs/new" element={<RunForm />} />
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

describe("RunForm duplicate", () => {
  it("starts from the original run's config, with its model, preset and seed count", async () => {
    renderDuplicate();
    expect(await screen.findByRole("heading", { name: "Duplicate and change one thing" })).toBeTruthy();
    expect((screen.getByLabelText("Model") as HTMLSelectElement).value).toBe("nanoscope.models.gpt2:GPT2");
    expect((screen.getByLabelText("n_layers") as HTMLInputElement).value).toBe("6");
    expect((screen.getByLabelText("Seeds") as HTMLInputElement).value).toBe("3");
    expect(screen.getByRole("region", { name: "Changes" }).textContent).toContain("0 changes");
    // what the original did differently from the defaults is sent, so the same seeds repeat it
    expect(await screen.findByText('echo {"n_layers":6}')).toBeTruthy();
  });

  it("counts one change against the original and sends it with everything the original had", async () => {
    renderDuplicate();
    await screen.findByRole("heading", { name: "Duplicate and change one thing" });
    const heads = screen.getByLabelText("n_heads");
    await userEvent.clear(heads);
    await userEvent.type(heads, "8");
    const changes = screen.getByRole("region", { name: "Changes" });
    expect(within(changes).getByText("1 change")).toBeTruthy();
    expect(changes.textContent).toContain("n_heads");
    expect(changes.textContent).toContain("4");
    expect(changes.textContent).toContain("8");
    expect(await screen.findByText('echo {"n_layers":6,"n_heads":8}')).toBeTruthy();
  });

  it("resets to the original", async () => {
    renderDuplicate();
    await screen.findByRole("heading", { name: "Duplicate and change one thing" });
    await userEvent.clear(screen.getByLabelText("n_heads"));
    await userEvent.type(screen.getByLabelText("n_heads"), "8");
    await userEvent.click(screen.getByRole("button", { name: "Reset to the original" }));
    expect((screen.getByLabelText("n_heads") as HTMLInputElement).value).toBe("4");
    await waitFor(() => expect(screen.getByRole("region", { name: "Changes" }).textContent).toContain("0 changes"));
  });

  it("links back to the original run", async () => {
    renderDuplicate();
    const link = await screen.findByRole("link", { name: REF });
    expect(link.getAttribute("href")).toBe(`/runs/${REF}`);
  });
});
