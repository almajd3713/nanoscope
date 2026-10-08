import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import models from "../../test/fixtures/models.json";
import presets from "../../test/fixtures/presets.json";
import { mockApi } from "../../test/fetch";
import { Providers } from "../app/providers";
import { resetSettingsCache, setShowCommands } from "../app/settings";
import { RunForm } from "./RunForm";

type Sent = { model: string; preset: string; kwargs: Record<string, unknown>; seeds: unknown };

// A stand-in validator with the library's own wording for the two cases the tests need.
function validate(sent: unknown) {
  const s = sent as Sent;
  const problems: object[] = [];
  for (const key of Object.keys(s.kwargs)) {
    if (key === "lr") {
      problems.push({
        code: "unknown_keyword", field: "lr", hint: "model parameters: d_model, n_layers, n_heads",
        message: "'lr' is neither a parameter of GPT2.__init__ nor a preset field (see nanoscope.Preset)",
      });
    }
    if (key === "n_heads" && typeof s.kwargs[key] !== "number") {
      problems.push({ code: "wrong_type", field: "n_heads", hint: "a parameter of GPT2.__init__", message: `n_heads must be int, got str '${String(s.kwargs[key])}'` });
    }
  }
  const n = typeof s.seeds === "number" ? s.seeds : 1;
  const refs = problems.length ? [] : Array.from({ length: n }, (_, i) => `${s.preset}/gpt2-dc677f03/seed-${i}`);
  return { ok: problems.length === 0, problems, refs };
}

function renderForm(path = "/runs/new", routes: Parameters<typeof mockApi>[0] = {}) {
  const seen = mockApi({
    "GET /api/models": { body: models },
    "GET /api/presets": { body: presets },
    "POST /api/validate/run": { body: validate },
    ...routes,
  });
  render(
    <Providers>
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route path="/runs/new" element={<RunForm />} />
          <Route path="/runs/*" element={<p>run page</p>} />
          <Route path="/runs" element={<p>runs list</p>} />
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

describe("RunForm", () => {
  it("builds its fields from the model's constructor and the preset, and sends nothing for defaults", async () => {
    const seen = renderForm();
    await screen.findByRole("heading", { name: "New run" });
    expect((screen.getByLabelText("Model") as HTMLSelectElement).value).toBe("nanoscope.models.gpt2:GPT2");
    expect((screen.getByLabelText("d_model") as HTMLInputElement).value).toBe("128");
    // vocab_size and context_length come from the preset's data: shown disabled, never sent
    expect((screen.getByLabelText("vocab_size") as HTMLInputElement).disabled).toBe(true);
    expect((screen.getByLabelText("max_steps") as HTMLInputElement).value).toBe("500");
    expect(screen.getAllByText("Set by the preset's data.").length).toBe(2);
    await waitFor(() => expect(seen.some((r) => r.path === "/api/validate/run")).toBe(true));
    const sent = seen.filter((r) => r.path === "/api/validate/run").at(-1)?.body as Sent;
    expect(sent).toEqual({ model: "nanoscope.models.gpt2:GPT2", preset: "tinystories-5min", kwargs: {}, seeds: 1 });
  });

  it("sends only what you changed, says where the runs land, and shows the command", async () => {
    renderForm();
    await screen.findByRole("heading", { name: "New run" });
    const heads = screen.getByLabelText("n_heads");
    await userEvent.clear(heads);
    await userEvent.type(heads, "8");
    await userEvent.clear(screen.getByLabelText("Seeds"));
    await userEvent.type(screen.getByLabelText("Seeds"), "3");
    expect(screen.getByText("changed from 4")).toBeTruthy();
    expect(await screen.findByText("tinystories-5min/gpt2-dc677f03/seed-2")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Train 3 seeds" })).toBeTruthy();
    const cmd = screen.getByRole("region", { name: "Equivalent command" });
    expect(cmd.textContent).toContain("nanoscope run nanoscope.models.gpt2:GPT2 --preset tinystories-5min --seeds 3 --set n_heads=8");
  });

  it("shows a problem under its field in the library's words, and will not train until it is fixed", async () => {
    renderForm();
    await screen.findByRole("heading", { name: "New run" });
    await userEvent.type(screen.getAllByLabelText("Keyword")[0]!, "lr");
    await userEvent.type(screen.getAllByLabelText("Value")[0]!, "0.01");
    expect(await screen.findByText(/'lr' is neither a parameter of GPT2.__init__ nor a preset field/)).toBeTruthy();
    expect(screen.getByText("model parameters: d_model, n_layers, n_heads")).toBeTruthy();
    expect((screen.getByRole("button", { name: "Train" }) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByText("Fix 1 problem to train")).toBeTruthy();
    const heads = screen.getByLabelText("n_heads");
    await userEvent.clear(heads);
    await userEvent.type(heads, "x");
    expect(await screen.findByText("n_heads must be int, got str 'x'")).toBeTruthy();
    expect(screen.getByText("Fix 2 problems to train")).toBeTruthy();
  });

  it("trains one request per seed and opens the runs", async () => {
    const seen = renderForm("/runs/new", {
      "POST /api/runs": { status: 202, body: (sent: unknown) => ({ ref: `tinystories-5min/gpt2/seed-${(sent as { seed: number }).seed}`, state: "queued" }) },
    });
    await screen.findByRole("heading", { name: "New run" });
    await userEvent.clear(screen.getByLabelText("Seeds"));
    await userEvent.type(screen.getByLabelText("Seeds"), "2");
    await userEvent.click(await screen.findByRole("button", { name: "Train 2 seeds" }));
    await screen.findByText("runs list");
    const posts = seen.filter((r) => r.method === "POST" && r.path === "/api/runs").map((r) => r.body);
    expect(posts).toEqual([
      { model: "nanoscope.models.gpt2:GPT2", preset: "tinystories-5min", seed: 0, kwargs: {}, compile: false, wandb: false },
      { model: "nanoscope.models.gpt2:GPT2", preset: "tinystories-5min", seed: 1, kwargs: {}, compile: false, wandb: false },
    ]);
  });

  it("opens the run page for a single seed", async () => {
    renderForm("/runs/new", { "POST /api/runs": { status: 202, body: { ref: "tinystories-5min/gpt2/seed-0", state: "queued" } } });
    await userEvent.click(await screen.findByRole("button", { name: "Train" }));
    expect(await screen.findByText("run page")).toBeTruthy();
  });

  it("shows only the six main preset fields until you ask for all of them", async () => {
    renderForm();
    await screen.findByRole("heading", { name: "New run" });
    expect(screen.queryByLabelText("grad_clip")).toBeNull();
    await userEvent.click(screen.getByRole("button", { name: /All 26 preset fields/ }));
    expect(screen.getByLabelText("grad_clip")).toBeTruthy();
    const preset = screen.getByRole("region", { name: "Preset settings" });
    expect(within(preset).getAllByRole("textbox").length).toBeGreaterThan(20);
  });

  it("uses a checkbox for a boolean part of the model", async () => {
    renderForm();
    await screen.findByRole("heading", { name: "New run" });
    await userEvent.selectOptions(screen.getByLabelText("Model"), "nanoscope.models.modern:Modern");
    const qk = screen.getByRole("checkbox", { name: "qk_norm" }) as HTMLInputElement;
    expect(qk.checked).toBe(true);
    await userEvent.click(qk);
    expect(await screen.findByText("changed")).toBeTruthy();
  });
});
