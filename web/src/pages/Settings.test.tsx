import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import data from "../../test/fixtures/data.json";
import server from "../../test/fixtures/settings.json";
import unlocksFirst from "../../test/fixtures/unlocks-first-run.json";
import { mockApi } from "../../test/fetch";
import { resetLevelCache } from "../app/level";
import { Providers } from "../app/providers";
import { resetSettingsCache } from "../app/settings";
import { Settings } from "./Settings";

const guided = { ...unlocksFirst, first_run: false, policy: "guided" };

function renderSettings(routes: Parameters<typeof mockApi>[0] = {}) {
  const seen = mockApi({
    "GET /api/learn/unlocks": { body: guided },
    "GET /api/data": { body: data },
    "GET /api/settings": { body: server },
    "GET /api/jobs": { body: [] },
    "GET /api/workers": { body: [] },
    ...routes,
  });
  render(
    <Providers>
      <MemoryRouter>
        <Settings />
      </MemoryRouter>
    </Providers>,
  );
  return seen;
}

beforeEach(() => {
  localStorage.clear();
  document.documentElement.removeAttribute("data-theme");
  vi.stubGlobal("matchMedia", () => ({ matches: false, addEventListener() {}, removeEventListener() {} }));
  act(() => {
    resetSettingsCache();
    resetLevelCache();
  });
});
afterEach(() => vi.unstubAllGlobals());

describe("Settings", () => {
  it("changes the theme, the level and the command toggle in this browser only", async () => {
    const seen = renderSettings();
    await screen.findByRole("heading", { name: "Settings" });
    await userEvent.click(screen.getByRole("radio", { name: "Dark" }));
    expect(document.documentElement.getAttribute("data-theme")).toBe("dark");
    await userEvent.click(screen.getByRole("radio", { name: /Research/ }));
    expect(localStorage.getItem("nanoscope.level")).toBe("Research");
    const toggle = screen.getByRole("checkbox", { name: /Show command-line equivalents/ }) as HTMLInputElement;
    expect(toggle.checked).toBe(false); // off by default
    await userEvent.click(toggle);
    expect(localStorage.getItem("nanoscope.showCommands")).toBe("1");
    // none of that is sent to the server
    expect(seen.filter((r) => r.method === "POST")).toEqual([]);
  });

  it("shows the gating policy and changes it through the API", async () => {
    const seen = renderSettings({ "POST /api/learn/policy": { body: { ...guided, policy: "open" } } });
    const guidedRadio = (await screen.findByRole("radio", { name: /^Guided/ })) as HTMLInputElement;
    await waitFor(() => expect(guidedRadio.checked).toBe(true));
    await userEvent.click(screen.getByRole("radio", { name: /^Open/ }));
    await waitFor(() => expect(seen.some((r) => r.method === "POST")).toBe(true));
    expect(seen.find((r) => r.method === "POST")?.body).toEqual({ policy: "open" });
  });

  it("lists each preset's data and prepares one as a job", async () => {
    const seen = renderSettings({ "POST /api/data/tinystories-30min/prepare": { status: 202, body: { id: 9, kind: "prepare-data", state: "queued" } } });
    await screen.findByText("tinystories-5min");
    const table = screen.getByRole("table");
    const done = within(table).getByText("tinystories-5min").closest("tr")!;
    expect(within(done).getByText("done")).toBeTruthy();
    expect(within(done).queryByRole("button", { name: "Prepare" })).toBeNull();
    const todo = within(table).getByText("tinystories-30min").closest("tr")!;
    expect(within(todo).getByText("not prepared")).toBeTruthy();
    await userEvent.click(within(todo).getByRole("button", { name: "Prepare" }));
    await waitFor(() => expect(seen.some((r) => r.path === "/api/data/tinystories-30min/prepare")).toBe(true));
  });

  it("shows what the server reports, with secrets only as set or not set", async () => {
    renderSettings();
    expect(await screen.findByText(`nanoscope ${server.version}`)).toBeTruthy();
    expect(screen.getByText(server.workspace)).toBeTruthy();
    expect(screen.getByText(`${server.listening.host}:${server.listening.port}`)).toBeTruthy();
    expect(screen.getByText("this machine only")).toBeTruthy();
    expect(screen.getByText("Offline jobs").nextElementSibling!.textContent).toBe("off");
    expect(screen.getByText("Hugging Face token").nextElementSibling!.textContent).toBe("set for the workers");
    expect(document.body.textContent).not.toMatch(/hf_[A-Za-z0-9]{10,}/);
  });

  it("says a secret is not set when no worker has it", async () => {
    renderSettings({ "GET /api/settings": { body: { ...server, workers: [{ ...server.workers[0], secrets: { HF_TOKEN: false, WANDB_API_KEY: false } }] } } });
    await screen.findByText("Hugging Face token");
    expect(screen.getByText("Hugging Face token").nextElementSibling!.textContent).toBe("not set for the workers");
    expect(screen.getByText(/W&B key/).nextElementSibling!.textContent).toBe("not set for the workers");
  });

  it("shows the library's error for a section that fails, and keeps the rest", async () => {
    renderSettings({
      "GET /api/data": { status: 422, problem: true, body: { title: "Cannot be done as asked", status: 422, detail: "cannot read the prepare files" } },
    });
    expect((await screen.findByRole("alert")).textContent).toContain("cannot read the prepare files");
    expect(await screen.findByText(`nanoscope ${server.version}`)).toBeTruthy();
  });
});
