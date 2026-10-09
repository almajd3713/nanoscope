import { act, render, screen } from "@testing-library/react";
import { axe } from "vitest-axe";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import checkpoints from "../test/fixtures/checkpoints.json";
import compare from "../test/fixtures/compare.json";
import curricula from "../test/fixtures/curricula.json";
import failed from "../test/fixtures/run-failed.json";
import inspectReport from "../test/fixtures/inspect-latest.json";
import jobs from "../test/fixtures/jobs.json";
import lesson04 from "../test/fixtures/lesson-04.json";
import models from "../test/fixtures/models.json";
import presets from "../test/fixtures/presets.json";
import runs from "../test/fixtures/runs.json";
import samples from "../test/fixtures/samples.json";
import unlocksFirst from "../test/fixtures/unlocks-first-run.json";
import { installEventSource } from "../test/eventsource";
import { mockApi } from "../test/fetch";
import { AppRoutes } from "./app/routes";
import { resetLevelCache, setLevel } from "./app/level";
import { Providers } from "./app/providers";
import { resetSettingsCache, setShowCommands } from "./app/settings";
import { Login } from "./pages/Login";

// Every main page through axe, with the command blocks on (the most markup) and at Tinker (the
// most controls). jsdom has no layout, so contrast is checked by the token pairs instead.
const guided = { ...unlocksFirst, first_run: false, policy: "guided" };
const REF = "tinystories-5min/mybigram/seed-0";

const API = {
  "GET /api/learn/unlocks": { body: guided },
  "GET /api/learn/progress": { body: { schema: 1, nanoscope: "0.4.0", lessons: {} } },
  "GET /api/curricula": { body: curricula },
  "GET /api/curricula/foundations/04-multi-head": { body: { ...lesson04, state: "started" } },
  "GET /api/files/lessons/foundations/04-multi-head/starter.py": { body: { path: "x", content: "x = 1\n", etag: "e" } },
  "GET /api/jobs": { body: jobs },
  "GET /api/workers": { body: [] },
  "GET /api/runs": { body: runs },
  [`GET /api/runs/${REF}`]: { body: failed },
  [`GET /api/runs/${REF}/metrics`]: { body: { rows: [], last_step: 0 } },
  [`GET /api/runs/${REF}/samples`]: { body: samples },
  [`GET /api/runs/${inspectReport.ref}`]: { body: failed },
  [`GET /api/runs/${inspectReport.ref}/checkpoints`]: { body: checkpoints },
  [`POST /api/runs/${inspectReport.ref}/inspect`]: { status: 202, body: { id: 1 } },
  "GET /api/jobs/1": { body: { id: 1, kind: "inspect", state: "done", result: inspectReport, error: null } },
  "GET /api/models": { body: models },
  "GET /api/presets": { body: presets },
  "POST /api/validate/run": { body: { ok: true, problems: [], refs: [`${REF}`] } },
  "POST /api/compare": { body: compare },
};

async function page(path: string, ready: string | RegExp) {
  mockApi(API);
  const view = render(
    <Providers>
      <MemoryRouter initialEntries={[path]}>
        <AppRoutes />
      </MemoryRouter>
    </Providers>,
  );
  await screen.findByRole("heading", { name: ready });
  return view;
}

beforeEach(() => {
  localStorage.clear();
  act(() => {
    resetSettingsCache();
    resetLevelCache();
    setShowCommands(true);
    setLevel("Tinker");
  });
  installEventSource();
});
afterEach(() => vi.unstubAllGlobals());

const check = async (container: HTMLElement) => {
  const results = await axe(container, { rules: { "color-contrast": { enabled: false } } });
  expect(results.violations.map((v) => `${v.id}: ${v.help} (${v.nodes.map((n) => n.target.join(" ")).join(", ")})`)).toEqual([]);
};

describe("accessibility (axe)", () => {
  it("lessons list", async () => {
    const { container } = await page("/learn", "Lessons");
    await screen.findByText("A bigram language model");
    await check(container);
  });

  it("a lesson", async () => {
    const { container } = await page("/learn/foundations/04-multi-head", "Multi-head attention");
    await check(container);
  });

  it("a run", async () => {
    const { container } = await page(`/runs/${REF}`, REF);
    await check(container);
  });

  it("inspect", async () => {
    const { container } = await page(`/inspect/${inspectReport.ref}`, "Attention");
    await check(container);
  });

  it("the runs list", async () => {
    const { container } = await page("/runs", "Runs");
    await screen.findByRole("table");
    await check(container);
  });

  it("the run form", async () => {
    const { container } = await page("/runs/new", "New run");
    await check(container);
  });

  it("compare", async () => {
    const { container } = await page("/compare?runs=bigram,modern,gpt2&preset=tinystories-5min", "Compare");
    await screen.findByRole("table");
    await check(container);
  });

  it("components", async () => {
    const { container } = await page("/components", "Components");
    await screen.findByRole("table");
    await check(container);
  });

  it("onboarding", async () => {
    const { container } = await page("/welcome", "How do you want to start?");
    await check(container);
  });

  it("login", async () => {
    mockApi(API);
    const { container } = render(<Login />);
    await check(container);
  });
});
