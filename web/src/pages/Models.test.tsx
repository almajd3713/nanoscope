import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import blocks from "../../test/fixtures/blocks-guided.json";
import models from "../../test/fixtures/models.json";
import { installEventSource } from "../../test/eventsource";
import { mockApi } from "../../test/fetch";
import { resetLevelCache, setLevel } from "../app/level";
import { Providers } from "../app/providers";
import { Models } from "./Models";

const REF = "lessons/foundations/01-bigram/starter.py:MyBigram";
const mine = {
  name: "ScaledMLP", family: "mlp", args: [], user: true, certified: false, reference: "naive_scaled",
  certification: { state: "stale" }, lock: { lockable: false, locked: false, lesson: null, how: null }, file: "/ws/blocks.py", line: 4,
};
const trace = (state: string, extra: object = {}) => ({
  id: 7, kind: "describe", state, lane: "interactive", owner: "local", payload: { model: `/ws/workspace/${REF}`, preset: "tinystories-5min", kwargs: {} },
  result: null, error: null, created_at: 1, ...extra,
});

beforeEach(() => {
  localStorage.clear();
  resetLevelCache();
  installEventSource();
});
afterEach(() => vi.unstubAllGlobals());

function open(routes: Parameters<typeof mockApi>[0] = {}) {
  const seen = mockApi({
    "GET /api/models": { body: models },
    "GET /api/jobs": { body: [] },
    "GET /api/blocks": { body: { ...blocks, blocks: [...blocks.blocks, mine] } },
    ...routes,
  });
  render(
    <Providers>
      <MemoryRouter>
        <Models />
      </MemoryRouter>
    </Providers>,
  );
  return seen;
}

describe("Models page", () => {
  it("lists the workspace's models, not the shipped ones, and each opens the model page", async () => {
    open();
    const link = await screen.findByRole("link", { name: "MyBigram" });
    expect(link.getAttribute("href")).toBe("/model/lessons/foundations/01-bigram/starter.py?class=MyBigram");
    expect(screen.queryByText("Modern")).toBeNull();
    expect(screen.getByText("not traced")).toBeTruthy();
  });

  it("shows params and FLOPs from the last trace", async () => {
    open({
      "GET /api/jobs": {
        body: [trace("failed", { id: 5, error: "old" }), trace("done", { result: { params: { total: 1250000, non_embedding: 726000 }, flops_per_token: 6500000 } })],
      },
    });
    const row = (await screen.findByRole("link", { name: "MyBigram" })).closest("tr")!;
    expect(within(row).getByText("1.25M")).toBeTruthy();
    expect(within(row).getByText("6.5M")).toBeTruthy();
    expect(within(row).getByText("traced")).toBeTruthy();
  });

  it("a trace is a job: the button queues one and the state follows it", async () => {
    const seen = open({
      [`POST /api/models/${REF}/describe`]: { status: 202, body: trace("queued") },
    });
    await userEvent.click(await screen.findByRole("button", { name: "Trace" }));
    await waitFor(() => expect(seen.some((s) => s.method === "POST" && s.path.endsWith("/describe"))).toBe(true));
  });

  it("a failed trace says why, word for word", async () => {
    open({ "GET /api/jobs": { body: [trace("failed", { error: "ShapeError: Block.attn: bad shape (/ws/m.py:9)" })] } });
    expect(await screen.findByText("ShapeError: Block.attn: bad shape (/ws/m.py:9)")).toBeTruthy();
    expect(screen.getByText("failed")).toBeTruthy();
  });

  it("hides your blocks below Extend", async () => {
    open();
    await screen.findByRole("link", { name: "MyBigram" });
    expect(screen.queryByRole("region", { name: "Your blocks" })).toBeNull();
  });

  it("at Extend lists your blocks with their certification and certifies one as a job", async () => {
    act(() => setLevel("Extend"));
    const seen = open({ "POST /api/blocks/ScaledMLP/certify": { status: 202, body: trace("queued", { kind: "certify" }) } });
    const region = await screen.findByRole("region", { name: "Your blocks" });
    expect(await within(region).findByText("ScaledMLP")).toBeTruthy();
    expect(within(region).getByText("changed since")).toBeTruthy();
    await userEvent.click(within(region).getByRole("button", { name: "Certify" }));
    await waitFor(() => expect(seen.some((s) => s.path === "/api/blocks/ScaledMLP/certify")).toBe(true));
    expect(await within(region).findByText(/The check is queued/)).toBeTruthy();
  });

  it("explains the first-time empty state", async () => {
    open({ "GET /api/models": { body: models.filter((m) => m.shipped) } });
    expect(await screen.findByText(/No model files yet/)).toBeTruthy();
  });
});
