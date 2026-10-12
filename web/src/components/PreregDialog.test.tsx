import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import realPreview from "../../test/fixtures/prereg-preview.json";
import { mockApi } from "../../test/fetch";
import { resetLevelCache } from "../app/level";
import { Providers } from "../app/providers";
import { diffRows, PreregDialog, type Preview } from "./PreregDialog";

// A real preview job result: the diff of a new record-mode study file (see the Python test that writes
// web/test/fixtures/prereg-preview.json).
const real = realPreview as Preview;
const DIFF = real.diff;
const preview = (over: Partial<Preview> = {}): Preview => ({ ...real, ...over });
const job = (id: number, state: string, result: unknown = null, error: string | null = null) => ({ id, kind: "x", state, lane: "interactive", owner: "local", payload: {}, result, error, created_at: 1 });

beforeEach(() => {
  localStorage.clear();
  resetLevelCache();
});
afterEach(() => vi.unstubAllGlobals());

function open(p: Preview, routes: Parameters<typeof mockApi>[0] = {}) {
  const seen = mockApi({
    "POST /api/studies/toy/preregister/preview": { status: 202, body: { study: "toy", job: job(1, "queued") } },
    "GET /api/jobs/1": { body: job(1, "done", p) },
    "POST /api/studies/toy/preregister/commit": { status: 202, body: { study: "toy", job: job(2, "queued") } },
    "GET /api/jobs/2": { body: job(2, "done", { commit: "a99fe9aa0e70b58a5d011120093e0935157b4f35" }) },
    ...routes,
  });
  render(
    <Providers>
      <PreregDialog name="toy" open onOpenChange={() => {}} />
    </Providers>,
  );
  return seen;
}

describe("diffRows", () => {
  it("dims the file headers and signs additions and deletions", () => {
    const rows = diffRows(DIFF);
    expect(rows.slice(0, 6).map((r) => r.kind)).toEqual(["head", "head", "head", "head", "head", "head"]);
    expect(rows.filter((r) => r.sign === "+")).toHaveLength(24);
    expect(rows.filter((r) => r.sign).map((r) => r.sign + r.text).slice(0, 3)).toEqual(["+schema = 1", '+name = "toy"', '+preset = "test-tiny"']);
    const edit = diffRows(["--- a/x", "+++ b/x", "@@ -1,2 +1,2 @@", " keep", '-mode = "explore"', '+mode = "record"', ""].join("\n"));
    expect(edit.map((r) => r.kind)).toEqual(["head", "head", "head", "ctx", "del", "add"]);
    expect(edit.filter((r) => r.sign).map((r) => r.sign + r.text)).toEqual(['−mode = "explore"', '+mode = "record"']);
  });
});

describe("PreregDialog", () => {
  it("shows the exact diff and message, then commits the previewed hash and shows the commit", async () => {
    const seen = open(preview());
    expect(await screen.findByText("exactly what the commit contains · 24 lines added, 0 removed")).toBeTruthy();
    expect(screen.getByText(/Committed-via: nanoscope/)).toBeTruthy();
    expect(screen.getByText(real.hash)).toBeTruthy();
    expect(screen.getByText(/A reader cannot tell how carefully this diff was/)).toBeTruthy();
    await userEvent.click(screen.getByRole("button", { name: "Commit preregistration" }));
    expect(await screen.findByText("a99fe9aa0e70b58a5d011120093e0935157b4f35")).toBeTruthy();
    expect(screen.getByText("Preregistration committed")).toBeTruthy();
    expect(seen.find((s) => s.path === "/api/studies/toy/preregister/commit")!.body).toEqual({ preview_hash: real.hash });
  });

  it("cannot commit while other files are changed, and lists them", async () => {
    open(preview({ unrelated: ["notes/ideas.md"] }));
    expect(await screen.findByText("Other files are changed")).toBeTruthy();
    expect(screen.getByText(/notes\/ideas\.md/)).toBeTruthy();
    expect((screen.getByRole("button", { name: "Commit preregistration" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("says when git has no identity or there is nothing to commit", async () => {
    open(preview({ identity: false, nothing_to_commit: true, diff: "" }));
    expect(await screen.findByText("Git has no identity")).toBeTruthy();
    expect(screen.getByText("Nothing to commit")).toBeTruthy();
    expect((screen.getByRole("button", { name: "Commit preregistration" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("shows the worker's refusal when the preview changed before the commit", async () => {
    open(preview(), { "GET /api/jobs/2": { body: job(2, "failed", null, "the preview changed since you looked; preview again") } });
    await userEvent.click(await screen.findByRole("button", { name: "Commit preregistration" }));
    await waitFor(() => expect(screen.getByText(/the preview changed since you looked/)).toBeTruthy());
  });
});
