import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { mockApi } from "../../test/fetch";
import { resetLevelCache } from "../app/level";
import { Providers } from "../app/providers";
import { CardExport, PROJECT_CARDS } from "./CardExport";

const content = JSON.stringify({ schema: 1, study: "toy", finals: {}, exported_at: "2026-10-08T21:04:26+00:00" }, null, 2);
const plan = { repo: PROJECT_CARDS, repo_type: "dataset", path_in_repo: "cards/toy.json", content, commit_message: "nanoscope: ablation card for toy", exported_at: "2026-10-08T21:04:26+00:00" };
const job = (state: string, result: unknown = null) => ({ id: 3, kind: "card-push", state, lane: "interactive", owner: "local", payload: {}, result, error: null, created_at: 1 });

beforeEach(() => {
  localStorage.clear();
  resetLevelCache();
});
afterEach(() => vi.unstubAllGlobals());

function open() {
  const seen = mockApi({
    "GET /api/studies/toy/card/upload": { body: plan },
    "POST /api/studies/toy/card/push": { status: 202, body: { study: "toy", repo: PROJECT_CARDS, path_in_repo: "cards/toy.json", job: job("queued") } },
    "GET /api/jobs/3": { body: job("done", { repo: PROJECT_CARDS, path_in_repo: "cards/toy.json" }) },
  });
  render(
    <Providers>
      <CardExport name="toy" open onOpenChange={() => {}} />
    </Providers>,
  );
  return seen;
}

describe("CardExport", () => {
  it("shows the whole card and offers it as a download, with no push on", async () => {
    const seen = open();
    const shown = await screen.findByLabelText("card.json");
    expect(shown.textContent).toBe(content);
    expect(screen.getByText(/card\.v1 · \d+ lines/)).toBeTruthy();
    const link = screen.getByRole("link", { name: /Download card\.json/ }) as HTMLAnchorElement;
    expect(decodeURIComponent(link.getAttribute("href")!.split(",")[1]!)).toBe(content);
    expect((screen.getByLabelText(/Also push it to a public Hugging Face dataset/) as HTMLInputElement).checked).toBe(false);
    expect(screen.queryByRole("button", { name: "Push card" })).toBeNull();
    expect(seen.some((s) => s.method === "POST")).toBe(false);
  });

  it("says exactly what uploads, and pushes the very card that was shown", async () => {
    const seen = open();
    await screen.findByLabelText("card.json");
    await userEvent.click(screen.getByLabelText(/Also push it to a public Hugging Face dataset/));
    expect(await screen.findByText("What uploads")).toBeTruthy();
    expect(screen.getByText("cards/toy.json")).toBeTruthy();
    expect(screen.getByText("nanoscope: ablation card for toy")).toBeTruthy();
    expect(screen.getByText(/Anyone can read a public dataset/)).toBeTruthy();
    expect(screen.getByText(/the text above, byte for byte/)).toBeTruthy();
    await userEvent.click(screen.getByRole("button", { name: "Push card" }));
    expect(await screen.findByRole("status")).toBeTruthy();
    expect(seen.find((s) => s.method === "POST")!.body).toEqual({ repo: PROJECT_CARDS, exported_at: plan.exported_at });
  });

  it("will not push to something that is not a dataset id", async () => {
    open();
    await screen.findByLabelText("card.json");
    await userEvent.click(screen.getByLabelText(/Also push it to a public Hugging Face dataset/));
    const field = await screen.findByLabelText("Dataset");
    await userEvent.clear(field);
    await userEvent.type(field, "nonsense");
    await waitFor(() => expect((screen.getByRole("button", { name: "Push card" }) as HTMLButtonElement).disabled).toBe(true));
    expect(screen.getByText("Use user/name")).toBeTruthy();
  });
});
