import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes, useParams } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import curricula from "../../test/fixtures/curricula.json";
import lesson01 from "../../test/fixtures/lesson-01.json";
import { mockApi } from "../../test/fetch";
import { Providers } from "../app/providers";
import { resetSettingsCache } from "../app/settings";
import { Lesson } from "./Lesson";

function ModelPage() {
  const { "*": file } = useParams();
  return <p>model page {file}</p>;
}

function RunPage() {
  const { "*": ref } = useParams();
  return <p>run page {ref}</p>;
}

function renderPage(routes: Parameters<typeof mockApi>[0]) {
  const seen = mockApi({ "GET /api/curricula": { body: curricula }, ...routes });
  render(
    <Providers>
      <MemoryRouter initialEntries={["/learn/foundations/01-bigram"]}>
        <Routes>
          <Route path="/learn/:path/:lesson" element={<Lesson />} />
          <Route path="/runs/*" element={<RunPage />} />
          <Route path="/model/*" element={<ModelPage />} />
        </Routes>
      </MemoryRouter>
    </Providers>,
  );
  return seen;
}

const STARTER = "class MyBigram(nn.Module):\n    pass\n";

beforeEach(() => {
  localStorage.clear();
  act(() => resetSettingsCache());
});
afterEach(() => vi.unstubAllGlobals());

describe("Lesson actions", () => {
  it("starting the lesson opens its starter file on the model page", async () => {
    let state = "not-started";
    const seen = renderPage({
      "GET /api/curricula/foundations/01-bigram": { body: () => ({ ...lesson01, state }) },
      "POST /api/curricula/foundations/01-bigram/start": {
        body: () => {
          state = "started";
          return { lesson: "foundations/01-bigram", copied: ["lessons/foundations/01-bigram/starter.py"], kept: [], state, policy: "guided", first_start: false, workspace: "lessons/foundations/01-bigram" };
        },
      },
    });
    await screen.findByRole("heading", { name: lesson01.title });
    expect((screen.getByRole("button", { name: "Train" }) as HTMLButtonElement).disabled).toBe(true);
    await userEvent.click(screen.getByRole("button", { name: "Start lesson" }));
    expect(await screen.findByText("model page lessons/foundations/01-bigram/starter.py")).toBeTruthy();
    expect(seen.some((r) => r.method === "POST" && r.path.endsWith("/start"))).toBe(true);
  });

  it("a started lesson shows its starter file and a way into the model page", async () => {
    renderPage({
      "GET /api/curricula/foundations/01-bigram": { body: { ...lesson01, state: "started" } },
      "GET /api/files/lessons/foundations/01-bigram/starter.py": {
        body: { path: "lessons/foundations/01-bigram/starter.py", content: STARTER, etag: "e1" },
      },
    });
    expect(await screen.findByText(/class MyBigram/)).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Start lesson" })).toBeNull();
    const open = screen.getByRole("link", { name: "Open in the model page" });
    expect(open.getAttribute("href")).toBe("/model/lessons/foundations/01-bigram/starter.py");
    await userEvent.click(open);
    expect(await screen.findByText("model page lessons/foundations/01-bigram/starter.py")).toBeTruthy();
  });

  it("Train runs the lesson's model with the level-0 defaults and opens the run page", async () => {
    const seen = renderPage({
      "GET /api/curricula/foundations/01-bigram": { body: { ...lesson01, state: "started" } },
      "GET /api/files/lessons/foundations/01-bigram/starter.py": {
        body: { path: "lessons/foundations/01-bigram/starter.py", content: STARTER, etag: "e1" },
      },
      "POST /api/runs": { status: 202, body: { ref: "tinystories-5min/lessons/foundations/01-bigram/starter/MyBigram/seed-0" } },
    });
    await userEvent.click(await screen.findByRole("button", { name: "Train" }));
    await waitFor(() => expect(screen.getByText(/run page tinystories-5min/)).toBeTruthy());
    const post = seen.find((r) => r.path === "/api/runs");
    expect(post?.body).toEqual({
      model: "lessons/foundations/01-bigram/starter.py:MyBigram",
      preset: "tinystories-5min",
      seed: 0,
      kwargs: {},
      compile: false,
      wandb: false,
    });
  });

  it("shows the library's refusal word for word", async () => {
    renderPage({
      "GET /api/curricula/foundations/01-bigram": { body: { ...lesson01, state: "started" } },
      "GET /api/files/lessons/foundations/01-bigram/starter.py": {
        body: { path: "lessons/foundations/01-bigram/starter.py", content: STARTER, etag: "e1" },
      },
      "POST /api/runs": {
        status: 422,
        problem: true,
        body: { title: "Cannot be done as asked", status: 422, detail: "unknown model 'x'; available: none" },
      },
    });
    await userEvent.click(await screen.findByRole("button", { name: "Train" }));
    expect((await screen.findByRole("alert")).textContent).toContain("unknown model 'x'; available: none");
  });
});
