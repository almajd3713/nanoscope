import { act, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import curricula from "../../test/fixtures/curricula.json";
import lesson04 from "../../test/fixtures/lesson-04.json";
import { mockApi } from "../../test/fetch";
import { Markdown } from "../components/Markdown";
import { Providers } from "../app/providers";
import { resetSettingsCache } from "../app/settings";
import { Lesson } from "./Lesson";

function renderPage() {
  mockApi({
    "GET /api/curricula/foundations/04-multi-head": { body: lesson04 },
    "GET /api/curricula": { body: curricula },
  });
  return render(
    <Providers>
      <MemoryRouter initialEntries={["/learn/foundations/04-multi-head"]}>
        <Routes>
          <Route path="/learn/:path/:lesson" element={<Lesson />} />
        </Routes>
      </MemoryRouter>
    </Providers>,
  );
}

beforeEach(() => {
  localStorage.clear();
  act(() => resetSettingsCache());
});
afterEach(() => vi.unstubAllGlobals());

describe("Lesson", () => {
  it("shows the title, state, intro and what passing unlocks", async () => {
    renderPage();
    expect(await screen.findByRole("heading", { name: "Multi-head attention" })).toBeTruthy();
    expect(screen.getByText("Passing unlocks")).toBeTruthy();
    expect(screen.getAllByText("block:Attention").length).toBeGreaterThan(0);
    expect(screen.getByText("cpu 0.5 min")).toBeTruthy();
    expect(screen.getByText(/split the vector into/)).toBeTruthy();
    expect(screen.getByText("torch.nn.MultiheadAttention, F.scaled_dot_product_attention")).toBeTruthy();
  });

  it("says the lock is advice and shows the outline with the current lesson marked", async () => {
    renderPage();
    expect(await screen.findByText(/The lock is advice/)).toBeTruthy();
    const outline = screen.getByRole("navigation", { name: "Foundations lessons" });
    expect(within(outline).getAllByRole("link")).toHaveLength(7); // the path title plus six lessons
    expect(within(outline).getByRole("link", { name: /Multi-head attention/ }).getAttribute("aria-current")).toBe("page");
  });

  it("has Surface, Deep and Reading tabs and switches between them", async () => {
    renderPage();
    await screen.findByRole("heading", { name: "Multi-head attention" });
    expect(screen.getAllByRole("tab").map((t) => t.textContent)).toEqual(["Surface", "Deep", "Reading"]);
    expect(screen.getByText(/split heads/)).toBeTruthy();
    await userEvent.click(screen.getByRole("tab", { name: "Deep" }));
    expect(screen.getByRole("tabpanel").textContent?.length).toBeGreaterThan(20);
  });

  it("shows the library's error when the lesson does not exist", async () => {
    mockApi({
      "GET /api/curricula/foundations/04-multi-head": {
        status: 404,
        problem: true,
        body: { title: "Not found", status: 404, detail: "unknown lesson 'foundations/04-multi-head'" },
      },
      "GET /api/curricula": { body: curricula },
    });
    render(
      <Providers>
        <MemoryRouter initialEntries={["/learn/foundations/04-multi-head"]}>
          <Routes>
            <Route path="/learn/:path/:lesson" element={<Lesson />} />
          </Routes>
        </MemoryRouter>
      </Providers>,
    );
    expect((await screen.findByRole("alert")).textContent).toContain("unknown lesson 'foundations/04-multi-head'");
  });
});

describe("Markdown", () => {
  it("does not render raw HTML or script links from lesson text", () => {
    const { container } = render(
      <Markdown>{'<script>alert(1)</script>\n\n<img src=x onerror="alert(1)">\n\n[bad](javascript:alert(1))\n\n`ok`'}</Markdown>,
    );
    expect(container.querySelector("script")).toBeNull();
    expect(container.querySelector("img")).toBeNull();
    expect(container.querySelector("a[href^='javascript']")).toBeNull();
    expect(screen.getByText("ok")).toBeTruthy();
  });

  it("renders tables and code", () => {
    const { container } = render(<Markdown>{"| a | b |\n|---|---|\n| `x` | y |\n\n```py\nz = 1\n```"}</Markdown>);
    expect(container.querySelectorAll("th")).toHaveLength(2);
    expect(container.querySelector("pre code")?.textContent).toContain("z = 1");
  });
});
