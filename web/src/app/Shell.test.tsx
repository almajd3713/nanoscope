import { act, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { mockApi } from "../../test/fetch";
import { Providers } from "./providers";
import { resetLevelCache } from "./level";
import { Shell } from "./Shell";

function renderShell(path = "/runs") {
  mockApi({ "GET /api/jobs": { body: [] }, "GET /api/workers": { body: [] } });
  return render(
    <Providers>
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route element={<Shell />}>
            <Route path="/runs" element={<p>runs page</p>} />
            <Route path="/learn" element={<p>learn page</p>} />
          </Route>
        </Routes>
      </MemoryRouter>
    </Providers>,
  );
}

beforeEach(() => {
  localStorage.clear();
  act(() => resetLevelCache());
  vi.stubGlobal("matchMedia", () => ({ matches: false, addEventListener() {}, removeEventListener() {} }));
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("Shell", () => {
  it("lists the nav in order and marks the current page", () => {
    renderShell("/runs");
    const nav = screen.getByRole("navigation", { name: "Main" });
    const names = within(nav).getAllByRole("link").map((a) => a.textContent);
    expect(names).toEqual(["Learn", "Runs", "Compare", "Studies", "Hardware", "Components"]);
    expect(within(nav).getByRole("link", { name: "Runs" }).getAttribute("aria-current")).toBe("page");
    expect(within(nav).getByRole("link", { name: "Learn" }).getAttribute("aria-current")).toBeNull();
    expect(screen.getByText("runs page")).toBeTruthy();
  });

  it("has the Settings link and a theme toggle", () => {
    renderShell();
    expect(screen.getByRole("link", { name: "Settings" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Switch to the dark theme" })).toBeTruthy();
  });

  it("starts on Learn and keeps a chosen level in localStorage", async () => {
    renderShell();
    const group = screen.getByRole("radiogroup", { name: "Level" });
    expect(within(group).getByRole("radio", { name: "Learn" }).getAttribute("aria-checked")).toBe("true");
    await userEvent.click(within(group).getByRole("radio", { name: "Research" }));
    expect(localStorage.getItem("nanoscope.level")).toBe("Research");
    expect(within(group).getByRole("radio", { name: "Research" }).getAttribute("aria-checked")).toBe("true");
  });

  it("reads the stored level on start", () => {
    localStorage.setItem("nanoscope.level", "Tinker");
    renderShell();
    const group = screen.getByRole("radiogroup", { name: "Level" });
    expect(within(group).getByRole("radio", { name: "Tinker" }).getAttribute("aria-checked")).toBe("true");
  });

  it("works when localStorage throws", async () => {
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    renderShell();
    const group = screen.getByRole("radiogroup", { name: "Level" });
    expect(within(group).getByRole("radio", { name: "Learn" }).getAttribute("aria-checked")).toBe("true");
    await userEvent.click(within(group).getByRole("radio", { name: "Extend" }));
    expect(within(group).getByRole("radio", { name: "Extend" }).getAttribute("aria-checked")).toBe("true");
  });

  it("does not let the level be unset by pressing it again", async () => {
    renderShell();
    const group = screen.getByRole("radiogroup", { name: "Level" });
    await userEvent.click(within(group).getByRole("radio", { name: "Learn" }));
    expect(within(group).getByRole("radio", { name: "Learn" }).getAttribute("aria-checked")).toBe("true");
  });
});
