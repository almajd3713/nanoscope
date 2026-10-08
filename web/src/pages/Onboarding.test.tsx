import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Providers } from "../app/providers";
import { resetSettingsCache } from "../app/settings";
import { mockApi } from "../../test/fetch";
import { Onboarding } from "./Onboarding";

function renderPage() {
  return render(
    <Providers>
      <MemoryRouter initialEntries={["/welcome"]}>
        <Routes>
          <Route path="/welcome" element={<Onboarding />} />
          <Route path="/learn" element={<p>lessons page</p>} />
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

describe("Onboarding", () => {
  it("starts on I'm learning and sends guided", async () => {
    const seen = mockApi({ "POST /api/learn/policy": { body: { policy: "guided" } } });
    renderPage();
    expect((screen.getByRole("radio", { name: /I'm learning/ }) as HTMLInputElement).checked).toBe(true);
    await userEvent.click(screen.getByRole("button", { name: "Continue to lessons" }));
    expect(await screen.findByText("lessons page")).toBeTruthy();
    expect(seen).toEqual([{ method: "POST", path: "/api/learn/policy", body: { policy: "guided" } }]);
  });

  it("sends open when I know this is chosen", async () => {
    const seen = mockApi({ "POST /api/learn/policy": { body: { policy: "open" } } });
    renderPage();
    await userEvent.click(screen.getByRole("radio", { name: /I know this/ }));
    await userEvent.click(screen.getByRole("button", { name: "Continue to lessons" }));
    await waitFor(() => expect(seen).toHaveLength(1));
    expect(seen[0]?.body).toEqual({ policy: "open" });
  });

  it("shows the library's error word for word and stays on the page", async () => {
    mockApi({
      "POST /api/learn/policy": {
        status: 422,
        problem: true,
        body: { title: "Cannot be done as asked", status: 422, detail: "policy must be one of guided, open, got 'x'" },
      },
    });
    renderPage();
    await userEvent.click(screen.getByRole("button", { name: "Continue to lessons" }));
    expect((await screen.findByRole("alert")).textContent).toContain("policy must be one of guided, open, got 'x'");
    expect(screen.queryByText("lessons page")).toBeNull();
  });
});
