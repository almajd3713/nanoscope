import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import list from "../../test/fixtures/study-list.json";
import { mockApi } from "../../test/fetch";
import { resetLevelCache, setLevel } from "../app/level";
import { Providers } from "../app/providers";
import { Studies } from "./Studies";

beforeEach(() => {
  localStorage.clear();
  resetLevelCache();
});
afterEach(() => vi.unstubAllGlobals());

function open(body: unknown = list) {
  mockApi({ "GET /api/studies": { body } });
  render(
    <Providers>
      <MemoryRouter>
        <Studies />
      </MemoryRouter>
    </Providers>,
  );
}

// The list is real API output (web/test/fixtures/study-list.json).
describe("Studies page", () => {
  it("lists studies and offers the builder from Research", async () => {
    setLevel("Research");
    open();
    expect(await screen.findByRole("link", { name: list[0]!.name })).toBeTruthy();
    expect(screen.getByRole("link", { name: "New study" })).toBeTruthy();
    expect(screen.getByRole("link", { name: list[0]!.spec! }).getAttribute("href")).toBe(`/studies/${list[0]!.name}/edit`);
  });

  it("shows the same list below Research without the builder controls", async () => {
    setLevel("Tinker");
    open();
    expect(await screen.findByRole("link", { name: list[0]!.name })).toBeTruthy();
    expect(screen.queryByRole("link", { name: "New study" })).toBeNull();
    expect(screen.queryByRole("link", { name: list[0]!.spec! })).toBeNull();
    expect(screen.getByText(list[0]!.spec!)).toBeTruthy();
  });

  it("explains an empty list, with the builder link only from Research", async () => {
    setLevel("Learn");
    open([]);
    expect(await screen.findByText("No studies yet")).toBeTruthy();
    expect(screen.queryByRole("link", { name: "Build one" })).toBeNull();
  });
});
