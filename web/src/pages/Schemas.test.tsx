import { act, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import list from "../../test/fixtures/schemas-list.json";
import inspect from "../../test/fixtures/schemas-inspect.json";
import { mockApi } from "../../test/fetch";
import { Providers } from "../app/providers";
import { resetSettingsCache, setShowCommands } from "../app/settings";
import { Schemas } from "./Schemas";

function open(path = "/schemas") {
  const seen = mockApi({
    "GET /api/schemas": { body: list },
    "GET /api/version": { body: { nanoscope: "0.5.0", schemas: list } },
    "GET /api/schemas/inspect": { body: inspect },
    "GET /api/schemas/author-check": { body: { ...inspect, title: "author-check.v1" } },
    "GET /api/schemas/config": { body: { ...inspect, title: "config.v1", properties: { seed: { type: "integer" } }, required: ["seed"] } },
  });
  render(
    <Providers>
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route path="/schemas" element={<Schemas />} />
          <Route path="/schemas/:name" element={<Schemas />} />
        </Routes>
      </MemoryRouter>
    </Providers>,
  );
  return seen;
}

beforeEach(() => {
  localStorage.clear();
  act(() => {
    resetSettingsCache();
    setShowCommands(true);
  });
});
afterEach(() => vi.unstubAllGlobals());

describe("Schemas", () => {
  it("lists every schema and shows one as a field table in the schema's own words", async () => {
    open("/schemas/inspect");
    const table = await screen.findByRole("table", { name: "Fields of inspect" });
    expect(await screen.findByText(`${Object.keys(list).length} JSON Schemas (2020-12)`, { exact: false })).toBeTruthy();
    expect(within(table).getByText("attention[].weights")).toBeTruthy();
    expect(within(table).getByText("The run inspected.")).toBeTruthy();
    const nav = screen.getByRole("navigation", { name: "Schemas" });
    expect(within(nav).getByRole("link", { name: /^inspect/ }).getAttribute("aria-current")).toBe("page");
    expect(screen.getByText(inspect.$id)).toBeTruthy();
  });

  it("finds a schema by name", async () => {
    open("/schemas/inspect");
    await screen.findByRole("table", { name: "Fields of inspect" });
    await userEvent.type(screen.getByLabelText("Find a schema"), "conf");
    const nav = screen.getByRole("navigation", { name: "Schemas" });
    expect(within(nav).getAllByRole("link").map((a) => a.textContent)).toEqual(["configv1"]);
  });

  it("opens the first schema when none is chosen", async () => {
    open("/schemas");
    expect(await screen.findByRole("table", { name: "Fields of author-check" })).toBeTruthy();
  });
});
