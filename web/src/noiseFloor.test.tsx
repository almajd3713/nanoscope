import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import compare from "../test/fixtures/compare.json";
import { installEventSource } from "../test/eventsource";
import { mockApi } from "../test/fetch";
import { resetLevelCache } from "./app/level";
import { Providers } from "./app/providers";
import { Compare } from "./pages/Compare";

beforeEach(() => {
  localStorage.clear();
  resetLevelCache();
  installEventSource();
});
afterEach(() => vi.unstubAllGlobals());

function open(doc: object) {
  mockApi({ "POST /api/compare": { body: doc } });
  render(
    <Providers>
      <MemoryRouter initialEntries={["/compare?runs=baselines/tinystories-5min/gpt2,baselines/tinystories-5min/modern"]}>
        <Compare />
      </MemoryRouter>
    </Providers>,
  );
}

describe("noise floor", () => {
  it("is on the compare page, with the library's words, beside what the seeds can resolve", async () => {
    open(compare);
    expect(await screen.findByText(/Seed noise on this preset is about 0\.014 bpb/)).toBeTruthy();
    expect(screen.getByText(/known to about ±0\.034 bpb/)).toBeTruthy();
  });

  it("says why there is none when the preset has no baselines", async () => {
    open({ ...compare, noise_floor: { metric: "val_bpb", preset: "x", sd: null, source: null, models: [], note: "no shipped baselines with 3 or more seeds for preset 'x'" } });
    expect(await screen.findByText(/no shipped baselines with 3 or more seeds for preset 'x'/)).toBeTruthy();
  });
});
