import { cleanup } from "@testing-library/react";
import { afterEach, expect, vi } from "vitest";
import * as axeMatchers from "vitest-axe/matchers";

afterEach(() => cleanup());
expect.extend(axeMatchers);

// uPlot needs a real canvas and matchMedia; jsdom has neither. Tests of Curve replace this mock.
vi.mock("uplot", () => ({
  default: class {
    setData() {}
    setSize() {}
    destroy() {}
  },
}));
vi.mock("uplot/dist/uPlot.min.css", () => ({}));

// jsdom has no ResizeObserver; the plots only use it to follow their container's width.
// (Assigned, not vi.stubGlobal: tests call vi.unstubAllGlobals, which would remove it.)
globalThis.ResizeObserver = class {
  observe() {}
  unobserve() {}
  disconnect() {}
};
