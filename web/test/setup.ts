import { cleanup } from "@testing-library/react";
import { afterEach, vi } from "vitest";

afterEach(() => cleanup());

// uPlot needs a real canvas and matchMedia; jsdom has neither. Tests of Curve replace this mock.
vi.mock("uplot", () => ({
  default: class {
    setData() {}
    setSize() {}
    destroy() {}
  },
}));
vi.mock("uplot/dist/uPlot.min.css", () => ({}));
