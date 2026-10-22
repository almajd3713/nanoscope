import { cleanup, configure } from "@testing-library/react";
import { afterEach, expect, vi } from "vitest";
import * as axeMatchers from "vitest-axe/matchers";

afterEach(() => cleanup());
// the graph lays itself out asynchronously; a loaded machine needs longer than the default second
configure({ asyncUtilTimeout: 5000 });
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

// Pages connect to the language server when the setting is on (the default). No test may reach a
// real one, such as the lsp service of a local stack: every socket is refused unless a test stubs
// its own. (Assigned for the same reason as ResizeObserver.)
globalThis.WebSocket = class {
  onopen: (() => void) | null = null;
  onmessage: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: (() => void) | null = null;
  constructor() {
    setTimeout(() => this.onerror?.(), 0);
  }
  send() {}
  close() {}
} as unknown as typeof WebSocket;
