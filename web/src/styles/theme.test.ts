import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { getChoice, initTheme, readToken, resolvedTheme, setChoice, subscribe } from "./theme";

type Listener = () => void;

function mockMedia(initial: boolean) {
  const state = { matches: initial, listeners: new Set<Listener>() };
  vi.stubGlobal("matchMedia", () => ({
    get matches() {
      return state.matches;
    },
    addEventListener: (_: string, fn: Listener) => state.listeners.add(fn),
    removeEventListener: (_: string, fn: Listener) => state.listeners.delete(fn),
  }));
  return {
    set(value: boolean) {
      state.matches = value;
      for (const fn of state.listeners) fn();
    },
  };
}

beforeEach(() => {
  localStorage.clear();
  document.documentElement.removeAttribute("data-theme");
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("theme", () => {
  it("defaults to system and sets no attribute", () => {
    mockMedia(false);
    initTheme();
    expect(getChoice()).toBe("system");
    expect(document.documentElement.hasAttribute("data-theme")).toBe(false);
  });

  it("keeps an explicit choice and removes the attribute for system", () => {
    mockMedia(false);
    setChoice("dark");
    expect(document.documentElement.getAttribute("data-theme")).toBe("dark");
    expect(getChoice()).toBe("dark");
    setChoice("system");
    expect(document.documentElement.hasAttribute("data-theme")).toBe(false);
    expect(getChoice()).toBe("system");
  });

  it("restores the stored choice on start", () => {
    mockMedia(false);
    localStorage.setItem("nanoscope.theme", "light");
    initTheme();
    expect(document.documentElement.getAttribute("data-theme")).toBe("light");
  });

  it("still works when localStorage throws", () => {
    mockMedia(false);
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    expect(getChoice()).toBe("system");
    const seen: string[] = [];
    subscribe((t) => seen.push(t));
    setChoice("dark");
    expect(document.documentElement.getAttribute("data-theme")).toBe("dark");
    expect(seen).toEqual(["dark"]);
  });

  it("tells subscribers when the OS preference changes while on system", () => {
    const media = mockMedia(false);
    initTheme();
    const seen: string[] = [];
    const off = subscribe((t) => seen.push(t));
    media.set(true);
    expect(seen).toEqual(["dark"]);
    expect(resolvedTheme()).toBe("dark");
    off();
    media.set(false);
    expect(seen).toEqual(["dark"]);
  });

  it("ignores OS changes when a theme is chosen", () => {
    const media = mockMedia(false);
    initTheme();
    setChoice("light");
    const seen: string[] = [];
    subscribe((t) => seen.push(t));
    media.set(true);
    expect(seen).toEqual([]);
  });

  it("reads a token from the root element", () => {
    document.documentElement.style.setProperty("--surface", " #f2f4f7 ");
    expect(readToken("surface")).toBe("#f2f4f7");
  });
});
