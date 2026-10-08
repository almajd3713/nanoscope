import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { resetSettingsCache, setShowCommands } from "../app/settings";
import { EquivalentCommand } from "./EquivalentCommand";

const CLI = "nanoscope run models/my_lm.py:MyLM --preset tinystories-5min --seeds 3";
const PY = 'from nanoscope import run\nrun(MyLM, preset="tinystories-5min", seeds=3)';

beforeEach(() => {
  localStorage.clear();
  act(() => resetSettingsCache());
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("EquivalentCommand", () => {
  it("renders nothing and takes no space while the setting is off (the default)", () => {
    const { container } = render(<EquivalentCommand cli={CLI} />);
    expect(container.innerHTML).toBe("");
  });

  it("shows the command once the setting is on, and offers a Python tab when given", async () => {
    act(() => setShowCommands(true));
    render(<EquivalentCommand cli={CLI} python={PY} />);
    expect(screen.getByText(CLI)).toBeTruthy();
    await userEvent.click(screen.getByRole("button", { name: "Python" }));
    expect(screen.getByText(/from nanoscope import run/)).toBeTruthy();
  });

  it("is off when localStorage throws, and still switches on for the page view", () => {
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    const { container } = render(<EquivalentCommand cli={CLI} />);
    expect(container.innerHTML).toBe("");
    act(() => setShowCommands(true));
    expect(screen.getByText(CLI)).toBeTruthy();
  });

  it("copies from the click handler, and says when it cannot", async () => {
    act(() => setShowCommands(true));
    const writeText = vi.fn().mockResolvedValue(undefined);
    vi.stubGlobal("navigator", { clipboard: { writeText } });
    render(<EquivalentCommand cli={CLI} />);
    await userEvent.click(screen.getByRole("button", { name: "Copy" }));
    expect(writeText).toHaveBeenCalledWith(CLI);
    expect(await screen.findByRole("button", { name: "Copied" })).toBeTruthy();
  });

  it("falls back to Select and copy when the clipboard is unavailable", async () => {
    act(() => setShowCommands(true));
    vi.stubGlobal("navigator", {});
    render(<EquivalentCommand cli={CLI} />);
    await userEvent.click(screen.getByRole("button", { name: "Copy" }));
    expect(screen.getByRole("button", { name: "Select and copy" })).toBeTruthy();
  });
});
