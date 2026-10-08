import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ConflictDialog } from "./ConflictDialog";

const DIFF = [
  "--- mylm.py (on disk)",
  "+++ mylm.py (yours)",
  "@@ -1,2 +1,2 @@",
  " class MyLM:",
  "-    width = 8",
  "+    width = 16",
  "",
].join("\n");

function show(extra: Partial<Parameters<typeof ConflictDialog>[0]> = {}) {
  const onKeepMine = vi.fn();
  const onTakeTheirs = vi.fn();
  const onOpenChange = vi.fn();
  render(
    <ConflictDialog open onOpenChange={onOpenChange} path="mylm.py" diff={DIFF} onKeepMine={onKeepMine} onTakeTheirs={onTakeTheirs} {...extra} />,
  );
  return { onKeepMine, onTakeTheirs, onOpenChange };
}

describe("ConflictDialog", () => {
  it("names the file and shows the server's diff with signs, not colour alone", () => {
    show();
    expect(screen.getByText("mylm.py changed on disk")).toBeTruthy();
    const diff = screen.getByLabelText("Differences");
    expect(diff.textContent).toContain("−    width = 8");
    expect(diff.textContent).toContain("+    width = 16");
    expect(diff.textContent).toContain("@@ -1,2 +1,2 @@");
  });

  it("keeps mine or takes theirs, and each button says which", async () => {
    const { onKeepMine, onTakeTheirs } = show();
    await userEvent.click(screen.getByRole("button", { name: "Take theirs" }));
    expect(onTakeTheirs).toHaveBeenCalledOnce();
    await userEvent.click(screen.getByRole("button", { name: "Keep mine" }));
    expect(onKeepMine).toHaveBeenCalledOnce();
  });

  it("can be put off without choosing", async () => {
    const { onOpenChange, onKeepMine, onTakeTheirs } = show();
    await userEvent.click(screen.getByRole("button", { name: "Decide later" }));
    expect(onOpenChange).toHaveBeenCalledWith(false);
    expect(onKeepMine).not.toHaveBeenCalled();
    expect(onTakeTheirs).not.toHaveBeenCalled();
  });

  it("disables both choices while one is being applied", () => {
    show({ busy: true });
    expect((screen.getByRole("button", { name: "Keep mine" }) as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByRole("button", { name: "Take theirs" }) as HTMLButtonElement).disabled).toBe(true);
  });
});
