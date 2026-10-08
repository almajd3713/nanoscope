import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import codeOnly from "../../test/fixtures/graph-code_only.json";
import opaque from "../../test/fixtures/graph-opaque_block.json";
import type { GClass } from "./flow";
import { GraphView } from "./GraphView";

const frame = (el: React.ReactNode) => <div style={{ width: 800, height: 600 }}>{el}</div>;

describe("opaque nodes and code-only classes", () => {
  it("an opaque call is marked, and Edit in code jumps to its line", async () => {
    const onEditInCode = vi.fn();
    render(frame(<GraphView cls={opaque.classes[0] as unknown as GClass} onEditInCode={onEditInCode} />));
    await screen.findByLabelText("Gated in attn");
    expect(screen.getAllByText("Not a registered block: shown, not editable")).toHaveLength(2);
    // react-flow hides a node until it has measured it, which jsdom never does: query by text
    fireEvent.click(screen.getAllByText("Edit in code, line 11")[0]!);
    expect(onEditInCode).toHaveBeenCalledWith(11);
  });

  it("a code-only class says why and offers the line", () => {
    const onEditInCode = vi.fn();
    render(<GraphView cls={codeOnly.classes[1] as unknown as GClass} onEditInCode={onEditInCode} />);
    expect(screen.getByText(/is code only: __init__ has an assignment/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /^Edit in code, line \d+$/ }));
    expect(onEditInCode).toHaveBeenCalledWith(12);
  });
});
