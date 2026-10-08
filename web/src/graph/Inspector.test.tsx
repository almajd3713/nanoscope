import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import blocks from "../../test/fixtures/blocks-guided.json";
import modern from "../../test/fixtures/graph-modern_like.json";
import { ApiProblem } from "../api/problem";
import type { CatalogBlock } from "../editor/completions";
import { buildFlow, type GClass } from "./flow";
import { Inspector, parseLiteral } from "./Inspector";

const cls = modern.classes[0] as unknown as GClass;
const catalog = blocks.blocks as unknown as CatalogBlock[];
const pick = (name: string) => buildFlow(cls).boxes.find((b) => b.name === name)!;

function show(name: string, extra: Partial<Parameters<typeof Inspector>[0]> = {}) {
  const onEdit = vi.fn();
  render(
    <MemoryRouter>
      <Inspector cls={cls} box={pick(name)} blocks={catalog} onEdit={onEdit} {...extra} />
    </MemoryRouter>,
  );
  return onEdit;
}

describe("parseLiteral", () => {
  it("reads numbers, constants and strings the way code writes them", () => {
    expect([parseLiteral("4"), parseLiteral("-2"), parseLiteral("1e-5"), parseLiteral("0.5")]).toEqual([4, -2, 1e-5, 0.5]);
    expect([parseLiteral("True"), parseLiteral("False"), parseLiteral("None")]).toEqual([true, false, null]);
    expect([parseLiteral('"pre"'), parseLiteral("post")]).toEqual(["pre", "post"]);
  });
});

describe("Inspector", () => {
  it("shows the block's options with the values written in the file", () => {
    show("Attention");
    expect(screen.getByRole("heading", { name: "Attention" })).toBeTruthy();
    expect((screen.getByLabelText("n_heads") as HTMLInputElement).value).toBe("4");
    expect((screen.getByLabelText("n_kv_heads") as HTMLInputElement).value).toBe("2");
    expect((screen.getByLabelText("qk_norm") as HTMLInputElement).value).toBe("True");
    expect((screen.getByLabelText("window") as HTMLInputElement).value).toBe(""); // not written
    expect(screen.queryByLabelText("pos")).toBeNull(); // a block-valued option is its own box
  });

  it("an edited option is one set_arg at the block's path", async () => {
    const onEdit = show("Attention");
    const field = screen.getByLabelText("n_heads");
    await userEvent.clear(field);
    await userEvent.type(field, "8{Enter}");
    expect(onEdit).toHaveBeenCalledTimes(1);
    expect(onEdit).toHaveBeenCalledWith([{ op: "set_arg", class: "MyModern", path: ["block", "attn"], arg: "n_heads", value: { kind: "literal", value: 8, span: null } }]);
  });

  it("adds an option that was not written, and removes one that is cleared", async () => {
    const onEdit = show("Attention");
    await userEvent.type(screen.getByLabelText("window"), "16");
    await userEvent.tab();
    expect(onEdit).toHaveBeenLastCalledWith([expect.objectContaining({ op: "set_arg", arg: "window", value: expect.objectContaining({ value: 16 }) })]);
    await userEvent.clear(screen.getByLabelText("n_kv_heads"));
    await userEvent.tab();
    expect(onEdit).toHaveBeenLastCalledWith([{ op: "remove_arg", class: "MyModern", path: ["block", "attn"], arg: "n_kv_heads" }]);
  });

  it("does nothing when the value is unchanged", async () => {
    const onEdit = show("Attention");
    await userEvent.click(screen.getByLabelText("n_heads"));
    await userEvent.tab();
    expect(onEdit).not.toHaveBeenCalled();
  });

  it("swaps a block for another of its family with one replace_block", async () => {
    const onEdit = show("RoPE");
    // RoPE is locked for this learner in the fixture, but it is the one in the file: still the current choice
    const select = screen.getByLabelText("Swap for") as HTMLSelectElement;
    expect(select.value).toBe("RoPE");
    await userEvent.selectOptions(select, "NoPE");
    expect(onEdit).toHaveBeenCalledWith([{ op: "replace_block", class: "MyModern", path: ["block", "attn", "pos"], node: { kind: "block", block: "NoPE", args: {}, span: null } }]);
  });

  it("lists a locked block but cannot pick it, and names its lesson", () => {
    show("RMSNorm", { box: { ...pick("RMSNorm"), family: "positional", name: "NoPE", path: ["block", "attn", "pos"] } });
    const option = within(screen.getByLabelText("Swap for")).getByRole("option", { name: /RoPE/ }) as HTMLOptionElement;
    expect(option.disabled).toBe(true);
    expect(option.textContent).toContain("locked, needs modern-block/02-rope");
    expect(screen.getByRole("link", { name: "modern-block/02-rope" }).getAttribute("href")).toBe("/learn/modern-block/02-rope");
  });

  it("says an opaque call is edited in the code", () => {
    const opaque = { ...pick("Attention"), kind: "opaque" as const, name: "Gated", line: 11 };
    show("Attention", { box: opaque });
    expect(screen.getByText(/not one the graph can edit, line 11/)).toBeTruthy();
    expect(screen.queryByLabelText("Swap for")).toBeNull();
  });

  it("shows the server's refusal word for word", () => {
    show("Attention", { error: new ApiProblem(422, { title: "Locked until you build it", detail: "block:RoPE is locked until you build it yourself in the lesson modern-block/02-rope." }) });
    expect(screen.getByRole("alert").textContent).toContain("block:RoPE is locked until you build it yourself");
  });

  it("a box nanoscope added has nothing to edit in the file", () => {
    show("Head", { box: { ...pick("Attention"), path: null, name: "Head", slot: "head" } });
    expect(screen.getByText(/added by nanoscope/)).toBeTruthy();
  });
});
