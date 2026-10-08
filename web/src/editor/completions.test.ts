import { describe, expect, it } from "vitest";
import { type Catalog, suggestionsAt } from "./completions";

const catalog: Catalog = {
  presets: ["tinystories-5min", "tinystories-30min", "fineweb-edu-1h"],
  blocks: [
    {
      name: "Attention",
      family: "attention",
      doc: "Multi-head attention.",
      args: [
        { name: "n_heads", type: "int", required: true },
        { name: "n_kv_heads", type: "int | None", required: false, default: null },
        { name: "window", type: "int | None", required: false, default: null },
      ],
      lock: { locked: false, lesson: "foundations/04-multi-head" },
    },
    { name: "RoPE", family: "positional", args: [{ name: "base", type: "float", required: false, default: 10000 }], lock: { locked: true, lesson: "modern-block/02-rope" } },
    { name: "RMSNorm", family: "norm", args: [], lock: { locked: false, lesson: null } },
    { name: "Block", family: "structure", args: [{ name: "attn", required: true }, { name: "mlp", required: true }, { name: "norm", required: true }] },
  ],
};

const labels = (text: string) => suggestionsAt(text, catalog).map((s) => s.label);

describe("completions", () => {
  it("offers block names where a value starts, with family and lock in the detail", () => {
    expect(labels("        block=Block(norm=RM")).toEqual(["RMSNorm"]);
    const rope = suggestionsAt("attn=Attention(n_heads=2, pos=Ro", catalog);
    expect(rope.map((s) => s.label)).toEqual(["RoPE"]);
    expect(rope[0]!.detail).toBe("positional, locked, needs modern-block/02-rope");
  });

  it("does not offer a block after a dot, or the name it already has", () => {
    expect(labels("nn.Att")).toEqual([]);
    expect(labels("x = Attention")).toEqual([]);
  });

  it("offers the keyword arguments of the call the cursor is in", () => {
    const found = suggestionsAt("Attention(", catalog);
    expect(found.map((s) => s.label)).toEqual(["n_heads", "n_kv_heads", "window"]);
    expect(found[0]).toMatchObject({ insertText: "n_heads=", kind: "argument", detail: "required: int" });
    expect(found[1]!.detail).toBe("optional: int | None = null");
  });

  it("filters by what is typed and leaves out arguments already given", () => {
    expect(labels("Attention(n_heads=4, n_")).toEqual(["n_kv_heads"]);
    expect(labels("Attention(n_heads=4, n_kv_heads=2, ")).toEqual(["window"]);
  });

  it("finds the innermost open call across lines and ignores brackets in strings and comments", () => {
    const text = ["Block(", "    norm=RMSNorm(),", "    attn=Attention(  # (the heads)", "        n_heads=2,", "        "].join("\n");
    expect(labels(text)).toEqual(["n_kv_heads", "window"]);
    expect(labels('Attention(n_heads=")", ')).toEqual(["n_kv_heads", "window"]);
    // back in Block after Attention closed
    expect(labels("Block(attn=Attention(n_heads=2), ")).toEqual(["mlp", "norm"]);
  });

  it("offers nothing for a call it does not know, or inside a value", () => {
    expect(labels("print(")).toEqual([]);
    expect(labels("Attention(n_heads=")).toEqual([]);
  });

  it("completes preset names inside the string", () => {
    expect(labels('run(Model, preset="tiny')).toEqual(["tinystories-5min", "tinystories-30min"]);
    expect(labels('get_preset("fine')).toEqual(["fineweb-edu-1h"]);
    expect(labels('print("tiny')).toEqual([]);
  });
});
