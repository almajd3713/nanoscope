import { describe, expect, it } from "vitest";
import { modulePath } from "./flow";
import { spanText, traceOf, type TraceRow } from "./trace";

const rows: TraceRow[] = [
  { path: "blocks.0.attn", input_shapes: [[8, 256, 64]], output_shapes: [[8, 256, 64]], params: 12320, flops_per_token: 270528 },
  { path: "tok_emb", input_shapes: [[8, 256]], output_shapes: [[8, 256, 64]], params: 262144, flops_per_token: 0 },
];

describe("traceOf", () => {
  it("formats a module the way describe does", () => {
    expect(traceOf(rows, "blocks.0.attn")).toEqual({ shape: "8x256x64", params: "12.3k", flops: "271k FLOP/token" });
    expect(traceOf(rows, "tok_emb")).toEqual({ shape: "8x256 → 8x256x64", params: "262k", flops: "0 FLOP/token" });
  });
  it("is null without a trace or a module", () => {
    expect(traceOf(null, "tok_emb")).toBeNull();
    expect(traceOf(rows, null)).toBeNull();
    expect(traceOf(rows, "blocks.9.mlp")).toBeNull();
  });
});

describe("modulePath", () => {
  it("maps graph paths to describe's module names", () => {
    expect(modulePath(["block", "norm"])).toBe("blocks.0.norm1");
    expect(modulePath(["block", "attn", "pos"])).toBe("blocks.0.attn.pos");
    expect(modulePath(["pattern", 2, "mlp"])).toBe("blocks.2.mlp");
    expect(modulePath(["final_norm"])).toBe("norm");
    expect(modulePath(["pos_emb"])).toBe("pos_emb");
    expect(modulePath(["q"])).toBeNull();
    expect(modulePath(null)).toBeNull();
  });
});

describe("spanText", () => {
  it("counts columns from 1", () => {
    expect(spanText({ line: 10, col: 29, end_line: 10, end_col: 40 })).toBe("line 10, cols 30 to 41");
    expect(spanText({ line: 10, col: 18, end_line: 11, end_col: 90 })).toBe("line 10, col 19 to line 11, col 91");
    expect(spanText(null)).toBe("");
  });
});
