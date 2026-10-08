import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import lesson04 from "../../test/fixtures/lesson-04.json";
import { mockApi } from "../../test/fetch";
import { fromChecks, fromDescribeError, fromValidation, useDescribeMarkers } from "./diagnostics";

afterEach(() => vi.unstubAllGlobals());

describe("fromValidation", () => {
  it("puts a locked use on its line, with the library's words", () => {
    const markers = fromValidation([
      { code: "locked", message: "line 12: block:RoPE is locked until you build it yourself in the lesson modern-block/02-rope.", hint: "block:RoPE unlocks in the lesson modern-block/02-rope" },
      { code: "unknown_keyword", field: "colour", message: "no keyword colour" },
    ]);
    expect(markers).toHaveLength(1); // the unknown keyword belongs to the run form
    expect(markers[0]).toMatchObject({ line: 12, severity: "error", source: "nanoscope", code: "locked" });
    expect(markers[0]!.message).toContain("block:RoPE is locked");
    expect(markers[0]!.message).toContain("modern-block/02-rope");
  });
});

describe("fromDescribeError", () => {
  const error = "ShapeError: Block.attn: mat1 and mat2 shapes cannot be multiplied (8x256 and 128x128) (/home/me/ws/models/my_lm.py:14)";
  it("marks the line the trace failed on", () => {
    const [marker] = fromDescribeError(error, "models/my_lm.py");
    expect(marker).toMatchObject({ line: 14, severity: "error", code: "shape" });
    expect(marker!.message).toBe("Block.attn: mat1 and mat2 shapes cannot be multiplied (8x256 and 128x128)");
  });
  it("ignores another file's error and any other kind of failure", () => {
    expect(fromDescribeError(error, "models/other.py")).toEqual([]);
    expect(fromDescribeError("ValueError: nope", "models/my_lm.py")).toEqual([]);
    expect(fromDescribeError(null, "models/my_lm.py")).toEqual([]);
  });
});

describe("fromChecks", () => {
  const source = "import torch\n\n\nclass MultiHead(nn.Module):\n    pass\n";
  const results = [
    { id: "built", passed: true, reason: "ok" },
    { id: "same-as-reference", passed: false, reason: "output differs from the reference: max abs diff 0.31 on inputs of shape [[2, 6, 16]] (tolerance 1e-05)" },
    { id: "built-by-hand", passed: false, reason: "uses F.scaled_dot_product_attention" },
  ];
  it("puts a failed equivalence result on its class, from the lesson's own checks", () => {
    const markers = fromChecks(results, lesson04.checks as never, source);
    expect(markers).toHaveLength(1);
    expect(markers[0]).toMatchObject({ line: 4, code: "equivalent", source: "nanoscope" });
    expect(markers[0]!.message).toContain("max abs diff 0.31");
  });
  it("says nothing when the class is not in this file", () => {
    expect(fromChecks(results, lesson04.checks as never, "x = 1\n")).toEqual([]);
  });
});

function wrapper({ children }: { children: ReactNode }) {
  return <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>{children}</QueryClientProvider>;
}

describe("useDescribeMarkers", () => {
  it("traces after a save and shows the shape error on its line", async () => {
    const seen = mockApi({
      "POST /api/models/models/my_lm.py:MyLM/describe": { status: 202, body: { id: 5, state: "queued" } },
      "GET /api/jobs/5": {
        body: { id: 5, state: "failed", error: "ShapeError: Block.mlp: bad shape (/ws/models/my_lm.py:21)" },
      },
    });
    const { result, rerender } = renderHook(({ etag }) => useDescribeMarkers("models/my_lm.py", etag, "models/my_lm.py:MyLM"), {
      wrapper,
      initialProps: { etag: "e1" },
    });
    await waitFor(() => expect(result.current).toHaveLength(1));
    expect(result.current[0]).toMatchObject({ line: 21, message: "Block.mlp: bad shape" });
    rerender({ etag: "e1" }); // the same save: no second job
    expect(seen.filter((s) => s.method === "POST")).toHaveLength(1);
  });

  it("shows nothing for a model that traces", async () => {
    mockApi({
      "POST /api/models/models/my_lm.py:MyLM/describe": { status: 202, body: { id: 6, state: "queued" } },
      "GET /api/jobs/6": { body: { id: 6, state: "done", error: null } },
    });
    const { result } = renderHook(() => useDescribeMarkers("models/my_lm.py", "e1", "models/my_lm.py:MyLM"), { wrapper });
    await waitFor(() => expect(result.current).toEqual([]));
  });
});
