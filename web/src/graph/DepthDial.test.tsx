import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import modern from "../../test/fixtures/graph-modern_like.json";
import { resetLevelCache, setLevel } from "../app/level";
import { defaultDepth, DepthDial } from "./DepthDial";
import type { GClass } from "./flow";
import { GraphView } from "./GraphView";
import type { NodeInfo } from "./nodes";
import type { TraceRow } from "./trace";

const cls = modern.classes[0] as unknown as GClass;

beforeEach(() => {
  localStorage.clear();
  resetLevelCache();
});
afterEach(() => vi.unstubAllGlobals());

const options = () => screen.getAllByRole("radio").map((r) => r.textContent);

describe("DepthDial", () => {
  it("a level only adds options: Learn has the diagram, Tinker adds detail, Research adds research", () => {
    const { unmount } = render(<DepthDial value="surface" onChange={() => {}} />);
    expect(options()).toEqual(["Surface"]);
    unmount();
    act(() => setLevel("Tinker"));
    const second = render(<DepthDial value="surface" onChange={() => {}} />);
    expect(options()).toEqual(["Surface", "Detailed"]);
    second.unmount();
    act(() => setLevel("Research"));
    render(<DepthDial value="surface" onChange={() => {}} />);
    expect(options()).toEqual(["Surface", "Detailed", "Research"]);
  });

  it("reports the chosen depth", async () => {
    act(() => setLevel("Research"));
    const onChange = vi.fn();
    render(<DepthDial value="surface" onChange={onChange} />);
    await userEvent.click(screen.getByRole("radio", { name: "Research" }));
    expect(onChange).toHaveBeenCalledWith("research");
  });

  it("a page opens at the deepest default its level can show", () => {
    expect(defaultDepth("Learn")).toBe("surface");
    expect(defaultDepth("Tinker")).toBe("detailed");
    expect(defaultDepth("Extend")).toBe("detailed");
  });
});

const rows: TraceRow[] = [
  { path: "blocks.0.attn", input_shapes: [[8, 256, 128]], output_shapes: [[8, 256, 128]], params: 49200, flops_per_token: 689000 },
];
const info = (name: string): NodeInfo | undefined =>
  name === "Attention" ? { reference: "naive_causal_attention", user: false, locked: false } : undefined;

const view = (depth: "surface" | "detailed" | "research") => (
  <div style={{ width: 800, height: 600 }}>
    <GraphView cls={cls} depth={depth} trace={rows} infoOf={info} />
  </div>
);

describe("what a box says at each depth", () => {
  it("surface: names only", async () => {
    render(view("surface"));
    await screen.findByLabelText("Attention in attn");
    expect(screen.queryByText("n_kv_heads=2")).toBeNull();
    expect(screen.queryByText(/8x256x128/)).toBeNull();
  });

  it("detailed: arguments, shape and parameters from the last trace", async () => {
    render(view("detailed"));
    await screen.findByLabelText("Attention in attn");
    expect(screen.getByText("n_kv_heads=2")).toBeTruthy();
    expect(screen.getByText("8x256x128 · 49.2k")).toBeTruthy();
  });

  it("research: where the call is in the file, equivalence and FLOPs", async () => {
    render(view("research"));
    await screen.findByLabelText("Attention in attn");
    await waitFor(() => expect(screen.getByText("reference naive_causal_attention, tested in the library's checks")).toBeTruthy());
    expect(screen.getByText("689k FLOP/token")).toBeTruthy();
    expect(screen.getAllByText(/^line \d+, cols \d+ to \d+$/).length).toBeGreaterThan(0);
  });
});
