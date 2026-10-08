import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import modern from "../../test/fixtures/graph-modern_like.json";
import type { GClass } from "./flow";
import { GraphView } from "./GraphView";
import type { NodeInfo } from "./nodes";

const frame = (el: React.ReactNode) => <div style={{ width: 800, height: 600 }}>{el}</div>;

const cls = modern.classes[0] as unknown as GClass;
const badges = (table: Record<string, NodeInfo>) => (name: string) => table[name];

describe("lock and certification badges", () => {
  it("a locked block in the file is dashed, locked, and names its lesson", async () => {
    render(frame(<GraphView cls={cls} infoOf={badges({ RoPE: { locked: true, lesson: "modern-block/02-rope" } })} />));
    const rope = await screen.findByLabelText("RoPE in pos");
    expect(rope.textContent).toContain("Pass modern-block/02-rope to use it");
    expect(rope.querySelector('[aria-label="locked"]')).toBeTruthy();
  });

  it("your own block shows its certification, in a word and a glyph", async () => {
    const table = { RMSNorm: { user: true, reference: "naive_scale_norm", certification: "certified" } };
    render(frame(<GraphView cls={cls} infoOf={badges(table)} />));
    const norm = await screen.findByLabelText("RMSNorm in norm");
    expect(norm.querySelector('[aria-label="certified"]')).toBeTruthy();
  });

  it("a stale or failed certification warns", async () => {
    render(frame(<GraphView cls={cls} infoOf={badges({ SwiGLU: { user: true, certification: "stale" }, Attention: { user: true, certification: "failed" } })} />));
    const mlp = await screen.findByLabelText("SwiGLU in mlp");
    expect(mlp.querySelector('[aria-label="changed since certified"]')).toBeTruthy();
    expect(screen.getByLabelText("Attention in attn").querySelector('[aria-label="failed its check"]')).toBeTruthy();
  });
});
