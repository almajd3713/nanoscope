// Where the boxes go. Computed from the graph on every parse and never saved: nothing about a
// model lives only in the browser, and a file edited elsewhere lays itself out afresh.
import ELK, { type ElkNode } from "elkjs/lib/elk.bundled.js";
import type { Box, Flow } from "./flow";

export type Depth = "surface" | "detailed" | "research";

export const BOX_WIDTH = 232;
const ROW = 18;
const PAD = 14;

export function boxHeight(box: Box, depth: Depth): number {
  if (box.kind === "opaque") return 56;
  const shown = depth === "surface" ? 0 : Math.min(box.args.length, 4);
  return 44 + (depth === "surface" ? 0 : ROW + shown * ROW);
}

export type Placed = { x: number; y: number; width: number; height: number };
export type Layout = { boxes: Record<string, Placed>; groups: Record<string, Placed>; width: number; height: number };

const elk = new ELK();

export async function layout(flow: Flow, depth: Depth): Promise<Layout> {
  const leaf = (b: Box) => ({ id: b.id, width: BOX_WIDTH, height: boxHeight(b, depth) });
  const groupOf = new Map(flow.boxes.map((b) => [b.id, b.parent]));
  const root = {
    id: "root",
    layoutOptions: {
      "elk.algorithm": "layered",
      "elk.direction": "DOWN",
      "elk.hierarchyHandling": "INCLUDE_CHILDREN",
      "elk.spacing.nodeNode": "28",
      "elk.layered.spacing.nodeNodeBetweenLayers": "36",
      "elk.padding": `[top=${PAD},left=${PAD},bottom=${PAD},right=${PAD}]`,
    },
    children: [
      ...flow.boxes.filter((b) => b.parent === null).map(leaf),
      ...flow.groups.map((g) => ({
        id: g.id,
        layoutOptions: { "elk.padding": `[top=${PAD + 22},left=${PAD},bottom=${PAD},right=${PAD}]`, "elk.direction": "DOWN" },
        children: flow.boxes.filter((b) => b.parent === g.id).map(leaf),
      })),
    ],
    edges: flow.arrows
      .filter((a) => groupOf.has(a.from) && groupOf.has(a.to))
      .map((a) => ({ id: a.id, sources: [a.from], targets: [a.to] })),
  };
  const done: { width?: number; height?: number; children?: ElkNode[] } = await elk.layout(root);
  const out: Layout = { boxes: {}, groups: {}, width: done.width ?? 0, height: done.height ?? 0 };
  const place = (n: { id: string; x?: number; y?: number; width?: number; height?: number }, into: Record<string, Placed>) => {
    into[n.id] = { x: n.x ?? 0, y: n.y ?? 0, width: n.width ?? BOX_WIDTH, height: n.height ?? 0 };
  };
  for (const child of done.children ?? []) {
    if (flow.groups.some((g) => g.id === child.id)) {
      place(child, out.groups);
      for (const inner of child.children ?? []) place(inner, out.boxes); // relative to the group
    } else {
      place(child, out.boxes);
    }
  }
  return out;
}
