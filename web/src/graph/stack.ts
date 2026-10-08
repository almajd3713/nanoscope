import type { Edit } from "./api";
import type { GClass, GNode } from "./flow";

export type Layer = { index: number; label: string };
export type Stack = { kind: "uniform"; layers: number | null; items: Layer[] } | { kind: "pattern"; items: Layer[]; layers: number | null };

type BlockNode = Extract<GNode, { kind: "block" }>;

const isBlock = (n: GNode | undefined): n is BlockNode => n?.kind === "block";

function windowOf(layer: GNode): string {
  const attn = isBlock(layer) ? layer.args["attn"] : undefined;
  const window = isBlock(attn) ? attn.args["window"] : undefined;
  return window && window.kind === "literal" ? `window=${String(window.value)}` : "global";
}

// The layers of a Decoder: one block repeated n_layers times, or a pattern of blocks repeated
// through them.
export function stackOf(cls: GClass): Stack | null {
  const args = cls.args;
  if (!args) return null;
  const n = args["n_layers"]?.kind === "literal" && typeof args["n_layers"].value === "number" ? args["n_layers"].value : null;
  const pattern = args["pattern"];
  if (pattern?.kind === "list") {
    return { kind: "pattern", layers: n, items: pattern.items.map((item, index) => ({ index, label: windowOf(item) })) };
  }
  if (isBlock(args["block"])) return { kind: "uniform", layers: n, items: [{ index: 0, label: windowOf(args["block"]) }] };
  return null;
}

// One more layer: n_layers + 1 for a single block, a copy of the last block for a pattern.
export function addLayerEdit(cls: GClass): Edit | null {
  const args = cls.args;
  if (!args) return null;
  const pattern = args["pattern"];
  if (pattern?.kind === "list") {
    const last = pattern.items[pattern.items.length - 1];
    return last ? { op: "add_layer", class: cls.name, path: ["pattern"], node: last } : null;
  }
  return { op: "add_layer", class: cls.name };
}

export function removeLayerEdit(cls: GClass, index: number | null): Edit {
  return index === null || cls.args?.["pattern"]?.kind !== "list"
    ? { op: "remove_layer", class: cls.name }
    : { op: "remove_layer", class: cls.name, path: ["pattern"], index };
}

export type Spec = { first: string; second: string; counts: [number, number] };

// "sliding:global 3:1": three layers of the first kind, then one of the second, repeated.
export function parsePattern(text: string): Spec | string {
  const m = /^\s*(\w+):(\w+)\s+(\d+):(\d+)\s*$/.exec(text);
  if (!m) return "write it as kind:kind a:b, for example sliding:global 3:1";
  const kinds = [m[1]!, m[2]!];
  const bad = kinds.find((k) => k !== "sliding" && k !== "global");
  if (bad) return `${bad} is not a layer kind here: use sliding or global`;
  const counts: [number, number] = [Number(m[3]), Number(m[4])];
  if (counts[0] + counts[1] === 0) return "a pattern needs at least one layer";
  return { first: kinds[0]!, second: kinds[1]!, counts };
}

function withWindow(layer: BlockNode, window: number | null): BlockNode {
  const copy = structuredClone(layer);
  const attn = copy.args["attn"];
  if (isBlock(attn)) {
    if (window === null) delete attn.args["window"];
    else attn.args["window"] = { kind: "literal", value: window };
  }
  return copy;
}

// The blocks a spec stands for, made from the model's own layer (its first pattern item, or its block).
export function patternItems(cls: GClass, spec: Spec, window: number): GNode[] | string {
  const args = cls.args ?? {};
  const pattern = args["pattern"];
  const base = pattern?.kind === "list" ? pattern.items[0] : args["block"];
  if (!isBlock(base)) return "this model has no layer block to build a pattern from";
  if (!isBlock(base.args["attn"])) return "the layer has no attention block to give a window";
  if (!Number.isInteger(window) || window < 1) return "the window is a whole number of tokens, at least 1";
  const make = (kind: string) => withWindow(base, kind === "sliding" ? window : null);
  return [
    ...Array.from({ length: spec.counts[0] }, () => make(spec.first)),
    ...Array.from({ length: spec.counts[1] }, () => make(spec.second)),
  ];
}
