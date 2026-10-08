// The graph of a model class (what POST /api/files/{path}/graph returns) as boxes and arrows.
// Pure: no layout and no React here. Each box remembers the path of its node in the graph, which
// is how an edit says what it changes (nanoscope/blocks/graph.py: apply_edits).

export type Span = { line: number; col: number; end_line: number; end_col: number };
export type GNode =
  | { kind: "literal"; value: unknown; span?: Span }
  | { kind: "param"; name: string; span?: Span }
  | { kind: "expr"; source: string; span?: Span }
  | { kind: "opaque"; call: string; source: string; reason?: string; span?: Span }
  | { kind: "list"; items: GNode[]; span?: Span }
  | { kind: "block"; block: string; args: Record<string, GNode>; family?: string; tier?: string; local?: boolean; span?: Span };
export type GClass = {
  name: string;
  kind: "decoder" | "composite" | "filled";
  line: number;
  representable: boolean;
  reason?: string | null;
  reason_line?: number | null;
  args?: Record<string, GNode>;
  slots?: string[];
  // a "filled" class (class OneHead(AttentionTemplate)) names the template it fills
  template?: string;
};
export type Path = (string | number)[];

export type BoxKind = "block" | "opaque" | "fixed";
export type Box = {
  id: string;
  kind: BoxKind;
  name: string; // the block's class name, as written
  slot: string | null; // the argument it fills ("attn")
  family: string | null;
  args: [string, string][]; // the plain-value arguments, as written in code
  path: Path | null; // where it sits in the graph; null for the boxes nanoscope adds (embedding, head)
  line: number | null;
  span: Span | null; // where its call sits in the file (1-based lines, 0-based columns)
  module: string | null; // the module path describe reports for it ("blocks.0.attn")
  parent: string | null; // the layer group it sits in
};
export type Group = { id: string; label: string; path: Path | null };
export type Arrow = { id: string; from: string; to: string; kind: "flow" | "arg" };
export type Flow = { boxes: Box[]; groups: Group[]; arrows: Arrow[] };

const SLOT_ORDER = ["norm", "attn", "mlp"];

export function pathId(path: Path): string {
  return path.join("/");
}

export function valueText(node: GNode): string {
  switch (node.kind) {
    case "literal":
      return typeof node.value === "string" ? JSON.stringify(node.value) : node.value === null ? "None" : node.value === true ? "True" : node.value === false ? "False" : String(node.value);
    case "param":
      return node.name;
    case "expr":
      return node.source;
    case "opaque":
      return node.source;
    case "list":
      return `[${node.items.map(valueText).join(", ")}]`;
    case "block":
      return `${node.block}(…)`;
  }
}

const SLOT_MODULE: Record<string, string> = { norm: "norm1", attn: "attn", mlp: "mlp" };

// The path of the module `describe` reports for a box: the first layer of a stack stands for the
// stack, a pattern item for its own layer.
export function modulePath(path: Path | null): string | null {
  if (!path || path.length === 0) return null;
  const [head, ...rest] = path;
  if (head === "final_norm") return "norm";
  if (head === "pos_emb") return "pos_emb";
  let layer: number;
  let slots: Path;
  if (head === "block") {
    layer = 0;
    slots = rest;
  } else if (head === "pattern" && typeof rest[0] === "number") {
    layer = rest[0];
    slots = rest.slice(1);
  } else {
    return null;
  }
  if (slots.length === 0) return `blocks.${layer}`;
  const [first, ...more] = slots as string[];
  return [`blocks.${layer}`, SLOT_MODULE[first!] ?? first, ...more].join(".");
}

function box(node: GNode, slot: string | null, path: Path, parent: string | null): Box {
  const line = node.span?.line ?? null;
  const span = node.span ?? null;
  const module = modulePath(path);
  if (node.kind === "block") {
    const args: [string, string][] = Object.entries(node.args)
      .filter(([, v]) => v.kind !== "block" && v.kind !== "list")
      .map(([k, v]) => [k, valueText(v)]);
    return { id: pathId(path), kind: "block", name: node.block, slot, family: node.family ?? (node.local ? "custom" : null), args, path, line, span, module, parent };
  }
  const name = node.kind === "opaque" ? node.call : valueText(node);
  return { id: pathId(path), kind: "opaque", name, slot, family: "custom", args: [], path, line, span, module, parent };
}

const LAYER_BLOCKS = ["Block", "BlockTemplate"];

type Ends = { first: string; last: string };

// Blocks nested in a block's arguments (Attention's pos=RoPE()) hang off it; a layer's own
// slots run in the order a transformer layer runs them, and the layer itself is the group, not a
// box. Returns the boxes data enters and leaves through.
function addBlock(flow: Flow, node: GNode, slot: string | null, path: Path, parent: string | null): Ends {
  const layer = node.kind === "block" && LAYER_BLOCKS.includes(node.block) && parent !== null;
  const own = layer ? null : box(node, slot, path, parent);
  if (own) flow.boxes.push(own);
  if (node.kind !== "block") return { first: own!.id, last: own!.id };
  const names = Object.keys(node.args).filter((k) => node.args[k]!.kind === "block" || node.args[k]!.kind === "opaque");
  const ordered = [...SLOT_ORDER.filter((k) => names.includes(k)), ...names.filter((k) => !SLOT_ORDER.includes(k))];
  const row: Ends[] = [];
  for (const name of ordered) {
    const ends = addBlock(flow, node.args[name]!, name, [...path, name], parent);
    if (layer) {
      const before = row[row.length - 1];
      if (before) flow.arrows.push({ id: `${before.last}>${ends.first}`, from: before.last, to: ends.first, kind: "flow" });
      row.push(ends);
    } else if (own) {
      flow.arrows.push({ id: `${own.id}>${ends.first}`, from: own.id, to: ends.first, kind: "arg" });
    }
  }
  if (layer) {
    const first = row[0]?.first ?? pathId(path);
    return { first, last: row[row.length - 1]?.last ?? first };
  }
  return { first: own!.id, last: own!.id };
}

function fixed(id: string, name: string, slot: string, family: string, args: [string, string][]): Box {
  return { id, kind: "fixed", name, slot, family, args, path: null, line: null, span: null, module: id === "tok_emb" || id === "head" ? id : null, parent: null };
}

// A Decoder class: embedding, the layers, the final norm, the head.
// A class that fills a template: its filled slots, in the order the template runs them.
function buildFilled(cls: GClass): Flow {
  const flow: Flow = { boxes: [], groups: [], arrows: [] };
  let before: string | null = null;
  for (const slot of cls.slots ?? []) {
    const node = cls.args?.[slot];
    if (!node || (node.kind === "literal" && node.value === null)) continue; // an empty slot has no box
    const own = addBlock(flow, node, slot, [slot], null);
    if (before) flow.arrows.push({ id: `${before}>${own.first}`, from: before, to: own.first, kind: "flow" });
    before = own.last;
  }
  return flow;
}

export function buildFlow(cls: GClass): Flow {
  const flow: Flow = { boxes: [], groups: [], arrows: [] };
  if (!cls.representable || !cls.args) return flow;
  if (cls.kind === "filled") return buildFilled(cls);
  if (cls.kind !== "decoder") return flow;
  const args = cls.args;
  let before: string | null = null;
  const connect = (ends: Ends) => {
    if (before) flow.arrows.push({ id: `${before}>${ends.first}`, from: before, to: ends.first, kind: "flow" });
    before = ends.last;
  };

  const dModel = args["d_model"] ? valueText(args["d_model"]) : "?";
  flow.boxes.push(fixed("tok_emb", "TokenEmbedding", "tok_emb", "embedding", [["d_model", dModel]]));
  connect({ first: "tok_emb", last: "tok_emb" });
  if (args["pos_emb"]) connect(addBlock(flow, args["pos_emb"], "pos_emb", ["pos_emb"], null));

  const layers = args["n_layers"] ? valueText(args["n_layers"]) : "?";
  if (args["pattern"]?.kind === "list") {
    const items = args["pattern"].items;
    flow.groups.push({ id: "layers", label: `pattern of ${items.length}, repeated through ${layers} layers`, path: ["pattern"] });
    const ends = items.map((item, i) => addBlock(flow, item, null, ["pattern", i], "layers"));
    ends.forEach((e, i) => {
      const next = ends[i + 1];
      if (next) flow.arrows.push({ id: `${e.last}>${next.first}`, from: e.last, to: next.first, kind: "flow" });
    });
    if (ends.length) connect({ first: ends[0]!.first, last: ends[ends.length - 1]!.last });
  } else if (args["block"]) {
    flow.groups.push({ id: "layers", label: `${layers} × layer`, path: ["block"] });
    connect(addBlock(flow, args["block"], null, ["block"], "layers"));
  }
  if (args["final_norm"]) connect(addBlock(flow, args["final_norm"], "final_norm", ["final_norm"], null));
  flow.boxes.push(fixed("head", "Head", "head", "head", []));
  connect({ first: "head", last: "head" });
  return flow;
}

// The node at `path` (an argument name selects an argument of the current call, an integer an
// item of the current list), or null.
export function nodeAt(cls: GClass, path: Path): GNode | null {
  let current: GNode | undefined;
  let args: Record<string, GNode> | undefined = cls.args;
  for (const step of path) {
    if (typeof step === "string") {
      current = args?.[step];
    } else {
      current = current?.kind === "list" ? current.items[step] : undefined;
    }
    if (!current) return null;
    args = current.kind === "block" ? current.args : undefined;
  }
  return current ?? null;
}
