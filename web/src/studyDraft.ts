// The study builder's form state and its conversion to and from a study spec (the JSON the API
// takes in `spec`). Pure, so the round trip is tested without a page. The library judges the
// values (POST /api/validate/study); this only gives typed text the JSON type it names.
import { parseValue, showValue } from "./runRequest";

export type VariantDraft = {
  name: string;
  model: string;
  // "ffn_hidden = 384, rope = false"
  kwargs: string;
  // a prediction for val_bpb, as typed ("" = none)
  predicted: string;
};

export type StudyDraft = {
  name: string;
  preset: string;
  // "4000000" tokens; "" = the preset's own steps. A FLOPs budget is kept as it was read.
  budget: string;
  seeds: string;
  baseline: string;
  match: "params" | "none";
  // percent, as typed: "2"
  tolerance: string;
  mode: "explore" | "record";
  variants: VariantDraft[];
  // spec keys the builder does not edit, carried through unchanged (so a round trip loses nothing)
  rest: Record<string, unknown>;
};

type Spec = Record<string, unknown>;
const EDITED = ["schema", "name", "preset", "budget", "seeds", "baseline", "match", "tolerance", "mode", "variants", "predictions"];

// "a = 1, b = [1, 2], c = false" -> pairs; commas inside brackets or quotes do not split.
export function splitPairs(text: string): [string, string][] {
  const parts: string[] = [];
  let depth = 0;
  let quote = "";
  let current = "";
  for (const ch of text) {
    if (quote) {
      if (ch === quote) quote = "";
    } else if (ch === '"' || ch === "'") {
      quote = ch;
    } else if (ch === "[" || ch === "{") {
      depth++;
    } else if (ch === "]" || ch === "}") {
      depth--;
    } else if (ch === "," && depth === 0) {
      parts.push(current);
      current = "";
      continue;
    }
    current += ch;
  }
  parts.push(current);
  return parts
    .map((p) => p.trim())
    .filter(Boolean)
    .map((p) => {
      const i = p.indexOf("=");
      return i < 0 ? [p, ""] : [p.slice(0, i).trim(), p.slice(i + 1).trim()];
    });
}

export function parseKwargs(text: string): Record<string, unknown> {
  return Object.fromEntries(splitPairs(text).map(([k, v]) => [k, parseValue(v, null)]));
}

export function showKwargs(kwargs: Record<string, unknown>): string {
  return Object.entries(kwargs)
    .map(([k, v]) => `${k} = ${showValue(v)}`)
    .join(", ");
}

// "0, 1, 2" -> [0, 1, 2]; "3" -> [0, 1, 2]; anything else is sent as typed for the API to refuse.
export function parseSeedList(text: string): unknown {
  const t = text.trim();
  if (/^\d+$/.test(t)) return Array.from({ length: Number(t) }, (_, i) => i);
  if (/^\d+(\s*,\s*\d+)*$/.test(t)) return t.split(",").map((x) => Number(x.trim()));
  return t;
}

export function fromSpec(spec: Spec): StudyDraft {
  const budget = spec["budget"] as Record<string, number> | undefined;
  const predictions = (spec["predictions"] ?? {}) as Record<string, Record<string, number>>;
  const rest = Object.fromEntries(Object.entries(spec).filter(([k]) => !EDITED.includes(k)));
  if (budget && !("tokens" in budget)) rest["budget"] = budget; // a FLOPs budget
  return {
    name: String(spec["name"] ?? ""),
    preset: String(spec["preset"] ?? "tinystories-5min"),
    budget: budget && "tokens" in budget ? String(budget["tokens"]) : "",
    seeds: ((spec["seeds"] as number[] | undefined) ?? [0, 1, 2]).join(", "),
    baseline: String(spec["baseline"] ?? ""),
    match: spec["match"] === "params" ? "params" : "none",
    tolerance: String(((spec["tolerance"] as number | undefined) ?? 0.02) * 100),
    mode: spec["mode"] === "record" ? "record" : "explore",
    variants: ((spec["variants"] as Spec[] | undefined) ?? []).map((v) => ({
      name: String(v["name"]),
      model: String(v["model"]),
      kwargs: showKwargs((v["kwargs"] as Record<string, unknown> | undefined) ?? {}),
      predicted: predictions[String(v["name"])]?.["val_bpb"]?.toString() ?? "",
    })),
    rest,
  };
}

export function toSpec(draft: StudyDraft): Spec {
  const spec: Spec = { schema: 1, name: draft.name, preset: draft.preset };
  Object.assign(spec, draft.rest);
  if (draft.budget.trim() !== "") spec["budget"] = { tokens: Number(draft.budget.replace(/[,_\s]/g, "")) };
  if (draft.match === "params") spec["match"] = "params";
  if (draft.baseline) spec["baseline"] = draft.baseline;
  spec["seeds"] = parseSeedList(draft.seeds);
  spec["mode"] = draft.mode;
  spec["tolerance"] = Number((Number(draft.tolerance) / 100).toPrecision(6));
  spec["variants"] = draft.variants.map((v) => {
    const kwargs = parseKwargs(v.kwargs);
    return { name: v.name, model: v.model, ...(Object.keys(kwargs).length ? { kwargs } : {}) };
  });
  const predictions = Object.fromEntries(
    draft.variants.filter((v) => v.predicted.trim() !== "").map((v) => [v.name, { val_bpb: Number(v.predicted) }]),
  );
  if (Object.keys(predictions).length) spec["predictions"] = predictions;
  return spec;
}

// The first free "variant-N" name.
export function freshName(variants: VariantDraft[]): string {
  let n = variants.length + 1;
  while (variants.some((v) => v.name === `variant-${n}`)) n++;
  return `variant-${n}`;
}
