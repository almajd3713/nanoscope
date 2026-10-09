// Turns what is typed in the run form into the request the API takes. Pure, so it is tested
// without a page. The library judges the values (POST /api/validate/run): this only gives
// them the JSON type their annotation names, and leaves the rest as typed.

export function parseValue(text: string, annotation: string | null): unknown {
  const options = (annotation ?? "").split("|").map((o) => o.trim()).filter(Boolean);
  const t = text.trim();
  if (options.includes("None") && (t === "" || t.toLowerCase() === "none" || t.toLowerCase() === "null")) {
    return null;
  }
  if (options.includes("bool") && (t === "true" || t === "false")) return t === "true";
  const numeric = options.includes("int") || options.includes("float");
  if (numeric && t !== "" && Number.isFinite(Number(t))) return Number(t);
  if (options.length === 0 || options.some((o) => !["int", "float", "bool", "str", "None"].includes(o))) {
    try {
      return JSON.parse(t);
    } catch {
      return text;
    }
  }
  return text;
}

// The text a default shows as in a field.
export function showValue(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "string") return value;
  return JSON.stringify(value);
}

// "3" -> 3 (seeds 0..2), "0, 4, 7" -> [0, 4, 7], anything else is sent as typed for the API to refuse.
export function parseSeeds(text: string): unknown {
  const t = text.trim();
  if (/^\d+$/.test(t)) return Number(t);
  if (/^\d+(\s*,\s*\d+)+$/.test(t)) return t.split(",").map((x) => Number(x.trim()));
  return t;
}

// "100, 500" -> [100, 500]; empty -> []; anything else is null (the form says so, nothing is sent).
export function parseSteps(text: string): number[] | null {
  const t = text.trim();
  if (t === "") return [];
  if (/^\d+(\s*,\s*\d+)*$/.test(t)) return t.split(",").map((x) => Number(x.trim()));
  return null;
}

export function seedList(seeds: unknown): number[] {
  if (typeof seeds === "number") return Array.from({ length: seeds }, (_, i) => i);
  if (Array.isArray(seeds)) return seeds as number[];
  return [];
}

export type Param = { name: string; annotation: string | null; default: unknown; required: boolean; from_data: boolean };

// Every field whose current text differs from its default becomes a keyword; the rest is not sent.
export function buildKwargs(
  fields: { name: string; annotation: string | null; default: unknown }[],
  current: (name: string) => string,
  extra: { key: string; value: string }[],
): Record<string, unknown> {
  const kwargs: Record<string, unknown> = {};
  for (const f of fields) {
    const text = current(f.name);
    if (text !== showValue(f.default)) kwargs[f.name] = parseValue(text, f.annotation);
  }
  for (const { key, value } of extra) {
    if (key.trim()) kwargs[key.trim()] = parseValue(value, null);
  }
  return kwargs;
}

// The CLI and Python calls for the same request.
export function commands(opts: {
  model: string;
  shippedClass: string | null;
  preset: string;
  seeds: unknown;
  kwargs: Record<string, unknown>;
  checkpointSteps?: number[];
}): { cli: string; python: string | undefined } {
  const lit = (v: unknown) => (typeof v === "string" ? `"${v}"` : v === null ? "None" : v === true ? "True" : v === false ? "False" : JSON.stringify(v));
  const seeds = typeof opts.seeds === "number" ? opts.seeds : undefined;
  const steps = opts.checkpointSteps ?? [];
  const sets = Object.entries(opts.kwargs).map(([k, v]) => `${k}=${showValue(v) || "None"}`);
  const cli = [
    `nanoscope run ${opts.model} --preset ${opts.preset}`,
    seeds !== undefined && seeds > 1 ? `--seeds ${seeds}` : "",
    steps.length ? `--checkpoint-steps ${steps.join(",")}` : "",
    sets.length ? `--set ${sets.join(" ")}` : "",
  ].filter(Boolean).join(" ");
  const python = opts.shippedClass
    ? [
        "from nanoscope import run",
        `from nanoscope.models import ${opts.shippedClass}`,
        "",
        `run(${[
          opts.shippedClass,
          `preset="${opts.preset}"`,
          seeds !== undefined && seeds > 1 ? `seeds=${seeds}` : "",
          steps.length ? `checkpoint_steps=[${steps.join(", ")}]` : "",
          ...Object.entries(opts.kwargs).map(([k, v]) => `${k}=${lit(v)}`),
        ].filter(Boolean).join(", ")})`,
      ].join("\n")
    : undefined;
  return { cli, python };
}
