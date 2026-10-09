// The inspect.v1 report (nanoscope/schemas/inspect.v1.json) and the small calculations the page
// makes over it. Nothing here computes a model number: weights, probabilities and ranks are the
// job's own.

export type Token = { id: number; text: string };
export type Guess = { id: number; text: string; p: number };
export type Next = Guess & { rank: number };
export type LensLayer = { name: string; positions: { top: Guess[]; next: Next | null }[] };
export type Attention = { module: string; layer: number; heads: number; weights: number[][][] };
export type InspectReport = {
  schema: number;
  ref: string;
  model: string;
  step: number;
  steps: number[];
  prompt: string;
  tokens: Token[];
  attention: Attention[];
  lens: { top_k: number; layers: LensLayer[] };
  notes?: string[];
};

export const MAX_PROMPT_TOKENS = 64;

// A token as the page shows it: quoted, so a leading space is visible.
export const quote = (text: string): string => JSON.stringify(text);

// The heat scale: five steps of the design system, blended between (weights 0..1); null is the
// masked future.
export function heat(v: number | null): string {
  if (v === null) return "var(--surface-sunken)";
  const x = Math.min(Math.max(v, 0), 1) * 4;
  const i = Math.min(3, Math.floor(x));
  const f = Math.round((x - i) * 100);
  return `color-mix(in oklab, var(--heat-${i + 1}) ${100 - f}%, var(--heat-${i + 2}))`;
}

// Row t of a head holds t+1 weights (causal); the cell at (t, j) is null for j > t.
export function cell(weights: number[][], t: number, j: number): number | null {
  return j <= t ? (weights[t]?.[j] ?? null) : null;
}

// The n largest weights of one row: position, token, weight.
export function topWeights(row: number[], tokens: Token[], n = 5): { pos: number; text: string; w: number }[] {
  return row
    .map((w, pos) => ({ pos, text: tokens[pos]?.text ?? "", w }))
    .sort((a, b) => b.w - a.w || a.pos - b.pos)
    .slice(0, n);
}

export const weightText = (w: number): string => w.toFixed(4);
export const probText = (p: number): string => p.toFixed(3);

export const LEGEND = [
  { label: "0 – 0.2", color: "var(--heat-1)" },
  { label: "0.2 – 0.4", color: "var(--heat-2)" },
  { label: "0.4 – 0.6", color: "var(--heat-3)" },
  { label: "0.6 – 0.8", color: "var(--heat-4)" },
  { label: "0.8 – 1", color: "var(--heat-5)" },
  { label: "future (masked)", color: "var(--surface-sunken)" },
];

export function inspectCommand(ref: string, prompt: string, step: number | null): { cli: string; python: string } {
  const stepArg = step === null ? "" : ` --step ${step}`;
  const cli = `nanoscope inspect ${ref}${stepArg} \\\n  --prompt ${quote(prompt)}`;
  const python = [
    "from nanoscope.inspect import inspect_checkpoint",
    "",
    `report = inspect_checkpoint(${quote(ref)}, ${quote(prompt)}${step === null ? "" : `, step=${step}`})`,
  ].join("\n");
  return { cli, python };
}
