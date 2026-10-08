import { countText } from "../format";

// The rows of a describe result (describe.v1 `modules`) a box reads its numbers from.
export type TraceRow = {
  path: string;
  input_shapes: number[][] | null;
  output_shapes: number[][] | null;
  params: number;
  flops_per_token: number;
};
export type Trace = { shape: string; params: string; flops: string };

const shapeText = (shapes: number[][] | null): string => (shapes && shapes[0] ? shapes[0].join("x") : "");

// "8x256x128" when a block keeps the shape, "8x256 → 8x256x128" when it changes it, as describe prints.
export function traceOf(rows: TraceRow[] | null | undefined, module: string | null): Trace | null {
  if (!rows || !module) return null;
  const row = rows.find((r) => r.path === module);
  if (!row) return null;
  const input = shapeText(row.input_shapes);
  const output = shapeText(row.output_shapes);
  const shape = input && output && input !== output ? `${input} → ${output}` : output || input;
  return { shape, params: countText(row.params), flops: `${countText(row.flops_per_token)} FLOP/token` };
}

// The span of a call as a person reads it: 1-based columns, "line 10, cols 30 to 41".
export function spanText(span: { line: number; col: number; end_line: number; end_col: number } | null): string {
  if (!span) return "";
  if (span.line === span.end_line) return `line ${span.line}, cols ${span.col + 1} to ${span.end_col + 1}`;
  return `line ${span.line}, col ${span.col + 1} to line ${span.end_line}, col ${span.end_col + 1}`;
}
