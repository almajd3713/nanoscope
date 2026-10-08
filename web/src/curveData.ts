// Turns metrics rows into the aligned arrays uPlot draws. Only arranges values the API gave:
// no smoothing, no mean, no interval.
export type Row = { step: number; val_bpb?: number | null; [key: string]: unknown };

export type EvalPoints = { steps: number[]; values: number[] };

export function evalPoints(rows: Row[]): EvalPoints {
  const steps: number[] = [];
  const values: number[] = [];
  for (const r of rows) {
    if (typeof r.val_bpb === "number") {
      steps.push(r.step);
      values.push(r.val_bpb);
    }
  }
  return { steps, values };
}

export type Aligned = {
  x: number[];
  run: (number | null)[];
  // the shipped seeds' range at each step, where every seed has an eval; null elsewhere
  low: (number | null)[];
  high: (number | null)[];
};

export function align(run: EvalPoints, seeds: EvalPoints[]): Aligned {
  const x = [...new Set([...run.steps, ...seeds.flatMap((s) => s.steps)])].sort((a, b) => a - b);
  const at = (p: EvalPoints) => new Map(p.steps.map((s, i) => [s, p.values[i] as number]));
  const mine = at(run);
  const theirs = seeds.map(at);
  const low: (number | null)[] = [];
  const high: (number | null)[] = [];
  for (const step of x) {
    const vals = theirs.map((m) => m.get(step)).filter((v): v is number => v !== undefined);
    const whole = seeds.length > 0 && vals.length === seeds.length;
    low.push(whole ? Math.min(...vals) : null);
    high.push(whole ? Math.max(...vals) : null);
  }
  return { x, run: x.map((s) => mine.get(s) ?? null), low, high };
}
