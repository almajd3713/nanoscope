// A study's state and counts from its runs. Runs that have no folder yet are queued or not
// started; the plan says how many there should be.
export type RunLine = { ref: string; state: string; step: number; max_steps: number; val_bpb: number | null };

export type StudyCounts = {
  total: number;
  done: number;
  running: number;
  failed: number;
  // not started: planned but no run folder yet, or queued
  waiting: number;
};

export function studyCounts(runs: RunLine[], total: number): StudyCounts {
  const count = (...states: string[]) => runs.filter((r) => states.includes(r.state)).length;
  const done = count("done");
  const running = count("running", "preparing");
  const failed = count("failed");
  return { total, done, running, failed, waiting: Math.max(0, total - done - running - failed - count("stopped", "cancelled")) };
}

// running while any run runs; done when every run is; failed/stopped when nothing is moving.
export function studyState(c: StudyCounts, runs: RunLine[]): "queued" | "running" | "done" | "failed" | "stopped" {
  if (c.running > 0) return "running";
  if (c.total > 0 && c.done === c.total) return "done";
  if (c.failed > 0) return "failed";
  if (runs.some((r) => r.state === "stopped" || r.state === "cancelled")) return "stopped";
  return c.done > 0 ? "stopped" : "queued";
}

// studies/<name>/<variant>/seed-<n>
export function parseRef(ref: string, study: string): { variant: string; seed: number } | null {
  const prefix = `studies/${study}/`;
  if (!ref.startsWith(prefix)) return null;
  const m = /^(.+)\/seed-(\d+)$/.exec(ref.slice(prefix.length));
  return m ? { variant: m[1]!, seed: Number(m[2]) } : null;
}
