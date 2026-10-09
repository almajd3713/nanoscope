import { describe, expect, it } from "vitest";
import { parseRef, studyCounts, studyState, type RunLine } from "./studyState";

const run = (state: string, ref = "studies/s/a/seed-0"): RunLine => ({ ref, state, step: 0, max_steps: 10, val_bpb: null });

describe("study state", () => {
  it("counts finished, running and waiting runs against the plan", () => {
    const runs = [run("done"), run("running"), run("failed")];
    expect(studyCounts(runs, 8)).toEqual({ total: 8, done: 1, running: 1, failed: 1, waiting: 5 });
  });

  it("is running while any run runs, done when all are, stopped when it halted part-way", () => {
    expect(studyState(studyCounts([run("running"), run("done")], 4), [run("running"), run("done")])).toBe("running");
    const all = [run("done"), run("done")];
    expect(studyState(studyCounts(all, 2), all)).toBe("done");
    const halted = [run("done"), run("stopped")];
    expect(studyState(studyCounts(halted, 4), halted)).toBe("stopped");
    expect(studyState(studyCounts([], 4), [])).toBe("queued");
    const failed = [run("done"), run("failed")];
    expect(studyState(studyCounts(failed, 2), failed)).toBe("failed");
  });

  it("reads the variant and seed out of a run ref, variant names with dashes included", () => {
    expect(parseRef("studies/m1-ablation/no-rope/seed-2", "m1-ablation")).toEqual({ variant: "no-rope", seed: 2 });
    expect(parseRef("studies/other/a/seed-0", "m1-ablation")).toBeNull();
  });
});
