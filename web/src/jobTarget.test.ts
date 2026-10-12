import { describe, expect, it } from "vitest";
import jobs from "../test/fixtures/jobs.json";
import { jobTarget } from "./jobTarget";

describe("jobTarget", () => {
  it("shows a workspace file the way you wrote it, and leaves other targets alone", () => {
    const describeJob = { ref: null, payload: { model: "/srv/ws/models/my_lm.py:MyLM" } };
    expect(jobTarget(describeJob, "/srv/ws")).toBe("models/my_lm.py:MyLM");
    expect(jobTarget(describeJob)).toBe("/srv/ws/models/my_lm.py:MyLM");
    expect(jobTarget({ ref: "studies/m1/gpt2/seed-0", payload: {} }, "/srv/ws")).toBe("studies/m1/gpt2/seed-0");
    for (const j of jobs as { ref: string | null; payload: Record<string, unknown> }[]) expect(typeof jobTarget(j, "/nowhere")).toBe("string");
  });
});
