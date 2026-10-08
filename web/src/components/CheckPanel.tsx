import { useLessonCheck } from "../useLessonCheck";
import { CheckResult } from "./CheckResult";
import { ProblemView } from "./ProblemView";

type JobView = NonNullable<ReturnType<typeof useLessonCheck>["job"]>;

export function CheckPanel({ lessonTitle, job, result, checkDefs, unlocks }: {
  lessonTitle: string;
  job: JobView;
  result: ReturnType<typeof useLessonCheck>["result"];
  checkDefs: { id: string; kind: string }[];
  unlocks: string[];
}) {
  const kinds = new Map(checkDefs.map((c) => [c.id, c.kind]));
  if (job.state === "queued" || job.state === "running") {
    return (
      <CheckResult
        status="running"
        title={lessonTitle}
        checks={checkDefs.map((c) => ({ id: c.id, kind: c.kind, passed: null, reason: "not run yet" }))}
        progress={job.state === "queued" ? "Waiting for a worker to start the check" : "Running the checks"}
      />
    );
  }
  if (job.state === "done" && result) {
    return (
      <CheckResult
        status={result.passed ? "passed" : "failed"}
        title={lessonTitle}
        checks={result.checks.map((c) => ({ ...c, kind: kinds.get(c.id) ?? "" }))}
      >
        {result.passed
          ? unlocks.length
            ? `Check passed. ${unlocks.map((u) => u.split(":").pop()).join(", ")} ${unlocks.length === 1 ? "is" : "are"} unlocked.`
            : "Check passed."
          : "Read the reasons, edit your file, and run the check again."}
      </CheckResult>
    );
  }
  return (
    <ProblemView
      title={job.state === "cancelled" ? "The check was cancelled" : "The check did not finish"}
      detail={job.error ?? ""}
    />
  );
}
