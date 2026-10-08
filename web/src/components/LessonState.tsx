import { Check, Circle, CircleHalf, LockSimple, X } from "../icons";
import { Tag } from "./Tag";

// progress.STATES verbatim; a not-started lesson with unmet prerequisites reads `locked`, as
// `nanoscope learn list` prints it. The lock is advice: the lesson still opens.
export function LessonState({ state, lockedBy }: { state: string; lockedBy: string[] }) {
  if (state === "not-started" && lockedBy.length > 0) return <Tag tone="muted" icon={LockSimple}>locked</Tag>;
  switch (state) {
    case "passed":
      return <Tag tone="good" icon={Check}>{state}</Tag>;
    case "failed":
      return <Tag tone="bad" icon={X}>{state}</Tag>;
    case "started":
    case "checking":
      return <Tag tone="neutral" icon={CircleHalf}>{state}</Tag>;
    default:
      return <Tag tone="muted" icon={Circle}>{state}</Tag>;
  }
}
