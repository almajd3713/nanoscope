import { Check, Circle, CircleHalf, Pause, Play, Prohibit, X } from "../icons";
import { Tag } from "./Tag";

// A run's or job's state as a glyph and the state's own word (status.STATES, or the queue).
// Running is not coloured: the step count and progress line beside it carry the live signal.
export function StateTag({ state }: { state: string }) {
  switch (state) {
    case "queued":
      return <Tag tone="muted" icon={Circle}>queued</Tag>;
    case "preparing":
      return <Tag tone="neutral" icon={CircleHalf}>preparing</Tag>;
    case "running":
      return <Tag tone="neutral" icon={Play}>running</Tag>;
    case "done":
      return <Tag tone="good" icon={Check}>done</Tag>;
    case "stopped":
      return <Tag tone="warn" icon={Pause}>stopped</Tag>;
    case "cancelling":
      return <Tag tone="warn" icon={Prohibit}>cancelling</Tag>;
    case "cancelled":
      return <Tag tone="muted" icon={Prohibit}>cancelled</Tag>;
    case "failed":
      return <Tag tone="bad" icon={X}>failed</Tag>;
    default:
      throw new Error(`unknown state ${JSON.stringify(state)}: add it to the design system first`);
  }
}
