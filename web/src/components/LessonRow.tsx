import { Link } from "react-router-dom";
import { LessonState } from "./LessonState";
import styles from "./LessonRow.module.css";

type Props = {
  to: string;
  number: string;
  title: string;
  summary: string;
  unlocks: string[];
  estimate: string;
  state: string;
  lockedBy: string[];
};

// One lesson in its path, as GET /api/curricula returns it.
export function LessonRow({ to, number, title, summary, unlocks, estimate, state, lockedBy }: Props) {
  const locked = state === "not-started" && lockedBy.length > 0;
  return (
    <Link to={to} className={styles.row}>
      <span className="value">{number}</span>
      <span className={styles.main}>
        <span className="body-strong">{title}</span>
        <span className={`small ${styles.muted}`}>{summary}</span>
      </span>
      <span className={styles.unlocks}>
        {unlocks.map((u) => (
          <span key={u} className="value">
            {u}
          </span>
        ))}
      </span>
      <span className="value">{estimate}</span>
      <span className={styles.state}>
        <LessonState state={state} lockedBy={lockedBy} />
        {locked && <span className={`caption ${styles.needs}`}>needs {lockedBy.join(", ")}</span>}
      </span>
    </Link>
  );
}
