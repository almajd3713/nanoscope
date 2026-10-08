import type { ReactNode } from "react";
import { ApiProblem } from "../api/problem";
import { X } from "../icons";
import styles from "./ProblemView.module.css";

type Props = {
  status?: number;
  title: string;
  detail: string;
  // actions the library names (a lesson link, a command)
  next?: ReactNode;
};

// An API error: the title, the status code and the detail word for word, never apologising.
export function ProblemView({ status, title, detail, next }: Props) {
  return (
    <div className={styles.problem} role="alert">
      <div className={`${styles.head} body-strong`}>
        <X size={16} aria-hidden="true" />
        <span>{title}</span>
        {status !== undefined && <span className={`${styles.status} value`}>{status}</span>}
      </div>
      {detail && <pre className={`${styles.detail} code`}>{detail}</pre>}
      {next && <div className={styles.next}>{next}</div>}
    </div>
  );
}

export function ProblemFromError({ error }: { error: unknown }) {
  if (error instanceof ApiProblem) {
    return <ProblemView status={error.status} title={error.title} detail={error.detail} />;
  }
  const detail = error instanceof Error ? error.message : String(error);
  return <ProblemView title="The page failed to draw" detail={detail} />;
}
