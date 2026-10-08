import type { ReactNode } from "react";
import { Check, Circle, X } from "../icons";
import { Tag } from "./Tag";
import styles from "./CheckResult.module.css";

export type CheckLine = {
  id: string;
  kind: string;
  // true, false, or null for a check that has not run yet
  passed: boolean | null;
  // the library's reason, word for word
  reason: string;
};

type Props = {
  status: "passed" | "failed" | "running";
  title: string;
  checks: CheckLine[];
  // a live line while running ("2 of 3 checks")
  progress?: string;
  // what passing changed, or the next step after a failure
  children?: ReactNode;
};

export function CheckResult({ status, title, checks, progress, children }: Props) {
  return (
    <section className={`${styles.result} ${status === "passed" ? styles.passed : status === "failed" ? styles.failed : ""}`} aria-label="Check result">
      <header className={styles.head}>
        <h2 className="heading">{title}</h2>
        {status === "passed" && <Tag tone="good" icon={Check}>passed</Tag>}
        {status === "failed" && <Tag tone="bad" icon={X}>failed</Tag>}
        {status === "running" && <Tag tone="neutral" icon={Circle}>checking</Tag>}
        {progress && <span className={`small ${styles.muted}`}>{progress}</span>}
      </header>
      <ul className={styles.list}>
        {checks.map((c) => (
          <li key={c.id} className={styles.row}>
            {c.passed === true && <Check size={16} className={styles.good} aria-label="passed" />}
            {c.passed === false && <X size={16} className={styles.bad} aria-label="failed" />}
            {c.passed === null && <Circle size={16} className={styles.muted} aria-label="not run yet" />}
            <span className="value">{c.id}</span>
            <span className={`caption ${styles.muted}`}>{c.kind}</span>
            <span className={`small ${styles.reason}`}>{c.reason}</span>
          </li>
        ))}
      </ul>
      {children && <p className={`small ${styles.foot}`}>{children}</p>}
    </section>
  );
}
