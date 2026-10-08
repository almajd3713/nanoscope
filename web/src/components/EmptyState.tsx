import type { ReactNode } from "react";
import { useShowCommands } from "../app/settings";
import styles from "./EmptyState.module.css";

type Props = { title: string; body: string; action?: ReactNode; command?: string };

// What is missing, what will appear once it exists, the one action that creates it, and the
// command (when Settings shows commands). No illustration, no "nothing here yet".
export function EmptyState({ title, body, action, command }: Props) {
  const showCommands = useShowCommands();
  return (
    <section className={styles.empty}>
      <h2 className="heading">{title}</h2>
      <p className={`body ${styles.muted}`}>{body}</p>
      {action && <div>{action}</div>}
      {command && showCommands && <pre className={`${styles.command} code-small`}>{command}</pre>}
    </section>
  );
}
