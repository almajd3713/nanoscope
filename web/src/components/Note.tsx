import type { ReactNode } from "react";
import { Info, Warning } from "../icons";
import styles from "./Note.module.css";

type Props = {
  // warn: the library's notes and cautions (default); info: a neutral explanation
  tone?: "warn" | "info";
  children: ReactNode;
};

export function Note({ tone = "warn", children }: Props) {
  const Glyph = tone === "warn" ? Warning : Info;
  return (
    <div className={`${styles.note} ${styles[tone]}`}>
      <Glyph className={styles.glyph} size={16} aria-hidden="true" />
      <p className={`${styles.text} small`}>{children}</p>
    </div>
  );
}
