import type { ReactNode } from "react";
import { Check, Info, Minus, X } from "../icons";
import styles from "./Verdict.module.css";

// The library's verdict word, with its glyph. `within noise` is a result, not a failure: it is
// never red or amber. The word comes from rows[].verdict verbatim.
export function Verdict({ verdict, children }: { verdict: string; children?: ReactNode }) {
  let word: ReactNode;
  switch (verdict) {
    case "better":
      word = (
        <span className={`${styles.word} ${styles.good} body`}>
          <Check size={16} aria-hidden="true" />
          better
        </span>
      );
      break;
    case "worse":
      word = (
        <span className={`${styles.word} ${styles.bad} body`}>
          <X size={16} aria-hidden="true" />
          worse
        </span>
      );
      break;
    case "within noise":
      word = (
        <span className={`${styles.word} ${styles.muted} body`}>
          <Minus size={16} aria-hidden="true" />
          within noise
        </span>
      );
      break;
    case "no CI":
      word = (
        <span className={`${styles.word} ${styles.muted} body`}>
          <Info size={16} aria-hidden="true" />
          no CI
        </span>
      );
      break;
    case "baseline":
      word = <span className={`${styles.word} ${styles.muted} body`}>baseline</span>;
      break;
    default:
      throw new Error(`unknown verdict ${JSON.stringify(verdict)}: add it to the design system first`);
  }
  return (
    <span className={styles.verdict}>
      {word}
      {children && <span className="body">{children}</span>}
    </span>
  );
}
