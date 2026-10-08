import type { ComponentType } from "react";
import styles from "./Tag.module.css";

type IconType = ComponentType<{ size?: number; "aria-hidden"?: boolean | "true" | "false" }>;

type Props = {
  tone: "neutral" | "muted" | "good" | "warn" | "bad";
  icon: IconType;
  // the state's own word, from the library
  children: string;
};

// A state is always a glyph and a word, never colour alone.
export function Tag({ tone, icon: Glyph, children }: Props) {
  return (
    <span className={`${styles.tag} ${styles[tone]} caption`}>
      <Glyph size={14} aria-hidden="true" />
      {children}
    </span>
  );
}
