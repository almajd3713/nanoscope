import type { ButtonHTMLAttributes } from "react";
import { Link, type LinkProps } from "react-router-dom";
import styles from "./Button.module.css";

type Look = {
  // primary: the action the page exists for (one per view region); danger is bordered, never filled
  variant?: "primary" | "secondary" | "quiet" | "danger";
  // lg only for a lesson page's single primary action
  size?: "md" | "lg" | "sm";
};

function classes({ variant = "secondary", size = "md" }: Look): string {
  return [styles.btn, variant !== "secondary" && styles[variant], size !== "md" && styles[size], "body-strong"]
    .filter(Boolean)
    .join(" ");
}

// The label says what happens to what ("Train 3 seeds", "Stop run"), in sentence case.
export function Button({ variant, size, className, ...rest }: Look & ButtonHTMLAttributes<HTMLButtonElement>) {
  return <button type="button" className={[classes({ variant, size }), className].filter(Boolean).join(" ")} {...rest} />;
}

export function ButtonLink({ variant, size, className, ...rest }: Look & LinkProps) {
  return <Link className={[classes({ variant, size }), className].filter(Boolean).join(" ")} {...rest} />;
}
