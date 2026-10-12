import type { ReactNode } from "react";
import { ButtonLink } from "./Button";
import styles from "./PhoneNotice.module.css";

type Props = { title: string; children?: ReactNode; back: { to: string; label: string }; actions?: ReactNode };

// What a page that needs a wide screen shows on a phone: why, and what can still be done from here.
export function PhoneNotice({ title, children, back, actions }: Props) {
  return (
    <section className={styles.notice} aria-label="Needs a wider screen">
      <h1 className="heading">{title}</h1>
      {children}
      <div className={styles.actions}>
        {actions}
        <ButtonLink to={back.to}>{back.label}</ButtonLink>
      </div>
    </section>
  );
}
