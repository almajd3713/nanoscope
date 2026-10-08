import styles from "./Readout.module.css";

type Props = { label: string; value: string; unit?: string; sub?: string };

// The one live number a page is about. At most two per page.
export function Readout({ label, value, unit, sub }: Props) {
  return (
    <div className={styles.readout}>
      <span className={`label ${styles.muted}`}>{label}</span>
      <span className={styles.line}>
        <span className="readout">{value}</span>
        {unit && <span className={`value ${styles.muted}`}>{unit}</span>}
      </span>
      {sub && <span className={`small ${styles.muted}`}>{sub}</span>}
    </div>
  );
}
