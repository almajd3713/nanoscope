import styles from "./Progress.module.css";

type Props = { step: number; total: number; unit?: string; eta?: string; extra?: string; label: string };

// A determinate line for anything with a known end. The fill is ink, not the accent: it is a
// measurement, not a control. No indeterminate animation.
export function Progress({ step, total, unit = "step", eta, extra, label }: Props) {
  const fraction = total > 0 ? Math.min(1, step / total) : 0;
  return (
    <div className={styles.progress}>
      <div
        className={styles.track}
        role="progressbar"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={total}
        aria-valuenow={Math.min(step, total)}
      >
        <div className={styles.fill} style={{ width: `${fraction * 100}%` }} />
      </div>
      <div className={`small ${styles.text}`}>
        <span className="value">
          {unit} {step} of {total}
        </span>
        {eta && <span className={`value ${styles.muted}`}>{eta} left</span>}
        {extra && <span className={`value ${styles.muted}`}>{extra}</span>}
      </div>
    </div>
  );
}
