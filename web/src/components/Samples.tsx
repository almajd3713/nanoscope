import { useState } from "react";
import { Tabs } from "./Tabs";
import styles from "./Samples.module.css";

export type Sample = { step: number; text: string };

// What the model wrote at each sample step, oldest to newest. The newest is shown until you
// pick another; a new sample does not move you off the one you chose.
export function Samples({ samples, caption }: { samples: Sample[]; caption?: string }) {
  const [chosen, setChosen] = useState<string | null>(null);
  if (samples.length === 0) return null;
  const latest = String(samples[samples.length - 1]!.step);
  const value = chosen !== null && samples.some((s) => String(s.step) === chosen) ? chosen : latest;
  return (
    <section className={styles.samples} aria-label="Samples">
      <div className={styles.head}>
        <h2 className="heading">Samples</h2>
        {caption && <span className={`small ${styles.muted}`}>{caption}</span>}
      </div>
      <Tabs
        value={value}
        onValueChange={setChosen}
        items={samples.map((s) => ({
          id: String(s.step),
          label: `step ${s.step}`,
          content: <p className={`${styles.text} code`}>{s.text}</p>,
        }))}
      />
    </section>
  );
}
