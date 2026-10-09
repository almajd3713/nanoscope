import type { ReactNode } from "react";
import styles from "./Segmented.module.css";

type Item = { id: string; label: ReactNode; title?: string };

type Props = {
  label: string;
  items: Item[];
  value: string;
  onChange: (id: string) => void;
};

// A few exclusive choices side by side (a checkpoint step, one step or across steps).
export function Segmented({ label, items, value, onChange }: Props) {
  return (
    <div className={styles.seg} role="radiogroup" aria-label={label}>
      {items.map((item) => (
        <button
          key={item.id}
          type="button"
          role="radio"
          aria-checked={item.id === value}
          title={item.title}
          className="body"
          onClick={() => onChange(item.id)}
        >
          {item.label}
        </button>
      ))}
    </div>
  );
}
