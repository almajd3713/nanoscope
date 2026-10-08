import { useId } from "react";
import { X } from "../icons";
import styles from "./Field.module.css";

type Props = {
  label: string;
  value: string;
  onChange?: (value: string) => void;
  // Enter or leaving the field: the value is final
  onCommit?: () => void;
  help?: string;
  disabled?: boolean;
  // a problem the library found with this field, word for word
  problem?: string;
  // "changed from 4": shown when the value differs from where it started
  changedFrom?: string;
  mono?: boolean;
};

export function Field({ label, value, onChange, onCommit, help, disabled, problem, changedFrom, mono = true }: Props) {
  const id = useId();
  const describedBy = [help && `${id}-help`, problem && `${id}-problem`].filter(Boolean).join(" ") || undefined;
  return (
    <div className={`${styles.field} ${problem ? styles.problem : ""}`}>
      <label className="label" htmlFor={id}>
        {label}
      </label>
      <input
        id={id}
        className={`${styles.input} ${mono ? "value" : "body"}`}
        value={value}
        disabled={disabled}
        aria-invalid={problem ? true : undefined}
        aria-describedby={describedBy}
        onChange={(e) => onChange?.(e.target.value)}
        onBlur={onCommit}
        onKeyDown={onCommit ? (e) => e.key === "Enter" && onCommit() : undefined}
      />
      {help && (
        <span id={`${id}-help`} className={`caption ${styles.muted}`}>
          {help}
        </span>
      )}
      {changedFrom !== undefined && <span className={`caption ${styles.changed}`}>changed from {changedFrom || "(empty)"}</span>}
      {problem && (
        <span id={`${id}-problem`} className={`small ${styles.message}`}>
          <X size={16} aria-hidden="true" />
          <span>{problem}</span>
        </span>
      )}
    </div>
  );
}
