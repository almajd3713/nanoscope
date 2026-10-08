import { useState } from "react";
import { useShowCommands } from "../app/settings";
import styles from "./EquivalentCommand.module.css";

type Props = {
  cli?: string;
  python?: string;
  label?: string;
};

// The CLI command or Python call that does what the screen does, built from the same request the
// screen sends. Renders nothing (and takes no space) unless Settings turns command-line
// equivalents on.
export function EquivalentCommand({ cli, python, label = "Equivalent command" }: Props) {
  const on = useShowCommands();
  const kinds = [cli !== undefined && "CLI", python !== undefined && "Python"].filter(Boolean) as ("CLI" | "Python")[];
  const [chosen, setChosen] = useState<"CLI" | "Python">("CLI");
  const [copied, setCopied] = useState<"idle" | "done" | "failed">("idle");
  if (!on || kinds.length === 0) return null;

  const kind = kinds.includes(chosen) ? chosen : kinds[0]!;
  const text = kind === "CLI" ? cli! : python!;

  const copy = () => {
    // Inside the click handler: browsers only allow the clipboard from a user gesture.
    const write = navigator.clipboard?.writeText(text);
    if (!write) {
      setCopied("failed");
      return;
    }
    write.then(
      () => setCopied("done"),
      () => setCopied("failed"),
    );
  };

  return (
    <section className={styles.cmd} aria-label={label}>
      <div className={styles.head}>
        <span className="label">{label}</span>
        {kinds.length > 1 && (
          <span className={styles.kinds}>
            {kinds.map((k) => (
              <button
                key={k}
                type="button"
                className={`${styles.kind} label`}
                aria-pressed={k === kind}
                onClick={() => setChosen(k)}
              >
                {k}
              </button>
            ))}
          </span>
        )}
        <span className={styles.spacer} />
        <button type="button" className={`${styles.copy} small`} onClick={copy}>
          {copied === "done" ? "Copied" : copied === "failed" ? "Select and copy" : "Copy"}
        </button>
      </div>
      <pre className={`${styles.code} code-small`}>{text}</pre>
    </section>
  );
}
