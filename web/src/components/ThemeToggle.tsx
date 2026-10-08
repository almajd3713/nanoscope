import { useSyncExternalStore } from "react";
import { Moon, Sun } from "../icons";
import { resolvedTheme, setChoice, subscribe } from "../styles/theme";
import styles from "./IconButton.module.css";

// Shows the theme you would switch to: a moon on light, a sun on dark.
export function ThemeToggle() {
  const theme = useSyncExternalStore(subscribe, resolvedTheme);
  const next = theme === "dark" ? "light" : "dark";
  const Glyph = next === "dark" ? Moon : Sun;
  return (
    <button
      type="button"
      className={styles.button}
      aria-label={`Switch to the ${next} theme`}
      title={`Switch to the ${next} theme`}
      onClick={() => setChoice(next)}
    >
      <Glyph size={20} aria-hidden="true" />
    </button>
  );
}
