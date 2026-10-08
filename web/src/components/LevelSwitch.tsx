import * as ToggleGroup from "@radix-ui/react-toggle-group";
import { LEVELS, type Level } from "../app/level";
import styles from "./LevelSwitch.module.css";

type Props = { value: Level; onChange: (level: Level) => void };

// Learn, Tinker, Research or Extend. It changes which controls are visible, nothing else.
export function LevelSwitch({ value, onChange }: Props) {
  return (
    <ToggleGroup.Root
      type="single"
      value={value}
      onValueChange={(v) => {
        // Radix reports "" when the pressed item is pressed again; a level is always chosen.
        if (v) onChange(v as Level);
      }}
      className={styles.switch}
      aria-label="Level"
    >
      {LEVELS.map((level) => (
        <ToggleGroup.Item key={level} value={level} className={`${styles.item} small`}>
          {level}
        </ToggleGroup.Item>
      ))}
    </ToggleGroup.Root>
  );
}
