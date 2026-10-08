import * as ToggleGroup from "@radix-ui/react-toggle-group";
import { useLevel } from "../app/level";
import { shows } from "../levels";
import type { Depth } from "./layout";
import styles from "./DepthDial.module.css";

const OPTIONS: { value: Depth; label: string; control: "graph.surface" | "graph.detailed" | "graph.research" }[] = [
  { value: "surface", label: "Surface", control: "graph.surface" },
  { value: "detailed", label: "Detailed", control: "graph.detailed" },
  { value: "research", label: "Research", control: "graph.research" },
];

// How much each box says: surface is the block diagram, detailed adds shapes, parameters and the
// arguments from the last trace, research adds where each call sits in the file and whether it
// matches its reference. A level only adds options.
export function DepthDial({ value, onChange }: { value: Depth; onChange: (depth: Depth) => void }) {
  const level = useLevel();
  return (
    <ToggleGroup.Root
      type="single"
      value={value}
      className={styles.dial}
      aria-label="Depth"
      onValueChange={(v) => v && onChange(v as Depth)}
    >
      {OPTIONS.filter((o) => shows(o.control, level)).map((o) => (
        <ToggleGroup.Item key={o.value} value={o.value} className={`${styles.item} body`}>
          {o.label}
        </ToggleGroup.Item>
      ))}
    </ToggleGroup.Root>
  );
}

// The depth a page opens at: what the level can show, never more.
export function defaultDepth(level: Parameters<typeof shows>[1]): Depth {
  return shows("graph.detailed", level) ? "detailed" : "surface";
}
