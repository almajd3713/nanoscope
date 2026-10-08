import { useState } from "react";
import { Button } from "../components/Button";
import { Field } from "../components/Field";
import { DotsSixVertical, Plus, Trash } from "../icons";
import type { Edit } from "./api";
import { carriesBlock, DRAG_TYPE } from "./dnd";
import type { GClass } from "./flow";
import { addLayerEdit, parsePattern, patternItems, removeLayerEdit, stackOf } from "./stack";
import styles from "./StackPanel.module.css";

export const DRAG_LAYER = "application/x-nanoscope-layer";

type Props = { cls: GClass; onEdit: (edits: Edit[]) => void; busy?: boolean };

// The layers of the model as a list you can edit: drop a Block to add a layer, drag a layer out
// (to the trash) or press its button to remove one, or type a pattern such as sliding:global 3:1.
export function StackPanel({ cls, onEdit, busy }: Props) {
  const stack = stackOf(cls);
  const [pattern, setPattern] = useState("sliding:global 3:1");
  const [window, setWindow] = useState("64");
  const [over, setOver] = useState<"add" | "trash" | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  if (!stack) return null;

  const spec = parsePattern(pattern);
  const items = typeof spec === "string" ? spec : patternItems(cls, spec, Number(window));
  const preview = typeof spec === "string" || typeof items === "string" ? [] : [...Array(spec.counts[0]).fill(spec.first), ...Array(spec.counts[1]).fill(spec.second)] as string[];
  const why = typeof spec === "string" ? spec : typeof items === "string" ? items : null;

  const add = () => {
    const edit = addLayerEdit(cls);
    if (edit) onEdit([edit]);
  };
  const remove = (index: number) => onEdit([removeLayerEdit(cls, stack.kind === "pattern" ? index : null)]);

  const zone = (kind: "add" | "trash") => ({
    onDragOver: (e: React.DragEvent) => {
      if (kind === "add" ? carriesBlock(e) : Array.from(e.dataTransfer.types).includes(DRAG_LAYER)) {
        e.preventDefault();
        setOver(kind);
      }
    },
    onDragLeave: () => setOver(null),
    onDrop: (e: React.DragEvent) => {
      setOver(null);
      if (kind === "add") {
        const name = e.dataTransfer.getData(DRAG_TYPE);
        if (!name) return;
        e.preventDefault();
        if (name !== "Block") setProblem(`${name} is not a layer: drop a Block to add a layer`);
        else {
          setProblem(null);
          add();
        }
      } else {
        const index = e.dataTransfer.getData(DRAG_LAYER);
        if (index === "") return;
        e.preventDefault();
        remove(Number(index));
      }
    },
  });

  return (
    <section className={styles.panel} aria-label="Layers">
      <div>
        <h2 className="heading">Layers</h2>
        <p className={`small ${styles.muted}`}>
          {stack.kind === "uniform" ? `One block, ${stack.layers ?? "n"} layers.` : `A pattern of ${stack.items.length}, repeated through ${stack.layers ?? "n"} layers.`}
        </p>
      </div>
      <ul className={styles.layers}>
        {stack.items.map((layer) => (
          <li
            key={layer.index}
            className={styles.layer}
            draggable={stack.kind === "pattern" ? stack.items.length > 1 : (stack.layers ?? 2) > 1}
            onDragStart={(e) => {
              e.dataTransfer.setData(DRAG_LAYER, String(layer.index));
              e.dataTransfer.effectAllowed = "move";
            }}
          >
            <DotsSixVertical size={16} aria-hidden="true" />
            <span className="value">Block</span>
            <span className={`caption ${styles.muted} ${styles.grow}`}>{stack.kind === "pattern" ? `layer ${layer.index}` : `× ${stack.layers ?? "n"}`}</span>
            <span className="value">{layer.label}</span>
            <Button size="sm" variant="quiet" disabled={busy} aria-label={stack.kind === "pattern" ? `Remove layer ${layer.index}` : "Remove a layer"} onClick={() => remove(layer.index)}>
              <Trash size={16} aria-hidden="true" />
            </Button>
          </li>
        ))}
      </ul>
      <div className={`${styles.zone} ${over === "add" ? styles.zoneOver : ""}`} {...zone("add")} role="group" aria-label="Add a layer">
        <Plus size={16} aria-hidden="true" />
        <span className="small">Drop a Block here to add a layer</span>
        <Button size="sm" disabled={busy} onClick={add}>
          Add a layer
        </Button>
      </div>
      <div className={`${styles.zone} ${styles.trash} ${over === "trash" ? styles.zoneOver : ""}`} {...zone("trash")} role="group" aria-label="Remove a layer by dropping it here">
        <Trash size={16} aria-hidden="true" />
        <span className="small">Drag a layer here to remove it</span>
      </div>
      {problem && <p className="small" role="alert">{problem}</p>}

      <Field label="Pattern" value={pattern} onChange={setPattern} problem={typeof spec === "string" ? spec : undefined} help="kind:kind a:b means a layers of the first kind, then b of the second" />
      <Field label="sliding window" value={window} onChange={setWindow} problem={typeof spec !== "string" && typeof items === "string" ? items : undefined} help="the window of a sliding layer, in tokens" />
      {preview.length > 0 && (
        <div className={styles.chips} aria-label="This writes">
          {preview.map((kind, i) => (
            <span key={i} className={`${styles.chip} caption ${kind === "global" ? styles.global : ""}`}>
              {kind}
            </span>
          ))}
        </div>
      )}
      <div className={styles.row}>
        <Button
          variant="primary"
          disabled={busy || why !== null}
          onClick={() => {
            if (typeof items !== "string" && typeof spec !== "string") onEdit([{ op: "set_pattern", class: cls.name, items }]);
          }}
        >
          Write pattern
        </Button>
      </div>
    </section>
  );
}
