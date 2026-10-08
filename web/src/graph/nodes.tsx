import { Handle, type NodeProps, Position } from "@xyflow/react";
import type { Box } from "./flow";
import type { Depth } from "./layout";
import styles from "./nodes.module.css";

export type BoxData = { box: Box; depth: Depth; selected: boolean; [key: string]: unknown };
export type GroupData = { label: string; [key: string]: unknown };

// A box in the model graph: the block's class name in mono, then its slot and family in words.
// Nodes are neutral: no colour per family; the family is a word.
export function BlockNodeView({ data }: NodeProps) {
  const { box, depth, selected } = data as BoxData;
  const shown = depth === "surface" ? [] : box.args.slice(0, 4);
  return (
    <div
      className={[styles.node, selected && styles.selected, box.kind === "opaque" && styles.opaque, box.kind === "fixed" && styles.fixed].filter(Boolean).join(" ")}
      aria-label={`${box.name}${box.slot ? ` in ${box.slot}` : ""}`}
    >
      <Handle type="target" position={Position.Top} isConnectable={false} />
      <div className={styles.head}>
        <span className={`${styles.name} body-strong`}>{box.name}</span>
      </div>
      <div className={`${styles.meta} caption`}>{[box.slot, box.family].filter(Boolean).join(" · ")}</div>
      {shown.length > 0 && (
        <ul className={`${styles.args} caption`}>
          {shown.map(([k, v]) => (
            <li key={k}>{`${k}=${v}`}</li>
          ))}
        </ul>
      )}
      <Handle type="source" position={Position.Bottom} isConnectable={false} />
    </div>
  );
}

export function GroupNodeView({ data }: NodeProps) {
  const { label } = data as GroupData;
  return <div className={`${styles.group} caption`} style={{ width: "100%", height: "100%" }}>{label}</div>;
}

export const nodeTypes = { block: BlockNodeView, group: GroupNodeView };
