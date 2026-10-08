import { Handle, type NodeProps, Position } from "@xyflow/react";
import { useState } from "react";
import { Button } from "../components/Button";
import { LockSimple, SealCheck, Warning } from "../icons";
import { carriesBlock, DRAG_TYPE } from "./dnd";
import type { Box } from "./flow";
import type { Depth } from "./layout";
import type { Trace } from "./trace";
import { spanText } from "./trace";
import styles from "./nodes.module.css";

// What the catalog says about a box's block: whether it is locked for this learner, and how its
// equivalence to a reference stands.
export type NodeInfo = {
  reference?: string | null;
  user?: boolean;
  certification?: string; // certified | failed | stale | uncertified (your own blocks)
  locked?: boolean;
  lesson?: string | null;
};

export type BoxData = {
  box: Box;
  depth: Depth;
  selected: boolean;
  info?: NodeInfo;
  trace?: Trace | null;
  // a block from the palette was dropped here
  onDropBlock?: (box: Box, name: string) => void;
  // jump the editor to the box's line
  onEditInCode?: (line: number) => void;
  [key: string]: unknown;
};

export type GroupData = {
  label: string;
  // per layer, from the blockstats of an open run: a rank 1..5 of its gradient norm and the number
  chips?: { name: string; heat: 1 | 2 | 3 | 4 | 5; title: string }[];
  onAddLayer?: () => void;
  onRemoveLayer?: () => void;
  [key: string]: unknown;
};

function Equivalence({ info }: { info: NodeInfo | undefined }) {
  if (!info) return null;
  if (info.user) {
    const state = info.certification ?? "uncertified";
    const text = {
      certified: `certified against ${info.reference ?? "its reference"}`,
      failed: "failed its check",
      stale: "changed since it was certified",
      uncertified: info.reference ? "not certified yet" : "no reference to check against",
    }[state] ?? state;
    return <li>{text}</li>;
  }
  return <li>{info.reference ? `reference ${info.reference}, tested in the library's checks` : "no reference"}</li>;
}

// A box in the model graph: the block's class name in mono, then its slot and family in words;
// more as the depth goes up. Nodes are neutral: no colour per family, the family is a word.
export function BlockNodeView({ data }: NodeProps) {
  const { box, depth, selected, info, trace, onDropBlock, onEditInCode } = data as BoxData;
  const [over, setOver] = useState(false);
  const droppable = onDropBlock !== undefined && box.path !== null && box.kind === "block";
  const shown = depth === "surface" ? [] : box.args.slice(0, 4);
  const locked = info?.locked === true;
  const sealed = info?.user === true && info.certification === "certified";
  const troubled = info?.user === true && (info.certification === "failed" || info.certification === "stale");
  return (
    <div
      className={[styles.node, selected && styles.selected, over && styles.over, box.kind === "opaque" && styles.opaque, box.kind === "fixed" && styles.fixed, locked && styles.lockedNode].filter(Boolean).join(" ")}
      aria-label={`${box.name}${box.slot ? ` in ${box.slot}` : ""}`}
      onDragOver={droppable ? (e) => {
        if (!carriesBlock(e)) return;
        e.preventDefault();
        e.dataTransfer.dropEffect = "copy";
        setOver(true);
      } : undefined}
      onDragLeave={droppable ? () => setOver(false) : undefined}
      onDrop={droppable ? (e) => {
        setOver(false);
        const name = e.dataTransfer.getData(DRAG_TYPE);
        if (!name) return;
        e.preventDefault();
        onDropBlock?.(box, name);
      } : undefined}
    >
      <Handle type="target" position={Position.Top} isConnectable={false} />
      <div className={styles.head}>
        <span className={`${styles.name} body-strong`}>{box.name}</span>
        {locked && <LockSimple size={16} aria-label="locked" />}
        {sealed && <SealCheck size={16} aria-label="certified" />}
        {troubled && <Warning size={16} aria-label={info?.certification === "failed" ? "failed its check" : "changed since certified"} />}
      </div>
      <div className={`${styles.meta} caption`}>{[box.slot, box.family].filter(Boolean).join(" · ")}</div>
      {locked && info?.lesson && <div className="caption">Pass {info.lesson} to use it</div>}
      {shown.length > 0 && (
        <ul className={`${styles.args} caption`}>
          {shown.map(([k, v]) => (
            <li key={k}>{`${k}=${v}`}</li>
          ))}
        </ul>
      )}
      {depth !== "surface" && trace && (
        <div className={`${styles.trace} caption`}>{[trace.shape, trace.params].filter(Boolean).join(" · ")}</div>
      )}
      {depth === "research" && (
        <ul className={`${styles.research} caption`}>
          {box.span && <li>{spanText(box.span)}</li>}
          {box.kind !== "opaque" && <Equivalence info={info} />}
          {trace && <li>{trace.flops}</li>}
        </ul>
      )}
      {box.kind === "opaque" && (
        <div className={styles.opaqueNote}>
          <span className="caption">Not a registered block: shown, not editable</span>
          {box.line !== null && onEditInCode && (
            <Button size="sm" variant="quiet" onClick={(e) => {
              e.stopPropagation();
              onEditInCode(box.line!);
            }}>
              Edit in code, line {box.line}
            </Button>
          )}
        </div>
      )}
      <Handle type="source" position={Position.Bottom} isConnectable={false} />
    </div>
  );
}

const HEAT = ["var(--heat-1)", "var(--heat-2)", "var(--heat-3)", "var(--heat-4)", "var(--heat-5)"];

export function GroupNodeView({ data }: NodeProps) {
  const { label, chips, onAddLayer, onRemoveLayer } = data as GroupData;
  return (
    <div className={`${styles.group} caption`} style={{ width: "100%", height: "100%" }}>
      <div className={styles.groupHead}>
        <span>{label}</span>
        {chips && chips.length > 0 && (
          <span className={styles.chips} aria-label="Layer gradient norms">
            {chips.map((c) => (
              <span key={c.name} className={styles.chip} title={c.title} aria-label={c.title}>
                <span className={styles.swatch} style={{ background: HEAT[c.heat - 1] }} />
                {c.name}
              </span>
            ))}
          </span>
        )}
        <span className={styles.spacer} />
        {onRemoveLayer && (
          <button type="button" className={styles.step} aria-label="Remove a layer" onClick={onRemoveLayer}>
            −
          </button>
        )}
        {onAddLayer && (
          <button type="button" className={styles.step} aria-label="Add a layer" onClick={onAddLayer}>
            +
          </button>
        )}
      </div>
    </div>
  );
}

export const nodeTypes = { block: BlockNodeView, group: GroupNodeView };
