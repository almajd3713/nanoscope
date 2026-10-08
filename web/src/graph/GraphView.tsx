import { Background, type Edge, type Node, ReactFlow } from "@xyflow/react";
import "@xyflow/react/dist/base.css";
import { useEffect, useMemo, useState } from "react";
import { Button } from "../components/Button";
import { buildFlow, type Box, type GClass } from "./flow";
import { type Depth, layout, type Layout } from "./layout";
import { type NodeInfo, nodeTypes } from "./nodes";
import { traceOf, type TraceRow } from "./trace";
import styles from "./GraphView.module.css";

type Props = {
  // one class of POST /api/files/{path}/graph
  cls: GClass;
  depth?: Depth;
  selected?: string | null;
  onSelect?: (box: Box | null) => void;
  // a palette block was dropped on a box
  onDropBlock?: (box: Box, name: string) => void;
  // what the catalog says about a block by name (lock, reference, certification)
  infoOf?: (name: string) => NodeInfo | undefined;
  // the modules of the last describe job, for shapes, parameters and FLOPs
  trace?: TraceRow[] | null;
  // blockstats of an open run, the newest line's blocks
  layerStats?: { name: string; grad_norm: number }[] | null;
  onEditInCode?: (line: number) => void;
  onAddLayer?: () => void;
  onRemoveLayer?: () => void;
};

// Rank each layer's gradient norm from 1 (lowest of this run) to 5 (highest), by where it falls
// between the lowest and the highest layer.
export function layerChips(stats: { name: string; grad_norm: number }[]): { name: string; heat: 1 | 2 | 3 | 4 | 5; title: string }[] {
  const norms = stats.map((s) => s.grad_norm);
  const lo = Math.min(...norms);
  const hi = Math.max(...norms);
  return stats.map((s) => {
    const rank = hi === lo ? 1 : Math.min(5, 1 + Math.floor(((s.grad_norm - lo) / (hi - lo)) * 4.999));
    return { name: s.name.replace("blocks.", ""), heat: rank as 1 | 2 | 3 | 4 | 5, title: `${s.name}: grad_norm ${Number(s.grad_norm.toPrecision(3))} (${rank} of 5)` };
  });
}

// The model as a diagram. It is a pure function of the parsed class: a file edited elsewhere
// parses again and the view lays itself out again; no layout is kept anywhere.
export function GraphView({ cls, depth = "detailed", selected = null, onSelect, onDropBlock, infoOf, trace = null, layerStats = null, onEditInCode, onAddLayer, onRemoveLayer }: Props) {
  const flow = useMemo(() => buildFlow(cls), [cls]);
  const [placed, setPlaced] = useState<{ flow: typeof flow; depth: Depth; layout: Layout } | null>(null);

  const extra: Record<string, number> = {};
  for (const b of flow.boxes) if (infoOf?.(b.name)?.locked) extra[b.id] = 1;
  const extraKey = JSON.stringify(extra);

  useEffect(() => {
    let current = true;
    void layout(flow, depth, JSON.parse(extraKey) as Record<string, number>).then((l) => {
      if (current) setPlaced({ flow, depth, layout: l });
    });
    return () => {
      current = false;
    };
  }, [flow, depth, extraKey]);

  if (!cls.representable) {
    return (
      <div className={styles.view}>
        <p className={`${styles.note} body`}>
          {cls.name} is code only: {cls.reason}. Edit it in the code.
          {onEditInCode && (cls.reason_line ?? cls.line) ? (
            <>
              {" "}
              <Button size="sm" onClick={() => onEditInCode((cls.reason_line ?? cls.line) as number)}>
                Edit in code, line {cls.reason_line ?? cls.line}
              </Button>
            </>
          ) : null}
        </p>
      </div>
    );
  }

  const ready = placed && placed.flow === flow && placed.depth === depth ? placed.layout : null;
  const nodes: Node[] = [];
  const edges: Edge[] = [];
  if (ready) {
    for (const g of flow.groups) {
      const p = ready.groups[g.id];
      if (p) nodes.push({ id: g.id, type: "group", position: { x: p.x, y: p.y }, data: { label: g.label, chips: layerStats ? layerChips(layerStats) : undefined, onAddLayer, onRemoveLayer }, style: { width: p.width, height: p.height }, draggable: false, selectable: false });
    }
    for (const b of flow.boxes) {
      const p = ready.boxes[b.id];
      if (!p) continue;
      nodes.push({
        id: b.id,
        type: "block",
        position: { x: p.x, y: p.y },
        parentId: b.parent ?? undefined,
        data: { box: b, depth, selected: selected === b.id, onDropBlock, info: infoOf?.(b.name), trace: traceOf(trace, b.module), onEditInCode },
        draggable: false,
      });
    }
    for (const a of flow.arrows) edges.push({ id: a.id, source: a.from, target: a.to, type: "smoothstep", style: a.kind === "arg" ? { strokeDasharray: "4 3" } : undefined });
  }

  return (
    <div className={styles.view} aria-label={`Graph of ${cls.name}`}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        fitView
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable
        onNodeClick={(_, node) => onSelect?.(flow.boxes.find((b) => b.id === node.id) ?? null)}
        onPaneClick={() => onSelect?.(null)}
        proOptions={{ hideAttribution: true }}
      >
        <Background />
      </ReactFlow>
    </div>
  );
}
