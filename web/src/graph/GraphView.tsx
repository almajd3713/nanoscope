import { Background, type Edge, type Node, ReactFlow } from "@xyflow/react";
import "@xyflow/react/dist/base.css";
import { useEffect, useMemo, useState } from "react";
import { buildFlow, type Box, type GClass } from "./flow";
import { type Depth, layout, type Layout } from "./layout";
import { nodeTypes } from "./nodes";
import styles from "./GraphView.module.css";

type Props = {
  // one class of POST /api/files/{path}/graph
  cls: GClass;
  depth?: Depth;
  selected?: string | null;
  onSelect?: (box: Box | null) => void;
};

// The model as a diagram. It is a pure function of the parsed class: a file edited elsewhere
// parses again and the view lays itself out again; no layout is kept anywhere.
export function GraphView({ cls, depth = "detailed", selected = null, onSelect }: Props) {
  const flow = useMemo(() => buildFlow(cls), [cls]);
  const [placed, setPlaced] = useState<{ flow: typeof flow; depth: Depth; layout: Layout } | null>(null);

  useEffect(() => {
    let current = true;
    void layout(flow, depth).then((l) => {
      if (current) setPlaced({ flow, depth, layout: l });
    });
    return () => {
      current = false;
    };
  }, [flow, depth]);

  if (!cls.representable) {
    return (
      <div className={styles.view}>
        <p className={`${styles.note} body`}>
          {cls.name} is code only: {cls.reason}. Edit it in the code.
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
      if (p) nodes.push({ id: g.id, type: "group", position: { x: p.x, y: p.y }, data: { label: g.label }, style: { width: p.width, height: p.height }, draggable: false, selectable: false });
    }
    for (const b of flow.boxes) {
      const p = ready.boxes[b.id];
      if (!p) continue;
      nodes.push({
        id: b.id,
        type: "block",
        position: { x: p.x, y: p.y },
        parentId: b.parent ?? undefined,
        data: { box: b, depth, selected: selected === b.id },
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
