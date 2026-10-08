import type { CatalogBlock } from "../editor/completions";
import type { Edit } from "./api";
import type { Box } from "./flow";
import { DRAG_TYPE } from "./Palette";

export { DRAG_TYPE };

export type DropPlan = { ok: true; edit: Edit } | { ok: false; message: string };

// What dropping `name` on `box` does: one replace_block at the box's path, the same patch the
// inspector's swap menu sends. A block of another family does not fit the slot, and a locked
// block is refused before anything is sent (the server refuses it too).
export function planDrop(className: string, box: Box, name: string, blocks: CatalogBlock[]): DropPlan {
  const dropped = blocks.find((b) => b.name === name);
  if (!dropped) return { ok: false, message: `${name} is not in the palette` };
  if (box.path === null) return { ok: false, message: `${box.name} is added by nanoscope; it is not a slot in your file` };
  if (box.kind === "opaque") return { ok: false, message: `${box.name} is a call the graph cannot edit; change it in the code` };
  if (dropped.lock?.locked) {
    return { ok: false, message: `${name} is locked until you build it yourself in the lesson ${dropped.lock.lesson ?? "that unlocks it"}` };
  }
  if (box.family !== null && dropped.family !== box.family) {
    return { ok: false, message: `${name} is a ${dropped.family} block; the ${box.slot ?? "slot"} takes a ${box.family} block` };
  }
  if (name === box.name) return { ok: false, message: `${box.name} is already there` };
  return { ok: true, edit: { op: "replace_block", class: className, path: box.path, node: { kind: "block", block: name, args: {}, span: null } } };
}

// True for a drag that carries a block from the palette.
export function carriesBlock(e: { dataTransfer: DataTransfer | null }): boolean {
  return Array.from(e.dataTransfer?.types ?? []).includes(DRAG_TYPE);
}
