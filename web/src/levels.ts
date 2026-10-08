// The single table of which controls each level shows (design system, Layout > Levels).
// A level only adds visibility: it never changes colors, density or layout, and a hidden
// field keeps its level-0 default and is never sent changed.
import { atLeast, type Level } from "./app/level";

export const CONTROLS = {
  // Learn
  lessons: "Learn",
  "graph.surface": "Learn",
  "graph.detailed": "Tinker",
  "graph.research": "Research",
  palette: "Learn",
  train: "Learn",
  "run.live": "Learn",
  "run.baseline": "Learn",
  "run.samples": "Learn",
  // Tinker
  editor: "Tinker",
  runForm: "Tinker",
  seeds: "Tinker",
  duplicate: "Tinker",
  compare: "Tinker",
  predictions: "Tinker",
  // Research
  studyBuilder: "Research",
  budgets: "Research",
  paramMatching: "Research",
  recordMode: "Research",
  "queue.devices": "Research",
  forestPlot: "Research",
  ablationCards: "Research",
  bench: "Research",
  // Extend
  workspaceTree: "Extend",
  blockRegistration: "Extend",
  certification: "Extend",
  authoringPreview: "Extend",
  schemasDocs: "Extend",
} as const satisfies Record<string, Level>;

export type ControlId = keyof typeof CONTROLS;

export function shows(control: ControlId, level: Level): boolean {
  return atLeast(level, CONTROLS[control]);
}

// Reset every field whose control the level hides to its default, so a request built by a
// lower level is the same request the level-0 defaults would make.
export function withoutHidden<T extends Record<string, unknown>>(
  values: T,
  defaults: T,
  controls: { [K in keyof T]?: ControlId },
  level: Level,
): T {
  const out = { ...values };
  for (const key of Object.keys(controls) as (keyof T)[]) {
    const control = controls[key];
    if (control && !shows(control, level)) out[key] = defaults[key];
  }
  return out;
}
