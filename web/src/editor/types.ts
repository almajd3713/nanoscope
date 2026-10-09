import type { Suggestion } from "./completions";
import type { LspClient } from "./lsp";

// A problem shown on a source line: ruff's, or nanoscope's own (P14.09, P14.11).
export type Marker = {
  line: number; // 1-based, as the server sends it
  column: number;
  endLine: number;
  endColumn: number;
  message: string;
  severity: "error" | "warning" | "info";
  // who says so ("ruff", "nanoscope"); shown beside the code
  source: string;
  code?: string | null;
};

export type SurfaceProps = {
  path: string;
  value: string;
  onChange: (value: string) => void;
  onSave: () => void;
  markers: Marker[];
  readOnly?: boolean;
  // move the cursor to a source line (the graph's "edit in code")
  revealLine?: number | null;
  // completions at the cursor, from the text before it
  complete?: (textBefore: string) => Suggestion[];
  // the language server for this file, when one is connected: hover, signatures, completions
  // and (through `markers`) its diagnostics
  lsp?: { client: LspClient; uri: string } | null;
};
