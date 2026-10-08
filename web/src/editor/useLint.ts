import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import { unwrap } from "../api/problem";
import type { Marker } from "./types";

type Diagnostic = {
  code: string | null;
  message: string;
  line: number;
  column: number;
  end_line: number;
  end_column: number;
  fixable?: boolean;
};

// ruff reports a syntax error with no code; names that cannot run (undefined names, bad
// syntax, E9xx) are errors, style findings are warnings.
const ERROR_CODES = /^(E9|F63|F7|F82)/;

export function toMarker(d: Diagnostic): Marker {
  const error = d.code === null || ERROR_CODES.test(d.code);
  return {
    line: d.line,
    column: d.column,
    endLine: d.end_line,
    endColumn: d.end_column,
    message: d.message,
    severity: error ? "error" : "warning",
    source: "ruff",
    code: d.code,
  };
}

// ruff's diagnostics for the saved file. Keyed by the file's ETag, so each save lints once and
// nothing is linted while the buffer is only edited.
export function useLint(path: string, etag: string | null): Marker[] {
  const lint = useQuery({
    queryKey: ["lint", path, etag],
    enabled: etag !== null,
    queryFn: async () => {
      const found = (await unwrap(api.POST("/api/files/{path}/lint", { params: { path: { path } } }))) as unknown as Diagnostic[];
      return found.map(toMarker);
    },
  });
  return lint.data ?? [];
}
