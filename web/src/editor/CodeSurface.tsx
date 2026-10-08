import { lazy, Suspense } from "react";
import type { SurfaceProps } from "./types";

const Monaco = lazy(() => import("./MonacoSurface"));

// The text area the page edits. Monaco loads on first use; until then a plain text area shows
// the same text, so a slow chunk never hides the file.
export function CodeSurface(props: SurfaceProps) {
  return (
    <Suspense
      fallback={
        <textarea
          aria-label={`Source of ${props.path}`}
          className="mono"
          style={{ width: "100%", height: "100%", minHeight: 240 }}
          value={props.value}
          readOnly
        />
      }
    >
      <Monaco {...props} />
    </Suspense>
  );
}
