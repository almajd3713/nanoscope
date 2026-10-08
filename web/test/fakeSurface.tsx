// Monaco needs a real browser. Component tests swap the surface for a text area with the same
// props, so they exercise the file logic (ETag, dirty, save) and the e2e tests cover Monaco.
import type { SurfaceProps } from "../src/editor/types";

export function CodeSurface({ path, value, onChange, onSave, markers, revealLine }: SurfaceProps) {
  return (
    <div>
      <textarea
        aria-label={`Source of ${path}`}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if ((e.ctrlKey || e.metaKey) && e.key === "s") {
            e.preventDefault();
            onSave();
          }
        }}
      />
      <ul aria-label="Markers">
        {markers.map((m, i) => (
          <li key={i}>{`${m.source} ${m.line}:${m.column} ${m.message}`}</li>
        ))}
      </ul>
      {revealLine ? <span data-testid="revealed">{revealLine}</span> : null}
    </div>
  );
}
