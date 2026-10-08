// Formats the library prints, reproduced so the page reads like `nanoscope learn list`.
// (nanoscope/learn/cli.py: _minutes and compute_text.)

export function minutesText(minutes: number): string {
  return minutes < 90 ? `${minutes} min` : `${Number((minutes / 60).toPrecision(3))} h`;
}

// `cpu 2 min / gpu 4 h`, or `-` when nothing is declared.
export function computeText(compute: Record<string, { estimate_minutes: number }>): string {
  const parts = Object.entries(compute).map(([name, c]) => `${name} ${minutesText(c.estimate_minutes)}`);
  return parts.join(" / ") || "-";
}
