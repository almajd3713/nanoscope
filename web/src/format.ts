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

// 13200000 -> "13.2M", 726000 -> "726k", 42 -> "42": an attached SI suffix, up to three
// significant digits (design system, Type).
export function countText(n: number): string {
  const units: [number, string][] = [[1e9, "G"], [1e6, "M"], [1e3, "k"]];
  for (const [size, suffix] of units) {
    if (Math.abs(n) >= size) return `${Number((n / size).toPrecision(3))}${suffix}`;
  }
  return String(n);
}

// Remaining seconds as m:ss or h:mm:ss ("0:41").
export function clockText(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = String(s % 60).padStart(2, "0");
  return h > 0 ? `${h}:${String(m).padStart(2, "0")}:${ss}` : `${m}:${ss}`;
}

// "2026-10-07 14:03" in the browser's time zone (24-hour, ISO date).
export function stampText(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
}
