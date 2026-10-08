import styles from "./ForestPlot.module.css";

export type ForestRow = {
  label: string;
  // rows[].delta from the API; null for the baseline
  delta: { mean: number; ci95_low: number | null; ci95_high: number | null } | null;
};

// A round step (1, 2, 2.5, 5 times a power of ten) for the axis. Layout only.
export function niceStep(span: number, target = 4): number {
  const raw = span / target;
  const power = 10 ** Math.floor(Math.log10(raw));
  const base = [1, 2, 2.5, 5, 10].find((m) => m * power >= raw) ?? 10;
  return base * power;
}

export function axis(values: number[]): { min: number; max: number; ticks: number[] } {
  const lo = Math.min(0, ...values);
  const hi = Math.max(0, ...values);
  const step = niceStep(hi - lo || 1);
  const min = Math.floor(lo / step) * step;
  const max = Math.ceil(hi / step) * step;
  const ticks: number[] = [];
  for (let t = min; t <= max + step / 2; t += step) ticks.push(Number(t.toFixed(10)));
  return { min, max, ticks };
}

const W = 420;
const LEFT = 84;
const RIGHT = 16;
const ROW = 28;
const TOP = 8;

// Each model's difference from the baseline with its 95% interval, drawn from the API's numbers
// only (rows[].delta). Lower is better, so left of the zero line is better.
export function ForestPlot({ caption, rows, unit }: { caption: string; rows: ForestRow[]; unit: string }) {
  const shown = rows.filter((r) => r.delta);
  const values = shown.flatMap((r) => [r.delta!.mean, r.delta!.ci95_low ?? r.delta!.mean, r.delta!.ci95_high ?? r.delta!.mean]);
  const { min, max, ticks } = axis(values);
  const x = (v: number) => LEFT + ((v - min) / (max - min)) * (W - LEFT - RIGHT);
  const height = TOP + rows.length * ROW + 44;
  const axisY = TOP + rows.length * ROW + 8;
  const text = shown
    .map((r) => `${r.label}: ${r.delta!.mean.toFixed(3)}${r.delta!.ci95_low !== null ? ` [${r.delta!.ci95_low.toFixed(3)}, ${r.delta!.ci95_high!.toFixed(3)}]` : ""}`)
    .join("; ");
  return (
    <figure className={styles.plot}>
      <svg viewBox={`0 0 ${W} ${height}`} role="img" aria-label={`${caption}. ${text}`}>
        {ticks.map((t) => (
          <line key={t} x1={x(t)} x2={x(t)} y1={TOP} y2={axisY} className={styles.grid} />
        ))}
        <line x1={x(0)} x2={x(0)} y1={TOP} y2={axisY} className={styles.zero} />
        {rows.map((r, i) => {
          const y = TOP + i * ROW + ROW / 2;
          const d = r.delta;
          return (
            <g key={r.label}>
              <text x={LEFT - 8} y={y + 4} textAnchor="end" className={styles.label}>
                {r.label}
              </text>
              {d && d.ci95_low !== null && d.ci95_high !== null && (
                <line x1={x(d.ci95_low)} x2={x(d.ci95_high)} y1={y} y2={y} className={styles.ci} />
              )}
              {d && <rect x={x(d.mean) - 3} y={y - 3} width={6} height={6} className={styles.point} />}
            </g>
          );
        })}
        <line x1={LEFT} x2={W - RIGHT} y1={axisY} y2={axisY} className={styles.zero} />
        {ticks.map((t) => (
          <text key={`t${t}`} x={x(t)} y={axisY + 14} textAnchor="middle" className={styles.tick}>
            {t === 0 ? "0" : t.toFixed(2).replace(/\.?0+$/, "").replace("-", "−")}
          </text>
        ))}
        <text x={x(0)} y={TOP - 1} textAnchor="middle" className={styles.tick}>
          no difference
        </text>
        <text x={LEFT} y={axisY + 30} textAnchor="start" className={styles.tick}>
          ← better
        </text>
        <text x={W - RIGHT} y={axisY + 30} textAnchor="end" className={styles.tick}>
          worse →
        </text>
      </svg>
      <figcaption className={`caption ${styles.caption}`}>
        {caption} ({unit}; lower is better)
      </figcaption>
    </figure>
  );
}
