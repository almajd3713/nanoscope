import { useEffect, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import { readToken, subscribe } from "../styles/theme";
import styles from "./Curve.module.css";

export type SeedLine = { seed: number; points: [number, number][] };
export type SetCurves = { label: string; baseline: boolean; seeds: SeedLine[] };

const HEIGHT = 300;

// One line per seed, in the set's series colour: seed 0 at full weight, the others thin and
// quiet. The baseline's seeds are dashed in the neutral baseline colour, with the range of its
// seeds as a band. Everything plotted is a point the API gave; nothing is averaged.
export function CompareCurves({ sets, xLabel, yLabel }: { sets: SetCurves[]; xLabel: string; yLabel: string }) {
  const host = useRef<HTMLDivElement>(null);
  const plotRef = useRef<uPlot | null>(null);

  const x = [...new Set(sets.flatMap((s) => s.seeds.flatMap((sd) => sd.points.map((p) => p[0]))))].sort((a, b) => a - b);
  const lines: { name: string; color: "series" | "baseline"; index: number; quiet: boolean; dashed: boolean; values: (number | null)[] }[] = [];
  let colour = 0;
  for (const set of sets) {
    const index = set.baseline ? -1 : colour++;
    set.seeds.forEach((sd, i) => {
      const at = new Map(sd.points);
      lines.push({
        name: i === 0 ? set.label : `${set.label} s${sd.seed}`,
        color: set.baseline ? "baseline" : "series",
        index,
        quiet: i > 0,
        dashed: set.baseline,
        values: x.map((v) => at.get(v) ?? null),
      });
    });
  }
  const base = sets.find((s) => s.baseline);
  // the range of the baseline's seeds at each step, where every one of them has a point
  const baseSeeds = lines.filter((l) => l.dashed).map((l) => l.values);
  const range = (pick: (v: number[]) => number) =>
    x.map((_, i) => {
      const vals = baseSeeds.map((v) => v[i]);
      return vals.length > 0 && vals.every((v) => typeof v === "number") ? pick(vals as number[]) : null;
    });
  const low = range((v) => Math.min(...v));
  const high = range((v) => Math.max(...v));
  const data = [x, ...lines.map((l) => l.values), low, high] as uPlot.AlignedData;
  const latest = useRef(data);
  useEffect(() => {
    latest.current = data;
  });

  const key = lines.map((l) => l.name).join("|");
  useEffect(() => {
    const el = host.current;
    if (!el) return;
    const build = () => {
      plotRef.current?.destroy();
      const font = (size: number) => `${size}px ${readToken("font-mono").split(",")[0]?.trim() ?? "monospace"}`;
      const muted = readToken("ink-muted");
      const axis = { stroke: muted, font: font(11), ticks: { show: false } };
      const colorOf = (l: (typeof lines)[number]) => (l.color === "baseline" ? readToken("baseline") : readToken(`series-${(l.index % 6) + 1}`));
      const opts: uPlot.Options = {
        width: el.clientWidth || 640,
        height: HEIGHT,
        legend: { show: false },
        cursor: { show: false },
        padding: [8, 16, 0, 0],
        scales: { x: { time: false } },
        axes: [
          { ...axis, grid: { show: false }, label: xLabel, labelFont: font(11) },
          { ...axis, grid: { show: true, stroke: readToken("grid"), width: 1 }, label: yLabel, labelFont: font(11), size: 56 },
        ],
        series: [
          {},
          ...lines.map((l) => ({
            label: l.name,
            stroke: colorOf(l),
            width: l.quiet ? 1 : 1.5,
            dash: l.dashed ? [5, 4] : undefined,
            spanGaps: true,
            alpha: l.quiet ? 0.6 : 1,
            points: { show: false },
          })),
          { label: "low", stroke: "transparent", width: 0, points: { show: false }, spanGaps: true },
          { label: "high", stroke: "transparent", width: 0, points: { show: false }, spanGaps: true },
        ],
        bands: base ? [{ series: [lines.length + 2, lines.length + 1], fill: readToken("band") }] : [],
      };
      plotRef.current = new uPlot(opts, latest.current, el);
    };
    build();
    const off = subscribe(build);
    const resize = new ResizeObserver(() => plotRef.current?.setSize({ width: el.clientWidth || 640, height: HEIGHT }));
    resize.observe(el);
    return () => {
      off();
      resize.disconnect();
      plotRef.current?.destroy();
      plotRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, xLabel, yLabel]);

  useEffect(() => {
    plotRef.current?.setData(data);
  });

  return (
    <figure className={styles.curve}>
      <div ref={host} className={styles.plot} />
    </figure>
  );
}
