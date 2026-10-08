import { useEffect, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import { align, type EvalPoints } from "../curveData";
import { readToken, subscribe } from "../styles/theme";
import styles from "./Curve.module.css";

type Props = {
  name: string;
  run: EvalPoints;
  // the shipped baseline: its label and each seed's eval points
  baseline?: { name: string; seeds: EvalPoints[] } | null;
  yLabel: string;
};

const HEIGHT = 280;

function options(width: number, hasBand: boolean, yLabel: string, name: string): uPlot.Options {
  const font = (size: number) => `${size}px ${readToken("font-mono").split(",")[0]?.trim() ?? "monospace"}`;
  const muted = readToken("ink-muted");
  const axis = {
    stroke: muted,
    font: font(11),
    ticks: { show: false },
  };
  return {
    width,
    height: HEIGHT,
    legend: { show: false },
    cursor: { show: false },
    padding: [8, 16, 0, 0],
    scales: { x: { time: false } },
    axes: [
      { ...axis, grid: { show: false }, label: "step", labelFont: font(11) },
      { ...axis, grid: { show: true, stroke: readToken("grid"), width: 1 }, label: yLabel, labelFont: font(11), size: 56 },
    ],
    series: [
      {},
      {
        label: name,
        stroke: readToken("series-1"),
        width: 1.5,
        spanGaps: true,
        points: { show: true, size: 3, fill: readToken("series-1"), stroke: readToken("surface-raised") },
      },
      { label: "low", stroke: "transparent", width: 0, points: { show: false }, spanGaps: true },
      { label: "high", stroke: "transparent", width: 0, points: { show: false }, spanGaps: true },
    ],
    bands: hasBand ? [{ series: [3, 2], fill: readToken("band") }] : [],
  };
}

// Validation bpb against step: the run's eval points on a line, and the shipped seeds' range as
// a band. Colours come from the tokens and are re-read when the theme changes.
export function Curve({ name, run, baseline, yLabel }: Props) {
  const host = useRef<HTMLDivElement>(null);
  const aligned = align(run, baseline?.seeds ?? []);
  const data = [aligned.x, aligned.run, aligned.low, aligned.high] as uPlot.AlignedData;
  const hasBand = aligned.low.some((v) => v !== null);
  const latest = useRef(data);
  const plotRef = useRef<uPlot | null>(null);
  useEffect(() => {
    latest.current = data;
  });

  useEffect(() => {
    const el = host.current;
    if (!el) return;
    const build = () => {
      plotRef.current?.destroy();
      plotRef.current = new uPlot(options(el.clientWidth || 640, hasBand, yLabel, name), latest.current, el);
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
  }, [hasBand, yLabel, name]);

  // New points arrive without rebuilding the plot.
  useEffect(() => {
    plotRef.current?.setData(data);
  });

  return (
    <figure className={styles.curve}>
      <div ref={host} className={styles.plot} />
      <div className={`caption ${styles.key}`}>
        <span>
          <span className={styles.swatch} />
          {name}
        </span>
        {hasBand && baseline && (
          <span>
            <span className={styles.band} />
            {baseline.name}
          </span>
        )}
      </div>
      <details className={`small ${styles.table}`}>
        <summary>Values as a table</summary>
        <div className={`value ${styles.values}`} style={{ ["--cols" as string]: aligned.x.length + 1 }}>
          <span className={styles.muted}>step</span>
          {aligned.x.map((s) => (
            <span key={`s${s}`}>{s}</span>
          ))}
          <span className={styles.muted}>bpb</span>
          {aligned.run.map((v, i) => (
            <span key={`v${aligned.x[i]}`}>{v === null ? "-" : v.toFixed(3)}</span>
          ))}
        </div>
      </details>
    </figure>
  );
}
