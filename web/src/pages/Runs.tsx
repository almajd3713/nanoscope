import { useQuery } from "@tanstack/react-query";
import { useCallback, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api } from "../api/client";
import { unwrap } from "../api/problem";
import { useLevel } from "../app/level";
import { ButtonLink } from "../components/Button";
import { EquivalentCommand } from "../components/EquivalentCommand";
import { ProblemFromError } from "../components/ProblemView";
import { StateTag } from "../components/StateTag";
import { useEvents } from "../hooks/useEvents";
import { shows } from "../levels";
import { MagnifyingGlass, Plus } from "../icons";
import { stampText } from "../format";
import styles from "./Runs.module.css";

const STATES = ["all", "running", "queued", "done", "stopped", "failed"] as const;
type Patch = { state?: string; step?: number; max_steps?: number; val_bpb?: number };

export function Runs() {
  const [params, setParams] = useSearchParams();
  const prefix = params.get("prefix") ?? "";
  const stateFilter = params.get("state") ?? "all";
  const baselines = params.get("baselines") === "1";
  const level = useLevel();
  const [selected, setSelected] = useState<string[]>([]);
  const [live, setLive] = useState<Record<string, Patch>>({});

  const runs = useQuery({
    queryKey: ["runs", prefix],
    queryFn: () => unwrap(api.GET("/api/runs", { params: { query: { prefix } } })),
  });
  const shipped = useQuery({
    queryKey: ["runs", "baselines"],
    queryFn: () => unwrap(api.GET("/api/runs", { params: { query: { prefix: "baselines" } } })),
    enabled: baselines,
  });

  const onEvent = useCallback((type: string, data: unknown) => {
    const d = data as Record<string, unknown> & { ref?: string };
    if (!d.ref) return;
    const ref = d.ref;
    setLive((prev) => {
      const patch: Patch = { ...prev[ref] };
      if (type === "state") {
        patch.state = d["state"] as string;
        patch.step = d["step"] as number;
        patch.max_steps = d["max_steps"] as number;
      } else if (type === "step") patch.step = d["step"] as number;
      else if (type === "eval" && typeof d["val_bpb"] === "number") patch.val_bpb = d["val_bpb"];
      return { ...prev, [ref]: patch };
    });
  }, []);
  const stream = useEvents(`/api/events?prefix=${encodeURIComponent(prefix)}`, {
    events: ["state", "step", "eval"],
    onEvent,
    onReset: () => setLive({}),
  });

  const set = (key: string, value: string | null) =>
    setParams((p) => {
      const next = new URLSearchParams(p);
      if (value === null || value === "" || value === "all") next.delete(key);
      else next.set(key, value);
      return next;
    });

  const error = runs.error ?? shipped.error;
  if (error) return <ProblemFromError error={error} />;

  const all = [...(runs.data ?? []), ...(baselines ? (shipped.data ?? []) : [])];
  // the library lists oldest first; the newest run is the one you are looking for
  const rows = all
    .map((r) => ({ ...r, ...live[r.ref] }))
    .filter((r) => stateFilter === "all" || r.state === stateFilter)
    .reverse();

  const toggle = (ref: string) => setSelected((s) => (s.includes(ref) ? s.filter((x) => x !== ref) : [...s, ref]));

  return (
    <div className={styles.page}>
      <header className={styles.head}>
        <h1 className="title">Runs</h1>
        <span className={`small ${styles.muted}`}>
          {rows.length} {rows.length === 1 ? "run" : "runs"} in <span className="value">runs/</span> ·{" "}
          {stream === "paused" ? <span className={styles.warn}>Live updates paused. Reconnecting…</span> : "live"}
        </span>
        <span className={styles.spacer} />
        {shows("compare", level) && (
          <ButtonLink
            to={`/compare?runs=${selected.join(",")}`}
            aria-disabled={selected.length < 2}
            onClick={(e) => selected.length < 2 && e.preventDefault()}
            title={selected.length < 2 ? "Select at least two runs to compare" : undefined}
          >
            Compare selected
          </ButtonLink>
        )}
        {shows("runForm", level) && (
          <ButtonLink to="/runs/new" variant="primary">
            <Plus size={16} aria-hidden="true" />
            New run
          </ButtonLink>
        )}
      </header>

      <div className={styles.filters}>
        <div className={styles.search}>
          <MagnifyingGlass size={16} aria-hidden="true" />
          <input
            aria-label="Filter by ref prefix"
            className="value"
            value={prefix}
            onChange={(e) => set("prefix", e.target.value)}
          />
        </div>
        <div className={`${styles.states} small`} role="group" aria-label="State">
          {STATES.map((s) => (
            <button key={s} type="button" aria-pressed={stateFilter === s} onClick={() => set("state", s)}>
              {s}
            </button>
          ))}
        </div>
        <label className={`small ${styles.check}`}>
          <input type="checkbox" checked={baselines} onChange={(e) => set("baselines", e.target.checked ? "1" : null)} />
          Show shipped baselines
        </label>
      </div>

      {!runs.data ? (
        <p className={`body ${styles.muted}`}>Loading runs…</p>
      ) : rows.length === 0 ? (
        <p className={`body ${styles.muted}`}>
          No runs match. {shows("runForm", level) ? (
            <>
              <Link to="/runs/new" className={styles.link}>Start one</Link> or run <span className="value">nanoscope run</span>.
            </>
          ) : (
            <>Train one from a lesson.</>
          )}
        </p>
      ) : (
        <div className={styles.wrap}>
          <table className={`${styles.table} small`}>
            <thead>
              <tr>
                <th className="label" scope="col">
                  <span aria-hidden="true"><span className="visually-hidden-label" aria-hidden="true" />nbsp;</span>
                </th>
                <th className="label" scope="col">State</th>
                <th className="label" scope="col">Run</th>
                <th className="label" scope="col">Model</th>
                <th className={`label ${styles.right}`} scope="col">Step</th>
                <th className={`label ${styles.right}`} scope="col">val_bpb</th>
                <th className="label" scope="col">Started</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.ref} data-selected={selected.includes(r.ref)}>
                  <td>
                    <input
                      type="checkbox"
                      aria-label={`Select ${r.ref}`}
                      checked={selected.includes(r.ref)}
                      onChange={() => toggle(r.ref)}
                    />
                  </td>
                  <td>
                    <StateTag state={r.state} />
                  </td>
                  <td>
                    <Link to={`/runs/${r.ref}`} className={`value ${styles.link}`}>
                      {r.ref}
                    </Link>
                  </td>
                  <td className={`value ${styles.muted}`}>{r.model ?? "–"}</td>
                  <td className={`value ${styles.right} ${r.step ? "" : styles.muted}`}>
                    {r.step || r.state === "running" || r.state === "failed" ? `${r.step} / ${r.max_steps}` : "–"}
                  </td>
                  <td className={`value ${styles.right} ${r.val_bpb == null ? styles.muted : ""}`}>
                    {r.val_bpb == null ? "–" : r.val_bpb.toFixed(3)}
                  </td>
                  <td className={`value ${styles.muted}`}>{r.started_at ? stampText(r.started_at) : "–"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <span className={`small ${styles.muted}`}>
        val_bpb is the latest evaluation: lower is better. A stopped run resumes from its last checkpoint; a cancelled one keeps its
        files.
      </span>
      <EquivalentCommand cli={`nanoscope status runs/${prefix}`.trimEnd()} />
    </div>
  );
}
