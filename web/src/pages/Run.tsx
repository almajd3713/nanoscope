import { useMutation, useQueries, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api/client";
import { ApiProblem, unwrap } from "../api/problem";
import { useLevel } from "../app/level";
import { shows } from "../levels";
import { Button, ButtonLink } from "../components/Button";
import { Curve } from "../components/Curve";
import { evalPoints, type Row } from "../curveData";
import { EquivalentCommand } from "../components/EquivalentCommand";
import { ProblemFromError } from "../components/ProblemView";
import { Progress } from "../components/Progress";
import { GeneratePanel } from "../components/GeneratePanel";
import { Samples } from "../components/Samples";
import { Readout } from "../components/Readout";
import { StateTag } from "../components/StateTag";
import { clockText, countText, stampText } from "../format";
import { ArrowClockwise, Copy, CopySimple, Stop, X } from "../icons";
import { useEvents } from "../hooks/useEvents";
import styles from "./Run.module.css";

const TERMINAL = ["done", "failed", "stopped", "cancelled"];
const LIVE_STATES = ["queued", "preparing", "running"];
const STALE_SECONDS = 120;

type RunError = { type: string; message: string; traceback?: string[] };
type Live = {
  state?: string;
  step?: number;
  maxSteps?: number;
  error?: RunError | null;
  device?: string | null;
  tokensPerSec?: number;
  secondsPerStep?: number;
  valBpb?: number;
  valStep?: number;
  at?: number; // when the last event arrived
};
type Baseline = {
  ref: string;
  n_seeds: number;
  metric: string;
  interval: [number, number] | null;
  value: number | null;
  inside: boolean | null;
};
type Summary = { final_step: number; final_val_bpb: number | null };

const bpb = (x: number) => x.toFixed(3);

function useNow(everyMs: number): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), everyMs);
    return () => clearInterval(id);
  }, [everyMs]);
  return now;
}

export function Run() {
  const ref = useParams()["*"] ?? "";
  const queryClient = useQueryClient();
  const detail = useQuery({
    queryKey: ["run", ref],
    queryFn: () => unwrap(api.GET("/api/runs/{ref}", { params: { path: { ref } } })),
    // A queued run has no folder yet: ask again until its worker makes one.
    refetchInterval: (q) => (q.state.error instanceof ApiProblem && q.state.error.status === 404 ? 1000 : false),
  });
  const missing = detail.error instanceof ApiProblem && detail.error.status === 404;
  const queued = useQuery({
    queryKey: ["jobs", "run", ref],
    queryFn: () => unwrap(api.GET("/api/jobs", { params: { query: { kind: "run", limit: 50 } } })),
    enabled: missing,
    refetchInterval: 1000,
    select: (jobs) => jobs.filter((j) => j.ref === ref && ["queued", "running"].includes(j.state)).at(-1),
  });
  const metrics = useQuery({
    queryKey: ["run", ref, "metrics"],
    queryFn: () => unwrap(api.GET("/api/runs/{ref}/metrics", { params: { path: { ref } } })),
  });
  const samples = useQuery({
    queryKey: ["run", ref, "samples"],
    queryFn: () => unwrap(api.GET("/api/runs/{ref}/samples", { params: { path: { ref } } })),
  });
  const baselineInfo = detail.data?.baseline as Baseline | null | undefined;
  const seedRefs = baselineInfo ? Array.from({ length: baselineInfo.n_seeds }, (_, i) => `${baselineInfo.ref}/seed-${i}`) : [];
  const seeds = useQueries({
    queries: seedRefs.map((seedRef) => ({
      queryKey: ["run", seedRef, "metrics"],
      queryFn: () => unwrap(api.GET("/api/runs/{ref}/metrics", { params: { path: { ref: seedRef } } })),
      staleTime: Infinity, // shipped files do not change
    })),
  });
  const [live, setLive] = useState<Live>({});
  const now = useNow(10000);
  const level = useLevel();
  const stop = useMutation({
    mutationFn: () => unwrap(api.POST("/api/runs/{ref}/stop", { params: { path: { ref } } })),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["run", ref] }),
  });
  const resume = useMutation({
    mutationFn: () => unwrap(api.POST("/api/runs/{ref}/resume", { params: { path: { ref } } })),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["run", ref] }),
  });

  const state = live.state ?? detail.data?.status?.state;
  // A run with no status.json (the shipped v0 baselines) has nothing to follow.
  const watching = detail.data !== undefined && LIVE_STATES.includes(state ?? "");

  const refetch = useCallback(() => queryClient.invalidateQueries({ queryKey: ["run", ref] }), [queryClient, ref]);
  const onEvent = useCallback(
    (type: string, data: unknown) => {
      const d = data as Record<string, unknown>;
      setLive((prev) => {
        const next: Live = { ...prev, at: Date.now() };
        if (type === "state") {
          next.state = d["state"] as string;
          next.step = d["step"] as number;
          next.maxSteps = d["max_steps"] as number;
          next.error = (d["error"] as RunError | null) ?? null;
          next.device = (d["device"] as string | null) ?? prev.device;
        } else if (type === "step") {
          next.step = d["step"] as number;
          if (typeof d["tokens_per_sec"] === "number") next.tokensPerSec = d["tokens_per_sec"];
          if (typeof d["elapsed"] === "number") next.secondsPerStep = d["elapsed"];
        } else if (type === "eval" && typeof d["val_bpb"] === "number") {
          next.valBpb = d["val_bpb"];
          next.valStep = d["step"] as number;
        }
        return next;
      });
      if (type === "sample") void queryClient.invalidateQueries({ queryKey: ["run", ref, "samples"] });
      if (type === "eval" || (type === "state" && TERMINAL.includes(d["state"] as string))) void refetch();
    },
    [refetch, queryClient, ref],
  );
  const stream = useEvents(watching ? `/api/runs/${ref}/events` : null, {
    events: ["state", "step", "eval", "sample"],
    onEvent,
    onReset: () => {
      setLive({});
      void refetch();
    },
    resumeParam: "since_step",
  });

  if (missing && queued.data) {
    return (
      <div className={styles.main}>
        <header className={styles.head}>
          <span className={`small ${styles.muted}`}>
            <Link to="/runs" className={styles.link}>
              Runs
            </Link>
          </span>
          <div className={styles.titleRow}>
            <h1 className={`title ${styles.ref}`}>{ref}</h1>
            <StateTag state={queued.data.state === "running" ? "preparing" : "queued"} />
            <span className={styles.spacer} />
            <Button variant="danger" disabled={stop.isPending} onClick={() => stop.mutate()}>
              <Stop size={16} aria-hidden="true" />
              Stop run
            </Button>
          </div>
          <span className={`small ${styles.muted}`}>
            {queued.data.state === "running" ? "A worker has it and is getting the run ready." : "Waiting for a worker to start this run."}
          </span>
        </header>
        {stop.error && <ProblemFromError error={stop.error} />}
      </div>
    );
  }
  if (detail.error) return <ProblemFromError error={detail.error} />;
  if (!detail.data) return <p className={`body ${styles.muted}`}>Loading run…</p>;

  const run = detail.data;
  const status = run.status;
  const config = run.config;
  const summary = run.summary as Summary;
  const baseline = run.baseline as Baseline | null;
  const stats = (config?.["stats"] ?? {}) as Record<string, number | string | undefined>;
  const preset = (config?.preset ?? {}) as Record<string, number | string>;

  const step = live.step ?? status?.step ?? 0;
  const maxSteps = live.maxSteps ?? status?.max_steps ?? (preset["max_steps"] as number | undefined) ?? 0;
  const error: RunError | null = (live.error !== undefined ? live.error : (status?.error as RunError | null)) ?? null;
  const device = live.device ?? (status?.device as string | undefined) ?? (stats["device"] as string | undefined);
  const valBpb = live.valBpb ?? summary.final_val_bpb;
  const valStep = live.valStep ?? summary.final_step;
  const running = state === "running";
  const eta = running && live.secondsPerStep && maxSteps > step ? clockText((maxSteps - step) * live.secondsPerStep) : undefined;
  const silentFor = live.at !== undefined ? (now - live.at) / 1000 : 0;

  const folder = ref.split("/").slice(0, -1).join("/");
  const range = baseline?.interval
    ? `${bpb(baseline.interval[0])} – ${bpb(baseline.interval[1])}`
    : undefined;
  const readoutSub =
    baseline && range && valBpb !== null
      ? baseline.inside === null
        ? `shipped range for one new run: ${range}`
        : `${baseline.inside ? "inside" : "outside"} the shipped range for one new run: ${range}`
      : undefined;

  return (
    <div className={styles.page}>
      <div className={styles.main}>
        <header className={styles.head}>
          <span className={`small ${styles.muted}`}>
            <Link to="/runs" className={styles.link}>
              Runs
            </Link>{" "}
            / <span className="value">{folder}</span>
          </span>
          <div className={styles.titleRow}>
            <h1 className={`title ${styles.ref}`}>{ref}</h1>
            {state && <StateTag state={state} />}
            {running && <span className="value">step {step}</span>}
            <span className={styles.spacer} />
            {["queued", "preparing", "running"].includes(state ?? "") && (
              <Button variant="danger" disabled={stop.isPending} onClick={() => stop.mutate()}>
                <Stop size={16} aria-hidden="true" />
                Stop run
              </Button>
            )}
            {["stopped", "cancelled"].includes(state ?? "") && (
              <Button disabled={resume.isPending} onClick={() => resume.mutate()}>
                <ArrowClockwise size={16} aria-hidden="true" />
                Resume
              </Button>
            )}
            {shows("duplicate", level) && (
              <ButtonLink to={`/runs/new?from=${encodeURIComponent(ref)}`}>
                <CopySimple size={16} aria-hidden="true" />
                Duplicate and change one thing
              </ButtonLink>
            )}
          </div>
          {stop.error && <ProblemFromError error={stop.error} />}
          {resume.error && <ProblemFromError error={resume.error} />}
          <span className={`small ${styles.muted}`}>
            {watching && stream === "paused" ? (
              <span className={styles.warn}>Live updates paused. Reconnecting…</span>
            ) : watching ? (
              "Live"
            ) : null}
            {watching && running && silentFor > STALE_SECONDS && ` · last update ${clockText(silentFor)} ago`}
            {device && (
              <>
                {" · "}
                <span className="value">{device}</span>
              </>
            )}
            {status?.updated_at && !watching && (
              <>
                {" · "}
                <span className="value">{stampText(status.updated_at)}</span>
              </>
            )}
          </span>
        </header>

        {state === "failed" && error && (
          <>
            <section className={styles.error} role="alert">
              <div className={styles.errorHead}>
                <X size={16} aria-hidden="true" />
                <span className="value-strong">{error.type}</span>
              </div>
              <p className="code">{error.message}</p>
              <span className={`small ${styles.muted}`}>
                From <span className="value">status.json</span>, word for word.
              </span>
            </section>
            {error.traceback && error.traceback.length > 0 && <Traceback lines={error.traceback} />}
          </>
        )}

        {(valBpb !== null || running || state === "done") && (
          <section className={styles.measure}>
            {valBpb !== null && valBpb !== undefined && (
              <Readout
                label={`Validation bpb, step ${valStep} (lower is better)`}
                value={bpb(valBpb)}
                unit="bpb"
                sub={readoutSub}
              />
            )}
            {live.tokensPerSec !== undefined && (
              <Readout label="Throughput" value={countText(live.tokensPerSec)} unit="tokens/s" />
            )}
            {maxSteps > 0 && (
              <div className={styles.progress}>
                <Progress step={step} total={maxSteps} eta={eta} label="Training progress" />
              </div>
            )}
          </section>
        )}

        {metrics.data && evalPoints(metrics.data.rows as Row[]).steps.length > 0 && (
          <section className={styles.panel}>
            <Curve
              name={ref}
              run={evalPoints(metrics.data.rows as Row[])}
              baseline={
                baseline && seeds.length > 0 && seeds.every((q) => q.data)
                  ? {
                      name: `shipped ${String(config?.model?.["class"] ?? "").toLowerCase()}, ${baseline.n_seeds} seeds`,
                      seeds: seeds.map((q) => evalPoints((q.data?.rows ?? []) as Row[])),
                    }
                  : null
              }
              yLabel="Validation bpb (lower is better)"
            />
          </section>
        )}

        {samples.data && samples.data.length > 0 && (
          <section className={styles.panel}>
            <Samples
              samples={samples.data}
              caption={
                typeof preset["sample_interval"] === "number"
                  ? `every ${preset["sample_interval"]} steps, ${preset["sample_length"]} tokens, temperature ${preset["sample_temperature"]}`
                  : undefined
              }
            />
          </section>
        )}

        {["done", "stopped"].includes(state ?? "") && (
          <section className={styles.panel}>
            <GeneratePanel runRef={ref} />
          </section>
        )}
      </div>

      <aside className={styles.aside}>
        <section className={styles.panel}>
          <h2 className={`label ${styles.muted}`}>Run</h2>
          <dl className={`small ${styles.facts}`}>
            <dt className={styles.muted}>Model</dt>
            <dd>
              <span className="value">{String(config?.model?.["class"] ?? "")}</span>
            </dd>
            <dt className={styles.muted}>Preset</dt>
            <dd>
              <span className="value">{preset["name"]}</span>
            </dd>
            <dt className={styles.muted}>Seed</dt>
            <dd>
              <span className="value">{config?.seed}</span>
            </dd>
            {typeof stats["n_params"] === "number" && (
              <>
                <dt className={styles.muted}>Params</dt>
                <dd>
                  <span className="value">
                    {countText(stats["n_params"])}
                    {typeof stats["n_non_embedding_params"] === "number" &&
                      ` (${countText(stats["n_non_embedding_params"])} non-embedding)`}
                  </span>
                </dd>
              </>
            )}
            {typeof status?.["started_at"] === "string" && (
              <>
                <dt className={styles.muted}>Started</dt>
                <dd>
                  <span className="value">{stampText(status["started_at"])}</span>
                </dd>
              </>
            )}
          </dl>
        </section>
        <section className={styles.panel}>
          <h2 className={`label ${styles.muted}`}>Files</h2>
          <span className={`value ${styles.files}`}>runs/{ref}/</span>
          <span className={`small ${styles.muted}`}>
            <span className="value">status.json</span> · <span className="value">metrics.jsonl</span> ·{" "}
            <span className="value">config.json</span>
          </span>
        </section>
        <EquivalentCommand cli={`nanoscope status runs/${folder}`} />
      </aside>
    </div>
  );
}

function Traceback({ lines }: { lines: string[] }) {
  const text = lines.join("\n");
  const [copied, setCopied] = useState(false);
  return (
    <section className={styles.trace} aria-label="Traceback">
      <header className={styles.traceHead}>
        <h2 className="heading">Traceback, last {lines.length} lines</h2>
        <Button
          variant="quiet"
          size="sm"
          onClick={() => {
            // inside the click handler: the clipboard needs a user gesture
            void navigator.clipboard?.writeText(text).then(() => setCopied(true));
          }}
        >
          <Copy size={16} aria-hidden="true" />
          {copied ? "Copied" : "Copy"}
        </Button>
      </header>
      <pre className={`${styles.traceBody} code-small`}>{text}</pre>
    </section>
  );
}
