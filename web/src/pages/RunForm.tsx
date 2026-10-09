import { keepPreviousData, useMutation, useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { api } from "../api/client";
import { unwrap } from "../api/problem";
import { Button } from "../components/Button";
import { EquivalentCommand } from "../components/EquivalentCommand";
import { Field } from "../components/Field";
import { ProblemFromError, ProblemView } from "../components/ProblemView";
import { CaretDown, CaretRight, Play, X } from "../icons";
import { buildKwargs, commands, parseSeeds, parseSteps, seedList, showValue } from "../runRequest";
import styles from "./RunForm.module.css";

type ModelDoc = {
  name: string;
  ref: string;
  doc: string;
  shipped: boolean;
  params: { name: string; annotation: string | null; default: unknown; required: boolean; from_data: boolean }[];
};
type PresetDoc = {
  name: string;
  fields: { name: string; type: string; default: unknown; help: string }[];
};
type Original = {
  ref: string;
  modelClass: string;
  preset: string;
  modelKwargs: Record<string, unknown>;
  presetFields: Record<string, unknown>;
  seeds: number[];
};

const PRIMARY_PRESET_FIELDS = ["max_steps", "batch_size", "learning_rate", "warmup_steps", "weight_decay", "eval_interval"];

function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const id = setTimeout(() => setV(value), ms);
    return () => clearTimeout(id);
  }, [value, ms]);
  return v;
}

export function RunForm() {
  const [params] = useSearchParams();
  const from = params.get("from");
  const models = useQuery({ queryKey: ["models"], queryFn: () => unwrap(api.GET("/api/models")) });
  const presets = useQuery({ queryKey: ["presets"], queryFn: () => unwrap(api.GET("/api/presets")) });
  const original = useQuery({
    queryKey: ["run", from],
    queryFn: () => unwrap(api.GET("/api/runs/{ref}", { params: { path: { ref: from ?? "" } } })),
    enabled: from !== null,
  });
  const folder = from ? from.split("/").slice(0, -1).join("/") : "";
  const siblings = useQuery({
    queryKey: ["runs", "prefix", folder],
    queryFn: () => unwrap(api.GET("/api/runs", { params: { query: { prefix: folder } } })),
    enabled: from !== null,
  });

  const error = models.error ?? presets.error ?? original.error;
  if (error) return <ProblemFromError error={error} />;
  if (!models.data || !presets.data || (from !== null && (!original.data || !siblings.data))) {
    return <p className={`body ${styles.muted}`}>Loading the form…</p>;
  }

  let origin: Original | null = null;
  if (from !== null && original.data?.config && siblings.data) {
    const config = original.data.config as unknown as { model: { class: string; kwargs?: Record<string, unknown> }; preset: Record<string, unknown>; seed: number };
    origin = {
      ref: from,
      modelClass: config.model.class,
      preset: String(config.preset["name"]),
      modelKwargs: config.model.kwargs ?? {},
      presetFields: config.preset,
      seeds: siblings.data.map((r) => Number(r.ref.split("/seed-").pop())).filter((n) => Number.isFinite(n)).sort((a, b) => a - b),
    };
  }
  return <Form models={models.data as ModelDoc[]} presets={presets.data as PresetDoc[]} origin={origin} />;
}

function Form({ models, presets, origin }: { models: ModelDoc[]; presets: PresetDoc[]; origin: Original | null }) {
  const navigate = useNavigate();
  const [modelRef, setModelRef] = useState(
    () =>
      (origin && models.find((m) => m.name === origin.modelClass)?.ref) ||
      models.find((m) => m.ref.endsWith(":GPT2"))?.ref ||
      models[0]?.ref ||
      "",
  );
  const [preset, setPreset] = useState(origin?.preset ?? "tinystories-5min");
  const [edits, setEdits] = useState<Record<string, string>>({});
  const [extra, setExtra] = useState<{ key: string; value: string }[]>([{ key: "", value: "" }]);
  const [seedsText, setSeedsText] = useState(origin && origin.seeds.length > 0 ? String(origin.seeds.length) : "1");
  const [allPreset, setAllPreset] = useState(false);

  const model = models.find((m) => m.ref === modelRef) ?? models[0]!;
  const presetDoc = presets.find((p) => p.name === preset) ?? presets[0]!;
  const sameAsOrigin = origin !== null && model.name === origin.modelClass;
  const samePreset = origin !== null && preset === origin.preset;

  const modelFields = model.params.filter((p) => !p.from_data);
  const presetFields = presetDoc.fields.filter((f) => f.name !== "name");

  // Where a field starts: the original run's value while duplicating, else the default.
  const originText = (key: string): string => {
    const [kind, name] = [key[0], key.slice(2)];
    if (kind === "m") {
      const p = model.params.find((x) => x.name === name);
      return sameAsOrigin && origin && name in origin.modelKwargs ? showValue(origin.modelKwargs[name]) : showValue(p?.default);
    }
    const f = presetDoc.fields.find((x) => x.name === name);
    return samePreset && origin && name in origin.presetFields ? showValue(origin.presetFields[name]) : showValue(f?.default);
  };
  const current = (key: string): string => edits[key] ?? originText(key);
  const setField = (key: string, value: string) => setEdits((e) => ({ ...e, [key]: value }));

  const kwargs = buildKwargs(
    [
      ...modelFields.map((p) => ({ name: `m:${p.name}`, annotation: p.annotation, default: p.default })),
      ...presetFields.map((f) => ({ name: `p:${f.name}`, annotation: f.type, default: f.default })),
    ],
    current,
    extra,
  );
  // the keys above carry a prefix to keep model and preset fields apart; the request does not
  const request = Object.fromEntries(Object.entries(kwargs).map(([k, v]) => [k.replace(/^[mp]:/, ""), v]));
  const seeds = parseSeeds(seedsText);
  const [stepsText, setStepsText] = useState("");
  const steps = parseSteps(stepsText);

  const changes = [...modelFields.map((p) => `m:${p.name}`), ...presetFields.map((f) => `p:${f.name}`)]
    .filter((key) => key in edits && edits[key] !== originText(key))
    .map((key) => ({ key, name: key.slice(2), from: originText(key), to: edits[key] as string }));

  const body = useDebounced(JSON.stringify({ model: modelRef, preset, kwargs: request, seeds, checkpoint_steps: steps ?? [] }), 300);
  const validation = useQuery({
    queryKey: ["validate", "run", body],
    queryFn: () => unwrap(api.POST("/api/validate/run", { body: JSON.parse(body) })),
    placeholderData: keepPreviousData,
  });
  const problems = validation.data?.problems ?? [];
  const refs = validation.data?.refs ?? [];

  const known = new Set([...modelFields.map((p) => p.name), ...presetFields.map((f) => f.name)]);
  const fieldProblem = (name: string) => problems.find((p) => p.field === name && known.has(name))?.message;
  const extraProblems = problems.filter((p) => p.field && !known.has(p.field) && !["seeds", "preset", "model"].includes(p.field));
  const wholeProblems = problems.filter((p) => !p.field);

  const train = useMutation({
    mutationFn: async () => {
      const list = seedList(seeds);
      const results = await Promise.all(
        list.map((seed) =>
          unwrap(
            api.POST("/api/runs", {
              body: { model: modelRef, preset, seed, kwargs: request, compile: false, wandb: false, checkpoint_steps: steps ?? [] },
            }),
          ) as Promise<{ ref: string }>,
        ),
      );
      return results;
    },
    onSuccess: (results) => {
      const first = results[0]?.ref ?? "";
      navigate(results.length === 1 ? `/runs/${first}` : `/runs?prefix=${encodeURIComponent(first.split("/").slice(0, -1).join("/"))}`);
    },
  });

  const count = seedList(seeds).length;
  const cmds = commands({
    model: modelRef,
    shippedClass: model.shipped ? model.name : null,
    preset,
    seeds,
    kwargs: request,
    checkpointSteps: steps ?? [],
  });

  const field = (key: string, label: string, help: string | undefined, disabled = false) => (
    <Field
      key={key}
      label={label}
      value={disabled ? "" : current(key)}
      disabled={disabled}
      help={disabled ? "Set by the preset's data." : help}
      problem={fieldProblem(key.slice(2))}
      changedFrom={!disabled && key in edits && edits[key] !== originText(key) ? originText(key) : undefined}
      onChange={(v) => setField(key, v)}
    />
  );

  const bools = modelFields.filter((p) => p.annotation === "bool");
  const shownPreset = allPreset ? presetFields : presetFields.filter((f) => PRIMARY_PRESET_FIELDS.includes(f.name));

  return (
    <div className={styles.page}>
      <div className={styles.main}>
        <header className={styles.head}>
          <span className={`small ${styles.muted}`}>
            {origin ? (
              <>
                <Link to={`/runs/${origin.ref}`} className={`value ${styles.link}`}>
                  {origin.ref}
                </Link>{" "}
                / duplicate
              </>
            ) : (
              <>
                <Link to="/runs" className={styles.link}>
                  Runs
                </Link>{" "}
                / new
              </>
            )}
          </span>
          <h1 className="title">{origin ? "Duplicate and change one thing" : "New run"}</h1>
          <p className={`body ${styles.muted}`} style={{ maxWidth: "68ch" }}>
            {origin
              ? "Every field starts from the original run's config. Change one thing, train the same seeds, and the comparison tells you what that one thing did."
              : "The fields come from the model's constructor and the preset. A field you leave alone keeps its default and is not sent."}
          </p>
        </header>

        {origin && (
          <section className={styles.changes} aria-label="Changes">
            <span className="body-strong">
              {changes.length} {changes.length === 1 ? "change" : "changes"}
            </span>
            {changes.map((c) => (
              <span key={c.key} className="value">
                {c.name} <span className={styles.muted}>{c.from || "(empty)"}</span> → <span className="value-strong">{c.to || "(empty)"}</span>
              </span>
            ))}
            <span className={styles.spacer} />
            <Button variant="quiet" size="sm" onClick={() => setEdits({})}>
              Reset to the original
            </Button>
          </section>
        )}

        {wholeProblems.map((p) => (
          <ProblemView key={p.message} title={p.code === "locked" ? "Locked until you build it" : "Cannot be done as asked"} detail={[p.message, p.hint].filter(Boolean).join("\n")} />
        ))}

        <section className={styles.panel} aria-label="Model settings">
          <div className={styles.row}>
            <label className={styles.selectField}>
              <span className="label">Model</span>
              <select className={`${styles.select} value`} value={modelRef} onChange={(e) => setModelRef(e.target.value)}>
                {models.map((m) => (
                  <option key={m.ref} value={m.ref}>
                    {m.ref}
                  </option>
                ))}
              </select>
            </label>
            {model.doc && <span className={`small ${styles.muted}`}>{model.doc.split("\n")[0]}</span>}
          </div>
          <div className={styles.grid}>
            {model.params
              .filter((p) => p.annotation !== "bool")
              .map((p) =>
                field(`m:${p.name}`, p.name, p.annotation ? `${p.annotation}` : undefined, p.from_data),
              )}
          </div>
          {bools.length > 0 && (
            <fieldset className={styles.parts}>
              <legend className="label">Parts</legend>
              {bools.map((p) => {
                const key = `m:${p.name}`;
                const changed = key in edits && edits[key] !== originText(key);
                return (
                  <label key={p.name} className={`small ${styles.part} ${changed ? styles.partChanged : ""}`}>
                    <input
                      type="checkbox"
                      checked={current(key) === "true"}
                      onChange={(e) => setField(key, e.target.checked ? "true" : "false")}
                    />
                    <span className={changed ? "value-strong" : "value"}>{p.name}</span>
                    {changed && <span className="caption">changed</span>}
                  </label>
                );
              })}
            </fieldset>
          )}
        </section>

        <section className={styles.panel} aria-label="Preset settings">
          <div className={styles.row}>
            <label className={styles.selectField}>
              <span className="label">Preset</span>
              <select className={`${styles.select} value`} value={preset} onChange={(e) => setPreset(e.target.value)}>
                {presets.map((p) => (
                  <option key={p.name} value={p.name}>
                    {p.name}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div className={styles.grid}>{shownPreset.map((f) => field(`p:${f.name}`, f.name, f.help))}</div>
          <Button variant="quiet" size="sm" onClick={() => setAllPreset((v) => !v)} aria-expanded={allPreset}>
            {allPreset ? <CaretDown size={16} aria-hidden="true" /> : <CaretRight size={16} aria-hidden="true" />}
            {allPreset ? "Fewer preset fields" : `All ${presetFields.length} preset fields`}
          </Button>
        </section>

        <section className={`${styles.panel} ${styles.seeds}`} aria-label="Seeds and other keywords">
          <Field
            label="Seeds"
            value={seedsText}
            onChange={setSeedsText}
            help={typeof seeds === "number" ? `Trains seeds ${seedList(seeds).join(", ")}. Three is the fewest that gives a confidence interval.` : undefined}
            problem={problems.find((p) => p.field === "seeds")?.message}
          />
          <Field
            label="Keep checkpoints at steps"
            value={stepsText}
            onChange={setStepsText}
            help="Optional, for example 100, 500, 2000. These are kept for good, so the Inspect page can look at the model at each one."
            problem={
              steps === null
                ? "Write step numbers separated by commas."
                : problems.find((p) => p.field === "checkpoint_steps")?.message
            }
          />
          <div className={styles.panel} style={{ padding: 0, border: 0, gap: "var(--space-2)" }}>
            <span className="label">Other keywords</span>
            {extra.map((row, i) => (
              <div key={i} className={styles.kv}>
                <input
                  aria-label="Keyword"
                  className="value"
                  value={row.key}
                  onChange={(e) => setExtra((rows) => rows.map((r, j) => (j === i ? { ...r, key: e.target.value } : r)).concat(i === rows.length - 1 && e.target.value ? [{ key: "", value: "" }] : []))}
                />
                <span className={styles.muted} style={{ textAlign: "center" }}>
                  =
                </span>
                <input
                  aria-label="Value"
                  className="value"
                  value={row.value}
                  onChange={(e) => setExtra((rows) => rows.map((r, j) => (j === i ? { ...r, value: e.target.value } : r)))}
                />
              </div>
            ))}
            {extraProblems.map((p) => (
              <div key={p.message} className={`small ${styles.bad}`} style={{ display: "flex", gap: "var(--space-1)" }}>
                <X size={16} aria-hidden="true" />
                <span>{p.message}</span>
              </div>
            ))}
            {extraProblems[0]?.hint && <div className={`small ${styles.muted}`}>{extraProblems[0].hint}</div>}
          </div>
        </section>

        {train.error && <ProblemFromError error={train.error} />}
        <div className={styles.submit}>
          <Button variant="primary" disabled={problems.length > 0 || train.isPending || count === 0 || steps === null} onClick={() => train.mutate()}>
            <Play size={16} aria-hidden="true" />
            {count > 1 ? `Train ${count} seeds` : "Train"}
          </Button>
          {problems.length > 0 && (
            <span className={`small ${styles.bad}`}>
              Fix {problems.length} {problems.length === 1 ? "problem" : "problems"} to train
            </span>
          )}
        </div>
      </div>

      <aside className={styles.aside}>
        <section className={styles.panel} style={{ gap: "var(--space-2)" }}>
          <h2 className={`label ${styles.muted}`}>Where the runs land</h2>
          {refs.length > 0 ? (
            refs.map((r) => (
              <span key={r} className={`value ${styles.refs}`}>
                {r}
              </span>
            ))
          ) : (
            <span className={`small ${styles.muted}`}>Fix the problems to see where the runs land.</span>
          )}
          <span className={`caption ${styles.muted}`}>
            The name is the model plus a hash of what differs from the defaults, so runs never collide. A seed that is already done is not
            trained again.
          </span>
        </section>
        <section className={styles.panel} style={{ gap: "var(--space-2)" }}>
          <h2 className={`label ${styles.muted}`}>Request, per seed</h2>
          <pre className={`${styles.code} code-small`}>
            {`POST /api/runs\n${JSON.stringify({ model: modelRef, preset, seed: seedList(seeds)[0] ?? 0, kwargs: request }, null, 2)}`}
          </pre>
        </section>
        <EquivalentCommand cli={cmds.cli} python={cmds.python} />
      </aside>
    </div>
  );
}
