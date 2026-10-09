import { VisuallyHidden } from "@radix-ui/react-visually-hidden";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api } from "../api/client";
import { unwrap } from "../api/problem";
import { Button } from "../components/Button";
import { EquivalentCommand } from "../components/EquivalentCommand";
import { Field } from "../components/Field";
import { Note } from "../components/Note";
import { ProblemFromError, ProblemView } from "../components/ProblemView";
import { countText } from "../format";
import { Plus, Trash } from "../icons";
import { freshName, fromSpec, toSpec, type StudyDraft, type VariantDraft } from "../studyDraft";
import { useSizes, useStudyValidation, type Problem, type SizeRow } from "../studies/useStudyChecks";
import styles from "./StudyBuilder.module.css";

type ModelDoc = { name: string; ref: string };
type PresetDoc = { name: string };
type Device = { name: string; kind: string; memory_total: number | null; memory_free: number | null };

const BLANK: StudyDraft = {
  name: "",
  preset: "tinystories-5min",
  budget: "",
  seeds: "0, 1, 2",
  baseline: "",
  match: "none",
  tolerance: "2",
  mode: "explore",
  variants: [],
  rest: {},
};

const gb = (bytes: number | null) => (bytes === null ? "" : `${(bytes / 1024 ** 3).toFixed(1)}`);

// A study as a form: the plan, the variants with their sizes, the compute it would use. The file
// it saves is the committable `studies/<name>.toml`; the library judges every value.
export function StudyBuilder() {
  const { name } = useParams();
  const saved = useQuery({
    queryKey: ["study-spec", name],
    queryFn: () => unwrap(api.GET("/api/studies/{name}/spec", { params: { path: { name: name ?? "" } } })),
    enabled: name !== undefined,
  });
  const models = useQuery({ queryKey: ["models"], queryFn: () => unwrap(api.GET("/api/models")) });
  const presets = useQuery({ queryKey: ["presets"], queryFn: () => unwrap(api.GET("/api/presets")) });
  const hardware = useQuery({ queryKey: ["hardware"], queryFn: () => unwrap(api.GET("/api/hardware")) });

  const error = saved.error ?? models.error ?? presets.error ?? hardware.error;
  if (error) return <ProblemFromError error={error} />;
  if (!models.data || !presets.data || !hardware.data || (name !== undefined && !saved.data)) {
    return <p className="body">Loading the builder…</p>;
  }
  const start = saved.data ? fromSpec(saved.data.spec as Record<string, unknown>) : BLANK;
  return (
    <Builder
      key={name ?? "new"}
      existing={name}
      start={start}
      savedToml={saved.data?.toml ?? null}
      models={models.data as unknown as ModelDoc[]}
      presets={presets.data as unknown as PresetDoc[]}
      devices={(hardware.data as unknown as { devices: Device[] }).devices}
    />
  );
}

function Builder({
  existing,
  start,
  savedToml,
  models,
  presets,
  devices,
}: {
  existing: string | undefined;
  start: StudyDraft;
  savedToml: string | null;
  models: ModelDoc[];
  presets: PresetDoc[];
  devices: Device[];
}) {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [draft, setDraft] = useState<StudyDraft>(start);
  const [picked, setPicked] = useState<string[]>(() => {
    const gpus = devices.filter((d) => d.kind === "cuda").map((d) => d.name);
    return gpus.length > 0 ? gpus : ["cpu"];
  });
  const [perDevice, setPerDevice] = useState("1");
  const set = <K extends keyof StudyDraft>(key: K, value: StudyDraft[K]) => setDraft((d) => ({ ...d, [key]: value }));
  const setVariant = (i: number, patch: Partial<VariantDraft>) =>
    setDraft((d) => ({ ...d, variants: d.variants.map((v, j) => (j === i ? { ...v, ...patch } : v)) }));

  const spec = useMemo(() => toSpec(draft), [draft]);
  const per = Math.max(1, Math.floor(Number(perDevice)) || 1);
  const validation = useStudyValidation(spec, picked, per);
  const verdict = validation.data;
  const sizes = useSizes(spec, verdict?.ok === true);
  const sizeRows = new Map<string, SizeRow>((sizes.data?.variants ?? []).map((r) => [r.name, r]));
  const outside = draft.match === "params" ? (sizes.data?.variants ?? []).filter((r) => !r.within) : [];

  const toml = verdict?.toml ?? null;
  const dirty = existing === undefined ? draft.name !== "" : toml !== null && toml !== savedToml;
  const problems: Problem[] = verdict?.problems ?? [];
  const wholeProblems = problems.filter((p) => !p.field || !p.field.startsWith("variants["));
  const total = draft.variants.length * (Array.isArray(spec["seeds"]) ? (spec["seeds"] as unknown[]).length : 0);

  const save = useMutation({
    mutationFn: () => unwrap(api.POST("/api/studies", { body: { spec, overwrite: existing !== undefined, devices: [], workers_per_device: 1 } })),
    onSuccess: (entry) => {
      void queryClient.invalidateQueries({ queryKey: ["study-spec"] });
      void queryClient.invalidateQueries({ queryKey: ["studies"] });
      const saved = (entry as unknown as { name: string }).name;
      if (existing === undefined) navigate(`/studies/${saved}/edit`, { replace: true });
    },
  });

  // Saving writes the validated file; a changed spec is "unsaved" until then.
  const saveState = save.isPending ? "saving" : existing === undefined ? "not saved yet" : dirty ? "unsaved changes" : "saved";

  const toggleDevice = (device: string, on: boolean) =>
    setPicked((p) => (on ? [...p, device] : p.filter((d) => d !== device)));

  const file = `studies/${draft.name || "name"}.toml`;
  const commandDevices = picked.join(",") || "cpu";

  return (
    <div className={styles.page}>
      <main className={styles.main}>
        <header className={styles.head}>
          <span className={`small ${styles.muted}`}>
            <Link to="/studies" className={styles.link}>Studies</Link> / <span className="value">{draft.name || "new"}</span>
          </span>
          <div className={styles.bar}>
            <h1 className="title">{draft.name || "New study"}</h1>
            <span className={`small ${styles.muted}`}>
              <span className="value">{file}</span> · {saveState}
            </span>
            <span className={styles.spacer} />
            <Button
              onClick={() => save.mutate()}
              disabled={!verdict?.ok || save.isPending}
              title={verdict?.ok ? undefined : "Fix the problems first"}
            >
              Save as TOML
            </Button>
          </div>
        </header>

        {save.error && <ProblemFromError error={save.error} />}
        {wholeProblems.map((p) => (
          <ProblemView key={`${p.field}${p.message}`} title="Cannot be done as asked" detail={[p.message, p.hint].filter(Boolean).join("\n")} />
        ))}

        <section className={styles.panel} aria-label="Plan">
          <h2 className="heading">Plan</h2>
          <div className={styles.grid}>
            <Field label="Name" value={draft.name} onChange={(v) => set("name", v)} disabled={existing !== undefined} help="Letters, digits, - and _" />
            <label className={styles.field}>
              <span className="label">Preset</span>
              <select className={`${styles.select} value`} value={draft.preset} onChange={(e) => set("preset", e.target.value)}>
                {presets.map((p) => (
                  <option key={p.name} value={p.name}>
                    {p.name}
                  </option>
                ))}
              </select>
            </label>
            <Field
              label="Budget (tokens)"
              value={draft.budget}
              onChange={(v) => set("budget", v)}
              help={draft.budget ? "The same for every variant." : "Empty: the preset's own steps."}
            />
            <Field label="Seeds" value={draft.seeds} onChange={(v) => set("seeds", v)} help="A count (3) or a list (0, 4, 7)." />
            <label className={styles.field}>
              <span className="label">Baseline</span>
              <select className={`${styles.select} value`} value={draft.baseline} onChange={(e) => set("baseline", e.target.value)}>
                <option value="">none</option>
                {draft.variants.map((v) => (
                  <option key={v.name} value={v.name}>
                    {v.name}
                  </option>
                ))}
              </select>
              <span className={`caption ${styles.muted}`}>Every other variant is compared with this one.</span>
            </label>
            <div className={styles.field}>
              <span className="label" id="match-label">Match</span>
              <div className={styles.inline}>
                <select
                  aria-labelledby="match-label"
                  className={`${styles.select} value`}
                  value={draft.match}
                  onChange={(e) => set("match", e.target.value as StudyDraft["match"])}
                >
                  <option value="params">params</option>
                  <option value="none">none</option>
                </select>
                <span className={`small ${styles.muted}`}>within</span>
                <input
                  className={`${styles.cell} ${styles.tol} value`}
                  aria-label="Tolerance (percent)"
                  value={draft.tolerance}
                  onChange={(e) => set("tolerance", e.target.value)}
                />
                <span className={`small ${styles.muted}`}>%</span>
              </div>
              <span className={`caption ${styles.muted}`}>Non-embedding parameter counts, from a worker's sizes job.</span>
            </div>
            <div className={styles.field}>
              <span className="label" id="mode-label">Mode</span>
              <div className={`${styles.seg} small`} role="radiogroup" aria-labelledby="mode-label">
                {(["explore", "record"] as const).map((m) => (
                  <button key={m} type="button" role="radio" aria-checked={draft.mode === m} onClick={() => set("mode", m)}>
                    {m === "explore" ? "Explore" : "Record"}
                  </button>
                ))}
              </div>
              <span className={`caption ${styles.muted}`}>
                Record runs only from a committed spec in a clean tree; its predictions are frozen once runs start.
              </span>
            </div>
          </div>
        </section>

        <section className={styles.panel} aria-label="Variants">
          <div className={styles.variantsHead}>
            <h2 className="heading">Variants</h2>
            <span className={`small ${styles.muted}`}>
              {draft.variants.length} {draft.variants.length === 1 ? "variant" : "variants"} × {total && draft.variants.length ? total / draft.variants.length : 0}{" "}
              seeds = {total} runs
              {sizes.isFetching ? " · counting sizes…" : sizes.data ? " · sizes counted" : ""}
            </span>
            <span className={styles.spacer} />
            <Button
              size="sm"
              onClick={() =>
                setDraft((d) => ({
                  ...d,
                  variants: [...d.variants, { name: freshName(d.variants), model: models[0]?.ref ?? "", kwargs: "", predicted: "" }],
                }))
              }
            >
              <Plus size={16} aria-hidden="true" />
              Add variant
            </Button>
          </div>
          {draft.variants.length === 0 ? (
            <p className={`body ${styles.muted}`}>A study needs at least one variant: a model and the keywords that make it different.</p>
          ) : (
            <div className={styles.wrap}>
              <table className={`${styles.table} small`}>
                <thead>
                  <tr>
                    <th className="label" scope="col">Name</th>
                    <th className="label" scope="col">Model</th>
                    <th className="label" scope="col">Keywords</th>
                    <th className={`label ${styles.right}`} scope="col">Non-emb params</th>
                    <th className={`label ${styles.right}`} scope="col">Over {sizes.data?.reference ?? "reference"}</th>
                    <th className={`label ${styles.right}`} scope="col">FLOP/token</th>
                    <th className={`label ${styles.right}`} scope="col">Predicted val_bpb</th>
                    <th className="label" scope="col">
                      <VisuallyHidden>Remove</VisuallyHidden>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {draft.variants.map((v, i) => {
                    const size = sizeRows.get(v.name);
                    const bad = draft.match === "params" && size !== undefined && !size.within;
                    const own = problems.filter((p) => p.field?.startsWith(`variants[${i}]`));
                    return (
                      <tr key={i} data-outside={bad}>
                        <td>
                          <input className={`${styles.cell} ${styles.name} value-strong`} aria-label={`Name of variant ${i + 1}`} value={v.name} onChange={(e) => setVariant(i, { name: e.target.value })} />
                        </td>
                        <td>
                          <select className={`${styles.select} value`} aria-label={`Model of ${v.name}`} value={v.model} onChange={(e) => setVariant(i, { model: e.target.value })}>
                            {!models.some((m) => m.ref === v.model) && <option value={v.model}>{v.model}</option>}
                            {models.map((m) => (
                              <option key={m.ref} value={m.ref}>
                                {m.ref}
                              </option>
                            ))}
                          </select>
                        </td>
                        <td>
                          <input
                            className={`${styles.cell} ${styles.kwargs} value`}
                            aria-label={`Keywords of ${v.name}`}
                            aria-invalid={own.length > 0 || undefined}
                            placeholder="ffn_hidden = 384, rope = false"
                            value={v.kwargs}
                            onChange={(e) => setVariant(i, { kwargs: e.target.value })}
                          />
                          {own.map((p) => (
                            <div key={p.message} className={`caption ${styles.outside}`}>{p.message}</div>
                          ))}
                        </td>
                        <td className={`value ${styles.right} ${bad ? styles.outside : ""}`}>{size ? size.non_embedding_params.toLocaleString("en-US") : "–"}</td>
                        <td className={`value ${styles.right} ${bad ? styles.outside : ""}`}>
                          {size ? (size.name === sizes.data?.reference ? "reference" : `${size.delta >= 0 ? "+" : ""}${(size.delta * 100).toFixed(1)}%`) : "–"}
                        </td>
                        <td className={`value ${styles.right}`}>{size ? countText(size.flops_per_token) : "–"}</td>
                        <td className={styles.right}>
                          <input
                            className={`${styles.cell} ${styles.predicted} value`}
                            aria-label={`Predicted val_bpb for ${v.name}`}
                            placeholder="–"
                            value={v.predicted}
                            onChange={(e) => setVariant(i, { predicted: e.target.value })}
                          />
                        </td>
                        <td>
                          <Button
                            variant="quiet"
                            size="sm"
                            aria-label={`Remove ${v.name}`}
                            onClick={() =>
                              setDraft((d) => ({
                                ...d,
                                baseline: d.baseline === v.name ? "" : d.baseline,
                                variants: d.variants.filter((_, j) => j !== i),
                              }))
                            }
                          >
                            <Trash size={16} aria-hidden="true" />
                          </Button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
          {outside.map((r) => (
            <ProblemView
              key={r.name}
              title={`${r.name} is outside the size tolerance`}
              detail={`${r.non_embedding_params.toLocaleString("en-US")} non-embedding params, ${(Math.abs(r.delta) * 100).toFixed(1)}% ${r.delta > 0 ? "above" : "below"} ${sizes.data?.reference}; match = params allows ${((sizes.data?.tolerance ?? 0) * 100).toFixed(0)}%.`}
            />
          ))}
          {sizes.error && <ProblemFromError error={sizes.error} />}
          {draft.variants.length > 2 && draft.baseline && (
            <Note tone="warn">
              {draft.variants.length - 1} variants are compared with one baseline; each interval is separate, so a lone small win can be luck.
            </Note>
          )}
        </section>
      </main>

      <aside className={styles.aside}>
        <section className={styles.panel} aria-label="Compute">
          <h2 className="label">Compute</h2>
          <fieldset className={styles.devices}>
            <legend className="label">Devices</legend>
            {devices.map((d) => (
              <label key={d.name} className={`small ${styles.device}`}>
                <input type="checkbox" checked={picked.includes(d.name)} onChange={(e) => toggleDevice(d.name, e.target.checked)} />
                <span className="value">{d.name}</span>
                {d.memory_free !== null && (
                  <span className={styles.muted}>
                    {gb(d.memory_free)} of {gb(d.memory_total)} GB free
                  </span>
                )}
              </label>
            ))}
          </fieldset>
          <Field label="Runs at once per device" value={perDevice} onChange={setPerDevice} />
          <div className={styles.estimate}>
            <span className="label">Estimate</span>
            <span className="body">{verdict?.estimate?.text ?? "–"}</span>
            <span className={`caption ${styles.muted}`}>From your bench results. Training steps only.</span>
          </div>
        </section>

        <section className={styles.panel} aria-label="What Save writes">
          <div className={styles.inline}>
            <h2 className="label">What Save writes</h2>
            <span className={`caption ${styles.muted} value`}>{file}</span>
          </div>
          <pre className={`${styles.toml} code`}>{toml ?? "Fix the problems above to see the file."}</pre>
        </section>

        <EquivalentCommand
          cli={`nanoscope study ${file} --dry-run --devices ${commandDevices} --workers-per-device ${per}`}
          python={`from nanoscope.study import load_study\n\nstudy = load_study("${file}")\nprint(study.sizes())\nprint(study.estimate(${JSON.stringify(picked)}, workers_per_device=${per})["text"])`}
        />
      </aside>
    </div>
  );
}
