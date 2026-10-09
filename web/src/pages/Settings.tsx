import { VisuallyHidden } from "@radix-ui/react-visually-hidden";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useSyncExternalStore } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import { unwrap } from "../api/problem";
import { LEVELS, type Level, setLevel, useLevel } from "../app/level";
import { setShowCommands, useShowCommands } from "../app/settings";
import { Button } from "../components/Button";
import { EquivalentCommand } from "../components/EquivalentCommand";
import { ProblemFromError } from "../components/ProblemView";
import { Tag } from "../components/Tag";
import { shows } from "../levels";
import { Check } from "../icons";
import { getChoice, setChoice, subscribe, type ThemeChoice } from "../styles/theme";
import styles from "./Settings.module.css";

const LEVEL_TEXT: Record<Level, string> = {
  Learn: "Lessons, the model graph and its palette, training, the live run page and samples.",
  Tinker: "Adds the code editor, the run form, seeds, duplicate and change one thing, and compare.",
  Research: "Adds studies, budgets, record mode, the queue and devices, forest plots and bench.",
  Extend: "Adds the workspace tree, your own blocks and their certification, and the schemas.",
};

const THEMES: { value: ThemeChoice; label: string }[] = [
  { value: "system", label: "Follow the system" },
  { value: "light", label: "Light" },
  { value: "dark", label: "Dark" },
];

type DataEntry = {
  preset: string;
  dataset: string;
  tokenizer: string;
  vocab_size: number | null;
  prepare: { stage: string; error?: string | null } | null;
};
type ServerSettings = {
  version: string;
  home: string;
  workspace: string;
  data: string;
  runs: string;
  listening: { host: string; port: number; loopback: boolean; token_required: boolean };
  jobs_offline: boolean;
  workers: { worker_id: string; device: string; secrets: Record<string, boolean> }[];
};

export function Settings() {
  const queryClient = useQueryClient();
  const theme = useSyncExternalStore(subscribe, getChoice);
  const level = useLevel();
  const showCommands = useShowCommands();
  const unlocks = useQuery({ queryKey: ["learn", "unlocks"], queryFn: () => unwrap(api.GET("/api/learn/unlocks")) });
  const data = useQuery({ queryKey: ["data"], queryFn: () => unwrap(api.GET("/api/data")) as unknown as Promise<DataEntry[]> });
  const server = useQuery({ queryKey: ["settings"], queryFn: () => unwrap(api.GET("/api/settings")) as unknown as Promise<ServerSettings> });

  const policy = useMutation({
    mutationFn: (value: "guided" | "open") => unwrap(api.POST("/api/learn/policy", { body: { policy: value } })),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["learn"] }),
  });
  const prepare = useMutation({
    mutationFn: (preset: string) => unwrap(api.POST("/api/data/{preset}/prepare", { params: { path: { preset } } })),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["data"] }),
  });

  const s = server.data;
  const hasSecret = (name: string) => (s?.workers ?? []).some((w) => w.secrets[name]);
  const address = s ? `${s.listening.host}:${s.listening.port}` : "";

  return (
    <div className={styles.page}>
      <header className={styles.head}>
        <h1 className="title">Settings</h1>
        <p className={`body ${styles.muted}`}>
          How nanoscope looks in this browser, how lessons gate blocks, the data on this machine, and how this server was started.
        </p>
      </header>

      <section className={styles.section} aria-labelledby="s-browser">
        <div className={styles.sectionHead}>
          <h2 className="heading" id="s-browser">This browser</h2>
          <span className={`small ${styles.muted}`}>saved here only, not on the server</span>
        </div>
        <div className={styles.panel}>
          <fieldset className={styles.group}>
            <legend className="label">Theme</legend>
            <div className={styles.row}>
              {THEMES.map((t) => (
                <label key={t.value} className={`small ${styles.option}`}>
                  <input type="radio" name="theme" checked={theme === t.value} onChange={() => setChoice(t.value)} />
                  {t.label}
                </label>
              ))}
            </div>
          </fieldset>
          <fieldset className={styles.group}>
            <legend className="label">Level</legend>
            {LEVELS.map((l) => (
              <label key={l} className={`small ${styles.levelOption}`}>
                <input type="radio" name="level" checked={level === l} onChange={() => setLevel(l)} />
                <span className="body-strong">{l}</span>
                <span className={styles.muted}>{LEVEL_TEXT[l]}</span>
              </label>
            ))}
            <span className={`caption ${styles.muted}`}>A level only shows or hides controls. Anything you leave hidden keeps its default.</span>
          </fieldset>
          <label className={styles.check}>
            <input type="checkbox" checked={showCommands} onChange={(e) => setShowCommands(e.target.checked)} />
            <span className={styles.stack}>
              <span className="body-strong">Show command-line equivalents</span>
              <span className={`small ${styles.muted}`}>
                Each screen can show the <span className="value">nanoscope</span> command or Python call that does the same thing. Turn this on if
                you script nanoscope, train on Kaggle or Colab, or want to learn the library itself.
              </span>
            </span>
          </label>
        </div>
      </section>

      <section className={styles.section} aria-labelledby="s-learning">
        <h2 className="heading" id="s-learning">Learning</h2>
        <div className={styles.panel} style={{ gap: "var(--space-3)" }}>
          {unlocks.error ? (
            <ProblemFromError error={unlocks.error} />
          ) : (
            <fieldset className={styles.group}>
              <legend className="label">Block gating</legend>
              {(
                [
                  ["guided", "Guided", "Bigger blocks unlock as you pass the lesson that builds them."],
                  ["open", "Open", "Every block is available. What you earned stays recorded."],
                ] as const
              ).map(([value, name, text]) => (
                <label key={value} className={`small ${styles.policyOption}`}>
                  <input
                    type="radio"
                    name="policy"
                    checked={unlocks.data?.policy === value}
                    disabled={!unlocks.data || policy.isPending}
                    onChange={() => policy.mutate(value)}
                  />
                  <span>
                    <span className="body-strong">{name}</span> <span className={`value ${styles.muted}`}>{value}</span>
                    <br />
                    <span className={styles.muted}>{text}</span>
                  </span>
                </label>
              ))}
            </fieldset>
          )}
          {policy.error && <ProblemFromError error={policy.error} />}
          <span className={`small ${styles.muted}`}>
            Saved in <span className="value">learn/unlocks.json</span> on the server. Each block's state is on{" "}
            <Link to="/components" className={styles.link}>Components</Link>.
          </span>
        </div>
      </section>

      <section className={styles.section} aria-labelledby="s-data">
        <div className={styles.sectionHead}>
          <h2 className="heading" id="s-data">Data</h2>
          <span className={`small ${styles.muted}`}>token caches for each preset, on this machine</span>
        </div>
        {data.error ? (
          <ProblemFromError error={data.error} />
        ) : (
          <div className={styles.wrap}>
            <table className={`${styles.table} small`}>
              <thead>
                <tr>
                  <th className="label" scope="col">Preset</th>
                  <th className="label" scope="col">Dataset</th>
                  <th className="label" scope="col">Tokenizer</th>
                  <th className="label" scope="col">State</th>
                  <th className="label" scope="col">
                    <VisuallyHidden>Action</VisuallyHidden>
                  </th>
                </tr>
              </thead>
              <tbody>
                {(data.data ?? []).map((d) => (
                  <tr key={d.preset}>
                    <td className="value">{d.preset}</td>
                    <td className={`value ${styles.muted}`}>{d.dataset}</td>
                    <td className={`value ${styles.muted}`}>
                      {d.tokenizer}
                      {d.vocab_size ? ` ${d.vocab_size}` : ""}
                    </td>
                    <td>
                      {d.prepare?.stage === "done" ? (
                        <Tag tone="good" icon={Check}>done</Tag>
                      ) : d.prepare ? (
                        <span className="small">{d.prepare.error ? `${d.prepare.stage}: ${d.prepare.error}` : d.prepare.stage}</span>
                      ) : (
                        <span className={`small ${styles.muted}`}>not prepared</span>
                      )}
                    </td>
                    <td>
                      {d.prepare?.stage !== "done" && (
                        <Button size="sm" disabled={prepare.isPending} onClick={() => prepare.mutate(d.preset)}>
                          Prepare
                        </Button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {prepare.error && <ProblemFromError error={prepare.error} />}
        <span className={`small ${styles.muted}`}>
          Preparing downloads the tokens from <span className="value">RedhouaneLazib/nanoscope-tokens</span> as a worker job; a run prepares its
          own data if you skip this.
        </span>
      </section>

      <section className={styles.section} aria-labelledby="s-server">
        <div className={styles.sectionHead}>
          <h2 className="heading" id="s-server">This server</h2>
          <span className={`small ${styles.muted}`}>read only: set when the server starts</span>
        </div>
        {server.error ? (
          <ProblemFromError error={server.error} />
        ) : (
          s && (
            <div className={styles.panel}>
              <dl className={`small ${styles.facts}`}>
                <dt className={styles.muted}>Version</dt>
                <dd><span className="value">nanoscope {s.version}</span></dd>
                <dt className={styles.muted}>Home</dt>
                <dd><span className="value">{s.home}</span> <span className={styles.muted}>runs, studies, learning progress, the queue</span></dd>
                <dt className={styles.muted}>Workspace</dt>
                <dd><span className="value">{s.workspace}</span> <span className={styles.muted}>your models, blocks and lesson files</span></dd>
                <dt className={styles.muted}>Data</dt>
                <dd><span className="value">{s.data}</span></dd>
                <dt className={styles.muted}>Address</dt>
                <dd>
                  <span className="value">{address}</span>{" "}
                  <span className={styles.muted}>{s.listening.loopback ? "this machine only" : "reachable from other machines; the token is required"}</span>
                </dd>
                <dt className={styles.muted}>Offline jobs</dt>
                <dd><span className="value">{s.jobs_offline ? "on" : "off"}</span></dd>
                <dt className={styles.muted}>Hugging Face token</dt>
                <dd className={styles.muted}>{hasSecret("HF_TOKEN") ? "set for the workers" : "not set for the workers"}</dd>
                <dt className={styles.muted}>W&amp;B key</dt>
                <dd className={styles.muted}>{hasSecret("WANDB_API_KEY") ? "set for the workers" : "not set for the workers"}</dd>
              </dl>
            </div>
          )
        )}
        <span className={`small ${styles.muted}`}>
          Change these in <span className="value">compose.yaml</span> and its <span className="value">.env</span>, then restart (see{" "}
          <span className="value">docs/deploy.md</span>). Secrets are never shown or sent to the browser. API reference:{" "}
          <a href="/api/docs" className={styles.link}>/api/docs</a>.{shows("schemasDocs", level) && (
            <>
              {" "}
              <Link to="/schemas" className={styles.link}>Schemas</Link>: every file format and API document.
            </>
          )}
        </span>
      </section>
      <EquivalentCommand cli={"nanoscope learn status\nnanoscope learn unlock --all\nnanoscope prepare-data tinystories-30min"} />
    </div>
  );
}
