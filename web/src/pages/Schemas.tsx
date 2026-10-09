import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api/client";
import { unwrap } from "../api/problem";
import { EmptyState } from "../components/EmptyState";
import { EquivalentCommand } from "../components/EquivalentCommand";
import { ProblemFromError } from "../components/ProblemView";
import { MagnifyingGlass } from "../icons";
import { fieldRows, type Schema } from "../schemas/rows";
import styles from "./Schemas.module.css";

// Every file nanoscope writes for other tools to read, and every document the API returns, follows
// one of these JSON Schemas. Read straight from GET /api/schemas, so the page cannot disagree
// with the server.
export function Schemas() {
  const { name } = useParams();
  const [filter, setFilter] = useState("");
  const list = useQuery({ queryKey: ["schemas"], queryFn: () => unwrap(api.GET("/api/schemas")) as unknown as Promise<Record<string, number>> });
  const version = useQuery({ queryKey: ["version"], queryFn: () => unwrap(api.GET("/api/version")) as unknown as Promise<{ nanoscope: string }> });
  const names = Object.keys(list.data ?? {}).sort();
  const shown = names.filter((n) => n.includes(filter.trim().toLowerCase()));
  const selected = name ?? names[0];
  const schema = useQuery({
    queryKey: ["schemas", selected],
    enabled: selected !== undefined,
    queryFn: () => unwrap(api.GET("/api/schemas/{name}", { params: { path: { name: selected ?? "" } } })) as unknown as Promise<Schema>,
  });

  const s = schema.data;
  const rows = s ? fieldRows(s) : [];
  const v = selected ? list.data?.[selected] : undefined;
  const strict = s?.additionalProperties === false;

  return (
    <div className={styles.page}>
      <header className={styles.head}>
        <span className={`small ${styles.muted}`}>
          <Link to="/settings" className={styles.link}>
            Settings
          </Link>{" "}
          / This server
        </span>
        <div className={styles.titleRow}>
          <h1 className="title">Schemas</h1>
          <span className={`small ${styles.muted}`}>
            {names.length} JSON Schemas (2020-12){version.data && <> · nanoscope <code className="value">{version.data.nanoscope}</code></>}
          </span>
          <span className={styles.spacer} />
          <a href="/api/docs" className={`small ${styles.link}`}>
            API reference (/docs)
          </a>
        </div>
        <span className={`small ${styles.muted} ${styles.measure}`}>
          Every file nanoscope writes for other tools to read, and every document the API returns, follows one of these. Each file carries <code className="value">schema</code> (its version) and <code className="value">nanoscope</code> (the version that wrote it); readers accept the current version and the one before it.
        </span>
      </header>

      {list.error && <ProblemFromError error={list.error} />}
      <div className={styles.cols}>
        <nav aria-label="Schemas" className={styles.nav}>
          <div className={styles.search}>
            <MagnifyingGlass size={16} aria-hidden="true" className={styles.glass} />
            <input className={`small ${styles.input}`} aria-label="Find a schema" placeholder="Find a schema" value={filter} onChange={(e) => setFilter(e.target.value)} />
          </div>
          {shown.map((n) => (
            <Link key={n} to={`/schemas/${n}`} aria-current={n === selected ? "page" : undefined} className={`${styles.item} ${n === selected ? styles.itemOn : ""}`}>
              <code className="value">{n}</code>
              <span className={`caption ${styles.muted}`}>v{list.data?.[n]}</span>
            </Link>
          ))}
          {shown.length === 0 && list.data && <span className={`small ${styles.muted} ${styles.none}`}>No schema is called that.</span>}
        </nav>

        <section className={styles.detail}>
          {schema.error && <ProblemFromError error={schema.error} />}
          {s && selected && (
            <>
              <div className={styles.panel}>
                <div className={styles.titleRow}>
                  <h2 className="heading">
                    <code className="value-strong">{selected}.v{v}</code>
                  </h2>
                  <span className={`small ${styles.muted}`}>current version</span>
                  <span className={styles.spacer} />
                  <a href={`/api/schemas/${selected}`} className={`small ${styles.link}`}>
                    {selected}.v{v}.json
                  </a>
                </div>
                {s.description && <p className="body">{s.description}</p>}
                <dl className={`small ${styles.facts}`}>
                  <dt className={styles.muted}>$id</dt>
                  <dd>
                    <code className="value">{s.$id}</code>
                  </dd>
                  <dt className={styles.muted}>Extra keys</dt>
                  <dd>{strict ? <>refused at the top level (<code className="value">additionalProperties: false</code>)</> : "allowed"}</dd>
                </dl>
              </div>
              {rows.length === 0 ? (
                <EmptyState title="No fields to list" body="This schema describes no properties of its own; open the JSON file for the whole of it." />
              ) : (
                <table className={styles.table} aria-label={`Fields of ${selected}`}>
                  <thead>
                    <tr>
                      <th className="label" scope="col">Field</th>
                      <th className="label" scope="col">Type</th>
                      <th className="label" scope="col">Required</th>
                      <th className="label" scope="col">Description</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((r, i) => (
                      <tr key={`${r.field}-${i}`}>
                        <td style={{ paddingLeft: 12 + r.depth * 16 }}>
                          <code className="value">{r.field}</code>
                        </td>
                        <td>
                          <code className={`value ${styles.muted}`}>{r.type}</code>
                        </td>
                        <td className="small">{r.required ? "yes" : "no"}</td>
                        <td className="small">{r.description || "–"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
              <span className={`caption ${styles.muted}`}>
                Rows come from the schema as served by <code className="value">GET /api/schemas/{selected}</code>; a dash is a field the schema does not describe.
              </span>
              <EquivalentCommand cli={`curl http://127.0.0.1:8765/api/schemas/${selected}`} python={`from nanoscope import schemas\n\nschemas.get("${selected}")      # the current version\nschemas.CURRENT             # every name and its version`} />
            </>
          )}
        </section>
      </div>
    </div>
  );
}
