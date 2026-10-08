import { Mark } from "../components/Mark";
import { Note } from "../components/Note";
import styles from "./Login.module.css";

// Shown when a request comes back 401. The token is a login link printed by `nanoscope serve`
// (or the api container's log); opening it once sets the cookie. No form: nothing to type.
export function Login() {
  return (
    <div className={styles.page}>
      <header className={styles.bar}>
        <Mark />
        <span className="heading">nanoscope</span>
      </header>
      <main className={styles.main}>
        <div className={styles.intro}>
          <h1 className="title">Sign in with the login link</h1>
          <p className="body">
            This server needs its access token. It printed a login link when it started. Open that link once in this
            browser; it sets a cookie and brings you back here.
          </p>
        </div>
        <section className={styles.panel}>
          <h2 className="heading">Running with docker compose</h2>
          <p className={`small ${styles.muted}`}>The api service logs the link on start. Find it with:</p>
          <pre className={`${styles.command} code-small`}>docker compose logs api | grep login</pre>
        </section>
        <section className={styles.panel}>
          <h2 className="heading">Running nanoscope serve yourself</h2>
          <p className={`small ${styles.muted}`}>
            Beyond this machine it prints a line like this in its terminal:
          </p>
          <pre className={`${styles.command} code-small`}>
            listening beyond this machine: sign in at http://&lt;host&gt;:8765/login?token=…
          </pre>
          <p className={`small ${styles.muted}`}>
            The token is also stored in <span className="value">~/.nanoscope/server/token</span>.
          </p>
        </section>
        <Note tone="warn">
          Whoever has the token can save Python files in your workspace and run them as your user. Share it only on a
          network you trust, through an SSH tunnel, or behind an HTTPS proxy.
        </Note>
      </main>
    </div>
  );
}
