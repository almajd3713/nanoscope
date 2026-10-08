import { Outlet } from "react-router-dom";
import { Mark } from "../components/Mark";
import styles from "./Shell.module.css";

// The first-run screens: the brand only, no nav, until a choice has been made.
export function BareLayout() {
  return (
    <div className={styles.shell}>
      <header className={styles.bar}>
        <Mark />
        <span className="heading">nanoscope</span>
      </header>
      <main className={styles.content}>
        <Outlet />
      </main>
    </div>
  );
}
