import { NavLink, Outlet } from "react-router-dom";
import { LevelSwitch } from "../components/LevelSwitch";
import { Mark } from "../components/Mark";
import { QueuePanel } from "../components/QueuePanel";
import { ThemeToggle } from "../components/ThemeToggle";
import { setLevel, useLevel } from "./level";
import styles from "./Shell.module.css";

// Models joins when the model page lands (P14.34).
export const NAV = [
  { to: "/learn", label: "Learn" },
  { to: "/runs", label: "Runs" },
  { to: "/compare", label: "Compare" },
  { to: "/studies", label: "Studies" },
  { to: "/hardware", label: "Hardware" },
  { to: "/components", label: "Components" },
] as const;

export function Shell() {
  const level = useLevel();
  return (
    <div className={styles.shell}>
      <header className={styles.bar}>
        <NavLink to="/" className={styles.brand} aria-label="nanoscope, home">
          <Mark />
          <span className="heading">nanoscope</span>
        </NavLink>
        <nav className={`${styles.nav} body`} aria-label="Main">
          {NAV.map((item) => (
            <NavLink key={item.to} to={item.to} className={styles.link}>
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className={`${styles.right} body`}>
          <LevelSwitch value={level} onChange={setLevel} />
          <NavLink to="/settings" className={styles.link}>
            Settings
          </NavLink>
          <ThemeToggle />
        </div>
      </header>
      <main className={styles.content}>
        <Outlet />
      </main>
      <QueuePanel />
    </div>
  );
}
