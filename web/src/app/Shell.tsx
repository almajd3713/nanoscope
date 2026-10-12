import { useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { LevelSwitch } from "../components/LevelSwitch";
import { Mark } from "../components/Mark";
import { QueuePanel } from "../components/QueuePanel";
import { ThemeToggle } from "../components/ThemeToggle";
import { CaretDown } from "../icons";
import { shows } from "../levels";
import { setLevel, useLevel, type Level } from "./level";
import { useUnlockOffer } from "./unlockOffer";
import styles from "./Shell.module.css";

export const NAV = [
  { to: "/learn", label: "Learn" },
  { to: "/models", label: "Models" },
  { to: "/runs", label: "Runs" },
  { to: "/compare", label: "Compare" },
  { to: "/studies", label: "Studies" },
  { to: "/hardware", label: "Hardware" },
  { to: "/components", label: "Components" },
] as const;

// The name of the page you are on, for the phone's top bar (where the nav is behind a menu).
function pageName(path: string): string {
  if (path.startsWith("/settings")) return "Settings";
  if (path.startsWith("/workspace")) return "Workspace";
  if (path.startsWith("/model/")) return "Models";
  return NAV.find((item) => path === item.to || path.startsWith(item.to + "/"))?.label ?? "";
}

export function Shell() {
  const level = useLevel();
  const location = useLocation();
  // the menu is open for the page it was opened on: going anywhere closes it
  const [openAt, setOpenAt] = useState<string | null>(null);
  const menu = openAt === location.pathname;
  const offer = useUnlockOffer();
  const choose = (next: Level) => {
    setLevel(next);
    void offer.onSwitch(next);
  };
  return (
    <div className={styles.shell}>
      <header className={`${styles.bar} ${menu ? styles.open : ""}`}>
        <NavLink to="/" className={styles.brand} aria-label="nanoscope, home">
          <Mark />
          <span className={`heading ${styles.wordmark}`}>nanoscope</span>
        </NavLink>
        <span className={`${styles.page} body-strong`}>{pageName(location.pathname)}</span>
        <button type="button" className={`${styles.menuButton} body-strong`} aria-expanded={menu} aria-controls="shell-menu" onClick={() => setOpenAt(menu ? null : location.pathname)}>
          Menu
          <CaretDown size={16} aria-hidden="true" />
        </button>
        <nav id="shell-menu" className={`${styles.nav} body`} aria-label="Main">
          {NAV.map((item) => (
            <NavLink key={item.to} to={item.to} className={styles.link}>
              {item.label}
            </NavLink>
          ))}
          {shows("workspaceTree", level) && (
            <NavLink to="/workspace" className={styles.link}>
              Workspace
            </NavLink>
          )}
        </nav>
        <div className={`${styles.right} body`}>
          <LevelSwitch value={level} onChange={choose} />
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
      {offer.dialog}
    </div>
  );
}
