import { describe, expect, it } from "vitest";
import { LEVELS } from "./app/level";
import { CONTROLS, shows, withoutHidden, type ControlId } from "./levels";

const ids = Object.keys(CONTROLS) as ControlId[];

describe("levels", () => {
  it("shows a control at its level and every level above, never below", () => {
    for (const id of ids) {
      const first = LEVELS.indexOf(CONTROLS[id]);
      LEVELS.forEach((level, i) => expect(shows(id, level), `${id} at ${level}`).toBe(i >= first));
    }
  });

  it("everything that Learn shows is still shown at every other level", () => {
    for (const id of ids.filter((x) => CONTROLS[x] === "Learn")) {
      for (const level of LEVELS) expect(shows(id, level)).toBe(true);
    }
  });

  it("each level adds controls", () => {
    for (const level of LEVELS) {
      expect(ids.some((id) => CONTROLS[id] === level), level).toBe(true);
    }
  });

  it("resets hidden fields to their level-0 defaults and keeps visible ones", () => {
    const defaults = { seeds: 1, lr: 0.001, record: false };
    const edited = { seeds: 5, lr: 0.01, record: true };
    const controls = { seeds: "seeds", record: "recordMode" } as const;
    expect(withoutHidden(edited, defaults, controls, "Learn")).toEqual({ seeds: 1, lr: 0.01, record: false });
    expect(withoutHidden(edited, defaults, controls, "Tinker")).toEqual({ seeds: 5, lr: 0.01, record: false });
    expect(withoutHidden(edited, defaults, controls, "Research")).toEqual(edited);
  });
});
