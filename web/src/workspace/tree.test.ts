import { describe, expect, it } from "vitest";
import { rows, sizeText, type Sources } from "./tree";

const file = (path: string) => ({ path, size: 500, modified: 1, etag: "e" });
const sources: Sources = {
  files: [
    "models/my_lm.py",
    "blocks/scale_norm.py",
    "studies/m1.toml",
    "curricula/my-course/path.toml",
    "curricula/my-course/01-layernorm/lesson.toml",
    "curricula/my-course/01-layernorm/starter.py",
    "lessons/foundations/01-bigram/starter.py",
    ".git/config",
    "notes.txt",
  ].map(file),
  models: [
    { name: "MyThing", ref: "curricula/my-course/01-layernorm/starter.py:MyThing" },
    { name: "Bigram", ref: "nanoscope.models.bigram:Bigram", shipped: true },
    { name: "MyLM", ref: "models/my_lm.py:MyLM" },
  ],
  blocks: [{ name: "ScaleNorm", file: "blocks/scale_norm.py", user: true, certified: null }],
  studies: [{ name: "m1", spec: "studies/m1.toml", runs_total: 0 }],
  lessons: [{ folder: "curricula/my-course/01-layernorm", id: "my-course/01-layernorm", files: ["lesson.toml"] }],
  changed: ["models/my_lm.py"],
};

describe("workspace rows", () => {
  const all = rows(sources, new Set());
  const by = (path: string) => all.find((r) => r.path === path)!;

  it("nests folders before files and hides tool folders", () => {
    expect(all.map((r) => r.path)).not.toContain(".git/config");
    expect(all.map((r) => r.path).slice(0, 3)).toEqual(["blocks", "blocks/scale_norm.py", "curricula"]);
    expect(by("curricula/my-course/01-layernorm/lesson.toml").depth).toBe(3);
  });

  it("says what nanoscope finds in a file and where it goes", () => {
    expect(by("models/my_lm.py").seen).toMatchObject({ kind: "model", link: "MyLM", to: "/model/models/my_lm.py?class=MyLM" });
    expect(by("blocks/scale_norm.py").seen).toMatchObject({ kind: "block", link: "ScaleNorm", note: "· not certified" });
    expect(by("studies/m1.toml").seen).toMatchObject({ kind: "study", to: "/studies/m1/edit", note: "· no runs yet" });
    expect(by("curricula/my-course/01-layernorm").seen).toMatchObject({ kind: "lesson", to: "/authoring/curricula/my-course/01-layernorm" });
    expect(by("curricula/my-course").seen.kind).toBe("path");
    expect(by("lessons/foundations/01-bigram").seen).toMatchObject({ kind: "lesson", to: "/learn/foundations/01-bigram" });
    expect(by("notes.txt").seen.kind).toBe("");
    expect(by("curricula/my-course/01-layernorm/starter.py").seen.kind).toBe("");
  });

  it("marks files git sees as changed", () => {
    expect(by("models/my_lm.py").changed).toBe(true);
    expect(by("blocks/scale_norm.py").changed).toBe(false);
  });

  it("leaves out the children of a closed folder", () => {
    const closed = rows(sources, new Set(["curricula"]));
    expect(closed.some((r) => r.path.startsWith("curricula/"))).toBe(false);
    expect(closed.some((r) => r.path === "curricula")).toBe(true);
  });

  it("writes sizes", () => {
    expect([sizeText(469), sizeText(2048), sizeText(3 * 1024 * 1024)]).toEqual(["469 B", "2.0 kB", "3.0 MB"]);
  });
});
