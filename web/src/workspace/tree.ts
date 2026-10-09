// The workspace as rows: every file the server can read, nested by folder, each with what
// nanoscope finds in it (a model, a block, a study, a lesson) and the page that works on it.
// Everything here is read off what the API already says; nothing opens or runs a file.

export type FileEntry = { path: string; size: number; modified: number; etag: string };
export type ModelRef = { name: string; ref: string; shipped?: boolean };
export type UserBlock = { name: string; file?: string | null; user?: boolean; certified?: string | boolean | null };
export type StudyRef = { name: string; spec?: string | null; runs_total?: number };
export type LessonFolder = { folder: string; id: string; files: string[] };

export type Sources = {
  files: FileEntry[];
  models: ModelRef[];
  blocks: UserBlock[];
  studies: StudyRef[];
  lessons: LessonFolder[];
  changed: string[];
};

export type Seen = { kind: string; link?: string; to?: string; note?: string };

export type Row = {
  key: string;
  path: string;
  name: string;
  depth: number;
  folder: boolean;
  file?: FileEntry;
  changed: boolean;
  seen: Seen;
};

// Files and folders nanoscope hides (the server skips them too; this is for a stale list).
export const HIDDEN = [".git", "__pycache__", ".venv", "node_modules", ".ipynb_checkpoints"];

function modelsIn(path: string, models: ModelRef[]): ModelRef[] {
  return models.filter((m) => !m.shipped && m.ref.startsWith(`${path}:`));
}

export function seeFile(path: string, s: Sources): Seen {
  // the classes in a lesson folder are the lesson's starter and solution, not your models
  if (path.startsWith("curricula/")) return { kind: "" };
  const models = modelsIn(path, s.models);
  if (models.length > 0) {
    const m = models[0]!;
    return { kind: "model", link: m.name, to: `/model/${path}?class=${encodeURIComponent(m.name)}`, note: models.length > 1 ? `· and ${models.length - 1} more` : undefined };
  }
  const block = s.blocks.find((b) => b.user && b.file === path);
  if (block) {
    return { kind: "block", link: block.name, to: `/model/${path}`, note: block.certified ? "· certified" : "· not certified" };
  }
  const study = s.studies.find((x) => x.spec === path);
  if (study) {
    return { kind: "study", link: study.name, to: `/studies/${study.name}/edit`, note: study.runs_total ? `· ${study.runs_total} runs` : "· no runs yet" };
  }
  if (path.endsWith(".py")) return { kind: "python", to: `/model/${path}` };
  return { kind: "" };
}

export function seeFolder(path: string, s: Sources): Seen {
  const lesson = s.lessons.find((l) => l.folder === path);
  if (lesson) return { kind: "lesson", link: lesson.id, to: `/authoring/${lesson.folder}`, note: "· authoring preview" };
  const parts = path.split("/");
  if (parts[0] === "curricula" && parts.length === 2) return { kind: "path" };
  const copy = /^lessons\/([^/]+)\/([^/]+)$/.exec(path);
  if (copy) return { kind: "lesson", link: `${copy[1]}/${copy[2]}`, to: `/learn/${copy[1]}/${copy[2]}`, note: "· your copy" };
  if (path === "lessons") return { kind: "your copies of lesson starters" };
  return { kind: "" };
}

// Rows in folder order (folders first, then files, by name), skipping the children of a folder
// that is closed.
export function rows(s: Sources, closed: ReadonlySet<string>): Row[] {
  type Node = { name: string; path: string; folders: Map<string, Node>; files: FileEntry[] };
  const root: Node = { name: "", path: "", folders: new Map(), files: [] };
  for (const file of s.files) {
    const parts = file.path.split("/");
    if (parts.some((p) => HIDDEN.includes(p))) continue;
    let node = root;
    for (const part of parts.slice(0, -1)) {
      const path = node.path ? `${node.path}/${part}` : part;
      let next = node.folders.get(part);
      if (!next) {
        next = { name: part, path, folders: new Map(), files: [] };
        node.folders.set(part, next);
      }
      node = next;
    }
    node.files.push(file);
  }
  const out: Row[] = [];
  const changed = new Set(s.changed);
  const walk = (node: Node, depth: number) => {
    for (const folder of [...node.folders.values()].sort((a, b) => a.name.localeCompare(b.name))) {
      out.push({ key: `d:${folder.path}`, path: folder.path, name: folder.name, depth, folder: true, changed: false, seen: seeFolder(folder.path, s) });
      if (!closed.has(folder.path)) walk(folder, depth + 1);
    }
    for (const file of [...node.files].sort((a, b) => a.path.localeCompare(b.path))) {
      out.push({
        key: `f:${file.path}`,
        path: file.path,
        name: file.path.split("/").at(-1)!,
        depth,
        folder: false,
        file,
        changed: changed.has(file.path),
        seen: seeFile(file.path, s),
      });
    }
  };
  walk(root, 0);
  return out;
}

export function sizeText(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} kB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}
