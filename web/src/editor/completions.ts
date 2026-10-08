// What the editor offers at the cursor: block names, a block's keyword arguments, preset names.
// Pure functions of the text before the cursor and the server's catalogs, so they are tested
// without Monaco; MonacoSurface wraps them in a completion provider.

export type CatalogArg = { name: string; type?: string | null; required: boolean; default?: unknown };
export type CatalogBlock = {
  name: string;
  family: string;
  doc?: string;
  args: CatalogArg[];
  lock?: { locked: boolean; lesson: string | null };
};
export type Catalog = { blocks: CatalogBlock[]; presets: string[] };

export type Suggestion = {
  label: string;
  insertText: string;
  kind: "block" | "argument" | "preset";
  detail: string;
  documentation?: string;
};

type Frame = { call: string; names: Set<string>; segment: string };

// Walk the text up to the cursor, tracking open calls. Strings and comments are skipped, so a
// parenthesis inside either does not count. Returns the open calls (innermost last) and, when
// the cursor sits inside a string, that string's text so far and what it is an argument of.
function scan(text: string): { frames: Frame[]; inString: { text: string; frame: Frame | null; quote: string; before: string } | null } {
  const frames: Frame[] = [];
  let i = 0;
  while (i < text.length) {
    const c = text[i]!;
    if (c === "#") {
      while (i < text.length && text[i] !== "\n") i++;
      continue;
    }
    if (c === '"' || c === "'") {
      const triple = text.startsWith(c.repeat(3), i);
      const quote = triple ? c.repeat(3) : c;
      let j = i + quote.length;
      while (j < text.length && !text.startsWith(quote, j)) j += text[j] === "\\" ? 2 : 1;
      if (j >= text.length) {
        const frame = frames[frames.length - 1] ?? null;
        return { frames, inString: { text: text.slice(i + quote.length), frame, quote, before: frame ? frame.segment : text.slice(Math.max(0, i - 40), i) } };
      }
      const open = frames[frames.length - 1];
      if (open) open.segment += text.slice(i, j + quote.length);
      i = j + quote.length;
      continue;
    }
    const top = frames[frames.length - 1];
    if (c === "(" ) {
      const call = /([A-Za-z_][\w.]*)\s*$/.exec(text.slice(0, i));
      frames.push({ call: call ? call[1]!.split(".").pop()! : "", names: new Set(), segment: "" });
    } else if (c === ")" || c === "]" || c === "}") {
      if (c === ")") frames.pop();
    } else if (c === "[" || c === "{") {
      // a list or dict inside a call: its commas are not the call's
      frames.push({ call: "", names: new Set(), segment: "" });
      i++;
      continue;
    } else if (c === "," && top) {
      top.segment = "";
    } else if (top) {
      top.segment += c;
      const kw = /^\s*([A-Za-z_]\w*)\s*=(?!=)$/.exec(top.segment);
      if (kw) top.names.add(kw[1]!);
    }
    i++;
  }
  return { frames, inString: null };
}

const byName = (blocks: CatalogBlock[]) => new Map(blocks.map((b) => [b.name, b]));

function lockText(b: CatalogBlock): string {
  return b.lock?.locked ? `locked, needs ${b.lock.lesson ?? "a lesson"}` : "";
}

export function suggestionsAt(textBefore: string, catalog: Catalog): Suggestion[] {
  const { frames, inString } = scan(textBefore);
  const blocks = byName(catalog.blocks);

  if (inString) {
    // preset="tinystories-…": the string is the value of a preset argument or get_preset(...)
    const inPreset = /preset\s*=\s*$/.test(inString.before) || inString.frame?.call === "get_preset" || /\bpreset\s*=\s*$/.test(inString.before);
    if (!inPreset) return [];
    return catalog.presets
      .filter((p) => p.startsWith(inString.text))
      .map((p) => ({ label: p, insertText: p, kind: "preset" as const, detail: "preset" }));
  }

  const word = /[A-Za-z_]\w*$/.exec(textBefore)?.[0] ?? "";
  const beforeWord = textBefore.slice(0, textBefore.length - word.length);
  if (/\.\s*$/.test(beforeWord)) return []; // an attribute, not a name we know

  const top = frames[frames.length - 1];
  const known = top ? blocks.get(top.call) : undefined;
  const atArgumentStart = top && /^\s*[A-Za-z_]*\s*$/.test(top.segment + word) && !/=/.test(top.segment);
  const out: Suggestion[] = [];
  if (top && known && atArgumentStart) {
    for (const arg of known.args) {
      if (top.names.has(arg.name) || !arg.name.startsWith(word)) continue;
      const type = arg.type ? `: ${arg.type}` : "";
      out.push({
        label: arg.name,
        insertText: `${arg.name}=`,
        kind: "argument",
        detail: `${arg.required ? "required" : "optional"}${type}${arg.required || arg.default === undefined ? "" : ` = ${String(arg.default)}`}`,
        documentation: known.doc,
      });
    }
  }
  // a block's name is wanted where a value starts: after "=", "(", "," or at the line start
  const valuePosition = /(=|\(|,|\[|^|\n)\s*$/.test(beforeWord);
  if (valuePosition && word.length > 0) {
    for (const b of catalog.blocks) {
      if (!b.name.startsWith(word) || b.name === word) continue;
      out.push({ label: b.name, insertText: b.name, kind: "block", detail: [b.family, lockText(b)].filter(Boolean).join(", "), documentation: b.doc });
    }
  }
  return out;
}
