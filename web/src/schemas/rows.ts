// A JSON Schema as a table of fields: one row per property, nested ones indented, with the type
// in words, whether it is required, and the schema's own description. Only what the schema says;
// a dash is a field it does not describe.

export type Schema = {
  $id?: string;
  title?: string;
  description?: string;
  type?: string | string[];
  const?: unknown;
  enum?: unknown[];
  oneOf?: Schema[];
  anyOf?: Schema[];
  $ref?: string;
  $defs?: Record<string, Schema>;
  properties?: Record<string, Schema>;
  required?: string[];
  items?: Schema;
  additionalProperties?: boolean | Schema;
  minimum?: number;
  maximum?: number;
};

export type FieldRow = { field: string; depth: number; type: string; required: boolean; description: string };

function resolve(node: Schema, root: Schema): Schema {
  if (node.$ref?.startsWith("#/$defs/")) return root.$defs?.[node.$ref.slice("#/$defs/".length)] ?? node;
  return node;
}

function typeOf(node: Schema, root: Schema): string {
  const n = resolve(node, root);
  if (n.const !== undefined) return `const ${JSON.stringify(n.const)}`;
  if (n.enum) return n.enum.map((v) => JSON.stringify(v)).join(" | ");
  const options = n.oneOf ?? n.anyOf;
  if (options) return options.map((o) => typeOf(o, root)).join(" | ");
  const t = Array.isArray(n.type) ? n.type.join(" | ") : (n.type ?? "any");
  if (n.type === "array" && n.items) return `array of ${typeOf(n.items, root)}`;
  return t;
}

// The object schema inside a node: itself, its array items, or a oneOf branch that has properties.
function objectOf(node: Schema, root: Schema): Schema | undefined {
  const n = resolve(node, root);
  if (n.properties) return n;
  if (n.items) return objectOf(n.items, root);
  for (const branch of n.oneOf ?? n.anyOf ?? []) {
    const found = objectOf(branch, root);
    if (found) return found;
  }
  return undefined;
}

export function fieldRows(schema: Schema): FieldRow[] {
  const out: FieldRow[] = [];
  const walk = (object: Schema, prefix: string, depth: number) => {
    for (const [name, raw] of Object.entries(object.properties ?? {})) {
      const node = resolve(raw, schema);
      const path = prefix ? `${prefix}.${name}` : name;
      const isArray = node.type === "array";
      out.push({
        field: depth > 2 ? `…${path.split(".").slice(-1)[0]}` : path,
        depth,
        type: typeOf(raw, schema),
        required: (object.required ?? []).includes(name),
        description: node.description ?? raw.description ?? "",
      });
      const inner = objectOf(raw, schema);
      if (inner) walk(inner, isArray ? `${path}[]` : path, depth + 1);
    }
  };
  walk(schema, "", 0);
  return out;
}
