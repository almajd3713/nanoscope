import { describe, expect, it } from "vitest";
import authorCheck from "../../test/fixtures/schema-author-check.json";
import config from "../../test/fixtures/schema-config.json";
import inspect from "../../test/fixtures/schema-inspect.json";
import { fieldRows, type Schema } from "./rows";

describe("fieldRows", () => {
  const rows = fieldRows(inspect as Schema);
  const row = (field: string) => rows.find((r) => r.field === field)!;

  it("lists top-level fields with their types, requiredness and the schema's own words", () => {
    expect(row("schema")).toMatchObject({ type: "const 1", required: true, depth: 0 });
    expect(row("steps")).toMatchObject({ type: "array of integer", description: "Every step the run has a checkpoint for (kept and archived), for a scrubber." });
    expect(row("notes").required).toBe(inspect.required.includes("notes"));
  });

  it("nests the properties of array items and objects", () => {
    expect(row("tokens[].text")).toMatchObject({ type: "string", depth: 1 });
    expect(row("attention[].weights").type).toBe("array of array of array of number");
    expect(row("lens.layers[].name").depth).toBe(2);
  });

  it("goes through a oneOf branch (a null or an object)", () => {
    const next = rows.find((r) => r.field.endsWith("next"))!;
    expect(next.type).toBe("null | object");
    expect(rows.some((r) => r.field.endsWith("rank"))).toBe(true);
  });

  it("tells an optional field from a required one", () => {
    const rows3 = fieldRows(config as Schema);
    expect(rows3.find((r) => r.field === "seed")?.required).toBe(true);
    expect(rows3.find((r) => r.field === "study")?.required).toBe(false);
    expect(rows3.find((r) => r.field === "model.rebuildable")?.required).toBe(false);
  });

  it("follows $defs references", () => {
    const rows2 = fieldRows(authorCheck as Schema);
    expect(rows2.find((r) => r.field === "checks[].starter")?.type).toBe("object");
    expect(rows2.some((r) => r.field.endsWith("passed"))).toBe(true);
  });
});
