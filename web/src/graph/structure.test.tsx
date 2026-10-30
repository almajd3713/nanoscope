import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { axe } from "vitest-axe";
import { describe, expect, it, vi } from "vitest";
import lesson04 from "../../test/fixtures/lesson-04.json";
import starter from "../../test/fixtures/structure-04-starter.json";
import solution from "../../test/fixtures/structure-06-solution.json";
import { builtAs, hintsFromChecks, type StructureReport, structureTree } from "./structure";
import { StructureView } from "./StructureView";

const gpt = solution as StructureReport;

describe("structureTree", () => {
  it("nests the rows and folds repeated layers into the first", () => {
    const root = structureTree(gpt.modules)!;
    expect(root.children.map((c) => c.name)).toEqual(["tok_emb", "pos_emb", "blocks", "ln_f", "head"]);
    const blocks = root.children[2]!;
    expect(blocks.children).toHaveLength(1);
    expect(blocks.children[0]).toMatchObject({ path: "blocks.0", type: "MyBlock", repeat: 4 });
    expect(blocks.children[0]!.children.map((c) => c.name)).toEqual(["ln1", "attn", "ln2", "fc", "proj"]);
  });

  it("keeps a list's items apart when they differ", () => {
    const rows = [
      { path: "", type: "M", block: null, depth: 0, params: 0, input_shapes: null, output_shapes: null, line: 1 },
      { path: "layers", type: "ModuleList", block: null, depth: 1, params: 0, input_shapes: null, output_shapes: null, line: 2 },
      { path: "layers.0", type: "Linear", block: null, depth: 2, params: 20, input_shapes: null, output_shapes: null, line: 2 },
      { path: "layers.1", type: "Linear", block: null, depth: 2, params: 30, input_shapes: null, output_shapes: null, line: 2 },
    ];
    expect(structureTree(rows)!.children[0]!.children.map((c) => [c.path, c.repeat])).toEqual([
      ["layers.0", 1],
      ["layers.1", 1],
    ]);
  });
});

describe("hintsFromChecks", () => {
  it("takes the first inputs of the lesson's equivalent check for that class", () => {
    expect(hintsFromChecks(lesson04.checks, "MultiHead")).toEqual({ inputs: lesson04.checks[1]!.inputs });
    expect(hintsFromChecks(lesson04.checks, "Other")).toEqual({});
    expect(hintsFromChecks([{ kind: "defines", class: "A", args: { d_model: 8 } }], "A")).toEqual({ args: { d_model: 8 } });
  });
});

describe("StructureView", () => {
  it("says what the code built and links each module to its line", async () => {
    const edit = vi.fn();
    const { container } = render(<StructureView structure={{ state: "done", report: gpt, error: null }} className="MyGPT2" onEditInCode={edit} />);
    expect(screen.getByText(builtAs(gpt))).toBeTruthy();
    const tree = screen.getByRole("list", { name: "Modules of MyGPT2" });
    const fc = within(tree).getByText("fc").closest("li")!;
    await userEvent.click(within(fc).getByRole("button", { name: /^line \d+$/ }));
    expect(edit).toHaveBeenCalledWith(gpt.modules.find((m) => m.path === "blocks.0.fc")!.line);
    const results = await axe(container, { rules: { "color-contrast": { enabled: false } } });
    expect(results.violations.map((v) => v.id)).toEqual([]);
  });

  it("an unfinished class: what stopped it and where, and that it has no layers yet", async () => {
    const report = starter as StructureReport;
    const { container } = render(<StructureView structure={{ state: "done", report, error: null }} className="MultiHead" />);
    expect(screen.getByText("MultiHead has no layers yet.", { exact: false })).toBeTruthy();
    // without the editor on the page the line is text, not a link
    expect(screen.queryByRole("button", { name: `line ${report.error!.line}` })).toBeNull();
    expect(screen.getByText(`line ${report.error!.line}`)).toBeTruthy();
    const results = await axe(container, { rules: { "color-contrast": { enabled: false } } });
    expect(results.violations.map((v) => v.id)).toEqual([]);
  });

  it("names the class it is waiting for", () => {
    render(<StructureView structure={{ state: "waiting", report: null, error: null }} className="OneHead" />);
    expect(screen.getByText("Waiting for a worker to trace OneHead…")).toBeTruthy();
  });
});
