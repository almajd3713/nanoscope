// @vitest-environment node
import stylelint from "stylelint";
import { describe, expect, it } from "vitest";

const lint = (file: string) => stylelint.lint({ files: file, configFile: "stylelint.config.js" });

describe("stylelint", () => {
  it("rejects each banned pattern in the fixture, once", async () => {
    const { results } = await lint("test/fixtures/banned.module.css");
    const rules = results.flatMap((r) => r.warnings.map((w) => w.rule)).sort();
    expect(rules).toEqual(
      [
        "at-rule-disallowed-list",
        "color-no-hex",
        "declaration-property-unit-disallowed-list",
        "declaration-property-value-disallowed-list",
        "declaration-property-value-disallowed-list",
        "declaration-property-value-disallowed-list",
        "declaration-property-value-disallowed-list",
        "function-disallowed-list",
        "function-disallowed-list",
        "function-disallowed-list",
      ].sort(),
    );
  });

  it("passes the generated tokens.css untouched", async () => {
    const { results } = await lint("src/styles/tokens.css");
    expect(results.flatMap((r) => r.warnings)).toEqual([]);
  });
});
