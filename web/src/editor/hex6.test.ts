import { describe, expect, it, vi } from "vitest";

// Importing monaco.ts loads the real editor, which needs a browser: stub it for this one function.
vi.mock("monaco-editor/editor/editor.api.js", () => ({}));
vi.mock("monaco-editor/features/register.all.js", () => ({}));
vi.mock("monaco-editor/languages/definitions/python/register.js", () => ({}));
vi.mock("monaco-editor/editor/editor.worker.js?worker", () => ({ default: class {} }));

describe("hex6", () => {
  it("gives Monaco six hex digits whatever the token is written as", async () => {
    const { colorHex, hex6 } = await import("./monaco");
    expect(hex6("#fff")).toBe("ffffff");
    expect(hex6("#0A4BC4")).toBe("0a4bc4");
    expect(hex6("rgba(10,75,196,0.16)")).toBe("0a4bc4");
    expect(colorHex("rgba(10,75,196,0.16)")).toBe("#0a4bc429");
    expect(hex6("#0a4bc429")).toBe("0a4bc4");
    expect(colorHex("#0a4bc429")).toBe("#0a4bc429");
    expect(colorHex("#fff")).toBe("#ffffff");
    expect(() => hex6("var(--x)")).toThrow(/cannot turn/);
  });
});
