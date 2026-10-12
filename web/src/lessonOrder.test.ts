import { describe, expect, it } from "vitest";
import curricula from "../test/fixtures/curricula.json";
import { neighbours } from "./lessonOrder";

describe("neighbours", () => {
  it("steps through a course and into the next one", () => {
    const first = curricula[0]!;
    const last = first.lessons[first.lessons.length - 1]!;
    expect(neighbours(curricula, first.lessons[0]!.id).previous).toBeNull();
    expect(neighbours(curricula, first.lessons[0]!.id).next?.id).toBe(first.lessons[1]!.id);
    expect(neighbours(curricula, last.id).next?.id).toBe(curricula[1]!.lessons[0]!.id);
  });

  it("knows nothing about an unknown lesson", () => {
    expect(neighbours(curricula, "x/01-y")).toEqual({ previous: null, next: null });
    expect(neighbours(undefined, "x/01-y")).toEqual({ previous: null, next: null });
  });
});
