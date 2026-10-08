import { render, screen } from "@testing-library/react";
import { useQuery } from "@tanstack/react-query";
import { describe, expect, it, vi } from "vitest";
import { ApiProblem, toProblem, unwrap } from "../api/problem";
import { makeQueryClient, Providers } from "../app/providers";
import { ProblemFromError, ProblemView } from "./ProblemView";

const LOCKED =
  "block:Attention is locked until you build it yourself in the lesson foundations/04-multi-head.\n  start the lesson:      nanoscope learn start foundations/04-multi-head\n  or unlock everything:  nanoscope learn unlock --all";

describe("ProblemView", () => {
  it("shows title, status and the detail word for word", () => {
    render(<ProblemView status={422} title="Locked until you build it" detail={LOCKED} />);
    expect(screen.getByText("Locked until you build it")).toBeTruthy();
    expect(screen.getByText("422")).toBeTruthy();
    expect(screen.getByText((_, el) => el?.tagName === "PRE" && el.textContent === LOCKED)).toBeTruthy();
  });

  it("turns a problem+json body into an ApiProblem and shows its detail verbatim", () => {
    const err = toProblem(
      { title: "Cannot be done as asked", status: 422, detail: "unknown preset 'x'; available: a, b" },
      { status: 422, statusText: "Unprocessable Entity" },
    );
    render(<ProblemFromError error={err} />);
    expect(screen.getByRole("alert").textContent).toContain("unknown preset 'x'; available: a, b");
  });

  it("falls back to the status text when the body is not a problem document", () => {
    const err = toProblem("upstream timed out", { status: 502, statusText: "Bad Gateway" });
    expect(err.title).toBe("Bad Gateway");
    expect(err.detail).toBe("upstream timed out");
  });

  it("unwrap throws an ApiProblem on an error response", async () => {
    const call = Promise.resolve({
      error: { title: "Not found", detail: "no run 'x'" },
      response: new Response(null, { status: 404 }),
    });
    await expect(unwrap(call)).rejects.toMatchObject({ status: 404, detail: "no run 'x'" });
  });

  it("the boundary shows a render error's message instead of a blank page", () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    function Boom(): never {
      throw new Error("kaboom");
    }
    render(
      <Providers>
        <Boom />
      </Providers>,
    );
    expect(screen.getByRole("alert").textContent).toContain("kaboom");
  });

  it("a failed query surfaces the detail verbatim and is not retried on a 4xx", async () => {
    let calls = 0;
    function Q() {
      const q = useQuery({
        queryKey: ["x"],
        queryFn: () => {
          calls += 1;
          throw new ApiProblem(404, { title: "Not found", detail: "no run 'x'" });
        },
      });
      return q.error ? <ProblemFromError error={q.error} /> : <p>loading</p>;
    }
    render(
      <Providers client={makeQueryClient()}>
        <Q />
      </Providers>,
    );
    expect((await screen.findByRole("alert")).textContent).toContain("no run 'x'");
    expect(calls).toBe(1);
  });
});
