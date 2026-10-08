import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { mockApi } from "../../test/fetch";
import { Providers } from "../app/providers";
import { Predict } from "./Predict";

const ID = "modern-block/07-assemble";
const progress = (extra: object = {}, lesson: object = {}) => ({
  schema: 1, nanoscope: "0.4.0", lessons: { [ID]: { state: "started", first_checked_at: null, ...lesson } }, ...extra,
});

function renderPredict() {
  return render(
    <Providers>
      <Predict lessonId={ID} />
    </Providers>,
  );
}

afterEach(() => vi.unstubAllGlobals());

describe("Predict", () => {
  it("records the prediction before the experiment, with only the fields you gave", async () => {
    const seen = mockApi({
      "GET /api/learn/progress": { body: progress() },
      "POST /api/curricula/modern-block/07-assemble/predict": {
        body: { lesson: ID, at: "2026-10-08T13:00:00+00:00", file: "lessons/modern-block/07-assemble/prediction.toml" },
      },
    });
    renderPredict();
    await userEvent.click(await screen.findByRole("radio", { name: "better" }));
    await userEvent.type(screen.getByLabelText("Low end of the difference"), "-0.2");
    await userEvent.type(screen.getByLabelText("High end of the difference"), "-0.05");
    await userEvent.click(screen.getByRole("button", { name: "Record prediction" }));
    await waitFor(() => expect(seen.some((r) => r.method === "POST")).toBe(true));
    expect(seen.find((r) => r.method === "POST")?.body).toEqual({ verdict: "better", low: -0.2, high: -0.05 });
  });

  it("shows the library's refusal word for word", async () => {
    mockApi({
      "GET /api/learn/progress": { body: progress() },
      "POST /api/curricula/modern-block/07-assemble/predict": {
        status: 422,
        problem: true,
        body: { title: "Cannot be done as asked", status: 422, detail: "prediction.toml: give both low and high, or neither" },
      },
    });
    renderPredict();
    await userEvent.type(await screen.findByLabelText("Low end of the difference"), "-0.2");
    await userEvent.click(screen.getByRole("button", { name: "Record prediction" }));
    expect((await screen.findByRole("alert")).textContent).toContain("give both low and high, or neither");
  });

  it("shows a recorded prediction as final, and no form", async () => {
    mockApi({ "GET /api/learn/progress": { body: progress({ predictions: { [ID]: { at: "2026-10-08T13:00:00+00:00" } } }) } });
    renderPredict();
    expect(await screen.findByText("Prediction recorded")).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Record prediction" })).toBeNull();
  });

  it("says it is too late once the lesson has been checked", async () => {
    mockApi({ "GET /api/learn/progress": { body: progress({}, { first_checked_at: "2026-10-08T12:38:01+00:00" }) } });
    renderPredict();
    expect(await screen.findByText(/too late for this lesson/)).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Record prediction" })).toBeNull();
  });
});
