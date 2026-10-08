import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useEvents } from "./useEvents";

class MockEventSource {
  static all: MockEventSource[] = [];
  onopen: (() => void) | null = null;
  onerror: (() => void) | null = null;
  closed = false;
  listeners = new Map<string, ((e: MessageEvent<string>) => void)[]>();
  constructor(
    public url: string,
    public init?: EventSourceInit,
  ) {
    MockEventSource.all.push(this);
  }
  addEventListener(type: string, fn: (e: MessageEvent<string>) => void) {
    this.listeners.set(type, [...(this.listeners.get(type) ?? []), fn]);
  }
  close() {
    this.closed = true;
  }
  open() {
    this.onopen?.();
  }
  fail() {
    this.onerror?.();
  }
  emit(type: string, data: unknown, id = "") {
    const e = new MessageEvent(type, { data: JSON.stringify(data), lastEventId: id });
    for (const fn of this.listeners.get(type) ?? []) fn(e);
  }
}

const last = () => MockEventSource.all[MockEventSource.all.length - 1]!;

beforeEach(() => {
  MockEventSource.all = [];
  vi.stubGlobal("EventSource", MockEventSource);
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

const OPTS = { events: ["step", "state"], resumeParam: "since_step" };

describe("useEvents", () => {
  it("opens with credentials, reports open, and parses events", () => {
    const seen: unknown[] = [];
    const { result } = renderHook(() =>
      useEvents("/api/runs/a/events", { ...OPTS, onEvent: (t, d, id) => seen.push([t, d, id]) }),
    );
    expect(result.current).toBe("connecting");
    expect(last().init?.withCredentials).toBe(true);
    act(() => last().open());
    expect(result.current).toBe("open");
    act(() => last().emit("step", { step: 7, loss: 2.5 }, "7"));
    expect(seen).toEqual([["step", { step: 7, loss: 2.5 }, "7"]]);
  });

  it("pauses on error, reconnects with backoff, and resumes from the last id", () => {
    const { result } = renderHook(() => useEvents("/api/runs/a/events", { ...OPTS, onEvent: () => {} }));
    act(() => last().open());
    act(() => last().emit("step", { step: 40 }, "40"));
    const first = last();
    act(() => first.fail());
    expect(first.closed).toBe(true);
    expect(result.current).toBe("paused");
    expect(MockEventSource.all).toHaveLength(1);
    act(() => void vi.advanceTimersByTime(1000));
    expect(MockEventSource.all).toHaveLength(2);
    expect(last().url).toBe("/api/runs/a/events?since_step=40");
    // a second failure waits twice as long
    act(() => last().fail());
    act(() => void vi.advanceTimersByTime(1999));
    expect(MockEventSource.all).toHaveLength(2);
    act(() => void vi.advanceTimersByTime(1));
    expect(MockEventSource.all).toHaveLength(3);
    act(() => last().open());
    expect(result.current).toBe("open");
  });

  it("on reset calls onReset and reconnects from the start", () => {
    const onReset = vi.fn();
    renderHook(() => useEvents("/api/runs/a/events", { ...OPTS, onEvent: () => {}, onReset }));
    act(() => last().open());
    act(() => last().emit("step", { step: 40 }, "40"));
    act(() => last().emit("reset", {}));
    expect(onReset).toHaveBeenCalledTimes(1);
    act(() => last().fail());
    act(() => void vi.advanceTimersByTime(1000));
    expect(last().url).toBe("/api/runs/a/events");
  });

  it("closes the stream on unmount and does not reconnect", () => {
    const { unmount } = renderHook(() => useEvents("/api/runs/a/events", { ...OPTS, onEvent: () => {} }));
    const es = last();
    unmount();
    expect(es.closed).toBe(true);
    act(() => void vi.advanceTimersByTime(60000));
    expect(MockEventSource.all).toHaveLength(1);
  });

  it("opens nothing for a null url", () => {
    renderHook(() => useEvents(null, { ...OPTS, onEvent: () => {} }));
    expect(MockEventSource.all).toHaveLength(0);
  });
});
