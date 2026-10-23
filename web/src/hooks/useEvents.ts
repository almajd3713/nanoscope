import { useEffect, useRef, useState } from "react";

export type EventsStatus = "connecting" | "open" | "paused";

type Options = {
  // SSE event names to listen for (docs/server.md lists them per stream).
  events: string[];
  onEvent: (type: string, data: unknown, id: string | null) => void;
  // The `reset` event: the metrics file was rewritten by a resume, so refetch.
  onReset?: () => void;
  // Query parameter that carries the last seen row id on reconnect (runs use `since_step`).
  // The server resumes from that parameter; it does not read Last-Event-ID.
  resumeParam?: string;
};

const FIRST_DELAY_MS = 1000;
const MAX_DELAY_MS = 15000;

function withParam(url: string, name: string, value: string): string {
  const u = new URL(url, window.location.origin);
  u.searchParams.set(name, value);
  return u.pathname + u.search;
}

type Subscriber = {
  events: Set<string>;
  onEvent: (type: string, data: unknown, id: string | null) => void;
  onReset: () => void;
  onStatus: (status: EventsStatus) => void;
};

// One EventSource for every subscriber of a url. A browser keeps at most six HTTP/1.1
// connections to one host and an open stream holds one, so a page whose panels each opened
// their own stream left no connection for its requests, which then waited forever.
// Reconnects with backoff, resumes from the last row id, and reports "paused" until it is back.
class Stream {
  private source: EventSource | null = null;
  private timer: ReturnType<typeof setTimeout> | undefined;
  private delay = FIRST_DELAY_MS;
  private lastId: string | null = null;
  private listening = new Set<string>();
  private stopped = false;
  status: EventsStatus = "connecting";
  readonly subscribers = new Set<Subscriber>();

  constructor(
    private url: string,
    private resumeParam?: string,
  ) {}

  add(sub: Subscriber): void {
    this.subscribers.add(sub);
    if (this.source) for (const type of sub.events) this.listen(type);
    else this.open();
  }

  remove(sub: Subscriber): boolean {
    this.subscribers.delete(sub);
    if (this.subscribers.size > 0) return false;
    this.stopped = true;
    clearTimeout(this.timer);
    this.source?.close();
    return true;
  }

  private setStatus(status: EventsStatus): void {
    this.status = status;
    for (const sub of this.subscribers) sub.onStatus(status);
  }

  private listen(type: string): void {
    if (this.listening.has(type) || !this.source) return;
    this.listening.add(type);
    this.source.addEventListener(type, (e) => {
      const m = e as MessageEvent<string>;
      if (m.lastEventId) this.lastId = m.lastEventId;
      let data: unknown = m.data;
      try {
        data = JSON.parse(m.data);
      } catch {
        /* keep the raw text */
      }
      for (const sub of this.subscribers) if (sub.events.has(type)) sub.onEvent(type, data, m.lastEventId || null);
    });
  }

  private open(): void {
    const target = this.lastId !== null && this.resumeParam ? withParam(this.url, this.resumeParam, this.lastId) : this.url;
    const es = new EventSource(target, { withCredentials: true });
    this.source = es;
    this.listening = new Set();
    es.onopen = () => {
      this.delay = FIRST_DELAY_MS;
      this.setStatus("open");
    };
    es.onerror = () => {
      // Take over from the browser's own retry so the next attempt can carry the resume id.
      es.close();
      if (this.stopped) return;
      this.setStatus("paused");
      this.timer = setTimeout(() => this.open(), this.delay);
      this.delay = Math.min(this.delay * 2, MAX_DELAY_MS);
    };
    for (const sub of this.subscribers) for (const type of sub.events) this.listen(type);
    es.addEventListener("reset", () => {
      this.lastId = null;
      for (const sub of this.subscribers) sub.onReset();
    });
  }
}

// Streams that resume from a row id carry that id in their url, so each subscriber keeps its own.
const shared = new Map<string, Stream>();

// Subscribe to one of the server's SSE streams. Cookie auth: EventSource sends the same-origin
// login cookie.
export function useEvents(url: string | null, options: Options): EventsStatus {
  // Keyed by stream so a new url reads as "connecting" without resetting state in an effect.
  const [state, setState] = useState<{ key: string; status: EventsStatus }>({ key: "", status: "connecting" });
  const latest = useRef(options);
  useEffect(() => {
    latest.current = options;
  });
  const eventsKey = options.events.join(",");
  const streamKey = `${url ?? ""}|${eventsKey}`;
  const resumeParam = options.resumeParam;

  useEffect(() => {
    if (!url) return;
    const sub: Subscriber = {
      events: new Set(eventsKey.split(",").filter(Boolean)),
      onEvent: (type, data, id) => latest.current.onEvent(type, data, id),
      onReset: () => latest.current.onReset?.(),
      onStatus: (status) => setState({ key: streamKey, status }),
    };
    let stream = resumeParam ? undefined : shared.get(url);
    if (!stream) {
      stream = new Stream(url, resumeParam);
      if (!resumeParam) shared.set(url, stream);
    } else if (stream.status !== "connecting") {
      sub.onStatus(stream.status);
    }
    stream.add(sub);
    const mine = stream;
    return () => {
      if (mine.remove(sub) && shared.get(url) === mine) shared.delete(url);
    };
  }, [url, eventsKey, streamKey, resumeParam]);

  return state.key === streamKey ? state.status : "connecting";
}
