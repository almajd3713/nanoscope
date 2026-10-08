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

// Subscribe to one of the server's SSE streams. Reconnects with backoff, resumes from the last
// row id, and reports "paused" (values stay; the page says so) until the stream is back.
// Cookie auth: EventSource sends the same-origin login cookie.
export function useEvents(url: string | null, options: Options): EventsStatus {
  // Keyed by stream so a new url reads as "connecting" without resetting state in an effect.
  const [state, setState] = useState<{ key: string; status: EventsStatus }>({ key: "", status: "connecting" });
  const latest = useRef(options);
  useEffect(() => {
    latest.current = options;
  });
  const eventsKey = options.events.join(",");
  const streamKey = `${url ?? ""}|${eventsKey}`;

  useEffect(() => {
    if (!url) return;
    let source: EventSource | null = null;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let delay = FIRST_DELAY_MS;
    let lastId: string | null = null;
    let stopped = false;
    const setStatus = (status: EventsStatus) => setState({ key: streamKey, status });

    const open = () => {
      const { resumeParam } = latest.current;
      const target = lastId !== null && resumeParam ? withParam(url, resumeParam, lastId) : url;
      const es = new EventSource(target, { withCredentials: true });
      source = es;
      es.onopen = () => {
        delay = FIRST_DELAY_MS;
        setStatus("open");
      };
      es.onerror = () => {
        // Take over from the browser's own retry so the next attempt can carry the resume id.
        es.close();
        if (stopped) return;
        setStatus("paused");
        timer = setTimeout(open, delay);
        delay = Math.min(delay * 2, MAX_DELAY_MS);
      };
      for (const type of eventsKey.split(",").filter(Boolean)) {
        es.addEventListener(type, (e) => {
          const m = e as MessageEvent<string>;
          if (m.lastEventId) lastId = m.lastEventId;
          let data: unknown = m.data;
          try {
            data = JSON.parse(m.data);
          } catch {
            /* keep the raw text */
          }
          latest.current.onEvent(type, data, m.lastEventId || null);
        });
      }
      es.addEventListener("reset", () => {
        lastId = null;
        latest.current.onReset?.();
      });
    };

    open();
    return () => {
      stopped = true;
      clearTimeout(timer);
      source?.close();
    };
  }, [url, eventsKey, streamKey]);

  return state.key === streamKey ? state.status : "connecting";
}
