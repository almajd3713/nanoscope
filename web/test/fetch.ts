import { vi } from "vitest";

export type Seen = { method: string; path: string; body: unknown };

// Replace fetch with a table of "METHOD /path" -> response. Returns what was requested.
export function mockApi(routes: Record<string, { status?: number | (() => number); body: unknown | ((sent: unknown) => unknown); problem?: boolean }>): Seen[] {
  const seen: Seen[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (request: Request) => {
      const url = new URL(request.url, "http://localhost");
      // the client encodes path parameters (a / becomes %2F); the server decodes them
      const pathname = decodeURIComponent(url.pathname);
      const text = request.method === "GET" ? "" : await request.text();
      seen.push({ method: request.method, path: pathname + url.search, body: text ? JSON.parse(text) : undefined });
      const hit = routes[`${request.method} ${pathname}`];
      if (!hit) return new Response(JSON.stringify({ title: "Not found", detail: `no mock for ${pathname}` }), { status: 404 });
      const sent = text ? JSON.parse(text) : undefined;
      const body = typeof hit.body === "function" ? (hit.body as (sent: unknown) => unknown)(sent) : hit.body;
      return new Response(JSON.stringify(body), {
        status: typeof hit.status === "function" ? hit.status() : (hit.status ?? 200),
        headers: { "content-type": hit.problem ? "application/problem+json" : "application/json" },
      });
    }),
  );
  return seen;
}
