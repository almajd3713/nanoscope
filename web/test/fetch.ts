import { vi } from "vitest";

export type Seen = { method: string; path: string; body: unknown };

// Replace fetch with a table of "METHOD /path" -> response. Returns what was requested.
export function mockApi(routes: Record<string, { status?: number; body: unknown; problem?: boolean }>): Seen[] {
  const seen: Seen[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (request: Request) => {
      const url = new URL(request.url, "http://localhost");
      const text = request.method === "GET" ? "" : await request.text();
      seen.push({ method: request.method, path: url.pathname + url.search, body: text ? JSON.parse(text) : undefined });
      const hit = routes[`${request.method} ${url.pathname}`];
      if (!hit) return new Response(JSON.stringify({ title: "Not found", detail: `no mock for ${url.pathname}` }), { status: 404 });
      return new Response(JSON.stringify(hit.body), {
        status: hit.status ?? 200,
        headers: { "content-type": hit.problem ? "application/problem+json" : "application/json" },
      });
    }),
  );
  return seen;
}
