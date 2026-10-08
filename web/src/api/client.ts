import createClient from "openapi-fetch";
import type { paths } from "./schema";

// Same origin: the API serves the built app, and Vite proxies /api in development.
export const api = createClient<paths>({
  // Absolute, because Request needs a full URL (this is the page's own origin).
  baseUrl: window.location.origin,
  credentials: "same-origin",
  // Looked up per call, so tests can replace fetch.
  fetch: (request) => globalThis.fetch(request),
});
