import createClient from "openapi-fetch";
import type { paths } from "./schema";

// Same origin: the API serves the built app, and Vite proxies /api in development.
export const api = createClient<paths>({ baseUrl: "", credentials: "same-origin" });
