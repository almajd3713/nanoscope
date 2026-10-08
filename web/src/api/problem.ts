// RFC 9457 problem+json as the server sends it (nanoscope/server/errors.py). The detail is the
// library's own message and is shown word for word.
export type ProblemDoc = {
  type?: string;
  title?: string;
  status?: number;
  detail?: string;
  instance?: string;
  // a locked block's 422 names the lesson that unlocks it
  lesson?: string;
  [extra: string]: unknown;
};

export class ApiProblem extends Error {
  readonly status: number;
  readonly title: string;
  readonly detail: string;
  readonly doc: ProblemDoc;

  constructor(status: number, doc: ProblemDoc) {
    const detail = doc.detail ?? "";
    super(detail || doc.title || `HTTP ${status}`);
    this.name = "ApiProblem";
    this.status = status;
    this.title = doc.title ?? `HTTP ${status}`;
    this.detail = detail;
    this.doc = doc;
  }
}

function isDoc(x: unknown): x is ProblemDoc {
  return typeof x === "object" && x !== null && !Array.isArray(x);
}

// The error body from openapi-fetch (parsed JSON, or text when it was not JSON) -> ApiProblem.
export function toProblem(error: unknown, response: Pick<Response, "status" | "statusText">): ApiProblem {
  if (isDoc(error)) return new ApiProblem(response.status, error);
  const text = typeof error === "string" ? error : "";
  return new ApiProblem(response.status, { title: response.statusText || `HTTP ${response.status}`, detail: text });
}

type Result<T> = { data?: T; error?: unknown; response: Response };

// Await an openapi-fetch call and throw an ApiProblem on an error response, so TanStack Query sees it.
export async function unwrap<T>(call: Promise<Result<T>>): Promise<T> {
  const { data, error, response } = await call;
  if (error !== undefined || !response.ok) throw toProblem(error, response);
  return data as T;
}
