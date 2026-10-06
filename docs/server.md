# The HTTP API

`nanoscope serve` puts the library behind HTTP, for the GUI and for anything else that wants
to read runs or start them. It adds nothing the library cannot do: every endpoint is a thin
wrapper, and the CLI equivalent of each is listed below. The full contract is
[`openapi.json`](openapi.json) (regenerate it with `make openapi`); a running server shows it
at `/api/docs`.

## Running it

```bash
pip install "nanoscope-lab[server]"
nanoscope serve --worker cpu          # http://127.0.0.1:8000/api, and a local worker
nanoscope serve --host 0.0.0.0        # reachable from other machines: needs the token
```

- The server itself never imports or runs your code. Anything that does (training, a study,
  `describe`, a lesson check, `generate`, `bench`, data preparation) is a **job** that a worker
  runs. Start a worker with `--worker DEVICE` (`cpu`, `cuda:0`) or separately with
  `nanoscope worker`. Without one, jobs stay `queued`; `GET /api/workers` shows who is there.
- Your files live in the workspace (`$NANOSCOPE_WORKSPACE`, default under `$NANOSCOPE_HOME`).
  Paths are confined to it; saves are atomic and need `If-Match` with the file's ETag.
- Runs started from the CLI or a notebook appear in the API and its event streams, because the
  server reads the same files they write.

## The token and the login URL

On `127.0.0.1` (the default) there is no login: only this machine can connect.

Beyond loopback every request needs a token. `serve` creates it in `$NANOSCOPE_HOME/server/token`
(mode 0600) and prints a login URL, `http://HOST:PORT/login?token=...`, which sets an HttpOnly
cookie. Scripts send `Authorization: Bearer <token>`.

**The token is a remote login.** Whoever holds it can save Python files in your workspace and
queue jobs that run them as your user. **The server speaks plain HTTP**, so on a shared network
the token and the cookie can be read by others. Beyond loopback, use only a network you trust,
an SSH tunnel (`ssh -L 8000:127.0.0.1:8000 gpu-box`, then serve on loopback there), or an HTTPS
reverse proxy. Hosted use by people you do not trust is a non-goal until there is a sandbox.

## Endpoints

Everything is under `/api`. Errors are RFC 9457 `application/problem+json`; the `detail` is the
library's own message word for word, and a block you have not unlocked is a 422 with
`type: "locked"` and the `lesson` that unlocks it.

| Resource | Endpoints | CLI equivalent |
|---|---|---|
| Presets | `GET /presets`, `/presets/{name}` | `nanoscope presets` |
| Models and blocks | `GET /models`, `/models/{ref}`, `/blocks`; `POST /models/{ref}/describe` (job) | `nanoscope describe`, `nanoscope blocks` |
| Files | `GET/PUT /files/{path}`, `GET /files`, `POST /files/{path}/lint` | your editor |
| Graph | `POST /files/{path}/graph`, `POST /files/{path}/graph/patch` | `nanoscope graph` |
| Validation | `POST /validate/run`, `/validate/study` | the checks `run()` does before it starts |
| Runs | `GET/POST /runs`, `GET /runs/{ref}` (+ `/metrics`, `/samples`, `/blockstats`, `/checkpoints`), `POST /runs/{ref}/stop`, `/resume`, `/generate` | `nanoscope run`, `status`, `stop` |
| Compare | `POST /compare` | `nanoscope compare` |
| Studies | `GET/POST /studies`, `GET /studies/{name}/report`, `POST /studies/{name}/run`, `/stop` | `nanoscope study`, `report`, `stop` |
| Learn | `GET /curricula`, `/curricula/{path}/{lesson}`; `POST .../start`, `.../check` (job); `GET /learn/progress`, `/learn/unlocks`; `POST /learn/unlock`, `/learn/policy` | `nanoscope learn ...` |
| Hardware and data | `GET /hardware`, `/hardware/bench`, `/data`; `POST /bench`, `/data/{preset}/prepare` (jobs) | `nanoscope bench`, `prepare-data` |
| Jobs and workers | `GET /jobs`, `/jobs/{id}`, `/workers`; `POST /jobs/{id}/cancel` | `nanoscope jobs` |
| Hub | `POST /sync/hub` (job) | `nanoscope run ... push_to_hub=` (the other end) |
| Schemas | `GET /schemas`, `/schemas/{name}`, `/version`, `/health` | `nanoscope/schemas/` |

A `POST` body has the defaults of the library function behind it, so `{"model": "bigram"}`
to `POST /runs` is `run(Bigram)`. A run that is already done is returned (200) instead of
queued again; otherwise the reply is the job (202) and the `ref` where the run will land.

## Server-sent events

Four streams (`text/event-stream`; they never end, so close them when done):

- `GET /runs/{ref}/events[?since_step=N]`: one run. `/events[?prefix=]`: every run under a
  prefix, each event carrying its `ref`; runs that appear later are followed from their first row.
- Event types: `state` (from `status.json`), `step` (at most four a second), `eval`, `sample`,
  `checkpoint`, `blockstats`, and `reset` (the metrics file was rewritten by a resume: refetch).
  Rows carry `id: <step>`, so a client can reconnect with `since_step`.
- `GET /files/events`: `change` `{path, kind, etag}` when a workspace file is added, edited or
  removed.
- `GET /learn/events`: `unlocks` or `progress` with the new document.
