# Deploying nanoscope with docker compose

`docker compose up` starts the API and one CPU worker on your machine. The API listens only on
`127.0.0.1` (port `8765`, set `NANOSCOPE_PORT` in `.env` to change it). It makes an access token
in the home volume on first start and logs the sign-in URL: `docker compose logs api`.

## Services and profiles

| Service | Starts with | What it is |
|---|---|---|
| `api` | `docker compose up` | the HTTP API and the GUI, on `127.0.0.1:8765` |
| `worker` | `docker compose up` | one CPU worker: runs every job (training, checks, generation) |
| `notebook` | `--profile notebook` | marimo editing `workspace/notebooks`, on `127.0.0.1:8766` (`NANOSCOPE_NOTEBOOK_PORT`) |
| `worker-gpu` | `--profile gpu` | a CUDA worker, see below |

The API never runs your code; workers and the notebook do. All of them run as uid 1000 with no
capabilities, a read-only root filesystem, and limits from `.env` (`NANOSCOPE_MEM_LIMIT`,
`NANOSCOPE_CPUS`, `NANOSCOPE_PIDS_LIMIT`). None of them can reach the Docker socket. Set
`NANOSCOPE_JOBS_OFFLINE=1` to make jobs that need the network (Hub sync, `push_to_hub`, W&B) fail
at once instead of reaching out.

`docker compose down` and `up` again keep everything: runs, the queue and the token live in
volumes. A run that was training when you stopped carries on from its last checkpoint once the
worker is back.

## Signing in from another machine

Published ports are bound to `127.0.0.1`, so nothing else on your network can connect. The login
URL in `docker compose logs api` is for this machine. The token is a remote login (see
[server.md](server.md#the-token-and-the-login-url)) and the API speaks plain HTTP, so reach a
remote box one of these ways:

- **SSH tunnel** (simplest): `ssh -L 8765:127.0.0.1:8765 gpu-box`, then open
  `http://127.0.0.1:8765` locally and sign in with the token from `docker compose logs api` there.
- **Tailscale**: put both machines on one tailnet and tunnel or proxy over it. Traffic between
  them is encrypted, but keep the published port on `127.0.0.1` and use `tailscale serve` (or an
  SSH tunnel over the tailnet) rather than publishing the port on `0.0.0.0`.
- **An HTTPS reverse proxy** on the host in front of `127.0.0.1:8765`.

Do not publish the port on `0.0.0.0` on a shared network.

## Using a GPU

```bash
docker compose --profile gpu up
```

This adds `worker-gpu`, which reserves every NVIDIA GPU (`count: all`). It needs the NVIDIA
container toolkit on the host. Set `NANOSCOPE_SLOTS` for the jobs that share a GPU at once.

For one worker per GPU, copy the `worker-gpu` service once per card and give each its own
device: `NVIDIA_VISIBLE_DEVICES=0` for the first, `1` for the second, and so on (and replace
`count: all` with `device_ids: ["0"]`). Every worker takes jobs from the same queue.

## Files owned by the wrong user

The containers run as uid 1000. A bind-mounted workspace (`NANOSCOPE_WORKSPACE`, default
`./workspace`) must be writable by that uid: on Linux, `chown -R 1000:1000 workspace`.

## Running a build of your own

The images come from `ghcr.io/almajd3713/nanoscope`. To build from a clone instead (a branch, or
changes the published image does not have yet):

```bash
docker compose build api && docker compose up --pull never
```

## Backing up

Everything you would miss is in the `nanoscope-home` volume (mounted at `/nanoscope`): `runs/`,
`studies/`, `learn/` (your progress), `server/token`, and `queue.db`, the job queue (SQLite, in WAL
mode, so it is several files: `queue.db`, `queue.db-wal`, `queue.db-shm`). Your models and studies
are in the workspace folder on the host, which is already yours to back up (it is a git repository
once you use preregistration commits).

Stop the stack first so the queue is not written while you copy it, then archive the volume:

```bash
docker compose down
docker run --rm -v nanoscope_nanoscope-home:/v -v "$PWD":/b alpine \
    tar czf /b/nanoscope-home.tgz -C /v .
docker compose up -d
```

(`nanoscope_` is the compose project name; `docker volume ls` shows the exact name.) To restore,
create the volume and extract the archive into it the same way. `nanoscope-data` (token caches)
and `hf-cache` can be rebuilt from the Hub and need no backup.

## WSL2 and Docker Desktop

On Windows, run `docker compose` from inside a WSL2 distribution with Docker Desktop's WSL
integration on (or Docker Engine installed in the distribution). Keep the clone, and so the
workspace folder, on the Linux filesystem (`~/nanoscope`), not under `/mnt/c`: bind mounts there
are slow, and file watching (the GUI's live updates) can miss changes. Give Docker enough memory
(`.wslconfig`, or Docker Desktop's resource settings) for `NANOSCOPE_MEM_LIMIT`. GPUs work through
the NVIDIA container toolkit in WSL2 with a recent Windows driver. On macOS, Docker Desktop is CPU
only.

## Apple silicon and AMD (native `serve`)

The images are CPU (and NVIDIA CUDA). Docker on macOS cannot reach Metal (MPS), and the compose
file has no ROCm image. For those GPUs run the server natively, in a Python environment that has
the right torch build:

```bash
pip install "nanoscope-lab[server]"
nanoscope serve --worker cpu      # or the torch device string of your GPU, e.g. cuda:0 on ROCm
```

This is the same API and GUI, with one worker process on that device, and it needs no Docker.
It runs on loopback by default, with no token. The device is passed to torch as given, but
nanoscope is only tested on CPU and NVIDIA CUDA: MPS and ROCm are untested here, so tell us what
you find.

## Kaggle and Colab

Free GPU sessions are a good place to run a study, and the Hub is the way back. In
[`notebooks/kaggle.ipynb`](../notebooks/kaggle.ipynb) set `RUNS_REPO` to a private Hub repo and
the study mirrors every run there as it trains (see [research.md](research.md#kaggle)). To see
those runs in your own stack, pull one with the API:

```bash
curl -H "Authorization: Bearer $TOKEN" -H 'content-type: application/json' \
     -d '{"repo": "you/nanoscope-runs", "ref": "tinystories-5min/bigram/seed-0"}' \
     http://127.0.0.1:8765/api/sync/hub
```

It is a job (it downloads checkpoints), so a worker needs `HF_TOKEN` in `.env` and
`NANOSCOPE_JOBS_OFFLINE` must not be set.
