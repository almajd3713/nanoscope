# Deploying nanoscope with docker compose

`docker compose up` starts the API and one CPU worker on your machine. The API listens only on
`127.0.0.1` (port `8765`, set `NANOSCOPE_PORT` in `.env` to change it). It makes an access token
in the home volume on first start and logs the sign-in URL: `docker compose logs api`.

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
