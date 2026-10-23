#!/usr/bin/env bash
# Start nanoscope with docker compose, adding the GPU worker when Docker can use an NVIDIA GPU.
#   ./up.sh            start (extra arguments go to `docker compose up`, e.g. ./up.sh -d)
#   ./up.sh --cpu      never start the GPU worker
#   ./up.sh --gpu      always start it (compose fails if Docker cannot reach a GPU)
set -euo pipefail
cd "$(dirname "$0")"

mode=auto
case "${1:-}" in
  --cpu) mode=cpu; shift ;;
  --gpu) mode=gpu; shift ;;
esac

# a GPU is usable when the driver answers and Docker has the nvidia runtime (or a CDI spec)
gpu_usable() {
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 || return 1
  local info
  info=$(docker info --format '{{range $name, $_ := .Runtimes}}{{$name}} {{end}}{{range .DiscoveredDevices}}{{.Source}} {{end}}' 2>/dev/null) || return 1
  [[ " $info" == *" nvidia "* || "$info" == *cdi* ]]
}

profiles="${COMPOSE_PROFILES:-}"
if [[ "$mode" == gpu ]] || { [[ "$mode" == auto ]] && gpu_usable; }; then
  profiles="${profiles:+$profiles,}gpu"
  echo "[nanoscope] GPU found: starting the GPU worker too (./up.sh --cpu to skip it)"
else
  echo "[nanoscope] no GPU usable from Docker: CPU worker only (see docs/deploy.md to set one up)"
fi

COMPOSE_PROFILES="$profiles" exec docker compose up "$@"
