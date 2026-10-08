#!/usr/bin/env bash
# The compose stack the end-to-end tests run against: its own project, port and workspace, so it
# never touches a stack you already run.
#   e2e/stack.sh up     build the image, start the API and a CPU worker, prepare the data
#   e2e/stack.sh reset  forget lessons, progress, runs and the queue, keeping the image and data
#   e2e/stack.sh down   stop it and delete its volumes
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export NANOSCOPE_PORT="${E2E_PORT:-18765}"
export NANOSCOPE_TOKEN="${E2E_TOKEN:-e2e-token}"
export NANOSCOPE_WORKSPACE="${E2E_WORKSPACE:-/tmp/nanoscope-e2e-workspace}"
PROJECT="nanoscope-e2e"
compose() { docker compose -p "$PROJECT" -f "$ROOT/compose.yaml" "$@"; }

case "${1:-}" in
  up)
    mkdir -p "$NANOSCOPE_WORKSPACE"
    chmod 777 "$NANOSCOPE_WORKSPACE"  # the containers write as uid 1000, whoever runs this
    compose build api
    compose up -d --pull never api worker
    for _ in $(seq 1 90); do
      curl -sf "http://127.0.0.1:$NANOSCOPE_PORT/api/health" >/dev/null && break
      sleep 1
    done
    curl -sf "http://127.0.0.1:$NANOSCOPE_PORT/api/health" >/dev/null || { compose logs --tail 40; exit 1; }
    # the lessons train on TinyStories: fetch the tokens once, outside the timed tests
    compose exec -T worker nanoscope prepare-data tinystories-5min
    ;;
  reset)
    compose exec -T worker sh -c 'rm -rf /nanoscope/learn /nanoscope/runs /nanoscope/queue.db* /nanoscope/jobs /nanoscope/workers /nanoscope/workspace/lessons'
    compose restart api worker
    for _ in $(seq 1 60); do
      curl -sf "http://127.0.0.1:$NANOSCOPE_PORT/api/health" >/dev/null && break
      sleep 1
    done
    ;;
  down)
    compose down -v --remove-orphans
    rm -rf "$NANOSCOPE_WORKSPACE"
    ;;
  *)
    echo "usage: $0 up|down" >&2
    exit 2
    ;;
esac
