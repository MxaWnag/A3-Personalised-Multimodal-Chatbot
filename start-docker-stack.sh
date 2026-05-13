#!/usr/bin/env bash
# One-shot bootstrap after `git clone`: Chroma + Ollama (pull models) + API + Streamlit.
# Prerequisites: Docker with Compose v2 (`docker compose`).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ ! -f docker-compose.yml ]]; then
  echo "error: docker-compose.yml not found in $ROOT" >&2
  exit 1
fi

compose() {
  docker compose "$@"
}

if ! docker compose version &>/dev/null; then
  echo "error: need Docker Compose v2 (docker compose). Install Docker Desktop or compose-plugin." >&2
  exit 1
fi

CHROMA_URL="${CHROMA_URL:-http://127.0.0.1:8001}"
OLLAMA_HOST_URL="${OLLAMA_HOST_URL:-http://127.0.0.1:11435}"
API_URL="${API_URL:-http://127.0.0.1:8000}"
CHROMA_WAIT_S="${CHROMA_WAIT_S:-120}"
OLLAMA_WAIT_S="${OLLAMA_WAIT_S:-120}"
API_WAIT_S="${API_WAIT_S:-900}"
SLEEP_S="${SLEEP_S:-3}"

log() { echo "[start-docker-stack] $*"; }

wait_http() {
  local name="$1" url="$2" max_s="$3"
  local deadline=$((SECONDS + max_s))
  while (( SECONDS < deadline )); do
    if curl -sf --connect-timeout 2 --max-time 5 "$url" >/dev/null; then
      log "$name is up ($url)"
      return 0
    fi
    sleep "$SLEEP_S"
  done
  echo "error: timeout waiting for $name ($url after ${max_s}s)" >&2
  return 1
}

wait_chroma() {
  local deadline=$((SECONDS + CHROMA_WAIT_S))
  while (( SECONDS < deadline )); do
    if curl -sf --connect-timeout 2 --max-time 5 "${CHROMA_URL}/api/v2/heartbeat" >/dev/null 2>&1; then
      log "Chroma heartbeat OK ($CHROMA_URL)"
      return 0
    fi
    if curl -sf --connect-timeout 2 --max-time 5 "${CHROMA_URL}/api/v1/heartbeat" >/dev/null 2>&1; then
      log "Chroma heartbeat OK ($CHROMA_URL v1)"
      return 0
    fi
    sleep "$SLEEP_S"
  done
  echo "error: timeout waiting for Chroma at $CHROMA_URL" >&2
  return 1
}

wait_api_ready() {
  local deadline=$((SECONDS + API_WAIT_S))
  while (( SECONDS < deadline )); do
    local body=""
    if body=$(curl -sf --connect-timeout 2 --max-time 20 "${API_URL}/health" 2>/dev/null); then
      if echo "$body" | grep -Eq '"agent_ready"[[:space:]]*:[[:space:]]*true'; then
        log "API agent ready ($API_URL)"
        return 0
      fi
    fi
    sleep "$SLEEP_S"
  done
  echo "error: timeout waiting for API (listening + agent_ready) at $API_URL (${API_WAIT_S}s)." >&2
  echo "hint: first index + embedding download can be slow; raise API_WAIT_S or check: compose logs -f api" >&2
  return 1
}

log "building images (api + streamlit)"
compose build api streamlit

log "starting Chroma + Ollama"
compose up -d chroma ollama

wait_chroma
wait_http "Ollama" "${OLLAMA_HOST_URL}/api/tags" "$OLLAMA_WAIT_S"

if [[ "${SKIP_OLLAMA_PULL:-0}" != "1" ]]; then
  log "pulling Ollama models (set SKIP_OLLAMA_PULL=1 to skip)"
  compose exec -T ollama ollama pull "${OLLAMA_CHAT_MODEL:-llama3.1:8b}"
  compose exec -T ollama ollama pull "${OLLAMA_VISION_MODEL:-llava:7b}"
else
  log "SKIP_OLLAMA_PULL=1 — ensure models exist in the ollama volume"
fi

log "starting API + Streamlit (API blocks until index is built; this can take many minutes)"
compose up -d api streamlit

wait_api_ready

log "done."
echo "  API:        $API_URL"
echo "  Streamlit:  http://127.0.0.1:8501"
echo "  Chroma:     $CHROMA_URL"
echo "  Ollama:     $OLLAMA_HOST_URL"
echo "Logs: docker compose logs -f api"
