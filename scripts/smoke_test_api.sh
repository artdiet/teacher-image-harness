#!/usr/bin/env bash
# Exercise every image-server endpoint with curl, the way a client on the network
# would. Run it from another computer (or from the Mac against its LAN address, not
# localhost, so the Host check and the firewall are part of the test).
#
#   IMAGE_SERVER_TEST_TOKEN=<device token> scripts/smoke_test_api.sh http://<mac-name>.local:7865
#
# Asks for the token if the variable is unset. Makes three real pictures (under a
# minute on a warm server), deletes the references it creates, and unloads the
# model at the end. Needs bash, curl and python3. Exit code 1 if anything failed.
set -uo pipefail

BASE=${1:?usage: $0 http://<server>:7865}
BASE=${BASE%/}
TOKEN=${IMAGE_SERVER_TEST_TOKEN:-}
if [[ -z "$TOKEN" ]]; then
  read -rsp "device token: " TOKEN
  echo
fi
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
pass=0
fail=0
H=(-H "Authorization: Bearer $TOKEN")
JSON=(-H "content-type: application/json")

check() { # name expected actual
  if [[ "$2" == "$3" ]]; then
    pass=$((pass + 1)); printf "  ok    %-52s %s\n" "$1" "$3"
  else
    fail=$((fail + 1)); printf "  FAIL  %-52s expected %s, got %s\n" "$1" "$2" "$3"
    head -c 300 "$OUT/body" 2>/dev/null | sed 's/^/        /'; echo
  fi
}
code() { curl -s -o "$OUT/body" -w '%{http_code}' --max-time 120 "$@"; }
field() { python3 -c "import json; d = json.load(open('$OUT/body')); print($1)" 2>/dev/null || echo "?"; }
wait_job() { # job_id -> final status
  local status
  [[ "$1" =~ ^job_[0-9a-f]{16}$ ]] || { echo "no-job"; return; }  # the POST failed
  for _ in $(seq 1 180); do
    code "${H[@]}" "$BASE/jobs/$1" >/dev/null
    status=$(field 'd["status"]')
    case "$status" in done | failed | cancelled) echo "$status"; return ;; esac
    sleep 1
  done
  echo timeout
}
# macOS bash 3.2 mangles JSON with commas inside "$( ... )": build bodies first.
body() { printf "$@"; }

echo "service"
check "GET /health (no token)" 200 "$(code --max-time 8 "$BASE/health")"
if [[ "$(cat "$OUT/body" 2>/dev/null)" == "" ]]; then
  echo "  the server didn't answer; on the Mac run scripts/check_lan.sh"; exit 1
fi
check "GET /health, unknown Host header" 400 "$(code -H 'Host: evil.example' "$BASE/health")"
check "GET /models, no token" 401 "$(code "$BASE/models")"
check "GET /models, wrong token" 401 "$(code -H 'Authorization: Bearer wrong' "$BASE/models")"
check "GET /models" 200 "$(code "${H[@]}" "$BASE/models")"
if ((fail)); then
  echo "  the token wasn't accepted, so the remaining checks would all fail too"
  echo "  (other computers need their own token: scripts/client_tokens.sh add <name> on the Mac)"
  exit 1
fi
echo "        models: $(field '", ".join(m["id"] for m in d["models"])')"

echo "jobs"
B=$(body '{"prompt":"a friendly owl holding a pencil","style":"icon","width":768,"height":768,"seed":11}')
check "POST /jobs (icon)" 202 "$(code "${H[@]}" "${JSON[@]}" -d "$B" "$BASE/jobs")"
JOB=$(field 'd["job_id"]')
start=$(date +%s)
check "  job finishes" done "$(wait_job "$JOB")"
echo "        $(($(date +%s) - start)) s, model $(field 'd["result"]["model"]')"
IMG=$(field 'd["result"]["image_id"]')
check "POST /jobs, unknown field" 422 "$(code "${H[@]}" "${JSON[@]}" -d '{"prompt":"x","colour":1}' "$BASE/jobs")"
check "POST /jobs, unknown model" 400 "$(code "${H[@]}" "${JSON[@]}" -d '{"prompt":"x","model":"nope"}' "$BASE/jobs")"
check "GET /jobs" 200 "$(code "${H[@]}" "$BASE/jobs?limit=5")"
check "GET /jobs/<unknown>" 404 "$(code "${H[@]}" "$BASE/jobs/job_0000000000000000")"

echo "images"
check "GET /images" 200 "$(code "${H[@]}" "$BASE/images?limit=3")"
check "GET /images/{id}?size=full" 200 "$(code "${H[@]}" "$BASE/images/$IMG?size=full")"
cp "$OUT/body" "$OUT/full.png"
check "GET /images/{id}?size=preview" 200 "$(code "${H[@]}" "$BASE/images/$IMG?size=preview")"
check "GET /images/{id}?size=preview&budget=12000" 200 "$(code "${H[@]}" "$BASE/images/$IMG?size=preview&budget=12000")"
check "  preview fits 12000 bytes" yes "$([[ $(wc -c <"$OUT/body") -le 12000 ]] && echo yes || echo no)"
check "GET /images/{id}/meta" 200 "$(code "${H[@]}" "$BASE/images/$IMG/meta")"
check "GET /images/<path escape>" 404 "$(code "${H[@]}" "$BASE/images/..%2F..%2Fetc%2Fpasswd")"
check "POST /images (upload)" 201 "$(code "${H[@]}" -F "file=@$OUT/full.png;type=image/png" "$BASE/images")"
UP=$(field 'd["image_id"]')
printf '%%PDF-1.7 not a picture' >"$OUT/fake.png"
check "POST /images, not a picture" 400 "$(code "${H[@]}" -F "file=@$OUT/fake.png;type=image/png" "$BASE/images")"

echo "references and editing"
check "POST /references (upload)" 201 "$(code "${H[@]}" -F "file=@$OUT/full.png;type=image/png" -F 'name=Smoke test owl' -F 'tags=smoke-test' "$BASE/references")"
REF1=$(field 'd["id"]')
B=$(body '{"from_image_id":"%s","name":"Smoke test owl 2","tags":["smoke-test"]}' "$IMG")
check "POST /references (from an image)" 201 "$(code "${H[@]}" "${JSON[@]}" -d "$B" "$BASE/references")"
REF2=$(field 'd["id"]')
check "GET /references?tag=smoke-test" 200 "$(code "${H[@]}" "$BASE/references?tag=smoke-test")"
check "GET /references/{id}" 200 "$(code "${H[@]}" "$BASE/references/$REF1")"
check "GET /references/{id}/image" 200 "$(code "${H[@]}" "$BASE/references/$REF1/image")"
B=$(body '{"prompt":"give the owl a tiny red scarf","source_image_id":"%s","reference_ids":["%s"],"seed":3}' "$UP" "$REF1")
check "POST /jobs (edit with a reference)" 202 "$(code "${H[@]}" "${JSON[@]}" -d "$B" "$BASE/jobs")"
start=$(date +%s)
check "  edit finishes" done "$(wait_job "$(field 'd["job_id"]')")"
echo "        $(($(date +%s) - start)) s"
check "DELETE /references/{id}" 204 "$(code "${H[@]}" -X DELETE "$BASE/references/$REF1")"
check "DELETE /references/{id} again" 404 "$(code "${H[@]}" -X DELETE "$BASE/references/$REF1")"
code "${H[@]}" -X DELETE "$BASE/references/$REF2" >/dev/null

echo "cancel"
code "${H[@]}" "${JSON[@]}" -d '{"prompt":"a castle on a hill","n":2,"seed":5}' "$BASE/jobs" >/dev/null
JOB=$(field 'd["job_id"]')
check "DELETE /jobs/{id}" 200 "$(code "${H[@]}" -X DELETE "$BASE/jobs/$JOB")"
check "  job ends cancelled" cancelled "$(wait_job "$JOB")"

echo "OpenAI-compatible (Open WebUI)"
check "GET /v1/models" 200 "$(code "${H[@]}" "$BASE/v1/models")"
check "GET /v1/status" 200 "$(code "${H[@]}" "$BASE/v1/status")"
B=$(body '{"prompt":"a red apple icon","size":"512x512","seed":1}')
check "POST /v1/images/generations" 200 "$(code "${H[@]}" "${JSON[@]}" -d "$B" "$BASE/v1/images/generations")"
check "POST /v1/unload" 200 "$(code "${H[@]}" -X POST "$BASE/v1/unload")"

echo
echo "passed $pass, failed $fail"
((fail == 0))
