#!/usr/bin/env bash
# Pull a gated HF model into the HF cache with resumable wget.
# `wget -c` resumes a half-finished blob (hf/hfdownloader kept stalling
# mid-file and restarting from zero). Files land in the normal cache layout
# (blobs/<sha> + snapshots/<commit>/<path> symlinks) so from_pretrained(repo_id)
# keeps working unchanged. Idempotent: complete blobs are skipped.
set -euo pipefail
# Arg wins; else DEPURPLE_MODEL (what the depurple pipeline edits); else e4b.
MODEL="${1:-${DEPURPLE_MODEL:-google/gemma-4-E4B-it}}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
TOKEN="${HF_TOKEN:-$(cat "$HF_HOME/token" 2>/dev/null || true)}"
CACHE="$HF_HOME/hub/models--${MODEL//\//--}"

# Manifest from the HF API: first line = commit sha, then "<blob-sha>\t<path>".
manifest=$(python3 - "$MODEL" "$TOKEN" <<'PY'
import json, sys, urllib.request
model, token = sys.argv[1], sys.argv[2]
def get(url):
    r = urllib.request.Request(url)
    if token: r.add_header("Authorization", "Bearer " + token)
    return json.load(urllib.request.urlopen(r))
sha = get(f"https://huggingface.co/api/models/{model}")["sha"]
print(sha)
for f in get(f"https://huggingface.co/api/models/{model}/tree/{sha}?recursive=1"):
    if f["type"] == "file":
        print((f.get("lfs") or f)["oid"] + "\t" + f["path"])
PY
)
SHA=$(head -1 <<<"$manifest")

AUTH=(); [ -n "$TOKEN" ] && AUTH=(--header "Authorization: Bearer $TOKEN")
mkdir -p "$CACHE/blobs" "$CACHE/snapshots/$SHA" "$CACHE/refs"
echo -n "$SHA" > "$CACHE/refs/main"

tail -n +2 <<<"$manifest" | while IFS=$'\t' read -r blob path; do
    dst="$CACHE/snapshots/$SHA/$path"
    mkdir -p "$(dirname "$dst")"
    wget -c "${AUTH[@]}" -O "$CACHE/blobs/$blob" \
        "https://huggingface.co/$MODEL/resolve/$SHA/$path"
    ln -srf "$CACHE/blobs/$blob" "$dst"
done
echo "done: $MODEL -> $CACHE/snapshots/$SHA"
