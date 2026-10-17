#!/usr/bin/env bash
# Download the pre-quantized MLX image models into the Hugging Face cache
# (~/.cache/huggingface/hub by default, outside this repo). Revisions are pinned
# to the commits the server was tested with (see image-server config.py).
# Safe to re-run: files that are already present are skipped.
#
#   scripts/download_models.sh            # all models
#   scripts/download_models.sh klein      # just one (klein | zimage | qwen)
#
# None of these repos are gated, so no Hugging Face token is needed.
set -euo pipefail
cd "$(dirname "$0")/.."

# Use the same cache location as the server: HF_HOME / HF_HUB_CACHE from .env.
if [[ -f .env ]]; then
  for key in HF_HOME HF_HUB_CACHE; do
    value=$(sed -n "s/^$key=//p" .env | tail -n 1 | sed -e 's/^"\(.*\)"$/\1/')
    if [[ -n "$value" && -z "${!key:-}" ]]; then export "$key=${value/#\~/$HOME}"; fi
  done
fi

HF=(uvx --from "huggingface_hub>=1.1.6,<2" hf)

download_klein() {
  # FLUX.2 [klein] 4B, 8-bit. Apache-2.0. The repo also holds a duplicate copy
  # under a stray top-level directory, so only the real folders are included.
  "${HF[@]}" download mflux-community/flux2-klein-4b-mflux-q8 \
    --revision ca49409e4ce21c6717ee87caadd9488e0b15208f \
    --include "text_encoder/*" --include "tokenizer/*" \
    --include "transformer/*" --include "vae/*"
}

download_zimage() {
  # Z-Image-Turbo, 8-bit. Apache-2.0.
  "${HF[@]}" download mflux-community/z-image-turbo-mflux-q8 \
    --revision 4ccff28917346aa9daae49cd3c477cf5260a242c
}

download_qwen() {
  # Qwen-Image-2.1-Turbo, 4-bit (text encoder quantized too).
  # Qwen Research License: research or evaluation use only.
  "${HF[@]}" download mflux-community/qwen-image-2-1-turbo-mflux-q4 \
    --revision bda8951ec6b8cc288a961432fe15c22433521a1d
}

targets=("$@")
[[ ${#targets[@]} -eq 0 ]] && targets=(klein zimage qwen)

for t in "${targets[@]}"; do
  echo "==> $t"
  "download_$t"
done
echo "==> done"
