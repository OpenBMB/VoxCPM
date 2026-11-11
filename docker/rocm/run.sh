#!/usr/bin/env bash
# Run the VoxCPM ROCm image with AMD GPU passthrough.
#
#   docker/rocm/run.sh                          # TTS / cloning / voice design WebUI
#   docker/rocm/run.sh python lora_ft_webui.py  # LoRA fine-tuning WebUI
#
# Everything is configurable through environment variables:
#
#   IMAGE          image                           (default: voxcpm:rocm, see build.sh)
#   ENGINE         podman | docker                 (default: podman if installed)
#   PORT           host + container port           (default: 8808)
#   DATA_ROOT      host dir holding models/ data/ lora/ output/
#                                                  (default: project root)
#   NETWORK, IP    optional container network / static IP
#   HSA_OVERRIDE_GFX_VERSION   auto-detected for RDNA2/RDNA3 cards outside the
#                  official ROCm list (e.g. RX 6700 XT -> 10.3.0); set to
#                  override, or to "none" to disable.
#   Any of VOXCPM_MODEL VOXCPM_DTYPE VOXCPM_OPTIMIZE ASR_BACKEND ASR_MODEL
#   ASR_LANGUAGE ASR_DEVICE TEXT_NUMBER_LANG HF_TOKEN MIOPEN_FIND_MODE
#   AUTO_SPLIT_WORDS AUTO_SPLIT_PAUSE DIT_STEPS are passed through.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

IMAGE="${IMAGE:-voxcpm:rocm}"
ENGINE="${ENGINE:-$(command -v podman >/dev/null && echo podman || echo docker)}"
PORT="${PORT:-8808}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT}"

# gfx target of the first GPU: MMmmss, e.g. 100301 = gfx1031
detect_gfx_override() {
  local v
  for f in /sys/class/kfd/kfd/topology/nodes/*/properties; do
    v=$(awk '/gfx_target_version/ {print $2}' "$f" 2>/dev/null)
    [ -n "$v" ] && [ "$v" != 0 ] && break
  done
  case "$v" in
    10030[1-9]) echo 10.3.0 ;;         # gfx1031..1036 -> gfx1030 kernels
    11000[1-9]) echo 11.0.0 ;;         # gfx1101..1103 -> gfx1100 kernels
    *) echo "" ;;
  esac
}
GFX="${HSA_OVERRIDE_GFX_VERSION:-$(detect_gfx_override)}"

mkdir -p "$DATA_ROOT"/{models,data,lora,output}

args=(run --rm --name "${NAME:-voxcpm}"
  -p "$PORT:$PORT" -e PORT="$PORT" -e GRADIO_SERVER_PORT="$PORT"
  --shm-size 4g
  --device /dev/kfd --device /dev/dri
  --security-opt label=disable
  -v "$DATA_ROOT/models:/app/models"
  -v "$DATA_ROOT/data:/app/data"
  -v "$DATA_ROOT/lora:/app/lora"
  -v "$DATA_ROOT/output:/app/output")

[ -t 0 ] && args+=(-it)

if [ "$ENGINE" = podman ]; then
  # rootless: container root = your user, so volume files stay yours.
  # keep-groups passes your video/render membership through to /dev/kfd.
  args+=(--group-add keep-groups)
else
  args+=(--user "$(id -u):$(id -g)" --group-add video --group-add render)
fi
# AppArmor hosts (Debian/Ubuntu) can block /dev/kfd ioctls
[ -d /sys/kernel/security/apparmor ] && args+=(--security-opt apparmor=unconfined)

[ -n "$GFX" ] && [ "$GFX" != none ] && args+=(-e HSA_OVERRIDE_GFX_VERSION="$GFX")
[ -n "${NETWORK:-}" ] && args+=(--network "$NETWORK")
[ -n "${IP:-}" ] && args+=(--ip "$IP")

for v in VOXCPM_MODEL VOXCPM_DTYPE VOXCPM_OPTIMIZE ASR_BACKEND ASR_MODEL \
         ASR_LANGUAGE ASR_DEVICE TEXT_NUMBER_LANG HF_TOKEN MIOPEN_FIND_MODE \
         AUTO_SPLIT_WORDS AUTO_SPLIT_PAUSE DIT_STEPS; do
  [ -n "${!v+x}" ] && args+=(-e "$v=${!v}")
done

echo "+ $ENGINE ${args[*]} $IMAGE $*" >&2
exec "$ENGINE" "${args[@]}" "$IMAGE" "$@"
