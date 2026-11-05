#!/usr/bin/env bash
# Build the VoxCPM ROCm image (and optionally push it to a registry).
#
#   docker/rocm/build.sh                                  # -> localhost/voxcpm:rocm
#   REGISTRY=ghcr.io/<user> PUSH=1 docker/rocm/build.sh   # also push to GHCR
#
# Settings (environment variables):
#
#   IMAGE_NAME     image name                          (default: voxcpm)
#   TAG            main tag                            (default: rocm)
#   REGISTRY       e.g. ghcr.io/<user>; tags the image there as well
#   PUSH           1 = push the registry tags
#   SHA_TAG        1 = also tag/push REGISTRY/IMAGE_NAME:TAG-<git sha> (default: 1)
#   LOCAL_TAG      0 = with REGISTRY set, skip the localhost/IMAGE_NAME:TAG tag
#   SOURCE_URL     repo URL stored in the OCI source label/annotation, which
#                  GHCR uses to link the package to a repository
#                  (default: the "origin" remote as https://github.com/...)
#   ENGINE         podman | docker                     (default: podman if installed)
#   Extra arguments are passed to "<engine> build", e.g. --no-cache or
#   --build-arg TORCH_VERSION=...
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ENGINE="${ENGINE:-$(command -v podman >/dev/null && echo podman || echo docker)}"
IMAGE_NAME="${IMAGE_NAME:-voxcpm}"
TAG="${TAG:-rocm}"
REGISTRY="${REGISTRY:-}"
PUSH="${PUSH:-0}"
SHA_TAG="${SHA_TAG:-1}"
LOCAL_TAG="${LOCAL_TAG:-1}"

REV="$(git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
SHORT="${REV:0:7}"
# git@github.com:owner/repo.git -> https://github.com/owner/repo
origin_url() {
  git -C "$PROJECT_ROOT" remote get-url origin 2>/dev/null \
    | sed -E 's#^git@([^:]+):#https://\1/#; s#\.git$##'
}
SOURCE_URL="${SOURCE_URL:-$(origin_url)}"
SOURCE_URL="${SOURCE_URL:-https://github.com/OpenBMB/VoxCPM}"

LOCAL_REF="localhost/$IMAGE_NAME:$TAG"
[ "$ENGINE" = docker ] && LOCAL_REF="$IMAGE_NAME:$TAG"

# Set both labels and manifest annotations: podman otherwise inherits the base
# image's org.opencontainers.image.source annotation (docker-library/python),
# and GHCR then can't link the package to SOURCE_URL.
meta=()
for kv in \
  "org.opencontainers.image.source=$SOURCE_URL" \
  "org.opencontainers.image.url=$SOURCE_URL/tree/$REV/docker/rocm" \
  "org.opencontainers.image.revision=$REV" \
  "org.opencontainers.image.description=VoxCPM2 TTS WebUI for AMD GPUs (ROCm)" \
  "org.opencontainers.image.licenses=Apache-2.0" \
  "org.opencontainers.image.base.name=docker.io/library/python:3.11-slim-bookworm"; do
  meta+=(--label "$kv")
  [ "$ENGINE" = podman ] && meta+=(--annotation "$kv")
done

remote_tags=()
if [ -n "$REGISTRY" ]; then
  remote_tags=("$TAG")
  [ "$SHA_TAG" = 1 ] && remote_tags+=("$TAG-$SHORT")
fi
names=()
[ "$LOCAL_TAG" = 1 ] || [ -z "$REGISTRY" ] && names+=("$LOCAL_REF")
for t in "${remote_tags[@]}"; do names+=("$REGISTRY/$IMAGE_NAME:$t"); done
tags=()
for n in "${names[@]}"; do tags+=(-t "$n"); done

echo "+ building ${names[*]} (rev $SHORT, source $SOURCE_URL)" >&2
"$ENGINE" build -f "$SCRIPT_DIR/Containerfile" "${meta[@]}" "${tags[@]}" "$@" "$PROJECT_ROOT"

if [ "$PUSH" = 1 ]; then
  [ -n "$REGISTRY" ] || { echo "PUSH=1 needs REGISTRY (e.g. ghcr.io/<user>)" >&2; exit 1; }
  for t in "${remote_tags[@]}"; do
    echo "+ pushing $REGISTRY/$IMAGE_NAME:$t" >&2
    "$ENGINE" push "$REGISTRY/$IMAGE_NAME:$t"
  done
fi

echo "Done: ${names[*]}" >&2
