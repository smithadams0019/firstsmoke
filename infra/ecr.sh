#!/usr/bin/env bash
# Create or reuse a private ECR repo per product, then build and push.
#
#   infra/ecr.sh <product> [options]
#
#     --context DIR      docker build context      (default products/<product>)
#     --dockerfile FILE  Dockerfile                (default <context>/Dockerfile)
#     --arch ARCH        linux/amd64 | linux/arm64 | both   (default linux/amd64)
#     --tag TAG          image tag  (default: the short git sha, plus 'latest')
#     --no-push          build only
#     --repo-only        create the repo and stop
#
# Architecture matters here. App Runner is x86 only, so the default is
# linux/amd64. A Graviton EC2 or Lambda arm64 target needs --arch linux/arm64.
# `--arch both` publishes a multi-arch manifest from one command.
#
#   DRY_RUN=1 infra/ecr.sh crackscope --arch both

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PRODUCT=""
CONTEXT=""
DOCKERFILE=""
ARCH="linux/amd64"
TAG=""
PUSH=1
REPO_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --context)    CONTEXT="$2"; shift 2 ;;
    --dockerfile) DOCKERFILE="$2"; shift 2 ;;
    --arch)       ARCH="$2"; shift 2 ;;
    --tag)        TAG="$2"; shift 2 ;;
    --no-push)    PUSH=0; shift ;;
    --repo-only)  REPO_ONLY=1; shift ;;
    -h|--help)    sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
    -*)           die "unknown option $1" ;;
    *)            PRODUCT="$1"; shift ;;
  esac
done

[[ -n "$PRODUCT" ]] || die "usage: infra/ecr.sh <product> [--arch linux/arm64] [--no-push]"
valid_slug "$PRODUCT"
require_aws

REPO="$(ecr_repo_name "$PRODUCT")"
URI="$(ecr_repo_uri "$PRODUCT")"
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# ---- repository ------------------------------------------------------------

log "ECR repository $REPO"
if aws ecr describe-repositories --repository-names "$REPO" >/dev/null 2>&1; then
  ok "repository exists"
else
  set_tags_cli "$PRODUCT"
  run aws ecr create-repository \
    --repository-name "$REPO" \
    --image-scanning-configuration scanOnPush=true \
    --image-tag-mutability MUTABLE \
    --encryption-configuration encryptionType=AES256 \
    --tags "${TAGS_CLI[@]}" >/dev/null
  ok "created $URI"
fi

# Untagged layers are pure cost. Keep the last 10 images, bin the rest.
log "lifecycle policy: keep the last 10 images"
run aws ecr put-lifecycle-policy --repository-name "$REPO" --lifecycle-policy-text '{
  "rules": [
    {
      "rulePriority": 1,
      "description": "expire untagged images after 1 day",
      "selection": {"tagStatus": "untagged", "countType": "sinceImagePushed",
                    "countUnit": "days", "countNumber": 1},
      "action": {"type": "expire"}
    },
    {
      "rulePriority": 2,
      "description": "keep only the 10 most recent images",
      "selection": {"tagStatus": "any", "countType": "imageCountMoreThan", "countNumber": 10},
      "action": {"type": "expire"}
    }
  ]
}' >/dev/null

if [[ "$REPO_ONLY" == "1" ]]; then
  ok "repository ready"
  printf '%s\n' "$URI"
  exit 0
fi

# ---- build context ---------------------------------------------------------

CONTEXT="${CONTEXT:-$ROOT/products/$PRODUCT}"
DOCKERFILE="${DOCKERFILE:-$CONTEXT/Dockerfile}"
[[ -d "$CONTEXT" ]] || die "no build context at $CONTEXT (pass --context)"
[[ -f "$DOCKERFILE" ]] || die "no Dockerfile at $DOCKERFILE (pass --dockerfile)"

if [[ -z "$TAG" ]]; then
  TAG="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo manual)"
  if ! git -C "$ROOT" diff --quiet 2>/dev/null; then
    TAG="${TAG}-dirty"
    warn "working tree is dirty; tagging $TAG"
  fi
fi

need docker
log "logging in to $REGISTRY"
if [[ "$DRY_RUN" != "1" ]]; then
  aws ecr get-login-password --region "$AWS_REGION" \
    | docker login --username AWS --password-stdin "$REGISTRY" >/dev/null
  ok "logged in"
else
  dim "[dry-run] docker login $REGISTRY"
fi

case "$ARCH" in
  both) PLATFORMS="linux/amd64,linux/arm64" ;;
  linux/amd64|linux/arm64) PLATFORMS="$ARCH" ;;
  *) die "--arch must be linux/amd64, linux/arm64 or both" ;;
esac

# buildx gives cross-arch builds and multi-arch manifests from one host. The
# builder is created once and reused; `docker buildx create` is not idempotent,
# so check first.
BUILDER="opencv26"
if ! docker buildx inspect "$BUILDER" >/dev/null 2>&1; then
  log "creating buildx builder $BUILDER"
  run docker buildx create --name "$BUILDER" --driver docker-container --bootstrap >/dev/null
fi
run docker buildx use "$BUILDER"

# A cross-arch build needs qemu registered on the host. Harmless to re-run.
if [[ "$PLATFORMS" == *","* || "$PLATFORMS" == "linux/arm64" ]]; then
  log "registering qemu for cross-architecture builds"
  run docker run --privileged --rm tonistiigi/binfmt --install arm64 >/dev/null 2>&1 || \
    warn "could not register qemu; an arm64 build will fail unless the host is arm64"
fi

BUILD_ARGS=(
  buildx build
  --platform "$PLATFORMS"
  --file "$DOCKERFILE"
  --tag "$URI:$TAG"
  --tag "$URI:latest"
  --build-arg "GIT_SHA=$TAG"
  --provenance false
  --cache-from "type=registry,ref=$URI:buildcache"
  --cache-to "type=registry,ref=$URI:buildcache,mode=max"
)
if [[ "$PUSH" == "1" ]]; then
  BUILD_ARGS+=(--push)
else
  # buildx cannot load a multi-arch manifest into the local daemon.
  [[ "$PLATFORMS" == *","* ]] && die "--no-push cannot be combined with --arch both"
  BUILD_ARGS+=(--load)
fi
BUILD_ARGS+=("$CONTEXT")

log "building $PLATFORMS from $CONTEXT"
run docker "${BUILD_ARGS[@]}"

if [[ "$PUSH" == "1" ]]; then
  ok "pushed $URI:$TAG"
  [[ "$DRY_RUN" == "1" ]] || aws ecr describe-images \
    --repository-name "$REPO" --image-ids "imageTag=$TAG" \
    --query 'imageDetails[0].{pushed:imagePushedAt,bytes:imageSizeInBytes,arch:imageManifestMediaType}' \
    --output table >&2 || true
else
  ok "built $URI:$TAG (not pushed)"
fi

printf '%s:%s\n' "$URI" "$TAG"
