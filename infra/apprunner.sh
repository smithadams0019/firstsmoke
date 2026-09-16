#!/usr/bin/env bash
# Create or update an App Runner service from an ECR image, wait, print the URL.
# This is the default path for a product's judge-accessible web endpoint.
#
#   infra/apprunner.sh <product> [options]
#
#     --tag TAG      image tag to deploy            (default latest)
#     --cpu N        0.25|0.5|1|2|4 vCPU            (default 0.25)
#     --memory N     0.5|1|2|3|4|... GB             (default 0.5)
#     --port N       container port                 (default 8000)
#     --env K=V      repeatable runtime env var
#     --delete       delete the service and stop
#     --status       print the current status and URL, change nothing
#
# App Runner is **x86 only** - there is no arm64 runtime. Build with
# `infra/ecr.sh <product>` (which defaults to linux/amd64). A COOL or Graviton
# workload cannot run here; use infra/graviton.sh for that.
#
# Cost: ~$2.52/month per service at idle on 0.25 vCPU / 0.5 GB, so five
# services is ~$12.60/month against the $80 budget. `--delete` between demo
# windows if that matters.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

PRODUCT=""
TAG="latest"
CPU="0.25 vCPU"
MEMORY="0.5 GB"
PORT="8000"
ACTION="deploy"
ENV_PAIRS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)    TAG="$2"; shift 2 ;;
    --cpu)    CPU="$2 vCPU"; shift 2 ;;
    --memory) MEMORY="$2 GB"; shift 2 ;;
    --port)   PORT="$2"; shift 2 ;;
    --env)    ENV_PAIRS+=("$2"); shift 2 ;;
    --delete) ACTION="delete"; shift ;;
    --status) ACTION="status"; shift ;;
    -h|--help) sed -n '2,25p' "${BASH_SOURCE[0]}"; exit 0 ;;
    -*)       die "unknown option $1" ;;
    *)        PRODUCT="$1"; shift ;;
  esac
done

[[ -n "$PRODUCT" ]] || die "usage: infra/apprunner.sh <product> [--tag TAG]"
valid_slug "$PRODUCT"
require_aws

SERVICE="opencv26-$PRODUCT"
IMAGE="$(ecr_repo_uri "$PRODUCT"):$TAG"
ACCESS_ROLE="opencv26-apprunner-ecr-access"

service_arn() {
  aws apprunner list-services \
    --query "ServiceSummaryList[?ServiceName=='$SERVICE'].ServiceArn | [0]" \
    --output text 2>/dev/null | grep -v '^None$' || true
}

service_field() {
  aws apprunner describe-service --service-arn "$1" --query "Service.$2" --output text
}

ARN="$(service_arn)"

# ---- status / delete -------------------------------------------------------

if [[ "$ACTION" == "status" ]]; then
  [[ -n "$ARN" ]] || die "no App Runner service named $SERVICE"
  printf 'status: %s\nurl:    https://%s\nimage:  %s\n' \
    "$(service_field "$ARN" Status)" \
    "$(service_field "$ARN" ServiceUrl)" \
    "$(service_field "$ARN" SourceConfiguration.ImageRepository.ImageIdentifier)"
  exit 0
fi

if [[ "$ACTION" == "delete" ]]; then
  [[ -n "$ARN" ]] || { warn "no service $SERVICE; nothing to delete"; exit 0; }
  confirm "delete App Runner service $SERVICE?" || die "aborted"
  run aws apprunner delete-service --service-arn "$ARN" >/dev/null
  ok "delete requested for $SERVICE"
  exit 0
fi

# ---- the ECR access role ---------------------------------------------------
# App Runner pulls from a private ECR repo through a service-linked role. One
# role is shared by all five products.

ensure_access_role() {
  if aws iam get-role --role-name "$ACCESS_ROLE" >/dev/null 2>&1; then
    dim "access role $ACCESS_ROLE exists"
    return
  fi
  log "creating ECR access role $ACCESS_ROLE"
  set_tags_cli shared
  run aws iam create-role --role-name "$ACCESS_ROLE" \
    --description "Lets App Runner pull opencv26 images from ECR" \
    --tags "${TAGS_CLI[@]}" \
    --assume-role-policy-document '{
      "Version": "2012-10-17",
      "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "build.apprunner.amazonaws.com"},
        "Action": "sts:AssumeRole"
      }]
    }' >/dev/null
  run aws iam attach-role-policy --role-name "$ACCESS_ROLE" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess
  # IAM is eventually consistent; App Runner rejects a role it cannot yet see.
  log "waiting for the role to propagate"
  run sleep 12
  ok "role ready"
}

ensure_access_role
ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ACCESS_ROLE}"

if [[ "$DRY_RUN" != "1" ]]; then
  aws ecr describe-images --repository-name "$(ecr_repo_name "$PRODUCT")" \
    --image-ids "imageTag=$TAG" >/dev/null 2>&1 \
    || die "no image $IMAGE - run infra/ecr.sh $PRODUCT first"
fi

ENV_JSON="{}"
if [[ ${#ENV_PAIRS[@]} -gt 0 ]]; then
  ENV_JSON="$(python3 -c '
import json, sys
print(json.dumps(dict(pair.split("=", 1) for pair in sys.argv[1:])))' "${ENV_PAIRS[@]}")"
fi

SOURCE_CONFIG="$(python3 -c '
import json, sys
image, port, env, role = sys.argv[1:5]
print(json.dumps({
  "ImageRepository": {
    "ImageIdentifier": image,
    "ImageRepositoryType": "ECR",
    "ImageConfiguration": {
      "Port": port,
      "RuntimeEnvironmentVariables": json.loads(env),
    },
  },
  "AutoDeploymentsEnabled": False,
  "AuthenticationConfiguration": {"AccessRoleArn": role},
}))' "$IMAGE" "$PORT" "$ENV_JSON" "$ROLE_ARN")"

HEALTH_CONFIG='{"Protocol":"HTTP","Path":"/healthz","Interval":10,"Timeout":5,"HealthyThreshold":1,"UnhealthyThreshold":5}'
INSTANCE_CONFIG="$(printf '{"Cpu":"%s","Memory":"%s"}' "$CPU" "$MEMORY")"

if [[ -n "$ARN" ]]; then
  log "updating $SERVICE to $IMAGE"
  run aws apprunner update-service \
    --service-arn "$ARN" \
    --source-configuration "$SOURCE_CONFIG" \
    --instance-configuration "$INSTANCE_CONFIG" \
    --health-check-configuration "$HEALTH_CONFIG" >/dev/null
else
  log "creating $SERVICE from $IMAGE ($CPU / $MEMORY)"
  run aws apprunner create-service \
    --service-name "$SERVICE" \
    --source-configuration "$SOURCE_CONFIG" \
    --instance-configuration "$INSTANCE_CONFIG" \
    --health-check-configuration "$HEALTH_CONFIG" \
    --tags "$(tags_json "$PRODUCT")" >/dev/null
  ARN="$(service_arn)"
fi

if [[ "$DRY_RUN" == "1" ]]; then
  dim "[dry-run] would wait for $SERVICE to become RUNNING"
  exit 0
fi

log "waiting for $SERVICE to go healthy (up to 15 minutes)"
deadline=$(( $(date +%s) + 900 ))
while :; do
  status="$(service_field "$ARN" Status)"
  case "$status" in
    RUNNING)
      url="$(service_field "$ARN" ServiceUrl)"
      ok "$SERVICE is RUNNING"
      log "checking https://$url/healthz"
      if curl -fsS --max-time 20 "https://$url/healthz" >&2; then
        printf '\n'
        ok "health check passed"
      else
        warn "service is RUNNING but /healthz did not answer yet"
      fi
      printf 'https://%s\n' "$url"
      exit 0
      ;;
    CREATE_FAILED|DELETE_FAILED)
      die "$SERVICE entered $status - check: aws apprunner list-operations --service-arn $ARN"
      ;;
    OPERATION_IN_PROGRESS|CREATING|UPDATING|DELETING|PAUSED|RESUMING)
      dim "status $status"
      ;;
    *)
      dim "status $status"
      ;;
  esac
  if [[ $(date +%s) -ge $deadline ]]; then
    die "timed out waiting for $SERVICE (last status $status)"
  fi
  sleep 15
done
