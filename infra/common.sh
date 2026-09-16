#!/usr/bin/env bash
# Shared settings and helpers for every infra script. Source this, do not run it.
#
# Everything here is idempotent by design: each script's job is to make the
# world match the desired state, whether or not it already does.

set -euo pipefail

export AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_DEFAULT_REGION="$AWS_REGION"
readonly PROJECT_TAG="opencv26"
readonly EXPECTED_ACCOUNT="<aws-account-id>"

# Nothing larger than this launches without being asked for explicitly.
readonly MAX_INSTANCE_DEFAULT="c8g.large"

DRY_RUN="${DRY_RUN:-0}"

c_red=$'\033[31m'; c_green=$'\033[32m'; c_yellow=$'\033[33m'
c_blue=$'\033[34m'; c_dim=$'\033[2m'; c_off=$'\033[0m'

log()  { printf '%s==>%s %s\n' "$c_blue" "$c_off" "$*" >&2; }
ok()   { printf '%s  ok%s %s\n' "$c_green" "$c_off" "$*" >&2; }
warn() { printf '%swarn%s %s\n' "$c_yellow" "$c_off" "$*" >&2; }
die()  { printf '%s err%s %s\n' "$c_red" "$c_off" "$*" >&2; exit 1; }
dim()  { printf '%s     %s%s\n' "$c_dim" "$*" "$c_off" >&2; }

# run <cmd...>  - echoes under DRY_RUN=1, executes otherwise.
run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '%s  [dry-run] %s%s\n' "$c_dim" "$*" "$c_off" >&2
    return 0
  fi
  "$@"
}

need() {
  command -v "$1" >/dev/null 2>&1 || die "$1 is not installed"
}

require_aws() {
  need aws
  local account
  account="$(aws sts get-caller-identity --query Account --output text 2>/dev/null)" \
    || die "AWS credentials are not configured (aws sts get-caller-identity failed)"
  if [[ "$account" != "$EXPECTED_ACCOUNT" ]]; then
    die "wrong AWS account: got $account, expected $EXPECTED_ACCOUNT. Check AWS_PROFILE."
  fi
  ACCOUNT_ID="$account"
  export ACCOUNT_ID
  dim "account $ACCOUNT_ID, region $AWS_REGION"
}

# Every script tags what it makes, so Cost Explorer can split the $80/month budget
# by product. Two formats because the AWS CLI is not consistent about it.
# Sets the TAGS_CLI array for the services that want Key=..,Value=.. shorthand
# (ec2, ecr, iam). It must be an array: the CLI needs one argv word per tag, and
# a quoted string of three pairs is rejected as a repeated "Value" key.
TAGS_CLI=()
set_tags_cli() {
  local product="${1:-shared}"
  TAGS_CLI=(
    "Key=Project,Value=$PROJECT_TAG"
    "Key=Product,Value=$product"
    "Key=ManagedBy,Value=infra-scripts"
  )
}

tags_json() {  # [{"Key":..,"Value":..}] style (s3, apprunner)
  local product="${1:-shared}"
  printf '[{"Key":"Project","Value":"%s"},{"Key":"Product","Value":"%s"},{"Key":"ManagedBy","Value":"infra-scripts"}]' \
    "$PROJECT_TAG" "$product"
}

valid_slug() {
  [[ "$1" =~ ^[a-z][a-z0-9-]{1,30}$ ]] || die "invalid product slug '$1' (lowercase, digits, hyphens)"
}

confirm() {
  [[ "${ASSUME_YES:-0}" == "1" ]] && return 0
  read -r -p "$1 [y/N] " reply
  [[ "$reply" == "y" || "$reply" == "Y" ]]
}

artifact_bucket() {
  printf 'opencv26-artifacts-%s' "$EXPECTED_ACCOUNT"
}

ecr_repo_name() { printf 'opencv26/%s' "$1"; }

ecr_repo_uri() {
  printf '%s.dkr.ecr.%s.amazonaws.com/%s' "$ACCOUNT_ID" "$AWS_REGION" "$(ecr_repo_name "$1")"
}
