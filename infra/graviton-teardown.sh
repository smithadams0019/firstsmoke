#!/usr/bin/env bash
# Terminate a Graviton instance launched by infra/graviton.sh.
#
#   infra/graviton-teardown.sh <instance-id> [--keep-sg] [--yes]
#
# Refuses to run without an instance id: there is no "terminate everything"
# mode, on purpose.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

INSTANCE_ID=""
KEEP_SG=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep-sg) KEEP_SG=1; shift ;;
    --yes)     ASSUME_YES=1; shift ;;
    -h|--help) sed -n '2,10p' "${BASH_SOURCE[0]}"; exit 0 ;;
    -*)        die "unknown option $1" ;;
    *)         INSTANCE_ID="$1"; shift ;;
  esac
done

if [[ -z "$INSTANCE_ID" ]]; then
  die "an instance id is required: infra/graviton-teardown.sh i-0123456789abcdef0
  List what is running:
    aws ec2 describe-instances --filters Name=tag:Project,Values=opencv26 \\
      Name=instance-state-name,Values=running \\
      --query 'Reservations[].Instances[].[InstanceId,InstanceType,Tags[?Key==\`Name\`].Value|[0]]' \\
      --output table"
fi
[[ "$INSTANCE_ID" =~ ^i-[0-9a-f]+$ ]] || die "'$INSTANCE_ID' is not an instance id"

require_aws

read -r state itype name < <(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].[State.Name,InstanceType,Tags[?Key==`Name`]|[0].Value]' \
  --output text 2>/dev/null) || die "no such instance: $INSTANCE_ID"

project="$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].Tags[?Key==`Project`]|[0].Value' --output text)"
if [[ "$project" != "$PROJECT_TAG" ]]; then
  die "$INSTANCE_ID is not tagged Project=$PROJECT_TAG (got '$project'). Refusing to touch it."
fi

log "$INSTANCE_ID  $itype  $name  ($state)"
confirm "terminate it?" || die "aborted"

run aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" >/dev/null
log "waiting for termination"
run aws ec2 wait instance-terminated --instance-ids "$INSTANCE_ID"
ok "terminated $INSTANCE_ID"

if [[ "$KEEP_SG" == "0" ]]; then
  SG_ID="$(aws ec2 describe-security-groups --filters Name=group-name,Values=opencv26-graviton-sg \
    --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo None)"
  if [[ "$SG_ID" != "None" && -n "$SG_ID" ]]; then
    in_use="$(aws ec2 describe-instances \
      --filters "Name=instance.group-id,Values=$SG_ID" \
                "Name=instance-state-name,Values=pending,running,stopping,stopped" \
      --query 'length(Reservations[].Instances[])' --output text)"
    if [[ "$in_use" == "0" ]]; then
      log "deleting the now-unused security group $SG_ID"
      run aws ec2 delete-security-group --group-id "$SG_ID" >/dev/null 2>&1 \
        || warn "could not delete $SG_ID yet; ENIs can take a minute to release"
    else
      dim "security group $SG_ID still has $in_use instance(s); leaving it"
    fi
  fi
fi

ok "done. Cost stops now."
