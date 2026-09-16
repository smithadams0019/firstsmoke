#!/usr/bin/env bash
# One shared artefact bucket: versioned, private, objects expire after 30 days.
#
#   infra/s3.sh                 create or reconcile
#   infra/s3.sh --show          print the current configuration
#   DRY_RUN=1 infra/s3.sh       print what it would do
#
# Products keep models, evidence frames and benchmark output under their own
# prefix: s3://<bucket>/<product>/...

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_aws

BUCKET="$(artifact_bucket)"

if [[ "${1:-}" == "--show" ]]; then
  aws s3api get-bucket-versioning --bucket "$BUCKET"
  aws s3api get-public-access-block --bucket "$BUCKET"
  aws s3api get-bucket-lifecycle-configuration --bucket "$BUCKET"
  aws s3api get-bucket-tagging --bucket "$BUCKET"
  exit 0
fi

log "artefact bucket s3://$BUCKET"

if aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  ok "bucket exists, reconciling settings"
else
  log "creating bucket"
  # us-east-1 rejects a LocationConstraint; every other region requires one.
  if [[ "$AWS_REGION" == "us-east-1" ]]; then
    run aws s3api create-bucket --bucket "$BUCKET" --region "$AWS_REGION"
  else
    run aws s3api create-bucket --bucket "$BUCKET" --region "$AWS_REGION" \
      --create-bucket-configuration "LocationConstraint=$AWS_REGION"
  fi
  run aws s3api wait bucket-exists --bucket "$BUCKET"
  ok "created"
fi

log "blocking all public access"
run aws s3api put-public-access-block --bucket "$BUCKET" \
  --public-access-block-configuration \
  'BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true'

log "enabling versioning"
run aws s3api put-bucket-versioning --bucket "$BUCKET" \
  --versioning-configuration Status=Enabled

log "enabling default encryption"
run aws s3api put-bucket-encryption --bucket "$BUCKET" \
  --server-side-encryption-configuration \
  '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"},"BucketKeyEnabled":true}]}'

log "lifecycle: expire objects after 30 days"
# Versioning is on, so non-current versions and aborted uploads need their own
# rules or the bucket keeps paying for data nobody can see.
run aws s3api put-bucket-lifecycle-configuration --bucket "$BUCKET" \
  --lifecycle-configuration '{
    "Rules": [
      {
        "ID": "expire-objects-after-30-days",
        "Status": "Enabled",
        "Filter": {"Prefix": ""},
        "Expiration": {"Days": 30},
        "NoncurrentVersionExpiration": {"NoncurrentDays": 7},
        "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 3}
      }
    ]
  }'

log "tagging"
run aws s3api put-bucket-tagging --bucket "$BUCKET" \
  --tagging "{\"TagSet\":$(tags_json shared)}"

ok "s3://$BUCKET ready"
printf 's3://%s\n' "$BUCKET"
