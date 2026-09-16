#!/usr/bin/env bash
# Launch a Graviton EC2 instance for the COOL benchmark arm.
#
#   infra/graviton.sh --ami <ami-id> [options]
#
#     --ami ID           required. The COOL AMI from AWS Marketplace, or a
#                        plain Ubuntu arm64 AMI for the pip-wheel arm.
#     --type TYPE        instance type          (default t4g.small)
#     --name NAME        Name tag               (default opencv26-graviton)
#     --product SLUG     Product tag            (default shared)
#     --key NAME         existing EC2 key pair for ssh
#     --volume GB        root volume size       (default 20)
#     --allow-ssh-from CIDR   ssh source        (default this machine's /32)
#     --allow-app-from CIDR   port 8000 source  (default 0.0.0.0/0)
#     --user-data FILE   cloud-init script
#
# Defaults are deliberately small. **t4g.small is free until 31 December 2026**
# (750 hrs/month aggregated across regions, all customers, existing accounts
# included) That is one free always-on box for the
# whole account, not one per product.
#
# COOL cannot run on t4g: it is a Marketplace AMI whose software fees are not
# covered by the free trial, and it supports c8g/m8g/r8g only. For a COOL arm
# pass `--type c8g.large` (2 vCPU, $0.0798/hr + $0.01/hr software) or larger.
#
# Anything above c8g.large has to be asked for by name; the script refuses to
# guess its way into a big bill.
#
# Tear down with: infra/graviton-teardown.sh <instance-id>

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

AMI=""
TYPE="t4g.small"
NAME="opencv26-graviton"
PRODUCT="shared"
KEY_NAME=""
VOLUME_GB=20
SSH_CIDR=""
APP_CIDR="0.0.0.0/0"
USER_DATA=""

# Sizes we will launch without an argument-level protest. Everything else needs
# --i-know-this-costs-money.
ALLOWED_DEFAULTS="t4g.micro t4g.small t4g.medium c7g.medium c7g.large c8g.large"
FORCE_BIG=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ami)             AMI="$2"; shift 2 ;;
    --type)            TYPE="$2"; shift 2 ;;
    --name)            NAME="$2"; shift 2 ;;
    --product)         PRODUCT="$2"; shift 2 ;;
    --key)             KEY_NAME="$2"; shift 2 ;;
    --volume)          VOLUME_GB="$2"; shift 2 ;;
    --allow-ssh-from)  SSH_CIDR="$2"; shift 2 ;;
    --allow-app-from)  APP_CIDR="$2"; shift 2 ;;
    --user-data)       USER_DATA="$2"; shift 2 ;;
    --i-know-this-costs-money) FORCE_BIG=1; shift ;;
    -h|--help)         sed -n '2,28p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *)                 die "unknown argument $1" ;;
  esac
done

[[ -n "$AMI" ]] || die "--ami is required. Subscribe to the COOL AMI on AWS Marketplace
  (prodview-fdvbfiewzuehs, 'Cloud Optimized OpenCV For AWS Graviton4') and pass
  its AMI id, or pass a plain Ubuntu arm64 AMI for the stock-wheel arm."
[[ "$AMI" =~ ^ami-[0-9a-f]+$ ]] || die "--ami does not look like an AMI id: $AMI"

if [[ " $ALLOWED_DEFAULTS " != *" $TYPE "* && "$FORCE_BIG" != "1" ]]; then
  die "$TYPE is larger than the c8g.large ceiling. If you really mean it, add
  --i-know-this-costs-money. c8g.4xlarge is \$0.64/hr plus \$0.04/hr of COOL
  software: about \$16 a day left running, against an \$80 monthly budget."
fi

case "$TYPE" in
  *g.*|*gd.*) : ;;  # Graviton families all carry a 'g'
  *) warn "$TYPE does not look like a Graviton type; the arm64 wheel will not run on it" ;;
esac

require_aws
need curl

if [[ -z "$SSH_CIDR" ]]; then
  my_ip="$(curl -fsS --max-time 8 https://checkip.amazonaws.com 2>/dev/null | tr -d '[:space:]' || true)"
  if [[ -n "$my_ip" ]]; then
    SSH_CIDR="${my_ip}/32"
    dim "restricting ssh to $SSH_CIDR"
  else
    die "could not determine this machine's IP; pass --allow-ssh-from CIDR explicitly.
  The script will not open port 22 to the world by default."
  fi
fi

SG_NAME="opencv26-graviton-sg"

# ---- security group: 22 and 8000, nothing else ----------------------------

VPC_ID="$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true \
  --query 'Vpcs[0].VpcId' --output text)"
[[ "$VPC_ID" != "None" ]] || die "no default VPC in $AWS_REGION"

SG_ID="$(aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC_ID" \
  --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo None)"

if [[ "$SG_ID" == "None" || -z "$SG_ID" ]]; then
  log "creating security group $SG_NAME"
  set_tags_cli "$PRODUCT"
  SG_ID="$(run aws ec2 create-security-group \
    --group-name "$SG_NAME" \
    --description "opencv26 Graviton benchmark: ssh and the app port only" \
    --vpc-id "$VPC_ID" \
    --tag-specifications "ResourceType=security-group,Tags=[{Key=Project,Value=$PROJECT_TAG},{Key=Product,Value=$PRODUCT},{Key=ManagedBy,Value=infra-scripts}]" \
    --query GroupId --output text)"
  ok "created $SG_ID"
else
  ok "reusing security group $SG_ID"
fi

authorize() {  # port cidr description - already-exists is not an error
  local port="$1" cidr="$2" desc="$3"
  if aws ec2 authorize-security-group-ingress --group-id "$SG_ID" \
      --ip-permissions "IpProtocol=tcp,FromPort=$port,ToPort=$port,IpRanges=[{CidrIp=$cidr,Description=$desc}]" \
      >/dev/null 2>&1; then
    ok "opened $port to $cidr"
  else
    dim "$port/$cidr already allowed"
  fi
}

if [[ "$DRY_RUN" != "1" ]]; then
  authorize 22 "$SSH_CIDR" "ssh"
  authorize 8000 "$APP_CIDR" "app"
else
  dim "[dry-run] would open 22 to $SSH_CIDR and 8000 to $APP_CIDR"
fi

# ---- launch ---------------------------------------------------------------

TAG_SPEC="ResourceType=instance,Tags=[{Key=Name,Value=$NAME},{Key=Project,Value=$PROJECT_TAG},{Key=Product,Value=$PRODUCT},{Key=ManagedBy,Value=infra-scripts}]"
VOL_TAG_SPEC="ResourceType=volume,Tags=[{Key=Name,Value=$NAME},{Key=Project,Value=$PROJECT_TAG},{Key=Product,Value=$PRODUCT},{Key=ManagedBy,Value=infra-scripts}]"

# Reuse an existing instance with the same Name tag rather than launching a
# second one. Re-running this script must not double the bill.
EXISTING="$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=$NAME" "Name=tag:Project,Values=$PROJECT_TAG" \
            "Name=instance-state-name,Values=pending,running,stopping,stopped" \
  --query 'Reservations[0].Instances[0].InstanceId' --output text 2>/dev/null || echo None)"

if [[ "$EXISTING" != "None" && -n "$EXISTING" ]]; then
  state="$(aws ec2 describe-instances --instance-ids "$EXISTING" \
    --query 'Reservations[0].Instances[0].State.Name' --output text)"
  warn "instance $EXISTING already carries Name=$NAME (state: $state)"
  if [[ "$state" == "stopped" ]]; then
    log "starting it rather than launching a second one"
    run aws ec2 start-instances --instance-ids "$EXISTING" >/dev/null
  fi
  INSTANCE_ID="$EXISTING"
else
  LAUNCH=(
    ec2 run-instances
    --image-id "$AMI"
    --instance-type "$TYPE"
    --security-group-ids "$SG_ID"
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$VOLUME_GB,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]"
    --metadata-options "HttpTokens=required,HttpEndpoint=enabled"
    --tag-specifications "$TAG_SPEC" "$VOL_TAG_SPEC"
    --query 'Instances[0].InstanceId' --output text
  )
  [[ -n "$KEY_NAME" ]] && LAUNCH+=(--key-name "$KEY_NAME")
  [[ -n "$USER_DATA" ]] && LAUNCH+=(--user-data "file://$USER_DATA")

  log "launching $TYPE from $AMI"
  INSTANCE_ID="$(run aws "${LAUNCH[@]}")"
  if [[ "$DRY_RUN" == "1" ]]; then
    dim "[dry-run] would wait for the instance and print its public DNS"
    exit 0
  fi
  ok "launched $INSTANCE_ID"
fi

log "waiting for $INSTANCE_ID to run"
run aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"

DNS="$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicDnsName' --output text)"
IP="$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)"

ok "$INSTANCE_ID is running"
dim "type $TYPE, sg $SG_ID, ssh from $SSH_CIDR"
[[ -n "$KEY_NAME" ]] && dim "ssh -i ~/.ssh/$KEY_NAME.pem ubuntu@$DNS"
dim "COOL venv, if this is the COOL AMI: source /opt/cool/venvs/python_3.12/bin/activate"
warn "this instance bills until you run: infra/graviton-teardown.sh $INSTANCE_ID"

printf '%s\n' "$DNS"
printf 'instance_id=%s\npublic_ip=%s\n' "$INSTANCE_ID" "$IP" >&2
