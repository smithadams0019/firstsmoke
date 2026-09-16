# What was created on AWS, and what it costs

Account <aws-account-id>. Everything tagged `Project=opencv26`, `Product=firstsmoke`.

**Live endpoint: https://afynmqkk9e.eu-west-1.awsapprunner.com**

## Why not us-east-1

The brief asked for us-east-1 and this is deployed in **eu-west-1** instead. The
reason is a hard account limit, not a preference:

```
An error occurred (InvalidRequestException) when calling the CreateService
operation: Account <aws-account-id> is restricted and can support only two App
Runner services per region at the moment.
```

At deploy time the two us-east-1 slots were held by `ugjcs-backend` and
`cairn-v2`, both unrelated production services belonging to other projects, and
deleting either to make room was not ours to do. us-east-2 held
`opencv26-preempt` and `recourse`; us-west-2 held `muster` and `gleaner`.
eu-west-1 had one free slot and it was taken.

Consequences worth knowing:

- The container image is in ECR in **both** us-east-1 and eu-west-1. App Runner
  can only pull from a repository in its own region, so the us-east-1 copy is
  now redundant and is listed below as a candidate for deletion.
- A judge in North America sees roughly 100 ms more latency than they would from
  Virginia. A scenario run takes about 32 s on the deployed 2 vCPU container
  against about 20 s on the development machine; almost all of that difference is
  the CPU, not the distance.
- The S3 artefact bucket stays in us-east-1, which is fine: nothing on the
  request path touches it.

## Resources

| Resource | Region | Identifier | State | Cost |
|---|---|---|---|---|
| App Runner service | eu-west-1 | `opencv26-firstsmoke`, 2 vCPU / 4 GB | **running, left running** | see below |
| App Runner ECR access role | eu-west-1 | `opencv26-apprunner-ecr-access` | active | free |
| ECR repository | eu-west-1 | `opencv26/firstsmoke` | 653 MB stored | $0.10 / GB-month |
| ECR repository | us-east-1 | `opencv26/firstsmoke` | 653 MB stored, **redundant** | $0.10 / GB-month |
| S3 objects | us-east-1 | `s3://opencv26-artifacts-<aws-account-id>/firstsmoke/` | 565 KB | negligible |

The S3 prefix holds `evaluation.json` (the full evaluation, 352 KB) and the
trained ONNX confirmer with its training report. The bucket has a 30-day object
expiry, which is shared policy; these are reproducible from the repo, so that is
fine.

## The hourly number

App Runner charges provisioned memory continuously and vCPU only while a request
is being served. At eu-west-1 list prices:

| Component | Rate | Hours | Cost |
|---|---|---|---|
| Provisioned memory, 4 GB | $0.00775 / GB-hour | continuous | **$0.0310 / hour** |
| Active vCPU, 2 vCPU | $0.078 / vCPU-hour | only while serving | $0.156 / hour *while active* |

**Idle: $0.031 an hour, which is $0.74 a day and about $22.60 for a 30-day
month.** That is the figure to plan on, because the service is left running
rather than scaled to zero.

Active cost is small in comparison. A scenario run is about 32 seconds of two
vCPUs, or $0.0014. A judge who runs all four bundled incidents costs about half
a penny. Even a hundred such visits over the judging window adds about $0.55.

Storage adds roughly $0.13 a month for the two ECR copies together.

**Total expected: about $23 for a month of continuous availability.** Deleting
the redundant us-east-1 ECR copy takes that to about $22.95, which is not worth
doing for the money but is worth doing for tidiness.

This is a deliberate choice under the instruction that cost is not the
constraint and the judge experience is. The alternative, scaling to zero between
demonstrations, would save about $20 a month and cost a judge a cold start on
their first click.

## Turning it off

```bash
AWS_REGION=eu-west-1 infra/apprunner.sh firstsmoke --delete   # asks first
aws ecr delete-repository --repository-name opencv26/firstsmoke \
    --region us-east-1 --force                                # the redundant copy
aws ecr delete-repository --repository-name opencv26/firstsmoke \
    --region eu-west-1 --force
aws s3 rm s3://opencv26-artifacts-<aws-account-id>/firstsmoke/ --recursive
```

## Checking it

```bash
AWS_REGION=eu-west-1 infra/apprunner.sh firstsmoke --status
curl -s https://afynmqkk9e.eu-west-1.awsapprunner.com/healthz
curl -s https://afynmqkk9e.eu-west-1.awsapprunner.com/version
```

`/version` prints the running OpenCV version, which for this entry is the number
that matters: it must say 5.0.0.
