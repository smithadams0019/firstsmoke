# infra

Shell scripts, not CDK or Terraform. Each one reconciles AWS to a desired state,
prints what it did, and is safe to re-run. Region `us-east-1`, account
`<aws-account-id>`, everything tagged `Project=opencv26` and `Product=<slug>`.

Set `DRY_RUN=1` on any of them to see the commands without running them.

| Script | What it does |
|---|---|
| `s3.sh` | The one artefact bucket: versioned, private, AES256, 30-day object expiry |
| `ecr.sh <product>` | Create or reuse a private repo, log in, buildx, push |
| `apprunner.sh <product>` | Create or update a service from ECR, wait, print the URL |
| `graviton.sh --ami <id>` | Launch a Graviton instance for the COOL benchmark |
| `graviton-teardown.sh <id>` | Terminate it, and the security group if nothing else uses it |

## The usual sequence

```bash
infra/s3.sh                                   # once, shared
infra/ecr.sh crackscope --context . --dockerfile products/crackscope/Dockerfile
infra/apprunner.sh crackscope                 # prints https://<id>.awsapprunner.com
infra/apprunner.sh crackscope --status
```

## Architecture flags

`ecr.sh` defaults to `--arch linux/amd64` because **App Runner is x86 only**.
For a Graviton or Lambda arm64 target pass `--arch linux/arm64`, or `--arch both`
for a multi-arch manifest. Cross-arch builds register qemu automatically.

## Things the scripts refuse to do

- `graviton.sh` will not launch anything above `c8g.large` without
  `--i-know-this-costs-money`. c8g.4xlarge is about $16 a day against an $80
  monthly budget.
- `graviton.sh` will not open port 22 to the world. It restricts ssh to this
  machine's `/32` unless you pass `--allow-ssh-from` explicitly.
- `graviton.sh` will not launch a second instance with a Name tag that already
  exists; it starts the stopped one instead.
- `graviton-teardown.sh` will not run without an instance id, and will not touch
  an instance that is not tagged `Project=opencv26`.
- `apprunner.sh --delete` asks before deleting.

## What exists right now

Created and verified:

- `s3://opencv26-artifacts-<aws-account-id>` — versioned, public access blocked,
  AES256, 30-day expiry, 7-day non-current version expiry, tagged.
- ECR `opencv26/foundation-probe` — scan-on-push, lifecycle keeping 10 images,
  tagged. One 147 MB `linux/amd64` image pushed from `products/_template`.

`apprunner.sh` and `graviton.sh` have been dry-run and their guard rails tested,
but no App Runner service or EC2 instance has been created — the first product
to deploy will be the first real run.
