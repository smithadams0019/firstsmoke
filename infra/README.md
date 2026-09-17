# infra

Shell scripts that reconcile AWS to a desired state, print what they did, and are safe
to re-run. The region comes from `AWS_REGION`, the account from `EXPECTED_ACCOUNT`, and
everything is tagged `Project=opencv26` and `Product=firstsmoke`. Set `DRY_RUN=1` to see the
commands without running them.

| Script | What it does |
|---|---|
| `s3.sh` | The artefact bucket: versioned, private, AES256, 30-day object expiry |
| `ecr.sh <product>` | Create or reuse a private ECR repo, log in, buildx, push |
| `apprunner.sh <product>` | Create or update the App Runner service from ECR, wait, print the URL |
| `common.sh` | Shared helpers and the account check |

```bash
infra/s3.sh                                   # once
infra/ecr.sh firstsmoke --context . --dockerfile products/firstsmoke/Dockerfile
infra/apprunner.sh firstsmoke                 # prints https://<id>.awsapprunner.com
infra/apprunner.sh firstsmoke --status
```

App Runner is x86 only, so `ecr.sh` builds `linux/amd64` by default.
`apprunner.sh --delete` asks before deleting.
