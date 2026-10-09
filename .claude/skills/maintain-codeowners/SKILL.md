---
name: maintain-codeowners
description: Audit or extend .github/CODEOWNERS, the human-review gate for sensitive paths. Use when asked whether a path should be gated, when auditing the list for stale or missing entries, or after a refactor moves code that a CODEOWNERS comment describes.
allowed-tools: Bash, Read, Edit, Grep, Glob, Task
---

# Maintain CODEOWNERS

`.github/CODEOWNERS` lists the paths where a PR needs one approval from
`@arthur-ai/arthur-engineers`. Everything else is left to the automated
reviewer. The file only works if every line is still true and the list stays
short enough that ordinary PRs are not slowed down. This skill keeps it that
way.

The same skill exists in arthur-engine, arthur-scope and unify-frontend; the
bar is shared, the details below are this repo's.

The gate only takes effect when "Require review from Code Owners" is on for
`main-ruleset`. Check it first and say so in the report if it is off; the rest
of the audit is moot until it is on:

```bash
gh api repos/arthur-ai/arthur-common/rules/branches/main \
  -q '.[] | select(.type=="pull_request") | .parameters.require_code_owner_review'
```

## Why a library needs its own gate

arthur-common is published to PyPI and imported by genai-engine, ml-engine and
arthur-scope. A change here ships to every consumer on their next dependency
bump, usually through a Renovate PR that nobody reads line by line. So the
question for each path is what it decides *in the consumers*: genai-engine's
`create_api_key` checks requested roles against `APIKeysRolesEnum`, and
`ToxicityConfig` supplies the default threshold stored on a guardrail rule.

## The bar for gating a path

A path belongs in CODEOWNERS only if **all three** hold:

1. **A mistake there is one of these**, matching a section of the file:
   - **secrets / config**: a credential (customer or ours) could be logged,
     sent to a different host, stored unencrypted, or sent over unverified TLS.
     Tests still pass; the customer has to rotate the secret. Not reversible
     with a revert.
   - **deploy / infra**: code or config ships inside the published wheel in a
     way that changes what consumers run, or is run by customers directly.
   - **ci/cd**: runs in a CI job that holds a protected-branch secret or the
     automator app token, or decides what goes into the published wheel
     (version, build config, release trigger).
   - **auth**: role and permission definitions, password policy, API-key role
     defaults, and the tests that pin them, as consumed by genai-engine and
     arthur-scope.
   - **guardrails**: changes guardrail results or rule defaults, so it needs
     evaluation and benchmarking before shipping.
   - **migrations**: schema changes. This repo has none; its models are data
     contracts, not database tables.
2. **It changes rarely.** Count commits over the last six months. If the file
   is edited in most feature PRs for its area, a gate there is drag, not
   safety. `agent_governance_schemas.py` (25 commits in six months) is left
   ungated for this reason.
3. **CODEOWNERS is the right fix.** If the risk is "a check only runs in
   pre-commit" or "a workflow runs PR code with secrets", the fix is a CI step
   or a workflow change. If the risk is "nobody tests that X stays the same",
   the fix is a test. Report those separately; do not gate a file to paper over
   them.

Two things that fail the bar on their own:

- **Renovate edits it.** Renovate's only manager is `pep621` with
  `includePaths` `**/pyproject.toml` and `**/uv.lock`. Gating either stops
  automerge for every dependency bump, because `renovate-auto-approve.yml`'s
  github-actions[bot] approval does not count as a code-owner approval.
  `pyproject.toml` also holds `[build-system]` and the published version; the
  right control for those is a CI check on the built wheel, not a gate.
- **Any file could do the same thing.** `aggregations/functions/__init__.py`
  imports every module in the folder, so gating one aggregation does nothing;
  pin metric ids and names with a test instead.

## Audit procedure

Run this when asked to audit, after a large refactor, or roughly quarterly.
Work against `origin/main` (`git fetch origin main` first). Report facts with
file paths and line numbers; a finding without a verified line is a guess.

### 1. Syntax and owner resolution

```bash
gh api "repos/arthur-ai/arthur-common/codeowners/errors?ref=<branch>"
```

Must return `[]`. `Unknown owner` on every line means the team is secret;
GitHub ignores secret teams.

### 2. Every pattern still matches a tracked file

A pattern that matches nothing is a path that moved. Find where it went and
re-point the line.

```bash
git ls-files > /tmp/tracked.txt
grep -E '^/' .github/CODEOWNERS | awk '{print $1}' | sed 's#^/##' | while read -r p; do
  case "$p" in
    */\*\*) n=$(grep -c "^${p%/**}/" /tmp/tracked.txt) ;;
    *)      n=$(grep -cx "$p" /tmp/tracked.txt) ;;
  esac
  [ "$n" = "0" ] && echo "NO MATCH: /$p"
done
```

### 3. Every comment is still true

Each `#` comment above a line states *why* that path is gated, as a factual
claim about the code ("NewApiKeyRequest's default role", "mint a token from
ARTHUR_GH_AUTOMATOR_APP_PRIVATE_KEY"). Code moves and the comment stays. For
each comment, grep for the thing it names and confirm it is still in that
file. When it has moved, gate the new location and fix the comment. Also
grep genai-engine and arthur-scope for the consumer the comment names: if
they stopped importing it, the line may no longer meet the bar.

### 4. Sweep for new candidates, per section

Use these searches as a starting point, then apply the bar to each hit.

| Section | Search |
|---|---|
| secrets / config | `secret`, `credential`, `password`, `api_key`, `token`, `verify_ssl`, `ssl`, `redact`; any model field that carries a credential or decides whether one is masked |
| deploy / infra | `[build-system]`, `[tool.hatch.build]` and other packaging config that changes the wheel's contents (report, do not gate: Renovate edits `pyproject.toml`) |
| ci/cd | workflow steps with `secrets.` in `env:` or `with:`, `id-token: write`, `environment:`, `create-github-app-token`; release triggers; `.bumpversion.cfg` |
| auth | `Role`, `Permission`, `APIKeysRolesEnum`, `UserPermission*`, password policy, default roles on request models; for each hit, find the consumer in genai-engine or arthur-scope that makes an access decision with it |
| guardrails | `RuleType`, `PIIEntityTypes`, `*Config` validators and `DEFAULT_*_THRESHOLD` constants that genai-engine stores on rules |
| migrations | none expected; say so |

For each candidate, get the churn and who makes it:

```bash
git log --oneline --since="6 months ago" origin/main -- <path> | wc -l
git log --format=%an --since="6 months ago" origin/main -- <path> | sort | uniq -c
```

If `git rev-parse --is-shallow-repository` prints `true`, the counts are
truncated to the clone's history; say so in the report.

### 5. Write the report

A table per outcome. Keep entries carry the path, the section, the one-line
reason tied to criterion 1, and the churn. Drop entries say which criterion
failed and, where the original worry was real but mis-aimed, what the right
fix is (a test, a CI step, a workflow change). Include dropped candidates in
the report: the reasoning is what stops the same idea coming back next audit.

## Editing the file

- Patterns start with `/` (anchored to the repo root). The owner starts at
  column 67; pad the path with spaces to column 66. A path longer than that
  gets two spaces.
- Put the line in the matching `# --- section ---`.
- Every added line gets a comment above it stating *why*, as a claim the next
  audit can verify by grep (name the function, enum, env var, token, or
  consumer that makes it sensitive). "Sensitive" on its own is not a reason.
- One line per file, not a glob over a directory, unless the whole directory
  meets the bar and changes rarely.
- Open the PR against `main`. The PR body lists each added path with its
  reason and churn, and each considered-and-dropped path with why. Editing
  `.github/**` is itself gated, so a human reviews the change.

## Verifying a change

1. `gh api "repos/arthur-ai/arthur-common/codeowners/errors?ref=<branch>"`
   returns `[]`.
2. The step-2 script prints no `NO MATCH`.
3. After merge, with the ruleset on: a throwaway PR touching a newly gated
   file gets a code-owner review request; a Renovate PR touching
   `pyproject.toml` or `uv.lock` does not.
