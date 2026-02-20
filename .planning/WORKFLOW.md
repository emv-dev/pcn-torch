# Workflow: pcn-torch

## Post-Phase Completion: GitHub Merge

After a phase is executed and verified, merge it into main via PR.

### Steps

1. **Push feature branch**
   ```bash
   git push -u origin ai/feature/<branch-name>
   ```

2. **Create PR** with:
   - Title: `Phase N: <Name> — <short description>`
   - Body: Summary, verification table, test plan (see template below)

3. **Wait for CI** — lint, typecheck, and test jobs must pass (enforced by ruleset)

4. **Merge PR** — squash merge into main

5. **Delete remote branch**
   ```bash
   git push origin --delete ai/feature/<branch-name>
   ```

6. **Start next phase** on a new branch
   ```bash
   git checkout main && git pull
   git checkout -b ai/feature/<next-branch-name>
   ```

### PR Body Template

```markdown
## Summary

- <bullet points of what was built>

## Verification

| Criterion | Status |
|-----------|--------|
| <criterion 1> | <pass/fail> |
| <criterion 2> | <pass/fail> |

Score: **N/N must-haves verified**

## Test plan

- [x] <test 1>
- [x] <test 2>
```

### Branch Naming

- Feature branches: `ai/feature/<feature-name>`
- Bugfix branches: `ai/bugfix/<bug-name>`

### Branch Protection (GitHub Ruleset on main)

- No direct pushes — PRs required
- No force pushes
- Status checks must pass: `lint`, `typecheck`, `test`
- Linear history required (squash merge)

---
*Created: 2026-02-20*
