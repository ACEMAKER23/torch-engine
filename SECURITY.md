# Security Policy

## Supported Versions

TorchEngine is a personal/portfolio project and does not currently publish versioned releases.

## Reporting a Vulnerability

Please **do not** open public issues for security-sensitive reports.

Instead, open a draft pull request with a minimal reproduction and a proposed fix **or** open a GitHub issue with only non-sensitive details and explicitly request a private follow-up in the comments.

If you are unsure whether something is security-sensitive, treat it as sensitive.

## Scope

Examples of issues to report:

- Memory safety vulnerabilities (e.g., out-of-bounds reads/writes, use-after-free)
- Unsafe deserialization or file handling
- Accidental credential leakage in the repository
- Build scripts that execute unexpected commands