# Security policy

## Supported version

Security fixes target the current `0.2.x` development line.

## Reporting

Do not open a public issue containing credentials, private memory, source code,
or an exploitable path. Use GitHub's private vulnerability reporting for this
repository. Include the affected revision, platform, reproduction steps, and a
minimal redacted example.

## Trust boundaries

- The repository scanner never executes analyzed source.
- Selected project roots are resolved, traversal is blocked, and outward
  symlinks are rejected.
- VCS, dependency, cache, build, hidden, secret-like, binary, oversized, and
  non-UTF-8 files are skipped by default.
- Kernel requests and responses are bounded and schema validated.
- Reserved memory prefixes and mismatched structured-record keys/layers are
  rejected.
- LTM snapshots are bounded, checksummed, validated, and reject symlink files.
- Model output is untrusted. Routing mutations require explicit user intent,
  and review findings require current source hashes, valid ranges, and optional
  symbol grounding.
- Raw model output and complete memory values are absent from normal logs.

`memory.delete` targets one canonical key. TST v0.2 does not expose destructive
bulk deletion through model routing; any future bulk operation must require
explicit confirmation outside model-generated output.

TST is not a sandbox for running untrusted code and does not provide multi-user
authentication, remote isolation, or cloud-provider security controls.
