# Contributing

## Before writing code

Open an issue describing the change. The decision path is the part of this package
where behaviour is constrained by design, and a patch that blurs the separation
between prediction and decision will be refused regardless of its quality.

Rules that are not negotiable:

- A learned component never decides admissibility. It produces an estimate that the
  deterministic operator consumes.
- Missing evidence is reported as missing. It is never coerced into a satisfied
  constraint or into a default value.
- Confidence is reported as unavailable when the versioned predictor provides none.
- Any change to the decision path carries a regression test that fails without it.

## Standards

```bash
pytest
ruff check .
mypy src
```

All three must pass. Public functions carry type annotations. Contracts are immutable
Pydantic models.

## Contributor Licence Agreement

This project is released under the AGPL-3.0 and is also offered under a separate
commercial licence. Dual licensing is only possible if a single party holds the rights
needed to grant both. Every contribution therefore requires a signed Contributor
Licence Agreement before it can be merged. See `CLA.md`.

The agreement does not transfer ownership of the contribution. It grants the project
maintainer the rights required to distribute it under both licences, and the
contributor keeps the right to use their own contribution however they wish.

To sign, send the completed agreement to houcinemessad@gmail.com and reference the
pull request.

## Commits

One logical change per commit. The message states what changes and why, not which
files were touched.
