# Contributing

Thanks for your interest in contributing to mflux-experiment!

## Getting Started

1. Fork the repository.
2. Run `make setup-dev` to install all dependencies and pre-commit hooks.
3. Create a feature branch: `git checkout -b feat/my-thing`.

## Development Workflow

```bash
# One-time setup (deps + pre-commit hooks)
make setup-dev

# Run all quality checks before pushing
make check
```

This runs:
- **Python**: Ruff lint, Ruff format check, Pyright type checking (`inference-service/`)
- **Frontend**: TypeScript type-check (`web-frontend/`)
- **C#**: `dotnet build --configuration Release` (`web-api/`)
- **Shell**: shellcheck on `start.sh` and `shutdown.sh`

## Pre-commit Hooks

Installed via `make setup-dev`. They run automatically on `git commit`:

- **Pre-commit**: trailing-whitespace, EOF fixer, YAML validation, large-file guard, merge-conflict detection, private-key detection, shellcheck, ruff lint + format
- **Pre-push**: all of the above + TruffleHog secret scanning

You can also run them manually:

```bash
pre-commit run --all-files       # all hooks on all files
pre-commit run ruff              # a single hook
```

## Making Changes

- Keep PRs focused on a single concern.
- Add or update tests where applicable.
- If you change the inference service API contract, update `shared/API_CONTRACT.md`.
- If you add a Makefile target, add a comment so `make` stays self-documenting.

## Project Structure

```
├── inference-service/    # FastAPI + mflux (MLX-native Flux)
│   └── app/
├── web-api/              # ASP.NET Core 9 API
│   ├── Controllers/
│   ├── Services/
│   └── Models/
├── web-frontend/         # React 19 + Vite + TypeScript
│   └── src/
└── shared/               # Cross-service contracts
    └── API_CONTRACT.md
```

## Commit Style

Conventional commits are appreciated but not enforced. Keep subject lines under 72 characters.

## Need Help?

Open a [discussion](https://github.com/stoyanov-x/mflux-playground/discussions) or an issue.
