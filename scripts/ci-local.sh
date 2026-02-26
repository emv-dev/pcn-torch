#!/usr/bin/env bash
# ci-local.sh -- Run the same CI checks locally that GitHub Actions runs.
# Mirrors: .github/workflows/ci.yml (lint, typecheck, test)
#
# Usage:
#   bash scripts/ci-local.sh          # Run all checks (lint, typecheck, test)
#   bash scripts/ci-local.sh --fix    # Auto-fix lint issues, then exit
#   bash scripts/ci-local.sh --help   # Show this help message

set -euo pipefail

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

step() {
    echo ""
    echo "=============================="
    echo " STEP: $1"
    echo "=============================="
    echo ""
}

usage() {
    echo "Usage: bash scripts/ci-local.sh [OPTIONS]"
    echo ""
    echo "Run the same CI checks locally that GitHub Actions runs."
    echo "Mirrors .github/workflows/ci.yml (lint, typecheck, test)."
    echo ""
    echo "Options:"
    echo "  --fix     Auto-fix lint issues (ruff check --fix + ruff format), then exit"
    echo "  --help    Show this help message"
    echo ""
    echo "Steps (in order):"
    echo "  1. Lint      - uv run ruff check . && uv run ruff format --check ."
    echo "  2. Typecheck - uv run mypy src/pcn_torch"
    echo "  3. Test      - uv run pytest tests/ -v --cov=pcn_torch --cov-report=term-missing"
}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

if [[ "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ "${1:-}" == "--fix" ]]; then
    step "Auto-fix lint (ruff check --fix)"
    uv run ruff check --fix .

    step "Auto-format (ruff format)"
    uv run ruff format .

    echo ""
    echo "=============================="
    echo " Lint fixes applied!"
    echo "=============================="
    exit 0
fi

# ---------------------------------------------------------------------------
# Sync dependencies (matches CI: each job runs uv sync --dev)
# ---------------------------------------------------------------------------

step "Syncing dependencies (uv sync --dev)"
uv sync --dev

# ---------------------------------------------------------------------------
# 1. Lint (matches CI: lint job)
# ---------------------------------------------------------------------------

step "Lint (ruff check)"
uv run ruff check .

step "Lint (ruff format --check)"
uv run ruff format --check .

# ---------------------------------------------------------------------------
# 2. Typecheck (matches CI: typecheck job)
# ---------------------------------------------------------------------------

step "Typecheck (mypy)"
uv run mypy src/pcn_torch

# ---------------------------------------------------------------------------
# 3. Test (matches CI: test job)
# ---------------------------------------------------------------------------

step "Test (pytest)"
uv run pytest tests/ -v --cov=pcn_torch --cov-report=term-missing

# ---------------------------------------------------------------------------
# Success
# ---------------------------------------------------------------------------

echo ""
echo "=============================="
echo " All CI checks passed!"
echo "=============================="
