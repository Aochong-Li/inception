#!/bin/bash
#
# Test runner script for StrongREJECT integration tests
#
# Usage:
#   ./tests/run_tests.sh                  # Run unit tests only
#   ./tests/run_tests.sh --integration    # Run integration tests only
#   ./tests/run_tests.sh --all            # Run all tests
#

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Activate virtual environment
if [ ! -f ".venv/bin/activate" ]; then
    echo "Error: Virtual environment not found at .venv/"
    echo "Please run: uv venv --python 3.12 .venv"
    exit 1
fi

source .venv/bin/activate

# Check if pytest is installed
if ! python -m pytest --version &> /dev/null; then
    echo "Installing pytest..."
    uv pip install pytest
fi

# Parse arguments
MODE="${1:---unit}"

case "$MODE" in
    --unit)
        echo "Running unit tests (no API calls)..."
        python -m pytest tests/test_strongreject_integration.py -v -m "not integration"
        ;;
    --integration)
        echo "Running integration tests (real API calls)..."
        if [ -z "$OPENAI_API_KEY" ]; then
            echo "Warning: OPENAI_API_KEY not set. Integration tests will be skipped."
        fi
        python -m pytest tests/test_strongreject_integration.py -v -m integration
        ;;
    --all)
        echo "Running all tests..."
        python -m pytest tests/test_strongreject_integration.py -v
        ;;
    *)
        echo "Usage: $0 [--unit|--integration|--all]"
        echo "  --unit         Run unit tests only (default)"
        echo "  --integration  Run integration tests only"
        echo "  --all          Run all tests"
        exit 1
        ;;
esac
