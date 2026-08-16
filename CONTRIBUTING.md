# Contributing to Watershed Rainfall Analyzer

Thank you for your interest in contributing! Please follow these steps:

## 1. Fork and Branch

Fork the repository and create a feature branch from `main`:

```bash
git checkout -b feature/your-feature-name
```

## 2. Set Up the Development Environment

```bash
pip install -e ".[dev]"
pre-commit install
```

## 3. Write Tests

Add tests in the `tests/` directory that cover your changes. Ensure all tests pass.

## 4. Run Checks

Before submitting, verify everything passes:

```bash
ruff check .
mypy watershed_analyzer
pytest
```

## 5. Submit a Pull Request

Push your branch and open a PR against `main`. Include a clear description of the change.
