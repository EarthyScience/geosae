# Development tasks: `make dev` installs the package with its development tools and the git hooks.

.PHONY: dev lint test docs check

dev:
	pip install -e ".[dev,docs]"
	pre-commit install

lint:
	pre-commit run --all-files

test:
	pytest -q

docs:
	mkdocs build --strict

# What CI enforces — run before pushing.
check: lint test docs
