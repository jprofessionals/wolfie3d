# Repository Guidelines

## Project Structure & Module Organization
- `src/wolfie3d/`: core package and entry points (`python -m wolfie3d`).
- `assets/`: game assets.
  - `assets/textures/`: wall textures (`bricks.png`, `stone.png`, `wood.png`, `metal.png`).
  - `assets/sprites/`: sprite images (e.g., `enemy.png`).
- `tests/`: pytest test suite (start new tests here).
- `docs/`: design notes and documentation.
- Tooling lives in `pyproject.toml` (ruff/black/mypy/pytest), env via `uv.lock`.

## Build, Test, and Development Commands
- Setup dev env: `uv sync --dev`.
- Run game: `uv run python -m wolfie3d` (no-enemies: `uv run python -m wolfie3d.game_no_enemies`).
- Lint: `uv run python -m ruff check .` (auto-fix: `uv run python -m ruff check . --fix`).
- Format: `uv run python -m black .`.
- Type check: `uv run python -m mypy src`.
- Tests: `uv run python -m pytest`.
- Coverage: `PYTHONPATH=src uv run python -m pytest --cov --cov-report=term-missing`.

## Coding Style & Naming Conventions
- Indentation: spaces, 4 spaces; max line length 100 (`.editorconfig`).
- Format with Black; lint with Ruff (rules: E,F,I,UP,B,C90; target `py310`).
- Imports: sorted; first‑party is `wolfie3d`.
- Naming: `snake_case` for modules/functions/variables, `PascalCase` for classes.
- Type hints required for new code (`mypy` strict-ish; `disallow_untyped_defs=true`).

## Testing Guidelines
- Framework: `pytest`; place tests in `tests/` as `test_*.py`.
- Keep tests fast and headless; do not require a display.
- Cover new logic added in `src/`; use fixtures for assets when possible.
- Aim for meaningful coverage; branch coverage is enabled in configuration.

## Commit & Pull Request Guidelines
- Commits: follow Conventional Commits (e.g., `feat:`, `fix:`, `chore:`, `docs:`). Examples in history: `feat(junie): ...`, `chore: ...`.
- Branches: `feat/<short-topic>`, `fix/<issue-id>`, `chore/<task>`.
- PRs: include a clear description, linked issues, and (when relevant) a short screenshot/GIF of the running game.
- CI/Local checklist: run Ruff, Black, Mypy, Pytest (and coverage) before requesting review.

## Security & Configuration Tips
- Ensure required textures exist under `assets/textures/`; optional sprites under `assets/sprites/`.
- Running the game uses an OpenGL 3.3 core context via Pygame; CI should avoid launching the windowed app.
