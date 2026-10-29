# Contributing to GPC

## Setup

GPC uses [uv](https://docs.astral.sh/uv/) for dependencies and [just](https://just.systems/)
as a command runner. `just` is installed into the virtual environment as a dev dependency.

```bash
git clone https://github.com/alexander-brady/gpc.git
cd gpc
uv sync                   # install the package with the dev and docs groups
source .venv/bin/activate # put just, ruff, pyrefly and pytest on your PATH
```

If you'd rather not activate the environment, prefix commands with `uv run`, e.g.
`uv run just test`.

## Commands

Every routine task has a `just` recipe. Run `just` (or `just list`) to see them all:

| Command | What it does |
| ------- | ------------ |
| `just list` | List all recipes, grouped. |
| `just ruff` | Format and lint all files, applying automatic fixes. |
| `just typecheck` | Type check all files with Pyrefly (alias: `just pyrefly`). |
| `just validate` | Check formatting, linting and types without changing anything. |
| `just test` | Run the full test suite with coverage. Extra arguments go to pytest. |
| `just test-one <target>` | Run specific tests with coverage, e.g. `just test-one tests/controllers`. |
| `just coverage` | Run the tests and write an HTML coverage report to `htmlcov/`. |
| `just docs` | Serve the docs locally with live reload. |
| `just docs-build` | Build the docs into `site/`. |
| `just clean` | Delete bytecode and cache files. |

## CI checks

GitHub Actions runs these on every pull request and push to `main`:

| Workflow | Runs when | Checks |
| -------- | --------- | ------ |
| **CI** | `gpc/`, `tests/`, `pyproject.toml` or `uv.lock` change | `just validate` (format, lint, type check) and `just test` |
| **Docs** | Pushes to `main` that touch the docs or `gpc/` | Builds the site and deploys it to GitHub Pages |
| **Publish** | A `v*` tag is pushed | Builds the package and publishes it to PyPI |

To catch failures before pushing, run the same checks locally:

```bash
just validate
just test
```

CI installs dependencies with `uv sync --locked`, which fails if `uv.lock` is out of date.
After changing dependencies in `pyproject.toml`, run `uv lock` and commit the updated lock file.

### Releasing

1. Bump `version` in `pyproject.toml` (e.g. with `uv version --bump minor`) and merge to `main`.
2. Tag the merge commit as `v<version>` and push the tag.

Publishing fails if the tag isn't on `main` or doesn't match the version in `pyproject.toml`.

## Testing

Tests live in `tests/` and mirror the package layout: tests for `gpc/controllers/rank.py`
go in `tests/controllers/test_rank.py`.

- **Use the shared fakes.** `tests/fakes.py` provides minimal, deterministic implementations
  of the abstract components (`FakePolicy`, `FakeWorldModel`, `SumObjective`), and
  `tests/conftest.py` exposes them as the `policy`, `world_model` and `objective` fixtures.
  Prefer these over defining new fakes in a test file.
- **Keep tests deterministic.** The fakes return predictable outputs, so assert exact values
  (`torch.testing.assert_close`) rather than just shapes where you can.
- **Test the error paths.** Validate inputs with clear `ValueError`s, and cover them with
  `pytest.raises(..., match=...)`.
- **Keep coverage up.** `just test` prints the lines each change leaves uncovered.

## Tensor shapes

GPC annotates tensor shapes in its type signatures using
[Pyrefly's tensor shape support](https://pyrefly.org/en/docs/tensor-shapes/), so
`just typecheck` catches shape mismatches before anything runs. The shape primitives come
from the `shape_extensions` package (installed with `pyrefly-torch-stubs`):

```python
from __future__ import annotations

from shape_extensions import IntVar
from torch import Tensor


def first_frame[B: IntVar, K: IntVar](
    frames: Tensor[[B, K, int, int, int]],
) -> Tensor[[B, int, int, int]]:
    return frames[:, 0]
```

- Declare dimensions as type parameters bound to `IntVar` (or `IntTuple` for a variable number
  of dimensions, spliced in with `Elements`). Use plain `int` for dimensions you don't track.
- Add `from __future__ import annotations` to modules that use shape annotations, so they're
  never evaluated at runtime.
- Use the letters from the docs' [tensor conventions](docs/quickstart/index.md#tensor-conventions)
  (`B`, `K`, `T`, `N`, `A`, ...) so signatures read consistently across the codebase.
- Also document shapes in docstrings, e.g. `frames (Tensor[B, K, C, H, W]): ...`, since the
  rendered API reference is where most readers will see them.

See the Pyrefly [getting started guide](https://pyrefly.org/en/docs/tensor-shapes-setup/) and
[API reference](https://pyrefly.org/en/docs/tensor-shapes-reference/) for the full syntax.

## Documentation

The docs are built with [Zensical](https://zensical.org/). Preview them with `just docs`.

- **API reference:** generated automatically, one page per module, from docstrings. Write
  [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
  for every public module, class and function. A new module gets its page on the next build.
  A new **subpackage** must also be added to the package pattern under
  `[project.plugins.api-autonav.module_options]` in `zensical.toml`, so its page doesn't
  repeat its re-exported members.
- **Guides:** hand-written pages live in `docs/` and are listed in the `nav` in `zensical.toml`.
  Link to API objects by their import path, e.g. `` [`GPCRank`][gpc.GPCRank] ``.
- **Code examples** in the quickstart should run as written, in page order. If you change an
  API they use, update them too.

## Best practices

- **Branch and open a pull request** rather than committing to `main`. Keep each PR focused on
  one change.
- **Use [Conventional Commits](https://www.conventionalcommits.org/)** prefixes, as in the
  existing history: `feat:`, `fix:`, `docs:`, `tests:`, `build:`, `refactor:`.
- **Run `just ruff` before committing.** Ruff handles formatting and import order.
- **Keep the public API explicit.** Export public names through the package `__init__.py` and
  its `__all__`.
- **Match the surrounding code.** Batch-first tensors, keyword-only options and validation of
  constructor arguments are the norm. Follow the existing patterns.
