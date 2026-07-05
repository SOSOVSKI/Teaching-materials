# Teaching materials

A [uv](https://docs.astral.sh/uv/)-managed monorepo collecting the helper
packages I write for the courses I teach. Each course's toolkit lives as its own
installable package under `packages/`, but they share one lockfile, one virtual
environment, and one test command.

## Packages

| Package | Course | What it does |
|---------|--------|--------------|
| [`constkit`](packages/constkit/) | Constitutive Modeling | Symbolic tensor algebra & kinematics toolkit (SymPy) for verifying derivations from L01–L03 and HW1. |
| [`symbolic-fem-workbench`](packages/symbolic-fem-workbench/) | Introduction to FEM | Symbolic finite-element workbench (SymPy): element matrices, weak forms, quadrature, elasticity — plus 14 companion notebooks. |

_More courses will be added here over time._

## Layout

```
Teaching-materials/
├── pyproject.toml          # workspace root (virtual — aggregates members)
├── uv.lock                 # single lockfile for the whole monorepo
├── packages/
│   ├── constkit/
│   │   ├── pyproject.toml   # package metadata (hatchling, src layout)
│   │   ├── README.md
│   │   ├── src/constkit/    # importable source
│   │   ├── tests/           # pytest suite
│   │   └── docs/            # USER_MANUAL.md, API_REFERENCE.md
│   └── symbolic-fem-workbench/
│       ├── pyproject.toml
│       ├── README.md
│       ├── src/symbolic_fem_workbench/
│       ├── tests/
│       └── notebooks/       # 14 companion .ipynb (no quarto scaffolding)
└── ...
```

## Getting started

```bash
uv sync --all-groups     # create .venv and install every package + dev tools
uv run pytest            # run all packages' test suites
```

## Adding a new course package

1. Scaffold a member under `packages/`:

   ```bash
   mkdir -p packages/<name>/src/<name> packages/<name>/{tests,docs}
   ```

2. Give it a `packages/<name>/pyproject.toml` with a `[build-system]`
   (hatchling) and the `src` layout:

   ```toml
   [tool.hatch.build.targets.wheel]
   packages = ["src/<name>"]
   ```

3. Register it with the workspace root's `pyproject.toml` — the
   `members = ["packages/*"]` glob picks it up automatically. Add it to the
   root `dependencies` (and `[tool.uv.sources]` with `{ workspace = true }`)
   if you want `uv sync` to install it into the shared environment.

4. Re-lock and sync:

   ```bash
   uv lock && uv sync --all-groups
   ```
