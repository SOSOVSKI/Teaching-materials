# constkit

Symbolic tensor algebra toolkit for the **Constitutive Modeling** course.

`constkit` lets students build, manipulate, and verify tensor expressions using
the same notation as the lecture notes. Every operation is fully symbolic
(powered by [SymPy](https://www.sympy.org)): you work with parameters like
`k`, `λ`, and `θ`, see LaTeX-rendered results in Jupyter, and substitute
numbers only when you want to check a specific case.

## Install

From the monorepo root (recommended — installs the whole workspace):

```bash
uv sync --all-groups
```

Or install just this package into any environment:

```bash
pip install -e packages/constkit          # editable, from the repo
# or, with optional numeric evaluation extras:
pip install -e "packages/constkit[numeric]"
```

The only hard dependency is `sympy>=1.12` (Python 3.10+).

## Quick start

```python
import constkit as ck
import sympy as sp

k = sp.Symbol("k", positive=True)
F = ck.deformation_gradient("simple_shear", k=k)
E = ck.green_lagrange(F)
ck.show(E, name="E")
```

## Documentation

- [User Manual](docs/USER_MANUAL.md) — tutorial-style guide, worked examples,
  and the full HW1 walkthrough.
- [API Reference](docs/API_REFERENCE.md) — every public function, class, and
  method.
- [Course notes (PDF)](docs/Constitutive-Models-in-Solid-Mechanics.pdf) — the
  full *Constitutive Models in Solid Mechanics* lecture book.

## Tests

```bash
uv run pytest packages/constkit          # from repo root
```

The suite doubles as a specification: HW1 problems 1–5 are exercised
symbolically and numerically, and `test_api_completeness.py` guarantees the
API reference stays in sync with the code.
