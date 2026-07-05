# symbolic-fem-workbench

A teaching-first **symbolic finite element** workbench built on
[SymPy](https://www.sympy.org), for the Introduction to FEM course.

It derives element matrices, weak forms, quadrature rules, and elasticity
operators *symbolically* — so students see the exact expressions (shape
functions, `B`-matrices, stiffness `BᵀDB`, load vectors) rather than only
numbers. Import name: `symbolic_fem_workbench`.

## Install

From the monorepo root (installs the whole workspace):

```bash
uv sync --all-groups
```

Or install just this package:

```bash
pip install -e packages/symbolic-fem-workbench
```

Dependencies (all mandatory, Python 3.11+): `sympy`, `numpy`, `matplotlib`,
`scipy`, `pyyaml`, plus the notebook stack `jupyter`, `ipywidgets`, `ipykernel`
so the companion notebooks run without any extra install step.

## Quick start

```python
import symbolic_fem_workbench as sfw

# Symbolic P1 stiffness on the reference triangle
problem = sfw.build_poisson_triangle_p1_local_problem()
print(problem)          # element stiffness matrix as SymPy expressions

# 2D isotropic elasticity constitutive matrix (plane stress)
D = sfw.plane_stress_D()
```

## Companion notebooks

`notebooks/` holds 14 Jupyter notebooks (`01_…`–`14_…`) covering strong→weak
form, Galerkin discretisation, 1D/2D/3D element stiffness, quadrature,
assembly, boundary conditions, elasticity, time-dependent FEM, and error
analysis.

```bash
uv run jupyter lab packages/symbolic-fem-workbench/notebooks
```

A few notebooks additionally use the **FEniCSx** stack (`fenics-dolfinx`,
`mpich`, `pyvista`), which is conda-only — install it from `conda-forge` if you
want to run those cells. The core symbolic notebooks need only this package.

> Quarto scaffolding from the source repo (`_quarto.yml`, `.quarto/`,
> `_freeze/`, `_output/`, `index.qmd`, and `*_files/` render artifacts) was
> intentionally **not** migrated — these are plain `.ipynb` notebooks.

## Course notes

- [Intro-FEM Lecture Notes (PDF)](docs/Intro-FEM-Lecture-Notes.pdf) — the
  compiled course book that these notebooks accompany.

## Tests

```bash
uv run pytest packages/symbolic-fem-workbench
```
