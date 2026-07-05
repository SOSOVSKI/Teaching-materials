"""Display and pretty-printing utilities for Jupyter notebooks."""

from __future__ import annotations

import sympy as sp

from constkit.tensor2 import Tensor2


def show(obj, name: str = "", simplify: bool = True) -> None:
    """Pretty-print a tensor or expression.

    In Jupyter notebooks, renders LaTeX. In a terminal, prints
    SymPy's pretty-printed form.

    Parameters
    ----------
    obj : Tensor2, sp.Matrix, sp.Expr, or sp.Array
        The object to display.
    name : str, optional
        Label to show before the expression (e.g., 'E', 'σ').
    simplify : bool
        Whether to apply sp.simplify before display.
    """
    if isinstance(obj, Tensor2):
        mat = obj.matrix
        label = name or obj.name
    elif isinstance(obj, sp.Matrix):
        mat = obj
        label = name
    elif isinstance(obj, sp.Array):
        _show_array(obj, name)
        return
    else:
        # Scalar expression
        expr = sp.simplify(obj) if simplify else obj
        if name:
            _display_latex(f"{name} = {sp.latex(expr)}")
        else:
            _display_latex(sp.latex(expr))
        return

    if simplify:
        mat = sp.simplify(mat)

    if label:
        _display_latex(f"{label} = {sp.latex(mat)}")
    else:
        _display_latex(sp.latex(mat))


def _display_latex(latex_str: str) -> None:
    """Display LaTeX in Jupyter or fall back to terminal print."""
    try:
        from IPython.display import Math, display

        display(Math(latex_str))
    except ImportError:
        # Not in Jupyter — use SymPy's pprint
        print(f"  {latex_str}")


def _show_array(arr: sp.Array, name: str) -> None:
    """Display a higher-order SymPy Array (e.g., 4th-order tensor)."""
    if name:
        print(f"{name}:")
    sp.pprint(arr)


def compare(A: Tensor2, B: Tensor2, names: tuple[str, str] = ("A", "B")) -> None:
    """Display two tensors side by side and their difference.

    Useful for verifying that two expressions are equal (difference = 0).
    """
    show(A, name=names[0])
    show(B, name=names[1])
    diff = A - B
    diff_simplified = sp.simplify(diff.matrix)
    is_zero = diff_simplified == sp.zeros(3, 3)
    show(Tensor2(diff_simplified), name=f"{names[0]} − {names[1]}")
    if is_zero:
        print("  ✓ Equal (difference is zero)")
    else:
        print("  ✗ Not equal")
