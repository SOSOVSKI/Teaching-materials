"""Symbolic second-order tensor in 3D.

Wraps a 3×3 SymPy Matrix with methods that mirror continuum-mechanics
lecture notation: ``trace()``, ``dev()``, ``I1()``, ``rotate(Q)``, etc.

Every method returns symbolic SymPy expressions. For numeric evaluation,
call ``.subs()`` or ``.evaluate()``.
"""

from __future__ import annotations

from typing import Any, Callable

import sympy as sp


class Tensor2:
    """A symbolic second-order tensor in 3D Cartesian coordinates.

    Parameters
    ----------
    matrix : sp.Matrix or array-like
        3×3 matrix of components.
    name : str, optional
        Display name (used in LaTeX rendering).

    Examples
    --------
    >>> import sympy as sp
    >>> A = Tensor2([[1, 2, 0], [2, 3, 1], [0, 1, 4]], name='A')
    >>> A.trace()
    8
    >>> A.I1()
    8
    """

    def __init__(self, matrix: Any, name: str = "") -> None:
        if isinstance(matrix, sp.Matrix):
            self._m = matrix
        else:
            self._m = sp.Matrix(matrix)
        if self._m.shape != (3, 3):
            raise ValueError(
                f"Tensor2 requires a 3×3 matrix, got {self._m.shape}"
            )
        self._name = name

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Display name used in LaTeX rendering. Mutable."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def matrix(self) -> sp.Matrix:
        """Underlying 3×3 SymPy Matrix."""
        return self._m

    def __getitem__(self, idx: tuple[int, int]) -> sp.Expr:
        return self._m[idx[0], idx[1]]

    # ------------------------------------------------------------------
    # Core tensor operations
    # ------------------------------------------------------------------

    @property
    def T(self) -> Tensor2:
        """Transpose: A^T."""
        return Tensor2(self._m.T, name=f"{self.name}^T" if self.name else "")

    def trace(self) -> sp.Expr:
        """Trace: tr(A) = A_ii."""
        return self._m.trace()

    def det(self) -> sp.Expr:
        """Determinant: det(A)."""
        return self._m.det()

    def inv(self) -> Tensor2:
        """Inverse: A^{-1}."""
        return Tensor2(
            self._m.inv(),
            name=f"{self.name}^{{-1}}" if self.name else "",
        )

    def sym(self) -> Tensor2:
        """Symmetric part: sym(A) = ½(A + A^T)."""
        return Tensor2(
            sp.Rational(1, 2) * (self._m + self._m.T),
            name=f"sym({self.name})" if self.name else "",
        )

    def skew(self) -> Tensor2:
        """Skew-symmetric part: skew(A) = ½(A − A^T)."""
        return Tensor2(
            sp.Rational(1, 2) * (self._m - self._m.T),
            name=f"skew({self.name})" if self.name else "",
        )

    def dev(self) -> Tensor2:
        """Deviatoric part: dev(A) = A − ⅓ tr(A) I."""
        return Tensor2(
            self._m - sp.Rational(1, 3) * self._m.trace() * sp.eye(3),
            name=f"dev({self.name})" if self.name else "",
        )

    # ------------------------------------------------------------------
    # Contractions
    # ------------------------------------------------------------------

    def dot(self, v: sp.Matrix) -> sp.Matrix:
        """Single contraction with a vector: A·v = A_ij v_j."""
        if isinstance(v, Tensor2):
            return Tensor2(self._m * v._m)
        return self._m * v

    def single_contract(self, B: Tensor2) -> Tensor2:
        """Tensor product (single contraction): (A·B)_ij = A_ik B_kj."""
        return Tensor2(
            self._m * B._m,
            name=f"{self.name}·{B.name}" if self.name and B.name else "",
        )

    def double_contract(self, B: Tensor2) -> sp.Expr:
        """Double contraction: A:B = A_ij B_ij.

        For symmetric tensors this equals the Frobenius inner product.
        """
        # Element-wise multiply and sum
        result = sp.Integer(0)
        for i in range(3):
            for j in range(3):
                result += self._m[i, j] * B._m[i, j]
        return result

    def outer(self, B: Tensor2) -> sp.Array:
        """Outer (dyadic) product: (A⊗B)_ijkl = A_ij B_kl.

        Returns a 4th-order SymPy Array with shape (3,3,3,3).
        """
        components = [
            [
                [[self._m[i, j] * B._m[k, l] for l in range(3)] for k in range(3)]
                for j in range(3)
            ]
            for i in range(3)
        ]
        return sp.Array(components)

    # ------------------------------------------------------------------
    # Principal invariants
    # ------------------------------------------------------------------

    def I1(self) -> sp.Expr:
        """First principal invariant: I₁ = tr(A)."""
        return self.trace()

    def I2(self) -> sp.Expr:
        """Second principal invariant: I₂ = ½[(tr A)² − tr(A²)]."""
        tr_A = self.trace()
        A2 = self._m * self._m
        tr_A2 = A2.trace()
        return sp.Rational(1, 2) * (tr_A**2 - tr_A2)

    def I3(self) -> sp.Expr:
        """Third principal invariant: I₃ = det(A)."""
        return self.det()

    # ------------------------------------------------------------------
    # Eigendecomposition and tensor functions
    # ------------------------------------------------------------------

    def eigendecomposition(self) -> dict:
        """Eigenvalues and eigenvectors of the tensor.

        Returns
        -------
        dict with keys:
            'eigenvalues' : list of sp.Expr
            'eigenvectors' : list of sp.Matrix (column vectors)

        Note: for fully symbolic tensors, eigenvalue expressions may be
        very long. Consider substituting numeric values first.
        """
        eig_data = self._m.eigenvects()
        eigenvalues = []
        eigenvectors = []
        for val, mult, vecs in eig_data:
            for _ in range(mult):
                eigenvalues.append(val)
            for v in vecs:
                eigenvectors.append(v.normalized())
        return {"eigenvalues": eigenvalues, "eigenvectors": eigenvectors}

    def spectral_function(self, f: Callable) -> Tensor2:
        """Apply a scalar function to a symmetric tensor spectrally.

        For a symmetric tensor with an orthonormal eigenbasis,

            f(A) = Σ_α f(λ_α) n_α ⊗ n_α

        Parameters
        ----------
        f : callable
            Scalar function to apply to each eigenvalue (e.g. ``sp.sqrt``).

        Returns
        -------
        Tensor2
            The tensor function f(A).

        Raises
        ------
        ValueError
            If the tensor is not symmetric. This orthonormal spectral
            representation is not valid for a general non-symmetric tensor.
        """
        symmetry_residual = sp.simplify(self._m - self._m.T)
        if symmetry_residual != sp.zeros(3, 3):
            raise ValueError(
                "spectral_function() requires a symmetric tensor. "
                "Use a general matrix-function construction for "
                "non-symmetric tensors."
            )
        eig = self.eigendecomposition()
        result = sp.zeros(3, 3)
        vals = eig["eigenvalues"]
        vecs = eig["eigenvectors"]
        for lam, n in zip(vals, vecs):
            n_col = sp.Matrix(n).reshape(3, 1)
            result += f(lam) * (n_col * n_col.T)
        return Tensor2(sp.simplify(result))

    def sqrt(self) -> Tensor2:
        """Square root of a symmetric tensor via spectral decomposition: √A."""
        return self.spectral_function(sp.sqrt)

    def log(self) -> Tensor2:
        """Logarithm of a symmetric tensor via spectral decomposition: ln(A)."""
        return self.spectral_function(sp.log)

    def exp(self) -> Tensor2:
        """Exponential of a symmetric tensor via spectral decomposition: exp(A)."""
        return self.spectral_function(sp.exp)

    # ------------------------------------------------------------------
    # Coordinate transformations
    # ------------------------------------------------------------------

    def rotate(self, Q: sp.Matrix) -> Tensor2:
        """Transform under rotation: A' = Q A Q^T.

        Parameters
        ----------
        Q : sp.Matrix
            3×3 orthogonal rotation matrix.
        """
        if isinstance(Q, Tensor2):
            Q = Q.matrix
        return Tensor2(
            sp.simplify(Q * self._m * Q.T),
            name=f"{self.name}'" if self.name else "",
        )

    # ------------------------------------------------------------------
    # Substitution and evaluation
    # ------------------------------------------------------------------

    def subs(self, *args, **kwargs) -> Tensor2:
        """Substitute symbolic values (delegates to SymPy Matrix.subs)."""
        return Tensor2(self._m.subs(*args, **kwargs), name=self.name)

    def evaluate(self, **values: float):
        """Substitute values and return a NumPy array.

        Requires numpy to be installed (optional dependency).
        """
        import numpy as np

        m = self._m.subs(values)
        return np.array(m.tolist(), dtype=float)

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other: Tensor2) -> Tensor2:
        if isinstance(other, Tensor2):
            return Tensor2(self._m + other._m)
        return Tensor2(self._m + sp.Matrix(other))

    def __sub__(self, other: Tensor2) -> Tensor2:
        if isinstance(other, Tensor2):
            return Tensor2(self._m - other._m)
        return Tensor2(self._m - sp.Matrix(other))

    def __mul__(self, scalar) -> Tensor2:
        return Tensor2(scalar * self._m, name=self.name)

    def __rmul__(self, scalar) -> Tensor2:
        return Tensor2(scalar * self._m, name=self.name)

    def __neg__(self) -> Tensor2:
        return Tensor2(-self._m, name=self.name)

    def __matmul__(self, other: Tensor2) -> Tensor2:
        """Matrix multiplication: A @ B."""
        if isinstance(other, Tensor2):
            return Tensor2(self._m * other._m)
        return Tensor2(self._m * sp.Matrix(other))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Tensor2):
            return sp.simplify(self._m - other._m) == sp.zeros(3, 3)
        return NotImplemented

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def latex(self) -> str:
        """LaTeX string representation."""
        return sp.latex(self._m)

    def _repr_latex_(self) -> str:
        """Jupyter notebook LaTeX rendering."""
        if self.name:
            return f"$${self.name} = {sp.latex(self._m)}$$"
        return f"$${sp.latex(self._m)}$$"

    def __repr__(self) -> str:
        if self.name:
            return f"Tensor2({self.name}):\n{self._m}"
        return f"Tensor2:\n{self._m}"

    def __str__(self) -> str:
        return str(self._m)


# ------------------------------------------------------------------
# Identity tensor
# ------------------------------------------------------------------

I = Tensor2(sp.eye(3), name="I")
"""The 3×3 identity tensor."""
