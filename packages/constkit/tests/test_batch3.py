"""Tests for batch-3 additions: outer_vec_tensor2, double_contract_4_x, Tensor4, notation."""

from __future__ import annotations

import sympy as sp
import pytest

from constkit.tensor2 import Tensor2
from constkit.tensor4 import Tensor4
from constkit.vector import outer_vec_tensor2
from constkit.calculus import (
    double_contract_4_2,
    double_contract_2_4,
    double_contract_4_4,
    identity_4th,
    symmetric_identity_4th,
)
from constkit.notation import (
    to_voigt,
    from_voigt,
    to_voigt_matrix,
    from_voigt_matrix,
    to_mandel,
    from_mandel,
    to_mandel_matrix,
    from_mandel_matrix,
    voigt_to_mandel,
    mandel_to_voigt,
    voigt_matrix_to_mandel,
    mandel_matrix_to_voigt,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _sym_tensor(prefix: str) -> Tensor2:
    """3×3 symmetric symbolic tensor."""
    vals = {(i, j): sp.Symbol(f"{prefix}{i+1}{j+1}") for i in range(3) for j in range(3)}
    m = sp.Matrix([[vals[i, j] for j in range(3)] for i in range(3)])
    sym_m = sp.Rational(1, 2) * (m + m.T)
    return Tensor2(sym_m)


def _zero_tensor4() -> sp.Array:
    a = sp.MutableDenseNDimArray([0]*81, (3, 3, 3, 3))
    return sp.Array(a)


# ------------------------------------------------------------------
# outer_vec_tensor2
# ------------------------------------------------------------------

class TestOuterVecTensor2:
    def test_shape(self):
        v = sp.Matrix([1, 0, 0])
        A = Tensor2(sp.eye(3))
        result = outer_vec_tensor2(v, A)
        assert result.shape == (3, 3, 3)

    def test_components(self):
        v = sp.Matrix([sp.Symbol('a'), sp.Symbol('b'), sp.Symbol('c')])
        A = Tensor2(sp.eye(3))
        result = outer_vec_tensor2(v, A)
        # result[i, j, k] = v_i * delta_jk
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    expected = v[i] * (1 if j == k else 0)
                    assert sp.simplify(result[i, j, k] - expected) == 0

    def test_zero_vector(self):
        v = sp.Matrix([0, 0, 0])
        A = Tensor2(sp.eye(3))
        result = outer_vec_tensor2(v, A)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    assert result[i, j, k] == 0


# ------------------------------------------------------------------
# double_contract_4_2, double_contract_2_4, double_contract_4_4
# ------------------------------------------------------------------

class TestDoubleContractFunctions:
    def setup_method(self):
        self.A = _sym_tensor('a')
        self.I4 = identity_4th()
        self.IS4 = symmetric_identity_4th()

    def test_4_2_identity(self):
        # I4 : A = A  (non-symmetric identity: (I4)_ijkl = delta_ik delta_jl)
        result = double_contract_4_2(self.I4, self.A)
        assert isinstance(result, Tensor2)
        # Transpose: I4_ijkl = delta_ik delta_jl → (I4:A)_ij = A_ji
        diff = sp.simplify(result.matrix - self.A.matrix.T)
        assert diff == sp.zeros(3, 3)

    def test_symmetric_identity_4_2(self):
        # IS4 : A = sym(A) for symmetric A → sym(A) = A
        A_sym = _sym_tensor('s')
        result = double_contract_4_2(self.IS4, A_sym)
        diff = sp.simplify(result.matrix - A_sym.matrix)
        assert diff == sp.zeros(3, 3)

    def test_2_4_identity(self):
        # A : I4 = transpose of A (same as 4_2 by symmetry of I4 construction)
        result = double_contract_2_4(self.A, self.I4)
        assert isinstance(result, Tensor2)
        diff = sp.simplify(result.matrix - self.A.matrix.T)
        assert diff == sp.zeros(3, 3)

    def test_4_4_scalar(self):
        # I4 : I4 → scalar = 9 (sum of delta_ik delta_jl delta_ik delta_jl)
        result = double_contract_4_4(self.I4, self.I4)
        assert sp.simplify(result - 9) == 0

    def test_4_4_symmetric_identity(self):
        # IS4 : IS4 → should be 6 (= 3*2 for 3D symmetric)
        result = double_contract_4_4(self.IS4, self.IS4)
        assert sp.simplify(result - 6) == 0


# ------------------------------------------------------------------
# Tensor4 class
# ------------------------------------------------------------------

class TestTensor4Construction:
    def test_from_array(self):
        a = sp.MutableDenseNDimArray([0]*81, (3, 3, 3, 3))
        T = Tensor4(a, name="C")
        assert T.name == "C"
        assert T.array.shape == (3, 3, 3, 3)

    def test_from_sp_array(self):
        a = sp.Array([0]*81, (3, 3, 3, 3))
        T = Tensor4(a)
        assert T.array.shape == (3, 3, 3, 3)

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            Tensor4(sp.zeros(3, 3))

    def test_name_property(self):
        T = Tensor4(sp.MutableDenseNDimArray([0]*81, (3, 3, 3, 3)), name="A")
        T.name = "B"
        assert T.name == "B"

    def test_getitem(self):
        a = sp.MutableDenseNDimArray([0]*81, (3, 3, 3, 3))
        a[0, 0, 0, 0] = sp.Integer(5)
        T = Tensor4(a)
        assert T[0, 0, 0, 0] == 5
        assert T[1, 0, 0, 0] == 0


class TestTensor4Contractions:
    def setup_method(self):
        # Build symmetric identity C_ijkl = ½(δ_ik δ_jl + δ_il δ_jk)
        a = sp.MutableDenseNDimArray([0]*81, (3, 3, 3, 3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        val = sp.Rational(1, 2) * (
                            (1 if i == k and j == l else 0)
                            + (1 if i == l and j == k else 0)
                        )
                        a[i, j, k, l] = val
        self.IS4 = Tensor4(a, name="IS")
        self.A = Tensor2(sp.Matrix([[1, 2, 3], [2, 5, 6], [3, 6, 9]]))

    def test_double_contract_symmetrizes(self):
        result = self.IS4.double_contract(self.A)
        assert isinstance(result, Tensor2)
        diff = sp.simplify(result.matrix - self.A.sym().matrix)
        assert diff == sp.zeros(3, 3)

    def test_rdouble_contract_symmetrizes(self):
        result = self.IS4.rdouble_contract(self.A)
        assert isinstance(result, Tensor2)
        diff = sp.simplify(result.matrix - self.A.sym().matrix)
        assert diff == sp.zeros(3, 3)

    def test_full_contract_scalar(self):
        result = self.IS4.full_contract(self.IS4)
        # IS4 : IS4 = 6 for 3D
        assert sp.simplify(result - 6) == 0


class TestTensor4Arithmetic:
    def setup_method(self):
        a = sp.MutableDenseNDimArray([0]*81, (3, 3, 3, 3))
        for i in range(3):
            a[i, i, i, i] = sp.Integer(1)
        self.T = Tensor4(a)

    def test_add(self):
        result = self.T + self.T
        assert result[0, 0, 0, 0] == 2
        assert result[1, 0, 0, 0] == 0

    def test_sub(self):
        result = self.T - self.T
        assert result[0, 0, 0, 0] == 0

    def test_scalar_mul(self):
        result = 3 * self.T
        assert result[0, 0, 0, 0] == 3

    def test_rmul(self):
        result = self.T * sp.Integer(2)
        assert result[0, 0, 0, 0] == 2

    def test_neg(self):
        result = -self.T
        assert result[0, 0, 0, 0] == -1


class TestTensor4Display:
    def test_repr(self):
        T = Tensor4(sp.MutableDenseNDimArray([0]*81, (3, 3, 3, 3)), name="C")
        assert "C" in repr(T)

    def test_repr_latex(self):
        T = Tensor4(sp.MutableDenseNDimArray([0]*81, (3, 3, 3, 3)), name="C")
        latex = T._repr_latex_()
        assert "C" in latex
        assert "$$" in latex


class TestTensor4PushPull:
    def test_push_pull_roundtrip(self):
        # Build simple C4 (diagonal)
        a = sp.MutableDenseNDimArray([0]*81, (3, 3, 3, 3))
        for i in range(3):
            a[i, i, i, i] = sp.Integer(1)
        C4 = Tensor4(a, name="C")

        k = sp.Symbol('k', positive=True)
        from constkit.kinematics import deformation_gradient
        F = deformation_gradient('simple_shear', k=k)

        c4 = C4.push_forward(F)
        C4_back = c4.pull_back(F)
        for i in range(3):
            for j in range(3):
                for kk in range(3):
                    for l in range(3):
                        diff = sp.simplify(C4_back[i, j, kk, l] - C4[i, j, kk, l])
                        assert diff == 0


class TestTensor4Symmetry:
    def setup_method(self):
        # Build a non-symmetric tensor
        a = sp.MutableDenseNDimArray([0]*81, (3, 3, 3, 3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        a[i, j, k, l] = sp.Integer((i + 1) * 10 + (j + 1))
        self.T = Tensor4(a)

    def test_minor_symmetrize(self):
        S = self.T.minor_symmetrize()
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        assert sp.simplify(S[i, j, k, l] - S[j, i, k, l]) == 0
                        assert sp.simplify(S[i, j, k, l] - S[i, j, l, k]) == 0

    def test_major_symmetrize(self):
        S = self.T.major_symmetrize()
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        assert sp.simplify(S[i, j, k, l] - S[k, l, i, j]) == 0


class TestTensor4SubsEvaluate:
    def test_subs(self):
        x = sp.Symbol('x')
        a = sp.MutableDenseNDimArray([0]*81, (3, 3, 3, 3))
        a[0, 0, 0, 0] = x
        T = Tensor4(a)
        T2 = T.subs(x, 5)
        assert T2[0, 0, 0, 0] == 5

    @pytest.mark.skip(reason="numpy optional dependency")
    def test_evaluate(self):
        x = sp.Symbol('x')
        a = sp.MutableDenseNDimArray([0]*81, (3, 3, 3, 3))
        a[0, 0, 0, 0] = x
        T = Tensor4(a)
        result = T.evaluate(x=3.0)
        assert abs(result[0, 0, 0, 0] - 3.0) < 1e-12


# ------------------------------------------------------------------
# Notation: Voigt
# ------------------------------------------------------------------

class TestVoigt:
    def setup_method(self):
        T11, T22, T33 = sp.symbols('T11 T22 T33')
        T23, T13, T12 = sp.symbols('T23 T13 T12')
        self.m = sp.Matrix([
            [T11, T12, T13],
            [T12, T22, T23],
            [T13, T23, T33],
        ])
        self.T = Tensor2(self.m)
        self.syms = (T11, T22, T33, T23, T13, T12)

    def test_to_voigt_ordering(self):
        v = to_voigt(self.T)
        T11, T22, T33, T23, T13, T12 = self.syms
        expected = sp.Matrix([T11, T22, T33, T23, T13, T12])
        assert sp.simplify(v - expected) == sp.zeros(6, 1)

    def test_from_voigt_roundtrip(self):
        v = to_voigt(self.T)
        T2 = from_voigt(v)
        diff = sp.simplify(T2.matrix - self.T.matrix)
        assert diff == sp.zeros(3, 3)

    def test_from_voigt_symmetry(self):
        v = sp.Matrix([1, 2, 3, 4, 5, 6])
        T = from_voigt(v)
        # must be symmetric
        diff = sp.simplify(T.matrix - T.matrix.T)
        assert diff == sp.zeros(3, 3)


class TestVoigtMatrix:
    def test_identity_4th_roundtrip(self):
        I4 = identity_4th()
        M = to_voigt_matrix(I4)
        C4 = from_voigt_matrix(M)
        # Only check entries covered by _PAIRS (symmetric entries)
        _PAIRS = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
        for i, j in _PAIRS:
            for k, l in _PAIRS:
                diff = sp.simplify(C4[i, j, k, l] - I4[i, j, k, l])
                assert diff == 0

    def test_shape(self):
        I4 = identity_4th()
        M = to_voigt_matrix(I4)
        assert M.shape == (6, 6)

    def test_from_voigt_matrix_shape(self):
        M = sp.eye(6)
        C4 = from_voigt_matrix(M)
        assert C4.shape == (3, 3, 3, 3)


# ------------------------------------------------------------------
# Notation: Mandel
# ------------------------------------------------------------------

class TestMandel:
    def setup_method(self):
        T11, T22, T33 = sp.symbols('T11 T22 T33')
        T23, T13, T12 = sp.symbols('T23 T13 T12')
        self.m = sp.Matrix([
            [T11, T12, T13],
            [T12, T22, T23],
            [T13, T23, T33],
        ])
        self.T = Tensor2(self.m)

    def test_to_mandel_diagonal_unchanged(self):
        m = to_mandel(self.T)
        T = self.T.matrix
        assert sp.simplify(m[0] - T[0, 0]) == 0
        assert sp.simplify(m[1] - T[1, 1]) == 0
        assert sp.simplify(m[2] - T[2, 2]) == 0

    def test_to_mandel_offdiag_scaled(self):
        m = to_mandel(self.T)
        T = self.T.matrix
        # index 3 → (1,2) i.e. T23
        assert sp.simplify(m[3] - sp.sqrt(2) * T[1, 2]) == 0
        assert sp.simplify(m[4] - sp.sqrt(2) * T[0, 2]) == 0
        assert sp.simplify(m[5] - sp.sqrt(2) * T[0, 1]) == 0

    def test_from_mandel_roundtrip(self):
        m = to_mandel(self.T)
        T2 = from_mandel(m)
        diff = sp.simplify(T2.matrix - self.T.matrix)
        assert diff == sp.zeros(3, 3)

    def test_inner_product_preserved(self):
        """Mandel inner product = tensor double contraction."""
        S = Tensor2(sp.Matrix([[1, 2, 3], [2, 5, 6], [3, 6, 9]]))
        T = Tensor2(sp.Matrix([[2, 1, 0], [1, 3, 2], [0, 2, 4]]))
        dc = S.double_contract(T)
        ms = to_mandel(S)
        mt = to_mandel(T)
        inner = (ms.T * mt)[0, 0]
        assert sp.simplify(inner - dc) == 0


class TestMandelMatrix:
    def test_inner_product_preserved(self):
        """A:C:B (tensor) == mA^T M mB (Mandel)."""
        IS4 = symmetric_identity_4th()
        A = Tensor2(sp.Matrix([[1, 2, 3], [2, 5, 6], [3, 6, 9]]))
        B = Tensor2(sp.Matrix([[2, 1, 0], [1, 3, 2], [0, 2, 4]]))

        # Tensor contraction: A : IS4 : B
        AIS = double_contract_2_4(A, IS4)
        tensor_result = AIS.double_contract(B)

        # Mandel representation
        M_mat = to_mandel_matrix(IS4)
        mA = to_mandel(A)
        mB = to_mandel(B)
        mandel_result = (mA.T * M_mat * mB)[0, 0]

        assert sp.simplify(tensor_result - mandel_result) == 0

    def test_roundtrip(self):
        IS4 = symmetric_identity_4th()
        M = to_mandel_matrix(IS4)
        C4_back = from_mandel_matrix(M)
        _PAIRS = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
        for i, j in _PAIRS:
            for k, l in _PAIRS:
                diff = sp.simplify(C4_back[i, j, k, l] - IS4[i, j, k, l])
                assert diff == 0


# ------------------------------------------------------------------
# Voigt ↔ Mandel conversions
# ------------------------------------------------------------------

class TestVoigtMandelConversions:
    def setup_method(self):
        T11, T22, T33 = sp.symbols('T11 T22 T33')
        T23, T13, T12 = sp.symbols('T23 T13 T12')
        self.T = Tensor2(sp.Matrix([
            [T11, T12, T13],
            [T12, T22, T23],
            [T13, T23, T33],
        ]))

    def test_voigt_to_mandel_vector(self):
        v = to_voigt(self.T)
        m_direct = to_mandel(self.T)
        m_converted = voigt_to_mandel(v)
        diff = sp.simplify(m_direct - m_converted)
        assert diff == sp.zeros(6, 1)

    def test_mandel_to_voigt_vector(self):
        m = to_mandel(self.T)
        v_direct = to_voigt(self.T)
        v_converted = mandel_to_voigt(m)
        diff = sp.simplify(v_direct - v_converted)
        assert diff == sp.zeros(6, 1)

    def test_voigt_mandel_matrix_roundtrip(self):
        IS4 = symmetric_identity_4th()
        V = to_voigt_matrix(IS4)
        M = voigt_matrix_to_mandel(V)
        M_direct = to_mandel_matrix(IS4)
        diff = sp.simplify(M - M_direct)
        assert diff == sp.zeros(6, 6)

    def test_mandel_voigt_matrix_roundtrip(self):
        IS4 = symmetric_identity_4th()
        M = to_mandel_matrix(IS4)
        V = mandel_matrix_to_voigt(M)
        V_direct = to_voigt_matrix(IS4)
        diff = sp.simplify(V - V_direct)
        assert diff == sp.zeros(6, 6)

    def test_vector_roundtrip(self):
        v = sp.Matrix([1, 2, 3, 4, 5, 6])
        assert sp.simplify(mandel_to_voigt(voigt_to_mandel(v)) - v) == sp.zeros(6, 1)

    def test_matrix_roundtrip(self):
        IS4 = symmetric_identity_4th()
        M = to_mandel_matrix(IS4)
        assert sp.simplify(voigt_matrix_to_mandel(mandel_matrix_to_voigt(M)) - M) == sp.zeros(6, 6)
