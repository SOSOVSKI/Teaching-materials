"""Tests for features added in Phase 2:
  - tensor2_derivative, scalar_tensor_derivative, scalar_tensor2_derivative
  - covariant_derivative_vector, covariant_derivative_tensor2
  - metric_from_F, spatial_metric_from_F, metric_from_F_inv
  - push_forward/pull_back for rank 1, 2, 4 tensors
  - index raising/lowering for rank 1, 2, 4 tensors
  - Tensor2.name property
"""

import sympy as sp

# Helper: zero sp.Array of given shape (sp.zeros only handles 2D)
def _zero_array(*shape):
    if len(shape) == 2:
        return sp.Array(sp.zeros(*shape))
    elif len(shape) == 3:
        n = shape[0]
        return sp.Array([[[0]*shape[2]]*shape[1]]*shape[0])
    elif len(shape) == 4:
        return sp.Array([[[[0]*shape[3]]*shape[2]]*shape[1]]*shape[0])
    raise ValueError(f"Unsupported shape {shape}")


import pytest

from constkit.tensor2 import Tensor2, I
from constkit.calculus import (
    identity_4th,
    lower_index,
    lower_index_tensor4,
    lower_index_vector,
    pull_back_covariant,
    pull_back_tensor2,
    pull_back_tensor4,
    pull_back_vector,
    push_forward_covariant,
    push_forward_tensor2,
    push_forward_tensor4,
    push_forward_vector,
    raise_index,
    raise_index_tensor4,
    raise_index_vector,
    raise_one_index,
    scalar_tensor_derivative,
    scalar_tensor2_derivative,
    symmetric_identity_4th,
    tensor2_derivative,
)
from constkit.coordinates import (
    cylindrical_basis,
    christoffel_symbols,
    covariant_derivative_vector,
    covariant_derivative_tensor2,
)
from constkit.kinematics import (
    metric_from_F,
    spatial_metric_from_F,
    metric_from_F_inv,
    right_cauchy_green,
    left_cauchy_green,
)


# ---------------------------------------------------------------------------
# Tensor2.name property
# ---------------------------------------------------------------------------

class TestTensor2NameProperty:
    def test_name_readable(self):
        A = Tensor2(sp.eye(3), name="A")
        assert A.name == "A"

    def test_name_mutable(self):
        A = Tensor2(sp.eye(3), name="old")
        A.name = "new"
        assert A.name == "new"

    def test_name_default_empty(self):
        A = Tensor2(sp.eye(3))
        assert A.name == ""

    def test_name_in_repr(self):
        A = Tensor2(sp.eye(3), name="Q")
        assert "Q" in repr(A)


# ---------------------------------------------------------------------------
# tensor2_derivative  ∂A_ij/∂B_kl
# ---------------------------------------------------------------------------

class TestTensor2Derivative:
    def test_identity_tensor_derivative(self):
        """∂A_ij/∂A_kl = δ_ik δ_jl  (4th-order identity)."""
        A = Tensor2(sp.Matrix([[sp.Symbol(f"a{i}{j}") for j in range(3)]
                               for i in range(3)]))
        result = tensor2_derivative(A, A)
        expected = identity_4th()
        assert result == expected

    def test_independent_tensors_zero(self):
        """∂A/∂B = 0 when A and B have disjoint symbols."""
        A = Tensor2(sp.Matrix([[sp.Symbol(f"a{i}{j}") for j in range(3)]
                               for i in range(3)]))
        B = Tensor2(sp.Matrix([[sp.Symbol(f"b{i}{j}") for j in range(3)]
                               for i in range(3)]))
        result = tensor2_derivative(A, B)
        assert result == _zero_array(3, 3, 3, 3)

    def test_transpose_derivative(self):
        """∂(Aᵀ)_ij/∂A_kl = δ_jk δ_il  (index swap)."""
        syms = [[sp.Symbol(f"a{i}{j}") for j in range(3)] for i in range(3)]
        A = Tensor2(sp.Matrix(syms))
        AT = A.T
        result = tensor2_derivative(AT, A)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        expected = 1 if (j == k and i == l) else 0
                        assert result[i, j, k, l] == expected

    def test_returns_array_shape(self):
        A = Tensor2(sp.Matrix([[sp.Symbol(f"a{i}{j}") for j in range(3)]
                               for i in range(3)]))
        result = tensor2_derivative(A, A)
        assert result.shape == (3, 3, 3, 3)


# ---------------------------------------------------------------------------
# scalar_tensor_derivative  ∂f/∂A_ij
# ---------------------------------------------------------------------------

class TestScalarTensorDerivative:
    def setup_method(self):
        self.syms = [[sp.Symbol(f"a{i}{j}") for j in range(3)] for i in range(3)]
        self.A = Tensor2(sp.Matrix(self.syms))

    def test_derivative_of_trace(self):
        """∂(tr A)/∂A = I."""
        f = self.A.trace()
        result = scalar_tensor_derivative(f, self.A)
        assert result == I

    def test_derivative_of_linear(self):
        """∂(A_12)/∂A = e_1 ⊗ e_2."""
        f = self.syms[0][1]
        result = scalar_tensor_derivative(f, self.A)
        expected = sp.zeros(3, 3)
        expected[0, 1] = 1
        assert result.matrix == expected

    def test_returns_tensor2(self):
        f = self.A.trace()
        result = scalar_tensor_derivative(f, self.A)
        assert isinstance(result, Tensor2)

    def test_derivative_of_det(self):
        """∂(det A)/∂A_ij at A=I equals δ_ij (cofactor of I)."""
        ident = Tensor2(sp.eye(3))
        syms = [[sp.Symbol(f"a{i}{j}") for j in range(3)] for i in range(3)]
        A_sym = Tensor2(sp.Matrix(syms))
        f = A_sym.det()
        result_sym = scalar_tensor_derivative(f, A_sym)
        result = result_sym.subs(
            {syms[i][j]: (1 if i == j else 0)
             for i in range(3) for j in range(3)}
        )
        assert sp.simplify(result.matrix - sp.eye(3)) == sp.zeros(3, 3)


# ---------------------------------------------------------------------------
# scalar_tensor2_derivative  ∂²f/(∂A_ij ∂B_kl)
# ---------------------------------------------------------------------------

class TestScalarTensor2Derivative:
    def test_independent_tensors_zero(self):
        """f = tr(A) depends only on A, so mixed derivative w.r.t. B is 0."""
        a_syms = [[sp.Symbol(f"a{i}{j}") for j in range(3)] for i in range(3)]
        b_syms = [[sp.Symbol(f"b{i}{j}") for j in range(3)] for i in range(3)]
        A = Tensor2(sp.Matrix(a_syms))
        B = Tensor2(sp.Matrix(b_syms))
        f = A.trace()
        result = scalar_tensor2_derivative(f, A, B)
        assert result == _zero_array(3, 3, 3, 3)

    def test_second_derivative_of_double_contraction(self):
        """∂²(A:B)/(∂A_ij ∂B_kl) = δ_ik δ_jl."""
        a_syms = [[sp.Symbol(f"a{i}{j}") for j in range(3)] for i in range(3)]
        b_syms = [[sp.Symbol(f"b{i}{j}") for j in range(3)] for i in range(3)]
        A = Tensor2(sp.Matrix(a_syms))
        B = Tensor2(sp.Matrix(b_syms))
        f = A.double_contract(B)
        result = scalar_tensor2_derivative(f, A, B)
        expected = identity_4th()
        assert result == expected

    def test_returns_array_shape(self):
        a_syms = [[sp.Symbol(f"a{i}{j}") for j in range(3)] for i in range(3)]
        b_syms = [[sp.Symbol(f"b{i}{j}") for j in range(3)] for i in range(3)]
        A = Tensor2(sp.Matrix(a_syms))
        B = Tensor2(sp.Matrix(b_syms))
        result = scalar_tensor2_derivative(A.trace(), A, B)
        assert result.shape == (3, 3, 3, 3)


# ---------------------------------------------------------------------------
# covariant_derivative_vector
# ---------------------------------------------------------------------------

class TestCovariantDerivativeVector:
    def setup_method(self):
        self.r, self.phi = sp.symbols("r phi", positive=True)
        self.coords = [self.r, self.phi, sp.Symbol("z")]
        self.g_cov = cylindrical_basis(self.r, self.phi)
        self.gamma = christoffel_symbols(self.g_cov, self.coords)

    def test_constant_vector_in_cartesian_is_zero(self):
        """In Cartesian coords (Γ=0) a constant vector has zero covariant derivative."""
        x, y, z = sp.symbols("x y z")
        coords = [x, y, z]
        # Cartesian basis: identity matrix rows
        g_cart = [sp.Matrix([1, 0, 0]), sp.Matrix([0, 1, 0]), sp.Matrix([0, 0, 1])]
        gamma_cart = christoffel_symbols(g_cart, coords)
        v = sp.Matrix([sp.Integer(1), sp.Integer(2), sp.Integer(3)])
        result = covariant_derivative_vector(v, gamma_cart, coords)
        assert result == _zero_array(3, 3)

    def test_covariant_variant_accepted(self):
        """Covariant variant runs without error."""
        v = sp.Matrix([self.r, sp.Integer(0), sp.Integer(0)])
        result = covariant_derivative_vector(v, self.gamma, self.coords, variant="covariant")
        assert result.shape == (3, 3)

    def test_invalid_variant_raises(self):
        v = sp.Matrix([self.r, sp.Integer(0), sp.Integer(0)])
        with pytest.raises(ValueError):
            covariant_derivative_vector(v, self.gamma, self.coords, variant="bad")

    def test_output_shape(self):
        v = sp.Matrix([self.r, sp.Integer(0), sp.Integer(0)])
        result = covariant_derivative_vector(v, self.gamma, self.coords)
        assert result.shape == (3, 3)


# ---------------------------------------------------------------------------
# covariant_derivative_tensor2
# ---------------------------------------------------------------------------

class TestCovariantDerivativeTensor2:
    def setup_method(self):
        x, y, z = sp.symbols("x y z")
        self.coords = [x, y, z]
        g_cart = [sp.Matrix([1, 0, 0]), sp.Matrix([0, 1, 0]), sp.Matrix([0, 0, 1])]
        self.gamma_cart = christoffel_symbols(g_cart, self.coords)

    def test_constant_tensor_in_cartesian_is_zero(self):
        """Constant tensor in flat (Cartesian) space has zero covariant derivative."""
        T = Tensor2(sp.Matrix([[1, 2, 0], [2, 3, 1], [0, 1, 4]]))
        result = covariant_derivative_tensor2(T, self.gamma_cart, self.coords)
        assert result == _zero_array(3, 3, 3)

    def test_output_shape(self):
        T = Tensor2(sp.eye(3))
        result = covariant_derivative_tensor2(T, self.gamma_cart, self.coords)
        assert result.shape == (3, 3, 3)

    def test_all_variants_run(self):
        T = Tensor2(sp.eye(3))
        for v in ("contravariant", "covariant", "mixed"):
            result = covariant_derivative_tensor2(T, self.gamma_cart, self.coords, variant=v)
            assert result.shape == (3, 3, 3)

    def test_invalid_variant_raises(self):
        T = Tensor2(sp.eye(3))
        with pytest.raises(ValueError):
            covariant_derivative_tensor2(T, self.gamma_cart, self.coords, variant="bad")


# ---------------------------------------------------------------------------
# metric_from_F, spatial_metric_from_F, metric_from_F_inv
# ---------------------------------------------------------------------------

class TestMetricFromF:
    def setup_method(self):
        k = sp.Symbol("k", positive=True)
        self.F_shear = Tensor2(
            sp.Matrix([[1, k, 0], [0, 1, 0], [0, 0, 1]]), name="F"
        )
        self.F_ident = Tensor2(sp.eye(3), name="F")

    def test_metric_from_F_equals_right_cauchy_green(self):
        """metric_from_F(F) = Fᵀ F = right_cauchy_green(F)."""
        G = metric_from_F(self.F_shear)
        C = right_cauchy_green(self.F_shear)
        assert sp.simplify(G.matrix - C.matrix) == sp.zeros(3, 3)

    def test_spatial_metric_equals_left_cauchy_green(self):
        """spatial_metric_from_F(F) = F Fᵀ = left_cauchy_green(F)."""
        g = spatial_metric_from_F(self.F_shear)
        b = left_cauchy_green(self.F_shear)
        assert sp.simplify(g.matrix - b.matrix) == sp.zeros(3, 3)

    def test_metric_from_F_inv_at_identity(self):
        """At F=I: G_inv = I."""
        G_inv = metric_from_F_inv(self.F_ident)
        assert sp.simplify(G_inv.matrix - sp.eye(3)) == sp.zeros(3, 3)

    def test_metric_inverse_consistency(self):
        """G · G_inv = I."""
        G = metric_from_F(self.F_shear)
        G_inv = metric_from_F_inv(self.F_shear)
        product = sp.simplify(G.matrix * G_inv.matrix)
        assert product == sp.eye(3)

    def test_metric_symmetry(self):
        """All three metrics are symmetric."""
        G = metric_from_F(self.F_shear)
        g = spatial_metric_from_F(self.F_shear)
        G_inv = metric_from_F_inv(self.F_shear)
        for M in (G, g, G_inv):
            assert sp.simplify(M.matrix - M.matrix.T) == sp.zeros(3, 3)

    def test_names(self):
        assert metric_from_F(self.F_ident).name == "G"
        assert spatial_metric_from_F(self.F_ident).name == "g"
        assert metric_from_F_inv(self.F_ident).name == "G_inv"

    def test_accepts_matrix_input(self):
        """Functions also accept plain sp.Matrix, not just Tensor2."""
        F_mat = sp.eye(3)
        G = metric_from_F(F_mat)
        assert isinstance(G, Tensor2)


# ---------------------------------------------------------------------------
# push_forward / pull_back — rank 1 (vectors)
# ---------------------------------------------------------------------------

class TestPushPullVector:
    def setup_method(self):
        self.F = Tensor2(sp.Matrix([[2, 0, 0], [0, 3, 0], [0, 0, 1]]))
        self.v = sp.Matrix([1, 1, 1])

    def test_push_forward_vector(self):
        result = push_forward_vector(self.v, self.F)
        expected = sp.Matrix([2, 3, 1])
        assert result == expected

    def test_pull_back_vector(self):
        v_spatial = push_forward_vector(self.v, self.F)
        v_back = pull_back_vector(v_spatial, self.F)
        assert sp.simplify(v_back - self.v) == sp.Matrix([0, 0, 0])

    def test_roundtrip(self):
        """push then pull-back gives identity."""
        v_sp = push_forward_vector(self.v, self.F)
        v_ref = pull_back_vector(v_sp, self.F)
        assert sp.simplify(v_ref - self.v) == sp.Matrix([0, 0, 0])


# ---------------------------------------------------------------------------
# push_forward / pull_back — rank 2 (contravariant and covariant)
# ---------------------------------------------------------------------------

class TestPushPullTensor2:
    def setup_method(self):
        k = sp.Symbol("k", positive=True)
        self.F = Tensor2(sp.Matrix([[1, k, 0], [0, 1, 0], [0, 0, 1]]))
        self.S = Tensor2(sp.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))

    def test_contravariant_roundtrip(self):
        """push_forward_tensor2 then pull_back_tensor2 = identity."""
        t = push_forward_tensor2(self.S, self.F)
        S_back = pull_back_tensor2(t, self.F)
        assert sp.simplify(S_back.matrix - self.S.matrix) == sp.zeros(3, 3)

    def test_covariant_roundtrip(self):
        """push_forward_covariant then pull_back_covariant = identity."""
        t = push_forward_covariant(self.S, self.F)
        S_back = pull_back_covariant(t, self.F)
        assert sp.simplify(S_back.matrix - self.S.matrix) == sp.zeros(3, 3)

    def test_push_at_identity_is_noop(self):
        """At F=I: push_forward_tensor2(T, I) = T."""
        F_ident = Tensor2(sp.eye(3))
        result = push_forward_tensor2(self.S, F_ident)
        assert sp.simplify(result.matrix - self.S.matrix) == sp.zeros(3, 3)


# ---------------------------------------------------------------------------
# push_forward / pull_back — rank 4
# ---------------------------------------------------------------------------

class TestPushPullTensor4:
    def setup_method(self):
        # Use simple stretch: F = diag(2, 1, 1)
        self.F = Tensor2(sp.Matrix([[2, 0, 0], [0, 1, 0], [0, 0, 1]]))
        # Use symmetric identity as reference moduli
        self.C4 = symmetric_identity_4th()

    def test_roundtrip(self):
        """push then pull-back recovers original."""
        c4 = push_forward_tensor4(self.C4, self.F)
        C4_back = pull_back_tensor4(c4, self.F)
        diff = sp.Array([
            [[[sp.simplify(C4_back[i, j, k, l] - self.C4[i, j, k, l])
               for l in range(3)] for k in range(3)]
              for j in range(3)] for i in range(3)
        ])
        assert diff == _zero_array(3, 3, 3, 3)

    def test_identity_F_is_noop(self):
        """At F=I and J=1: push_forward_tensor4(C4, I) = C4."""
        F_ident = Tensor2(sp.eye(3))
        result = push_forward_tensor4(self.C4, F_ident)
        diff = sp.Array([
            [[[sp.simplify(result[i, j, k, l] - self.C4[i, j, k, l])
               for l in range(3)] for k in range(3)]
              for j in range(3)] for i in range(3)
        ])
        assert diff == _zero_array(3, 3, 3, 3)

    def test_output_shape(self):
        result = push_forward_tensor4(self.C4, self.F)
        assert result.shape == (3, 3, 3, 3)


# ---------------------------------------------------------------------------
# Index raising / lowering
# ---------------------------------------------------------------------------

class TestIndexRaiseLower:
    def setup_method(self):
        # Use flat metric (g_ij = δ_ij) so raising = identity operation
        self.g_flat = sp.eye(3)
        self.T = Tensor2(sp.Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        self.v = sp.Matrix([1, 2, 3])

    def test_raise_index_vector_flat(self):
        """With flat metric, v^i = v_i."""
        result = raise_index_vector(self.v, self.g_flat)
        assert result == self.v

    def test_lower_index_vector_flat(self):
        """With flat metric, v_i = v^i."""
        result = lower_index_vector(self.v, self.g_flat)
        assert result == self.v

    def test_raise_lower_roundtrip_vector(self):
        """Raise then lower with same metric = identity."""
        g = sp.Matrix([[2, 0, 0], [0, 3, 0], [0, 0, 1]])
        g_inv = g.inv()
        v_up = raise_index_vector(self.v, g_inv)
        v_down = lower_index_vector(v_up, g)
        assert sp.simplify(v_down - self.v) == sp.Matrix([0, 0, 0])

    def test_raise_index_flat(self):
        """With flat metric, raise_index(T) = T."""
        result = raise_index(self.T, self.g_flat)
        assert sp.simplify(result.matrix - self.T.matrix) == sp.zeros(3, 3)

    def test_lower_index_flat(self):
        """With flat metric, lower_index(T) = T."""
        result = lower_index(self.T, self.g_flat)
        assert sp.simplify(result.matrix - self.T.matrix) == sp.zeros(3, 3)

    def test_raise_lower_roundtrip_tensor2(self):
        """Raise then lower = identity."""
        g = sp.Matrix([[2, 0, 0], [0, 3, 0], [0, 0, 1]])
        g_inv = g.inv()
        T_up = raise_index(self.T, g_inv)
        T_down = lower_index(T_up, g)
        assert sp.simplify(T_down.matrix - self.T.matrix) == sp.zeros(3, 3)

    def test_raise_one_index_first(self):
        """With flat metric, raise first index = T itself."""
        result = raise_one_index(self.T, self.g_flat, which="first")
        assert sp.simplify(result.matrix - self.T.matrix) == sp.zeros(3, 3)

    def test_raise_one_index_second(self):
        """With flat metric, raise second index = T itself."""
        result = raise_one_index(self.T, self.g_flat, which="second")
        assert sp.simplify(result.matrix - self.T.matrix) == sp.zeros(3, 3)

    def test_raise_one_index_invalid(self):
        with pytest.raises(ValueError):
            raise_one_index(self.T, self.g_flat, which="third")

    def test_raise_lower_roundtrip_tensor4(self):
        """Raise then lower all 4 indices = identity."""
        C4 = identity_4th()
        g = sp.Matrix([[2, 0, 0], [0, 3, 0], [0, 0, 1]])
        g_inv = g.inv()
        C4_up = raise_index_tensor4(C4, g_inv)
        C4_down = lower_index_tensor4(C4_up, g)
        diff = sp.Array([
            [[[sp.simplify(C4_down[i, j, k, l] - C4[i, j, k, l])
               for l in range(3)] for k in range(3)]
              for j in range(3)] for i in range(3)
        ])
        assert diff == _zero_array(3, 3, 3, 3)

    def test_raise_index_tensor4_flat(self):
        """With flat metric, raise_index_tensor4 = identity."""
        C4 = symmetric_identity_4th()
        result = raise_index_tensor4(C4, self.g_flat)
        diff = sp.Array([
            [[[sp.simplify(result[i, j, k, l] - C4[i, j, k, l])
               for l in range(3)] for k in range(3)]
              for j in range(3)] for i in range(3)
        ])
        assert diff == _zero_array(3, 3, 3, 3)
