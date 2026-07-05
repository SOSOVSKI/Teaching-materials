"""Shared fixtures for HW1 test suite."""

import sympy as sp
import pytest

from constkit.tensor2 import Tensor2


# ------------------------------------------------------------------
# Symbolic parameters
# ------------------------------------------------------------------

@pytest.fixture
def k_sym():
    return sp.Symbol("k", positive=True)


@pytest.fixture
def theta_sym():
    return sp.Symbol("theta")


@pytest.fixture
def r_sym():
    return sp.Symbol("r", positive=True)


@pytest.fixture
def phi_sym():
    return sp.Symbol("phi")


# ------------------------------------------------------------------
# Numeric tensors (for substitution checks)
# ------------------------------------------------------------------

@pytest.fixture
def u_vec():
    return sp.Matrix([1, 2, 3])


@pytest.fixture
def v_vec():
    return sp.Matrix([4, 5, 6])


@pytest.fixture
def S_tensor():
    """Symmetric tensor for Problem 1."""
    return Tensor2(
        sp.Matrix([[2, 1, 0], [1, 3, 1], [0, 1, 4]]),
        name="S",
    )


@pytest.fixture
def T_tensor():
    """Second tensor for Problem 1."""
    return Tensor2(
        sp.Matrix([[1, 0, 2], [0, 1, 0], [2, 0, 3]]),
        name="T",
    )
