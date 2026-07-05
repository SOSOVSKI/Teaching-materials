"""constkit — Symbolic tensor algebra toolkit for constitutive modeling.

Provides a SymPy-based symbolic engine for verifying tensor operations,
kinematics, and stress measures covered in L01–L03 of a constitutive
modeling course.

Quick start::

    import constkit as ck
    import sympy as sp

    k = sp.Symbol('k', positive=True)
    F = ck.deformation_gradient('simple_shear', k=k)
    E = ck.green_lagrange(F)
    ck.show(E, name='E')
"""

from constkit.tensor2 import Tensor2
from constkit.tensor4 import Tensor4
from constkit.vector import (
    angle,
    cross,
    dot,
    norm,
    outer_product,
    outer_vec_tensor2,
    triple_product,
)
from constkit.index_notation import kronecker, levi_civita
from constkit.invariants import (
    I1,
    I2,
    I3,
    J2_invariant,
    invariant_derivative,
)
from constkit.kinematics import (
    almansi,
    deformation_gradient,
    green_lagrange,
    infinitesimal_strain,
    isochoric_split,
    left_cauchy_green,
    metric_from_F,
    metric_from_F_inv,
    polar_decomposition,
    principal_stretches,
    right_cauchy_green,
    spatial_metric_from_F,
    velocity_gradient,
)
from constkit.coordinates import (
    contravariant_basis,
    covariant_derivative_tensor2,
    covariant_derivative_vector,
    cylindrical_basis,
    metric_tensor,
    spherical_basis,
    verify_biorthogonality,
)
from constkit.stress import (
    cauchy_to_kirchhoff,
    cauchy_to_pk1,
    cauchy_to_pk2,
    check_objectivity,
    pk1_to_cauchy,
    pk2_to_cauchy,
    verify_stress_transformations,
)
from constkit.rates import jaumann_rate, oldroyd_rate, truesdell_rate
from constkit.calculus import (
    double_contract_2_4,
    double_contract_4_2,
    double_contract_4_4,
    identity_4th,
    invariant_gradient,
    lower_index,
    lower_index_tensor4,
    lower_index_vector,
    pull_back,
    pull_back_covariant,
    pull_back_tensor2,
    pull_back_tensor4,
    pull_back_vector,
    push_forward,
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
    tensor_derivative,
    tensor2_derivative,
)
from constkit.transforms import rotation_matrix
from constkit.display import show
from constkit.notation import (
    from_mandel,
    from_mandel_matrix,
    from_voigt,
    from_voigt_matrix,
    mandel_matrix_to_voigt,
    mandel_to_voigt,
    to_mandel,
    to_mandel_matrix,
    to_voigt,
    to_voigt_matrix,
    voigt_matrix_to_mandel,
    voigt_to_mandel,
)

__version__ = "0.1.0"

__all__ = [
    # Tensor classes
    "Tensor2",
    "Tensor4",
    # Vector operations
    "angle",
    "cross",
    "dot",
    "norm",
    "outer_product",
    "outer_vec_tensor2",
    "triple_product",
    # Index notation
    "kronecker",
    "levi_civita",
    # Invariants
    "I1",
    "I2",
    "I3",
    "J2_invariant",
    "invariant_derivative",
    # Kinematics
    "almansi",
    "deformation_gradient",
    "green_lagrange",
    "infinitesimal_strain",
    "isochoric_split",
    "left_cauchy_green",
    "metric_from_F",
    "metric_from_F_inv",
    "polar_decomposition",
    "principal_stretches",
    "right_cauchy_green",
    "spatial_metric_from_F",
    "velocity_gradient",
    # Coordinates
    "contravariant_basis",
    "covariant_derivative_tensor2",
    "covariant_derivative_vector",
    "cylindrical_basis",
    "metric_tensor",
    "spherical_basis",
    "verify_biorthogonality",
    # Stress
    "cauchy_to_kirchhoff",
    "cauchy_to_pk1",
    "cauchy_to_pk2",
    "check_objectivity",
    "pk1_to_cauchy",
    "pk2_to_cauchy",
    "verify_stress_transformations",
    # Rates
    "jaumann_rate",
    "oldroyd_rate",
    "truesdell_rate",
    # Calculus & contractions
    "double_contract_2_4",
    "double_contract_4_2",
    "double_contract_4_4",
    "identity_4th",
    "invariant_gradient",
    "lower_index",
    "lower_index_tensor4",
    "lower_index_vector",
    "pull_back",
    "pull_back_covariant",
    "pull_back_tensor2",
    "pull_back_tensor4",
    "pull_back_vector",
    "push_forward",
    "push_forward_covariant",
    "push_forward_tensor2",
    "push_forward_tensor4",
    "push_forward_vector",
    "raise_index",
    "raise_index_tensor4",
    "raise_index_vector",
    "raise_one_index",
    "scalar_tensor_derivative",
    "scalar_tensor2_derivative",
    "symmetric_identity_4th",
    "tensor_derivative",
    "tensor2_derivative",
    # Transforms / display
    "rotation_matrix",
    "show",
    # Notation (Voigt / Mandel)
    "from_mandel",
    "from_mandel_matrix",
    "from_voigt",
    "from_voigt_matrix",
    "mandel_matrix_to_voigt",
    "mandel_to_voigt",
    "to_mandel",
    "to_mandel_matrix",
    "to_voigt",
    "to_voigt_matrix",
    "voigt_matrix_to_mandel",
    "voigt_to_mandel",
]
