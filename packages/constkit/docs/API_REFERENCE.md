# constkit API Reference

Complete documentation for every public function, class, and method.

---

## `constkit.tensor2` — Second-Order Tensor Class

### `class Tensor2`

A symbolic second-order tensor in 3D Cartesian coordinates. Wraps a 3×3 SymPy Matrix.

**Constructor:**

```python
Tensor2(matrix, name="")
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `matrix` | `sp.Matrix` or array-like | 3×3 matrix of components |
| `name` | `str` | Display name for LaTeX rendering |

Raises `ValueError` if the matrix is not 3×3.

```python
A = Tensor2([[1, 2, 0], [2, 3, 1], [0, 1, 4]], name='A')
```

---

#### Properties

##### `Tensor2.matrix -> sp.Matrix`

The underlying 3×3 SymPy Matrix. Read-only access to raw data.

##### `Tensor2.T -> Tensor2`

Transpose: $A^T$, where $(A^T)_{ij} = A_{ji}$.

##### `Tensor2.name -> str`

Display name used in LaTeX rendering. Mutable — assign freely after construction.

---

#### Access

##### `Tensor2.__getitem__(idx: tuple[int, int]) -> sp.Expr`

Component access: `A[i, j]` returns $A_{ij}$.

```python
A[0, 1]  # returns component A₁₂
```

---

#### Core Operations

##### `Tensor2.trace() -> sp.Expr`

$$\mathrm{tr}(\mathbf{A}) = A_{ii}$$

```python
A.trace()  # 8
```

##### `Tensor2.det() -> sp.Expr`

$$\det(\mathbf{A})$$

```python
A.det()  # 18
```

##### `Tensor2.inv() -> Tensor2`

$$\mathbf{A}^{-1}$$

Raises `NonInvertibleMatrixError` if singular.

##### `Tensor2.sym() -> Tensor2`

Symmetric part:

$$\mathrm{sym}(\mathbf{A}) = \tfrac{1}{2}(\mathbf{A} + \mathbf{A}^T)$$

##### `Tensor2.skew() -> Tensor2`

Skew-symmetric part:

$$\mathrm{skew}(\mathbf{A}) = \tfrac{1}{2}(\mathbf{A} - \mathbf{A}^T)$$

##### `Tensor2.dev() -> Tensor2`

Deviatoric part:

$$\mathrm{dev}(\mathbf{A}) = \mathbf{A} - \tfrac{1}{3}\mathrm{tr}(\mathbf{A})\,\mathbf{I}$$

---

#### Contractions

##### `Tensor2.dot(v: sp.Matrix) -> sp.Matrix`

Single contraction with a vector:

$$(\mathbf{A} \cdot \mathbf{v})_i = A_{ij} v_j$$

If `v` is a `Tensor2`, returns a `Tensor2` (matrix product).

```python
A.dot(sp.Matrix([1, 0, 0]))  # first column of A
```

##### `Tensor2.single_contract(B: Tensor2) -> Tensor2`

Tensor-tensor single contraction (matrix product):

$$(\mathbf{A} \cdot \mathbf{B})_{ij} = A_{ik} B_{kj}$$

```python
C = A.single_contract(B)
```

##### `Tensor2.double_contract(B: Tensor2) -> sp.Expr`

Double contraction (Frobenius inner product for symmetric tensors):

$$\mathbf{A} : \mathbf{B} = A_{ij} B_{ij}$$

```python
A.double_contract(B)  # scalar
```

##### `Tensor2.outer(B: Tensor2) -> sp.Array`

Outer (dyadic) product producing a 4th-order tensor:

$(\mathbf{A} \otimes \mathbf{B})_{ijkl} = A_{ij} B_{kl}$

Returns `sp.Array` with shape (3, 3, 3, 3).

---

#### Principal Invariants

##### `Tensor2.I1() -> sp.Expr`

$$I_1 = \mathrm{tr}(\mathbf{A})$$

##### `Tensor2.I2() -> sp.Expr`

$$I_2 = \tfrac{1}{2}\left[(\mathrm{tr}\,\mathbf{A})^2 - \mathrm{tr}(\mathbf{A}^2)\right]$$

##### `Tensor2.I3() -> sp.Expr`

$$I_3 = \det(\mathbf{A})$$

---

#### Eigendecomposition and Tensor Functions

##### `Tensor2.eigendecomposition() -> dict`

Computes eigenvalues and normalized eigenvectors.

**Returns:** `dict` with keys `'eigenvalues'` (list of `sp.Expr`) and `'eigenvectors'` (list of `sp.Matrix`).

```python
eig = A.eigendecomposition()
eig['eigenvalues']   # [3 - sqrt(3), 3, 3 + sqrt(3)]
eig['eigenvectors']  # [Matrix(...), Matrix(...), Matrix(...)]
```

##### `Tensor2.spectral_function(f: Callable) -> Tensor2`

Apply a scalar function to a **symmetric** tensor via spectral decomposition:

$$f(\mathbf{A}) = \sum_{\alpha} f(\lambda_\alpha)\, \mathbf{n}_\alpha \otimes \mathbf{n}_\alpha$$

This representation assumes an orthonormal eigenbasis, so it is only
valid for symmetric tensors.

| Parameter | Type | Description |
|-----------|------|-------------|
| `f` | callable | Scalar function (e.g. `sp.sqrt`, `sp.log`) |

```python
A.spectral_function(lambda x: x**2)  # same as A @ A for symmetric A
```

##### `Tensor2.sqrt() -> Tensor2`

$$\sqrt{\mathbf{A}} = \sum_\alpha \sqrt{\lambda_\alpha}\, \mathbf{n}_\alpha \otimes \mathbf{n}_\alpha$$

Defined here for symmetric tensors.

##### `Tensor2.log() -> Tensor2`

$$\ln(\mathbf{A}) = \sum_\alpha \ln(\lambda_\alpha)\, \mathbf{n}_\alpha \otimes \mathbf{n}_\alpha$$

Defined here for symmetric tensors.

##### `Tensor2.exp() -> Tensor2`

$$\exp(\mathbf{A}) = \sum_\alpha \exp(\lambda_\alpha)\, \mathbf{n}_\alpha \otimes \mathbf{n}_\alpha$$

Defined here for symmetric tensors.

---

#### Coordinate Transformations

##### `Tensor2.rotate(Q: sp.Matrix) -> Tensor2`

Transform under an orthogonal rotation:

$$\mathbf{A}' = \mathbf{Q}\,\mathbf{A}\,\mathbf{Q}^T$$

| Parameter | Type | Description |
|-----------|------|-------------|
| `Q` | `sp.Matrix` or `Tensor2` | 3×3 orthogonal rotation matrix |

---

#### Substitution and Evaluation

##### `Tensor2.subs(*args, **kwargs) -> Tensor2`

Substitute symbolic values. Delegates to `sp.Matrix.subs()`.

```python
A.subs(k, 1)               # substitute k → 1
A.subs([(k, 1), (m, 2)])   # multiple substitutions
```

##### `Tensor2.evaluate(**values) -> numpy.ndarray`

Substitute values and return a NumPy float64 array. Requires `numpy`.

```python
A.evaluate(k=1.0, m=2.0)  # returns ndarray shape (3, 3)
```

---

#### Arithmetic Operators

| Operator | Expression | Result |
|----------|------------|--------|
| `A + B` | Addition | `Tensor2` |
| `A - B` | Subtraction | `Tensor2` |
| `s * A` or `A * s` | Scalar multiplication | `Tensor2` |
| `-A` | Negation | `Tensor2` |
| `A @ B` | Matrix multiplication | `Tensor2` |
| `A == B` | Equality (via `sp.simplify`) | `bool` |

---

#### Display

##### `Tensor2.latex() -> str`

LaTeX string of the matrix components.

##### `Tensor2._repr_latex_() -> str`

Jupyter auto-rendering hook. Returns `$$name = [matrix]$$`.

---

### Module-Level Constant

##### `I`

The 3×3 identity tensor: `Tensor2(sp.eye(3), name="I")`.

A module-level constant — import directly:

```python
from constkit.tensor2 import I
```

---

## `constkit.vector` — Vector Operations

All functions accept lists, tuples, or `sp.Matrix` objects and coerce to 3×1 column vectors internally.

### `dot(u, v) -> sp.Expr`

Inner product: $\mathbf{u} \cdot \mathbf{v} = u_i v_i$

### `cross(u, v) -> sp.Matrix`

Cross product: $(\mathbf{u} \times \mathbf{v})_k = \varepsilon_{kij} u_i v_j$. Returns 3×1 column vector.

### `norm(v) -> sp.Expr`

Euclidean norm: $|\mathbf{v}| = \sqrt{\mathbf{v} \cdot \mathbf{v}}$

### `outer_product(u, v) -> sp.Matrix`

Outer product: $(\mathbf{u} \otimes \mathbf{v})_{ij} = u_i v_j$. Returns 3×3 matrix.

### `triple_product(u, v, w) -> sp.Expr`

Scalar triple product: $[\mathbf{u}, \mathbf{v}, \mathbf{w}] = \mathbf{u} \cdot (\mathbf{v} \times \mathbf{w})$

Equals the signed volume of the parallelepiped spanned by the three vectors.

### `angle(u, v) -> sp.Expr`

Angle between vectors: $\theta = \arccos\left(\frac{\mathbf{u} \cdot \mathbf{v}}{|\mathbf{u}|\,|\mathbf{v}|}\right)$

### `outer_vec_tensor2(v, A) -> sp.Array`

Outer product of a vector with a rank-2 tensor: $(\mathbf{v} \otimes \mathbf{A})_{ijk} = v_i A_{jk}$

| Parameter | Type | Description |
|-----------|------|-------------|
| `v` | array-like | 3-component vector |
| `A` | `Tensor2` or `sp.Matrix` | 3×3 rank-2 tensor |

Returns `sp.Array` with shape (3, 3, 3).

---

## `constkit.index_notation` — Index Notation Helpers

### `kronecker(i: int, j: int) -> int`

Kronecker delta: $\delta_{ij}$. Returns 1 if `i == j`, else 0.

### `levi_civita(i: int, j: int, k: int) -> int`

Levi-Civita permutation symbol: $\varepsilon_{ijk}$. Uses **1-based** indices.

| Value | Condition |
|-------|-----------|
| +1 | even permutation of (1,2,3) |
| -1 | odd permutation of (1,2,3) |
| 0 | any two indices equal |

### `epsilon_delta_identity(j: int, k: int, m: int, n: int) -> int`

The epsilon-delta identity: $\varepsilon_{ijk}\varepsilon_{imn} = \delta_{jm}\delta_{kn} - \delta_{jn}\delta_{km}$

Uses **1-based** indices.

### `kronecker_matrix() -> sp.Matrix`

Returns `sp.eye(3)` — the Kronecker delta as a 3×3 identity matrix.

### `levi_civita_tensor() -> sp.Array`

Returns the full 3×3×3 Levi-Civita tensor as a SymPy Array. Internally 0-indexed, values correspond to 1-based definition.

---

## `constkit.invariants` — Principal and Deviatoric Invariants

### `I1(A: Tensor2 | sp.Matrix) -> sp.Expr`

$$I_1(\mathbf{A}) = \mathrm{tr}(\mathbf{A}) = A_{ii}$$

### `I2(A: Tensor2 | sp.Matrix) -> sp.Expr`

$$I_2(\mathbf{A}) = \tfrac{1}{2}\left[(\mathrm{tr}\,\mathbf{A})^2 - \mathrm{tr}(\mathbf{A}^2)\right]$$

### `I3(A: Tensor2 | sp.Matrix) -> sp.Expr`

$$I_3(\mathbf{A}) = \det(\mathbf{A})$$

### `J2_invariant(A: Tensor2) -> sp.Expr`

Second deviatoric invariant:

$$J_2 = \tfrac{1}{2}\,\mathbf{A}' : \mathbf{A}', \quad \mathbf{A}' = \mathrm{dev}(\mathbf{A})$$

Used in von Mises yield criteria.

### `J3_invariant(A: Tensor2) -> sp.Expr`

Third deviatoric invariant: $J_3 = \det(\mathrm{dev}(\mathbf{A}))$

### `invariant_derivative(n: int, A: Tensor2) -> Tensor2`

Closed-form derivative of the $n$-th principal invariant w.r.t. $\mathbf{A}$:

| $n$ | Formula | Result |
|-----|---------|--------|
| 1 | $\partial I_1 / \partial \mathbf{A} = \mathbf{I}$ | Identity |
| 2 | $\partial I_2 / \partial \mathbf{A} = I_1 \mathbf{I} - \mathbf{A}^T$ | |
| 3 | $\partial I_3 / \partial \mathbf{A} = I_3 \mathbf{A}^{-T}$ | Cofactor |

Raises `ValueError` if $n \notin \{1, 2, 3\}$.

---

## `constkit.transforms` — Rotation Matrices

### `rotation_matrix(angle, axis: int = 3) -> sp.Matrix`

Rotation matrix about a coordinate axis (counterclockwise).

| Parameter | Type | Description |
|-----------|------|-------------|
| `angle` | `sp.Expr` or float | Rotation angle in radians |
| `axis` | `int` | 1 (x), 2 (y), or 3 (z) |

Returns a 3×3 orthogonal matrix with $\det(\mathbf{Q}) = +1$.

**Axis 3 (z-axis):**

$$\mathbf{Q} = \begin{pmatrix} \cos\theta & \sin\theta & 0 \\ -\sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{pmatrix}$$

### `rotation_matrix_axis_angle(axis_vec: sp.Matrix, angle) -> sp.Matrix`

Rotation matrix from axis-angle representation via Rodrigues' formula:

$$\mathbf{Q} = \mathbf{I} + \sin\theta\, \mathbf{N} + (1 - \cos\theta)\, \mathbf{N}^2$$

where $\mathbf{N}$ is the skew-symmetric matrix of the unit axis vector.

| Parameter | Type | Description |
|-----------|------|-------------|
| `axis_vec` | `sp.Matrix` | 3×1 unit vector defining rotation axis |
| `angle` | `sp.Expr` | Rotation angle in radians |

---

## `constkit.kinematics` — Deformation and Strain

### `deformation_gradient(mode: str, **params) -> Tensor2`

Construct a symbolic deformation gradient for a named deformation mode.

| Mode | Parameters | $\mathbf{F}$ |
|------|-----------|---|
| `'simple_shear'` | `k` | $\begin{pmatrix}1 & k & 0\\0 & 1 & 0\\0 & 0 & 1\end{pmatrix}$ |
| `'uniaxial'` | `lam` | $\mathrm{diag}(\lambda, 1/\sqrt{\lambda}, 1/\sqrt{\lambda})$ |
| `'biaxial'` | `lam1`, `lam2` | $\mathrm{diag}(\lambda_1, \lambda_2, 1/\lambda_1\lambda_2)$ |
| `'rotation'` | `theta`, `axis` (default 3) | Rotation matrix |
| `'custom'` | `F` (3×3 matrix) | User-provided |

### `right_cauchy_green(F) -> Tensor2`

$$\mathbf{C} = \mathbf{F}^T \mathbf{F}$$

Reference-configuration (material) deformation tensor.

### `left_cauchy_green(F) -> Tensor2`

$$\mathbf{b} = \mathbf{F}\,\mathbf{F}^T$$

Current-configuration (spatial/Finger) deformation tensor.

### `green_lagrange(F) -> Tensor2`

$$\mathbf{E} = \tfrac{1}{2}(\mathbf{C} - \mathbf{I}) = \tfrac{1}{2}(\mathbf{F}^T\mathbf{F} - \mathbf{I})$$

Reference-configuration finite strain measure.

### `almansi(F) -> Tensor2`

$$\mathbf{e} = \tfrac{1}{2}(\mathbf{I} - \mathbf{b}^{-1}) = \tfrac{1}{2}(\mathbf{I} - (\mathbf{F}\mathbf{F}^T)^{-1})$$

Current-configuration (Euler-Almansi) finite strain measure.

### `infinitesimal_strain(F) -> Tensor2`

$$\boldsymbol{\varepsilon} = \mathrm{sym}(\mathbf{F} - \mathbf{I}) = \tfrac{1}{2}(\nabla\mathbf{u} + \nabla\mathbf{u}^T)$$

Valid only for small deformations ($|\nabla\mathbf{u}| \ll 1$).

### `jacobian(F) -> sp.Expr`

$$J = \det(\mathbf{F})$$

Volume ratio between current and reference configurations. Must be $> 0$.

**Note:** This function is defined in `kinematics.py` but is not re-exported from `__init__.py`. Import it directly:

```python
from constkit.kinematics import jacobian
```

### `polar_decomposition(F) -> dict`

$$\mathbf{F} = \mathbf{R}\,\mathbf{U} = \mathbf{V}\,\mathbf{R}$$

Computes $\mathbf{U} = \sqrt{\mathbf{C}}$, $\mathbf{R} = \mathbf{F}\mathbf{U}^{-1}$, $\mathbf{V} = \mathbf{R}\mathbf{U}\mathbf{R}^T$.

**Returns:** `dict` with keys:

| Key | Type | Description |
|-----|------|-------------|
| `'R'` | `Tensor2` | Rotation tensor ($\mathbf{R}^T\mathbf{R} = \mathbf{I}$, $\det\mathbf{R} = 1$) |
| `'U'` | `Tensor2` | Right stretch tensor (symmetric, positive-definite, reference config) |
| `'V'` | `Tensor2` | Left stretch tensor (symmetric, positive-definite, current config) |
| `'stretches'` | `list` | Principal stretches $\lambda_i$ (eigenvalues of $\mathbf{U}$) |

### `isochoric_split(F) -> dict`

$$\mathbf{F} = J^{1/3}\,\bar{\mathbf{F}}, \quad \det\bar{\mathbf{F}} = 1$$

**Returns:** `dict` with keys:

| Key | Type | Description |
|-----|------|-------------|
| `'F_bar'` | `Tensor2` | Isochoric deformation gradient |
| `'J'` | `sp.Expr` | Jacobian determinant |
| `'C_bar'` | `Tensor2` | Isochoric right Cauchy-Green tensor ($\bar{\mathbf{C}} = J^{-2/3}\mathbf{C}$) |

### `principal_stretches(F) -> list`

Principal stretches: $\lambda_i = \sqrt{\text{eigenvalue}_i(\mathbf{C})}$.

Returns a list of symbolic expressions.

### `velocity_gradient(F, F_dot) -> dict`

$$\mathbf{L} = \dot{\mathbf{F}}\,\mathbf{F}^{-1}, \quad \mathbf{D} = \mathrm{sym}(\mathbf{L}), \quad \mathbf{W} = \mathrm{skew}(\mathbf{L})$$

| Parameter | Type | Description |
|-----------|------|-------------|
| `F` | `Tensor2` | Deformation gradient |
| `F_dot` | `Tensor2` | Time derivative of F |

**Returns:** `dict` with keys `'L'` (velocity gradient), `'D'` (rate of deformation), `'W'` (spin tensor).

### `metric_from_F(F) -> Tensor2`

Reference-configuration metric tensor (pull-back of the Euclidean metric):

$$\mathbf{G} = \mathbf{F}^T\mathbf{F}$$

Identical numerically to the right Cauchy-Green tensor, but named `'G'` for use in curvilinear reference-frame computations. Symmetric and positive-definite.

### `spatial_metric_from_F(F) -> Tensor2`

Current-configuration (spatial) metric tensor (push-forward of the Euclidean metric):

$$\mathbf{g} = \mathbf{F}\mathbf{F}^T$$

Identical numerically to the left Cauchy-Green tensor, but named `'g'`. Symmetric and positive-definite.

### `metric_from_F_inv(F) -> Tensor2`

Contravariant (inverse) reference metric:

$$\mathbf{G}^{-1} = \mathbf{F}^{-1}\mathbf{F}^{-T}$$

Satisfies $\mathbf{G}\,\mathbf{G}^{-1} = \mathbf{I}$. Named `'G_inv'`.

Note: $(\mathbf{F}^T\mathbf{F})^{-1} = \mathbf{F}^{-1}\mathbf{F}^{-T}$, which differs from $\mathbf{F}^{-T}\mathbf{F}^{-1} = (\mathbf{F}\mathbf{F}^T)^{-1}$ (the inverse spatial metric).

---

## `constkit.coordinates` — Curvilinear Coordinates

### `cylindrical_basis(r, phi) -> list[sp.Matrix]`

Covariant basis vectors for cylindrical coordinates $(r, \phi, z)$.

Mapping: $x^1 = r\cos\phi$, $x^2 = r\sin\phi$, $x^3 = z$.

**Returns:** `[g_r, g_φ, g_z]` — three 3×1 column vectors.

$$\mathbf{g}_r = (\cos\phi, \sin\phi, 0)^T, \quad \mathbf{g}_\phi = (-r\sin\phi, r\cos\phi, 0)^T, \quad \mathbf{g}_z = (0, 0, 1)^T$$

### `spherical_basis(R, theta, phi) -> list[sp.Matrix]`

Covariant basis vectors for spherical coordinates $(R, \theta, \phi)$.

Mapping: $x^1 = R\sin\theta\cos\phi$, $x^2 = R\sin\theta\sin\phi$, $x^3 = R\cos\theta$.

**Returns:** `[g_R, g_θ, g_φ]` — three 3×1 column vectors.

### `metric_tensor(g_cov: list[sp.Matrix]) -> sp.Matrix`

Metric tensor from covariant basis vectors:

$$g_{ij} = \mathbf{g}_i \cdot \mathbf{g}_j$$

Returns a 3×3 symmetric positive-definite SymPy Matrix.

### `contravariant_basis(g_cov: list[sp.Matrix]) -> list[sp.Matrix]`

Contravariant (reciprocal) basis vectors satisfying $\mathbf{g}_i \cdot \mathbf{g}^j = \delta^j_i$.

$$\mathbf{g}^1 = \frac{\mathbf{g}_2 \times \mathbf{g}_3}{\mathbf{g}_1 \cdot (\mathbf{g}_2 \times \mathbf{g}_3)}, \quad \text{(cyclic)}$$

Raises `ValueError` if the basis is linearly dependent.

### `verify_biorthogonality(g_cov, g_contra) -> bool`

Checks all 9 dot products $\mathbf{g}_i \cdot \mathbf{g}^j = \delta^j_i$.

Returns `True` if all pass. Raises `AssertionError` with details if any product fails.

### `christoffel_symbols(g_cov, coords) -> sp.Array`

Christoffel symbols of the second kind:

$$\Gamma^k_{ij} = \tfrac{1}{2}\,g^{km}\left(\frac{\partial g_{mj}}{\partial \theta^i} + \frac{\partial g_{im}}{\partial \theta^j} - \frac{\partial g_{ij}}{\partial \theta^m}\right)$$

| Parameter | Type | Description |
|-----------|------|-------------|
| `g_cov` | `list[sp.Matrix]` | Covariant basis vectors |
| `coords` | `list[sp.Symbol]` | Coordinate symbols $[\theta^1, \theta^2, \theta^3]$ |

Returns `sp.Array` with shape (3, 3, 3) where `result[k, i, j]` = $\Gamma^k_{ij}$.

### `covariant_derivative_vector(v, christoffel, coord_symbols, variant) -> sp.Array`

Covariant derivative of a vector field.

| Parameter | Type | Description |
|-----------|------|-------------|
| `v` | `sp.Matrix` | 3×1 vector of components |
| `christoffel` | `sp.Array` | Christoffel symbols (3×3×3), as from `christoffel_symbols` |
| `coord_symbols` | `list[sp.Symbol]` | Coordinate symbols |
| `variant` | `str` | `'contravariant'` (default) or `'covariant'` |

**Contravariant** (upper-index vector $v^j$):
$$(\nabla_i v)^j = \frac{\partial v^j}{\partial \theta^i} + \Gamma^j_{ik}\,v^k$$

**Covariant** (lower-index covector $v_j$):
$$(\nabla_i v)_j = \frac{\partial v_j}{\partial \theta^i} - \Gamma^k_{ij}\,v_k$$

Returns `sp.Array` with shape (3, 3) where `result[i, j]` = $(\nabla_i v)[j]$.

### `covariant_derivative_tensor2(T, christoffel, coord_symbols, variant) -> sp.Array`

Covariant derivative of a rank-2 tensor field.

| Parameter | Type | Description |
|-----------|------|-------------|
| `T` | `Tensor2` or `sp.Matrix` | 3×3 tensor components |
| `christoffel` | `sp.Array` | Christoffel symbols (3×3×3) |
| `coord_symbols` | `list[sp.Symbol]` | Coordinate symbols |
| `variant` | `str` | `'contravariant'`, `'covariant'`, or `'mixed'` (default: `'contravariant'`) |

**Contravariant** ($T^{ij}$):
$$(\nabla_k T)^{ij} = \frac{\partial T^{ij}}{\partial \theta^k} + \Gamma^i_{km}\,T^{mj} + \Gamma^j_{km}\,T^{im}$$

**Covariant** ($T_{ij}$):
$$(\nabla_k T)_{ij} = \frac{\partial T_{ij}}{\partial \theta^k} - \Gamma^m_{ki}\,T_{mj} - \Gamma^m_{kj}\,T_{im}$$

**Mixed** ($T^i{}_j$):
$$(\nabla_k T)^i{}_j = \frac{\partial T^i{}_j}{\partial \theta^k} + \Gamma^i_{km}\,T^m{}_j - \Gamma^m_{kj}\,T^i{}_m$$

Returns `sp.Array` with shape (3, 3, 3) where `result[k, i, j]` = $(\nabla_k T)[i, j]$.

---

## `constkit.stress` — Stress Measure Conversions

All functions accept `Tensor2` for stress and deformation gradient arguments.

### `cauchy_to_pk1(sigma, F) -> Tensor2`

$$\mathbf{P} = J\,\boldsymbol{\sigma}\,\mathbf{F}^{-T}$$

1st Piola-Kirchhoff stress (two-point tensor, generally non-symmetric).

### `cauchy_to_pk2(sigma, F) -> Tensor2`

$$\mathbf{S} = J\,\mathbf{F}^{-1}\,\boldsymbol{\sigma}\,\mathbf{F}^{-T}$$

2nd Piola-Kirchhoff stress (reference-config, symmetric).

### `cauchy_to_kirchhoff(sigma, F) -> Tensor2`

$$\boldsymbol{\tau} = J\,\boldsymbol{\sigma}$$

Kirchhoff stress (current-config, symmetric).

### `pk2_to_cauchy(S, F) -> Tensor2`

$$\boldsymbol{\sigma} = J^{-1}\,\mathbf{F}\,\mathbf{S}\,\mathbf{F}^T$$

### `pk1_to_cauchy(P, F) -> Tensor2`

$$\boldsymbol{\sigma} = J^{-1}\,\mathbf{P}\,\mathbf{F}^T$$

### `verify_stress_transformations(sigma, F, Q=None, sigma_star=None) -> dict`

Verify the standard stress transformation laws under a superposed rigid-body rotation.

| Parameter | Type | Description |
|-----------|------|-------------|
| `sigma` | `Tensor2` | Cauchy stress |
| `F` | `Tensor2` | Deformation gradient |
| `Q` | `sp.Matrix` or `None` | Rotation matrix. If `None`, uses a symbolic rotation about z. |
| `sigma_star` | `Tensor2`, `sp.Matrix`, or `None` | Optional rotated Cauchy stress to compare against $Q\sigma Q^T$ |

**Returns:** `dict` with keys:

| Key | Type | Meaning |
|-----|------|---------|
| `'sigma_matches_rotation'` | `bool` | $\boldsymbol{\sigma}^* = \mathbf{Q}\boldsymbol{\sigma}\mathbf{Q}^T$ |
| `'tau_matches_rotation'` | `bool` | $\boldsymbol{\tau}^* = \mathbf{Q}\boldsymbol{\tau}\mathbf{Q}^T$ |
| `'S_invariant'` | `bool` | $\mathbf{S}^* = \mathbf{S}$ |
| `'P_two_point'` | `bool` | $\mathbf{P}^* = \mathbf{Q}\mathbf{P}$ |
| `'details'` | `dict` | Difference matrices (should be zero) |

If `sigma_star` is omitted, the function uses the theoretical transform
$Q\sigma Q^T$ and checks internal consistency of the conversion formulas.
If `sigma_star` is supplied, the Cauchy-stress check becomes a genuine
constitutive objectivity test.

### `check_objectivity(sigma, F, Q=None, sigma_star=None) -> dict`

Deprecated alias for `verify_stress_transformations(...)`.

---

## `constkit.rates` — Objective Stress Rates

All three rate functions share the same signature pattern.

### `jaumann_rate(T, T_dot, W) -> Tensor2`

Jaumann (corotational) rate:

$$\overset{\triangledown}{\mathbf{T}} = \dot{\mathbf{T}} - \mathbf{W}\mathbf{T} + \mathbf{T}\mathbf{W}$$

| Parameter | Type | Description |
|-----------|------|-------------|
| `T` | `Tensor2` | Objective tensor (e.g. Cauchy stress) |
| `T_dot` | `Tensor2` | Material time derivative of T |
| `W` | `Tensor2` | Spin tensor (skew-symmetric part of velocity gradient) |

### `oldroyd_rate(T, T_dot, L) -> Tensor2`

Oldroyd (upper-convected) rate:

$$\overset{\triangle}{\mathbf{T}} = \dot{\mathbf{T}} - \mathbf{L}\mathbf{T} - \mathbf{T}\mathbf{L}^T$$

| Parameter | Type | Description |
|-----------|------|-------------|
| `T` | `Tensor2` | Objective tensor |
| `T_dot` | `Tensor2` | Material time derivative |
| `L` | `Tensor2` | Velocity gradient |

### `truesdell_rate(T, T_dot, L) -> Tensor2`

Truesdell rate:

$$\overset{\circ}{\mathbf{T}} = \dot{\mathbf{T}} - \mathbf{L}\mathbf{T} - \mathbf{T}\mathbf{L}^T + (\mathrm{tr}\,\mathbf{L})\,\mathbf{T}$$

Same parameter pattern as `oldroyd_rate`.

---

## `constkit.calculus` — Tensor Calculus

### `tensor_derivative(expr: sp.Expr, A: Tensor2) -> sp.Matrix`

Derivative of a scalar expression with respect to a tensor:

$$\left(\frac{\partial f}{\partial \mathbf{A}}\right)_{ij} = \frac{\partial f}{\partial A_{ij}}$$

| Parameter | Type | Description |
|-----------|------|-------------|
| `expr` | `sp.Expr` | Scalar symbolic expression depending on elements of A |
| `A` | `Tensor2` | The tensor to differentiate w.r.t. |

Returns a 3×3 `sp.Matrix`.

### `invariant_gradient(n: int, A: Tensor2) -> Tensor2`

Alias for `constkit.invariants.invariant_derivative`. See that entry for details.

### `symmetric_identity_4th() -> sp.Array`

Fourth-order symmetric identity tensor:

$$\mathbb{I}^s_{ijkl} = \tfrac{1}{2}(\delta_{ik}\delta_{jl} + \delta_{il}\delta_{jk})$$

This is $\partial \mathbf{A} / \partial \mathbf{A}$ for a **symmetric** tensor.

Returns `sp.Array` with shape (3, 3, 3, 3).

### `identity_4th() -> sp.Array`

Fourth-order identity tensor (non-symmetric):

$$\mathbb{I}_{ijkl} = \delta_{ik}\delta_{jl}$$

This is $\partial \mathbf{A} / \partial \mathbf{A}$ for a **general** (non-symmetric) tensor.

Returns `sp.Array` with shape (3, 3, 3, 3).

### `push_forward(S: Tensor2, F: Tensor2) -> Tensor2`

Push-forward of a **covariant** 2-tensor from reference to current configuration:

$$\varphi_*(\mathbf{S}) = \mathbf{F}^{-T}\,\mathbf{S}\,\mathbf{F}^{-1}$$

This is the generic component transformation for a covariant tensor. It
is not the PK2-to-Kirchhoff stress conversion.

### `pull_back(t: Tensor2, F: Tensor2) -> Tensor2`

Pull-back of a **contravariant** 2-tensor from current to reference configuration:

$$\varphi^*(\mathbf{t}) = \mathbf{F}^T\,\mathbf{t}\,\mathbf{F}$$

This is the generic component transformation for a contravariant tensor.
It is not the Kirchhoff-to-PK2 stress conversion.

### `push_forward_vector(v: sp.Matrix, F: Tensor2) -> sp.Matrix`

Push-forward of a contravariant vector (reference → current):

$$\mathbf{v}_\text{spatial} = \mathbf{F}\,\mathbf{v}_\text{material}$$

### `pull_back_vector(v: sp.Matrix, F: Tensor2) -> sp.Matrix`

Pull-back of a contravariant vector (current → reference):

$$\mathbf{v}_\text{material} = \mathbf{F}^{-1}\,\mathbf{v}_\text{spatial}$$

### `push_forward_tensor2(T: Tensor2, F: Tensor2) -> Tensor2`

Contravariant push-forward of a rank-2 tensor (reference → current):

$$\mathbf{t} = \mathbf{F}\,\mathbf{T}\,\mathbf{F}^T$$

Used e.g. to map the 2nd PK stress $\mathbf{S}$ to the Kirchhoff stress $\boldsymbol{\tau} = \mathbf{F}\mathbf{S}\mathbf{F}^T$.

### `pull_back_tensor2(t: Tensor2, F: Tensor2) -> Tensor2`

Contravariant pull-back of a rank-2 tensor (current → reference):

$$\mathbf{T} = \mathbf{F}^{-1}\,\mathbf{t}\,\mathbf{F}^{-T}$$

Inverse of `push_forward_tensor2`.

### `push_forward_covariant(T: Tensor2, F: Tensor2) -> Tensor2`

Covariant push-forward (reference → current). Alias for `push_forward`:

$$\mathbf{t} = \mathbf{F}^{-T}\,\mathbf{T}\,\mathbf{F}^{-1}$$

### `pull_back_covariant(t: Tensor2, F: Tensor2) -> Tensor2`

Covariant pull-back (current → reference). Alias for `pull_back`:

$$\mathbf{T} = \mathbf{F}^T\,\mathbf{t}\,\mathbf{F}$$

### `push_forward_tensor4(C4: sp.Array, F: Tensor2) -> sp.Array`

Piola push-forward of a 4th-order tensor (reference → current):

$$c_{ijkl} = \frac{1}{J}\,F_{iI}\,F_{jJ}\,F_{kK}\,F_{lL}\,C_{IJKL}$$

Maps reference-config material tangent moduli $\mathbb{C}$ to spatial tangent moduli $\mathbb{c}$.

| Parameter | Type | Description |
|-----------|------|-------------|
| `C4` | `sp.Array` | Shape (3,3,3,3) — reference-config 4th-order tensor |
| `F` | `Tensor2` | Deformation gradient |

Returns `sp.Array` with shape (3, 3, 3, 3).

### `pull_back_tensor4(c4: sp.Array, F: Tensor2) -> sp.Array`

Piola pull-back of a 4th-order tensor (current → reference):

$$C_{IJKL} = J\,(F^{-1})_{Ii}\,(F^{-1})_{Jj}\,(F^{-1})_{Kk}\,(F^{-1})_{Ll}\,c_{ijkl}$$

Inverse of `push_forward_tensor4`. Returns `sp.Array` with shape (3, 3, 3, 3).

### `raise_index_vector(v: sp.Matrix, g_contra: sp.Matrix) -> sp.Matrix`

Raise the index of a covariant vector: $v^i = g^{ij} v_j$

### `lower_index_vector(v: sp.Matrix, g_cov: sp.Matrix) -> sp.Matrix`

Lower the index of a contravariant vector: $v_i = g_{ij} v^j$

### `raise_index(T_cov: Tensor2, g_contra: sp.Matrix) -> Tensor2`

Raise both indices of a covariant rank-2 tensor:

$$T^{ij} = g^{ik}\,g^{jl}\,T_{kl}$$

Equivalent to $\mathbf{g}^{-1}\,\mathbf{T}\,\mathbf{g}^{-1}$ when $g$ is symmetric.

### `lower_index(T_cont: Tensor2, g_cov: sp.Matrix) -> Tensor2`

Lower both indices of a contravariant rank-2 tensor:

$$T_{ij} = g_{ik}\,g_{jl}\,T^{kl}$$

### `raise_one_index(T: Tensor2, g_contra: sp.Matrix, which) -> Tensor2`

Raise a single index of a mixed rank-2 tensor.

| `which` | Formula |
|---------|---------|
| `'first'` | $T^i{}_j = g^{ik}\,T_{kj}$ |
| `'second'` | $T_i{}^j = T_{ik}\,g^{kj}$ |

### `raise_index_tensor4(C4: sp.Array, g_contra: sp.Matrix) -> sp.Array`

Raise all four indices of a covariant 4th-order tensor:

$$C^{ijkl} = g^{iI}\,g^{jJ}\,g^{kK}\,g^{lL}\,C_{IJKL}$$

Returns `sp.Array` with shape (3, 3, 3, 3).

### `lower_index_tensor4(c4: sp.Array, g_cov: sp.Matrix) -> sp.Array`

Lower all four indices of a contravariant 4th-order tensor:

$$C_{ijkl} = g_{iI}\,g_{jJ}\,g_{kK}\,g_{lL}\,C^{IJKL}$$

Returns `sp.Array` with shape (3, 3, 3, 3).

### `tensor2_derivative(A: Tensor2, B: Tensor2) -> sp.Array`

Derivative of a rank-2 tensor with respect to another rank-2 tensor (4th-order result):

$$\left(\frac{\partial \mathbf{A}}{\partial \mathbf{B}}\right)_{ijkl} = \frac{\partial A_{ij}}{\partial B_{kl}}$$

| Parameter | Type | Description |
|-----------|------|-------------|
| `A` | `Tensor2` | Numerator tensor |
| `B` | `Tensor2` | Denominator tensor |

Returns `sp.Array` with shape (3, 3, 3, 3). When `A = B`, the result is the 4th-order identity `identity_4th()`.

### `scalar_tensor_derivative(f: sp.Expr, A: Tensor2) -> Tensor2`

Derivative of a scalar expression with respect to a tensor, returned as a `Tensor2`:

$$\left(\frac{\partial f}{\partial \mathbf{A}}\right)_{ij} = \frac{\partial f}{\partial A_{ij}}$$

| Parameter | Type | Description |
|-----------|------|-------------|
| `f` | `sp.Expr` | Scalar symbolic expression depending on components of A |
| `A` | `Tensor2` | The tensor to differentiate w.r.t. |

Returns a `Tensor2`. Equivalent to `tensor_derivative` but enables chained tensor operations.

```python
# ∂(tr A)/∂A = I
f = A.trace()
dfdA = scalar_tensor_derivative(f, A)  # → identity Tensor2
```

### `double_contract_4_2(C4: sp.Array, A: Tensor2) -> Tensor2`

Right double contraction of a rank-4 tensor with a rank-2 tensor:

$$B_{ij} = \mathbb{C}_{ijkl}\, A_{kl}$$

This is the standard constitutive operation $\boldsymbol{\sigma} = \mathbb{C}:\boldsymbol{\varepsilon}$.

| Parameter | Type | Description |
|-----------|------|-------------|
| `C4` | `sp.Array` | Shape (3,3,3,3) fourth-order tensor |
| `A` | `Tensor2` | Rank-2 tensor (e.g. strain) |

Returns `Tensor2`.

### `double_contract_2_4(A: Tensor2, C4: sp.Array) -> Tensor2`

Left double contraction of a rank-2 tensor with a rank-4 tensor:

$$B_{kl} = A_{ij}\, \mathbb{C}_{ijkl}$$

| Parameter | Type | Description |
|-----------|------|-------------|
| `A` | `Tensor2` | Rank-2 tensor |
| `C4` | `sp.Array` | Shape (3,3,3,3) fourth-order tensor |

Returns `Tensor2`.

### `double_contract_4_4(C4: sp.Array, D4: sp.Array) -> sp.Expr`

Full double contraction of two rank-4 tensors to a scalar:

$$s = \mathbb{C}_{ijkl}\, \mathbb{D}_{ijkl}$$

| Parameter | Type | Description |
|-----------|------|-------------|
| `C4` | `sp.Array` | Shape (3,3,3,3) fourth-order tensor |
| `D4` | `sp.Array` | Shape (3,3,3,3) fourth-order tensor |

Returns `sp.Expr` scalar.

### `scalar_tensor2_derivative(f: sp.Expr, A: Tensor2, B: Tensor2) -> sp.Array`

Second mixed derivative of a scalar with respect to two rank-2 tensors:

$$\left(\frac{\partial^2 f}{\partial \mathbf{A}\,\partial \mathbf{B}}\right)_{ijkl} = \frac{\partial^2 f}{\partial A_{ij}\,\partial B_{kl}}$$

| Parameter | Type | Description |
|-----------|------|-------------|
| `f` | `sp.Expr` | Scalar symbolic expression depending on both A and B |
| `A` | `Tensor2` | First differentiation tensor |
| `B` | `Tensor2` | Second differentiation tensor |

Returns `sp.Array` with shape (3, 3, 3, 3). When `f = A:B`, the result is `identity_4th()`.

---

## `constkit.tensor4` — Fourth-Order Tensor Class

`Tensor4` wraps a `sp.Array` of shape (3,3,3,3) with continuum-mechanics methods.

### `Tensor4(array, name="")`

Constructor.

| Parameter | Type | Description |
|-----------|------|-------------|
| `array` | `sp.Array`, `sp.MutableDenseNDimArray`, or array-like | Shape (3,3,3,3) components |
| `name` | `str` | Display name for LaTeX rendering |

Raises `ValueError` if shape is not (3,3,3,3).

#### `Tensor4.name -> str`

`@property` — display name, mutable via assignment.

#### `Tensor4.array -> sp.Array`

`@property` — underlying (3,3,3,3) SymPy Array.

#### `Tensor4.double_contract(B: Tensor2) -> Tensor2`

Right double contraction: $(C:A)_{ij} = C_{ijkl} A_{kl}$.

#### `Tensor4.rdouble_contract(A: Tensor2) -> Tensor2`

Left double contraction: $(A:C)_{kl} = A_{ij} C_{ijkl}$.

#### `Tensor4.full_contract(D: Tensor4) -> sp.Expr`

Full double contraction to scalar: $s = C_{ijkl} D_{ijkl}$.

#### `Tensor4.push_forward(F: Tensor2) -> Tensor4`

Piola push-forward: $c_{ijkl} = J^{-1} F_{iI} F_{jJ} F_{kK} F_{lL} C_{IJKL}$.

#### `Tensor4.pull_back(F: Tensor2) -> Tensor4`

Piola pull-back: $C_{IJKL} = J F^{-1}_{Ii} F^{-1}_{Jj} F^{-1}_{Kk} F^{-1}_{Ll} c_{ijkl}$.

#### `Tensor4.minor_symmetrize() -> Tensor4`

Minor symmetrization: $C_{(ij)(kl)} = \frac{1}{4}(C_{ijkl} + C_{jikl} + C_{ijlk} + C_{jilk})$.

#### `Tensor4.major_symmetrize() -> Tensor4`

Major symmetrization: $C_{ijkl}^\text{sym} = \frac{1}{2}(C_{ijkl} + C_{klij})$.

#### `Tensor4.subs(*args, **kwargs) -> Tensor4`

Substitute symbolic values component-wise.

#### `Tensor4.evaluate(**values) -> np.ndarray`

Substitute values and return a NumPy array of shape (3,3,3,3). Requires `numpy`.

#### `Tensor4.latex() -> str`

LaTeX string of the underlying array.

---

## `constkit.notation` — Voigt and Mandel Notation

Ordering convention (both Voigt and Mandel):

| Index α | Component |
|---------|-----------|
| 0 | (1,1) |
| 1 | (2,2) |
| 2 | (3,3) |
| 3 | (2,3) |
| 4 | (1,3) |
| 5 | (1,2) |

**Voigt** uses no scaling. **Mandel** multiplies off-diagonal entries (α ≥ 3) by $\sqrt{2}$, preserving the tensor inner product $\mathbf{T}:\mathbf{S} = \mathbf{m}_T^T \mathbf{m}_S$.

### `to_voigt(T: Tensor2) -> sp.Matrix`

Convert symmetric rank-2 tensor to 6-component Voigt vector. Returns shape (6,1).

### `from_voigt(v: sp.Matrix) -> Tensor2`

Reconstruct symmetric `Tensor2` from 6-component Voigt vector.

### `to_voigt_matrix(C4: sp.Array) -> sp.Matrix`

Convert rank-4 tensor to 6×6 Voigt stiffness matrix: $M_{\alpha\beta} = C_{ijkl}$.

### `from_voigt_matrix(M: sp.Matrix) -> sp.Array`

Reconstruct rank-4 tensor (with major and minor symmetry) from 6×6 Voigt matrix. Returns `sp.Array` of shape (3,3,3,3).

### `to_mandel(T: Tensor2) -> sp.Matrix`

Convert symmetric rank-2 tensor to 6-component Mandel vector ($\sqrt{2}$ scaling on off-diagonal). Returns shape (6,1).

### `from_mandel(v: sp.Matrix) -> Tensor2`

Reconstruct symmetric `Tensor2` from 6-component Mandel vector.

### `to_mandel_matrix(C4: sp.Array) -> sp.Matrix`

Convert rank-4 tensor to 6×6 Mandel stiffness matrix: $M_{\alpha\beta} = s_\alpha s_\beta C_{ijkl}$ where $s_\alpha = 1$ (α < 3) or $\sqrt{2}$ (α ≥ 3). Preserves double contraction: $\mathbf{A}:\mathbb{C}:\mathbf{B} = \mathbf{m}_A^T \mathbf{M} \mathbf{m}_B$.

### `from_mandel_matrix(M: sp.Matrix) -> sp.Array`

Reconstruct rank-4 tensor from 6×6 Mandel matrix. Returns `sp.Array` of shape (3,3,3,3).

### `voigt_to_mandel(v: sp.Matrix) -> sp.Matrix`

Convert Voigt 6-vector to Mandel by multiplying off-diagonal entries by $\sqrt{2}$.

### `mandel_to_voigt(v: sp.Matrix) -> sp.Matrix`

Convert Mandel 6-vector to Voigt by dividing off-diagonal entries by $\sqrt{2}$.

### `voigt_matrix_to_mandel(C: sp.Matrix) -> sp.Matrix`

Convert 6×6 Voigt stiffness matrix to Mandel: $M_{\alpha\beta} = s_\alpha s_\beta V_{\alpha\beta}$.

### `mandel_matrix_to_voigt(M: sp.Matrix) -> sp.Matrix`

Convert 6×6 Mandel stiffness matrix to Voigt: $V_{\alpha\beta} = M_{\alpha\beta} / (s_\alpha s_\beta)$.

---

## `constkit.display` — Pretty Printing

### `show(obj, name="", simplify=True) -> None`

Pretty-print a tensor, matrix, or expression.

| Parameter | Type | Description |
|-----------|------|-------------|
| `obj` | `Tensor2`, `sp.Matrix`, `sp.Expr`, or `sp.Array` | Object to display |
| `name` | `str` | Label (e.g. `'E'`, `'σ'`) |
| `simplify` | `bool` | Apply `sp.simplify` before display |

In Jupyter: renders LaTeX via `IPython.display.Math`. In terminal: prints LaTeX string.

```python
ck.show(E, name='E')   # renders: E = [matrix]
```

### `compare(A: Tensor2, B: Tensor2, names=("A", "B")) -> None`

Display two tensors, their difference, and whether they are equal.

```python
from constkit.display import compare
compare(E_computed, E_expected, names=("computed", "expected"))
# Shows both, difference matrix, and ✓/✗
```

---

## Cross-Reference: Where to Find Each Operation

| Lecture Topic | Function(s) |
|---|---|
| Inner/dot product | `vector.dot` |
| Cross product | `vector.cross` |
| Scalar triple product | `vector.triple_product` |
| Outer product (vectors) | `vector.outer_product` |
| Vector norm, angle | `vector.norm`, `vector.angle` |
| Kronecker delta | `index_notation.kronecker`, `.kronecker_matrix()` |
| Levi-Civita symbol | `index_notation.levi_civita`, `.levi_civita_tensor()` |
| Epsilon-delta identity | `index_notation.epsilon_delta_identity` |
| Tensor trace, det, inv | `Tensor2.trace()`, `.det()`, `.inv()` |
| Transpose | `Tensor2.T` |
| Symmetric/skew decomposition | `Tensor2.sym()`, `.skew()` |
| Deviatoric part | `Tensor2.dev()` |
| Double contraction A:B | `Tensor2.double_contract(B)` |
| Single contraction A·B | `Tensor2.single_contract(B)` |
| Tensor outer product A⊗B | `Tensor2.outer(B)` |
| Principal invariants I₁, I₂, I₃ | `Tensor2.I1()`, `.I2()`, `.I3()` or `invariants.I1()` etc. |
| Deviatoric invariants J₂, J₃ | `invariants.J2_invariant`, `invariants.J3_invariant` |
| Invariant derivatives | `invariants.invariant_derivative`, `calculus.invariant_gradient` |
| Eigenvalues/eigenvectors | `Tensor2.eigendecomposition()` |
| Symmetric tensor functions (√, ln, exp) | `Tensor2.sqrt()`, `.log()`, `.exp()`, `.spectral_function(f)` |
| Rotation matrices | `transforms.rotation_matrix`, `.rotation_matrix_axis_angle` |
| Tensor rotation A' = QAQᵀ | `Tensor2.rotate(Q)` |
| Cylindrical coordinates | `coordinates.cylindrical_basis` |
| Spherical coordinates | `coordinates.spherical_basis` |
| Metric tensor | `coordinates.metric_tensor` |
| Contravariant basis | `coordinates.contravariant_basis` |
| Biorthogonality check | `coordinates.verify_biorthogonality` |
| Christoffel symbols | `coordinates.christoffel_symbols` |
| Deformation gradient F | `kinematics.deformation_gradient` |
| Right Cauchy-Green C | `kinematics.right_cauchy_green` |
| Left Cauchy-Green b | `kinematics.left_cauchy_green` |
| Green-Lagrange strain E | `kinematics.green_lagrange` |
| Almansi strain e | `kinematics.almansi` |
| Infinitesimal strain ε | `kinematics.infinitesimal_strain` |
| Jacobian J = det F | `kinematics.jacobian` |
| Polar decomposition F=RU=VR | `kinematics.polar_decomposition` |
| Isochoric split | `kinematics.isochoric_split` |
| Principal stretches | `kinematics.principal_stretches` |
| Velocity gradient L, D, W | `kinematics.velocity_gradient` |
| Cauchy ↔ PK1 | `stress.cauchy_to_pk1`, `stress.pk1_to_cauchy` |
| Cauchy ↔ PK2 | `stress.cauchy_to_pk2`, `stress.pk2_to_cauchy` |
| Cauchy → Kirchhoff | `stress.cauchy_to_kirchhoff` |
| Stress transformation check | `stress.verify_stress_transformations` |
| Jaumann rate | `rates.jaumann_rate` |
| Oldroyd rate | `rates.oldroyd_rate` |
| Truesdell rate | `rates.truesdell_rate` |
| ∂f/∂A (scalar→tensor) | `calculus.tensor_derivative` |
| 4th-order identity (sym) | `calculus.symmetric_identity_4th` |
| 4th-order identity (general) | `calculus.identity_4th` |
| Push-forward | `calculus.push_forward` |
| Pull-back | `calculus.pull_back` |
| Display / LaTeX | `display.show`, `display.compare` |
