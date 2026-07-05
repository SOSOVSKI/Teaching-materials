# constkit User Manual

A symbolic tensor algebra toolkit for the Constitutive Modeling course.

`constkit` lets you build, manipulate, and verify tensor expressions using the same notation as the lecture notes. Every operation is fully symbolic — you work with parameters like $k$, $\lambda$, and $\theta$, see LaTeX-rendered results in Jupyter, and substitute numbers only when you want to check a specific case.

---

## Installation

constkit requires **Python 3.10+** and **SymPy** (the only hard dependency).

**Option A — install from the course folder:**

```bash
cd course/
pip install -e .
```

**Option B — just add to your Python path** (no install needed):

```python
import sys
sys.path.insert(0, "/path/to/course")
import constkit as ck
```

**Option C — in a Jupyter notebook** (if the notebook is inside `course/`):

```python
import constkit as ck  # works if notebook is in or below course/
```

For numeric evaluation (parameter sweeps, plotting), install the optional dependencies:

```bash
pip install numpy scipy matplotlib
```

---

## Quick Start

```python
import constkit as ck
import sympy as sp

# Create a symbolic tensor
A = ck.Tensor2([[2, 1, 0],
                [1, 3, 1],
                [0, 1, 4]], name='A')

# Principal invariants
print("I₁ =", A.I1())   # 9
print("I₂ =", A.I2())   # 24
print("I₃ =", A.I3())   # 18

# Deviatoric part
A_dev = A.dev()
ck.show(A_dev, name="A'")

# Eigenvalues
eig = A.eigendecomposition()
for lam in eig['eigenvalues']:
    print("λ =", lam)
```

That's it. Every operation returns a SymPy expression — you can simplify, substitute, differentiate, or display it in LaTeX.

---

## Module Walkthrough

The modules follow the lecture progression. Work through them in order as the course advances.

### L01 — Vectors and Tensor Algebra

**Vectors** (`constkit.vector`):

```python
import constkit as ck
import sympy as sp

u = sp.Matrix([1, 2, 3])
v = sp.Matrix([4, 5, 6])

ck.dot(u, v)              # 32
ck.cross(u, v)            # Matrix([-3, 6, -3])
ck.norm(u)                # sqrt(14)
ck.triple_product(u, v, sp.Matrix([1, 0, 0]))  # scalar triple product
ck.outer_product(u, v)    # 3×3 matrix: u ⊗ v
```

**Index notation** (`constkit.index_notation`):

```python
ck.kronecker(1, 1)        # 1
ck.kronecker(1, 2)        # 0
ck.levi_civita(1, 2, 3)   # 1
ck.levi_civita(2, 1, 3)   # -1
ck.levi_civita(1, 1, 3)   # 0
```

**Second-order tensors** (`constkit.tensor2`):

```python
# Create from a matrix
S = ck.Tensor2([[2, 1, 0],
                [1, 3, 1],
                [0, 1, 4]], name='S')

# Core operations
S.trace()                  # 9
S.det()                    # 18
S.inv()                    # Tensor2 — the inverse
S.T                        # transpose
S.sym()                    # symmetric part: ½(S + Sᵀ)
S.skew()                   # skew part: ½(S − Sᵀ)
S.dev()                    # deviatoric: S − ⅓ tr(S) I

# Contractions
T = ck.Tensor2([[1, 0, 2],
                [0, 1, 0],
                [2, 0, 3]], name='T')

S.dot(sp.Matrix([1, 0, 0]))       # S·v — vector result
S.single_contract(T)               # S·T — tensor result
S.double_contract(T)               # S:T — scalar result (= 17)

# Invariants
S.I1(), S.I2(), S.I3()
```

**Coordinate transformations** (`constkit.transforms`):

```python
theta = sp.Symbol('theta')
Q = ck.rotation_matrix(theta, axis=3)   # rotation about z-axis

# Transform a tensor: S' = Q S Qᵀ
S_rot = S.rotate(Q)

# Verify invariants are preserved
sp.simplify(S_rot.I1() - S.I1())  # 0
sp.simplify(S_rot.I3() - S.I3())  # 0
```

**Spectral decomposition and tensor functions for symmetric tensors**:

```python
# Eigenvalues and eigenvectors
eig = S.eigendecomposition()
print(eig['eigenvalues'])    # [3 - sqrt(3), 3, 3 + sqrt(3)]
print(eig['eigenvectors'])   # list of normalized eigenvectors

# Tensor functions via spectral decomposition
# These helpers assume S is symmetric, so the eigenvectors are orthonormal.
sqrt_S = S.sqrt()            # √S
ln_S = S.log()               # ln(S)
exp_S = S.exp()              # exp(S)

# Verify: √S · √S = S
diff = (sqrt_S @ sqrt_S) - S
sp.simplify(diff.matrix)     # zero matrix
```

**Principal invariants and their derivatives** (`constkit.invariants`):

```python
# Free functions (work on Tensor2 or plain sp.Matrix)
ck.I1(S)    # tr(S)
ck.I2(S)    # ½[(tr S)² − tr(S²)]
ck.I3(S)    # det(S)

# Invariant derivatives (closed-form, from L02)
dI1 = ck.invariant_derivative(1, S)   # ∂I₁/∂S = I
dI2 = ck.invariant_derivative(2, S)   # ∂I₂/∂S = I₁ I − Sᵀ
dI3 = ck.invariant_derivative(3, S)   # ∂I₃/∂S = I₃ S⁻ᵀ
```

### L01 (continued) — Curvilinear Coordinates

```python
import constkit as ck
import sympy as sp

r, phi, z = sp.symbols('r phi z', positive=True)

# Covariant basis vectors for cylindrical coordinates
g_cov = ck.cylindrical_basis(r, phi)
ck.show(g_cov[0], name='g_r')    # [cos φ, sin φ, 0]
ck.show(g_cov[1], name='g_φ')    # [-r sin φ, r cos φ, 0]
ck.show(g_cov[2], name='g_z')    # [0, 0, 1]

# Metric tensor
g = ck.metric_tensor(g_cov)
ck.show(g, name='g_{ij}')        # diag(1, r², 1)

# Contravariant (reciprocal) basis
g_contra = ck.contravariant_basis(g_cov)

# Verify biorthogonality: g_i · gʲ = δⱼᵢ
ck.verify_biorthogonality(g_cov, g_contra)   # True
```

For spherical coordinates:

```python
R, theta, phi = sp.symbols('R theta phi', positive=True)
g_sph = ck.spherical_basis(R, theta, phi)
g_sph_metric = ck.metric_tensor(g_sph)
# diag(1, R², R² sin²θ)
```

### L02–L03 — Kinematics

**Deformation gradient presets**:

```python
import constkit as ck
import sympy as sp

k = sp.Symbol('k', positive=True)

# Simple shear
F = ck.deformation_gradient('simple_shear', k=k)
ck.show(F, name='F')
# ⎡1  k  0⎤
# ⎢0  1  0⎥
# ⎣0  0  1⎦

# Other modes
F_uni = ck.deformation_gradient('uniaxial', lam=sp.Symbol('lambda'))
F_bi  = ck.deformation_gradient('biaxial', lam1=sp.Rational(3,2), lam2=sp.Rational(4,3))
F_rot = ck.deformation_gradient('rotation', theta=sp.pi/6, axis=3)
F_cust = ck.deformation_gradient('custom', F=[[1, 0.5, 0], [0, 1, 0], [0, 0, 1]])
```

**Cauchy-Green tensors and strain measures**:

```python
C = ck.right_cauchy_green(F)     # C = Fᵀ F
b = ck.left_cauchy_green(F)      # b = F Fᵀ
E = ck.green_lagrange(F)         # E = ½(C − I)
e = ck.almansi(F)                # e = ½(I − b⁻¹)
eps = ck.infinitesimal_strain(F) # ε = sym(F − I)

# Compare strain measures at small vs large shear
E_small = E.subs(k, sp.Rational(1, 100))
e_small = e.subs(k, sp.Rational(1, 100))
# E₁₂ ≈ e₁₂ ≈ ε₁₂ = k/2  (they agree for small k)

E_large = E.subs(k, 1)
e_large = e.subs(k, 1)
# E₂₂ = +1/2,  e₂₂ = −1/2  (they diverge for large k!)
```

**Polar decomposition**:

```python
result = ck.polar_decomposition(F.subs(k, 1))
R = result['R']     # rotation tensor
U = result['U']     # right stretch (reference config)
V = result['V']     # left stretch (current config)
stretches = result['stretches']   # principal stretches λᵢ

# Verify: F = R U
diff = F.subs(k, 1).matrix - R.matrix * U.matrix
sp.simplify(diff)   # zero
```

**Velocity gradient**:

```python
k_dot = sp.Symbol('k_dot')
F_dot = ck.Tensor2(sp.Matrix([[0, k_dot, 0], [0, 0, 0], [0, 0, 0]]))

result = ck.velocity_gradient(F, F_dot)
L = result['L']    # velocity gradient: L = Ḟ F⁻¹
D = result['D']    # rate of deformation: D = sym(L)
W = result['W']    # spin tensor: W = skew(L)
```

### L03 — Stress Measures

```python
import constkit as ck
import sympy as sp

sigma_0 = sp.Symbol('sigma_0', positive=True)
k = sp.Symbol('k', positive=True)

F = ck.deformation_gradient('simple_shear', k=k)
sigma = ck.Tensor2(sigma_0 * sp.Matrix([[1,0,0],[0,0,0],[0,0,0]]))

# Convert between stress measures
P = ck.cauchy_to_pk1(sigma, F)      # P = J σ F⁻ᵀ
S = ck.cauchy_to_pk2(sigma, F)      # S = J F⁻¹ σ F⁻ᵀ
tau = ck.cauchy_to_kirchhoff(sigma, F)  # τ = J σ

# Reverse conversions
sigma_back = ck.pk2_to_cauchy(S, F)    # σ = J⁻¹ F S Fᵀ
sigma_back2 = ck.pk1_to_cauchy(P, F)   # σ = J⁻¹ P Fᵀ
```

**Stress transformation check**:

```python
result = ck.verify_stress_transformations(sigma, F)
print(result['sigma_matches_rotation'])   # True — σ* = Q σ Qᵀ
print(result['tau_matches_rotation'])     # True — τ* = Q τ Qᵀ
print(result['S_invariant'])              # True — S* = S
print(result['P_two_point'])              # True — P* = Q P
```

### L02 — Objective Rates

```python
import constkit as ck
import sympy as sp

# Cauchy stress and its material time derivative
sigma = ck.Tensor2(sp.Matrix([[100, 0, 0], [0, 0, 0], [0, 0, 0]]))
sigma_dot = ck.Tensor2(sp.zeros(3, 3))

# Spin tensor from simple shear
W = ck.Tensor2(sp.Matrix([[0, sp.Rational(1,2), 0],
                           [-sp.Rational(1,2), 0, 0],
                           [0, 0, 0]]))

# Jaumann rate: σ̊ = σ̇ − Wσ + σW
sigma_J = ck.jaumann_rate(sigma, sigma_dot, W)
ck.show(sigma_J)    # Non-zero even though σ̇ = 0!

# Oldroyd and Truesdell rates use L instead of W:
L = ck.Tensor2(sp.Matrix([[0, 1, 0], [0, 0, 0], [0, 0, 0]]))
sigma_O = ck.oldroyd_rate(sigma, sigma_dot, L)
sigma_Tr = ck.truesdell_rate(sigma, sigma_dot, L)
```

### L02 — Tensor Calculus

```python
import constkit as ck
import sympy as sp

# Create a symbolic tensor with explicit symbol entries
a, b, c, d, e, f = sp.symbols('a b c d e f')
A = ck.Tensor2([[a, d, e], [d, b, f], [e, f, c]], name='A')

# Derivative of a scalar w.r.t. a tensor
from constkit.calculus import tensor_derivative
tr_A = A.trace()
dtr = tensor_derivative(tr_A, A)   # ∂(tr A)/∂A = I

det_A = A.det()
ddet = tensor_derivative(det_A, A) # ∂(det A)/∂A = cofactor matrix

# Closed-form invariant gradients
dI1 = ck.invariant_gradient(1, A)  # I
dI2 = ck.invariant_gradient(2, A)  # I₁ I − Aᵀ
dI3 = ck.invariant_gradient(3, A)  # I₃ A⁻ᵀ

# Push-forward and pull-back of generic covariant/contravariant tensors
from constkit.calculus import push_forward, pull_back
F = ck.deformation_gradient('simple_shear', k=sp.Symbol('k'))
S = ck.Tensor2(sp.eye(3))   # some covariant reference-config tensor
s_pushed = push_forward(S, F)   # φ_*(S) = F⁻ᵀ S F⁻¹
S_pulled = pull_back(s_pushed, F)  # φ*(s) = Fᵀ s F  (recovers S)

# For stress measures, use the dedicated conversions instead:
tau = ck.cauchy_to_kirchhoff(sigma, F)  # τ = J σ
S_pk2 = ck.cauchy_to_pk2(sigma, F)      # S = J F⁻¹ σ F⁻ᵀ

# Fourth-order identity tensors
from constkit.calculus import symmetric_identity_4th, identity_4th
I_sym = symmetric_identity_4th()    # ∂A/∂A for symmetric A
I_full = identity_4th()             # ∂A/∂A for general A
```

---

## Common Workflows

### "Verify my hand calculation"

You solved a problem on paper and want to check the answer:

```python
import constkit as ck
import sympy as sp

# Your hand calculation: C for simple shear with k=1
# You got C = [[1, 1, 0], [1, 2, 0], [0, 0, 1]]

F = ck.deformation_gradient('simple_shear', k=1)
C = ck.right_cauchy_green(F)
print(C.matrix)
# Matrix([[1, 1, 0], [1, 2, 0], [0, 0, 1]])  ✓
```

### "Explore how a quantity depends on a parameter"

```python
import constkit as ck
import sympy as sp

k = sp.Symbol('k', positive=True)
F = ck.deformation_gradient('simple_shear', k=k)
E = ck.green_lagrange(F)

# See E₁₂ and E₂₂ as functions of k
print("E_12 =", E[0,1])   # k/2
print("E_22 =", E[1,1])   # k²/2

# The shear component is linear, the normal component is quadratic.
# That's why small-strain theory drops the k² term!
```

### "Check stress transformations or constitutive objectivity"

```python
import constkit as ck
import sympy as sp

k = sp.Symbol('k', positive=True)
sigma_0 = sp.Symbol('sigma_0', positive=True)
F = ck.deformation_gradient('simple_shear', k=k)
sigma = ck.Tensor2(sigma_0 * sp.Matrix([[1,0,0],[0,0,0],[0,0,0]]))

result = ck.verify_stress_transformations(sigma, F)
# result['sigma_matches_rotation'] → True  (σ* = Q σ Qᵀ)
# result['S_invariant']            → True  (2nd PK stress is invariant)
# result['P_two_point']            → True  (1st PK transforms as QP)

# You can also supply a specific rotation and a rotated constitutive response:
Q = ck.rotation_matrix(sp.pi/4, axis=3)
sigma_star = ck.Tensor2(Q * sigma.matrix * Q.T)
result = ck.verify_stress_transformations(sigma, F, Q, sigma_star=sigma_star)
```

### "Work in curvilinear coordinates"

```python
import constkit as ck
import sympy as sp
from constkit.coordinates import christoffel_symbols

r, phi, z = sp.symbols('r phi z', positive=True)

# Build the full coordinate machinery
g_cov = ck.cylindrical_basis(r, phi)
g_contra = ck.contravariant_basis(g_cov)
g_metric = ck.metric_tensor(g_cov)

# Verify the reciprocal basis is correct
ck.verify_biorthogonality(g_cov, g_contra)

# Christoffel symbols
Gamma = christoffel_symbols(g_cov, [r, phi, z])
# Γ^r_φφ = -r
print("Γ^r_φφ =", Gamma[0, 1, 1])
# Γ^φ_rφ = 1/r
print("Γ^φ_rφ =", Gamma[1, 0, 1])
```

---

## Tips for Jupyter Notebooks

### LaTeX rendering

The `show()` function automatically renders LaTeX when called in a Jupyter cell:

```python
import constkit as ck
import sympy as sp

k = sp.Symbol('k')
F = ck.deformation_gradient('simple_shear', k=k)
E = ck.green_lagrange(F)

ck.show(E, name='E')   # renders: E = [matrix in LaTeX]
```

`Tensor2` objects also render automatically in Jupyter (via `_repr_latex_`):

```python
E  # just put the variable alone in a cell — Jupyter renders LaTeX
```

### Side-by-side comparison

```python
from constkit.display import compare

E = ck.green_lagrange(F)
e = ck.almansi(F)
compare(E, e, names=("E", "e"))
# Shows both tensors and their difference, plus ✓/✗ for equality
```

### Substituting numeric values

```python
# Symbolic result
E = ck.green_lagrange(ck.deformation_gradient('simple_shear', k=sp.Symbol('k')))
ck.show(E, name='E')       # symbolic

# Substitute a specific k
E_num = E.subs(sp.Symbol('k'), sp.Rational(1, 2))
ck.show(E_num, name='E(k=½)')   # exact rational

# Get a NumPy array for plotting (requires numpy)
import numpy as np
E_array = E.evaluate(k=0.5)
print(type(E_array))  # <class 'numpy.ndarray'>
```

---

## HW1 Worked Examples

These examples show exactly how to use constkit to verify each homework problem.

### Problem 1 — Tensor Operations

```python
import constkit as ck
import sympy as sp

u = sp.Matrix([1, 2, 3])
v = sp.Matrix([4, 5, 6])
S = ck.Tensor2([[2, 1, 0], [1, 3, 1], [0, 1, 4]], name='S')
T = ck.Tensor2([[1, 0, 2], [0, 1, 0], [2, 0, 3]], name='T')

# (a) Outer product, single contraction, double contraction
print("u ⊗ v =\n", ck.outer_product(u, v))
print("S·v =", S.dot(v))
print("S:T =", S.double_contract(T))

# (b) Invariants
print("I₁ =", S.I1(), "  I₂ =", S.I2(), "  I₃ =", S.I3())

# (c) Principal values and directions
eig = S.eigendecomposition()
for i, lam in enumerate(eig['eigenvalues']):
    print(f"λ_{i+1} = {sp.simplify(lam)}")

# (d) Rotate by 45° about x₃
Q = ck.rotation_matrix(sp.pi/4, axis=3)
S_rot = S.rotate(Q)
ck.show(S_rot, name="S'")
# Verify invariants preserved:
assert sp.simplify(S_rot.I1() - S.I1()) == 0
assert sp.simplify(S_rot.I3() - S.I3()) == 0
```

### Problem 2 — Tensor Functions

```python
import constkit as ck
import sympy as sp

A = ck.Tensor2([[2, 1, 0], [1, 3, 1], [0, 1, 4]], name='A')

# (a) Characteristic equation
i1, i2, i3 = A.I1(), A.I2(), A.I3()
lam = sp.Symbol('lambda')
char_eq = -lam**3 + i1*lam**2 - i2*lam + i3
print("Characteristic equation:", char_eq, "= 0")

# (b) ln(A) for symmetric A
ln_A = A.log()
# Verify: tr(ln A) = ln(det A)
assert sp.simplify(ln_A.trace() - sp.log(A.det())) == 0

# (c) √A for symmetric A
sqrt_A = A.sqrt()
# Verify: √A · √A = A
reconstructed = sqrt_A @ sqrt_A
assert sp.simplify(reconstructed.matrix - A.matrix) == sp.zeros(3, 3)

# (d) ∂A/∂A for symmetric A
from constkit.calculus import symmetric_identity_4th
I_sym = symmetric_identity_4th()
print("I^s_1212 =", I_sym[0,1,0,1])  # 1/2 — yes, symmetric

# (e) ∂A/∂A for non-symmetric A
from constkit.calculus import identity_4th
I_full = identity_4th()
print("I_1212 =", I_full[0,1,0,1])   # 1 — not the same
```

### Problem 3 — Curvilinear Coordinates

```python
import constkit as ck
import sympy as sp

r, phi = sp.symbols('r phi', positive=True)

# (a) Covariant basis
g_cov = ck.cylindrical_basis(r, phi)
for i, name in enumerate(['g_r', 'g_φ', 'g_z']):
    ck.show(g_cov[i], name=name)

# Metric tensor
g = ck.metric_tensor(g_cov)
ck.show(g, name='g_{ij}')

# (b) Contravariant basis
g_contra = ck.contravariant_basis(g_cov)
for i, name in enumerate(['g^r', 'g^φ', 'g^z']):
    ck.show(g_contra[i], name=name)

# (c) Biorthogonality
ck.verify_biorthogonality(g_cov, g_contra)
print("Biorthogonality verified ✓")

# Evaluate at r=2, φ=π/6 to get concrete numbers
for i in range(3):
    v = g_cov[i].subs([(r, 2), (phi, sp.pi/6)])
    print(f"g_{i+1}(r=2, φ=30°) = {v.T}")
```

### Problem 4 — Kinematics

```python
import constkit as ck
import sympy as sp

k = sp.Symbol('k', positive=True)
F = ck.deformation_gradient('simple_shear', k=k)

# (a) F
ck.show(F, name='F')
print("J =", F.det())  # 1 — isochoric

# (b) C and b
C = ck.right_cauchy_green(F)
b = ck.left_cauchy_green(F)
ck.show(C, name='C')
ck.show(b, name='b')

# (c) Green-Lagrange and Almansi
E = ck.green_lagrange(F)
e = ck.almansi(F)
ck.show(E, name='E')
ck.show(e, name='e')

# (d) Compare at k=0.01 and k=1.0
for k_val in [sp.Rational(1, 100), sp.Integer(1)]:
    E_val = E.subs(k, k_val)
    e_val = e.subs(k, k_val)
    print(f"\nk = {k_val}:")
    print(f"  E_12 = {E_val[0,1]},  e_12 = {e_val[0,1]}")
    print(f"  E_22 = {E_val[1,1]},  e_22 = {e_val[1,1]}")
```

### Problem 5 — Stress Measures

```python
import constkit as ck
import sympy as sp

sigma_0 = sp.Symbol('sigma_0', positive=True)
F = ck.deformation_gradient('simple_shear', k=sp.Integer(1))
sigma = ck.Tensor2(sigma_0 * sp.Matrix([[1,0,0],[0,0,0],[0,0,0]]))

# (a) First Piola-Kirchhoff
P = ck.cauchy_to_pk1(sigma, F)
ck.show(P, name='P')

# (b) Second Piola-Kirchhoff
S = ck.cauchy_to_pk2(sigma, F)
ck.show(S, name='S')

# (c) Stress transformation laws
result = ck.verify_stress_transformations(sigma, F)
print("σ follows QσQᵀ:", result['sigma_matches_rotation'])
print("S invariant:", result['S_invariant'])
print("P two-point:", result['P_two_point'])

# (d) Jaumann rate
sigma_dot = ck.Tensor2(sp.zeros(3, 3))
W = ck.Tensor2(sp.Matrix([[0, sp.Rational(1,2), 0],
                           [-sp.Rational(1,2), 0, 0],
                           [0, 0, 0]]))
sigma_J = ck.jaumann_rate(sigma, sigma_dot, W)
ck.show(sigma_J, name='σ̊_Jaumann')
# σ̊₁₂ = σ₀/2, showing that even with σ̇=0, the Jaumann rate is non-zero
# because the spin W corrects for the material rotation.
```

---

## Troubleshooting

**"SymPy is very slow for my expression"** — Fully symbolic 3×3 eigenvalue problems can produce enormous radical expressions. If `eigendecomposition()`, `sqrt()`, or `log()` is slow, substitute numeric values first with `.subs(k, 1)` and then call the operation on the substituted symmetric tensor.

**"My matrix isn't 3×3"** — constkit works exclusively in 3D. All tensors must be 3×3. For 2D problems, embed them in 3D (set the third row/column to identity-like values).

**"simplify() doesn't show zero"** — SymPy's simplifier is heuristic. If you expect zero, try `sp.trigsimp()` or `sp.simplify(sp.expand(...))`. For numeric verification, substitute concrete values and check.

**"I need Voigt notation"** — Voigt conversions are planned for Phase 2 (numeric evaluation layer). For now, work with full 3×3 tensors.
