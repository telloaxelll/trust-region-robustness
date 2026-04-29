# Comparative Robustness of Second-Order Optimization Methods Under Non-Convexity & Noise

## Overview

This project investigates trust region optimization methods and compares their robustness and convergence behavior against classical Newton's Method. The central question we seek to explore is: when the quadratic model of a function is only reliable within a bounded neighborhood, how does constraining the step size affect convergence, stability, and practical performance?

---

## Background

### Newton's Method

Newton's Method is a second-order optimization algorithm that computes a step by solving the Newton system:

$$p_k = -H_k^{-1} \nabla f(x_k)$$

where $H_k = \nabla^2 f(x_k)$ is the Hessian at the current iterate $x_k$. The method builds a local quadratic model:

$$m_k(p) = f(x_k) + \nabla f(x_k)^\top p + \frac{1}{2} p^\top H_k \, p$$

and takes the step that minimizes this model exactly, without restriction.

**Strengths:**
- Quadratic convergence near a solution when $H_k$ is positive definite and Lipschitz continuous
- Exact second-order information makes each step highly informed

**Weaknesses:**
- Requires the Hessian to be positive definite; indefinite or singular Hessians cause the method to fail or diverge
- The full Newton step may overshoot when far from the solution, where the quadratic model is inaccurate
- Computing and factoring the Hessian is $O(n^3)$ per iteration, expensive for large-scale problems
- No built-in globalization: the method can diverge without a line search or other safeguard

---

### Trust Region Methods

Trust region methods address Newton's weaknesses by restricting each step to a region where the quadratic model is trusted to be a good approximation of $f$. The subproblem solved at each iteration is:

$$\min_{p \in \mathbb{R}^n} \quad m_k(p) = f(x_k) + g_k^\top p + \frac{1}{2} p^\top B_k \, p$$
$$\text{subject to} \quad \|p\| \leq \Delta_k$$

where $g_k = \nabla f(x_k)$, $B_k$ is either the true Hessian or an approximation, and $\Delta_k > 0$ is the trust region radius.

The radius is updated based on the **reduction ratio**:

$$\rho_k = \frac{f(x_k) - f(x_k + p_k)}{m_k(0) - m_k(p_k)}$$

which measures how well the model predicted the actual decrease. If $\rho_k$ is close to 1, the model is accurate and $\Delta_k$ is expanded; if $\rho_k$ is small or negative, the step is rejected and $\Delta_k$ is contracted.

**Strengths:**
- Globally convergent under mild assumptions, even from poor starting points
- Handles indefinite Hessians naturally — the constraint regularizes the subproblem
- Radius adaptation provides automatic step length control without a separate line search
- Can recover from regions where the quadratic model is poor

**Weaknesses:**
- The trust region subproblem itself requires an inner solver, adding complexity
- Overhead per iteration is higher than a plain Newton step when the subproblem is solved exactly
- Convergence rate near a solution matches Newton's (superlinear/quadratic), but the constant may be worse

---

## Key Algorithmic Variants

### Exact Trust Region Subproblem (Moré–Sorensen)

The exact solution to the trust region subproblem satisfies:

$$(B_k + \lambda I) p^* = -g_k, \quad \lambda \geq 0, \quad \lambda(\Delta_k - \|p^*\|) = 0$$

This is the **secular equation** approach. The scalar $\lambda$ acts as a Lagrange multiplier enforcing the radius constraint. When $B_k$ is positive definite and the unconstrained Newton step lies inside $\Delta_k$, then $\lambda = 0$ and the trust region step coincides with the Newton step.

### Cauchy Point and Dogleg Methods

For efficiency, the subproblem need not be solved exactly. Two common approximate solvers:

- **Cauchy Point:** Minimize the model along the steepest descent direction, then clip to the trust region boundary. Cheap, but ignores curvature.
- **Dogleg Method:** Interpolate between the Cauchy point and the full Newton step. When the Newton step is inside $\Delta_k$, take it. When it is outside, follow the dogleg path from the Cauchy point toward the Newton point, stopping at the boundary. Requires $B_k$ positive definite.

### Steihaug–Toint Conjugate Gradient (CG) Method

For large-scale problems, the subproblem is solved approximately using truncated CG:

- Run CG on $B_k p = -g_k$ until either convergence, the iterate exits the trust region, or a direction of negative curvature is encountered
- If negative curvature is detected, move to the trust region boundary along that direction

This avoids forming or factoring $B_k$ explicitly, requiring only matrix-vector products with $B_k$.

---

## Convergence Comparison

| Property | Newton's Method | Trust Region |
|---|---|---|
| Local convergence rate | Quadratic | Quadratic (exact subproblem) |
| Global convergence | Not guaranteed without line search | Guaranteed under mild conditions |
| Indefinite Hessian handling | Fails or requires modification | Handled naturally via $\lambda$ |
| Step acceptance | Always accepts Newton step | Rejects steps with $\rho_k < \eta$ |
| Cost per iteration | $O(n^3)$ Hessian factor | $O(n^3)$ (exact) or $O(n \cdot \text{cg\_iters})$ (inexact) |
| Robustness far from solution | Low | High |

Near a local minimizer $x^*$ where $\nabla^2 f(x^*)$ is positive definite, both methods are equivalent: the trust region radius grows unbounded, $\lambda \to 0$, and the trust region step converges to the Newton step. The trust region framework thus generalizes Newton's Method, recovering it as a special case.

---

## The Role of the Hessian Approximation

When the exact Hessian is unavailable or too expensive, $B_k$ can be a quasi-Newton approximation (e.g., SR1, BFGS). The SR1 update is particularly natural for trust region methods because it can produce indefinite approximations while still preserving useful curvature information, something BFGS avoids by construction.

The trust region framework tolerates an indefinite $B_k$ by adding a sufficiently large $\lambda I$ shift, ensuring $(B_k + \lambda I)$ is positive definite when solving the subproblem.

---

## Robustness Analysis

The robustness of trust region methods stems from three mechanisms:

1. **Guaranteed descent:** The acceptance criterion $\rho_k \geq \eta > 0$ ensures the objective decreases on every accepted step.
2. **Adaptive scaling:** $\Delta_k$ shrinks automatically when the model is inaccurate, preventing large errant steps.
3. **Regularization of curvature:** Even when $B_k$ is indefinite, the constrained subproblem has a well-defined solution.

Newton's Method lacks all three of these properties without explicit augmentation (e.g., modified Cholesky, line search, Levenberg–Marquardt regularization), each of which partially approximates what trust region methods provide by design.

---

## References

- Nocedal, J. & Wright, S. J. (2006). *Numerical Optimization* (2nd ed.). Springer. Chapters 4–7.
- Conn, A. R., Gould, N. I. M., & Toint, Ph. L. (2000). *Trust Region Methods*. SIAM.
- Moré, J. J. & Sorensen, D. C. (1983). Computing a trust region step. *SIAM Journal on Scientific and Statistical Computing*, 4(3), 553–572.
- Steihaug, T. (1983). The conjugate gradient method and trust regions in large scale optimization. *SIAM Journal on Numerical Analysis*, 20(3), 626–637.
