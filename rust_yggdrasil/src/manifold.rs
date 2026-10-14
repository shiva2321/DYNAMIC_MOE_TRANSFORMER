use std::f32::consts::PI;

/// Configuration defining the dimensions and Riemannian metric parameters of the Yggdrasil manifold.
/// Manifold M = H^(d_H) x T^(d_T) x R^+
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct YggdrasilConfig {
    pub dim_h: u32,
    pub dim_t: u32,
    pub c: f32,
    pub eps: f32,
    pub weight_h: f32,
    pub weight_t: f32,
    pub weight_r: f32,
}

impl Default for YggdrasilConfig {
    fn default() -> Self {
        Self {
            dim_h: 16,
            dim_t: 16,
            c: 1.0,
            eps: 1e-5,
            weight_h: 1.0,
            weight_t: 1.0,
            weight_r: 1.0,
        }
    }
}

impl YggdrasilConfig {
    #[inline(always)]
    pub fn dim_total(&self) -> usize {
        (self.dim_h + self.dim_t + 1) as usize
    }

    #[inline(always)]
    pub fn max_hyperbolic_norm(&self) -> f32 {
        (1.0 - self.eps) / self.c.sqrt()
    }
}

// ---------------------------------------------------------------------------
// Numerically robust elementary mathematical functions
// ---------------------------------------------------------------------------

/// Numerically stable arsinh(x) = ln(x + sqrt(x^2 + 1))
#[inline(always)]
pub fn stable_arsinh(x: f32) -> f32 {
    (x + (x * x + 1.0).sqrt()).ln()
}

/// Numerically stable artanh(x) = 0.5 * ln((1 + x) / (1 - x))
/// Clamps input into [-1.0 + eps, 1.0 - eps] to prevent infinite logarithms.
#[inline(always)]
pub fn stable_artanh(x: f32) -> f32 {
    let clamped = x.clamp(-0.9999999, 0.9999999);
    0.5 * ((1.0 + clamped) / (1.0 - clamped)).ln()
}

/// Branchless modular wrapping into [-PI, PI)
#[inline(always)]
pub fn wrap_angle(theta: f32) -> f32 {
    let two_pi = 2.0 * PI;
    let mut val = theta % two_pi;
    if val >= PI {
        val -= two_pi;
    } else if val < -PI {
        val += two_pi;
    }
    val
}

/// Shortest path angular difference on the circle S^1
#[inline(always)]
pub fn shortest_angular_diff(t1: f32, t2: f32) -> f32 {
    wrap_angle(t1 - t2)
}

// ---------------------------------------------------------------------------
// Core Manifold Geometry Primitives
// ---------------------------------------------------------------------------

/// Projects arbitrary vector into the open Poincare ball B_c^(d_H)
#[inline]
pub fn project_hyperbolic(x: &mut [f32], c: f32, eps: f32) {
    let max_r = (1.0 - eps) / c.sqrt();
    let mut sq_norm = 0.0f32;
    for &v in x.iter() {
        sq_norm += v * v;
    }
    let norm = (sq_norm + 1e-18).sqrt();
    if norm > max_r {
        let scale = max_r / norm;
        for v in x.iter_mut() {
            *v *= scale;
        }
    }
}

/// Normalizes toroidal coordinates into canonical domain [-PI, PI)
#[inline]
pub fn project_torus(theta: &mut [f32]) {
    for t in theta.iter_mut() {
        *t = wrap_angle(*t);
    }
}

/// Projects a composite state vector [H; T; R]
pub fn project_state(point: &mut [f32], config: &YggdrasilConfig) {
    let dh = config.dim_h as usize;
    let dt = config.dim_t as usize;
    assert!(point.len() >= dh + dt + 1, "State vector buffer too small");

    project_hyperbolic(&mut point[0..dh], config.c, config.eps);
    project_torus(&mut point[dh..dh + dt]);
    
    // Radial coordinate must be strictly positive (scale)
    let r_idx = dh + dt;
    if point[r_idx] < config.eps {
        point[r_idx] = config.eps;
    }
}

/// Geodesic distance on the Poincare Ball with curvature c
/// Uses the exact, singularity-free identity: d = (2 / sqrt(c)) * arsinh(sqrt(xi))
/// where xi = (c * ||u - v||^2) / ((1 - c||u||^2) * (1 - c||v||^2))
#[inline]
pub fn poincare_distance(u: &[f32], v: &[f32], c: f32) -> f32 {
    debug_assert_eq!(u.len(), v.len());
    let mut diff_sq = 0.0f32;
    let mut u_sq = 0.0f32;
    let mut v_sq = 0.0f32;

    for i in 0..u.len() {
        let d = u[i] - v[i];
        diff_sq += d * d;
        u_sq += u[i] * u[i];
        v_sq += v[i] * v[i];
    }

    if diff_sq < 1e-14 {
        return 0.0;
    }

    let denom_u = (1.0 - c * u_sq).max(1e-7);
    let denom_v = (1.0 - c * v_sq).max(1e-7);
    let xi = (c * diff_sq) / (denom_u * denom_v);
    
    (2.0 / c.sqrt()) * stable_arsinh(xi.max(0.0).sqrt())
}

/// Geodesic distance on the Clifford Flat Torus T^(d_T)
#[inline]
pub fn torus_distance(u: &[f32], v: &[f32]) -> f32 {
    debug_assert_eq!(u.len(), v.len());
    let mut sum_sq = 0.0f32;
    for i in 0..u.len() {
        let diff = shortest_angular_diff(u[i], v[i]);
        sum_sq += diff * diff;
    }
    sum_sq.sqrt()
}

/// Composite Riemannian distance on the Yggdrasil Manifold
#[inline]
pub fn composite_distance(u: &[f32], v: &[f32], config: &YggdrasilConfig) -> f32 {
    let dh = config.dim_h as usize;
    let dt = config.dim_t as usize;
    let r_idx = dh + dt;

    let d_h = poincare_distance(&u[0..dh], &v[0..dh], config.c);
    let d_t = torus_distance(&u[dh..r_idx], &v[dh..r_idx]);
    // Intensity is multiplicative (r -> r * s): the invariant metric is |ln r_u - ln r_v|, floored at eps
    // (same definition as geometry/hybrid.py and the CUDA kernel).
    let d_r = (u[r_idx].max(config.eps).ln() - v[r_idx].max(config.eps).ln()).abs();

    let d_sq = config.weight_h * d_h * d_h 
             + config.weight_t * d_t * d_t 
             + config.weight_r * d_r * d_r;
    d_sq.sqrt()
}

/// Mobius addition in the Poincare ball: u (plus_c) v
/// Invariant: (-v) (plus_c) (v (plus_c) u) = u (Left-gyrogroup identity)
#[inline]
pub fn mobius_add(u: &[f32], v: &[f32], out: &mut [f32], c: f32) {
    debug_assert_eq!(u.len(), v.len());
    debug_assert_eq!(u.len(), out.len());

    let mut dot = 0.0f32;
    let mut u_sq = 0.0f32;
    let mut v_sq = 0.0f32;

    for i in 0..u.len() {
        dot += u[i] * v[i];
        u_sq += u[i] * u[i];
        v_sq += v[i] * v[i];
    }

    let alpha = 1.0 + 2.0 * c * dot + c * v_sq;
    let beta = 1.0 - c * u_sq;
    let denom = (1.0 + 2.0 * c * dot + c * c * u_sq * v_sq).max(1e-10);

    for i in 0..u.len() {
        out[i] = (alpha * u[i] + beta * v[i]) / denom;
    }
}

/// Riemannian Exponential map on the Poincare Ball: exp_u^c(v)
/// Maps a tangent vector v in T_u B to a point on the manifold
#[inline]
pub fn exp_map(u: &[f32], v: &[f32], out: &mut [f32], c: f32) {
    debug_assert_eq!(u.len(), v.len());
    debug_assert_eq!(u.len(), out.len());

    let mut u_sq = 0.0f32;
    let mut v_sq = 0.0f32;
    for i in 0..u.len() {
        u_sq += u[i] * u[i];
        v_sq += v[i] * v[i];
    }

    let v_norm = v_sq.sqrt();
    if v_norm < 1e-10 {
        out.copy_from_slice(u);
        return;
    }

    let lambda_u = 2.0 / (1.0 - c * u_sq).max(1e-7);
    let sqrt_c = c.sqrt();
    let arg = 0.5 * sqrt_c * lambda_u * v_norm;
    let factor = arg.tanh() / (sqrt_c * v_norm);

    // Temp scaled vector w = factor * v
    let mut w = vec![0.0f32; u.len()];
    for i in 0..u.len() {
        w[i] = factor * v[i];
    }

    mobius_add(u, &w, out, c);
}

/// Riemannian Logarithmic map on the Poincare Ball: log_u^c(v)
/// Maps a point v on the manifold to a tangent vector in T_u B
#[inline]
pub fn log_map(u: &[f32], v: &[f32], out: &mut [f32], c: f32) {
    debug_assert_eq!(u.len(), v.len());
    debug_assert_eq!(u.len(), out.len());

    let mut neg_u = vec![0.0f32; u.len()];
    for i in 0..u.len() {
        neg_u[i] = -u[i];
    }

    let mut diff = vec![0.0f32; u.len()];
    mobius_add(&neg_u, v, &mut diff, c);

    let mut diff_sq = 0.0f32;
    let mut u_sq = 0.0f32;
    for i in 0..u.len() {
        diff_sq += diff[i] * diff[i];
        u_sq += u[i] * u[i];
    }

    let diff_norm = diff_sq.sqrt();
    if diff_norm < 1e-10 {
        out.fill(0.0);
        return;
    }

    let lambda_u = 2.0 / (1.0 - c * u_sq).max(1e-7);
    let sqrt_c = c.sqrt();
    let scaled_norm = (sqrt_c * diff_norm).min(0.999999);
    let artanh_val = stable_artanh(scaled_norm);
    let factor = (2.0 / (sqrt_c * lambda_u)) * (artanh_val / diff_norm);

    for i in 0..u.len() {
        out[i] = factor * diff[i];
    }
}

/// Toroidal binding (Lie group addition modulo 2PI)
#[inline]
pub fn torus_bind(u: &[f32], v: &[f32], out: &mut [f32]) {
    debug_assert_eq!(u.len(), v.len());
    debug_assert_eq!(u.len(), out.len());
    for i in 0..u.len() {
        out[i] = wrap_angle(u[i] + v[i]);
    }
}

/// Toroidal unbinding / inverse translation: (u - v) mod 2PI
#[inline]
pub fn torus_unbind(u: &[f32], v: &[f32], out: &mut [f32]) {
    debug_assert_eq!(u.len(), v.len());
    debug_assert_eq!(u.len(), out.len());
    for i in 0..u.len() {
        out[i] = wrap_angle(u[i] - v[i]);
    }
}

/// Riemannian Gradient Retraction step:
/// Translates Euclidean gradient grad_euc to Riemannian gradient by metric inverse,
/// then takes a geodesic step using exp_map.
#[inline]
pub fn riemannian_retraction(
    point: &[f32],
    grad_euc: &[f32],
    lr: f32,
    out: &mut [f32],
    c: f32,
    eps: f32,
) {
    debug_assert_eq!(point.len(), grad_euc.len());
    debug_assert_eq!(point.len(), out.len());

    let mut p_sq = 0.0f32;
    for &val in point.iter() {
        p_sq += val * val;
    }

    // Inverse metric tensor scale for Poincare ball: ((1 - c||x||^2)^2) / 4
    let factor = (1.0 - c * p_sq).max(1e-7);
    let inv_metric = (factor * factor) / 4.0;

    let mut step = vec![0.0f32; point.len()];
    for i in 0..point.len() {
        step[i] = -lr * inv_metric * grad_euc[i];
    }

    exp_map(point, &step, out, c);
    project_hyperbolic(out, c, eps);
}
