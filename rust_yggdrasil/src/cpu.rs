use crate::manifold::*;
use rayon::prelude::*;

/// Multi-threaded batch pairwise distance matrix computation on CPU using Rayon.
/// Matrix A has shape (N, D), Matrix B has shape (M, D).
/// Output Matrix has shape (N, M) in row-major layout.
pub fn batch_pairwise_distance_cpu(
    a: &[f32],
    b: &[f32],
    n: usize,
    m: usize,
    config: &YggdrasilConfig,
    out: &mut [f32],
) {
    let d = config.dim_total();
    assert_eq!(a.len(), n * d, "Matrix A size mismatch");
    assert_eq!(b.len(), m * d, "Matrix B size mismatch");
    assert_eq!(out.len(), n * m, "Output matrix size mismatch");

    out.par_chunks_mut(m)
        .enumerate()
        .for_each(|(i, row)| {
            let a_point = &a[i * d..(i + 1) * d];
            for j in 0..m {
                let b_point = &b[j * d..(j + 1) * d];
                row[j] = composite_distance(a_point, b_point, config);
            }
        });
}

/// Multi-threaded batch Mobius addition: out[i] = u[i] (plus_c) v[i]
pub fn batch_mobius_add_cpu(
    u: &[f32],
    v: &[f32],
    n: usize,
    d: usize,
    c: f32,
    out: &mut [f32],
) {
    assert_eq!(u.len(), n * d);
    assert_eq!(v.len(), n * d);
    assert_eq!(out.len(), n * d);

    out.par_chunks_mut(d)
        .enumerate()
        .for_each(|(i, out_point)| {
            let u_point = &u[i * d..(i + 1) * d];
            let v_point = &v[i * d..(i + 1) * d];
            mobius_add(u_point, v_point, out_point, c);
        });
}

/// Multi-threaded batch projection of state vectors into the valid manifold domain
pub fn batch_project_cpu(
    points: &mut [f32],
    n: usize,
    config: &YggdrasilConfig,
) {
    let d = config.dim_total();
    assert_eq!(points.len(), n * d);

    points.par_chunks_mut(d).for_each(|point| {
        project_state(point, config);
    });
}

/// Multi-threaded batch Riemannian exponential map
pub fn batch_exp_map_cpu(
    u: &[f32],
    v: &[f32],
    n: usize,
    d: usize,
    c: f32,
    out: &mut [f32],
) {
    assert_eq!(u.len(), n * d);
    assert_eq!(v.len(), n * d);
    assert_eq!(out.len(), n * d);

    out.par_chunks_mut(d)
        .enumerate()
        .for_each(|(i, out_point)| {
            let u_point = &u[i * d..(i + 1) * d];
            let v_point = &v[i * d..(i + 1) * d];
            exp_map(u_point, v_point, out_point, c);
        });
}

/// Multi-threaded batch Riemannian logarithmic map
pub fn batch_log_map_cpu(
    u: &[f32],
    v: &[f32],
    n: usize,
    d: usize,
    c: f32,
    out: &mut [f32],
) {
    assert_eq!(u.len(), n * d);
    assert_eq!(v.len(), n * d);
    assert_eq!(out.len(), n * d);

    out.par_chunks_mut(d)
        .enumerate()
        .for_each(|(i, out_point)| {
            let u_point = &u[i * d..(i + 1) * d];
            let v_point = &v[i * d..(i + 1) * d];
            log_map(u_point, v_point, out_point, c);
        });
}

/// Multi-threaded batch Torus binding
pub fn batch_torus_bind_cpu(
    u: &[f32],
    v: &[f32],
    n: usize,
    d: usize,
    out: &mut [f32],
) {
    assert_eq!(u.len(), n * d);
    assert_eq!(v.len(), n * d);
    assert_eq!(out.len(), n * d);

    out.par_chunks_mut(d)
        .enumerate()
        .for_each(|(i, out_point)| {
            let u_point = &u[i * d..(i + 1) * d];
            let v_point = &v[i * d..(i + 1) * d];
            torus_bind(u_point, v_point, out_point);
        });
}

/// Multi-threaded batch Riemannian gradient retraction
pub fn batch_riemannian_retract_cpu(
    points: &[f32],
    grads: &[f32],
    n: usize,
    d: usize,
    lr: f32,
    c: f32,
    eps: f32,
    out: &mut [f32],
) {
    assert_eq!(points.len(), n * d);
    assert_eq!(grads.len(), n * d);
    assert_eq!(out.len(), n * d);

    out.par_chunks_mut(d)
        .enumerate()
        .for_each(|(i, out_point)| {
            let p_point = &points[i * d..(i + 1) * d];
            let g_point = &grads[i * d..(i + 1) * d];
            riemannian_retraction(p_point, g_point, lr, out_point, c, eps);
        });
}
