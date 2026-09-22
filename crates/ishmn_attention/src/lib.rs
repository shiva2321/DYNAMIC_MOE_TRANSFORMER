use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray4, PyReadonlyArray1, PyReadonlyArray3, PyReadonlyArray4};
use ndarray::Array4;
use rayon::prelude::*;

/// Apply continuous SO(2) rotations along orthogonal 2D planes of S^{D-1}.
/// Preserves ||u|| == 1 strictly without normalization drift.
///
/// Args:
///   u: [B, H, T, D] directional unit vectors
///   freqs: [D / 2] frequencies
///   start_pos: starting temporal index
#[pyfunction]
#[pyo3(signature = (u, freqs, start_pos=0))]
fn riemannian_so2_winding<'py>(
    py: Python<'py>,
    u: PyReadonlyArray4<'py, f32>,
    freqs: PyReadonlyArray1<'py, f32>,
    start_pos: usize,
) -> PyResult<Bound<'py, PyArray4<f32>>> {
    let u_view = u.as_array();
    let shape = u_view.shape();
    let (b, h, t, d) = (shape[0], shape[1], shape[2], shape[3]);

    if d % 2 != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "hyperspace_dim D must be even for SO(2) 2D planes",
        ));
    }
    let num_planes = d / 2;
    let freqs_view = freqs.as_array();
    if freqs_view.len() != num_planes {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "freqs length ({}) must equal D / 2 ({})",
            freqs_view.len(),
            num_planes
        )));
    }

    let freqs_slice: Vec<f32> = freqs_view.iter().copied().collect();
    let mut out = vec![0.0f32; b * h * t * d];
    let u_slice = u_view.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("u must be contiguous in memory")
    })?;

    // Parallelize across (b * h * t) rows
    let row_len = d;
    out.par_chunks_exact_mut(row_len)
        .enumerate()
        .for_each(|(idx, out_row)| {
            let t_idx = (idx % t) + start_pos;
            let in_row = &u_slice[idx * row_len..(idx + 1) * row_len];

            for p in 0..num_planes {
                let theta = (t_idx as f32) * freqs_slice[p];
                let cos_t = theta.cos();
                let sin_t = theta.sin();

                let u0 = in_row[2 * p];
                let u1 = in_row[2 * p + 1];

                out_row[2 * p] = u0 * cos_t - u1 * sin_t;
                out_row[2 * p + 1] = u0 * sin_t + u1 * cos_t;
            }
        });

    let arr = Array4::from_shape_vec((b, h, t, d), out)
        .map_err(|e: ndarray::ShapeError| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

/// Tiled Online Riemannian Softmax Attention on S^{D-1} x R+
///
/// Implements Milakov & Gimelshein / Dao et al. online softmax recurrence
/// generalized to Riemannian manifold score functions:
///   s_{ij} = kappa_eff * <u_i, u_j> - beta * (ln r_i - ln r_j)^2
///
/// Args:
///   u_q: [B, H, T, D] rotated query directions on S^{D-1}
///   u_k: [B, H, S, D] rotated key directions on S^{D-1}
///   ln_r_q: [B, H, T] query log-radius
///   ln_r_k: [B, H, S] key log-radius
///   v: [B, H, S, d_v] value vectors
///   kappa: [H] concentration per head
///   beta: [H] radial precision per head
///   is_causal: boolean causal masking
///   enable_entropy_scaling: whether to scale kappa by sqrt(ln(S + 1))
///   block_size: tile block size along sequence S
#[pyfunction]
#[pyo3(signature = (
    u_q,
    u_k,
    ln_r_q,
    ln_r_k,
    v,
    kappa,
    beta,
    is_causal=false,
    enable_entropy_scaling=true,
    block_size=64
))]
fn tiled_riemannian_attention<'py>(
    py: Python<'py>,
    u_q: PyReadonlyArray4<'py, f32>,
    u_k: PyReadonlyArray4<'py, f32>,
    ln_r_q: PyReadonlyArray3<'py, f32>,
    ln_r_k: PyReadonlyArray3<'py, f32>,
    v: PyReadonlyArray4<'py, f32>,
    kappa: PyReadonlyArray1<'py, f32>,
    beta: PyReadonlyArray1<'py, f32>,
    is_causal: bool,
    enable_entropy_scaling: bool,
    block_size: usize,
) -> PyResult<Bound<'py, PyArray4<f32>>> {
    let q_view = u_q.as_array();
    let k_view = u_k.as_array();
    let v_view = v.as_array();
    let lrq_view = ln_r_q.as_array();
    let lrk_view = ln_r_k.as_array();
    let kappa_view = kappa.as_array();
    let beta_view = beta.as_array();

    let (b, h, t, d) = (q_view.shape()[0], q_view.shape()[1], q_view.shape()[2], q_view.shape()[3]);
    let (b_k, h_k, s, d_k) = (k_view.shape()[0], k_view.shape()[1], k_view.shape()[2], k_view.shape()[3]);
    let d_v = v_view.shape()[3];

    if b != b_k || h != h_k || d != d_k {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Dimension mismatch between Query and Key",
        ));
    }

    let entropy_factor = if enable_entropy_scaling && s > 1 {
        ((s as f32) + 1.0).ln().sqrt()
    } else {
        1.0
    };

    let kappa_vec: Vec<f32> = kappa_view
        .iter()
        .map(|&k| (k.clamp(0.1, 30.0)) * entropy_factor)
        .collect();
    let beta_vec: Vec<f32> = beta_view
        .iter()
        .map(|&b_val| b_val.clamp(0.01, 10.0))
        .collect();

    let u_q_slice = q_view.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("u_q must be contiguous")
    })?;
    let u_k_slice = k_view.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("u_k must be contiguous")
    })?;
    let lrq_slice = lrq_view.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("ln_r_q must be contiguous")
    })?;
    let lrk_slice = lrk_view.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("ln_r_k must be contiguous")
    })?;
    let v_slice = v_view.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("v must be contiguous")
    })?;

    let mut out = vec![0.0f32; b * h * t * d_v];
    let bc = if block_size > 0 { block_size } else { 64 };

    // Parallelize across (b, h, t) queries using Rayon
    out.par_chunks_exact_mut(d_v)
        .enumerate()
        .for_each(|(query_flat_idx, out_vec)| {
            let t_idx = query_flat_idx % t;
            let bh_idx = query_flat_idx / t;
            let h_idx = bh_idx % h;
            let b_idx = bh_idx / h;

            let kappa_eff = kappa_vec[h_idx];
            let beta_eff = beta_vec[h_idx];

            let q_offset = (b_idx * (h * t * d)) + (h_idx * (t * d)) + (t_idx * d);
            let q_vec = &u_q_slice[q_offset..q_offset + d];

            let lrq_offset = (b_idx * (h * t)) + (h_idx * t) + t_idx;
            let lrq_val = lrq_slice[lrq_offset];

            let k_bh_offset = (b_idx * (h * s * d)) + (h_idx * (s * d));
            let lrk_bh_offset = (b_idx * (h * s)) + (h_idx * s);
            let v_bh_offset = (b_idx * (h * s * d_v)) + (h_idx * (s * d_v));

            let max_k = if is_causal {
                (t_idx + 1).min(s)
            } else {
                s
            };

            if max_k == 0 {
                return;
            }

            let mut running_max = -f32::INFINITY;
            let mut running_sum = 0.0f32;
            let mut o_acc = vec![0.0f32; d_v];
            let mut tile_scores = vec![0.0f32; bc];

            let mut k_start = 0;
            while k_start < max_k {
                let k_end = (k_start + bc).min(max_k);
                let tile_len = k_end - k_start;
                let mut tile_max = -f32::INFINITY;

                for i in 0..tile_len {
                    let k_idx = k_start + i;
                    let k_offset = k_bh_offset + (k_idx * d);
                    let k_vec = &u_k_slice[k_offset..k_offset + d];

                    // Geodesic angular similarity on S^{D-1}
                    let mut dot = 0.0f32;
                    for dim in 0..d {
                        dot += q_vec[dim] * k_vec[dim];
                    }

                    // Scale-invariant radial depth error
                    let lrk_val = lrk_slice[lrk_bh_offset + k_idx];
                    let diff_ln_r = lrq_val - lrk_val;
                    let s_radial = -(diff_ln_r * diff_ln_r);

                    let score = kappa_eff * dot + beta_eff * s_radial;
                    tile_scores[i] = score;
                    if score > tile_max {
                        tile_max = score;
                    }
                }

                // Online Softmax Recurrence
                let new_max = running_max.max(tile_max);
                let alpha = (running_max - new_max).exp();

                running_sum *= alpha;
                for d_idx in 0..d_v {
                    o_acc[d_idx] *= alpha;
                }

                for i in 0..tile_len {
                    let k_idx = k_start + i;
                    let p = (tile_scores[i] - new_max).exp();
                    running_sum += p;

                    let v_offset = v_bh_offset + (k_idx * d_v);
                    let v_vec = &v_slice[v_offset..v_offset + d_v];
                    for d_idx in 0..d_v {
                        o_acc[d_idx] += p * v_vec[d_idx];
                    }
                }

                running_max = new_max;
                k_start += bc;
            }

            // Final normalization
            if running_sum > 0.0 {
                let inv_sum = 1.0 / running_sum;
                for d_idx in 0..d_v {
                    out_vec[d_idx] = o_acc[d_idx] * inv_sum;
                }
            }
        });

    let arr = Array4::from_shape_vec((b, h, t, d_v), out)
        .map_err(|e: ndarray::ShapeError| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

#[pymodule]
fn ishmn_attention_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(riemannian_so2_winding, m)?)?;
    m.add_function(wrap_pyfunction!(tiled_riemannian_attention, m)?)?;
    Ok(())
}
