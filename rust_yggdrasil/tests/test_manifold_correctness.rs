use yggdrasil_engine::manifold::*;
use yggdrasil_engine::cpu::*;
use yggdrasil_engine::gpu::YggdrasilGpuEngine;

#[test]
fn test_poincare_metric_axioms() {
    let c = 1.0f32;
    let u = vec![0.1f32, -0.2, 0.3];
    let v = vec![0.4f32, 0.1, -0.1];

    // Identity
    let d_uu = poincare_distance(&u, &u, c);
    assert_eq!(d_uu, 0.0f32, "Self distance must be 0");

    // Symmetry
    let d_uv = poincare_distance(&u, &v, c);
    let d_vu = poincare_distance(&v, &u, c);
    assert!((d_uv - d_vu).abs() < 1e-6, "Poincare distance must be symmetric");
    assert!(d_uv > 0.0, "Distance must be strictly positive");

    // Extreme boundary stability test (near norm = 1.0)
    let near_boundary_1 = vec![0.9999f32, 0.0, 0.0];
    let near_boundary_2 = vec![0.0f32, 0.9999, 0.0];
    let d_bound = poincare_distance(&near_boundary_1, &near_boundary_2, c);
    assert!(!d_bound.is_nan() && !d_bound.is_infinite());
    assert!(d_bound > 10.0, "Boundary points must have large hyperbolic distance");
}

#[test]
fn test_poincare_left_cancellation_identity() {
    let c = 1.0f32;
    let u = vec![0.2f32, -0.1, 0.3];
    let v = vec![-0.15f32, 0.25, -0.05];

    // v_plus_u = v (+) u
    let mut v_plus_u = vec![0.0f32; 3];
    mobius_add(&v, &u, &mut v_plus_u, c);

    // neg_v = -v
    let neg_v = vec![-v[0], -v[1], -v[2]];

    // recovered = (-v) (+) (v (+) u)
    let mut recovered = vec![0.0f32; 3];
    mobius_add(&neg_v, &v_plus_u, &mut recovered, c);

    for i in 0..3 {
        assert!(
            (recovered[i] - u[i]).abs() < 1e-6,
            "Left-cancellation failed at dim {}: got {}, expected {}",
            i, recovered[i], u[i]
        );
    }
}

#[test]
fn test_torus_zero_drift_1000_hops() {
    let dim = 8;
    let original = vec![1.2f32, -2.1, 0.5, 3.0, -0.1, 0.8, -1.5, 2.7];
    let mut state = original.clone();

    // Perform 1000 random shifts and inverse shifts
    for step in 0..1000 {
        let shift: Vec<f32> = (0..dim).map(|i| ((step * 7 + i * 13) as f32).sin() * 2.0).collect();
        let mut shifted = vec![0.0f32; dim];
        torus_bind(&state, &shift, &mut shifted);

        let mut unshifted = vec![0.0f32; dim];
        torus_unbind(&shifted, &shift, &mut unshifted);
        state = unshifted;
    }

    let drift = torus_distance(&original, &state);
    assert!(
        drift < 1e-5,
        "Torus suffered accumulated drift after 1000 hops: {}",
        drift
    );
}

#[test]
fn test_safe_boundary_retraction() {
    let config = YggdrasilConfig {
        dim_h: 4,
        dim_t: 4,
        c: 1.0,
        eps: 1e-5,
        ..Default::default()
    };

    // Vector with huge hyperbolic magnitude (norm = 10.0)
    let mut state = vec![
        5.0f32, -5.0, 5.0, -5.0, // Hyperbolic (violates boundary)
        10.0, -15.0, 20.0, -25.0, // Torus (violates [-PI, PI))
        -5.0, // Radial (violates positive scale)
    ];

    project_state(&mut state, &config);

    // Verify Hyperbolic norm
    let mut sq_norm = 0.0f32;
    for &x in &state[0..4] {
        sq_norm += x * x;
    }
    let norm = sq_norm.sqrt();
    let max_r = config.max_hyperbolic_norm();
    assert!(norm <= max_r + 1e-6, "Hyperbolic norm exceeded boundary: {}", norm);

    // Verify Torus bounds
    for (i, &theta) in state[4..8].iter().enumerate() {
        assert!(
            theta >= -std::f32::consts::PI && theta < std::f32::consts::PI,
            "Torus dim {} not wrapped: {}", i, theta
        );
    }

    // Verify Radial positivity
    assert!(state[8] >= config.eps, "Radial coordinate not positive: {}", state[8]);
}

#[test]
fn test_exp_log_map_inversion() {
    let c = 1.0f32;
    let base = vec![0.1f32, -0.15, 0.2];
    let tangent = vec![0.05f32, 0.1, -0.08];

    let mut point = vec![0.0f32; 3];
    exp_map(&base, &tangent, &mut point, c);

    let mut recovered_tangent = vec![0.0f32; 3];
    log_map(&base, &point, &mut recovered_tangent, c);

    for i in 0..3 {
        assert!(
            (recovered_tangent[i] - tangent[i]).abs() < 1e-5,
            "Exp/Log inverse mismatch at dim {}: got {}, expected {}",
            i, recovered_tangent[i], tangent[i]
        );
    }
}

#[test]
fn test_cuda_rtx3060_execution_and_accuracy() {
    let engine = match YggdrasilGpuEngine::new(0) {
        Ok(e) => e,
        Err(err) => {
            eprintln!("Skipping GPU test: {}", err);
            return;
        }
    };

    println!("Testing on hardware: {}", engine.ctx.device_name);

    let config = YggdrasilConfig {
        dim_h: 8,
        dim_t: 8,
        c: 1.0,
        eps: 1e-5,
        weight_h: 1.0,
        weight_t: 1.0,
        weight_r: 1.0,
    };
    let d = config.dim_total(); // 17

    let n = 128;
    let m = 64;

    let mut a_points = vec![0.0f32; n * d];
    let mut b_points = vec![0.0f32; m * d];

    for i in 0..n * d {
        a_points[i] = ((i as f32) * 0.1).sin() * 0.2;
    }
    for j in 0..m * d {
        b_points[j] = ((j as f32) * 0.15).cos() * 0.2;
    }

    // Run CPU reference
    let mut cpu_out = vec![0.0f32; n * m];
    batch_pairwise_distance_cpu(&a_points, &b_points, n, m, &config, &mut cpu_out);

    // Run GPU execution
    let mut gpu_out = vec![0.0f32; n * m];
    engine.pairwise_distance_host(&a_points, &b_points, n, m, &config, &mut gpu_out)
        .expect("GPU pairwise distance failed");

    // Cross-verify CPU vs GPU results
    let mut max_diff = 0.0f32;
    for i in 0..n * m {
        let diff = (cpu_out[i] - gpu_out[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    println!("Max CPU vs GPU pairwise distance difference: {:.2e}", max_diff);
    assert!(
        max_diff < 1e-5,
        "GPU pairwise distance deviated from CPU reference by {}",
        max_diff
    );
}
