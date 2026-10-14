#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// ---------------------------------------------------------------------------
// Inline GPU mathematical helpers with strict boundary guards
// ---------------------------------------------------------------------------

__device__ __forceinline__ float gpu_stable_arsinh(float x) {
    return logf(x + sqrtf(x * x + 1.0f));
}

__device__ __forceinline__ float gpu_wrap_angle(float theta) {
    float two_pi = 2.0f * (float)M_PI;
    float val = fmodf(theta, two_pi);
    if (val >= (float)M_PI) {
        val -= two_pi;
    } else if (val < -(float)M_PI) {
        val += two_pi;
    }
    return val;
}

// Shortest signed arc between two angles (difference d of two angles -> [-pi, pi]).
// fmodf is a slow software loop on GPUs; for |d| < 8*pi a fused round-and-subtract is exact enough
// (error <= 1 ulp of 8*pi), larger inputs fall back to fmodf. Used by the distance kernel only: the canonical
// [-pi, pi) wrap above stays in project/bind/unbind.
__device__ __forceinline__ float gpu_wrap_diff(float d) {
    const float two_pi = 6.28318530717958647692f;
    if (fabsf(d) < 25.0f) {
        return d - two_pi * rintf(d * 0.15915494309189533577f);
    }
    return gpu_wrap_angle(d);
}

// ---------------------------------------------------------------------------
// Kernel 1: High-Throughput Pairwise Distance Matrix on Yggdrasil Manifold
// Grid: ( (M + 15)/16, (N + 15)/16 ), Block: (16, 16)
// ---------------------------------------------------------------------------
extern "C" __global__ void k_yggdrasil_pairwise_distance(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ out_dist,
    int N,
    int M,
    int dim_h,
    int dim_t,
    float c,
    float eps,
    float weight_h,
    float weight_t,
    float weight_r
) {
    int tx = threadIdx.x; // 0..15 (col within tile)
    int ty = threadIdx.y; // 0..15 (row within tile)
    int col = blockIdx.x * 16 + tx;
    int row = blockIdx.y * 16 + ty;
    int total_dim = dim_h + dim_t + 1;
    int tid = ty * 16 + tx; // 0..255

    // Shared memory for tile of A (16 rows) and B (16 cols)
    __shared__ float s_A[16][36];
    __shared__ float s_B[16][36];
    __shared__ float s_u_sq[16];
    __shared__ float s_v_sq[16];
    __shared__ float s_log_ra[16];
    __shared__ float s_log_rb[16];

    // Coalesced loading into shared memory
    int total_elements = 16 * total_dim;
    for (int idx = tid; idx < total_elements; idx += 256) {
        int r = idx / total_dim;
        int d = idx % total_dim;
        int global_r = blockIdx.y * 16 + r;
        s_A[r][d] = (global_r < N) ? A[global_r * total_dim + d] : 0.0f;
    }
    for (int idx = tid; idx < total_elements; idx += 256) {
        int r = idx / total_dim;
        int d = idx % total_dim;
        int global_c = blockIdx.x * 16 + r;
        s_B[r][d] = (global_c < M) ? B[global_c * total_dim + d] : 0.0f;
    }
    __syncthreads();

    // Precompute norms and logs per row/col (only 16 threads needed)
    if (tid < 16) {
        int r = tid;
        float u2 = 0.0f;
        for (int k = 0; k < dim_h; ++k) {
            float ak = s_A[r][k];
            u2 += ak * ak;
        }
        s_u_sq[r] = u2;
        s_log_ra[r] = logf(fmaxf(s_A[r][total_dim - 1], eps));

        float v2 = 0.0f;
        for (int k = 0; k < dim_h; ++k) {
            float bk = s_B[r][k];
            v2 += bk * bk;
        }
        s_v_sq[r] = v2;
        s_log_rb[r] = logf(fmaxf(s_B[r][total_dim - 1], eps));
    }
    __syncthreads();

    if (row >= N || col >= M) return;

    // 1. Hyperbolic Poincare Geodesic Distance
    float diff_sq_h = 0.0f;
    #pragma unroll 4
    for (int k = 0; k < dim_h; ++k) {
        float d = s_A[ty][k] - s_B[tx][k];
        diff_sq_h += d * d;
    }

    float d_h = 0.0f;
    if (diff_sq_h > 1e-14f) {
        float denom_u = fmaxf(1.0f - c * s_u_sq[ty], 1e-7f);
        float denom_v = fmaxf(1.0f - c * s_v_sq[tx], 1e-7f);
        float xi = (c * diff_sq_h) / (denom_u * denom_v);
        d_h = (2.0f / sqrtf(c)) * gpu_stable_arsinh(sqrtf(fmaxf(xi, 0.0f)));
    }

    // 2. Toroidal Clifford Shortest-Path Distance
    float d_t_sq = 0.0f;
    #pragma unroll 4
    for (int k = 0; k < dim_t; ++k) {
        float diff = gpu_wrap_diff(s_A[ty][dim_h + k] - s_B[tx][dim_h + k]);
        d_t_sq += diff * diff;
    }
    float d_t = sqrtf(d_t_sq);

    // 3. Radial Shell Distance
    float d_r = fabsf(s_log_ra[ty] - s_log_rb[tx]);

    // Composite Metric
    float composite_sq = weight_h * (d_h * d_h) 
                       + weight_t * (d_t * d_t) 
                       + weight_r * (d_r * d_r);

    out_dist[row * M + col] = sqrtf(composite_sq);
}

__device__ __forceinline__ float gpu_wrap_diff_fast(float d) {
    const float two_pi = 6.28318530717958647692f;
    const float inv_two_pi = 0.15915494309189533577f;
    return d - two_pi * rintf(d * inv_two_pi);
}

#define TILE_N 64
#define TILE_M 64
#define STRIDE 36

extern "C" __global__ void k_yggdrasil_pairwise_distance_fast(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ out_dist,
    int N,
    int M,
    int dim_h,
    int dim_t,
    float c,
    float eps,
    float weight_h,
    float weight_t,
    float weight_r
) {
    __shared__ float s_A[TILE_N][STRIDE];
    __shared__ float s_B[TILE_M][STRIDE];
    __shared__ float s_u_sq[TILE_N];
    __shared__ float s_v_sq[TILE_M];
    __shared__ float s_log_ra[TILE_N];
    __shared__ float s_log_rb[TILE_M];

    int tx = threadIdx.x; // 0..15
    int ty = threadIdx.y; // 0..15
    int tid = ty * 16 + tx; // 0..255
    int total_dim = dim_h + dim_t + 1;

    int block_row = blockIdx.y * TILE_N;
    int block_col = blockIdx.x * TILE_M;

    // Coalesced load of A tile (64 rows x total_dim)
    int n_elements = TILE_N * total_dim;
    for (int i = tid; i < n_elements; i += 256) {
        int r = i / total_dim;
        int d = i % total_dim;
        int gr = block_row + r;
        s_A[r][d] = (gr < N) ? A[gr * total_dim + d] : 0.0f;
    }

    // Coalesced load of B tile (64 rows x total_dim)
    for (int i = tid; i < n_elements; i += 256) {
        int r = i / total_dim;
        int d = i % total_dim;
        int gc = block_col + r;
        s_B[r][d] = (gc < M) ? B[gc * total_dim + d] : 0.0f;
    }
    __syncthreads();

    // Precompute norms and logs
    if (tid < TILE_N) {
        int r = tid;
        float u2 = 0.0f;
        #pragma unroll 4
        for (int k = 0; k < dim_h; ++k) {
            float ak = s_A[r][k];
            u2 += ak * ak;
        }
        s_u_sq[r] = u2;
        s_log_ra[r] = logf(fmaxf(s_A[r][total_dim - 1], eps));
    }
    if (tid < TILE_M) {
        int r = tid;
        float v2 = 0.0f;
        #pragma unroll 4
        for (int k = 0; k < dim_h; ++k) {
            float bk = s_B[r][k];
            v2 += bk * bk;
        }
        s_v_sq[r] = v2;
        s_log_rb[r] = logf(fmaxf(s_B[r][total_dim - 1], eps));
    }
    __syncthreads();

    // Each thread computes 4 rows x 4 cols = 16 outputs
    int r_base = ty * 4;
    int c_base = tx * 4;

    float dot_h[4][4];
    float d_t_sq[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            dot_h[i][j] = 0.0f;
            d_t_sq[i][j] = 0.0f;
        }
    }

    // 1. Hyperbolic GEMM via dot product in shared memory
    for (int k = 0; k < dim_h; ++k) {
        float a_k[4];
        float b_k[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) a_k[i] = s_A[r_base + i][k];
        #pragma unroll
        for (int j = 0; j < 4; ++j) b_k[j] = s_B[c_base + j][k];

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                dot_h[i][j] = fmaf(a_k[i], b_k[j], dot_h[i][j]);
            }
        }
    }

    // 2. Torus loop with branchless wrap
    for (int k = 0; k < dim_t; ++k) {
        float a_k[4];
        float b_k[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) a_k[i] = s_A[r_base + i][dim_h + k];
        #pragma unroll
        for (int j = 0; j < 4; ++j) b_k[j] = s_B[c_base + j][dim_h + k];

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float d = gpu_wrap_diff_fast(a_k[i] - b_k[j]);
                d_t_sq[i][j] = fmaf(d, d, d_t_sq[i][j]);
            }
        }
    }

    // 3. Epilogue and write
    float inv_sqrt_c = 1.0f / sqrtf(c);
    float v_sq_reg[4];
    float denom_v_reg[4];
    float log_rb_reg[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        v_sq_reg[j] = s_v_sq[c_base + j];
        denom_v_reg[j] = fmaxf(1.0f - c * v_sq_reg[j], 1e-7f);
        log_rb_reg[j] = s_log_rb[c_base + j];
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int gr = block_row + r_base + i;
        if (gr >= N) continue;

        float u_sq = s_u_sq[r_base + i];
        float denom_u = fmaxf(1.0f - c * u_sq, 1e-7f);
        float log_ra = s_log_ra[r_base + i];

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int gc = block_col + c_base + j;
            if (gc >= M) continue;

            float d2_h = fmaxf(u_sq + v_sq_reg[j] - 2.0f * dot_h[i][j], 0.0f);
            float d_h = 0.0f;
            if (d2_h > 1e-14f) {
                float xi = (c * d2_h) / (denom_u * denom_v_reg[j]);
                d_h = (2.0f * inv_sqrt_c) * gpu_stable_arsinh(sqrtf(fmaxf(xi, 0.0f)));
            }

            float d_t = sqrtf(d_t_sq[i][j]);
            float d_r = fabsf(log_ra - log_rb_reg[j]);

            float comp_sq = weight_h * (d_h * d_h) + weight_t * (d_t * d_t) + weight_r * (d_r * d_r);
            out_dist[gr * M + gc] = sqrtf(comp_sq);
        }
    }
}


// ---------------------------------------------------------------------------
// Kernel 2: Batch Mobius Addition in the Poincare Ball
// out[i] = u[i] (plus_c) v[i]
// ---------------------------------------------------------------------------
extern "C" __global__ void k_poincare_mobius_add(
    const float* __restrict__ U,
    const float* __restrict__ V,
    float* __restrict__ out,
    int N,
    int dim_h,
    float c
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const float* u_ptr = U + i * dim_h;
    const float* v_ptr = V + i * dim_h;
    float* out_ptr = out + i * dim_h;

    float dot = 0.0f;
    float u_sq = 0.0f;
    float v_sq = 0.0f;

    #pragma unroll 4
    for (int k = 0; k < dim_h; ++k) {
        float uk = u_ptr[k];
        float vk = v_ptr[k];
        dot += uk * vk;
        u_sq += uk * uk;
        v_sq += vk * vk;
    }

    float alpha = 1.0f + 2.0f * c * dot + c * v_sq;
    float beta = 1.0f - c * u_sq;
    float denom = fmaxf(1.0f + 2.0f * c * dot + c * c * u_sq * v_sq, 1e-10f);

    #pragma unroll 4
    for (int k = 0; k < dim_h; ++k) {
        out_ptr[k] = (alpha * u_ptr[k] + beta * v_ptr[k]) / denom;
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: Batch Projection & Boundary Retraction on GPU
// ---------------------------------------------------------------------------
extern "C" __global__ void k_yggdrasil_project(
    float* __restrict__ points,
    int N,
    int dim_h,
    int dim_t,
    float c,
    float eps
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int total_dim = dim_h + dim_t + 1;
    float* p_ptr = points + i * total_dim;

    // 1. Hyperbolic Retraction: norm <= (1 - eps) / sqrt(c)
    float max_r = (1.0f - eps) / sqrtf(c);
    float sq_norm = 0.0f;
    #pragma unroll 4
    for (int k = 0; k < dim_h; ++k) {
        sq_norm += p_ptr[k] * p_ptr[k];
    }
    float norm = sqrtf(sq_norm + 1e-18f);
    if (norm > max_r) {
        float scale = max_r / norm;
        #pragma unroll 4
        for (int k = 0; k < dim_h; ++k) {
            p_ptr[k] *= scale;
        }
    }

    // 2. Torus Canonical Projection: [-PI, PI)
    float* torus_ptr = p_ptr + dim_h;
    #pragma unroll 4
    for (int k = 0; k < dim_t; ++k) {
        torus_ptr[k] = gpu_wrap_angle(torus_ptr[k]);
    }

    // 3. Radial Retraction: r >= eps
    int r_idx = dim_h + dim_t;
    if (p_ptr[r_idx] < eps) {
        p_ptr[r_idx] = eps;
    }
}

// ---------------------------------------------------------------------------
// Kernel 4: Batch Torus Lie Group Shift / Binding
// out[idx] = (U[idx] + V[idx]) mod 2PI
// ---------------------------------------------------------------------------
extern "C" __global__ void k_torus_bind(
    const float* __restrict__ U,
    const float* __restrict__ V,
    float* __restrict__ out,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    out[idx] = gpu_wrap_angle(U[idx] + V[idx]);
}

// ---------------------------------------------------------------------------
// Kernel 5: Batch Torus Unbind / Inverse Shift
// out[idx] = (U[idx] - V[idx]) mod 2PI
// ---------------------------------------------------------------------------
extern "C" __global__ void k_torus_unbind(
    const float* __restrict__ U,
    const float* __restrict__ V,
    float* __restrict__ out,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    out[idx] = gpu_wrap_angle(U[idx] - V[idx]);
}

// ---------------------------------------------------------------------------
// Kernel 6: Riemannian Gradient Retraction on Poincare Ball
//   p' = p (+) exp-step( -lr * ((1 - c|p|^2)^2 / 4) * g )
// The Riemannian step is a multiple of the Euclidean gradient (w = kappa * g), so everything needs only the
// three scalars |p|^2, |g|^2 and <p, g>: no per-thread array, no limit on dim_h (the first version kept
// the step in a 64-float local array and left every dimension >= 64 unwritten).
// ---------------------------------------------------------------------------
extern "C" __global__ void k_riemannian_retract(
    const float* __restrict__ points,
    const float* __restrict__ grads,
    float lr,
    float* __restrict__ out,
    int N,
    int dim_h,
    float c,
    float eps
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const float* p_ptr = points + (size_t)i * dim_h;
    const float* g_ptr = grads + (size_t)i * dim_h;
    float* out_ptr = out + (size_t)i * dim_h;

    float p_sq = 0.0f, g_sq = 0.0f, pg = 0.0f;
    for (int k = 0; k < dim_h; ++k) {
        float pk = p_ptr[k];
        float gk = g_ptr[k];
        p_sq += pk * pk;
        g_sq += gk * gk;
        pg += pk * gk;
    }

    float factor = fmaxf(1.0f - c * p_sq, 1e-7f);
    float inv_metric = (factor * factor) * 0.25f;                       // inverse metric (1 - c|p|^2)^2 / 4
    float v_norm = fabsf(lr) * inv_metric * sqrtf(g_sq);                // |v| of the tangent step v = -lr * inv_metric * g

    if (v_norm < 1e-10f) {
        for (int k = 0; k < dim_h; ++k) out_ptr[k] = p_ptr[k];
        return;
    }

    float sqrt_c = sqrtf(c);
    float lambda_p = 2.0f / factor;                                     // conformal factor at p
    float scale_v = tanhf(0.5f * sqrt_c * lambda_p * v_norm) / (sqrt_c * v_norm);
    float kappa = -lr * inv_metric * scale_v;                           // w = exp-map step = kappa * g

    float dot = kappa * pg;                                             // <p, w>
    float w_sq = kappa * kappa * g_sq;                                  // |w|^2
    float alpha = 1.0f + 2.0f * c * dot + c * w_sq;
    float beta = 1.0f - c * p_sq;
    float denom = fmaxf(1.0f + 2.0f * c * dot + c * c * p_sq * w_sq, 1e-10f);

    // |p (+) w|^2 in closed form, to apply the boundary retraction without a second pass
    float res_sq = (alpha * alpha * p_sq + 2.0f * alpha * beta * kappa * pg + beta * beta * kappa * kappa * g_sq) / (denom * denom);
    float res_norm = sqrtf(res_sq + 1e-18f);
    float max_r = (1.0f - eps) / sqrt_c;
    float r_scale = (res_norm > max_r) ? (max_r / res_norm) : 1.0f;

    float inv_denom = r_scale / denom;
    for (int k = 0; k < dim_h; ++k) {
        out_ptr[k] = (alpha * p_ptr[k] + beta * kappa * g_ptr[k]) * inv_denom;
    }
}
