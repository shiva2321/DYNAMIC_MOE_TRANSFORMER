use crate::cuda_driver::*;
use crate::manifold::YggdrasilConfig;
use std::ffi::c_void;
use std::sync::Arc;

pub const EMBEDDED_PTX: &[u8] = include_bytes!("gpu_kernels.ptx");

/// Safe GPU execution engine for the Yggdrasil Geometric Manifold on NVIDIA GPUs
pub struct YggdrasilGpuEngine {
    pub ctx: Arc<CudaDeviceContext>,
    module: CudaModule,
    fn_pairwise_dist: CUfunction,
    fn_pairwise_dist_fast: CUfunction,
    fn_mobius_add: CUfunction,
    fn_project: CUfunction,
    fn_torus_bind: CUfunction,
    fn_torus_unbind: CUfunction,
    fn_riemannian_retract: CUfunction,
}

impl YggdrasilGpuEngine {
    /// Initializes GPU context on specified device and loads compiled PTX kernels
    pub fn new(device_ordinal: i32) -> Result<Self, String> {
        let ctx = CudaDeviceContext::init_device(device_ordinal)?;
        let module = CudaModule::load_from_ptx(Arc::clone(&ctx.driver), EMBEDDED_PTX)?;

        let fn_pairwise_dist = module.get_function("k_yggdrasil_pairwise_distance")?;
        let fn_pairwise_dist_fast = module.get_function("k_yggdrasil_pairwise_distance_fast")?;
        let fn_mobius_add = module.get_function("k_poincare_mobius_add")?;
        let fn_project = module.get_function("k_yggdrasil_project")?;
        let fn_torus_bind = module.get_function("k_torus_bind")?;
        let fn_torus_unbind = module.get_function("k_torus_unbind")?;
        let fn_riemannian_retract = module.get_function("k_riemannian_retract")?;

        Ok(Self {
            ctx,
            module,
            fn_pairwise_dist,
            fn_pairwise_dist_fast,
            fn_mobius_add,
            fn_project,
            fn_torus_bind,
            fn_torus_unbind,
            fn_riemannian_retract,
        })
    }


    /// Computes N x M Pairwise Distance Matrix directly on GPU using raw device pointers (Zero-Copy)
    pub fn pairwise_distance_gpu(
        &self,
        a_dptr: u64,
        b_dptr: u64,
        out_dptr: u64,
        n: usize,
        m: usize,
        config: &YggdrasilConfig,
    ) -> Result<(), String> {
        let block = (16u32, 16u32, 1u32);
        let grid = (
            ((m as u32 + block.0 - 1) / block.0).max(1),
            ((n as u32 + block.1 - 1) / block.1).max(1),
            1u32,
        );

        let mut a_ptr = a_dptr as *mut c_void;
        let mut b_ptr = b_dptr as *mut c_void;
        let mut out_ptr = out_dptr as *mut c_void;
        let mut n_val = n as i32;
        let mut m_val = m as i32;
        let mut dim_h_val = config.dim_h as i32;
        let mut dim_t_val = config.dim_t as i32;
        let mut c_val = config.c;
        let mut eps_val = config.eps;
        let mut w_h = config.weight_h;
        let mut w_t = config.weight_t;
        let mut w_r = config.weight_r;

        let params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut b_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut dim_h_val as *mut _ as *mut c_void,
            &mut dim_t_val as *mut _ as *mut c_void,
            &mut c_val as *mut _ as *mut c_void,
            &mut eps_val as *mut _ as *mut c_void,
            &mut w_h as *mut _ as *mut c_void,
            &mut w_t as *mut _ as *mut c_void,
            &mut w_r as *mut _ as *mut c_void,
        ];

        self.module.launch(self.fn_pairwise_dist, grid, block, 0, params)?;
        self.ctx.synchronize()?;
        Ok(())
    }

    /// Computes N x M Pairwise Distance Matrix using 64x64 tiled high-throughput kernel
    pub fn pairwise_distance_gpu_fast(
        &self,
        a_dptr: u64,
        b_dptr: u64,
        out_dptr: u64,
        n: usize,
        m: usize,
        config: &YggdrasilConfig,
    ) -> Result<(), String> {
        let block = (16u32, 16u32, 1u32);
        let grid = (
            ((m as u32 + 63) / 64).max(1),
            ((n as u32 + 63) / 64).max(1),
            1u32,
        );

        let mut a_ptr = a_dptr as *mut c_void;
        let mut b_ptr = b_dptr as *mut c_void;
        let mut out_ptr = out_dptr as *mut c_void;
        let mut n_val = n as i32;
        let mut m_val = m as i32;
        let mut dim_h_val = config.dim_h as i32;
        let mut dim_t_val = config.dim_t as i32;
        let mut c_val = config.c;
        let mut eps_val = config.eps;
        let mut w_h = config.weight_h;
        let mut w_t = config.weight_t;
        let mut w_r = config.weight_r;

        let params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut b_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut dim_h_val as *mut _ as *mut c_void,
            &mut dim_t_val as *mut _ as *mut c_void,
            &mut c_val as *mut _ as *mut c_void,
            &mut eps_val as *mut _ as *mut c_void,
            &mut w_h as *mut _ as *mut c_void,
            &mut w_t as *mut _ as *mut c_void,
            &mut w_r as *mut _ as *mut c_void,
        ];

        self.module.launch(self.fn_pairwise_dist_fast, grid, block, 0, params)?;
        self.ctx.synchronize()?;
        Ok(())
    }


    /// End-to-end pairwise distance with automatic host-to-device buffer management
    pub fn pairwise_distance_host(
        &self,
        a: &[f32],
        b: &[f32],
        n: usize,
        m: usize,
        config: &YggdrasilConfig,
        out: &mut [f32],
    ) -> Result<(), String> {
        let d = config.dim_total();
        assert_eq!(a.len(), n * d);
        assert_eq!(b.len(), m * d);
        assert_eq!(out.len(), n * m);

        let buf_a = CudaDeviceBuffer::allocate(&self.ctx, a.len() * 4)?;
        let buf_b = CudaDeviceBuffer::allocate(&self.ctx, b.len() * 4)?;
        let buf_out = CudaDeviceBuffer::allocate(&self.ctx, out.len() * 4)?;

        buf_a.copy_from_host(a)?;
        buf_b.copy_from_host(b)?;

        self.pairwise_distance_gpu(buf_a.dptr, buf_b.dptr, buf_out.dptr, n, m, config)?;

        buf_out.copy_to_host(out)?;
        Ok(())
    }

    /// Batch Mobius Addition on GPU
    pub fn mobius_add_gpu(
        &self,
        u_dptr: u64,
        v_dptr: u64,
        out_dptr: u64,
        n: usize,
        dim_h: usize,
        c: f32,
    ) -> Result<(), String> {
        let block_size = 256u32;
        let grid_size = ((n as u32 + block_size - 1) / block_size).max(1);

        let mut u_ptr = u_dptr as *mut c_void;
        let mut v_ptr = v_dptr as *mut c_void;
        let mut out_ptr = out_dptr as *mut c_void;
        let mut n_val = n as i32;
        let mut dim_h_val = dim_h as i32;
        let mut c_val = c;

        let params: Vec<*mut c_void> = vec![
            &mut u_ptr as *mut _ as *mut c_void,
            &mut v_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
            &mut dim_h_val as *mut _ as *mut c_void,
            &mut c_val as *mut _ as *mut c_void,
        ];

        self.module.launch(
            self.fn_mobius_add,
            (grid_size, 1, 1),
            (block_size, 1, 1),
            0,
            params,
        )?;
        self.ctx.synchronize()?;
        Ok(())
    }

    /// Batch Projection & Boundary Retraction on GPU
    pub fn project_gpu(
        &self,
        points_dptr: u64,
        n: usize,
        config: &YggdrasilConfig,
    ) -> Result<(), String> {
        let block_size = 256u32;
        let grid_size = ((n as u32 + block_size - 1) / block_size).max(1);

        let mut p_ptr = points_dptr as *mut c_void;
        let mut n_val = n as i32;
        let mut dim_h_val = config.dim_h as i32;
        let mut dim_t_val = config.dim_t as i32;
        let mut c_val = config.c;
        let mut eps_val = config.eps;

        let params: Vec<*mut c_void> = vec![
            &mut p_ptr as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
            &mut dim_h_val as *mut _ as *mut c_void,
            &mut dim_t_val as *mut _ as *mut c_void,
            &mut c_val as *mut _ as *mut c_void,
            &mut eps_val as *mut _ as *mut c_void,
        ];

        self.module.launch(
            self.fn_project,
            (grid_size, 1, 1),
            (block_size, 1, 1),
            0,
            params,
        )?;
        self.ctx.synchronize()?;
        Ok(())
    }

    /// Batch Torus Lie Group Shift / Binding on GPU
    pub fn torus_bind_gpu(
        &self,
        u_dptr: u64,
        v_dptr: u64,
        out_dptr: u64,
        total_elements: usize,
    ) -> Result<(), String> {
        let block_size = 256u32;
        let grid_size = ((total_elements as u32 + block_size - 1) / block_size).max(1);

        let mut u_ptr = u_dptr as *mut c_void;
        let mut v_ptr = v_dptr as *mut c_void;
        let mut out_ptr = out_dptr as *mut c_void;
        let mut total_val = total_elements as i32;

        let params: Vec<*mut c_void> = vec![
            &mut u_ptr as *mut _ as *mut c_void,
            &mut v_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut total_val as *mut _ as *mut c_void,
        ];

        self.module.launch(
            self.fn_torus_bind,
            (grid_size, 1, 1),
            (block_size, 1, 1),
            0,
            params,
        )?;
        self.ctx.synchronize()?;
        Ok(())
    }

    /// Batch Torus Lie Group Unbinding / Inverse Shift on GPU
    pub fn torus_unbind_gpu(
        &self,
        u_dptr: u64,
        v_dptr: u64,
        out_dptr: u64,
        total_elements: usize,
    ) -> Result<(), String> {
        let block_size = 256u32;
        let grid_size = ((total_elements as u32 + block_size - 1) / block_size).max(1);

        let mut u_ptr = u_dptr as *mut c_void;
        let mut v_ptr = v_dptr as *mut c_void;
        let mut out_ptr = out_dptr as *mut c_void;
        let mut total_val = total_elements as i32;

        let params: Vec<*mut c_void> = vec![
            &mut u_ptr as *mut _ as *mut c_void,
            &mut v_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut total_val as *mut _ as *mut c_void,
        ];

        self.module.launch(
            self.fn_torus_unbind,
            (grid_size, 1, 1),
            (block_size, 1, 1),
            0,
            params,
        )?;
        self.ctx.synchronize()?;
        Ok(())
    }

    /// Riemannian gradient retraction on GPU for manifold parameter updates
    pub fn riemannian_retract_gpu(
        &self,
        points_dptr: u64,
        grads_dptr: u64,
        out_dptr: u64,
        n: usize,
        dim_h: usize,
        lr: f32,
        c: f32,
        eps: f32,
    ) -> Result<(), String> {
        let block_size = 256u32;
        let grid_size = ((n as u32 + block_size - 1) / block_size).max(1);

        let mut p_ptr = points_dptr as *mut c_void;
        let mut g_ptr = grads_dptr as *mut c_void;
        let mut out_ptr = out_dptr as *mut c_void;
        let mut lr_val = lr;
        let mut n_val = n as i32;
        let mut dim_h_val = dim_h as i32;
        let mut c_val = c;
        let mut eps_val = eps;

        let params: Vec<*mut c_void> = vec![
            &mut p_ptr as *mut _ as *mut c_void,
            &mut g_ptr as *mut _ as *mut c_void,
            &mut lr_val as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
            &mut dim_h_val as *mut _ as *mut c_void,
            &mut c_val as *mut _ as *mut c_void,
            &mut eps_val as *mut _ as *mut c_void,
        ];

        self.module.launch(
            self.fn_riemannian_retract,
            (grid_size, 1, 1),
            (block_size, 1, 1),
            0,
            params,
        )?;
        self.ctx.synchronize()?;
        Ok(())
    }
}
