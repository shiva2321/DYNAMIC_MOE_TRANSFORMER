pub mod cuda_driver;
pub mod cpu;
pub mod gpu;
pub mod manifold;

use std::cell::RefCell;
use std::ffi::{c_char, CString};
use std::panic::catch_unwind;

pub use cpu::*;
pub use gpu::*;
pub use manifold::*;

thread_local! {
    static LAST_ERROR: RefCell<String> = RefCell::new(String::new());
}

fn set_error(err: String) {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = err;
    });
}

#[no_mangle]
pub extern "C" fn yggdrasil_get_last_error(buf: *mut c_char, max_len: usize) -> i32 {
    if buf.is_null() || max_len == 0 {
        return -1;
    }
    LAST_ERROR.with(|e| {
        let err = e.borrow();
        let c_str = CString::new(err.clone()).unwrap_or_default();
        let bytes = c_str.as_bytes_with_nul();
        let copy_len = bytes.len().min(max_len);
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr() as *const c_char, buf, copy_len);
            if copy_len > 0 {
                *buf.add(copy_len - 1) = 0;
            }
        }
        0
    })
}

// ---------------------------------------------------------------------------
// C-ABI GPU Lifecycle & Kernel Execution
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn yggdrasil_gpu_init(device_ordinal: i32) -> *mut YggdrasilGpuEngine {
    let result = catch_unwind(|| {
        match YggdrasilGpuEngine::new(device_ordinal) {
            Ok(engine) => Box::into_raw(Box::new(engine)),
            Err(e) => {
                set_error(e);
                std::ptr::null_mut()
            }
        }
    });
    result.unwrap_or(std::ptr::null_mut())
}

#[no_mangle]
pub extern "C" fn yggdrasil_gpu_destroy(engine: *mut YggdrasilGpuEngine) {
    if !engine.is_null() {
        let _ = catch_unwind(|| {
            unsafe {
                drop(Box::from_raw(engine));
            }
        });
    }
}

#[no_mangle]
pub extern "C" fn yggdrasil_gpu_pairwise_distance(
    engine: *mut YggdrasilGpuEngine,
    a_dptr: u64,
    b_dptr: u64,
    out_dptr: u64,
    n: usize,
    m: usize,
    dim_h: u32,
    dim_t: u32,
    c: f32,
    eps: f32,
    weight_h: f32,
    weight_t: f32,
    weight_r: f32,
) -> i32 {
    if engine.is_null() {
        set_error("Engine handle is null".into());
        return -1;
    }
    let config = YggdrasilConfig {
        dim_h,
        dim_t,
        c,
        eps,
        weight_h,
        weight_t,
        weight_r,
    };
    let eng = unsafe { &*engine };
    match eng.pairwise_distance_gpu(a_dptr, b_dptr, out_dptr, n, m, &config) {
        Ok(_) => 0,
        Err(e) => {
            set_error(e);
            -2
        }
    }
}

#[no_mangle]
pub extern "C" fn yggdrasil_gpu_pairwise_distance_fast(
    engine: *mut YggdrasilGpuEngine,
    a_dptr: u64,
    b_dptr: u64,
    out_dptr: u64,
    n: usize,
    m: usize,
    dim_h: u32,
    dim_t: u32,
    c: f32,
    eps: f32,
    weight_h: f32,
    weight_t: f32,
    weight_r: f32,
) -> i32 {
    if engine.is_null() {
        set_error("Engine handle is null".into());
        return -1;
    }
    let config = YggdrasilConfig {
        dim_h,
        dim_t,
        c,
        eps,
        weight_h,
        weight_t,
        weight_r,
    };
    let eng = unsafe { &*engine };
    match eng.pairwise_distance_gpu_fast(a_dptr, b_dptr, out_dptr, n, m, &config) {
        Ok(_) => 0,
        Err(e) => {
            set_error(e);
            -2
        }
    }
}

#[no_mangle]
pub extern "C" fn yggdrasil_gpu_mobius_add(
    engine: *mut YggdrasilGpuEngine,
    u_dptr: u64,
    v_dptr: u64,
    out_dptr: u64,
    n: usize,
    dim_h: usize,
    c: f32,
) -> i32 {
    if engine.is_null() {
        set_error("Engine handle is null".into());
        return -1;
    }
    let eng = unsafe { &*engine };
    match eng.mobius_add_gpu(u_dptr, v_dptr, out_dptr, n, dim_h, c) {
        Ok(_) => 0,
        Err(e) => {
            set_error(e);
            -2
        }
    }
}

#[no_mangle]
pub extern "C" fn yggdrasil_gpu_project(
    engine: *mut YggdrasilGpuEngine,
    points_dptr: u64,
    n: usize,
    dim_h: u32,
    dim_t: u32,
    c: f32,
    eps: f32,
) -> i32 {
    if engine.is_null() {
        set_error("Engine handle is null".into());
        return -1;
    }
    let config = YggdrasilConfig {
        dim_h,
        dim_t,
        c,
        eps,
        ..Default::default()
    };
    let eng = unsafe { &*engine };
    match eng.project_gpu(points_dptr, n, &config) {
        Ok(_) => 0,
        Err(e) => {
            set_error(e);
            -2
        }
    }
}

#[no_mangle]
pub extern "C" fn yggdrasil_gpu_torus_bind(
    engine: *mut YggdrasilGpuEngine,
    u_dptr: u64,
    v_dptr: u64,
    out_dptr: u64,
    total_elements: usize,
) -> i32 {
    if engine.is_null() {
        set_error("Engine handle is null".into());
        return -1;
    }
    let eng = unsafe { &*engine };
    match eng.torus_bind_gpu(u_dptr, v_dptr, out_dptr, total_elements) {
        Ok(_) => 0,
        Err(e) => {
            set_error(e);
            -2
        }
    }
}

#[no_mangle]
pub extern "C" fn yggdrasil_gpu_torus_unbind(
    engine: *mut YggdrasilGpuEngine,
    u_dptr: u64,
    v_dptr: u64,
    out_dptr: u64,
    total_elements: usize,
) -> i32 {
    if engine.is_null() {
        set_error("Engine handle is null".into());
        return -1;
    }
    let eng = unsafe { &*engine };
    match eng.torus_unbind_gpu(u_dptr, v_dptr, out_dptr, total_elements) {
        Ok(_) => 0,
        Err(e) => {
            set_error(e);
            -2
        }
    }
}

#[no_mangle]
pub extern "C" fn yggdrasil_gpu_riemannian_retract(
    engine: *mut YggdrasilGpuEngine,
    points_dptr: u64,
    grads_dptr: u64,
    out_dptr: u64,
    n: usize,
    dim_h: usize,
    lr: f32,
    c: f32,
    eps: f32,
) -> i32 {
    if engine.is_null() {
        set_error("Engine handle is null".into());
        return -1;
    }
    let eng = unsafe { &*engine };
    match eng.riemannian_retract_gpu(points_dptr, grads_dptr, out_dptr, n, dim_h, lr, c, eps) {
        Ok(_) => 0,
        Err(e) => {
            set_error(e);
            -2
        }
    }
}

// ---------------------------------------------------------------------------
// C-ABI Multi-Threaded Host / CPU Execution
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn yggdrasil_cpu_pairwise_distance(
    a_ptr: *const f32,
    b_ptr: *const f32,
    out_ptr: *mut f32,
    n: usize,
    m: usize,
    dim_h: u32,
    dim_t: u32,
    c: f32,
    eps: f32,
    weight_h: f32,
    weight_t: f32,
    weight_r: f32,
) -> i32 {
    if a_ptr.is_null() || b_ptr.is_null() || out_ptr.is_null() {
        set_error("Null pointer in CPU pairwise distance".into());
        return -1;
    }
    let config = YggdrasilConfig {
        dim_h,
        dim_t,
        c,
        eps,
        weight_h,
        weight_t,
        weight_r,
    };
    let d = config.dim_total();
    let a = unsafe { std::slice::from_raw_parts(a_ptr, n * d) };
    let b = unsafe { std::slice::from_raw_parts(b_ptr, m * d) };
    let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, n * m) };

    batch_pairwise_distance_cpu(a, b, n, m, &config, out);
    0
}

#[no_mangle]
pub extern "C" fn yggdrasil_cpu_mobius_add(
    u_ptr: *const f32,
    v_ptr: *const f32,
    out_ptr: *mut f32,
    n: usize,
    dim_h: usize,
    c: f32,
) -> i32 {
    if u_ptr.is_null() || v_ptr.is_null() || out_ptr.is_null() {
        set_error("Null pointer in CPU mobius add".into());
        return -1;
    }
    let u = unsafe { std::slice::from_raw_parts(u_ptr, n * dim_h) };
    let v = unsafe { std::slice::from_raw_parts(v_ptr, n * dim_h) };
    let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, n * dim_h) };

    batch_mobius_add_cpu(u, v, n, dim_h, c, out);
    0
}

#[no_mangle]
pub extern "C" fn yggdrasil_cpu_project(
    points_ptr: *mut f32,
    n: usize,
    dim_h: u32,
    dim_t: u32,
    c: f32,
    eps: f32,
) -> i32 {
    if points_ptr.is_null() {
        set_error("Null pointer in CPU project".into());
        return -1;
    }
    let config = YggdrasilConfig {
        dim_h,
        dim_t,
        c,
        eps,
        ..Default::default()
    };
    let d = config.dim_total();
    let points = unsafe { std::slice::from_raw_parts_mut(points_ptr, n * d) };
    batch_project_cpu(points, n, &config);
    0
}

#[no_mangle]
pub extern "C" fn yggdrasil_cpu_torus_bind(
    u_ptr: *const f32,
    v_ptr: *const f32,
    out_ptr: *mut f32,
    n: usize,
    dim_t: usize,
) -> i32 {
    if u_ptr.is_null() || v_ptr.is_null() || out_ptr.is_null() {
        set_error("Null pointer in CPU torus bind".into());
        return -1;
    }
    let u = unsafe { std::slice::from_raw_parts(u_ptr, n * dim_t) };
    let v = unsafe { std::slice::from_raw_parts(v_ptr, n * dim_t) };
    let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, n * dim_t) };

    batch_torus_bind_cpu(u, v, n, dim_t, out);
    0
}
