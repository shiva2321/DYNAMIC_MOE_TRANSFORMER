#![allow(non_snake_case, dead_code, unused_imports)]

use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::Arc;

pub type CUresult = i32;
pub type CUdevice = i32;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUdeviceptr = u64;
pub type CUstream = *mut c_void;

pub const CUDA_SUCCESS: CUresult = 0;

// Windows dynamic library loading FFI
extern "system" {
    fn LoadLibraryA(lpLibFileName: *const c_char) -> *mut c_void;
    fn GetProcAddress(hModule: *mut c_void, lpProcName: *const c_char) -> *mut c_void;
    fn FreeLibrary(hLibModule: *mut c_void) -> i32;
}

#[derive(Clone)]
pub struct CudaDriver {
    pub cu_init: unsafe extern "system" fn(flags: u32) -> CUresult,
    pub cu_device_get: unsafe extern "system" fn(device: *mut CUdevice, ordinal: i32) -> CUresult,
    pub cu_device_get_name: unsafe extern "system" fn(name: *mut c_char, len: i32, dev: CUdevice) -> CUresult,
    pub cu_device_primary_ctx_retain: unsafe extern "system" fn(pctx: *mut CUcontext, dev: CUdevice) -> CUresult,
    pub cu_ctx_set_current: unsafe extern "system" fn(ctx: CUcontext) -> CUresult,
    pub cu_mem_alloc_v2: unsafe extern "system" fn(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult,
    pub cu_mem_free_v2: unsafe extern "system" fn(dptr: CUdeviceptr) -> CUresult,
    pub cu_memcpy_htod_v2: unsafe extern "system" fn(dstDevice: CUdeviceptr, srcHost: *const c_void, ByteCount: usize) -> CUresult,
    pub cu_memcpy_dtoh_v2: unsafe extern "system" fn(dstHost: *mut c_void, srcDevice: CUdeviceptr, ByteCount: usize) -> CUresult,
    pub cu_memcpy_dtod_v2: unsafe extern "system" fn(dstDevice: CUdeviceptr, srcDevice: CUdeviceptr, ByteCount: usize) -> CUresult,
    pub cu_module_load_data: unsafe extern "system" fn(module: *mut CUmodule, image: *const c_void) -> CUresult,
    pub cu_module_get_function: unsafe extern "system" fn(hfunc: *mut CUfunction, hmod: CUmodule, name: *const c_char) -> CUresult,
    pub cu_launch_kernel: unsafe extern "system" fn(
        f: CUfunction,
        gridDimX: u32, gridDimY: u32, gridDimZ: u32,
        blockDimX: u32, blockDimY: u32, blockDimZ: u32,
        sharedMemBytes: u32,
        hStream: CUstream,
        kernelParams: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CUresult,
    pub cu_ctx_synchronize: unsafe extern "system" fn() -> CUresult,
}

impl CudaDriver {
    pub fn load() -> Result<Self, String> {
        let dll_name = CString::new("nvcuda.dll").map_err(|e| e.to_string())?;
        let handle = unsafe { LoadLibraryA(dll_name.as_ptr()) };
        if handle.is_null() {
            return Err("Failed to load nvcuda.dll - NVIDIA driver not found".into());
        }

        unsafe fn get_sym<T>(handle: *mut c_void, name: &str) -> Result<T, String> {
            let c_name = CString::new(name).map_err(|e| e.to_string())?;
            let proc = GetProcAddress(handle, c_name.as_ptr());
            if proc.is_null() {
                return Err(format!("CUDA symbol not found: {}", name));
            }
            Ok(std::mem::transmute_copy(&proc))
        }

        unsafe {
            Ok(Self {
                cu_init: get_sym(handle, "cuInit")?,
                cu_device_get: get_sym(handle, "cuDeviceGet")?,
                cu_device_get_name: get_sym(handle, "cuDeviceGetName")?,
                cu_device_primary_ctx_retain: get_sym(handle, "cuDevicePrimaryCtxRetain")?,
                cu_ctx_set_current: get_sym(handle, "cuCtxSetCurrent")?,
                cu_mem_alloc_v2: get_sym(handle, "cuMemAlloc_v2")?,
                cu_mem_free_v2: get_sym(handle, "cuMemFree_v2")?,
                cu_memcpy_htod_v2: get_sym(handle, "cuMemcpyHtoD_v2")?,
                cu_memcpy_dtoh_v2: get_sym(handle, "cuMemcpyDtoH_v2")?,
                cu_memcpy_dtod_v2: get_sym(handle, "cuMemcpyDtoD_v2")?,
                cu_module_load_data: get_sym(handle, "cuModuleLoadData")?,
                cu_module_get_function: get_sym(handle, "cuModuleGetFunction")?,
                cu_launch_kernel: get_sym(handle, "cuLaunchKernel")?,
                cu_ctx_synchronize: get_sym(handle, "cuCtxSynchronize")?,
            })
        }
    }
}

/// Safe RAII GPU Context
pub struct CudaDeviceContext {
    pub driver: Arc<CudaDriver>,
    pub device: CUdevice,
    pub ctx: CUcontext,
    pub device_name: String,
}

impl CudaDeviceContext {
    pub fn init_device(device_ordinal: i32) -> Result<Arc<Self>, String> {
        let driver = Arc::new(CudaDriver::load()?);
        unsafe {
            let res = (driver.cu_init)(0);
            if res != CUDA_SUCCESS {
                return Err(format!("cuInit failed with code: {}", res));
            }

            let mut dev: CUdevice = 0;
            let res = (driver.cu_device_get)(&mut dev, device_ordinal);
            if res != CUDA_SUCCESS {
                return Err(format!("cuDeviceGet failed with code: {}", res));
            }

            let mut name_buf = [0i8; 256];
            (driver.cu_device_get_name)(name_buf.as_mut_ptr(), 256, dev);
            let dev_name = CStr::from_ptr(name_buf.as_ptr())
                .to_string_lossy()
                .into_owned();

            let mut ctx: CUcontext = std::ptr::null_mut();
            let res = (driver.cu_device_primary_ctx_retain)(&mut ctx, dev);
            if res != CUDA_SUCCESS {
                return Err(format!("cuDevicePrimaryCtxRetain failed with code: {}", res));
            }

            let res = (driver.cu_ctx_set_current)(ctx);
            if res != CUDA_SUCCESS {
                return Err(format!("cuCtxSetCurrent failed with code: {}", res));
            }

            Ok(Arc::new(Self {
                driver,
                device: dev,
                ctx,
                device_name: dev_name,
            }))
        }
    }

    pub fn synchronize(&self) -> Result<(), String> {
        unsafe {
            let res = (self.driver.cu_ctx_synchronize)();
            if res != CUDA_SUCCESS {
                return Err(format!("cuCtxSynchronize failed with code: {}", res));
            }
            Ok(())
        }
    }
}

/// RAII GPU Device Memory Buffer. Guaranteed memory safety: frees device memory on drop.
pub struct CudaDeviceBuffer {
    driver: Arc<CudaDriver>,
    pub dptr: CUdeviceptr,
    pub bytes: usize,
}

impl CudaDeviceBuffer {
    pub fn allocate(ctx: &CudaDeviceContext, bytes: usize) -> Result<Self, String> {
        let mut dptr: CUdeviceptr = 0;
        unsafe {
            let res = (ctx.driver.cu_mem_alloc_v2)(&mut dptr, bytes);
            if res != CUDA_SUCCESS {
                return Err(format!("cuMemAlloc failed (bytes: {}) with code: {}", bytes, res));
            }
        }
        Ok(Self {
            driver: Arc::clone(&ctx.driver),
            dptr,
            bytes,
        })
    }

    pub fn copy_from_host(&self, host_data: &[f32]) -> Result<(), String> {
        let req_bytes = host_data.len() * std::mem::size_of::<f32>();
        if req_bytes > self.bytes {
            return Err("Host data exceeds buffer capacity".into());
        }
        unsafe {
            let res = (self.driver.cu_memcpy_htod_v2)(
                self.dptr,
                host_data.as_ptr() as *const c_void,
                req_bytes,
            );
            if res != CUDA_SUCCESS {
                return Err(format!("cuMemcpyHtoD failed with code: {}", res));
            }
        }
        Ok(())
    }

    pub fn copy_to_host(&self, host_data: &mut [f32]) -> Result<(), String> {
        let req_bytes = host_data.len() * std::mem::size_of::<f32>();
        if req_bytes > self.bytes {
            return Err("Host slice too small for device buffer".into());
        }
        unsafe {
            let res = (self.driver.cu_memcpy_dtoh_v2)(
                host_data.as_mut_ptr() as *mut c_void,
                self.dptr,
                req_bytes,
            );
            if res != CUDA_SUCCESS {
                return Err(format!("cuMemcpyDtoH failed with code: {}", res));
            }
        }
        Ok(())
    }
}

impl Drop for CudaDeviceBuffer {
    fn drop(&mut self) {
        if self.dptr != 0 {
            unsafe {
                (self.driver.cu_mem_free_v2)(self.dptr);
            }
            self.dptr = 0;
        }
    }
}

/// Loaded CUDA Module containing GPU kernel functions
pub struct CudaModule {
    pub driver: Arc<CudaDriver>,
    pub module: CUmodule,
}

impl CudaModule {
    pub fn load_from_ptx(driver: Arc<CudaDriver>, ptx_bytes: &[u8]) -> Result<Self, String> {
        let mut module: CUmodule = std::ptr::null_mut();
        unsafe {
            let res = (driver.cu_module_load_data)(
                &mut module,
                ptx_bytes.as_ptr() as *const c_void,
            );
            if res != CUDA_SUCCESS {
                return Err(format!("cuModuleLoadData failed with code: {}", res));
            }
        }
        Ok(Self { driver, module })
    }

    pub fn get_function(&self, name: &str) -> Result<CUfunction, String> {
        let c_name = CString::new(name).map_err(|e| e.to_string())?;
        let mut func: CUfunction = std::ptr::null_mut();
        unsafe {
            let res = (self.driver.cu_module_get_function)(&mut func, self.module, c_name.as_ptr());
            if res != CUDA_SUCCESS {
                return Err(format!("cuModuleGetFunction failed for '{}' with code: {}", name, res));
            }
        }
        Ok(func)
    }

    pub fn launch(
        &self,
        func: CUfunction,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: u32,
        mut params: Vec<*mut c_void>,
    ) -> Result<(), String> {
        unsafe {
            let res = (self.driver.cu_launch_kernel)(
                func,
                grid.0, grid.1, grid.2,
                block.0, block.1, block.2,
                shared_mem,
                std::ptr::null_mut(),
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            );
            if res != CUDA_SUCCESS {
                return Err(format!("cuLaunchKernel failed with code: {}", res));
            }
        }
        Ok(())
    }
}
