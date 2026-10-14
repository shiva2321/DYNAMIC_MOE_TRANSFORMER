import ctypes
import os

torch_lib = r"C:\Users\Asta\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\lib"
os.add_dll_directory(torch_lib)
os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
nvrtc_path = os.path.join(torch_lib, "nvrtc64_120_0.dll")
nvrtc = ctypes.WinDLL(nvrtc_path)
cuda = ctypes.WinDLL("nvcuda.dll")

# Init CUDA Driver
assert cuda.cuInit(0) == 0, "cuInit failed"
device = ctypes.c_int()
assert cuda.cuDeviceGet(ctypes.byref(device), 0) == 0, "cuDeviceGet failed"
ctx = ctypes.c_void_p()
assert cuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), device) == 0, "cuDevicePrimaryCtxRetain failed"
assert cuda.cuCtxSetCurrent(ctx) == 0, "cuCtxSetCurrent failed"

# Compile CUDA C to PTX with NVRTC
src = b"""
extern "C" __global__ void saxpy(float a, const float *x, const float *y, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a * x[i] + y[i];
    }
}
"""

prog = ctypes.c_void_p()
res = nvrtc.nvrtcCreateProgram(ctypes.byref(prog), src, b"saxpy.cu", 0, None, None)
assert res == 0, f"nvrtcCreateProgram failed: {res}"

opts = [b"--gpu-architecture=compute_86"]
c_opts = (ctypes.c_char_p * len(opts))(*opts)
res = nvrtc.nvrtcCompileProgram(prog, len(opts), c_opts)
if res != 0:
    log_size = ctypes.c_size_t()
    nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(log_size))
    log = ctypes.create_string_buffer(log_size.value)
    nvrtc.nvrtcGetProgramLog(prog, log)
    print("Compile error:", log.value.decode("utf-8"))
assert res == 0, f"nvrtcCompileProgram failed: {res}"

ptx_size = ctypes.c_size_t()
nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(ptx_size))
ptx = ctypes.create_string_buffer(ptx_size.value)
nvrtc.nvrtcGetPTX(prog, ptx)
print("PTX successfully compiled! Size:", ptx_size.value, "bytes")

# Load PTX module into CUDA driver
module = ctypes.c_void_p()
res = cuda.cuModuleLoadData(ctypes.byref(module), ptx.raw)
assert res == 0, f"cuModuleLoadData failed: {res}"

kernel = ctypes.c_void_p()
res = cuda.cuModuleGetFunction(ctypes.byref(kernel), module, b"saxpy")
assert res == 0, f"cuModuleGetFunction failed: {res}"

print("Kernel function acquired successfully from CUDA driver!")
