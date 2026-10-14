"""Rebuild src/gpu_kernels.ptx from src/gpu_kernels.cu with NVRTC, the CUDA runtime compiler that ships inside
PyTorch's CUDA wheels. No CUDA toolkit or nvcc is needed.

    python rust_yggdrasil/build_ptx.py            # compute_86 (RTX 30xx); pass --arch compute_89 etc. for other GPUs
    cd rust_yggdrasil && cargo build --release    # the PTX is embedded into the DLL with include_bytes!
"""

import argparse
import ctypes
import glob
import os
import sys
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="compute_86", help="CUDA target architecture, or comma-separated architectures")
    ap.add_argument("--multi-arch", action="store_true", help="Compile for multiple standard architectures (75, 80, 86, 89)")
    args = ap.parse_args()
    here = Path(__file__).resolve().parent
    src = (here / "src" / "gpu_kernels.cu").read_bytes()
    lib_dir = Path(torch.__file__).resolve().parent / "lib"
    cands = sorted(glob.glob(str(lib_dir / "nvrtc64_*.dll")) + glob.glob(str(lib_dir / "libnvrtc.so*")))
    if not cands:
        print("no NVRTC library found next to torch; install a CUDA build of PyTorch or use nvcc -ptx", file=sys.stderr)
        return 1
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(lib_dir))
    nvrtc = ctypes.CDLL(cands[0])
    prog = ctypes.c_void_p()
    assert nvrtc.nvrtcCreateProgram(ctypes.byref(prog), src, b"gpu_kernels.cu", 0, None, None) == 0

    if args.multi_arch:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            primary_arch = f"compute_{major}{minor}"
        else:
            primary_arch = "compute_75"
    else:
        primary_arch = args.arch
    opts = [f"--gpu-architecture={primary_arch}".encode()]
    c_opts = (ctypes.c_char_p * len(opts))(*opts)
    res = nvrtc.nvrtcCompileProgram(prog, len(opts), c_opts)
    size = ctypes.c_size_t()
    nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(size))
    log = ctypes.create_string_buffer(size.value)
    nvrtc.nvrtcGetProgramLog(prog, log)
    if size.value > 1:
        print(log.value.decode(errors="replace"))
    if res != 0:
        print(f"NVRTC compilation failed with code {res}", file=sys.stderr)
        return 1
    nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(size))
    ptx = ctypes.create_string_buffer(size.value)
    nvrtc.nvrtcGetPTX(prog, ptx)

    # cuModuleLoadData reads the PTX as a C string: the NUL terminator NVRTC appends MUST stay in the file
    # (without it the driver reads past the end and fails with CUDA_ERROR_INVALID_PTX, code 218)
    nul_byte = ptx.raw[size.value - 1]
    assert nul_byte == 0 or nul_byte == b"\x00"[0], f"Expected NUL terminator at offset {size.value - 1}, found {nul_byte}"
    print(f"verified NUL terminator at byte offset {size.value - 1} (byte={nul_byte})")

    (here / "src" / "gpu_kernels.ptx").write_bytes(ptx.raw[: size.value])
    print(f"wrote src/gpu_kernels.ptx ({size.value} bytes) for {primary_arch} with {Path(cands[0]).name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

