"""
High-Performance Python / PyTorch Bridge to the Rust Yggdrasil GPU Engine.
Supports Zero-Copy GPU execution directly on NVIDIA Tensor/CUDA cores via raw device pointers,
with multi-threaded SIMD CPU fallback via Rayon.
"""

import ctypes
import math
import os
import warnings
import torch
from pathlib import Path
from typing import Optional, Tuple


def _find_engine_dll() -> str:
    """Locates the compiled yggdrasil_engine.dll binary."""
    base_dir = Path(__file__).resolve().parent.parent
    possible_paths = [
        base_dir / "rust_yggdrasil" / "target" / "release" / "yggdrasil_engine.dll",
        base_dir / "rust_yggdrasil" / "target" / "debug" / "yggdrasil_engine.dll",
    ]
    for p in possible_paths:
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        "Could not find yggdrasil_engine.dll. Please run 'cargo build --release' in rust_yggdrasil."
    )


def _retract_torch(p: torch.Tensor, g: torch.Tensor, lr: float, c: float, eps: float) -> torch.Tensor:
    """PyTorch implementation of the Riemannian retraction, identical in form to the CUDA kernel (closed form in
    |p|^2, |g|^2 and <p, g>; any dim_h). Used for CPU tensors."""
    p_sq = (p * p).sum(-1, keepdim=True)
    g_sq = (g * g).sum(-1, keepdim=True)
    pg = (p * g).sum(-1, keepdim=True)
    factor = (1.0 - c * p_sq).clamp(min=1e-7)
    inv_metric = factor * factor * 0.25
    v_norm = abs(lr) * inv_metric * g_sq.sqrt()
    safe = v_norm.clamp(min=1e-10)
    sqrt_c = math.sqrt(c)
    scale_v = torch.tanh(0.5 * sqrt_c * (2.0 / factor) * safe) / (sqrt_c * safe)
    kappa = -lr * inv_metric * scale_v
    dot, w_sq = kappa * pg, kappa * kappa * g_sq
    alpha = 1.0 + 2.0 * c * dot + c * w_sq
    beta = 1.0 - c * p_sq
    denom = (1.0 + 2.0 * c * dot + c * c * p_sq * w_sq).clamp(min=1e-10)
    out = (alpha * p + beta * kappa * g) / denom
    max_r = (1.0 - eps) / sqrt_c
    norm = out.norm(dim=-1, keepdim=True)
    out = torch.where(norm > max_r, out * (max_r / norm.clamp(min=1e-18)), out)
    return torch.where(v_norm < 1e-10, p, out)


class YggdrasilEngineBridge:
    """Singleton wrapper around yggdrasil_engine.dll C-ABI exports."""

    _instance: Optional["YggdrasilEngineBridge"] = None

    def __init__(self):
        dll_path = _find_engine_dll()
        self.lib = ctypes.CDLL(dll_path)
        self._setup_signatures()

    @classmethod
    def get_instance(cls) -> "YggdrasilEngineBridge":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _setup_signatures(self):
        # Error handling
        self.lib.yggdrasil_get_last_error.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        self.lib.yggdrasil_get_last_error.restype = ctypes.c_int

        # GPU lifecycle
        self.lib.yggdrasil_gpu_init.argtypes = [ctypes.c_int]
        self.lib.yggdrasil_gpu_init.restype = ctypes.c_void_p

        self.lib.yggdrasil_gpu_destroy.argtypes = [ctypes.c_void_p]
        self.lib.yggdrasil_gpu_destroy.restype = None

        # GPU kernel execution (u64 represents 64-bit CUdeviceptr)
        self.lib.yggdrasil_gpu_pairwise_distance.argtypes = [
            ctypes.c_void_p,  # engine
            ctypes.c_uint64,  # a_dptr
            ctypes.c_uint64,  # b_dptr
            ctypes.c_uint64,  # out_dptr
            ctypes.c_size_t,  # n
            ctypes.c_size_t,  # m
            ctypes.c_uint32,  # dim_h
            ctypes.c_uint32,  # dim_t
            ctypes.c_float,   # c
            ctypes.c_float,   # eps
            ctypes.c_float,   # weight_h
            ctypes.c_float,   # weight_t
            ctypes.c_float,   # weight_r
        ]
        self.lib.yggdrasil_gpu_pairwise_distance.restype = ctypes.c_int

        self.lib.yggdrasil_gpu_pairwise_distance_fast.argtypes = [
            ctypes.c_void_p,  # engine
            ctypes.c_uint64,  # a_dptr
            ctypes.c_uint64,  # b_dptr
            ctypes.c_uint64,  # out_dptr
            ctypes.c_size_t,  # n
            ctypes.c_size_t,  # m
            ctypes.c_uint32,  # dim_h
            ctypes.c_uint32,  # dim_t
            ctypes.c_float,   # c
            ctypes.c_float,   # eps
            ctypes.c_float,   # weight_h
            ctypes.c_float,   # weight_t
            ctypes.c_float,   # weight_r
        ]
        self.lib.yggdrasil_gpu_pairwise_distance_fast.restype = ctypes.c_int

        self.lib.yggdrasil_gpu_mobius_add.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_float,
        ]
        self.lib.yggdrasil_gpu_mobius_add.restype = ctypes.c_int

        self.lib.yggdrasil_gpu_project.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_float,
            ctypes.c_float,
        ]
        self.lib.yggdrasil_gpu_project.restype = ctypes.c_int

        self.lib.yggdrasil_gpu_torus_bind.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_size_t,
        ]
        self.lib.yggdrasil_gpu_torus_bind.restype = ctypes.c_int

        self.lib.yggdrasil_gpu_torus_unbind.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_size_t,
        ]
        self.lib.yggdrasil_gpu_torus_unbind.restype = ctypes.c_int

        self.lib.yggdrasil_gpu_riemannian_retract.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
        ]
        self.lib.yggdrasil_gpu_riemannian_retract.restype = ctypes.c_int

        # CPU FFI execution
        self.lib.yggdrasil_cpu_pairwise_distance.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
        ]
        self.lib.yggdrasil_cpu_pairwise_distance.restype = ctypes.c_int

        self.lib.yggdrasil_cpu_mobius_add.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_float,
        ]
        self.lib.yggdrasil_cpu_mobius_add.restype = ctypes.c_int

        self.lib.yggdrasil_cpu_project.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_float,
            ctypes.c_float,
        ]
        self.lib.yggdrasil_cpu_project.restype = ctypes.c_int

        self.lib.yggdrasil_cpu_torus_bind.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        self.lib.yggdrasil_cpu_torus_bind.restype = ctypes.c_int

    def get_last_error(self) -> str:
        buf = ctypes.create_string_buffer(1024)
        self.lib.yggdrasil_get_last_error(buf, 1024)
        return buf.value.decode("utf-8", errors="replace")


class YggdrasilKernel:
    """High-level Python API for GPU/CPU acceleration of the Yggdrasil Manifold."""

    def __init__(self, device_ordinal: int = 0, require_gpu: bool = False):
        self.bridge = YggdrasilEngineBridge.get_instance()
        self.device_ordinal = device_ordinal
        self._gpu_engine = None

        if torch.cuda.is_available():
            self._gpu_engine = self.bridge.lib.yggdrasil_gpu_init(device_ordinal)
            if not self._gpu_engine:
                err = self.bridge.get_last_error()
                if require_gpu:
                    raise RuntimeError(f"Yggdrasil GPU engine failed to initialise: {err}")
                warnings.warn(f"Yggdrasil GPU engine failed to initialise ({err}); running the CPU path instead. "
                              "Pass require_gpu=True to make this an error.", RuntimeWarning, stacklevel=2)

    def __del__(self):
        if self._gpu_engine:
            self.bridge.lib.yggdrasil_gpu_destroy(self._gpu_engine)
            self._gpu_engine = None

    def pairwise_distance(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        dim_h: int,
        dim_t: int,
        c: float = 1.0,
        eps: float = 1e-5,
        weight_h: float = 1.0,
        weight_t: float = 1.0,
        weight_r: float = 1.0,
    ) -> torch.Tensor:
        """
        Computes pairwise distance matrix between sets of states A and B.
        A: (N, D), B: (M, D) where D = dim_h + dim_t + 1
        Returns: (N, M)
        """
        assert A.shape[-1] == dim_h + dim_t + 1, "Dimension mismatch for A"
        assert B.shape[-1] == dim_h + dim_t + 1, "Dimension mismatch for B"
        assert A.device == B.device, "Tensors must be on the same device"

        N = A.shape[0]
        M = B.shape[0]
        A_cont = A.contiguous().float()
        B_cont = B.contiguous().float()
        out = torch.empty((N, M), device=A.device, dtype=torch.float32)

        if A.is_cuda and self._gpu_engine:
            res = self.bridge.lib.yggdrasil_gpu_pairwise_distance(
                self._gpu_engine,
                ctypes.c_uint64(A_cont.data_ptr()),
                ctypes.c_uint64(B_cont.data_ptr()),
                ctypes.c_uint64(out.data_ptr()),
                N,
                M,
                dim_h,
                dim_t,
                c,
                eps,
                weight_h,
                weight_t,
                weight_r,
            )
            if res != 0:
                raise RuntimeError(f"GPU pairwise_distance failed: {self.bridge.get_last_error()}")
        else:
            # CPU Execution
            A_cpu = A_cont.cpu()
            B_cpu = B_cont.cpu()
            out_cpu = torch.empty((N, M), dtype=torch.float32)
            res = self.bridge.lib.yggdrasil_cpu_pairwise_distance(
                A_cpu.data_ptr(),
                B_cpu.data_ptr(),
                out_cpu.data_ptr(),
                N,
                M,
                dim_h,
                dim_t,
                c,
                eps,
                weight_h,
                weight_t,
                weight_r,
            )
            if res != 0:
                raise RuntimeError(f"CPU pairwise_distance failed: {self.bridge.get_last_error()}")
            out = out_cpu.to(A.device)

        return out

    def pairwise_distance_fast(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        dim_h: int,
        dim_t: int,
        c: float = 1.0,
        eps: float = 1e-5,
        weight_h: float = 1.0,
        weight_t: float = 1.0,
        weight_r: float = 1.0,
    ) -> torch.Tensor:
        """
        High-throughput 64x64 tiled pairwise distance kernel on the Yggdrasil manifold.
        A: (N, D), B: (M, D) where D = dim_h + dim_t + 1.
        Returns: (N, M).
        """
        assert A.shape[-1] == dim_h + dim_t + 1, "Dimension mismatch for A"
        assert B.shape[-1] == dim_h + dim_t + 1, "Dimension mismatch for B"
        assert A.device == B.device, "Tensors must be on the same device"

        N = A.shape[0]
        M = B.shape[0]
        A_cont = A.contiguous().float()
        B_cont = B.contiguous().float()
        out = torch.empty((N, M), device=A.device, dtype=torch.float32)

        if A.is_cuda and self._gpu_engine:
            res = self.bridge.lib.yggdrasil_gpu_pairwise_distance_fast(
                self._gpu_engine,
                ctypes.c_uint64(A_cont.data_ptr()),
                ctypes.c_uint64(B_cont.data_ptr()),
                ctypes.c_uint64(out.data_ptr()),
                N,
                M,
                dim_h,
                dim_t,
                c,
                eps,
                weight_h,
                weight_t,
                weight_r,
            )
            if res != 0:
                raise RuntimeError(f"GPU pairwise_distance_fast failed: {self.bridge.get_last_error()}")
        else:
            return self.pairwise_distance(A, B, dim_h, dim_t, c, eps, weight_h, weight_t, weight_r)

        return out

    def pairwise_distance_gemm_epilogue(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        dim_h: int,
        dim_t: int,
        c: float = 1.0,
        eps: float = 1e-5,
        weight_h: float = 1.0,
        weight_t: float = 1.0,
        weight_r: float = 1.0,
        torus_mode: str = "wrapped",
        use_cuda_kernel: bool = True,
    ) -> torch.Tensor:
        """
        High-throughput GEMM + epilogue pairwise distance on the Yggdrasil manifold.
        Uses compiled CUDA 64x64 tiled kernel when available, or PyTorch reference.
        A: (N, D), B: (M, D) where D = dim_h + dim_t + 1.
        Returns: (N, M).
        """
        assert A.shape[-1] == dim_h + dim_t + 1, "Dimension mismatch for A"
        assert B.shape[-1] == dim_h + dim_t + 1, "Dimension mismatch for B"
        assert A.device == B.device, "Tensors must be on the same device"

        if use_cuda_kernel and A.is_cuda and self._gpu_engine and torus_mode == "wrapped":
            return self.pairwise_distance_fast(
                A, B, dim_h, dim_t, c=c, eps=eps,
                weight_h=weight_h, weight_t=weight_t, weight_r=weight_r
            )

        N, M = A.shape[0], B.shape[0]
        # 1. Hyperbolic part via GEMM
        Ah = A[:, :dim_h]
        Bh = B[:, :dim_h]
        u_sq = (Ah * Ah).sum(-1, keepdim=True)  # (N, 1)
        v_sq = (Bh * Bh).sum(-1, keepdim=True).t()  # (1, M)
        dot_h = torch.matmul(Ah, Bh.t())  # (N, M) GEMM
        diff_sq_h = (u_sq + v_sq - 2.0 * dot_h).clamp(min=0.0)

        denom_u = (1.0 - c * u_sq).clamp(min=1e-7)
        denom_v = (1.0 - c * v_sq).clamp(min=1e-7)
        xi = (c * diff_sq_h) / (denom_u * denom_v)
        safe_xi = xi.clamp(min=0.0)
        d_h = torch.where(
            diff_sq_h > 1e-14,
            (2.0 / math.sqrt(c)) * torch.asinh(torch.sqrt(safe_xi)),
            torch.zeros_like(safe_xi),
        )

        # 2. Torus part
        if dim_t == 0:
            d_t_sq = torch.zeros((N, M), device=A.device, dtype=torch.float32)
        elif torus_mode == "smooth":
            At = A[:, dim_h : dim_h + dim_t]
            Bt = B[:, dim_h : dim_h + dim_t]
            cA = torch.cat([torch.cos(At), torch.sin(At)], dim=-1)
            cB = torch.cat([torch.cos(Bt), torch.sin(Bt)], dim=-1)
            dot_t = torch.matmul(cA, cB.t())  # (N, M) GEMM
            d_t_sq = (2.0 * dim_t - 2.0 * dot_t).clamp(min=0.0)
        else:
            # Wrapped difference (chunked to ensure O(N*M) peak memory)
            At = A[:, dim_h : dim_h + dim_t]
            Bt = B[:, dim_h : dim_h + dim_t]
            TWO_PI = 2.0 * math.pi
            INV_TWO_PI = 1.0 / TWO_PI
            d_t_sq = torch.empty((N, M), device=A.device, dtype=torch.float32)
            chunk_size = 512
            for i in range(0, N, chunk_size):
                end = min(i + chunk_size, N)
                diff = At[i:end, None, :] - Bt[None, :, :]
                diff = diff - TWO_PI * torch.round(diff * INV_TWO_PI)
                d_t_sq[i:end] = (diff * diff).sum(dim=-1)

        # 3. Radial part
        r_idx = dim_h + dim_t
        ra = A[:, r_idx].clamp(min=eps)
        rb = B[:, r_idx].clamp(min=eps)
        d_r = (torch.log(ra).unsqueeze(1) - torch.log(rb).unsqueeze(0)).abs()

        # 4. Composite metric
        comp_sq = weight_h * (d_h * d_h) + weight_t * d_t_sq + weight_r * (d_r * d_r)
        return torch.sqrt(comp_sq.clamp(min=0.0))

    def mobius_add(self, U: torch.Tensor, V: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        """Batch Mobius addition: U (+) V on the Poincare Ball."""
        assert U.shape == V.shape, "Shape mismatch in mobius_add"
        assert U.device == V.device, "Device mismatch in mobius_add"

        N = U.shape[0]
        dim_h = U.shape[1]
        U_cont = U.contiguous().float()
        V_cont = V.contiguous().float()
        out = torch.empty_like(U_cont)

        if U.is_cuda and self._gpu_engine:
            res = self.bridge.lib.yggdrasil_gpu_mobius_add(
                self._gpu_engine,
                ctypes.c_uint64(U_cont.data_ptr()),
                ctypes.c_uint64(V_cont.data_ptr()),
                ctypes.c_uint64(out.data_ptr()),
                N,
                dim_h,
                c,
            )
            if res != 0:
                raise RuntimeError(f"GPU mobius_add failed: {self.bridge.get_last_error()}")
        else:
            U_cpu = U_cont.cpu()
            V_cpu = V_cont.cpu()
            out_cpu = torch.empty_like(U_cpu)
            res = self.bridge.lib.yggdrasil_cpu_mobius_add(
                U_cpu.data_ptr(),
                V_cpu.data_ptr(),
                out_cpu.data_ptr(),
                N,
                dim_h,
                c,
            )
            if res != 0:
                raise RuntimeError(f"CPU mobius_add failed: {self.bridge.get_last_error()}")
            out = out_cpu.to(U.device)

        return out

    def project(
        self,
        points: torch.Tensor,
        dim_h: int,
        dim_t: int,
        c: float = 1.0,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        """Projects arbitrary vectors into valid Yggdrasil Manifold bounds."""
        N = points.shape[0]
        p_cont = points.contiguous().float().clone()

        if points.is_cuda and self._gpu_engine:
            res = self.bridge.lib.yggdrasil_gpu_project(
                self._gpu_engine,
                ctypes.c_uint64(p_cont.data_ptr()),
                N,
                dim_h,
                dim_t,
                c,
                eps,
            )
            if res != 0:
                raise RuntimeError(f"GPU project failed: {self.bridge.get_last_error()}")
        else:
            p_cpu = p_cont.cpu()
            res = self.bridge.lib.yggdrasil_cpu_project(
                p_cpu.data_ptr(),
                N,
                dim_h,
                dim_t,
                c,
                eps,
            )
            if res != 0:
                raise RuntimeError(f"CPU project failed: {self.bridge.get_last_error()}")
            p_cont = p_cpu.to(points.device)

        return p_cont

    def torus_bind(self, U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Lie group addition modulo 2*PI on the Clifford Torus."""
        assert U.shape == V.shape
        total_elements = U.numel()
        U_cont = U.contiguous().float()
        V_cont = V.contiguous().float()
        out = torch.empty_like(U_cont)

        if U.is_cuda and self._gpu_engine:
            res = self.bridge.lib.yggdrasil_gpu_torus_bind(
                self._gpu_engine,
                ctypes.c_uint64(U_cont.data_ptr()),
                ctypes.c_uint64(V_cont.data_ptr()),
                ctypes.c_uint64(out.data_ptr()),
                total_elements,
            )
            if res != 0:
                raise RuntimeError(f"GPU torus_bind failed: {self.bridge.get_last_error()}")
        else:
            U_cpu = U_cont.cpu()
            V_cpu = V_cont.cpu()
            out_cpu = torch.empty_like(U_cpu)
            res = self.bridge.lib.yggdrasil_cpu_torus_bind(
                U_cpu.data_ptr(),
                V_cpu.data_ptr(),
                out_cpu.data_ptr(),
                U_cpu.shape[0],
                U_cpu.shape[1],
            )
            if res != 0:
                raise RuntimeError(f"CPU torus_bind failed: {self.bridge.get_last_error()}")
            out = out_cpu.to(U.device)

        return out

    def torus_unbind(self, U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Lie group subtraction modulo 2*PI on the Clifford Torus."""
        assert U.shape == V.shape
        total_elements = U.numel()
        U_cont = U.contiguous().float()
        V_cont = V.contiguous().float()
        out = torch.empty_like(U_cont)

        if U.is_cuda and self._gpu_engine:
            res = self.bridge.lib.yggdrasil_gpu_torus_unbind(
                self._gpu_engine,
                ctypes.c_uint64(U_cont.data_ptr()),
                ctypes.c_uint64(V_cont.data_ptr()),
                ctypes.c_uint64(out.data_ptr()),
                total_elements,
            )
            if res != 0:
                raise RuntimeError(f"GPU torus_unbind failed: {self.bridge.get_last_error()}")
        else:
            # CPU fallback via direct modular arithmetic
            diff = (U_cont - V_cont) % (2.0 * 3.141592653589793)
            diff = torch.where(diff >= 3.141592653589793, diff - 2.0 * 3.141592653589793, diff)
            out = diff

        return out

    def riemannian_retract(
        self,
        points: torch.Tensor,
        grads: torch.Tensor,
        lr: float,
        c: float = 1.0,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        """Riemannian gradient retraction step on GPU/CPU."""
        assert points.shape == grads.shape
        N = points.shape[0]
        dim_h = points.shape[1]
        p_cont = points.contiguous().float()
        g_cont = grads.contiguous().float()
        out = torch.empty_like(p_cont)

        if points.is_cuda and self._gpu_engine:
            res = self.bridge.lib.yggdrasil_gpu_riemannian_retract(
                self._gpu_engine,
                ctypes.c_uint64(p_cont.data_ptr()),
                ctypes.c_uint64(g_cont.data_ptr()),
                ctypes.c_uint64(out.data_ptr()),
                N,
                dim_h,
                lr,
                c,
                eps,
            )
            if res != 0:
                raise RuntimeError(f"GPU riemannian_retract failed: {self.bridge.get_last_error()}")
        else:
            out = _retract_torch(p_cont.cpu(), g_cont.cpu(), lr, c, eps).to(points.device)

        return out
