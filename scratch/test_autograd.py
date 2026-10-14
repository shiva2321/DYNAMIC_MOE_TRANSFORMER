import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from geometry.gpu_kernel import YggdrasilKernel

class YggdrasilDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, dim_h, dim_t, c, eps, w_h, w_t, w_r, kernel=None):
        if kernel is None:
            kernel = YggdrasilKernel()
        
        # Kernel execution (GPU accelerated on RTX 3060)
        dist = kernel.pairwise_distance(A, B, dim_h, dim_t, c, eps, w_h, w_t, w_r)
        
        ctx.save_for_backward(A, B, dist)
        ctx.dim_h = dim_h
        ctx.dim_t = dim_t
        ctx.c = c
        ctx.eps = eps
        ctx.w_h = w_h
        ctx.w_t = w_t
        ctx.w_r = w_r
        return dist

    @staticmethod
    def backward(ctx, grad_output):
        A, B, dist = ctx.saved_tensors
        dim_h = ctx.dim_h
        dim_t = ctx.dim_t
        c = ctx.c
        w_h = ctx.w_h
        w_t = ctx.w_t
        w_r = ctx.w_r

        N, D = A.shape
        M, _ = B.shape

        # Expand for pairwise operations: A_exp (N, 1, D), B_exp (1, M, D)
        A_exp = A.unsqueeze(1)
        B_exp = B.unsqueeze(0)

        # 1. Hyperbolic Backward
        u = A_exp[..., :dim_h]
        v = B_exp[..., :dim_h]
        diff_u_v = u - v
        diff_sq = (diff_u_v ** 2).sum(dim=-1, keepdim=True)
        u_sq = (u ** 2).sum(dim=-1, keepdim=True)
        v_sq = (v ** 2).sum(dim=-1, keepdim=True)

        alpha_u = torch.clamp(1.0 - c * u_sq, min=1e-7)
        alpha_v = torch.clamp(1.0 - c * v_sq, min=1e-7)
        xi = (c * diff_sq) / (alpha_u * alpha_v)
        xi = torch.clamp(xi, min=0.0)

        # Safe derivative of d_H^2 wrt xi
        # d_H = 2/sqrt(c) * arsinh(sqrt(xi))
        sqrt_xi = torch.sqrt(xi + 1e-18)
        d_h = (2.0 / (c ** 0.5)) * torch.asinh(sqrt_xi)
        # derivative factor: (2 * d_h) / (sqrt(c) * sqrt(xi * (xi + 1)))
        denom_xi = torch.clamp(torch.sqrt((xi + 1e-14) * (xi + 1.0)), min=1e-7)
        dd_dxi = torch.where(xi < 1e-10, torch.full_like(xi, 4.0 / c), (2.0 * d_h) / ((c ** 0.5) * denom_xi))

        # dxi / du = 2c/(alpha_u * alpha_v) * [ (u - v) + (c * diff_sq / alpha_u) * u ]
        dxi_du = (2.0 * c / (alpha_u * alpha_v)) * (diff_u_v + (c * diff_sq / alpha_u) * u)
        dxi_dv = (2.0 * c / (alpha_u * alpha_v)) * (-diff_u_v + (c * diff_sq / alpha_v) * v)

        grad_d2_u = w_h * dd_dxi * dxi_du
        grad_d2_v = w_h * dd_dxi * dxi_dv

        # 2. Torus Backward
        t_u = A_exp[..., dim_h:dim_h+dim_t]
        t_v = B_exp[..., dim_h:dim_h+dim_t]
        delta = (t_u - t_v) % (2.0 * 3.141592653589793)
        delta = torch.where(delta >= 3.141592653589793, delta - 2.0 * 3.141592653589793, delta)
        delta = torch.where(delta < -3.141592653589793, delta + 2.0 * 3.141592653589793, delta)
        grad_d2_tu = w_t * 2.0 * delta
        grad_d2_tv = -w_t * 2.0 * delta

        # 3. Radial Backward
        r_u = A_exp[..., -1:]
        r_v = B_exp[..., -1:]
        diff_r = r_u - r_v
        grad_d2_ru = w_r * 2.0 * diff_r
        grad_d2_rv = -w_r * 2.0 * diff_r

        # Assemble composite gradient: d(D^2) / dA and d(D^2) / dB
        grad_d2_A = torch.cat([grad_d2_u, grad_d2_tu, grad_d2_ru], dim=-1)
        grad_d2_B = torch.cat([grad_d2_v, grad_d2_tv, grad_d2_rv], dim=-1)

        # Scale by d(D) / d(D^2) = 1 / (2 * D)
        inv_2dist = 1.0 / (2.0 * torch.clamp(dist.unsqueeze(-1), min=1e-7))
        grad_d_A = grad_d2_A * inv_2dist
        grad_d_B = grad_d2_B * inv_2dist

        # Multiply by grad_output and sum over respective batch dimensions
        grad_out_exp = grad_output.unsqueeze(-1)
        grad_A = (grad_out_exp * grad_d_A).sum(dim=1)
        grad_B = (grad_out_exp * grad_d_B).sum(dim=0)

        return grad_A, grad_B, None, None, None, None, None, None, None, None

def test_autograd_function():
    print("Testing YggdrasilDistanceFunction autograd backward pass...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kernel = YggdrasilKernel(0)

    dim_h, dim_t = 4, 4
    total_dim = dim_h + dim_t + 1
    N, M = 4, 3

    # Small random coordinates strictly within interior
    torch.manual_seed(123)
    A = torch.randn(N, total_dim, device=device, dtype=torch.float64) * 0.1
    B = torch.randn(M, total_dim, device=device, dtype=torch.float64) * 0.1
    A[:, -1] = A[:, -1].abs() + 0.5
    B[:, -1] = B[:, -1].abs() + 0.5

    A.requires_grad_(True)
    B.requires_grad_(True)

    # Convert to float32 for kernel call inside custom function, test gradcheck
    # Note: gradcheck requires float64 for high-precision finite differences
    def func(a_in, b_in):
        a_f32 = a_in.float()
        b_f32 = b_in.float()
        dist = YggdrasilDistanceFunction.apply(a_f32, b_f32, dim_h, dim_t, 1.0, 1e-5, 1.0, 1.0, 1.0, kernel)
        return dist.double()

    test_passed = torch.autograd.gradcheck(func, (A, B), eps=1e-4, atol=1e-3, rtol=1e-3)
    print("Gradcheck result:", test_passed)
    assert test_passed, "Autograd gradient check failed!"
    print("YggdrasilDistanceFunction backward pass matches numerical gradients perfectly!")

if __name__ == "__main__":
    test_autograd_function()
