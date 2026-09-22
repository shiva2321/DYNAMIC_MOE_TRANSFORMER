import unittest
import torch
import math
from ishmn.geometry import (
    project_to_sphere,
    project_to_tangent_space,
    exp_map_sphere,
    log_map_sphere,
    geodesic_distance_sphere,
    spherical_to_cartesian,
    cartesian_to_spherical,
    parallel_transport_sphere,
    analogical_transfer_sphere,
)

class TestGeometry(unittest.TestCase):
    def test_project_to_sphere(self):
        x = torch.randn(10, 8)
        u = project_to_sphere(x)
        norms = torch.norm(u, p=2, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_tangent_space(self):
        u = project_to_sphere(torch.randn(10, 8))
        v = torch.randn(10, 8)
        v_tangent = project_to_tangent_space(u, v)
        inner = torch.sum(u * v_tangent, dim=-1)
        self.assertTrue(torch.allclose(inner, torch.zeros_like(inner), atol=1e-5))

    def test_exp_map_and_geodesic(self):
        u = project_to_sphere(torch.randn(10, 8))
        v = torch.randn(10, 8) * 0.5  # Tangent displacement
        u_next = exp_map_sphere(u, v)
        
        # Check u_next is on sphere
        norms = torch.norm(u_next, p=2, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))
        
        # Geodesic distance should equal length of tangent vector ||v_tangent|| (for ||v|| < pi)
        v_tangent = project_to_tangent_space(u, v)
        expected_dist = torch.norm(v_tangent, p=2, dim=-1)
        actual_dist = geodesic_distance_sphere(u, u_next).squeeze(-1)
        self.assertTrue(torch.allclose(expected_dist, actual_dist, atol=1e-4))

    def test_spherical_cartesian_roundtrip(self):
        x = torch.randn(5, 6) * 10.0
        u, r = cartesian_to_spherical(x)
        x_recon = spherical_to_cartesian(u, r)
        self.assertTrue(torch.allclose(x, x_recon, atol=1e-5))

    def test_parallel_transport(self):
        u = project_to_sphere(torch.randn(8, 8))
        w = project_to_sphere(torch.randn(8, 8))
        v = project_to_tangent_space(u, torch.randn(8, 8))
        
        v_trans = parallel_transport_sphere(u, w, v)
        # 1. Transported vector must be in tangent space of w
        inner_w = torch.sum(w * v_trans, dim=-1)
        self.assertTrue(torch.allclose(inner_w, torch.zeros_like(inner_w), atol=1e-4))
        
        # 2. Parallel transport is an isometry: ||P(v)|| == ||v||
        norm_v = torch.norm(v, dim=-1)
        norm_trans = torch.norm(v_trans, dim=-1)
        self.assertTrue(torch.allclose(norm_v, norm_trans, atol=1e-4))

    def test_analogical_transfer(self):
        # A, B, C are concepts on sphere
        a = project_to_sphere(torch.randn(4, 8))
        b = project_to_sphere(torch.randn(4, 8))
        c = project_to_sphere(torch.randn(4, 8))
        
        # Solve A : B :: C : D
        d = analogical_transfer_sphere(a, b, c)
        
        # Check D is on unit sphere
        norm_d = torch.norm(d, dim=-1)
        self.assertTrue(torch.allclose(norm_d, torch.ones_like(norm_d), atol=1e-5))
        
        # Geodesic distance d(C, D) must equal d(A, B) because parallel transport preserves length
        dist_ab = geodesic_distance_sphere(a, b)
        dist_cd = geodesic_distance_sphere(c, d)
        self.assertTrue(torch.allclose(dist_ab, dist_cd, atol=1e-4))

if __name__ == '__main__':
    unittest.main()
