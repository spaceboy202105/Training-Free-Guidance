import torch
import torch.nn as nn
from typing import List, Union
from .utils import check_grad_fn, rescale_grad

class TangentPointEnergyGuidance:
    """
    Guidance class for avoiding self-intersection in WireFrame generation using TPE energy.
    """

    def __init__(self, args):
        self.args = args
        self.device = args.device

        # Initialize any necessary parameters or models here
        # For example, if TPE energy calculation requires specific parameters:
        # self.tpe_param = args.tpe_param
        self.num_samples_per_edge = getattr(args, "num_samples_per_edge", 32)
        self.tpe_beta = getattr(args, "tpe_beta", 2.0)
        self.tpe_eta = getattr(args, "tpe_eta", 4.0)
        self.tpe_eps = getattr(args, "tpe_eps", 1e-4)
        self.use_sobolev_gradient = getattr(args, "use_sobolev_gradient", False)
        self.use_laplacian_smoothing = getattr(args, "use_laplacian_smoothing", False)

    def _build_fixed_vertex_mask(self, point_mask: torch.Tensor) -> torch.Tensor:
        """
        Build a mask for fixed spline vertices (endpoints of each spline segment).

        Args:
            point_mask: [B, L, P] valid point mask

        Returns:
            fixed_mask: [B, L, P] True for fixed vertices
        """
        if point_mask is None:
            return None

        bsz, max_loops, max_points = point_mask.shape
        device = point_mask.device

        point_indices = torch.arange(max_points, device=device)
        is_vertex = (point_indices % self.num_samples_per_edge == 0) | (
            point_indices % self.num_samples_per_edge == self.num_samples_per_edge - 1
        )
        is_vertex = is_vertex.view(1, 1, -1).expand(bsz, max_loops, max_points)
        return is_vertex & point_mask

    def compute_intersection_energy(self, x, loop_mask=None, point_mask=None):
        """
        Compute the self-intersection energy (TPE) for the wireframe x.

        Args:
            x: Tensor [batch_size, max_loops, max_points, 3]
            loop_mask: Tensor [batch_size, max_loops] (Bool)
            point_mask: Tensor [batch_size, max_loops, max_points] (Bool)

        Returns:
            energy: A scalar tensor representing the intersection penalty.
                   Higher energy means more intersections.
        """
        if loop_mask is None:
            loop_mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        if point_mask is None:
            point_mask = torch.ones(x.shape[:3], dtype=torch.bool, device=x.device)
        eps = self.tpe_eps
        beta = self.tpe_beta
        eta = self.tpe_eta

        total_energy = x.new_zeros(())
        batch_size, max_loops, max_points, _ = x.shape

        for b in range(batch_size):
            for l in range(max_loops):
                if not loop_mask[b, l]:
                    continue

                valid_mask = point_mask[b, l]
                if valid_mask.sum() < 4:
                    continue

                points = x[b, l][valid_mask]
                num_points = points.shape[0]
                if num_points < 4:
                    continue

                points_next = torch.roll(points, shifts=-1, dims=0)
                edge_vec = points_next - points
                edge_len = torch.linalg.norm(edge_vec, dim=-1).clamp_min(eps)
                edge_tangent = edge_vec / edge_len.unsqueeze(-1)

                if num_points < 4:
                    continue

                endpoints_i = torch.stack([points, points_next], dim=1)  # [E, 2, 3]
                endpoints_j = endpoints_i

                d = endpoints_i[:, None, :, None, :] - endpoints_j[None, :, None, :, :]
                dist = torch.linalg.norm(d, dim=-1).clamp_min(eps)

                tangent = edge_tangent[:, None, None, None, :]
                cross = torch.cross(tangent.expand_as(d), d, dim=-1)
                cross_norm = torch.linalg.norm(cross, dim=-1).clamp_min(eps)

                kernel = (cross_norm ** beta) / (dist ** eta)
                kernel_avg = kernel.mean(dim=(-1, -2))

                idx = torch.arange(num_points, device=x.device)
                diff = (idx[:, None] - idx[None, :]).abs()
                adjacent = (diff == 0) | (diff == 1) | (diff == num_points - 1)

                pair_mask = ~adjacent
                length_weight = edge_len[:, None] * edge_len[None, :]
                energy = (kernel_avg * length_weight * pair_mask).sum()
                total_energy = total_energy + energy

        return total_energy

    @torch.enable_grad()
    def get_guidance(self, x_need_grad, func=lambda x:x, post_process=lambda x:x, return_logp=False, check_grad=True, **kwargs):
        """
        Compute the guidance gradient based on intersection energy.
        """

        if check_grad:
            check_grad_fn(x_need_grad)

        # Apply any necessary transformations (e.g., VAE decoding)
        x = post_process(func(x_need_grad))

        # Calculate energy (loss)
        # We want to minimize energy, so we maximize negative energy (log_prob = -energy)
        energy = self.compute_intersection_energy(
            x,
            loop_mask=kwargs.get("loop_mask", None),
            point_mask=kwargs.get("point_mask", None),
        )
        log_probs = -energy

        if return_logp:
            return log_probs

        # Calculate gradient of the objective w.r.t x_need_grad
        # We want to move x in the direction that maximizes log_probs (minimizes energy)
        grad_l2 = torch.autograd.grad(log_probs.sum(), x_need_grad)[0]

        point_mask = kwargs.get("point_mask", None)
        if point_mask is not None:
            grad_l2 = grad_l2 * point_mask.unsqueeze(-1).float()

        fixed_mask = self._build_fixed_vertex_mask(point_mask)
        if fixed_mask is not None:
            grad_l2 = grad_l2.masked_fill(fixed_mask.unsqueeze(-1), 0.0)

        if self.use_sobolev_gradient:                                                                                                                                                                                                        
              # 选项 A: 完整分数阶 (根据论文)                                                                                                                                                                                                  
              A_matrix = self.build_fractional_sobolev_matrix(x)
              grad_final = torch.linalg.solve(A_matrix, grad_l2)
        elif self.use_laplacian_smoothing:
              # 选项 B: 简单 Laplacian 平滑 (H1)
              L_matrix = self.get_laplacian(x)
              grad_final = torch.linalg.solve(torch.eye(N) + self.lambda * L_matrix, grad_l2)
        else:
              grad_final = grad_l2              
        # Rescale gradient (standard practice in this codebase)
        return rescale_grad(grad_final, clip_scale=1.0, **kwargs)
