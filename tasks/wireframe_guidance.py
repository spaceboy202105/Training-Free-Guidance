import torch
import torch.nn as nn
from typing import List, Union

try:
    from .utils import check_grad_fn, rescale_grad
except Exception:
    def check_grad_fn(x_need_grad):
        assert x_need_grad.requires_grad, "x_need_grad should require grad"

    def rescale_grad(grad: torch.Tensor, clip_scale, **kwargs):
        node_mask = kwargs.get('node_mask', None)
        scale = (grad ** 2).mean(dim=-1)
        if node_mask is not None:
            scale = scale * node_mask.float()
            denom = node_mask.float().sum(dim=-1).clamp_min(1.0)
            scale = scale.sum(dim=-1) / denom
            clipped_scale = torch.clamp(scale, max=clip_scale)
            coef = clipped_scale / (scale + 1e-12)
            grad = grad * coef.view(-1, 1, 1)
        return grad

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
        # NOTE: keep public API; delegate to vectorized implementation
        return self.compute_intersection_energy_vectorized(
            x,
            loop_mask=loop_mask,
            point_mask=point_mask,
            return_per_sample=False,
        )

    def compute_intersection_energy_vectorized(
        self,
        x: torch.Tensor,
        loop_mask=None,
        point_mask=None,
        return_per_sample: bool = False,
    ) -> torch.Tensor:
        """
        Vectorized TPE energy over all valid loops (removes Python loops over batch/loops).

        x: [B, L, P, 3]
        loop_mask: [B, L] bool
        point_mask: [B, L, P] bool
        """
        if loop_mask is None:
            loop_mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        if point_mask is None:
            point_mask = torch.ones(x.shape[:3], dtype=torch.bool, device=x.device)

        eps = self.tpe_eps
        beta = self.tpe_beta
        eta = self.tpe_eta

        B, L, P, _ = x.shape

        # Select valid (b,l) loops
        bl = loop_mask.view(-1)  # [B*L]
        if bl.sum() == 0:
            if return_per_sample:
                return x.new_zeros((B,))
            return x.new_zeros(())

        flat_indices = torch.nonzero(bl, as_tuple=False).squeeze(-1)  # [N]
        sample_ids = torch.div(flat_indices, L, rounding_mode='floor')  # [N]

        X = x.view(B * L, P, 3)[bl]                 # [N, P, 3]
        PM = point_mask.view(B * L, P)[bl]          # [N, P]

        # If too few valid points in a loop, it contributes 0
        valid_counts = PM.sum(dim=1)                # [N]
        valid_loop = valid_counts >= 4
        if valid_loop.sum() == 0:
            if return_per_sample:
                return x.new_zeros((B,))
            return x.new_zeros(())

        X = X[valid_loop]
        PM = PM[valid_loop]
        sample_ids = sample_ids[valid_loop]

        # Zero-out invalid points (keeps tensor dense for vectorization)
        X = X * PM.unsqueeze(-1).to(X.dtype)

        # Edges: i -> i+1 (cyclic)
        X_next = torch.roll(X, shifts=-1, dims=1)   # [N, P, 3]
        edge_vec = X_next - X                       # [N, P, 3]
        edge_len = torch.linalg.norm(edge_vec, dim=-1).clamp_min(eps)  # [N, P]
        edge_tangent = edge_vec / edge_len.unsqueeze(-1)               # [N, P, 3]

        # Valid edge if both endpoints valid
        edge_valid = PM & torch.roll(PM, shifts=-1, dims=1)            # [N, P]
        edge_len = edge_len * edge_valid.to(edge_len.dtype)           # mask out invalid edges

        # Build pairwise segment endpoint diffs:
        # endpoints_i: [N, P, 2, 3] with [start, end]
        endpoints_i = torch.stack([X, X_next], dim=2)                  # [N, P, 2, 3]
        endpoints_j = endpoints_i

        # d: [N, P, P, 2, 2, 3] (broadcast)
        d = endpoints_i[:, :, None, :, None, :] - endpoints_j[:, None, :, None, :, :]

        dist = torch.linalg.norm(d, dim=-1).clamp_min(eps)             # [N, P, P, 2, 2]

        tangent = edge_tangent[:, :, None, None, None, :]              # [N, P, 1, 1, 1, 3]
        cross = torch.cross(tangent.expand_as(d), d, dim=-1)
        cross_norm = torch.linalg.norm(cross, dim=-1).clamp_min(eps)   # [N, P, P, 2, 2]

        kernel = (cross_norm ** beta) / (dist ** eta)                  # [N, P, P, 2, 2]
        kernel_avg = kernel.mean(dim=(-1, -2))                         # [N, P, P]

        # Adjacent mask (exclude i==j and neighbors, cyclic)
        idx = torch.arange(P, device=x.device)
        diff = (idx[:, None] - idx[None, :]).abs()                     # [P, P]
        adjacent = (diff == 0) | (diff == 1) | (diff == P - 1)         # [P, P]
        pair_mask = (~adjacent).to(kernel_avg.dtype)                   # [P, P]

        # Also exclude pairs where either edge is invalid
        edge_pair_valid = (edge_valid.to(kernel_avg.dtype)[:, :, None] *
                           edge_valid.to(kernel_avg.dtype)[:, None, :])  # [N, P, P]

        length_weight = edge_len[:, :, None] * edge_len[:, None, :]    # [N, P, P]

        loop_energy = (kernel_avg * length_weight * edge_pair_valid * pair_mask).sum(dim=(1, 2))
        energy_per_sample = x.new_zeros((B,))
        energy_per_sample.scatter_add_(0, sample_ids, loop_energy)

        if return_per_sample:
            return energy_per_sample
        return energy_per_sample.sum()

    @torch.enable_grad()
    def get_guidance(self, x_need_grad, func=lambda x:x, post_process=lambda x:x, return_logp=False, check_grad=True, **kwargs):
        """
        Compute the guidance gradient based on intersection energy.
        """

        if check_grad:
            check_grad_fn(x_need_grad)

        # Apply any necessary transformations (e.g., VAE decoding)
        x_pnt,loop_mask,point_mask = post_process(func(x_need_grad))

        # Calculate energy (loss)
        # We want to minimize energy, so we maximize negative energy (log_prob = -energy)
        energy = self.compute_intersection_energy_vectorized(
            x_pnt,
            loop_mask=loop_mask,
            point_mask=point_mask,
            return_per_sample=True,
        )
        log_probs = -energy

        if return_logp:
            return log_probs

        # Calculate gradient of the objective w.r.t x_need_grad
        # We want to move x in the direction that maximizes log_probs (minimizes energy)
        grad_l2 = torch.autograd.grad(log_probs.sum(), x_need_grad)[0]

        point_mask_for_grad = kwargs.get("point_mask", point_mask)
        if point_mask_for_grad is not None and point_mask_for_grad.shape == grad_l2.shape[:-1]:
            grad_l2 = grad_l2 * point_mask_for_grad.unsqueeze(-1).float()

            fixed_mask = self._build_fixed_vertex_mask(point_mask_for_grad)
            if fixed_mask is not None:
                grad_l2 = grad_l2.masked_fill(fixed_mask.unsqueeze(-1), 0.0)

        if self.use_sobolev_gradient:                                                                                                                                                                                                        
              # 选项 A: 完整分数阶 (根据论文)                   
            grad_final = grad_l2              
                                                                                                                                                                                             
            #   A_matrix = self.build_fractional_sobolev_matrix(x)
            #   grad_final = torch.linalg.solve(A_matrix, grad_l2)
        elif self.use_laplacian_smoothing:
            grad_final = grad_l2              
              # 选项 B: 简单 Laplacian 平滑 (H1)
            #   L_matrix = self.get_laplacian(x)
            #   grad_final = torch.linalg.solve(torch.eye(N) + self.lambda * L_matrix, grad_l2)
        else:
              grad_final = grad_l2              
        # Rescale gradient (standard practice in this codebase)
        return rescale_grad(grad_final, clip_scale=1.0, **kwargs)
