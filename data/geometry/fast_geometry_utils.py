import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

class DepthToNormals(nn.Module):
    def __init__(
        self,
        H: int,
        W: int,
        fx: float = 320.0,
        fy: float = 320.0,
        cx: float = 320.0,
        cy: float = 240.0,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()
        self.device = device
        
        cx = cx if cx is not None else (W - 1) / 2
        cy = cy if cy is not None else (H - 1) / 2

        u = torch.arange(W, device=device).view(1, W).expand(H, W)
        v = torch.arange(H, device=device).view(H, 1).expand(H, W)
        rays = torch.stack(((u - cx) / fx, (v - cy) / fy, torch.ones_like(u)), dim=0).to(self.device)
        self.register_buffer("pixel_rays", rays, persistent=False)

        kx = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=device
        )
        ky = kx.t()
        self.register_buffer("kernels", torch.stack([kx, ky]), persistent=False)

    @staticmethod
    def _forward_impl(depth, pixel_rays, kernels):
        pts3d = depth.unsqueeze(0) * pixel_rays  # (3,H,W)

        b, c, h, w = 1, 3, *depth.shape
        spatial_pad = [
            kernels.size(1) // 2,
            kernels.size(1) // 2,
            kernels.size(2) // 2,
            kernels.size(2) // 2,
        ]
        x = F.pad(pts3d.unsqueeze(0).reshape(b * c, 1, h, w), spatial_pad, "replicate")
        grad = F.conv2d(x, kernels[:, None]).reshape(b, c, 2, h, w)
        dzdx, dzdy = grad[:, :, 0], grad[:, :, 1]
        normal = torch.cross(dzdx, dzdy, dim=1)
        normal = F.normalize(normal, dim=1).squeeze(0).permute(1, 2, 0)  # (H,W,3)
        return normal, pts3d

    def forward(self, depth: torch.Tensor):
        norm, pts = self._forward_impl(depth, self.pixel_rays, self.kernels)
        return norm, pts


def depth_to_normals(depth, fx, fy, cx, cy):
    in_dim = depth.dim()
    
    if in_dim == 2:
        depth = depth[None]
        
    B, H, W = depth.shape
    device = depth.device

    def expand_param(x):
        x = torch.as_tensor(x, device=device)
        if x.ndim == 0:
            x = x.expand(B)  # No memory duplication, just broadcasting
        assert x.shape == (B,)
        return x.view(B, 1, 1)

    fx = expand_param(fx)
    fy = expand_param(fy)
    cx = expand_param(cx)
    cy = expand_param(cy)

    u = torch.arange(W, device=device).view(1, 1, W).expand(B, H, W)
    v = torch.arange(H, device=device).view(1, H, 1).expand(B, H, W)
    ones = torch.ones_like(u)

    rays = torch.stack([(u - cx) / fx, (v - cy) / fy, ones], dim=1)  # (B, 3, H, W)
    pts3d = depth.unsqueeze(1) * rays  # (B, 3, H, W)

    kx = torch.tensor([[-1.0, 0.0, 1.0],
                       [-2.0, 0.0, 2.0],
                       [-1.0, 0.0, 1.0]], device=device)
    ky = kx.t()
    kernels = torch.stack([kx, ky])  # (2, 3, 3)

    x = F.pad(pts3d.view(B * 3, 1, H, W), [1, 1, 1, 1], mode="replicate")
    grad = F.conv2d(x, kernels[:, None]).view(B, 3, 2, H, W)
    dzdx, dzdy = grad[:, :, 0], grad[:, :, 1]

    normal = torch.cross(dzdx, dzdy, dim=1)
    normal = F.normalize(normal, dim=1)

    normals_out = normal.permute(0, 2, 3, 1)  # (B, H, W, 3)
    pts3d_out = pts3d                        # (B, 3, H, W)

    if in_dim == 2:
        return normals_out[0], pts3d_out[0]
    return normals_out, pts3d_out
