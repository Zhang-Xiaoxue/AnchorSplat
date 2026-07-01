# coding=utf-8
# Adapted from
# https://github.com/nerfstudio-project/gsplat/blob/main/gsplat/rendering.py
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from typing import Dict, Optional, Tuple
from typing_extensions import Literal
import math
import torch
from torch import Tensor

from meta_gauss_render import spherical_harmonics, flash_gaussian_build_mask, gaussian_sort, calc_render, get_render_schedule_cpp
from depth_anything_3.model.utils.projection_three_dims_gaussian_fused import projection_three_dims_gaussian_fused
import acl


def assert_check(condition, message):
    if not condition:
        raise ValueError(message)


def validate_inputs(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    colors: Tensor,
    viewmats: Tensor,
    Ks: Tensor,
    render_mode: str,
    sh_degree: Optional[int],
    N: int,
    C: int
) -> None:

    assert_check(means.shape == (N, 3), f"Invalid shape for means: {means.shape}")
    assert_check(quats.shape == (N, 4), f"Invalid shape for quats: {quats.shape}")
    assert_check(scales.shape == (N, 3), f"Invalid shape for scales: {scales.shape}")
    assert_check(opacities.shape == (N,), f"Invalid shape for opacities: {opacities.shape}")
    assert_check(viewmats.shape == (C, 4, 4), f"Invalid shape for viewmats: {viewmats.shape}")
    assert_check(Ks.shape == (C, 3, 3), f"Invalid shape for Ks: {Ks.shape}")
    assert_check(render_mode in ["RGB", "D", "ED", "RGB+D", "RGB+ED"], f"Invalid render_mode: {render_mode}")

    if sh_degree is None:
        # treat colors as post-activation values, should be in shape [ N, D] or [C, N, D]
        assert_check((colors.dim() == 2 and colors.shape[0] == N) or
                    (colors.dim() == 3 and colors.shape[:2] == (C, N)),
                    f"Invalid shape for colors: {colors.shape}")
    else:
        # treat colors as SH coefficients, should be in shape [N, K, 3] or [C, N, K, 3]
        # Allowing for activating partial SH bands
        assert_check((colors.dim() == 3 and colors.shape[0] == N and colors.shape[2] == 3) or
                    (colors.dim() == 4 and colors.shape[:2] == (C, N) and colors.shape[3] == 3),
                    f"Invalid shape for colors: {colors.shape}")

        assert_check((sh_degree + 1) ** 2 <= colors.shape[-2], f"Invalid sh_degree for colors shape: {colors.shape}")


class Rasterizer:
    def tile2image(self, rendered_image, height, width, channel_dim=3):
        rendered_image = rendered_image.reshape(math.ceil(self.padded_height/self.tile_size),
                                                math.ceil(self.padded_width/self.tile_size), self.tile_size, self.tile_size, -1)
        rendered_image = rendered_image.permute(0, 2, 1, 3, 4)
        rendered_image = rendered_image.reshape(math.ceil(self.padded_height/self.tile_size)*self.tile_size,
                                                math.ceil(self.padded_width/self.tile_size)*self.tile_size, -1)
        return rendered_image.permute(2, 0, 1)[:, :height, :width]

    def get_render_input(self, means2d, colors, opacities, conics, depths, tile_offsets, _cam_view):
        cf_means2 = means2d[0, _cam_view]
        cf_colors3 = colors[0, _cam_view]
        cf_opacity = opacities[0, _cam_view]

        inv_x_0 = conics[0, _cam_view, 0, :]
        inv_x_1 = conics[0, _cam_view, 1, :]
        inv_x_2 = conics[0, _cam_view, 2, :]

        cf_depths = depths[_cam_view]
        tile_size = self.tile_size
        pix_coords = self.pix_coord.reshape(self.padded_height//tile_size, tile_size, self.padded_width//tile_size, tile_size, 2) \
            .permute(0, 2, 1, 3, 4).reshape(self.padded_height//tile_size*self.padded_width//tile_size, tile_size*tile_size, 2) \
            .permute(0, 2, 1).to(torch.float32).contiguous()
        # nums: 每个tile对应的gs数量
        nums = torch.cat([tile_offsets[_cam_view][:1], tile_offsets[_cam_view][1:] - tile_offsets[_cam_view][:-1]])
        # lb_sched：cat[每个vector core要处理的tile数目的cumsum，依次对应的tile id，依次对应的tile offset]
        lb_sched = get_render_schedule_cpp(nums.cpu().to(torch.int64), acl.get_device_capability(colors.device.index, 1)[0]).clone().detach().to(colors.device, torch.int64)

        return (cf_means2, cf_colors3, cf_opacity, inv_x_0, inv_x_1, inv_x_2, cf_depths, pix_coords, lb_sched)

    def ascend_rasterize_splats(
        self,
        w2c: Tensor,
        Knorm: Tensor,
        width: int,
        height: int,
        tile_size: int,
        splats: dict,
        active_sh_degree: int,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        if not hasattr(self, "tile_grid"):
            self.tile_size = tile_size
            self.padded_width = math.ceil(width/tile_size)*tile_size
            self.padded_height = math.ceil(height/tile_size)*tile_size
            self.tile_grid = torch.stack(torch.meshgrid(torch.arange(0, self.padded_height, tile_size), \
                                                        torch.arange(0, self.padded_width, tile_size), indexing='ij'), dim=-1).view(-1, 2).to(splats["mean"].device)
            self.pix_coord = torch.stack(torch.meshgrid(torch.arange(self.padded_width), torch.arange(self.padded_height), indexing='xy'), dim=-1).to(splats["mean"].device)

        means = splats["mean"]  # splats["means"]  # [N, 3]
        quats = splats["rotation"]  # splats["quats"]  # [N, 4]
        scales = splats["scale"]  # splats["scales"]  # torch.exp(splats["scales"])  # [N, 3]
        opacities = splats["opacity"].flatten()  # splats["opacities"]  # torch.sigmoid(splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        colors = splats["color"].unflatten(-1, (-1, 3))  # torch.cat([splats["sh0"], splats["shN"]], 1)  # [N, K, 3]

        rasterize_mode = "classic"
        Ks = Knorm * Knorm.new_tensor((width, height, 1))[:, None]
        # render_colors, render_depth, info = self._ascend_rasterization(
        render_colors, render_depth = self._ascend_rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=w2c,  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            tile_size=tile_size,
            rasterize_mode=rasterize_mode,
            sh_degree=active_sh_degree,
            camera_model="pinhole",
            **kwargs,
        )
        # return render_colors, render_depth, info
        return render_colors, render_depth

    def inverse_cov2d_v2(self, cov2_00, cov2_01, cov2_11, scale=1):
        det = cov2_00 * cov2_11 - cov2_01 * cov2_01
        inv_x_0 = cov2_11 / det * scale
        inv_x_1 = -cov2_01 / det * scale
        inv_x_2 = cov2_00 / det * scale
        return inv_x_0, inv_x_1, inv_x_2

    def _ascend_rasterization(
        self,
        means: Tensor,  # [..., N, 3]
        quats: Tensor,  # [..., N, 4]
        scales: Tensor,  # [..., N, 3]
        opacities: Tensor,  # [..., N]
        colors: Tensor,  # [..., (C,) N, D] or [..., (C,) N, K, 3]
        viewmats: Tensor,  # [..., C, 4, 4]
        Ks: Tensor,  # [..., C, 3, 3]
        width: int,
        height: int,
        near_plane: float = 0.01,
        far_plane: float = 1e10,
        eps2d: float = 0.3,
        sh_degree: Optional[int] = None,
        tile_size: int = 64,
        render_mode: Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"] = "RGB",
        rasterize_mode: Literal["classic", "antialiased"] = "classic",
        camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole",
    ) -> Tuple[Tensor, Tensor, Dict]:
        """A version of rasterization() that utilizes on PyTorch's autograd.
        Rasterize a set of 3D Gaussians (N) to a batch of image planes (C).

        .. note::
            This function relies on gsplat's CUDA backend for some computation, but the
            entire differentiable graph is built with PyTorch (and nerfacc), so
            back-propagation is handled by PyTorch's autograd.

        ..note::
            Compared to rasterization(), this function does not support some arguments such as
            `packed`, `sparse_grad` and `absgrad`.
        """

        N = means.shape[0]
        C = viewmats.shape[0]
        B = 1
        validate_inputs(means, quats, scales, opacities, colors, viewmats, Ks, render_mode, sh_degree, N, C)

        # # Colors are SH coefficients, with shape [N, K, 3] or [C, N, K, 3]
        # camtoworlds = torch.inverse(viewmats) # [C, 4, 4]
        # if colors.dim() == 3:
        #     # Turn [N, K, 3] into [C, N, K, 3]
        #     shs = colors.expand(C, -1, -1, -1)
        # else:
        #     # colors is already [C, N, K, 3]
        #     shs = colors

        # # build colors
        # rays_o = camtoworlds[0, :3, 3]
        # rays_d = torch.nn.functional.normalize(means - rays_o, dim=-1, eps=.05)
        # k = (sh_degree+1)**2
        # colors = spherical_harmonics(sh_degree, rays_d.reshape(B, N, 3), shs[0, :, :k, :].reshape(B, N, k, 3))
        # colors = (colors+0.5).clip(min=0.0)
        colors = colors.movedim(0, 2).mul(.28209479177387814).add(.5).clamp(0)

        # ascend gauss projection
        means2d, depths, conics, opacities, radius, covars2d, colors, cnt = projection_three_dims_gaussian_fused(
            means.reshape(B, N, 3),
            colors,
            None,
            quats.reshape(B, N, 4),
            scales.reshape(B, N, 3),
            opacities.reshape(B, N),
            viewmats.reshape(B, C, 4, 4).contiguous(),
            Ks.reshape(B, C, 3, 3),
            width,
            height,
            0.3,
            0.05
        )
        camera_ids, gaussian_ids = None, None

        # ascend gauss sort
        with torch.no_grad():
            mask = flash_gaussian_build_mask(means2d, opacities[None, :], conics, covars2d, cnt[None, :], self.tile_grid.float(), width, height, tile_size)
            sorted_gs_ids = []
            tile_offsets = []
            for _cam_view in range(0, C):
                cf_sorted_gs_ids, cf_tile_offsets = gaussian_sort(mask[0, _cam_view], depths[0, _cam_view])
                sorted_gs_ids.append(cf_sorted_gs_ids)
                tile_offsets.append(cf_tile_offsets)

        render_colors = []
        render_depths = []
        for _cam_view in range(0, C):
            input = self.get_render_input(means2d, colors, opacities, conics, depths, tile_offsets, _cam_view)
            cf_means2, cf_colors3, cf_opacity, inv_x_0, inv_x_1, inv_x_2, cf_depths, pix_coords, lb_sched = input

            # ascend rasterize to pixels
            cf_render_colors, cf_render_depths = calc_render(cf_means2,
                                                            inv_x_0, inv_x_1, inv_x_2,
                                                            cf_opacity,
                                                            cf_colors3,
                                                            cf_depths,
                                                            pix_coords,
                                                            lb_sched,
                                                            sorted_gs_ids[_cam_view]
                                                            )
            cf_render_colors = self.tile2image(cf_render_colors.permute(1, 2, 0), height, width)
            cf_render_depths = self.tile2image(cf_render_depths.permute(1, 2, 0), height, width)

            render_colors.append(cf_render_colors)
            render_depths.append(cf_render_depths)
        render_colors = torch.stack(render_colors)
        render_depths = torch.stack(render_depths)

        # meta = {
        #     "gaussian_ids": gaussian_ids,
        #     "means2d": means2d,
        #     "radii": radius,
        #     "width": width,
        #     "height": height,
        #     "n_cameras": C,
        # }
        # return render_colors, render_depths, meta
        return render_colors, render_depths
