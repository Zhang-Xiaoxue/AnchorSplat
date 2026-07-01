from typing_extensions import Literal

import torch, torch_npu
from torch.autograd import Function
from torch.nn import Module
from torch import Tensor
import torch.nn.functional as F

import meta_gauss_render._C

class ProjectionThreeDimsGaussianFused(Function):
    @staticmethod
    def forward(ctx,
                means: torch.Tensor,
                colors: torch.Tensor,
                covars: torch.Tensor = None,
                quat: torch.Tensor = None,
                scales: torch.Tensor = None,
                opacities: torch.Tensor = None,
                viewmats: torch.Tensor = None,
                Ks: torch.Tensor = None,
                width: int = 0,
                height: int = 0,
                eps: float = 0.3,
                near_plane: float = 0.01,
                far_plane: float = 1e10,
                calc_compensations: bool = False,
                camera_model: str = "pinhole"):

        if quat is not None:
            if scales is None:
                raise ValueError("'quat' and 'scales' are required together.")
            if covars is not None:
                raise ValueError("Invalid parameter combination: 'covars' and ('quat', 'scales')pair are mutually exclusive.")
            quat = quat.permute((0, 2, 1)).contiguous()
            scales = scales.permute((0, 2, 1)).contiguous()
            covars = meta_gauss_render._C.quat_scales_to_covars(quat, scales)
        else:
            covars = covars.permute((0, 2, 3, 1)).contiguous()

        means = means.permute((0, 2, 1)).contiguous()

        means2d, depths, conics, compensations, det, radius, covars2d = meta_gauss_render._C.projection_three_dims_gaussian_forward(
                means,
                covars,
                opacities,
                viewmats,
                Ks,
                width,
                height,
                eps,
                calc_compensations,
                camera_model
        )
        
        det = det.squeeze(-2)
        depths = depths.squeeze(-2)
        if calc_compensations:
            compensations = compensations.squeeze(-2)
        else:
            compensations = None

        means_culling, colors_culling, means2d_culling, depths_culling, radius_culling, covars2d_culling, conics_culling, opacities_culling, filter, cnt = meta_gauss_render._C.gaussian_filter(
                means,
                colors,
                det,
                opacities,
                means2d,
                depths,
                radius,
                conics,
                covars2d,
                compensations,
                width,
                height,
                near_plane,
                far_plane)

        ctx.save_for_backward(means, conics, viewmats, quat, scales, Ks, filter, compensations)
        ctx.width = width
        ctx.height = height
        return means2d_culling, depths_culling, conics_culling, opacities_culling, radius_culling, covars2d_culling, colors_culling, cnt

    @staticmethod
    def backward(
        ctx, *v_args
    ):
        means, conics, viewmats, quats, scales, Ks, filter, compensations = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        v_means2d, v_depths, v_conics, v_opacities_culling, v_radii, v_covars2d, v_colors_culling, v_cnt = v_args
        v_pW, v_quats, v_scales, v_R, v_colors, v_opacities = meta_gauss_render._C.fully_fused_projection_bwd(
                means,
                quats,
                scales,
                conics,
                viewmats,
                Ks,
                v_means2d,
                v_depths,
                v_conics,
                v_colors_culling,
                v_opacities_culling,
                filter,
                compensations,
                width,
                height
        )

        # return v_pW, v_colors, None, v_quats, v_scales, v_opacities, \
        #         None, None, None, None, None, None, None, None, None
        # mask = (filter[0, ..., None] & filter.new_tensor((1, 2, 4, 8, 16, 32, 64, 128))).flatten(-2).bool()[:, :v_pW.shape[1]]
        # v_pW.mul_(mask[..., None])
        # v_colors.mul_(mask[:, None])
        # v_quats.mul_(mask[..., None])
        # v_scales.mul_(mask[..., None])
        # v_opacities.mul_(mask)
        return v_pW.nan_to_num_(), v_colors.nan_to_num_(), None, v_quats.nan_to_num_(), v_scales.nan_to_num_(), v_opacities.nan_to_num_(), \
            None, None, None, None, None, None, None, None, None

projection_three_dims_gaussian_fused = ProjectionThreeDimsGaussianFused.apply