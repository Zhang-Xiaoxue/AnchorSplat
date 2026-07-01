#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from easydict import EasyDict as edict


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)  # 0
    P[1, 2] = (top + bottom) / (top - bottom)  # 0
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def get_projection_matrix_from_K(K, width, height, znear=0.01, zfar=100.0):
    """
    Convert camera intrinsics (K) to 4x4 OpenGL-style projection matrix.
    Args:
        K: [3, 3] intrinsic matrix
        width, height: image dimensions
        znear, zfar: near and far clipping planes
    Returns:
        [4, 4] projection matrix (torch.Tensor)
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    fovx = 2 * math.atan(0.5 * width / fx)
    fovy = 2 * math.atan(0.5 * height / fy)

    tan_fovx = math.tan(0.5 * fovx) * 1.3
    tan_fovy = math.tan(0.5 * fovy) * 1.3

    proj = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0,1)

    return proj, tan_fovx, tan_fovy


def get_camera(intr, w2c, width, height, device='npu:0'):
    """convert intrinsics and extrinsics matrices to viewpoint camera

    Args:
        intrs (torch.Tensor): intrisics, already times with img width and hieght [3,3]
        w2cs (torch.Tensor): w2c matrices [4,4]
        width (int): render img width
        height (int): render img height
        device (str, optional): _description_. Defaults to 'npu:0'.
    """
    viewpoint_cam = edict()
    viewpoint_cam.world_view_transform = w2c.to(device).T

    viewpoint_cam.camera_center = torch.inverse(viewpoint_cam.world_view_transform.float())[:3,3].to(device)
    viewpoint_cam.image_width = width
    viewpoint_cam.image_height = height

    focal_length_x = intr[0,0]
    focal_length_y = intr[1,1]
    # FovY = focal2fov(focal_length_y, height)
    # FovX = focal2fov(focal_length_x, width)

    proj, tan_fovx, tan_fovy = get_projection_matrix_from_K(intr, width, height, znear=0.01, zfar=1e10)
    viewpoint_cam.projection_matrix = proj.to(device)

    return viewpoint_cam, tan_fovx, tan_fovy, focal_length_x, focal_length_y


def render_metagauss_perview(gauss_render,
    gaussian: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    resolution: tuple,
    output_depth: bool = True
):
    """
    Args:
        gauss_render (AscendGaussRender): _description_
        gausssian (torch.Tensor): _description_
        extrinsics (torch.Tensor): (normalized) extrinsics
        intrinsics (torch.Tensor): normalized intrinsics
        resolution (tuple): [H, W]
        render_mode (str): _description_
        sh_degree (int, optional): _description_. Defaults to 0.

    Raises:
        TypeError: _description_

    Returns:
        _type_: _description_
    """
    height, width = resolution
    intrinsics = intrinsics * intrinsics.new_tensor((width, height, 1))[:, None]

    # if len(extrinsics.shape) == 4 and len(intrinsics.shape) == 4:
    #     extrinsics = extrinsics.squeeze(0)
    #     intrinsics = intrinsics.squeeze(0)
    # elif len(extrinsics.shape) == 2 and len(intrinsics.shape) == 2:
    #     extrinsics = extrinsics.unsqueeze(0)
    #     intrinsics = intrinsics.unsqueeze(0)
    # else:
    #     raise TypeError

    # assert extrinsics.shape[0] > 1 or intrinsics.shape[0] > 1

    viewpoint_cam, tan_fovx, tan_fovy, focal_x, focal_y = get_camera(intrinsics[0], extrinsics[0], width, height, device=gaussian.device)

    render_pkg = gauss_render(
        viewpoint_cam, gaussian,
        tan_fovx, tan_fovy, focal_x, focal_y,
        output_depth=output_depth
    )
    return render_pkg