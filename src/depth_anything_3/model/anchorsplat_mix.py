from addict import Dict
from omegaconf import OmegaConf
from torch import accelerator, autocast, inference_mode, no_grad, nn, Tensor, tensor
from torchvision.transforms.v2 import Normalize

from depth_anything_3.api import DepthAnything3
from depth_anything_3.cfg import create_object
from depth_anything_3.model.da3 import DepthAnything3Net
from depth_anything_3.model.feat_cnn import FeatureCNN
from depth_anything_3.model.point_sampler import VoxelSampler
from depth_anything_3.model.point_net import PointBackbone, Refiner
from depth_anything_3.model.gs_head import BaseGaussianHead
from depth_anything_3.model.renderer import Renderer
from depth_anything_3.utils.pose_align import align_poses_umeyama


@autocast(accelerator.current_accelerator().type, enabled=False)
def _align_to_input_extrinsics(
    extrinsics: Tensor,
    prediction: Dict[str, Tensor],
    align_to_input_ext_scale: bool = True,
    ransac_view_thresh: int = 10
) -> Dict[str, Tensor]:
    """Align depth map to input extrinsics"""
    _, _, scale, _ = align_poses_umeyama(
        prediction.extrinsics.squeeze(0).detach().float().cpu().numpy(),
        extrinsics.squeeze(0).float().cpu().numpy(),
        ransac=len(extrinsics) >= ransac_view_thresh,
        return_aligned=True,
        random_state=42,
    )
    if align_to_input_ext_scale: prediction.depth = prediction.depth / scale
    return prediction


class AnchorSplatNet(nn.Module):
    '''
    AnchorSplat network for anchor-aligned GS prediction based on multi-view RGBD inputs.

    This network consists of:
    - 2D backbone: lightweight CNN
    - Anchor sampler: voxel downsampling or TSDF fusion
    - 3D backbone: Point Transformer
    - GS head: anchor-aligned GS prediction
    - Refiner (optional): GS refiner based on render error
    '''

    def __init__(self, depth_net: str, feat_cnn: str, anchor_sampler: dict, point_net: dict,
                 gs_head: dict, refiner: dict, render_resolution: tuple[int, int], num_kv: int):
        super().__init__()
        self.normalize = Normalize((.485, .456, .406), (.229, .224, .225))
        self.depth_net: DepthAnything3Net = DepthAnything3.from_pretrained(depth_net).model
        self.depth_net.eval().requires_grad_(False)
        self.feat_cnn: FeatureCNN = create_object(OmegaConf.create(feat_cnn))
        self.anchor_sampler: VoxelSampler = create_object(OmegaConf.create(anchor_sampler))
        self.point_net: PointBackbone = create_object(OmegaConf.create(point_net))
        self.gs_head: BaseGaussianHead = create_object(OmegaConf.create(gs_head))
        if refiner:
            self.refiner: Refiner = create_object(OmegaConf.create(refiner))
            self.refiner.sampler = self.anchor_sampler
            self.refiner.head = self.gs_head
        else: self.refiner = None
        self.renderer = Renderer(render_resolution)
        self.intrinsics_scale = nn.Buffer(tensor([*render_resolution[::-1], 1.]).view(3, 1))
        self.num_kv = num_kv

    def forward(self, data: Dict[str, Tensor], nv_only: bool = False) -> Dict[str, Tensor]:
        images = self.normalize(data.images)
        with inference_mode():
            intrinsics = data.intrinsics * self.intrinsics_scale
            prediction = self.depth_net.forward(images, data.extrinsics, intrinsics)
            try: prediction = _align_to_input_extrinsics(data.extrinsics, prediction)
            except: pass  # print(f'Umeyama alignment failed for {data.dataset[0]} {data.scene[0]}')
        with inference_mode(self.refiner is not None):
            depth = prediction.depth.to(images.dtype)
            feature = self.feat_cnn.forward(images, depth, data.intrinsics, data.extrinsics)
            feature = feature.movedim(1, -1).unflatten(0, data.images.shape[:2])
            anchors, feature, voxel_sizes = self.anchor_sampler.forward(
                depth, feature, data.intrinsics, data.extrinsics)
            feature = self.point_net.forward(feature, anchors, voxel_sizes)
        gs = self.gs_head.forward(
            feature[:, self.num_kv:], anchors[:, self.num_kv:], voxel_sizes)
        render = self.renderer.forward(data, gs, nv_only and self.refiner is None)
        outputs = Dict(depth=depth, gs=gs, render=render)
        if self.refiner is not None:
            with no_grad():
                gs_kv = self.gs_head.forward(
                    feature[:, :self.num_kv], anchors[:, :self.num_kv], voxel_sizes)
                render_images = self.normalize(render.images)
                gs = Dict({k: v.detach() for k, v in gs.items()})
            gs = self.refiner.forward(gs, gs_kv, render_images, images,
                depth, data.intrinsics, data.extrinsics, voxel_sizes)
            render = self.renderer.forward(data, gs, nv_only)
            outputs.update(refine=Dict(depth=depth, gs=gs, render=render))
        return outputs
