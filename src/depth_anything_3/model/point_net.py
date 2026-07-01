from functools import partial
from omegaconf import OmegaConf
import torch
from torch import nn, Tensor
from torch.nn import Module, Sequential
from torch.utils.checkpoint import checkpoint_sequential
from torchvision.models import resnet18, ResNet18_Weights

try:
    from torch_npu import npu_fusion_attention
    NPU_AVAILABLE = True
except:
    from torch.nn.functional import scaled_dot_product_attention
    NPU_AVAILABLE = False

from depth_anything_3.cfg import create_object
from depth_anything_3.model.utils.attention import Attention
from depth_anything_3.model.utils.block import Block
from depth_anything_3.model.point_sampler import \
    aggregate_feature, contract, RADIUS, VoxelSampler
from depth_anything_3.model.gs_head import BaseGaussianHead, clamp, GaussianMLP


class SimpleAttention(Attention):
    def __init__(self, dim: int, num_heads: int, **kwargs):
        super().__init__(dim, num_heads, **kwargs)
        if NPU_AVAILABLE: self.spda = lambda q, k, v: npu_fusion_attention(
            q, k, v,
            head_num=num_heads,
            input_layout='BSH',
            scale=self.scale
        )[0]
        else: self.spda = lambda q, k, v: scaled_dot_product_attention(
            q.unflatten(2, (self.num_heads, self.head_dim)).transpose(1, 2),
            k.unflatten(2, (self.num_heads, self.head_dim)).transpose(1, 2),
            v.unflatten(2, (self.num_heads, self.head_dim)).transpose(1, 2)
        ).transpose(1, 2).flatten(2)

    def forward(self, x: Tensor, pos=None, attn_mask=None) -> Tensor:
        q, k, v = self.qkv.forward(x).chunk(3, 2)
        x = self.spda(q, k, v)
        x = self.proj(x)
        return x


class SampleAttention(SimpleAttention):
    def __init__(self, dim: int, num_heads: int, num_kv: int, **kwargs):
        super().__init__(dim, num_heads, **kwargs)
        self.num_kv = num_kv

    def forward(self, x: Tensor, pos=None, attn_mask=None) -> Tensor:
        q, k, v = self.qkv.forward(x).chunk(3, 2)
        x = self.spda(q, k[:, :self.num_kv], v[:, :self.num_kv])
        x = self.proj(x)
        return x


class PointBackbone(Module):
    def __init__(self, dim_in: int, dim: int, num_heads: int, depth: int, num_kv: int,
                 num_pe_freqs: int, pe_contract: bool = False, checkpoint: bool = True):
        super().__init__()
        self.feat_proj = nn.Linear(dim_in, dim) if dim_in != dim else nn.Identity()
        freqs = torch.logspace(0, num_pe_freqs - 1, num_pe_freqs, 2)
        self.freqs = nn.Buffer(torch.pi / 2 / RADIUS * freqs)
        self.coord_proj = nn.Linear(3 + 6 * num_pe_freqs, dim)
        self.pe_contract = pe_contract  # whether to apply PE on contracted coordinates
        self.depth = depth
        attn = partial(SampleAttention, num_kv=num_kv)
        self.blocks = Sequential(*[Block(dim, num_heads, attn_class=attn) for _ in range(depth)])
        self.checkpoint = checkpoint

    def forward(self, feature: Tensor, coords: Tensor, voxel_sizes: Tensor = None) -> Tensor:
        if self.pe_contract: coords = .5 * contract(coords)  # norm within RADIUS
        embed = (coords[..., None] * self.freqs).flatten(2)
        embed = torch.cat((coords, embed.cos(), embed.sin()), 2)
        feature = self.feat_proj.forward(feature) + self.coord_proj.forward(embed)
        return checkpoint_sequential(self.blocks, self.depth, feature, False) \
            if self.checkpoint else self.blocks(feature)


class Refiner(nn.Module):
    sampler: VoxelSampler
    head: BaseGaussianHead

    # def __init__(self, backbone: dict, mlp: dict):
    #     super().__init__()
    #     feat_cnn = resnet18(weights=ResNet18_Weights.DEFAULT)
    #     layers = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4']
    #     self.feat_cnn = nn.Sequential(*(getattr(feat_cnn, l) for l in layers))
    #     self.num_kv = backbone['num_kv']
    #     self.backbone: PointBackbone = create_object(OmegaConf.create(backbone))
    #     self.mlp: GaussianMLP = create_object(OmegaConf.create(mlp))

    # def forward(self, gs: dict[str, Tensor], gs_kv: dict[str, Tensor], render: Tensor,
    #             images: Tensor, depth: Tensor, intrinsics: Tensor, extrinsics: Tensor,
    #             voxel_sizes: Tensor) -> dict[str, Tensor]:
    #     # sample kv coordinates
    #     coords = contract(gs.mean) if self.sampler.contract_scene else gs.mean
    #     anchors, _ = self.sampler.sample_batch(coords, self.num_kv)
    #     coords = torch.cat((anchors, coords), 1)

    #     # pointwise render error feature
    #     error = (images - render).movedim(2, -1)
    #     error = aggregate_feature(coords, error, depth, intrinsics, extrinsics)
    #     feature = self.feat_cnn(images.flatten(0, 1)) - self.feat_cnn(render.flatten(0, 1))
    #     feature = feature.movedim(1, -1).unflatten(0, images.shape[:2])
    #     feature = aggregate_feature(coords, feature, depth, intrinsics, extrinsics)
    #     gs_vals = torch.cat(list(gs.values()), 2).to(feature.dtype)
    #     gs_kv_vals = torch.cat(list(gs_kv.values()), 2).to(feature.dtype)
    #     feature = torch.cat((torch.cat((gs_kv_vals, gs_vals), 1), error, feature), 2)

    #     # GS update
    #     update = self.backbone.forward(feature, coords, voxel_sizes)
    #     update = self.mlp.forward(torch.cat((gs_vals, update[:, self.num_kv:]), 2))
    #     update = update.split_with_sizes([v.shape[2] for v in gs.values()], 2)
    #     # return {k: self.head.act[k](x + self.head.inv_act[k](v, voxel_sizes), voxel_sizes)
    #     #         for x, (k, v) in zip(update, gs.items())}
    #     offset_max = self.head.offset_max * voxel_sizes
    #     scale_max = self.head.scale_max * voxel_sizes
    #     gs.mean = gs.mean + clamp(update[0], -offset_max, offset_max)
    #     gs.opacity = clamp(gs.opacity + update[1], 0, 1)
    #     gs.scale = clamp(gs.scale + update[2], torch.zeros_like(scale_max), scale_max)
    #     gs.rotation = nn.functional.normalize(gs.rotation + update[3], dim=-1)
    #     gs.color = gs.color + update[4] * self.head.sh_mask
    #     return gs