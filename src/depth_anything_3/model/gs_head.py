from abc import abstractmethod
from addict import Dict
import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Callable


class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: Tensor, min: Tensor | None = None, max: Tensor | None = None):
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return grad_output.clone(), None, None

clamp: Callable[..., Tensor] = ClampWithGrad.apply


class GaussianMLP(nn.Module):
    def __init__(self, dim_in: int, dim: int, dim_out: int, depth: int,
                 input_norm: bool, activation: str, zero_init: bool):
        super().__init__()
        self.net = nn.Sequential()
        if input_norm: self.net.append(nn.LayerNorm(dim_in))
        if depth == 1: self.net.append(nn.Linear(dim_in, dim_out))
        else:
            activation = getattr(nn, activation)()
            self.net.append(nn.Linear(dim_in, dim))
            self.net.append(activation)
            for _ in range(depth - 2):
                self.net.append(nn.Linear(dim, dim))
                self.net.append(activation)
            self.net.append(nn.Linear(dim, dim_out))
        if zero_init:
            nn.init.normal_(self.net[-1].weight, std=1e-3)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class BaseGaussianHead(nn.Module):
    def __init__(
        self,
        num_gs: int,
        sh_degree: int,
        offset_max: float = 10,
        opacity_bias: float = .1,
        scale_act: str = 'sigmoid',
        scale_bias: float = 1,
        scale_max: float = 2,
        rotation_bias: tuple[float, float, float, float] = (1., 0., 0., 0.)
    ):
        super().__init__()
        self.num_gs = num_gs
        self.sh_mask = nn.Buffer(torch.ones(3 * (1 + sh_degree) ** 2))
        for i in range(1, sh_degree): self.sh_mask[3 * i ** 2:3 * (i + 1) ** 2] = .25 ** i

        inv_sigmoid = lambda x: math.log(x / (1 - x))
        inv_softplus = lambda x: math.log(math.expm1(x))
        self.offset_max = offset_max
        self.opacity_bias = inv_sigmoid(opacity_bias)
        self.scale_bias = inv_softplus(scale_bias) if scale_act == 'softplus' \
            else inv_sigmoid(scale_bias / scale_max)
        self.scale_max = scale_max
        self.rotation_bias = nn.Buffer(torch.tensor(rotation_bias))

        self.act: dict[str, Callable[..., Tensor]] = Dict(  # apply scale and bias
            mean=lambda x, s: self.offset_max * s * F.tanh(x),
            opacity=lambda x, s: F.sigmoid(x + self.opacity_bias),
            scale=lambda x, s:
                s * ClampWithGrad.apply(F.softplus(x + self.scale_bias), None, self.scale_max) \
                if scale_act == 'softplus' else (scale_max * s * F.sigmoid(x + self.scale_bias)),
            rotation=lambda x, s: F.normalize(x + self.rotation_bias, dim=-1),
            color=lambda x, s: self.sh_mask * x
        )
        # self.inv_act: dict[str, Callable[..., Tensor]] = Dict(  # remove scale and bias
        #     mean=lambda x, s: (x / (offset_max * s)).atanh(),
        #     opacity=lambda x, s: ((y := x.clamp(.05, .95)) / (1 - y)).log() - self.opacity_bias,
        #     scale=lambda x, s:
        #         (x / (scale_max * s)).expm1().log() - self.scale_bias if scale_act == 'softplus' \
        #         else ((y := (x / (scale_max * s)).clamp(.05, .95)) / (1 - y)).log() - self.scale_bias,
        #     rotation=lambda x, s: x - self.rotation_bias,
        #     color=lambda x, s: x / self.sh_mask
        # )

    @abstractmethod
    def forward(self, x: Tensor, anchors: Tensor, voxel_sizes: Tensor) -> Dict[str, Tensor]: pass


class SharedGaussianHead(BaseGaussianHead):
    def __init__(
        self,
        num_gs: int,
        sh_degree: int,
        mlp_args: dict,
        gs_args: dict
    ):
        super().__init__(num_gs, sh_degree, **gs_args)
        dim_out = (11 + 3 * (1 + sh_degree) ** 2) * num_gs
        self.mlp = GaussianMLP(dim_out=dim_out, **mlp_args)

    def forward(self, x: Tensor, anchors: Tensor, voxel_sizes: Tensor) -> Dict[str, Tensor]:
        B, N, _ = x.shape
        voxel_sizes = voxel_sizes.view(B, 1, 1)
        x = self.mlp.forward(x).view(B, N * self.num_gs, -1)
        gs = Dict()
        for attr, xi in zip(('mean', 'opacity', 'scale', 'rotation', 'color'),
                            x.split((3, 1, 3, 4, x.shape[2] - 11), 2)):
            gs[attr] = self.act[attr](xi.float(), voxel_sizes)
        gs.mean = anchors.repeat_interleave(self.num_gs, 1) + gs.mean
        return gs


class SplitGaussianHead(BaseGaussianHead):
    def __init__(
        self,
        num_gs: int,
        sh_degree: int,
        mlp_args: dict,
        gs_args: dict
    ):
        super().__init__(num_gs, sh_degree, **gs_args)
        color_dim = 3 * (1 + sh_degree) ** 2
        self.mlp: dict[str, GaussianMLP] = nn.ModuleDict(dict(
            mean=GaussianMLP(dim_out=3 * num_gs, **mlp_args),
            opacity=GaussianMLP(dim_out=num_gs, **mlp_args),
            scale=GaussianMLP(dim_out=3 * num_gs, **mlp_args),
            rotation=GaussianMLP(dim_out=4 * num_gs, **mlp_args),
            color=GaussianMLP(dim_out=color_dim * num_gs, **mlp_args)
        ))

    def forward(self, x: Tensor, anchors: Tensor, voxel_sizes: Tensor) -> Dict[str, Tensor]:
        B, N, _ = x.shape
        voxel_sizes = voxel_sizes.view(B, 1, 1)
        gs = Dict()
        for attr in 'mean', 'opacity', 'scale', 'rotation', 'color':
            y = self.mlp[attr].forward(x).view(B, N * self.num_gs, -1)
            gs[attr] = self.act[attr](y.float(), voxel_sizes)
        gs.mean = anchors.repeat_interleave(self.num_gs, 1) + gs.mean
        return gs
