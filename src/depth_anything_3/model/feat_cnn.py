from torch import cat, linspace, nn, Tensor
from torch.nn.functional import interpolate, normalize


class FeatureCNN(nn.Module):
    def __init__(
        self,
        resolution: tuple[int, int],
        dim_down: tuple[int, int, int],
        dim_bottleneck: tuple[int, int],
        dim_out: int
    ):
        super().__init__()
        self.resolution = resolution
        self.down = nn.Sequential(
            nn.Conv2d(10, dim_down[0], 3, 2, 1),
            nn.BatchNorm2d(dim_down[0]),
            nn.ReLU(True),
            nn.Conv2d(dim_down[0], dim_down[1], 3, 2, 1),
            nn.BatchNorm2d(dim_down[1]),
            nn.ReLU(True),
            nn.Conv2d(dim_down[1], dim_down[2], 3, 1, 1),
            nn.BatchNorm2d(dim_down[2]),
            nn.ReLU(True)
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(dim_down[-1], dim_bottleneck[0], 3, 1, 1),
            nn.BatchNorm2d(dim_bottleneck[0]),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim_bottleneck[0], dim_bottleneck[1], 3, 1, 1),
            nn.BatchNorm2d(dim_bottleneck[1]),
            nn.ReLU(True)
        )
        self.out = nn.Conv2d(dim_down[-1] + dim_bottleneck[-1], dim_out, 1)

    def plucker_embed(self, intrinsics: Tensor, extrinsics: Tensor):
        B, V = intrinsics.shape[:2]
        H, W = self.resolution
        device, dtype = intrinsics.device, intrinsics.dtype
        x = linspace(.5 / W, 1 - .5 / W, W, device=device, dtype=dtype)
        y = linspace(.5 / H, 1 - .5 / H, H, device=device, dtype=dtype)[:, None]
        rays_d = intrinsics.new_ones(B, V, 3, H, W)
        rays_d[:, :, 0] = (x - intrinsics[:, :, :1, 2:]) / intrinsics[:, :, :1, :1]
        rays_d[:, :, 1] = (y - intrinsics[:, :, 1:2, 2:]) / intrinsics[:, :, 1:2, 1:2]
        rays_d = normalize(rays_d, dim=-1)
        rays_o = extrinsics[:, :, :3, :3].mT @ -extrinsics[:, :, :3, 3:]
        cross = rays_d.cross(rays_o[..., None], 2)
        return rays_d, cross

    def forward(self, images: Tensor, depth: Tensor, intrinsics: Tensor, extrinsics: Tensor) -> Tensor:
        depth = interpolate(depth.flatten(0, 1).unsqueeze(1), self.resolution)
        rays_d, cross = self.plucker_embed(intrinsics, extrinsics)
        x = cat((images.flatten(0, 1), depth, rays_d.flatten(0, 1), cross.flatten(0, 1)), 1)
        x = self.down(x)
        b = self.bottleneck(x)
        return self.out(cat((x, b), 1))
