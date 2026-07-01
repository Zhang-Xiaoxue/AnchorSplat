from addict import Dict
import torch
try:
    from depth_anything_3.model.utils.rasterizer import Rasterizer
    GS_VERSION = 'npu'
except ImportError:
    from gsplat import rasterization
    GS_VERSION = 'gsplat'


class Renderer(torch.nn.Module):
    '''
    3DGS render wrapper class that automatically selects from:
    - NPU: https://gitcode.com/cann/cann-recipes-spatial-intelligence/
    - GPU: https://github.com/nerfstudio-project/gsplat
    '''
    def __init__(self, render_resolution: tuple[int, int]):
        super().__init__()
        if GS_VERSION == 'npu': self.rasterizer = Rasterizer()
        self.render_resolution = render_resolution

    @torch.autocast(torch.accelerator.current_accelerator().type, enabled=False)
    def forward(self, data: Dict, gs: Dict, nv_only: bool = False) -> Dict[str, torch.Tensor]:
        B, V = data.extrinsics_nv.shape[:2]
        extrinsics = data.extrinsics_nv.float() if nv_only \
            else torch.cat((data.extrinsics, data.extrinsics_nv), 1).float()
        intrinsics = data.intrinsics_nv.float() if nv_only \
            else torch.cat((data.intrinsics, data.intrinsics_nv), 1).float()
        if GS_VERSION == 'npu':
            render = []
            for i in range(B):
                splats = {k: v[i].float() for k, v in gs.items()}
                for j in range(extrinsics.shape[1]):
                    image, depth = self.rasterizer.ascend_rasterize_splats(
                        w2c=extrinsics[i, j:j + 1],
                        Knorm=intrinsics[i, j:j + 1],
                        width=self.render_resolution[1],
                        height=self.render_resolution[0],
                        tile_size=32,
                        splats=splats,
                        active_sh_degree=0
                    )
                    render.append(torch.cat((image, depth), 1))
            render = torch.cat(render).unflatten(0, intrinsics.shape[:2])
            outputs = Dict(
                images_nv=render[:, -V:, :3].clamp(0, 1),
                depth_nv=render[:, -V:, 3]
            )
            if not nv_only: outputs.update(
                images=render[:, :V, :3].clamp(0, 1),
                depth=render[:, :V, 3]
            )
        else:  # GS_VERSION == 'gsplat'
            H, W = self.render_resolution
            render = rasterization(
                gs.mean,
                gs.rotation,
                gs.scale,
                gs.opacity[..., 0],
                gs.color[..., None, :],
                extrinsics,
                intrinsics * intrinsics.new_tensor((W, H, 1.))[:, None],
                W,
                H,
                near_plane=0.05,
                sh_degree=0,
                render_mode='RGB+ED'
            )[0].movedim(4, 2)
            outputs = Dict(
                images_nv=render[:, -V:, :3].clamp(0, 1),
                depth_nv=render[:, -V:, 3]
            )
            if not nv_only: outputs.update(
                images=render[:, :V, :3].clamp(0, 1),
                depth=render[:, :V, 3]
            )
        return outputs
