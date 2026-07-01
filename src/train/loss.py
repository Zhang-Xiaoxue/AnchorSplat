from addict import Dict
from collections import OrderedDict
import torch
from torch.nn import functional as F
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.functional.image.lpips import _NoTrainLpips

LPIPS_NET = _NoTrainLpips()

def psnr(x: torch.Tensor, y: torch.Tensor, reduce: bool = True) -> torch.Tensor:
    val = F.mse_loss(x, y, reduction='none').mean((-3, -2, -1)).log10_().mul_(-10)
    return val.mean() if reduce else val

def ssim(x: torch.Tensor, y: torch.Tensor, reduce: bool = True) -> torch.Tensor:
    val = structural_similarity_index_measure(
        x.flatten(0, 1), y.flatten(0, 1), data_range=1, reduction='none')
    return val.mean() if reduce else val.view(x.shape[:2])

@torch.autocast(torch.accelerator.current_accelerator().type, enabled=False)
def lpips(x: torch.Tensor, y: torch.Tensor, reduce: bool = True) -> torch.Tensor:
    if next(LPIPS_NET.parameters()).device != x.device: LPIPS_NET.to(x.device)
    val = LPIPS_NET.forward(x.flatten(0, 1).float(), y.flatten(0, 1).float(), normalize=True)
    return val.mean() if reduce else val.view(x.shape[:2])

def absrel(x: torch.Tensor, y: torch.Tensor, valid_mask: torch.Tensor | None = None,
           reduce: bool = True) -> torch.Tensor:
    diff = (x - y).abs() / y.clamp(.01)
    if valid_mask is None: return diff.mean(None if reduce else (-2, -1))
    diff = diff.mul_(valid_mask).sum((-2, -1)) / valid_mask.sum((-2, -1)).clamp(1)
    return diff.mean() if reduce else diff

def delta1(x: torch.Tensor, y: torch.Tensor, threshold: float = 1.25,
           valid_mask: torch.Tensor | None = None, reduce: bool = True) -> torch.Tensor:
    bit_mat = torch.max(x / y, y / x) < threshold
    if valid_mask is None: return bit_mat.float().mean(None if reduce else (-2, -1))
    bit_mat = bit_mat.mul_(valid_mask).sum((-2, -1)) / valid_mask.sum((-2, -1)).clamp(1)
    return bit_mat.mean() if reduce else bit_mat


def train_metrics(inputs: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor],
                  config: Dict[str, float]) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    render = outputs.render
    masks = inputs.masks[:, :, None] if 'masks' in inputs else inputs.images.new_ones(1, 1, 1)
    masks_nv = inputs.masks_nv[:, :, None] if 'masks_nv' in inputs else inputs.images.new_ones(1, 1, 1)
    loss = Dict()
    if config.get(k := 'rgb_l1'):
        loss[k] = config[k] * F.l1_loss(render.images * masks, inputs.images * masks)
    if config.get(k := 'rgb_ssim'):
        loss[k] = config[k] * (1 - ssim(render.images * masks, inputs.images * masks))
    if config.get(k := 'rgb_lpips'):
        loss[k] = config[k] * lpips(render.images * masks, inputs.images * masks)
    if config.get(k := 'rgb_nv_l1'):
        loss[k] = config[k] * F.l1_loss(render.images_nv * masks_nv, inputs.images_nv * masks_nv)
    if config.get(k := 'rgb_nv_ssim'):
        loss[k] = config[k] * (1 - ssim(render.images_nv * masks_nv, inputs.images_nv * masks_nv))
    if config.get(k := 'rgb_nv_lpips'):
        loss[k] = config[k] * lpips(render.images_nv * masks_nv, inputs.images_nv * masks_nv)
    if config.get(k := 'depth_l1'):
        loss[k] = config[k] * F.l1_loss(render.depth * masks[:, :, 0], outputs.depth * masks[:, :, 0])
    if config.get(k := 'disparity_reg_l1'):
        loss[k] = config[k] * render.depth.clamp(.1).reciprocal().mean()
    if config.get(k := 'gs_scale_prod'):
        loss[k] = config[k] * outputs.gs.scale.prod(-1).mean()
    if config.get(k := 'gs_opacity_l1'):
        loss[k] = config[k] * outputs.gs.opacity.mean()
    for k, v in loss.items(): loss[k] = v.clamp_max(1)
    loss_total = sum(loss.values())
    metrics = Dict()
    with torch.inference_mode():
        metrics.loss = loss_total
        metrics.psnr = psnr(render.images * masks, inputs.images * masks)
        metrics.psnr_nv = psnr(render.images_nv * masks_nv, inputs.images_nv * masks_nv)
        metrics.ssim = ssim(render.images * masks, inputs.images * masks)
        metrics.ssim_nv = ssim(render.images_nv * masks_nv, inputs.images_nv * masks_nv)
        metrics.lpips = lpips(render.images * masks, inputs.images * masks)
        metrics.lpips_nv = lpips(render.images_nv * masks_nv, inputs.images_nv * masks_nv)
        metrics.update({'loss/' + k: v for k, v in loss.items()})
    return loss_total, metrics


@torch.autocast(torch.accelerator.current_accelerator().type, enabled=False)
def test_metrics(inputs: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor],
                 center_crop: bool = False) -> tuple[list[str], torch.Tensor]:
    '''
    Returns:
        (keys, values) (list[str], torch.Tensor (B, V, K)):
        metric names and corresponding per-scene per-view values,
        metrics stacked along the final dim
    '''
    x, y = outputs.render.images_nv, inputs.images_nv
    xd, yd = outputs.render.depth_nv, outputs.depth  # inputs.depth_nv
    if center_crop:
        S = min(x.shape[-2:])
        I, J = (x.shape[-2] - S) // 2, (x.shape[-1] - S) // 2
        x, y = x[..., I:I + S, J:J + S], y[..., I:I + S, J:J + S]
        xd, yd = xd[..., I:I + S, J:J + S], yd[..., I:I + S, J:J + S]
    if 'masks_nv' in inputs:
        masks_nv = inputs.masks_nv[..., I:I + S, J:J + S]
        x = x * masks_nv[:, :, None]
        y = y * masks_nv[:, :, None]
    else: masks_nv = None
    metrics = OrderedDict()
    metrics['image/psnr'] = psnr(x, y, reduce=False)
    metrics['image/ssim'] = ssim(x, y, reduce=False)
    metrics['image/lpips'] = lpips(x, y, reduce=False)
    metrics['depth/absrel'] = absrel(xd, yd, masks_nv, reduce=False)
    metrics['depth/delta1'] = delta1(xd, yd, 1.25, masks_nv, reduce=False)
    return metrics.keys(), torch.stack(list(metrics.values()), -1)
