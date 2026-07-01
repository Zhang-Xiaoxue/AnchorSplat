from addict import Dict
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from glob import glob
import imageio
from matplotlib.pyplot import get_cmap
import numpy as np
from omegaconf import OmegaConf
import os
from pathlib import Path
from shutil import copytree
from time import perf_counter
import torch
from torch import distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn import functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.io import write_png
from tqdm import tqdm
try:
    import torch_npu.npu
    import torch_npu.optim
    from torch_npu import profiler as prof
    DEVICE_TYPE = 'npu'
except: DEVICE_TYPE = 'cuda'

from depth_anything_3.cfg import create_object, load_config
from depth_anything_3.model.anchorsplat_mix import AnchorSplatNet
from depth_anything_3.utils.camera_trj_helpers import interpolate_camera
# from depth_anything_3.model.utils.utils3d_transforms import generate_cameras
from depth_anything_3.utils.gsply_helpers import export_ply, inverse_sigmoid
from depth_anything_3.utils.logger import logger
from train.dataset_mix import MixedDataset
from train.loss import train_metrics, test_metrics

colormap = get_cmap('Spectral')
colormap._init()
colormap = torch.tensor((255 * colormap._lut[:256, :3].T).round(), dtype=torch.uint8)


@torch.inference_mode()
def resize_inputs(inputs: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]):
    B, V, _, H, W = outputs.render.images_nv.shape
    if inputs.images.shape[-2:] != (H, W):
        inputs.images = F.interpolate(
            inputs.images.flatten(0, 1), (H, W), mode='area').unflatten(0, (B, V))
        inputs.images_nv = F.interpolate(
            inputs.images_nv.flatten(0, 1), (H, W), mode='area').unflatten(0, (B, V))
    if 'depth' in inputs and inputs.depth.shape[-2:] != (H, W):
        inputs.depth = F.interpolate(inputs.depth, (H, W))
        if 'depth_nv' in inputs: inputs.depth_nv = F.interpolate(inputs.depth_nv, (H, W))
    if 'depth' not in outputs: outputs.depth = inputs.depth
    if 'masks' in inputs and inputs.masks.shape[-2:] != (H, W):
        with torch.inference_mode(False):
            inputs.masks = F.interpolate(inputs.masks.float(), (H, W)).bool()
            inputs.masks_nv = F.interpolate(inputs.masks_nv.float(), (H, W)).bool()


@torch.inference_mode()
def visualize(inputs: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    MAX_VIEWS, MAX_WIDTH = 8, 224
    V = inputs.images.shape[1]
    indices = ... if V <= MAX_VIEWS else torch.linspace(0, V - 1, MAX_VIEWS, dtype=torch.int)
    depth = torch.stack((outputs.depth[0, indices], outputs.render.depth[0, indices]), 1)
    factor = MAX_WIDTH / outputs.depth.shape[-1]
    if factor < 1: depth = F.interpolate(depth, scale_factor=factor, mode='area')
    depth = depth.mul_(256 / depth.amax((1, 2, 3), True)).int().clamp_(0, 255).cpu()
    depth = colormap[:, depth.movedim(0, 2).flatten(0, 1).flatten(1)]
    vis = torch.cat((
        inputs.images[0, indices], outputs.render.images[0, indices],
        inputs.images_nv[0, indices], outputs.render.images_nv[0, indices]
    ), 2)
    if factor < 1: vis = F.interpolate(vis, scale_factor=factor, mode='area')
    vis = vis.clamp_(0, 1).movedim(0, 2).flatten(2).mul_(255).round_().byte().cpu()
    return torch.cat((vis[:, :depth.shape[1]], depth, vis[:, depth.shape[1]:]), 1)


def train_epoch(
    model: torch.nn.parallel.DistributedDataParallel,
    sampler: DistributedSampler,
    loader: DataLoader,
    epoch_range: tuple[int, int],
    config: Dict,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    writer: SummaryWriter,
    log_dir: str,
    profiler: torch.profiler.profile,
):
    rank0 = not dist.get_rank()
    getattr(torch, DEVICE_TYPE).empty_cache()
    model.train()
    model.module.depth_net.eval()
    refine = model.module.refiner is not None
    if profiler is not None: profiler.start()

    for epoch in range(*epoch_range):
        if sampler is not None: sampler.set_epoch(epoch)
        pbar = tqdm(loader, f'Epoch {epoch}', disable=not rank0)
        for inputs in pbar:
            log_step = scheduler.last_epoch
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor): inputs[k] = v.to(DEVICE_TYPE)
            outputs = model(inputs)
            resize_inputs(inputs, outputs)
            loss, metrics = train_metrics(inputs, outputs, config.loss)
            if refine:
                loss_refine, metrics_refine = train_metrics(inputs, outputs.refine, config.loss)
                loss = loss + loss_refine
                metrics.update({'refine/' + k: v for k, v in metrics_refine.items()})
                metrics.update({'refine-diff/' + k: v - metrics[k]
                                for k, v in metrics_refine.items() if 'loss' not in k})
            if log_step > 100: loss = loss.clamp_max(1)  # zero gradients for large loss
            loss.backward()

            if rank0 and not log_step % config.steps.vis_every:
                vis = visualize(inputs, outputs)
                filename = f'{log_step:05}-{inputs.dataset[0]}-{inputs.scene[0]}.png'
                write_png(vis, f'{log_dir}/train/{filename}', 9)
                if refine:
                    vis = visualize(inputs, outputs.refine)
                    filename = f'{filename[:-4]}-refine.png'
                    write_png(vis, f'{log_dir}/train/{filename}', 9)
            log_values = torch.stack(list(metrics.values())).detach()
            dist.reduce(log_values, 0, dist.ReduceOp.AVG)
            if rank0:
                for k, v in zip(metrics.keys(), log_values):
                    metrics[k] = v = v.item()
                    writer.add_scalar('train/' + k, v, log_step, new_style=True)
                pbar.set_postfix(metrics)
            if profiler is not None: profiler.step()

            if hasattr(optimizer, 'clip_grad_norm_fused_'): optimizer.clip_grad_norm_fused_(1)
            else: torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    if rank0: writer.flush()
    if profiler is not None: profiler.stop()


@torch.inference_mode()
def val_epoch(
    model: torch.nn.parallel.DistributedDataParallel,
    loader: DataLoader,
    config: Dict,
    log_step: int,
    writer: SummaryWriter,
    log_dir: str,
) -> float | None:
    rank0 = not dist.get_rank()
    refine = model.module.refiner is not None
    getattr(torch, DEVICE_TYPE).empty_cache()
    model.eval()
    pbar = tqdm(loader, f'Evaluating {loader.dataset.name}', disable=not rank0)
    metrics_all = 0
    for inputs in pbar:
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor): inputs[k] = v.to(DEVICE_TYPE)
        outputs = model(inputs)
        resize_inputs(inputs, outputs)
        _, metrics = train_metrics(inputs, outputs, config.loss)
        if refine:
            _, metrics_refine = train_metrics(inputs, outputs.refine, config.loss)
            metrics.update({'refine/' + k: v for k, v in metrics_refine.items()})
        if rank0:
            vis = visualize(inputs, outputs)
            filename = f'{log_step:05}-{inputs.dataset[0]}-{inputs.scene[0]}.png'
            write_png(vis, f'{log_dir}/val/{filename}', 9)
            if refine:
                vis = visualize(inputs, outputs.refine)
                filename = f'{filename[:-4]}.png'
                write_png(vis, f'{log_dir}/val/{filename}', 9)
            pbar.set_postfix({k: v.item() for k, v in metrics.items()})
        metrics_all += torch.stack(list(metrics.values()))
    metrics_all /= max(1, len(loader))
    dist.reduce(metrics_all, 0, dist.ReduceOp.AVG)
    key = 'refine/psnr_nv' if refine else 'psnr_nv'
    psnr_nv = None
    if rank0:
        for k, v in zip(metrics.keys(), metrics_all.tolist()):
            if k == key: psnr_nv = v
            writer.add_scalar(f'val_{inputs.dataset[0]}/' + k, v, log_step, new_style=True)
        writer.flush()
    return psnr_nv


@torch.inference_mode()
def test_epoch(
    model: torch.nn.parallel.DistributedDataParallel,
    loader: DataLoader,
    config: Dict,
    log_dir: str,
    executor: ThreadPoolExecutor
):
    rank0 = not dist.get_rank()
    refine = model.module.refiner is not None
    getattr(torch, DEVICE_TYPE).empty_cache()
    model = model.float().eval().requires_grad_(False)
    metrics, scenes, runtime = [], [], 0

    pbar = tqdm(loader, f'Testing {loader.dataset.name}', disable=not rank0)
    for inputs in pbar:
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor): inputs[k] = v.to(DEVICE_TYPE)
        if config.get('warmup') and not runtime: model(inputs)  # dummy batch

        scenes.extend(inputs.scene)
        start = perf_counter()
        outputs = model(inputs, nv_only=config.nv_only)
        runtime += perf_counter() - start

        resize_inputs(inputs, outputs)
        keys, metrics_single = test_metrics(inputs, outputs, config.center_crop)
        if refine:
            keys_refine, metrics_refine = test_metrics(inputs, outputs.refine, config.center_crop)
            keys = list(keys) + ['refine/' + k for k in keys_refine]
            metrics_single = torch.cat((metrics_single, metrics_refine), -1)
        metrics.append(metrics_single)
        if rank0:
            values = metrics_single.mean((0, 1)).tolist()
            pbar.set_postfix({k: v for k, v in zip(keys, values)})

        if not config.nv_only:
            vis = visualize(inputs, outputs)
            filename = f'{inputs.datset[0]}-{inputs.scene[0]}.png'
            executor.submit(write_png, vis, f'{log_dir}/test/{filename}', 9)
            if 'refine' in outputs:
                vis = visualize(inputs, outputs.refine)
                filename = f'{inputs.datset[0]}-{inputs.scene[0]}-rerfine.png'
                executor.submit(write_png, vis, f'{log_dir}/test/{filename}', 9)
        if config.export_gs:
            gs = Dict(
                means=outputs.gs.mean[0],
                scales=outputs.gs.scale[0].log(),
                rotations=outputs.gs.rotation[0],
                harmonics=outputs.gs.color[0, ..., None],
                opacities=inverse_sigmoid(outputs.gs.opacity[0, :, 0])
            )
            filename = f'{inputs.dataset[0]}-{inputs.scene[0]}.ply'
            executor.submit(export_ply, **gs, path=Path(f'{log_dir}/test/{filename}'))
        if config.render_video:
            data = Dict()
            # data.extrinsics_nv, data.intrinsics_nv = generate_cameras(
            #     center=outputs.gs.mean.mean(1), **config.render_video)
            data.intrinsics_nv, data.extrinsics_nv = interpolate_camera(
                inputs.intrinsics, inputs.extrinsics, config.render_video.num_views)
            render = model.module.renderer.forward(data, outputs.gs, True)
            images = render.images_nv.mul_(255).clamp_(0, 255).byte().cpu().movedim(2, -1).numpy()
            depths = render.depth_nv.mul_(256 / render.depth_nv.amax((1, 2, 3), True))
            depths = colormap[:, depths.clamp_(0, 255).int().cpu()].movedim(0, -1).numpy()
            for dataset, scene, image, depth in zip(inputs.dataset, inputs.scene, images, depths):
                filename = f'{log_dir}/test/{dataset}-{scene}-image.mp4'
                executor.submit(imageio.v3.imwrite, filename, image, fps=60)
                filename = f'{log_dir}/test/{dataset}-{scene}-depth.mp4'
                executor.submit(imageio.v3.imwrite, filename, depth, fps=60)

    metrics = torch.cat(metrics)
    metrics_all = metrics.new_zeros(dist.get_world_size(), *metrics.shape)
    scenes_all = [None] * dist.get_world_size() if rank0 else None
    runtime = torch.tensor(runtime / len(scenes), device=DEVICE_TYPE)
    dist.all_gather_into_tensor(metrics_all, metrics)
    dist.gather_object(scenes, scenes_all)
    dist.reduce(runtime, 0, dist.ReduceOp.AVG)
    dist.barrier()

    if rank0:
        scenes = []
        for l in scenes_all: scenes.extend(l)
        unique, scenes_set = [], set()
        for i, scene in enumerate(scenes):
            if scene not in scenes_set:
                unique.append(i)
                scenes_set.add(scene)
        metrics = metrics_all.flatten(0, 1)[unique]
        metrics_np = {k: metrics[..., i].cpu().numpy() for i, k in enumerate(keys)}
        np.savez(f'{log_dir}/test/metrics_{inputs.dataset[0]}', **metrics_np)
        metrics = metrics.mean((0, 1)).tolist()
        lines = [f'runtime: {runtime.item()}\n']
        for k, v in zip(keys, metrics): lines.append(f'{k}: {v}\n')
        with open(f'{log_dir}/test/metrics_{inputs.dataset[0]}.txt', 'w') as file:
            file.writelines(lines)


@record
def main():
    # config and arguments
    parser = ArgumentParser()
    parser.add_argument('--config', help='Path to config yaml')
    parser.add_argument('--output', help='Logging directory')
    parser.add_argument('-o', '--override', action='append',
                        help='Override configs with parent.child=value')
    parser.add_argument('--ckpt', help='Load checkpoint')
    parser.add_argument('--test', action='store_true', help='Run test only')
    parser.add_argument('--profile', action='store_true', help='Profiling')
    parser.add_argument('--debug', action='store_true', help='Nan debugging')
    args = parser.parse_args()
    config = load_config(args.config, args.override)

    # pytorch distributed settings
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.accelerator.set_device_index(local_rank)
    dist.init_process_group('hccl' if DEVICE_TYPE == 'npu' else 'nccl')
    if DEVICE_TYPE == 'npu': torch_npu.npu.set_device(local_rank)
    else: torch.cuda.set_device(local_rank)
    rank0 = not dist.get_rank()

    # datasets
    if not args.test:
        config.dataset.training = True
        train_dataset: MixedDataset = create_object(config.dataset)
        train_sampler = DistributedSampler(train_dataset, drop_last=True)
        train_loader = DataLoader(
            train_dataset, config.trainer.batch_size,
            sampler=train_sampler, num_workers=int(not args.debug),
            pin_memory=True, persistent_workers=not args.debug
        )
        config.dataset.training = False
        val_dataset: MixedDataset = create_object(config.dataset)
        val_loaders: list[DataLoader] = []
        for dataset in val_dataset.datasets:
            val_loaders.append(DataLoader(
                dataset, config.trainer.batch_size,
                sampler=DistributedSampler(dataset, shuffle=False),
                num_workers=int(not args.debug),
                pin_memory=True, persistent_workers=not args.debug
            ))

    # torch_npu.npu.memory._record_memory_history()

    # model and optimzier
    model: AnchorSplatNet = create_object(config.model)
    dtype = torch.bfloat16 if config.trainer.dtype == 'bf16' \
        else torch.half if config.trainer.dtype == 'fp16' else torch.float
    model.to(DEVICE_TYPE, dtype)
    if config.trainer.refiner_only:
        model.feat_cnn.requires_grad_(False)
        model.point_net.requires_grad_(False)
        model.gs_head.requires_grad_(False)
    model = torch.nn.parallel.DistributedDataParallel(model, [local_rank], local_rank)
    params = [p for p in model.parameters() if p.requires_grad]
    if rank0: print('Number of trainable parameters:', sum(p.numel() for p in params))
    if DEVICE_TYPE == 'npu' and dtype != torch.bfloat16:
        optimizer = torch_npu.optim.NpuFusedAdamW(params, config.trainer.lr)
    else: optimizer = torch.optim.AdamW(params, config.trainer.lr)
    lr_lambda = lambda x: min(1, x / config.trainer.steps.warmup)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # checkpoint
    try:
        file = args.ckpt if args.ckpt \
            else sorted(glob(f'{args.output}/ckpt/epoch_*.ckpt'))[-1]
        ckpt = torch.load(file, DEVICE_TYPE)
        best_epoch = int(file.split('epoch_')[-1][:-5])
        model.load_state_dict(ckpt[0], False)
        if best_epoch:
            optimizer.load_state_dict(ckpt[1])
            scheduler.load_state_dict(ckpt[2])
        best_metric = ckpt[3] if len(ckpt) == 4 else -torch.inf
    except:
        assert not args.test, 'No available checkpoints for test'
        ckpt, best_epoch, best_metric = None, 0, -torch.inf

    # debugging and profiling
    if args.debug: torch.autograd.set_detect_anomaly(True)
    if rank0 and args.profile and DEVICE_TYPE == 'npu':
        profiler = prof.profile(
            activities=[prof.ProfilerActivity.CPU,
                        prof.ProfilerActivity.NPU],
            schedule=prof.schedule(0, 5, 1, 1, 1),
            on_trace_ready=prof.tensorboard_trace_handler(f'{args.output}/profile'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            with_modules=True,
            experimental_config=prof._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                l2_cache=True, msprof_tx=True, mstx=True,
                export_type=[torch_npu.profiler.ExportType.Db],
                sys_io=True, sys_interconnection=True
            )
        )
    else: profiler = None

    # logging
    if not args.test:
        max_epochs = config.trainer.epochs.total
        interval = config.trainer.epochs.eval_every
    if rank0 and not args.test:
        for subdir in 'src', 'ckpt', 'train', 'val', 'tensorboard':
            os.makedirs(os.path.join(args.output, subdir), exist_ok=True)
        copytree('.', os.path.join(args.output, 'src'), dirs_exist_ok=True)
        logger.info(f'Training epoch {best_epoch} — {max_epochs} ({len(train_loader)} steps / epoch)')
        logger.info(f'Evaluating every {interval} epochs')
        writer = SummaryWriter(os.path.join(args.output, 'tensorboard'))
        OmegaConf.save(config, os.path.join(args.output, 'config.yaml'), True)
    else: writer = None

    # training and validation
    if not args.test:
        with torch.autocast(DEVICE_TYPE, dtype):
            for start in range(best_epoch, max_epochs, interval):
                end = min(start + interval, max_epochs)
                train_epoch(model, train_sampler, train_loader, (start, end), config.trainer,
                            optimizer, scheduler, writer, args.output, profiler)
                if rank0: metric = 0
                for i, loader in enumerate(val_loaders):
                    psnr_nv = val_epoch(
                        model, loader, config.trainer, scheduler.last_epoch, writer, args.output)
                    if rank0: metric += val_dataset.weights[i] * psnr_nv
                if rank0:
                    best, periodic = metric > best_metric, not (end % config.trainer.epochs.ckpt_every)
                    if best or periodic:
                        ckpt = model.state_dict(), optimizer.state_dict(), scheduler.state_dict(), metric
                        torch.save(ckpt, f'{args.output}/ckpt/epoch_{end:04}.ckpt')
                    if best:
                        if best_epoch and best_epoch % config.trainer.epochs.ckpt_every:
                            os.remove(f'{args.output}/ckpt/epoch_{best_epoch:04}.ckpt')
                        best_epoch, best_metric = end, metric
        if rank0: writer.close()
        del train_dataset, train_sampler, train_loader, optimizer, scheduler, \
            val_dataset, val_loaders, ckpt, profiler, writer

    # torch_npu.npu.memory._dump_snapshot()

    # testing
    config.dataset.training = False
    config.dataset.num_views = config.trainer.test.num_views
    test_dataset: MixedDataset = create_object(config.dataset)
    test_loaders: list[DataLoader] = []
    for dataset in test_dataset.datasets:
        test_loaders.append(DataLoader(
            dataset, config.trainer.batch_size,
            sampler=DistributedSampler(dataset, shuffle=False),
            num_workers=int(not args.debug),
            pin_memory=True, persistent_workers=not args.debug
        ))
    if rank0: os.makedirs(os.path.join(args.output, 'test'), exist_ok=True)
    with torch.autocast(DEVICE_TYPE, dtype), ThreadPoolExecutor(4) as executor:
        for loader in test_loaders:
            test_epoch(model, loader, config.trainer.test, args.output, executor)
    dist.destroy_process_group()


if __name__ == '__main__': main()