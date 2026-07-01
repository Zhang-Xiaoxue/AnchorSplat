from addict import Dict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from glob import glob
from itertools import accumulate
import numpy as np
from os import listdir
from random import choice, choices, sample
import torch
from torchvision.io import decode_image
from torchvision.transforms.v2.functional import InterpolationMode, resize


def affine_inverse(extrinsics: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(extrinsics)
    out[..., :3, :3] = RT = extrinsics[..., :3, :3].mT
    out[..., :3, 3:] = RT @ -extrinsics[..., :3, 3:]
    out[..., 3, 3] = 1
    return out


def normalize_extrinsics(extrinsics: torch.Tensor, num_views: int) -> tuple[torch.Tensor, float]:
    '''Normalize input and novel view extrinsics
    according to input view (first `num_views`) distances'''
    extrinsics = extrinsics @ affine_inverse(extrinsics[:1])
    translations = affine_inverse(extrinsics[:num_views])[:, :3, 3]
    median_dist = translations.norm(dim=1).median().clamp(.1)
    extrinsics[:, :3, 3] /= median_dist
    return extrinsics


class BaseDataset:
    def __init__(self, name: str, roots: list[str], num_views: int, resolution: tuple[int, int],
                 dtype: str, training: bool, executor: ThreadPoolExecutor):
        self.name = name
        self.roots = roots
        self.num_views = num_views
        self.resolution = resolution
        self.dtype = torch.half if dtype == 'fp16' \
            else torch.bfloat16 if dtype == 'bf16' else torch.float
        self.training = training
        self.executor = executor

    def __len__(self) -> int:
        return len(self.roots)

    def load_single_image(self, root: str, filename: str) -> torch.Tensor:
        image = decode_image(f'{root}/{filename}')
        image = resize(image[None], self.resolution, InterpolationMode.BILINEAR)
        return (image / 255).to(self.dtype)

    def load_single_depth(self, root: str, scale: float, filename: str) -> torch.Tensor:
        depth = decode_image(f'{root}/{filename}')
        depth = resize(depth, self.resolution, InterpolationMode.NEAREST_EXACT)
        return (1e-3 / scale * depth).to(self.dtype)

    def load_images(self, root: str, filenames: list[str]) -> list[torch.Tensor]:
        load_single_image = partial(self.load_single_image, root)
        return list(self.executor.map(load_single_image, filenames))

    def load_depths(self, root: str, scale: float, filenames: list[str]) -> list[torch.Tensor]:
        load_single_depth = partial(self.load_single_depth, root, scale)
        return list(self.executor.map(load_single_depth, filenames))

    def __getitem__(self, idx: int) -> Dict: ...


class AnchorSplatDataset(BaseDataset):
    def sample_indices(self, length: int) -> np.ndarray:
        # only support scenes with no less than (2 * self.num_views) images
        max_interval = length // self.num_views
        if not self.training:
            indices = 2 * np.arange(self.num_views)
            indices_all = np.concatenate((indices, indices + 1))
            return (length - 1 - indices_all[-1]) // 2 + indices_all
        interval = np.random.randint(min(2, max_interval), 1 + min(5, max_interval))
        indices = np.arange(0, self.num_views * interval, interval)
        indices += np.random.choice(interval // 2, self.num_views)
        indices += np.random.randint(0, length - indices[-1])
        remain = np.zeros(length, bool)
        remain[max(0, indices[0] - interval):indices[-1] + interval] = True
        remain[indices] = False
        remain = remain.nonzero()[0]
        if remain.shape[0] < self.num_views:  # add input views to novel views
            samples = np.random.choice(indices, self.num_views - remain.shape[0])
            remain = np.concatenate((remain, samples))
        indices_nv = np.random.choice(remain, self.num_views, False)
        indices_nv.sort()
        return np.concatenate((indices, indices_nv))

    def __getitem__(self, idx: int):
        root = self.roots[idx]
        viewInfo = np.load(f'{root}/viewInfo.npz')
        files = viewInfo['image_filenames']
        indices_all = self.sample_indices(files.shape[0])
        extrinsics_all = torch.tensor(viewInfo['T_w2c'][indices_all])
        extrinsics_all = normalize_extrinsics(extrinsics_all, self.num_views)

        images_all = self.load_images(f'{root}/images', files[indices_all])
        indices, indices_nv = np.split(indices_all, [self.num_views])
        data = Dict(
            dataset=self.name,
            scene=root.split('/')[-1],
            intrinsics=torch.tensor(viewInfo['K_norm'][indices]).to(self.dtype),
            extrinsics=extrinsics_all[:self.num_views].to(self.dtype),
            images=torch.concatenate(images_all[:self.num_views]),
            intrinsics_nv=torch.tensor(viewInfo['K_norm'][indices_nv]).to(self.dtype),
            extrinsics_nv=extrinsics_all[self.num_views:].to(self.dtype),
            images_nv=torch.concatenate(images_all[self.num_views:])
        )
        if not self.training:  # load ground truth depth for validation
            pass
            # depth_all = self.load_depths(...)
            # data.update(
            #     depth=torch.concatenate(depth_all[:self.num_views]),
            #     depth_nv=torch.concatenate(depth_all[self.num_views:])
            # )
        return data


class Reliev3RDataset(BaseDataset):
    def __getitem__(self, idx: int):
        # sample dataset and scene
        while True:
            try:
                root = self.roots[idx]
                scene = root.split('/')[-1]
                image_root = f'{root}/rgb'
                files = sorted(listdir(image_root))
                length = len(files)
                info = np.load(f'{root}/{scene}.npy', allow_pickle=True).item()
                assert length == info['intr_mat'].shape[0] >= 2 * self.num_views
                break
            except: idx = (idx + 1) % len(self.roots)

        # sample frames
        if self.training:  # random
            frames = sample(range(length), 2 * self.num_views)
            frames = sorted(frames[:self.num_views]) + sorted(frames[self.num_views:])
        else:  # deterministic
            interval = length // (2 * self.num_views)
            frames = list(range(length))[:2 * interval * self.num_views:interval]
            frames = frames[::2] + frames[1::2]
        files = [files[i] for i in frames]

        # load data
        images_all = self.load_images(image_root, files)
        intrinsics = torch.tensor(info['intr_mat'][frames])
        extrinsics = torch.tensor(info['extr_mat'][frames])
        if self.name == 'dl3dv':  # opengl to opencv
            T_row = extrinsics.new_tensor(
                ((0, 1, 0, 0), (1, 0, 0, 0), (0, 0, -1, 0), (0, 0, 0, 1)))
            T_col = extrinsics.new_tensor(
                ((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0), (0, 0, 0, 1)))
            extrinsics = T_row @ extrinsics @ T_col
        extrinsics = affine_inverse(extrinsics)  # c2w to w2c
        extrinsics = normalize_extrinsics(extrinsics, self.num_views)
        data = Dict(
            dataset=self.name,
            scene=scene,
            intrinsics=intrinsics[:self.num_views].to(self.dtype),
            extrinsics=extrinsics[:self.num_views].to(self.dtype),
            images=torch.concatenate(images_all[:self.num_views]),
            intrinsics_nv=intrinsics[self.num_views:].to(self.dtype),
            extrinsics_nv=extrinsics[self.num_views:].to(self.dtype),
            images_nv=torch.concatenate(images_all[self.num_views:])
        )
        return data


class MixedDataset(torch.utils.data.Dataset):
    def __init__(self, datasets: dict[str, dict[str, str | float]], num_views: int,
                 resolution: tuple[int, int], dtype: str, training: bool):
        super().__init__()
        self.executor = ThreadPoolExecutor(2 * num_views)
        self.datasets: list[BaseDataset] = []
        datasets = {k: Dict(v) for k, v in datasets.items()}
        for name, config in datasets.items():
            if config.type == 'anchorsplat':
                filename = f'train/data/{name}_{"train" if training else "val"}.txt'
                with open(filename) as f: scenes = f.read().split()
                roots = [f'{config.root}/{s}' for s in scenes]
                dataset_class = AnchorSplatDataset
            elif config.type == 'reliev3r':
                roots = sorted(glob(f'{config.root}/sub_*/*'))
                size = round(len(roots) * config.val_ratio)
                roots = roots[:-size] if training else roots[-size:]
                dataset_class = Reliev3RDataset
            else: raise NotImplementedError
            self.datasets.append(dataset_class(
                name, roots, num_views, resolution, dtype, training, self.executor))
        self.cum_lengths = list(accumulate(len(d) for d in self.datasets))
        self.weights = list(d.weight for d in datasets.values())
        self.cum_weights = list(accumulate(self.weights))

    def __len__(self) -> int:
        return 8192

    def __getitem__(self, idx: int):
        dataset = choices(self.datasets, cum_weights=self.cum_weights)[0]
        return choice(dataset)
