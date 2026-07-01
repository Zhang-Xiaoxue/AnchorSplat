import torch
from torch import nn, Tensor
from torch.nn import functional as F

RADIUS = 10

def contract(x: torch.Tensor):
    x_norm = x.norm(dim=-1, keepdim=True, dtype=x.dtype)
    inv_x_norm = RADIUS * x_norm.clamp(RADIUS).reciprocal_()
    x_warp = inv_x_norm * (2 - inv_x_norm) * x
    return x.lerp(x_warp, x_norm.gt_(RADIUS))

def inv_contract(x: torch.Tensor):
    x_norm = x.norm(dim=-1, keepdim=True, dtype=x.dtype)
    x_unit = F.normalize(x, dim=-1).to(x.dtype)
    x_warp = RADIUS ** 2 * x_unit / (2 * RADIUS - x_norm).clamp(.01)
    return x.lerp(x_warp, x_norm.gt_(RADIUS))


def depth_to_world(depth: Tensor, intrinsics: Tensor, extrinsics: Tensor) -> Tensor:
    '''
    Args:
        depth (Tensor (B, V, H, W)): initial depth map prediction
        intrinsics (Tensor (B, V, 3, 3)): normalized intrinsics (0 < u,v < 1)
        extrinsics (Tensor (B, V, 4, 4)): normalized extrinsics
    Returns:
        points (Tensor (B, N, 3)): projected points
    '''
    # Project depth maps to world space with normalized intrinsics and extrinsics
    B, V, H, W = depth.shape
    x = torch.arange(W, device=depth.device, dtype=depth.dtype) / W
    y = torch.arange(H, device=depth.device, dtype=depth.dtype) / H
    rays = depth.new_ones(B, V, H, W, 4)
    rays[..., 0] = (x - intrinsics[..., :1, 2:]) / intrinsics[..., :1, :1]
    rays[..., 1] = (y.view(H, 1) - intrinsics[..., 1:2, 2:]) / intrinsics[..., 1:2, 1:2]
    rays[..., :3] *= depth[..., None]
    # Only applicable to orthonormal rotation matrix with RR^T=I
    # points = (rays.flatten(2, 3) - extrinsics[..., None, :3, 3]) @ extrinsics[..., :3, :3]
    points = torch.einsum('bvij,bvhwj->bvhwi', extrinsics.float().inverse().to(rays.dtype)[:, :, :3], rays)
    return points.flatten(1, 3)


def aggregate_feature(anchors: Tensor, feature: Tensor, depth: Tensor, intrinsics: Tensor,
                      extrinsics: Tensor, depth_tol: float = .05) -> Tensor:
    '''
    Args:
        anchors (Tensor (B, N, 3)):
        feature (Tensor (B, V, H, W, D)):
        depth (Tensor (B, V, H, W)):
        intrinsics (Tensor (B, V, 3, 3)):
        extrinsics (Tensor (B, V, 4, 4)):
        depth_tol (float):
    Returns:
        feature (Tensor (B, N, D)):
    '''
    coords = torch.cat((anchors, torch.ones_like(anchors[..., :1])), 2)                 # (B, N, 4)
    coords = torch.einsum('bvij,bnj->bvni', intrinsics @ extrinsics[:, :, :3], coords)  # (B, V, N, 3)
    coords = coords[..., :2] / (depth_proj := coords[..., 2:]).clamp(1e-3)              # (B, V, N, 2)
    coords_samp = (2 * coords - 1).float().view(-1, 1, coords.shape[2], 2)              # (BV, 1, N, 2)
    depth_samp = depth.flatten(0, 1)[:, None].float()                                   # (BV, 1, H, W)
    depth_samp = F.grid_sample(depth_samp, coords_samp, 'nearest').to(depth.dtype)      # (BV, 1, 1, N)
    depth_diff = depth_samp.view_as(depth_proj) - depth_proj                            # (B, V, N, 1)
    valid = depth_proj.gt(0) & depth_diff.abs().lt(depth_tol)                           # (B, V, N, 1)
    valid = ((coords.ge(0) & coords.le(1)).all(3, True) & valid).to(feature.dtype)      # (B, V, N, 1)
    feature_samp = feature.flatten(0, 1).movedim(3, 1).float()                          # (BV, D, h, w)
    feature = F.grid_sample(feature_samp, coords_samp).to(feature.dtype)                # (BV, D, 1, N)
    feature = feature.transpose(1, 3).reshape(*valid.shape[:3], -1)                     # (B, V, N, D)
    feature = (feature * valid).sum(1) / valid.sum(1).clamp(1)                          # (B, N, D)
    return feature


class VoxelSampler(nn.Module):
    def __init__(self, num_points: int | list[int] | tuple[int], init_voxel_size: float,
                 scale_factor: float, max_ratio: float, max_iters: int,
                 depth_tol: float = .05, contract_scene: bool = False):
        '''
        Args:
            num_points (int | list[int] | tuple[int]): number of anchor points to sample;
                enable multi-scale sampling if provided as a sequence
            init_voxel_size (float): initial value for voxel size search
            scale_factor (float): minimal voxel size multiplier or divisor `scale_factor > 1` during search
            max_ratio (float): maximal ratio of sampled voxels before random dropout
            max_iters (int): maximal iterations for voxel size search if `num_points` is not reached
            depth_tol (float): determine visibility according to difference between pixel and anchor depth
            contract_scene (bool): whether to contract unbound point cloud into unit ball before sampling
        '''
        super().__init__()
        self.num_points = [num_points] if isinstance(num_points, int) else num_points
        self.init_voxel_size = init_voxel_size
        self.factor = scale_factor
        self.max_ratio = max_ratio
        self.max_iters = max_iters
        self.depth_tol = depth_tol
        self.contract_scene = contract_scene

    @torch.inference_mode()
    def hash(self, points: Tensor, voxel_size: float) -> Tensor:
        indices = (points / voxel_size).round().long() & 0x1FFFFF  # 21-bit signed int
        return indices[:, 0] << 42 | indices[:, 1] << 21 | indices[:, 2]  # 63-bit packed int

    @torch.inference_mode()
    def unhash(self, indices: Tensor) -> Tensor:
        return torch.stack((indices << 1, indices << 22, indices << 43), 1) >> 43

    @torch.inference_mode()
    def search_voxel_size(self, points: Tensor, n_points: int) -> float:
        '''
        Args:
            points (Tensor (N, 3)): points to sample
            n_points (int): desired number of anchors
        Returns:
            voxel_size (float): voxel size for downsampling
        '''
        voxel_size, prev_ratio = self.init_voxel_size, torch.nan
        for _ in range(self.max_iters):
            ratio = self.hash(points, voxel_size).unique(False).shape[0] / n_points
            if 1 <= ratio <= self.max_ratio or ratio <= prev_ratio < 1: break
            voxel_size *= min(ratio ** .5, 1 / self.factor) if ratio < 1 else max(ratio ** .5, self.factor)
            prev_ratio = ratio
        return voxel_size

    def sample(self, points: Tensor, n_points: int, voxel_size: float) -> Tensor:
        '''
        Args:
            points (Tensor (N, 3)):
            n_points (int):
            voxel_size (float):
        Returns:
            anchors (Tensor (K, 3)):
        '''
        indices = self.unhash(self.hash(points, voxel_size).unique(False))
        anchors = indices.to(points.dtype).mul_(voxel_size)
        choice = lambda x, n: torch.randperm(x.shape[0], device=x.device)[:n]
        if anchors.shape[0] >= n_points:
            samples = choice(anchors, n_points)
            return anchors[samples]  #, indices[samples]
        samples = choice(points, n_points - anchors.shape[0])
        # indices_sampled = (points[samples] / voxel_size).round().long()
        return torch.cat((anchors, points[samples]))  #, torch.cat((indices, indices_sampled))

    def sample_batch(self, points: Tensor, n_points: int) -> tuple[Tensor, Tensor]:
        '''
        Args:
            points (Tensor (B, N, 3)):
            n_points (int):
        Returns:
            (anchors, voxel_sizes) (Tensor (B, K, 3), (B,)):
        '''
        anchors = []
        voxel_sizes = []
        for i in range(points.shape[0]):
            voxel_sizes.append(self.search_voxel_size(points[i], n_points))
            anchors.append(self.sample(points[i], n_points, voxel_sizes[-1]))
        anchors = torch.stack(anchors)
        return anchors, anchors.new_tensor(voxel_sizes)

    def forward(self, depth: Tensor, feature: Tensor, intrinsics: Tensor, extrinsics: Tensor) -> tuple[Tensor, ...]:
        '''
        Args:
            depth (Tensor (B, V, H, W)): initial depth map prediction
            feature (Tensor (B, V, H/P, W/P, D)): depth net backbone feature
            intrinsics (Tensor (B, V, 3, 3)): normalized intrinsics (0 < u,v < 1)
            extrinsics (Tensor (B, V, 4, 4)): normalized extrinsics
        Returns:
            (anchors, feature, voxel_sizes) (Tensor (B, N, 3), (B, N, D), (B,)):
                sampled anchors, feature and voxel sizes
        '''
        points = depth_to_world(depth, intrinsics, extrinsics)
        anchors_all = depth.new_empty(depth.shape[0], sum(self.num_points), 3)
        # indices_all = torch.empty_like(anchors_all, dtype=torch.long)
        if self.contract_scene: points = contract(points)  # points[i][depth[i].ravel() > 0]
        end = 0
        for n_points in self.num_points:
            start, end = end, end + n_points
            anchors_all[:, start:end], voxel_sizes = self.sample_batch(points, n_points)
        if self.contract_scene: anchors_all = inv_contract(anchors_all)
        feature = aggregate_feature(anchors_all, feature, depth, intrinsics, extrinsics, self.depth_tol)
        return anchors_all, feature, voxel_sizes


class ClosePackedSampler(VoxelSampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        offset = torch.eye(3, dtype=torch.long)
        self.offset = nn.Buffer(torch.cat((offset, -offset)))

    @torch.inference_mode()
    def hash(self, points: Tensor, voxel_size: float) -> Tensor:
        points = points / voxel_size
        rounded = points.round()
        indices = rounded.long()
        # Keep cube corners and face centers with 2 | i + j + k; filter edge centers
        odd = indices & 1
        odd = odd[:, 0] ^ odd[:, 1] ^ odd[:, 2]
        # Determine closest FCC lattice based on decimal parts of scaled coordinates
        offset = points - rounded
        offset = torch.cat((offset, -offset), 1)
        indices += odd[:, None] & self.offset[offset.argmax(1)]
        indices &= 0x1FFFFF
        return indices[:, 0] << 42 | indices[:, 1] << 21 | indices[:, 2]


class ConsistentVoxelSamplerV2(VoxelSampler):
    def __init__(
        self,
        *args,
        # --- multi-view consistency ---
        min_views: int = 2,
        max_mismatch: int = 1,
        depth_tol_rel: float = 0.01,
        consistency_k: int | None = 6,      # V=16 推荐 6~8
        align_corners: bool = False,
        mask_depth_mode: str = "nearest",   # "nearest" or "bilinear"

        # --- mixed sampling ---
        good_ratio: float = 0.90,           # 基础比例（当 adaptive_mix=False 时固定）
        bad_weight_power: float = 2.0,      # support 加权指数
        max_candidates: int = 200000,       # 限制候选点规模（每个池）
        fallback_min_points: int = 20000,   # good 池太小触发 fallback

        # --- adaptive mix（推荐开）---
        adaptive_mix: bool = True,
        min_good_ratio: float = 0.60,
        max_good_ratio: float = 0.95,
        target_keep_ratio: float = 0.50,    # keep_frac=0.5 时比例约为 good_ratio

        # --- logging / coverage ---
        coverage_voxel_size: float = 0.20,  # 覆盖率统计体素大小（米）
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.min_views = int(min_views)
        self.max_mismatch = int(max_mismatch)
        self.depth_tol_rel = float(depth_tol_rel)
        self.consistency_k = consistency_k
        self.align_corners = bool(align_corners)
        self.mask_depth_mode = str(mask_depth_mode)

        self.good_ratio = float(good_ratio)
        self.bad_weight_power = float(bad_weight_power)
        self.max_candidates = int(max_candidates)
        self.fallback_min_points = int(fallback_min_points)

        self.adaptive_mix = bool(adaptive_mix)
        self.min_good_ratio = float(min_good_ratio)
        self.max_good_ratio = float(max_good_ratio)
        self.target_keep_ratio = float(target_keep_ratio)

        self.coverage_voxel_size = float(coverage_voxel_size)
        self.verbose = bool(verbose)
        # print(f'[CONFIG] ConsistentVoxelSamplerV2: {self.__dict__}')

    # -------------------------
    # utils
    # -------------------------
    def _tol_of(self, z: Tensor) -> Tensor:
        # tol(z) = abs + rel*z
        return z.new_tensor(self.depth_tol) + self.depth_tol_rel * z.abs()

    @staticmethod
    def _rand_choice_idx(n: int, k: int, device) -> Tensor:
        if k <= 0:
            return torch.empty((0,), dtype=torch.long, device=device)
        if n <= k:
            return torch.arange(n, device=device)
        return torch.randperm(n, device=device)[:k]

    @staticmethod
    def _weighted_choice_idx(w: Tensor, k: int) -> Tensor:
        if k <= 0:
            return torch.empty((0,), dtype=torch.long, device=w.device)
        if w.numel() <= k:
            return torch.arange(w.numel(), device=w.device)
        w = w.float().clamp_min(0)
        if torch.all(w <= 0):
            return torch.randperm(w.numel(), device=w.device)[:k]
        return torch.multinomial(w, k, replacement=False)

    @torch.no_grad()
    def _camera_centers(self, extrinsics: Tensor) -> Tensor:
        # extrinsics: world->cam, camera center in world: C = -R^T t
        R = extrinsics[..., :3, :3]
        t = extrinsics[..., :3, 3]
        C = -torch.einsum("bvij,bvj->bvi", R.transpose(-1, -2), t)
        return C

    # 更靠谱的 voxel hash（降低碰撞）
    @torch.no_grad()
    def _hash_vox(self, vox: torch.Tensor) -> torch.Tensor:
        v = vox.to(torch.int64)
        return (v[:, 0] * 73856093) ^ (v[:, 1] * 19349663) ^ (v[:, 2] * 83492791)

    @torch.no_grad()
    def _voxel_unique_count(self, points: Tensor, vs: float) -> int:
        if points.numel() == 0:
            return 0
        vox = torch.round(points / vs).to(torch.int64)
        key = self._hash_vox(vox)
        return int(key.unique().numel())

    @torch.no_grad()
    def _voxel_representative_indices(self, points: Tensor, vs: float) -> Tensor:
        """
        points: (N,3)（一般传 points.detach()）
        返回：每个 voxel 选一个代表点的 index
        """
        if points.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=points.device)
        vox = torch.round(points / vs).to(torch.int64)
        key = self._hash_vox(vox)
        sk, order = torch.sort(key)
        first = torch.ones_like(sk, dtype=torch.bool)
        first[1:] = sk[1:] != sk[:-1]
        return order[first]

    @torch.no_grad()
    def search_voxel_size(self, points: Tensor, n_points: int) -> float:
        """
        override：用更稳的 hash 做 unique 计数，避免原 hash 碰撞导致 ratio 失真
        """
        voxel_size, prev_ratio = float(self.init_voxel_size), float("nan")
        for _ in range(self.max_iters):
            vox = torch.round(points / voxel_size).to(torch.int64)
            ratio = self._hash_vox(vox).unique().numel() / max(n_points, 1)
            if (1 <= ratio <= self.max_ratio) or (ratio <= prev_ratio < 1):
                break
            voxel_size *= min(ratio ** 0.5, 1 / self.factor) if ratio < 1 else max(ratio ** 0.5, self.factor)
            prev_ratio = ratio
        return voxel_size

    # -------------------------
    # multi-view consistency (no_grad)
    # -------------------------
    @torch.no_grad()
    def multiview_consistency_mask(
        self,
        points_world: Tensor,   # (B,V,H,W,3)
        depth: Tensor,          # (B,V,H,W) meters
        intrinsics: Tensor,     # (B,V,3,3) normalized 0~1
        extrinsics: Tensor,     # (B,V,4,4) world->cam
    ) -> tuple[Tensor, Tensor]:
        B, V, H, W = depth.shape
        device = depth.device

        # choose neighbor views
        if self.consistency_k is None or self.consistency_k >= V:
            nbr_idx = torch.arange(V, device=device)[None, None, :].expand(B, V, V)  # (B,V,V)
        else:
            k = int(self.consistency_k)
            C = self._camera_centers(extrinsics.float())  # (B,V,3)
            dist = (C[:, :, None, :] - C[:, None, :, :]).pow(2).sum(-1)  # (B,V,V)
            nbr_idx = dist.topk(k, dim=-1, largest=False).indices        # (B,V,k)

        P = (intrinsics @ extrinsics[:, :, :3]).float()  # (B,V,3,4)

        keep = torch.zeros((B, V, H, W), device=device, dtype=torch.bool)
        support = torch.zeros((B, V, H, W), device=device, dtype=torch.int16)

        ones = torch.ones((B, H, W, 1), device=device, dtype=points_world.dtype)

        for s in range(V):
            Xw = points_world[:, s]                 # (B,H,W,3)
            Xh = torch.cat([Xw, ones], dim=-1)      # (B,H,W,4)

            idx = nbr_idx[:, s]                     # (B,K)
            K = idx.shape[-1]

            P_t = P.gather(1, idx[:, :, None, None].expand(B, K, 3, 4))      # (B,K,3,4)
            d_t = depth.gather(1, idx[:, :, None, None].expand(B, K, H, W)).float()

            proj = torch.einsum("bkij,bhwj->bkhwi", P_t, Xh.float())          # (B,K,H,W,3)
            z = proj[..., 2].clamp_min(1e-6)                                 # (B,K,H,W)
            uv = proj[..., :2] / z[..., None]                                # (B,K,H,W,2) 0~1

            grid = (uv * 2.0 - 1.0).reshape(B * K, H, W, 2)                  # (B*K,H,W,2)
            d_in = d_t.reshape(B * K, 1, H, W)
            d_samp = F.grid_sample(
                d_in, grid,
                mode=self.mask_depth_mode,            # nearest/bilinear
                padding_mode="zeros",
                align_corners=self.align_corners
            ).reshape(B, K, H, W)

            in_frame = (uv >= 0).all(-1) & (uv <= 1).all(-1) & (z > 0) & (d_samp > 0)

            tol = self._tol_of(z).to(d_samp.dtype)
            diff = d_samp - z.to(d_samp.dtype)

            # occlusion-aware: target 更近 => 被遮挡 => 忽略该视角，不计 mismatch
            occluded = in_frame & (d_samp < (z.to(d_samp.dtype) - tol))
            visible = in_frame & (~occluded)

            match = visible & (diff.abs() < tol)
            mismatch = visible & (~match)

            match_cnt = match.sum(dim=1)
            mismatch_cnt = mismatch.sum(dim=1)

            keep[:, s] = (match_cnt >= self.min_views) & (mismatch_cnt <= self.max_mismatch)
            support[:, s] = match_cnt.to(torch.int16)

        return keep, support

    # -------------------------
    # forward (needs grads for anchors/feat)
    # -------------------------
    def forward(self, depth: Tensor, feature: Tensor, intrinsics: Tensor, extrinsics: Tensor):
        """
        不用 inference_mode：允许 anchors/feature 相关梯度回传。
        mask/投票/离散索引选择都在 no_grad（选择不可导，但选中的点坐标仍可导）。
        """
        B, V, H, W = depth.shape
        device, dtype = depth.device, depth.dtype

        # pixel center in 0~1
        x = (torch.arange(W, device=device, dtype=dtype) + 0.5) / W
        y = (torch.arange(H, device=device, dtype=dtype) + 0.5) / H

        rays = depth.new_ones(B, V, H, W, 4)
        rays[..., 0] = (x - intrinsics[..., :1, 2:]) / intrinsics[..., :1, :1]
        rays[..., 1] = (y.view(H, 1) - intrinsics[..., 1:2, 2:]) / intrinsics[..., 1:2, 1:2]
        rays[..., :3] *= depth[..., None]

        # cam->world（extrinsics 是 world->cam）
        c2w = extrinsics.float().inverse().to(dtype)[:, :, :3]
        points_world = torch.einsum('bvij,bvhwj->bvhwi', c2w, rays)  # ✅对 depth 可导

        with torch.no_grad():
            keep_mask, support = self.multiview_consistency_mask(points_world, depth, intrinsics, extrinsics)

        points_flat = points_world.flatten(1, 3)                  # (B, V*H*W, 3) ✅可导
        keep_flat = keep_mask.flatten(1, 3)                       # (B, N) no_grad
        supp_flat = support.flatten(1, 3).to(torch.int16)         # (B, N) no_grad
        valid_flat = (depth.flatten(1, 3) > 0)                    # (B, N)

        total_N = sum(self.num_points)
        anchors_all = depth.new_empty((B, total_N, 3))
        voxel_sizes = depth.new_empty(B)

        # rank info（DDP 时 b 永远 0 很正常，加 rank 才看得懂）
        rank, world = 0, 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world = torch.distributed.get_world_size()

        for i in range(B):
            pts = points_flat[i]          # (N,3) ✅
            valid = valid_flat[i]
            good_m = keep_flat[i] & valid
            bad_m = (~keep_flat[i]) & valid

            n_valid = int(valid.sum().item())
            n_good_full = int(good_m.sum().item())
            keep_ratio = (n_good_full / max(n_valid, 1)) * 100.0

            # full pools (for stats)
            pts_valid_full = pts[valid]
            pts_good_full = pts[good_m]
            pts_bad_full = pts[bad_m]

            # sampling pools (may be truncated)
            pts_good = pts_good_full
            pts_bad = pts_bad_full
            supp_bad = supp_flat[i][bad_m].float()

            # truncate candidates for speed
            if self.max_candidates > 0 and pts_good.shape[0] > self.max_candidates:
                with torch.no_grad():
                    idx = self._rand_choice_idx(pts_good.shape[0], self.max_candidates, device)
                pts_good = pts_good[idx]

            if self.max_candidates > 0 and pts_bad.shape[0] > self.max_candidates:
                with torch.no_grad():
                    w = (supp_bad + 1.0).pow(self.bad_weight_power)
                    idx = self._weighted_choice_idx(w, self.max_candidates)
                pts_bad = pts_bad[idx]
                supp_bad = supp_bad[idx]

            # contract (optional)
            if self.contract_scene:
                pts_good_c = contract(pts_good)
                pts_bad_c = contract(pts_bad)
                pts_all_c = contract(pts_valid_full)
            else:
                pts_good_c, pts_bad_c, pts_all_c = pts_good, pts_bad, pts_valid_full

            end = 0
            scale_logs = []  # (n_points, n_good, n_bad, rep_cnt, vs, fallback, good_ratio_eff)
            anchors_g_list, anchors_b_list = [], []

            for n_points in self.num_points:
                start, end = end, end + n_points

                # fallback when good pool is too small
                if pts_good_c.shape[0] < max(self.fallback_min_points, n_points // 2):
                    with torch.no_grad():
                        vs = self.search_voxel_size(pts_all_c.detach(), n_points)
                        rep = self._voxel_representative_indices(pts_all_c.detach(), vs)
                        rep_cnt = int(rep.numel())
                        if rep.numel() >= n_points:
                            sel = rep[self._rand_choice_idx(rep.numel(), n_points, device)]
                        else:
                            extra = self._rand_choice_idx(pts_all_c.shape[0], n_points - rep.numel(), device)
                            sel = torch.cat([rep, extra], 0)

                    anchors = pts_all_c[sel]  # ✅对 depth 可导
                    anchors_all[i, start:end] = anchors[:n_points]
                    scale_logs.append((n_points, int(anchors.shape[0]), 0, rep_cnt, float(vs), True, 0.0))

                    # anchors 已经是这个尺度选出来的点
                    if self.contract_scene:
                        anchors_g_list.append(inv_contract(anchors).detach())
                    else:
                        anchors_g_list.append(anchors.detach())
                    anchors_b_list.append(anchors.new_empty((0, 3)))

                    continue

                # -------- adaptive good/bad ratio --------
                keep_frac = n_good_full / max(n_valid, 1)  # 0~1
                if self.adaptive_mix:
                    # keep_frac 低 => 更需要 bad 来补覆盖
                    ratio_eff = self.good_ratio * (keep_frac / max(self.target_keep_ratio, 1e-6))
                    ratio_eff = float(max(self.min_good_ratio, min(self.max_good_ratio, ratio_eff)))
                else:
                    ratio_eff = self.good_ratio

                n_good_target = int(round(n_points * ratio_eff))
                n_good = min(n_good_target, pts_good_c.shape[0])
                n_bad = n_points - n_good

                with torch.no_grad():
                    vs = self.search_voxel_size(pts_good_c.detach(), max(n_good, 1))
                    rep = self._voxel_representative_indices(pts_good_c.detach(), vs)
                    rep_cnt = int(rep.numel())
                    if rep.numel() >= n_good:
                        sel_g = rep[self._rand_choice_idx(rep.numel(), n_good, device)]
                    else:
                        extra = self._rand_choice_idx(pts_good_c.shape[0], n_good - rep.numel(), device)
                        sel_g = torch.cat([rep, extra], 0)

                anchors_g = pts_good_c[sel_g] if n_good > 0 else pts_good_c.new_empty((0, 3))

                if n_bad > 0 and pts_bad_c.shape[0] > 0:
                    with torch.no_grad():
                        w = (supp_bad + 1.0).pow(self.bad_weight_power)
                        sel_b = self._weighted_choice_idx(w, min(n_bad, pts_bad_c.shape[0]))
                    anchors_b = pts_bad_c[sel_b]
                else:
                    anchors_b = pts_good_c.new_empty((0, 3))

                anchors = torch.cat([anchors_g, anchors_b], 0)
                if self.contract_scene:
                    anchors_g_list.append(inv_contract(anchors_g).detach())
                    anchors_b_list.append(inv_contract(anchors_b).detach())
                else:
                    anchors_g_list.append(anchors_g.detach())
                    anchors_b_list.append(anchors_b.detach())

                # still not enough -> pad from all valid
                if anchors.shape[0] < n_points:
                    need = n_points - anchors.shape[0]
                    with torch.no_grad():
                        extra = self._rand_choice_idx(pts_all_c.shape[0], need, device)
                    anchors = torch.cat([anchors, pts_all_c[extra]], 0)

                anchors_all[i, start:end] = anchors[:n_points]
                scale_logs.append((n_points, int(anchors_g.shape[0]), int(anchors_b.shape[0]),
                                   rep_cnt, float(vs), False, ratio_eff))

            if self.contract_scene:
                anchors_all[i] = inv_contract(anchors_all[i])

            # # coverage stats
            # with torch.no_grad():
            #     anc_all = anchors_all[i].detach()
            #     anc_g = torch.cat(anchors_g_list, 0) if len(anchors_g_list) else anc_all.new_empty((0, 3))
            #     anc_b = torch.cat(anchors_b_list, 0) if len(anchors_b_list) else anc_all.new_empty((0, 3))

            #     cand_valid_u = self._voxel_unique_count(pts_valid_full.detach(), self.coverage_voxel_size)
            #     cand_good_u  = self._voxel_unique_count(pts_good_full.detach(),  self.coverage_voxel_size)

            #     anc_u   = self._voxel_unique_count(anc_all, self.coverage_voxel_size)
            #     anc_g_u = self._voxel_unique_count(anc_g,   self.coverage_voxel_size)
            #     anc_b_u = self._voxel_unique_count(anc_b,   self.coverage_voxel_size)

            #     cov_valid     = anc_u   / max(cand_valid_u, 1) * 100.0   # 总 anchors 覆盖所有 valid 候选（<=100）
            #     cov_good_only = anc_g_u / max(cand_good_u,  1) * 100.0   # 仅 good anchors 覆盖 good 候选（<=100）
            #     bad_contrib   = anc_b_u / max(cand_valid_u, 1) * 100.0   # bad anchors 对 valid 的覆盖贡献（<=100）
            #     share_bad     = anc_b_u / max(anc_u, 1) * 100.0          # bad 在最终覆盖中的占比（<=100）

            # if self.verbose:
            #     print(f"[ConsistentVoxelSamplerV2][rank {rank}/{world}] b={i} "
            #           f"keep_ratio={keep_ratio:.2f}% "
            #           f"cov_valid@{self.coverage_voxel_size:.2f}m={cov_valid:.2f}% "
            #           f"cov_good_only@{self.coverage_voxel_size:.2f}m={cov_good_only:.2f}% "
            #           f"bad_contrib={bad_contrib:.2f}% share_bad={share_bad:.2f}% "
            #           f"good_pool={pts_good_full.shape[0]} bad_pool={pts_bad_full.shape[0]} valid_pool={pts_valid_full.shape[0]}")
            #     for (npt, ng, nb, rep_cnt, vs, fb, r_eff) in scale_logs:
            #         if fb:
            #             print(f"  - N={npt}: vs={vs:.4f} rep={rep_cnt} good={ng} bad={nb} (fallback)")
            #         else:
            #             print(f"  - N={npt}: ratio_eff={r_eff:.2f} vs={vs:.4f} rep={rep_cnt} good={ng} bad={nb}")

            voxel_sizes[i] = vs

        # feature aggregation（可导）
        feat = aggregate_feature(anchors_all, feature, depth, intrinsics, extrinsics, self.depth_tol)
        return anchors_all, feat, voxel_sizes

    # -------------------------
    # visualization helpers
    # -------------------------
    # def plot_keep_mask_grid(
    #     self,
    #     keep_mask: torch.Tensor,
    #     support: torch.Tensor | None = None,
    #     b: int = 0,
    #     save_path: str | None = None,
    #     max_cols: int = 4
    # ):
    #     """
    #     keep_mask: (B,V,H,W) bool
    #     support:   (B,V,H,W) int (optional)
    #     """
    #     km = keep_mask[b].detach().cpu().numpy().astype(np.float32)  # (V,H,W)
    #     V = km.shape[0]
    #     cols = max_cols
    #     rows = (V + cols - 1) // cols

    #     fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    #     axes = np.array(axes).reshape(-1)

    #     for i in range(rows * cols):
    #         ax = axes[i]
    #         ax.axis("off")
    #         if i >= V:
    #             continue

    #         ax.imshow(km[i], vmin=0, vmax=1)
    #         keep_ratio = km[i].mean() * 100.0
    #         title = f"view {i}  keep={keep_ratio:.2f}%"

    #         if support is not None:
    #             sp = support[b, i].detach().cpu().numpy()
    #             title += f"  sup_mean={sp.mean():.2f}"

    #         ax.set_title(title, fontsize=10)

    #     plt.tight_layout()
    #     if save_path is not None:
    #         plt.savefig(save_path, dpi=200, bbox_inches="tight")
    #     return fig

    # @torch.no_grad()
    # def compute_keep_mask_for_vis(self, depth: Tensor, intrinsics: Tensor, extrinsics: Tensor):
    #     """
    #     复现 forward 里 points_world 的构造，然后单独拿 keep_mask 做可视化
    #     """
    #     B, V, H, W = depth.shape
    #     device = depth.device
    #     dtype = depth.dtype

    #     x = (torch.arange(W, device=device, dtype=dtype) + 0.5) / W
    #     y = (torch.arange(H, device=device, dtype=dtype) + 0.5) / H

    #     rays = depth.new_ones(B, V, H, W, 4)
    #     rays[..., 0] = (x - intrinsics[..., :1, 2:]) / intrinsics[..., :1, :1]
    #     rays[..., 1] = (y.view(H, 1) - intrinsics[..., 1:2, 2:]) / intrinsics[..., 1:2, 1:2]
    #     rays[..., :3] *= depth[..., None]

    #     c2w = extrinsics.float().inverse().to(dtype)[:, :, :3]
    #     points_world = torch.einsum('bvij,bvhwj->bvhwi', c2w, rays)

    #     keep_mask, support = self.multiview_consistency_mask(points_world, depth, intrinsics, extrinsics)
    #     return keep_mask, support


# class TSDFSampler(VoxelSampler):
#     def __init__(self, num_points: int | list[int] | tuple[int], init_voxel_size: float, scale_factor: float,
#                  max_ratio: float, max_iters: int, trunc_voxels: float, thresh_voxels: float):
#         '''
#         Args:
#             num_points (int | list[int] | tuple[int]): number of anchor points to sample;
#                 enable multi-scale sampling if provided as a sequence
#             init_voxel_size (float): initial value for voxel size search
#             scale_factor (float): minimal voxel size multiplier or divisor `scale_factor > 1` during search
#             max_ratio (float): maximal ratio of sampled voxels before random dropout
#             max_iters (int): maximal iterations for voxel size search if `num_points` is not reached
#             trunc_voxels (float): truncated number of voxels for SDF,
#                 `trunc_dist = trunc_voxels * voxel_size`
#             thresh_voxels (float): threshold number of voxels for surface extraction,
#                 `thresh_dist = thresh_voxels * voxel_size`
#         '''
#         super().__init__(num_points, init_voxel_size, scale_factor, max_ratio, max_iters)
#         self.trunc_voxels = trunc_voxels
#         self.thresh_voxels = thresh_voxels

#     def sample(self, points: Tensor, depth: Tensor, intrinsics: Tensor, extrinsics: Tensor,
#                bbox_min: Tensor, bbox_max: Tensor, voxel_size: float) -> Tensor:
#         '''
#         Args:
#             depth (Tensor (V, H, W)):
#             intrinsics (Tensor (V, 3, 3)):
#             extrinsics (Tensor (V, 4, 4)):
#             bbox_min (Tensor (3)):
#             bbox_max (Tensor (3)):
#             voxel_size (float):
#         Returns:
#             anchors (Tensor (K, 3)):
#         '''
#         V, H, W = depth.shape
#         grid = [torch.arange(bbox_min[i], bbox_max[i], voxel_size, dtype=depth.dtype, device=depth.device) for i in range(3)]
#         grid = torch.stack(torch.meshgrid(*grid, indexing='ij')).flatten(1)
#         coords = intrinsics @ (extrinsics[:, :3, :3] @ grid + extrinsics[:, :3, 3:])
#         coords[:, :2] /= coords[:, 2:]
#         x, y, z = (W * coords[:, 0]).floor().int().clamp(0, W - 1), (H * coords[:, 1]).floor().int().clamp(0, H - 1), coords[:, 2]
#         dist = depth[torch.arange(V, device=depth.device)[:, None].expand_as(z), y, x] - z
#         trunc_dist = self.trunc_voxels * voxel_size
#         valid = coords[:, :2].gt(0).all(1) & coords[:, :2].lt(1).all(1) & z.gt(trunc_dist) & dist.abs().lt(trunc_dist)
#         valid = (dist * valid).sum(0).abs() / valid.sum(0) < self.thresh_voxels * voxel_size
#         return grid[:, valid].T
