# datasets/SuperquadricSDFDataset.py

import os
from typing import Tuple, List

import numpy as np
import torch
import torch.utils.data as data

from .io import IO
from .build import DATASETS
from utils.logger import *

# Import your SQ implicit tools
from procedural_data_gen.sq_implicit import (
    global_bounding_box_for_all_SQs,
    multi_sq_implicit_union,
    sample_grid_in_box,
)


@DATASETS.register_module()
class SuperquadricSDFDataset(data.Dataset):
    """
    Dataset for superquadric-based point clouds with SDF supervision.

    It assumes the dataset was generated with `generate_pointzero_like_dataset`
    in `mode="enriched"` and `storage="lmdb"`, i.e.:

      - main key:           "<key>"            -> (N, 3) float32 points
      - sidecar sq params:  "<key>.sq_params"  -> (S, 11) float32 superquadric params
      - (optionally labels: "<key>.labels"    -> (N,) int32 SQ indices per point)

    For now, SDF query points are sampled on a regular 3D grid inside a global
    bounding box of the union of all SQs in a cloud.

    __getitem__ returns:
        taxonomy_id: str  (currently "random" in your SQ generator)
        model_id:    str  (e.g. "train:00000042")
        payload:     tuple:
                        (points, query_points, sdf_values, grid_shape)

        where:
          - points:        (P, 3) float32 tensor, normalized (same as ZeroVerse)
          - query_points:  (M, 3) float32 tensor (grid flattened)
          - sdf_values:    (M,)   float32 tensor, union field (f-1), <0 inside, >0 outside
          - grid_shape:    (3,)   long tensor [nx, ny, nz] (so you can reshape if needed)
    """

    def __init__(self, config):
        # --- common config fields (mirrors ZeroVerse) ---
        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.use_lmdb = config.USE_LMDB
        self.subset = config.subset

        # How many points to sample from the cloud (Point-MAE input)
        # ZeroVerse uses both N_POINTS (upper) and npoints (lower); we mirror that.
        self.npoints = getattr(config, "N_POINTS", None)
        if self.npoints is None:
            self.npoints = getattr(config, "npoints", None)
        if self.npoints is None:
            raise ValueError("SuperquadricSDFDataset: config must define N_POINTS or npoints")

        # Whether to merge train+test splits (copied from ZeroVerse)
        self.whole = getattr(config, "whole", False)

        # Sampling method for SDF queries: currently only 'grid' implemented
        # but we expose a knob so we can add 'random', 'near_surface', ... later.
        self.sampling_method = getattr(config, "SQ_SDF_SAMPLING", "grid")

        # Grid resolution for 'grid' sampling (nx = ny = nz by default)
        self.grid_resolution = getattr(config, "SQ_SDF_GRID_RES", 32)

        # Total number of query points when we subsample / balance occupancy.
        # Default: full grid.
        self.n_query_points = getattr(
            config, "SQ_SDF_N_QUERY", self.grid_resolution ** 3
        )

        # Parameters for occupancy-balanced sampling
        # width of "near surface" band in implicit units (f-1)
        self.occ_band_width = getattr(config, "SQ_OCC_BAND", 0.1)
        # fraction of inside/outside samples taken from near-surface band
        self.occ_near_fraction = getattr(config, "SQ_OCC_NEAR_FRACTION", 0.5)
        # fraction of total points that should be inside (≈ class balance)
        self.occ_inside_fraction = getattr(config, "SQ_OCC_INSIDE_FRACTION", 0.5)

        # How we sample SDF / occupancy query points
        self.sampling_method = getattr(config, "SQ_SDF_SAMPLING", "grid")

        # For surface-jitter sampling: how many query points per surface point
        # If npoints = 1024 and jitter_factor = 8 → 1024 * 8 = 8192 queries.
        self.jitter_factor = getattr(config, "SQ_JITTER_FACTOR", 8)

        # Standard deviation of Gaussian noise for jitter (in same units as pts / SQ)
        self.jitter_sigma = getattr(config, "SQ_JITTER_SIGMA", 0.1)

        # --- build file list from {subset}.txt (same pattern as ZeroVerse) ---
        self.data_list_file = os.path.join(self.data_root, f"{self.subset}.txt")
        test_data_list_file = os.path.join(self.data_root, "test.txt")

        print_log(f"[SuperquadricSDFDataset] sample out {self.npoints} points "
                  f"for point cloud input", logger="SuperquadricSDF")

        print_log(f"[SuperquadricSDFDataset] Open file {self.data_list_file}",
                  logger="SuperquadricSDF")
        with open(self.data_list_file, "r") as f:
            lines = f.readlines()

        if self.whole:
            with open(test_data_list_file, "r") as f:
                test_lines = f.readlines()
            print_log(f"[SuperquadricSDFDataset] Open file {test_data_list_file}",
                      logger="SuperquadricSDF")
            lines = test_lines + lines

        self.file_list: List[dict] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            taxonomy_id = "random"  # same as SQ generator
            model_id = line.split(".")[0]  # "train:00000042" etc, without extension
            self.file_list.append({
                "taxonomy_id": taxonomy_id,
                "model_id": model_id,
                "file_path": line,           # e.g. "train:00000042"
            })

        print_log(f"[SuperquadricSDFDataset] {len(self.file_list)} instances were loaded",
                  logger="SuperquadricSDF")

        # Permutation buffer for random subsampling of cloud points
        self.permutation = np.arange(self.npoints)

        if not self.use_lmdb:
            raise NotImplementedError(
                "SuperquadricSDFDataset currently assumes USE_LMDB=True "
                "(enriched LMDB with .sq_params sidecar)."
            )

    # ---------- small helpers copied/adapted from ZeroVerse ----------

    def pc_norm(self, pc: np.ndarray) -> np.ndarray:
        """Normalize point cloud to zero-mean and unit sphere.

        pc: (N, C) numpy array
        return: (N, C) numpy array
        """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / (m + 1e-8)
        return pc

    def random_sample(self, pc: np.ndarray, num: int) -> np.ndarray:
        """Randomly subsample 'num' points from pc."""
        if pc.shape[0] < num:
            raise ValueError(f"Point cloud has {pc.shape[0]} points, "
                             f"but requested {num}")
        np.random.shuffle(self.permutation)
        idx = self.permutation[:num]
        return pc[idx]

    # ---------- core logic ----------

    def _load_points_and_sq_params(self, sample: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load point cloud (N,3) and SQ params (S,11) for one sample from LMDB.
        """
        key = sample["file_path"]  # e.g. "train:00000042"

        # Main point cloud value from LMDB
        pts = IO.get_lmdb(key, self.pc_path)  # (N,3) float32
        if pts is None:
            raise RuntimeError(f"SuperquadricSDFDataset: couldn't load points for key={key!r}")

        # Sidecar SQ params from LMDB
        sq_params = IO.get_lmdb(key + ".sq_params", self.pc_path)  # (S,11) float32
        if sq_params is None:
            raise RuntimeError(f"SuperquadricSDFDataset: couldn't load sq_params for key={key!r}")

        # Ensure arrays are in the expected shape/dtype
        pts = np.asarray(pts, dtype=np.float32)
        sq_params = np.asarray(sq_params, dtype=np.float32)

        if sq_params.ndim != 2 or sq_params.shape[1] != 11:
            raise ValueError(f"sq_params for key={key!r} must have shape (S,11); "
                             f"got {sq_params.shape}")

        return pts, sq_params

    def _sample_sdf_grid(
        self,
        sq_params_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given torch SQ params (S,11), sample a regular 3D grid of query points and
        evaluate the union implicit field.

        Returns:
            query_points: (M, 3) float32 tensor on CPU
            sdf_values:  (M,)   float32 tensor on CPU
            grid_shape:  (3,)   long tensor [nx, ny, nz]
        """
        # Global bounding box over all SQs
        min_xyz, max_xyz = global_bounding_box_for_all_SQs(sq_params_t)  # (3,),(3,)

        # Grid sampling
        points_grid, grid_shape = sample_grid_in_box(
            min_xyz,
            max_xyz,
            resolution=self.grid_resolution,
            device=sq_params_t.device,
            dtype=sq_params_t.dtype,
        )  # (M,3), (nx,ny,nz)

        # Evaluate union implicit field (f-1) at all grid points
        sdf_vals = multi_sq_implicit_union(points_grid, sq_params_t, signed=True)  # (M,)

        # Pack grid_shape as a tensor so collate_fn can deal with it easily
        grid_shape_t = torch.tensor(grid_shape, dtype=torch.long)

        # All on CPU (you'll move to GPU in your training loop)
        return (
            points_grid.detach().cpu(),
            sdf_vals.detach().cpu(),
            grid_shape_t,
        )
    
    def _sample_occupancy_balanced(
        self,
        sq_params_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample query points with roughly balanced occupancy labels and a bias
        towards points near the SQ surface.

        Returns:
            query_points: (M, 3) float32 tensor on CPU
            sdf_values:  (M,)   float32 tensor on CPU  (still the implicit field!)
            grid_shape:  (3,)   long tensor [M, 1, 1]  (degenerate "grid")
        """
        # Global bounding box over all SQs
        min_xyz, max_xyz = global_bounding_box_for_all_SQs(sq_params_t)  # (3,),(3,)

        # Full candidate grid
        points_grid, grid_shape = sample_grid_in_box(
            min_xyz,
            max_xyz,
            resolution=self.grid_resolution,
            device=sq_params_t.device,
            dtype=sq_params_t.dtype,
        )  # (M_full,3), (nx,ny,nz)

        # Signed implicit field (f-1)
        sdf_vals = multi_sq_implicit_union(points_grid, sq_params_t, signed=True)  # (M_full,)
        occ_vals = sdf_vals <= 0.0
        sdf_abs = sdf_vals.abs()

        device = sq_params_t.device

        # Masks for near-surface and inside/outside
        band = self.occ_band_width
        near_mask = sdf_abs < band

        inside_mask = occ_vals
        outside_mask = ~occ_vals

        def _nz(mask: torch.Tensor) -> torch.Tensor:
            idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            return idx

        inside_idx = _nz(inside_mask)
        outside_idx = _nz(outside_mask)

        # If for some degenerate reason all points are inside or outside,
        # just fall back to uniform subsampling of the grid.
        if inside_idx.numel() == 0 or outside_idx.numel() == 0:
            num_total = min(points_grid.shape[0], self.n_query_points)
            perm = torch.randperm(points_grid.shape[0], device=device)[:num_total]
            points_sub = points_grid[perm]
            sdf_sub = sdf_vals[perm]
            grid_shape_t = torch.tensor([points_sub.shape[0], 1, 1], dtype=torch.long)
            return points_sub.detach().cpu(), sdf_sub.detach().cpu(), grid_shape_t

        near_inside_idx = _nz(inside_mask & near_mask)
        far_inside_idx = _nz(inside_mask & ~near_mask)
        near_outside_idx = _nz(outside_mask & near_mask)
        far_outside_idx = _nz(outside_mask & ~near_mask)

        # How many samples do we want in total / per class?
        num_total = min(self.n_query_points, points_grid.shape[0])
        num_inside = int(num_total * self.occ_inside_fraction)
        num_outside = num_total - num_inside

        num_near_inside = int(num_inside * self.occ_near_fraction)
        num_far_inside = num_inside - num_near_inside
        num_near_outside = int(num_outside * self.occ_near_fraction)
        num_far_outside = num_outside - num_near_outside

        def _sample_from_two_sets(primary: torch.Tensor,
                                  secondary: torch.Tensor,
                                  k: int) -> torch.Tensor:
            """Sample k indices, preferring 'primary' but falling back to 'secondary'."""
            if k <= 0:
                return torch.empty(0, dtype=torch.long, device=device)
            if primary.numel() >= k:
                perm = torch.randperm(primary.numel(), device=device)[:k]
                return primary[perm]

            need_secondary = k - primary.numel()

            if primary.numel() > 0:
                perm_p = torch.randperm(primary.numel(), device=device)
                primary_sel = primary[perm_p]
            else:
                primary_sel = torch.empty(0, dtype=torch.long, device=device)

            if secondary.numel() == 0:
                # Last resort: sample anywhere from the full grid
                all_idx = torch.arange(points_grid.shape[0], device=device)
                rand_idx = torch.randint(0, all_idx.numel(), (need_secondary,), device=device)
                secondary_sel = all_idx[rand_idx]
            elif secondary.numel() >= need_secondary:
                perm_s = torch.randperm(secondary.numel(), device=device)[:need_secondary]
                secondary_sel = secondary[perm_s]
            else:
                rand_idx = torch.randint(0, secondary.numel(), (need_secondary,), device=device)
                secondary_sel = secondary[rand_idx]

            return torch.cat([primary_sel, secondary_sel], dim=0)

        inside_sel = torch.cat([
            _sample_from_two_sets(near_inside_idx, far_inside_idx, num_near_inside),
            _sample_from_two_sets(far_inside_idx, near_inside_idx, num_far_inside),
        ], dim=0)

        outside_sel = torch.cat([
            _sample_from_two_sets(near_outside_idx, far_outside_idx, num_near_outside),
            _sample_from_two_sets(far_outside_idx, near_outside_idx, num_far_outside),
        ], dim=0)

        idx = torch.cat([inside_sel, outside_sel], dim=0)

        points_sub = points_grid[idx]
        sdf_sub = sdf_vals[idx]

        # Degenerate "grid": we just record how many points we kept
        grid_shape_t = torch.tensor([points_sub.shape[0], 1, 1], dtype=torch.long)

        return (
            points_sub.detach().cpu(),
            sdf_sub.detach().cpu(),
            grid_shape_t,
        )


    def __getitem__(self, idx: int):
        sample = self.file_list[idx]

        # --- load point cloud + SQ params from LMDB ---
        pts_np, sq_params_np = self._load_points_and_sq_params(sample)

        # --- subsample + normalize point cloud for encoder input ---
        pts_surface_np = self.random_sample(pts_np, self.npoints)  # (P,3)
        #pts_np = self.pc_norm(pts_np)                             # (P,3)
        pts_t = torch.from_numpy(pts_surface_np).float()           # (P,3)

        # --- convert SQ params to torch ---
        sq_params_t = torch.from_numpy(sq_params_np).float()  # (S,11)

        # --- sample SDF queries according to sampling_method ---
        if self.sampling_method == "grid":
            query_points_t, sdf_vals_t, grid_shape_t = self._sample_sdf_grid(sq_params_t)
        elif self.sampling_method == "occ_balanced":
            query_points_t, sdf_vals_t, grid_shape_t = self._sample_occupancy_balanced(sq_params_t)
        elif self.sampling_method == "sdf_balanced":
            query_points_t, sdf_vals_t, grid_shape_t = self._sample_occupancy_balanced(sq_params_t)
        elif self.sampling_method == "surface_jitter":
            query_points_t, sdf_vals_t, grid_shape_t = self._sample_sdf_surface_jitter(sq_params_t, pts_surface_np)
        elif self.sampling_method == "uniform_bbox":
            query_points_t, sdf_vals_t, grid_shape_t = self._sample_sdf_uniform_bbox(sq_params_t)
        else:
            raise NotImplementedError(
                f"SuperquadricSDFDataset: sampling_method={self.sampling_method!r} "
                f"not implemented."
            )

        payload = (pts_t, query_points_t, sdf_vals_t, grid_shape_t)

        return sample["taxonomy_id"], sample["model_id"], payload

    def __len__(self) -> int:
        return len(self.file_list)

    def _sample_sdf_surface_jitter(
            self,
            sq_params_t: torch.Tensor,
            pts_surface_np: np.ndarray,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Sample SDF query points by jittering subsampled surface points.

            For each point in pts_surface_np (P,3) we generate `jitter_factor` new
            query points by adding Gaussian noise with std `jitter_sigma`.

            Returns:
                query_points: (M, 3) float32 tensor on CPU
                sdf_values:  (M,) float32 tensor on CPU
                grid_shape:  (3,) long tensor = [M, 1, 1] (degenerate grid)
            """
            device = sq_params_t.device
            dtype = sq_params_t.dtype

            # Convert subsampled surface points to torch on same device/dtype as sq_params.
            pts_surface_t = torch.from_numpy(pts_surface_np).to(device=device, dtype=dtype)  # (P,3)
            num_surface = pts_surface_t.shape[0]

            # How many jittered queries in total
            factor = int(self.jitter_factor)
            if factor <= 0:
                raise ValueError(f"SQ_JITTER_FACTOR must be > 0, got {factor}")

            # Repeat each surface point 'factor' times → (P * factor, 3)
            pts_rep = pts_surface_t.unsqueeze(1).repeat(1, factor, 1).reshape(-1, 3)

            # Add Gaussian noise
            sigma = float(self.jitter_sigma)
            if sigma > 0.0:
                noise = torch.randn_like(pts_rep) * sigma
                query_points = pts_rep + noise
            else:
                query_points = pts_rep

            # Compute SDF / implicit values at these query points
            sdf_vals = multi_sq_implicit_union(query_points, sq_params_t, signed=True)  # (M,)

            # "Grid shape" is just a degenerate (M,1,1) so downstream code still works
            M = query_points.shape[0]
            grid_shape_t = torch.tensor([M, 1, 1], dtype=torch.long)

            # Return on CPU (rest of dataset code expects CPU tensors)
            return (
                query_points.detach().cpu().float(),   # (M,3)
                sdf_vals.detach().cpu().float(),       # (M,)
                grid_shape_t,
            )
    
    def _sample_sdf_uniform_bbox(
        self,
        sq_params_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample SDF query points uniformly at random inside the global
        bounding box of all superquadrics for this shape.

        Returns:
            query_points: (M, 3) float32 tensor on CPU
            sdf_values:  (M,) float32 tensor on CPU
            grid_shape:  (3,) long tensor [M, 1, 1] (degenerate grid)
        """
        device = sq_params_t.device
        dtype = sq_params_t.dtype

        # Global bounding box over all SQs: (3,), (3,)
        min_xyz, max_xyz = global_bounding_box_for_all_SQs(sq_params_t)

        # Number of random query points
        M = int(self.n_query_points)
        if M <= 0:
            raise ValueError(f"SQ_SDF_N_QUERY must be > 0, got {M}")

        # Uniform random in [0,1]^3 → scale to [min_xyz, max_xyz]
        # shape: (M, 3)
        u = torch.rand((M, 3), device=device, dtype=dtype)
        query_points = min_xyz[None, :] + u * (max_xyz[None, :] - min_xyz[None, :])

        # Evaluate SQ implicit function at these points
        sdf_vals = multi_sq_implicit_union(query_points, sq_params_t, signed=True)  # (M,)

        # Degenerate "grid" shape, so downstream still works
        grid_shape_t = torch.tensor([M, 1, 1], dtype=torch.long)

        return (
            query_points.detach().cpu().float(),  # (M,3)
            sdf_vals.detach().cpu().float(),      # (M,)
            grid_shape_t,
        )