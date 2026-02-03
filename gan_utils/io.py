# io.py

import os
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable, List, Tuple, Union
import torch.nn.functional as F


def _ensure_three_channels(arr: np.ndarray) -> np.ndarray:
    """
    Enforce (H, W, 3) layout:
    - If grayscale (H, W), stack to 3 channels
    - If 4+ channels (e.g., RGBA), trim to first 3
    """
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[..., :3]
    return arr


def _normalize_to_float(arr: np.ndarray) -> np.ndarray:
    """
    Convert to float32 in [0,1] by dividing by 255 if input dtype suggests 8-bit,
    otherwise scale by max if needed (safe fallback).
    """
    arr = arr.astype(np.float32)
    # Heuristic: if max > 1 and <= 255, likely uint8-style images
    maxv = arr.max() if arr.size > 0 else 1.0
    if maxv > 1.0 and maxv <= 255.0:
        arr = arr / 255.0
    else:
        # For arbitrary ranges, avoid division-by-zero
        if maxv > 0:
            arr = arr / maxv
    return arr


def _to_torch_image(arr: np.ndarray) -> torch.Tensor:
    """
    Convert (H, W, C) numpy array in [0,1] float32 to torch tensor (C, H, W) float32.
    """
    arr = _ensure_three_channels(arr)
    arr = _normalize_to_float(arr)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class H5ImageDataset(Dataset):
    """
    Faster HDF5 dataset:
    - Opens the HDF5 file lazily once per worker.
    - Avoids file handle duplication.
    - Much faster for large datasets (50k–200k images).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        h5_path: str,
        key_col: str = "file_name",
        transforms: Optional[Callable] = None,
        verify_keys: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.key_col = key_col
        self.h5_path = h5_path
        self.transforms = transforms

        if self.key_col not in self.df.columns:
            raise KeyError(f"Expected column '{self.key_col}' in df")

        # Unique keys
        self.keys = self.df[self.key_col].astype(str).unique().tolist()

        # DO NOT OPEN HDF5 HERE
        self._h5 = None

        # Optional key verification
        if verify_keys:
            with h5py.File(self.h5_path, "r") as f:
                missing = [k for k in self.keys if k not in f]
            if missing:
                raise KeyError(f"{len(missing)} keys missing, e.g. {missing[:5]}")

    def _init_h5(self):
        """Open HDF5 lazily once per worker."""
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        self._init_h5()

        key = self.keys[idx]
        if key not in self._h5:
            raise KeyError(f"Image key '{key}' not found in HDF5")

        arr = self._h5[key][:]  # numpy array (H,W,C)

        # Convert to torch tensor
        img = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        
        img = F.interpolate(img.unsqueeze(0), size=(512,512), mode="bilinear", align_corners=False).squeeze(0)


        if self.transforms:
            img = self.transforms(img)

        return img



# class H5ImageDataset(Dataset):
#     """
#     Minimal dataset: returns images only.
#     - df must contain a column 'file_name' which matches keys in the HDF5 file.
#     - h5_path points to the HDF5 file with datasets under those keys.
#     - Optional transforms(img) can be applied to the torch tensor (C,H,W).
#     """

#     def __init__(
#         self,
#         df: pd.DataFrame,
#         h5_path: str,
#         key_col: str = "file_name",
#         transforms: Optional[Callable] = None,
#         verify_keys: bool = False,
#     ):
#         self.df = df.reset_index(drop=True)
#         self.key_col = key_col
#         self.h5_path = h5_path
#         self.transforms = transforms

#         # Group one record per image key; if df has multiple rows per image, dedupe
#         if self.key_col not in self.df.columns:
#             raise KeyError(f"Expected column '{self.key_col}' in df")
#         self.keys = self.df[self.key_col].astype(str).unique().tolist()

#         # Open HDF5 once and hold handle
#         self.h5f = h5py.File(self.h5_path, "r")

#         if verify_keys:
#             missing = [k for k in self.keys if k not in self.h5f]
#             if len(missing) > 0:
#                 raise KeyError(f"{len(missing)} image keys not found in HDF5, e.g. {missing[:5]}")

#     def __len__(self) -> int:
#         return len(self.keys)

#     def __getitem__(self, idx: int) -> torch.Tensor:
#         key = self.keys[idx]
#         if key not in self.h5f:
#             raise KeyError(f"Image key '{key}' not found in HDF5")
#         arr = self.h5f[key][:]
#         img = _to_torch_image(arr)  # (C,H,W) float32 in [0,1]
#         if self.transforms:
#             img = self.transforms(img)
#         return img

# New Code

        img = _to_torch_image(arr) # (C,H,W) float32 in [0,1] 
    
    # --- NEW: pad to 512×512 --- 
        C, H, W = img.shape 
        pad_h = 512 - H 
        pad_w = 512 - W 
        
        if pad_h < 0 or pad_w < 0: 
            raise ValueError(f"Image is larger than 512×512: got {H}×{W}") 
        
        # F.pad pads in (left, right, top, bottom) order for 2D images 
        # For (C,H,W), we pad width then height: (W_left, W_right, H_top, H_bottom) 
        
        img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h)) 
        
        # ---------------------------------- 
        
        if self.transforms: 
            img = self.transforms(img) 
        return img


class H5ImageCondDataset(Dataset):
    """
    Returns (image, cond) pair:
    - df provides one row per image with metadata columns used to create a conditioning vector.
    - h5_path is the HDF5 file with datasets under keys from df[key_col].
    - cond_cols lists the df columns to extract as conditioning features (float32).
    - Optional transforms(img, cond) can be applied. If a transform expects only img, handle accordingly.

    Example:
        cond_cols = ["area", "source_id"] or any numeric features you want to condition on.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        h5_path: str,
        key_col: str = "file_name",
        cond_cols: Optional[List[str]] = None,
        transforms: Optional[Callable] = None,
        fill_missing: float = 0.0,
        verify_keys: bool = False,
    ):
        self.h5_path = h5_path
        self.key_col = key_col
        self.cond_cols = cond_cols or []
        self.transforms = transforms
        self.fill_missing = fill_missing

        # One record per image_id/file_name; if multiple rows exist per image, pick first
        if self.key_col not in df.columns:
            raise KeyError(f"Expected column '{self.key_col}' in df")

        # If df has multiple rows per image, aggregate to first for conditioning
        grouped = df.groupby(self.key_col).agg(**{
            col: (col, "first") for col in self.cond_cols
        }).reset_index()

        self.records = grouped.reset_index(drop=True).to_dict(orient="records")
        self.h5f = h5py.File(self.h5_path, "r")

        # Prepare column existence and type safety
        for col in self.cond_cols:
            if col not in grouped.columns:
                raise KeyError(f"Conditioning column '{col}' not found in df")

        if verify_keys:
            missing = [rec[self.key_col] for rec in self.records if rec[self.key_col] not in self.h5f]
            if len(missing) > 0:
                raise KeyError(f"{len(missing)} image keys not found in HDF5, e.g. {missing[:5]}")

    def __len__(self) -> int:
        return len(self.records)

    def _make_cond_vector(self, rec: dict) -> torch.Tensor:
        vals: List[float] = []
        for col in self.cond_cols:
            v = rec.get(col, self.fill_missing)
            # Convert non-numerics safely
            try:
                v = float(v)
            except Exception:
                v = self.fill_missing
            vals.append(v)
        return torch.tensor(vals, dtype=torch.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rec = self.records[idx]
        key = str(rec[self.key_col])
        if key not in self.h5f:
            raise KeyError(f"Image key '{key}' not found in HDF5")

        arr = self.h5f[key][:]
        img = _to_torch_image(arr)  # (C,H,W)
        cond = self._make_cond_vector(rec)  # (D,)

        if self.transforms:
            try:
                img, cond = self.transforms(img, cond)
            except TypeError:
                # If transform only expects img
                img = self.transforms(img)

        return img, cond


class H5ImageUniformBinSamplerDataset(Dataset):
    """
    Uniformly sample images across bins of a chosen df numeric column.
    - df: must have key_col and bin_col (numeric)
    - bins: list of (left, right) tuples defining bin ranges
    - per_bin: number of samples to draw per bin (with replacement if fewer available)
    - returns images only (GAN-friendly). If you also need conditioning, combine with H5ImageCondDataset logic.

    Example:
        bins = [(0, 100), (100, 1000), (1000, 10000)]
        bin_col = "area"  # or any numeric metric you care about
    """

    def __init__(
        self,
        df: pd.DataFrame,
        h5_path: str,
        key_col: str = "file_name",
        bin_col: str = "area",
        bins: Optional[List[Tuple[float, float]]] = None,
        per_bin: int = 1000,
        transforms: Optional[Callable] = None,
        verify_keys: bool = False,
    ):
        self.h5_path = h5_path
        self.key_col = key_col
        self.bin_col = bin_col
        self.bins = bins or [(0, 1e2), (1e2, 1e3), (1e3, 1e4), (1e4, 1e6)]
        self.per_bin = per_bin
        self.transforms = transforms

        if self.key_col not in df.columns:
            raise KeyError(f"Expected column '{self.key_col}' in df")
        if self.bin_col not in df.columns:
            raise KeyError(f"Expected bin column '{self.bin_col}' in df")

        # Prepare numeric values for binning; coerce invalids to NaN then drop
        df = df.copy()
        df[self.bin_col] = pd.to_numeric(df[self.bin_col], errors="coerce")
        df = df.dropna(subset=[self.bin_col, self.key_col])

        subsampled_keys: List[str] = []

        for (l, r) in self.bins:
            in_bin = df[(df[self.bin_col] >= l) & (df[self.bin_col] < r)]
            keys = in_bin[self.key_col].astype(str).unique()
            if len(keys) == 0:
                continue
            if len(keys) < self.per_bin:
                chosen = np.random.choice(keys, self.per_bin, replace=True)
            else:
                chosen = np.random.choice(keys, self.per_bin, replace=False)
            subsampled_keys.extend(list(chosen))

        self.keys = subsampled_keys
        self.h5f = h5py.File(self.h5_path, "r")

        if verify_keys:
            missing = [k for k in self.keys if k not in self.h5f]
            if len(missing) > 0:
                raise KeyError(f"{len(missing)} sampled keys not found in HDF5, e.g. {missing[:5]}")

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> torch.Tensor:
        key = self.keys[idx]
        if key not in self.h5f:
            raise KeyError(f"Image key '{key}' not found in HDF5")
        arr = self.h5f[key][:]
        img = _to_torch_image(arr)  # (C,H,W)
        if self.transforms:
            img = self.transforms(img)
        return img

# import h5py
# import pandas as pd
# # from io import H5ImageDataset   # assuming your new io.py is in the same folder
# from torch.utils.data import DataLoader

# # 1. Load your metadata (parquet or CSV with file_name column)
# df = pd.read_parquet("mask_labels_multiclass_composites_fractions.gzip")
# STACK_DIR = "stacks_fractions/"

# import os
# import h5py
# import numpy as np


# out_file = "all_images.h5"

# with h5py.File(out_file, "w") as h5f:
#     for fname in sorted(os.listdir(STACK_DIR)):
#         if fname.endswith(".npz"):
#             path = os.path.join(STACK_DIR, fname)
#             stack = np.load(path)
#             for key in stack.files:
#                 arr = stack[key]
#                 # store each image as a dataset under its key
#                 if key not in h5f:
#                     h5f.create_dataset(key, data=arr, compression="gzip")
# print("Conversion complete → all_images.h5")


# # 2. Point to your HDF5 file
# h5_path = "all_images.h5"

# # 3. Create dataset and dataloader
# dataset = H5ImageDataset(df, h5_path)
# loader = DataLoader(dataset, batch_size=4, shuffle=True)

# # 4. Grab one batch
# batch = next(iter(loader))
# print("Batch shape:", batch.shape)   # should be (4, 3, H, W)

# # 5. Inspect one image tensor
# img = batch[0]
# print("Image tensor stats:", img.min().item(), img.max().item(), img.shape)





# import os
# from torch.utils.data import Dataset
# import pandas as pd
# import numpy as np
# import tables as tb
# import torch
# from functions import charge2img, normalize_and_convert_to_img

# class StereoVeristasDataGenNorm(Dataset):
#     def __init__(self, input_file, size_threshold=100, imsize=96, mode='intersect'):
#         self.h5file = input_file
#         self.size_threshold = size_threshold
#         self.img_size = imsize
#         self.h5_table = tb.open_file(self.h5file)
        
#         self.T1 = np.array(self.h5_table.root.dl1.event.telescope.images.tel_001.cols.image)
#         self.T1_events = np.array(self.h5_table.root.dl1.event.telescope.parameters.tel_001.cols.event_id)
        
#         self.T2 = np.array(self.h5_table.root.dl1.event.telescope.images.tel_002.cols.image)
#         self.T2_events = np.array(self.h5_table.root.dl1.event.telescope.parameters.tel_002.cols.event_id)
        
#         self.T3 = np.array(self.h5_table.root.dl1.event.telescope.images.tel_003.cols.image)
#         self.T3_events = np.array(self.h5_table.root.dl1.event.telescope.parameters.tel_003.cols.event_id)
        
#         self.T4 = np.array(self.h5_table.root.dl1.event.telescope.images.tel_004.cols.image)
#         self.T4_events = np.array(self.h5_table.root.dl1.event.telescope.parameters.tel_004.cols.event_id)
        
#         self.T1_cond = np.where(np.sum(self.T1, axis=1)>= size_threshold)
#         self.T2_cond = np.where(np.sum(self.T2, axis=1)>= size_threshold)
#         self.T3_cond = np.where(np.sum(self.T3, axis=1)>= size_threshold)
#         self.T4_cond = np.where(np.sum(self.T4, axis=1)>= size_threshold)
        
#         if mode == 'intersect':
#             self.t1_intersect_events = np.intersect1d(np.intersect1d(np.intersect1d(self.T1_events[self.T1_cond], self.T2_events), self.T3_events), self.T4_events)
#             self.t2_intersect_events = np.intersect1d(np.intersect1d(np.intersect1d(self.T1_events, self.T2_events[self.T2_cond]), self.T3_events), self.T4_events)
#             self.t3_intersect_events = np.intersect1d(np.intersect1d(np.intersect1d(self.T1_events, self.T2_events), self.T3_events[self.T3_cond]), self.T4_events)
#             self.t4_intersect_events = np.intersect1d(np.intersect1d(np.intersect1d(self.T1_events, self.T2_events), self.T3_events), self.T4_events[self.T4_cond])

#             self.all_intersected_events = np.unique(np.concatenate([self.t1_intersect_events, self.t2_intersect_events, self.t3_intersect_events, self.t4_intersect_events]))
#             print(f'Found total samples {self.all_intersected_events.shape[0]} meeting size thresh {self.size_threshold}')
#         else:
#             self.t1_intersect_events = np.union1d(np.union1d(np.union1d(self.T1_events[self.T1_cond], self.T2_events), self.T3_events), self.T4_events)
#             self.t2_intersect_events = np.union1d(np.union1d(np.union1d(self.T1_events, self.T2_events[self.T2_cond]), self.T3_events), self.T4_events)
#             self.t3_intersect_events = np.union1d(np.union1d(np.union1d(self.T1_events, self.T2_events), self.T3_events[self.T3_cond]), self.T4_events)
#             self.t4_intersect_events = np.union1d(np.union1d(np.union1d(self.T1_events, self.T2_events), self.T3_events), self.T4_events[self.T4_cond])
            
#             self.all_intersected_events = np.unique(np.concatenate([self.t1_intersect_events, self.t2_intersect_events, self.t3_intersect_events, self.t4_intersect_events]))
#             print(f'Found total samples {self.all_intersected_events.shape[0]} meeting size thresh {self.size_threshold}')


#     def __len__(self):
#         return(len(self.all_intersected_events))
    
#     def __getitem__(self, index):
        
#         intersect_event = self.all_intersected_events[index]
        
#         try:
#             t1_array = normalize_and_convert_to_img(self.T1[np.where(self.T1_events == intersect_event)][0])
#         except IndexError:
#             t1_array = normalize_and_convert_to_img(np.zeros(499))
        
#         try:
#             t2_array = normalize_and_convert_to_img(self.T2[np.where(self.T2_events == intersect_event)][0])
#         except IndexError:
#             t2_array = normalize_and_convert_to_img(np.zeros(499))
            
#         try:
#             t3_array = normalize_and_convert_to_img(self.T3[np.where(self.T3_events == intersect_event)][0])
#         except IndexError:
#             t3_array = normalize_and_convert_to_img(np.zeros(499))
        
#         try:
#             t4_array = normalize_and_convert_to_img(self.T4[np.where(self.T4_events == intersect_event)][0])
#         except IndexError:
#             t4_array = normalize_and_convert_to_img(np.zeros(499))
        
#         return torch.cat((t1_array,t2_array,t3_array,t4_array), dim=0)
    

# class CondStereoVeristasDataGenNorm(Dataset):
#     def __init__(self, input_file, size_threshold=100, imsize=96, mode='intersect'):
#         self.h5file = input_file
#         self.size_threshold = size_threshold
#         self.img_size = imsize
#         self.h5_table = tb.open_file(self.h5file)
        
#         self.needed_hp = ['hillas_r', 'hillas_length', 'hillas_width', 'hillas_skewness', 'hillas_kurtosis']
        
#         self.T1 = np.array(self.h5_table.root.dl1.event.telescope.images.tel_001.cols.image)
#         self.T1_events = np.array(self.h5_table.root.dl1.event.telescope.parameters.tel_001.cols.event_id)
#         self.T1_hp = np.moveaxis(np.array([np.array(self.h5_table.root.dl1.event.telescope.parameters.tel_001.col(i)) for i in self.needed_hp]),0,1)
        
        
#         self.T2 = np.array(self.h5_table.root.dl1.event.telescope.images.tel_002.cols.image)
#         self.T2_events = np.array(self.h5_table.root.dl1.event.telescope.parameters.tel_002.cols.event_id)
#         self.T2_hp = np.moveaxis(np.array([np.array(self.h5_table.root.dl1.event.telescope.parameters.tel_002.col(i)) for i in self.needed_hp]),0,1)
        
#         self.T3 = np.array(self.h5_table.root.dl1.event.telescope.images.tel_003.cols.image)
#         self.T3_events = np.array(self.h5_table.root.dl1.event.telescope.parameters.tel_003.cols.event_id)
#         self.T3_hp = np.moveaxis(np.array([np.array(self.h5_table.root.dl1.event.telescope.parameters.tel_003.col(i)) for i in self.needed_hp]),0,1)
        
#         self.T4 = np.array(self.h5_table.root.dl1.event.telescope.images.tel_004.cols.image)
#         self.T4_events = np.array(self.h5_table.root.dl1.event.telescope.parameters.tel_004.cols.event_id)
#         self.T4_hp = np.moveaxis(np.array([np.array(self.h5_table.root.dl1.event.telescope.parameters.tel_004.col(i)) for i in self.needed_hp]),0,1)
        
#         self.T1_cond = np.where(np.sum(self.T1, axis=1)>= size_threshold)
#         self.T2_cond = np.where(np.sum(self.T2, axis=1)>= size_threshold)
#         self.T3_cond = np.where(np.sum(self.T3, axis=1)>= size_threshold)
#         self.T4_cond = np.where(np.sum(self.T4, axis=1)>= size_threshold)
        
#         if mode == 'intersect':
#             self.t1_intersect_events = np.intersect1d(np.intersect1d(np.intersect1d(self.T1_events[self.T1_cond], self.T2_events), self.T3_events), self.T4_events)
#             self.t2_intersect_events = np.intersect1d(np.intersect1d(np.intersect1d(self.T1_events, self.T2_events[self.T2_cond]), self.T3_events), self.T4_events)
#             self.t3_intersect_events = np.intersect1d(np.intersect1d(np.intersect1d(self.T1_events, self.T2_events), self.T3_events[self.T3_cond]), self.T4_events)
#             self.t4_intersect_events = np.intersect1d(np.intersect1d(np.intersect1d(self.T1_events, self.T2_events), self.T3_events), self.T4_events[self.T4_cond])

#             self.all_intersected_events = np.unique(np.concatenate([self.t1_intersect_events, self.t2_intersect_events, self.t3_intersect_events, self.t4_intersect_events]))
#             print(f'Found total samples {self.all_intersected_events.shape[0]} meeting size thresh {self.size_threshold}')
#         else:
#             self.t1_intersect_events = np.union1d(np.union1d(np.union1d(self.T1_events[self.T1_cond], self.T2_events), self.T3_events), self.T4_events)
#             self.t2_intersect_events = np.union1d(np.union1d(np.union1d(self.T1_events, self.T2_events[self.T2_cond]), self.T3_events), self.T4_events)
#             self.t3_intersect_events = np.union1d(np.union1d(np.union1d(self.T1_events, self.T2_events), self.T3_events[self.T3_cond]), self.T4_events)
#             self.t4_intersect_events = np.union1d(np.union1d(np.union1d(self.T1_events, self.T2_events), self.T3_events), self.T4_events[self.T4_cond])
            
#             self.all_intersected_events = np.unique(np.concatenate([self.t1_intersect_events, self.t2_intersect_events, self.t3_intersect_events, self.t4_intersect_events]))
#             print(f'Found total samples {self.all_intersected_events.shape[0]} meeting size thresh {self.size_threshold}')


#     def __len__(self):
#         return(len(self.all_intersected_events))
    
    
#     def __getitem__(self, index):
#         intersect_event = self.all_intersected_events[index]
        
#         try:
#             t1_charge_array = self.T1[np.where(self.T1_events == intersect_event)][0]
#             t1_array = normalize_and_convert_to_img(t1_charge_array)
#             t1_size = np.sum(t1_charge_array)
#             if t1_size!=0:
#                 t1_size = np.log10(t1_size)
#             else:
#                 t1_size = -1
#         except IndexError:
#             t1_array = normalize_and_convert_to_img(np.zeros(499))
#             t1_size = -1
        
#         try:
#             t2_charge_array = self.T2[np.where(self.T2_events == intersect_event)][0]
#             t2_array = normalize_and_convert_to_img(t2_charge_array)
#             t2_size = np.sum(t2_charge_array)
#             if t2_size!=0:
#                 t2_size = np.log10(t2_size)
#             else:
#                 t2_size = -1
#         except IndexError:
#             t2_array = normalize_and_convert_to_img(np.zeros(499))
#             t2_size = -1
            
#         try:
#             t3_charge_array = self.T3[np.where(self.T3_events == intersect_event)][0]
#             t3_array = normalize_and_convert_to_img(t3_charge_array)
#             t3_size = np.sum(t3_charge_array)
#             if t3_size!=0:
#                 t3_size = np.log10(t3_size)
#             else:
#                 t3_size = -1
#         except IndexError:
#             t3_array = normalize_and_convert_to_img(np.zeros(499))
#             t3_size = -1
        
#         try:
#             t4_charge_array = self.T4[np.where(self.T4_events == intersect_event)][0]
#             t4_array = normalize_and_convert_to_img(t4_charge_array)
#             t4_size = np.sum(t4_charge_array)
#             if t4_size!=0:
#                 t4_size = np.log10(t4_size)
#             else:
#                 t4_size = -1
#         except IndexError:
#             t4_array = normalize_and_convert_to_img(np.zeros(499))
#             t4_size = -1
        
#         t1_hp = np.concatenate([[t1_size],self.T1_hp[np.where(self.T1_events == intersect_event)][0]])
#         t2_hp = np.concatenate([[t2_size], self.T2_hp[np.where(self.T2_events == intersect_event)][0]])
#         t3_hp = np.concatenate([[t3_size], self.T3_hp[np.where(self.T3_events == intersect_event)][0]])
#         t4_hp = np.concatenate([[t4_size], self.T4_hp[np.where(self.T4_events == intersect_event)][0]])
        
#         hillas_conds = np.concatenate([t1_hp, t2_hp, t3_hp, t4_hp])
#         hillas_conds[np.where(np.isnan(hillas_conds))] = 0
        
#         return torch.cat((t1_array,t2_array,t3_array,t4_array), dim=0), torch.Tensor(hillas_conds).to(torch.float32)#torch.Tensor([t1_size, t2_size, t3_size, t4_size]).to(torch.float32)


# class CondStereoVeristasDataGenNormUniformSampling(Dataset):
#     def __init__(self, input_file, size_threshold=100, imsize=96, mode='intersect', per_size_sampling=2500):
#         self.h5file = input_file
#         self.size_threshold = size_threshold
#         self.img_size = imsize
#         self.h5_table = tb.open_file(self.h5file)
#         self.per_size_sampling = per_size_sampling
        
#         self.needed_hp = ['camera_frame_hillas_r', 'camera_frame_hillas_length', 'camera_frame_hillas_width', 'camera_frame_hillas_skewness', 'camera_frame_hillas_kurtosis']
        
#         self.T1 = np.array(self.h5_table.root.dl1.event.telescope.images.tel_001.cols.image)
#         self.T1_events = np.array(self.h5_table.root.dl1.event.telescope.parameters.tel_001.cols.event_id)
#         self.T1_hp = np.moveaxis(np.array([np.array(self.h5_table.root.dl1.event.telescope.parameters.tel_001.col(i)) for i in self.needed_hp]),0,1)
        
        
#         self.T2 = np.array(self.h5_table.root.dl1.event.telescope.images.tel_002.cols.image)
#         self.T2_events = np.array(self.h5_table.root.dl1.event.telescope.parameters.tel_002.cols.event_id)
#         self.T2_hp = np.moveaxis(np.array([np.array(self.h5_table.root.dl1.event.telescope.parameters.tel_002.col(i)) for i in self.needed_hp]),0,1)
        
#         self.T3 = np.array(self.h5_table.root.dl1.event.telescope.images.tel_003.cols.image)
#         self.T3_events = np.array(self.h5_table.root.dl1.event.telescope.parameters.tel_003.cols.event_id)
#         self.T3_hp = np.moveaxis(np.array([np.array(self.h5_table.root.dl1.event.telescope.parameters.tel_003.col(i)) for i in self.needed_hp]),0,1)
        
#         self.T4 = np.array(self.h5_table.root.dl1.event.telescope.images.tel_004.cols.image)
#         self.T4_events = np.array(self.h5_table.root.dl1.event.telescope.parameters.tel_004.cols.event_id)
#         self.T4_hp = np.moveaxis(np.array([np.array(self.h5_table.root.dl1.event.telescope.parameters.tel_004.col(i)) for i in self.needed_hp]),0,1)
        
#         self.T1_cond = np.where(np.sum(self.T1, axis=1)>= size_threshold)
#         self.T2_cond = np.where(np.sum(self.T2, axis=1)>= size_threshold)
#         self.T3_cond = np.where(np.sum(self.T3, axis=1)>= size_threshold)
#         self.T4_cond = np.where(np.sum(self.T4, axis=1)>= size_threshold)
        
#         if mode == 'intersect':
#             self.t1_intersect_events = np.intersect1d(np.intersect1d(np.intersect1d(self.T1_events[self.T1_cond], self.T2_events), self.T3_events), self.T4_events)
#             self.t2_intersect_events = np.intersect1d(np.intersect1d(np.intersect1d(self.T1_events, self.T2_events[self.T2_cond]), self.T3_events), self.T4_events)
#             self.t3_intersect_events = np.intersect1d(np.intersect1d(np.intersect1d(self.T1_events, self.T2_events), self.T3_events[self.T3_cond]), self.T4_events)
#             self.t4_intersect_events = np.intersect1d(np.intersect1d(np.intersect1d(self.T1_events, self.T2_events), self.T3_events), self.T4_events[self.T4_cond])

#             self.all_intersected_events = np.unique(np.concatenate([self.t1_intersect_events, self.t2_intersect_events, self.t3_intersect_events, self.t4_intersect_events]))
#             print(f'Found total samples {self.all_intersected_events.shape[0]} meeting size thresh {self.size_threshold}')
#         else:
#             self.t1_intersect_events = np.union1d(np.union1d(np.union1d(self.T1_events[self.T1_cond], self.T2_events), self.T3_events), self.T4_events)
#             self.t2_intersect_events = np.union1d(np.union1d(np.union1d(self.T1_events, self.T2_events[self.T2_cond]), self.T3_events), self.T4_events)
#             self.t3_intersect_events = np.union1d(np.union1d(np.union1d(self.T1_events, self.T2_events), self.T3_events[self.T3_cond]), self.T4_events)
#             self.t4_intersect_events = np.union1d(np.union1d(np.union1d(self.T1_events, self.T2_events), self.T3_events), self.T4_events[self.T4_cond])
            
#             self.all_intersected_events = np.unique(np.concatenate([self.t1_intersect_events, self.t2_intersect_events, self.t3_intersect_events, self.t4_intersect_events]))
#             print(f'Found total samples {self.all_intersected_events.shape[0]} meeting size thresh {self.size_threshold}')
    
#         idx_t1_events = np.array([np.where(self.T1_events == iii)[0] for iii in self.all_intersected_events])
#         t1_charge_array = np.squeeze(self.T1[idx_t1_events], axis=1)
#         t1_sum_array = np.log10(np.sum(t1_charge_array, axis=1))
        
#         idx_t2_events = np.array([np.where(self.T2_events == iii)[0] for iii in self.all_intersected_events])
#         t2_charge_array = np.squeeze(self.T2[idx_t2_events], axis=1)
#         t2_sum_array = np.log10(np.sum(t2_charge_array, axis=1))

#         idx_t3_events = np.array([np.where(self.T3_events == iii)[0] for iii in self.all_intersected_events])
#         t3_charge_array = np.squeeze(self.T3[idx_t3_events], axis=1)
#         t3_sum_array = np.log10(np.sum(t3_charge_array, axis=1))

#         idx_t4_events = np.array([np.where(self.T4_events == iii)[0] for iii in self.all_intersected_events])
#         t4_charge_array = np.squeeze(self.T4[idx_t4_events], axis=1)
#         t4_sum_array = np.log10(np.sum(t4_charge_array, axis=1))
        
#         subsampled_events = []

#         for size_bins in [[2,2.5], [2.5,3], [3, 3.5], [3.5,4], [4,6]]:
#             lbin, rbin = size_bins
#             required_cond = np.where((t1_sum_array>=lbin) & (t1_sum_array<rbin) & ((t2_sum_array>=lbin) & (t2_sum_array<rbin)) & 
#                              ((t3_sum_array>=lbin) & (t3_sum_array<rbin)) & ((t4_sum_array>=lbin) & (t4_sum_array<rbin)))[0]
#             chosen_events = self.all_intersected_events[required_cond]
#             if len(chosen_events)<self.per_size_sampling:
#                 chosen_events = np.random.choice(chosen_events, self.per_size_sampling, replace=True)
#             else:
#                 chosen_events = np.random.choice(chosen_events, self.per_size_sampling, replace=False)
#             subsampled_events.extend(list(chosen_events))
#         self.subsampled_events = np.array(subsampled_events)


#     def __len__(self):
#         return(len(self.subsampled_events))
    
    
#     def __getitem__(self, index):
#         intersect_event = self.subsampled_events[index]
        
#         try:
#             t1_charge_array = self.T1[np.where(self.T1_events == intersect_event)][0]
#             t1_array = normalize_and_convert_to_img(t1_charge_array)
#             t1_size = np.sum(t1_charge_array)
#             if t1_size!=0:
#                 t1_size = np.log10(t1_size)
#             else:
#                 t1_size = -1
#         except IndexError:
#             t1_array = normalize_and_convert_to_img(np.zeros(499))
#             t1_size = -1
        
#         try:
#             t2_charge_array = self.T2[np.where(self.T2_events == intersect_event)][0]
#             t2_array = normalize_and_convert_to_img(t2_charge_array)
#             t2_size = np.sum(t2_charge_array)
#             if t2_size!=0:
#                 t2_size = np.log10(t2_size)
#             else:
#                 t2_size = -1
#         except IndexError:
#             t2_array = normalize_and_convert_to_img(np.zeros(499))
#             t2_size = -1
            
#         try:
#             t3_charge_array = self.T3[np.where(self.T3_events == intersect_event)][0]
#             t3_array = normalize_and_convert_to_img(t3_charge_array)
#             t3_size = np.sum(t3_charge_array)
#             if t3_size!=0:
#                 t3_size = np.log10(t3_size)
#             else:
#                 t3_size = -1
#         except IndexError:
#             t3_array = normalize_and_convert_to_img(np.zeros(499))
#             t3_size = -1
        
#         try:
#             t4_charge_array = self.T4[np.where(self.T4_events == intersect_event)][0]
#             t4_array = normalize_and_convert_to_img(t4_charge_array)
#             t4_size = np.sum(t4_charge_array)
#             if t4_size!=0:
#                 t4_size = np.log10(t4_size)
#             else:
#                 t4_size = -1
#         except IndexError:
#             t4_array = normalize_and_convert_to_img(np.zeros(499))
#             t4_size = -1
        
#         t1_hp = np.concatenate([[t1_size],self.T1_hp[np.where(self.T1_events == intersect_event)][0]])
#         t2_hp = np.concatenate([[t2_size], self.T2_hp[np.where(self.T2_events == intersect_event)][0]])
#         t3_hp = np.concatenate([[t3_size], self.T3_hp[np.where(self.T3_events == intersect_event)][0]])
#         t4_hp = np.concatenate([[t4_size], self.T4_hp[np.where(self.T4_events == intersect_event)][0]])
        
#         hillas_conds = np.concatenate([t1_hp, t2_hp, t3_hp, t4_hp])
#         hillas_conds[np.where(np.isnan(hillas_conds))] = 0
        
#         return torch.cat((t1_array,t2_array,t3_array,t4_array), dim=0), torch.Tensor(hillas_conds).to(torch.float32)#torch.Tensor([t1_size, t2_size, t3_size, t4_size]).to(torch.float32)
    

# class VeritasDataGen(Dataset):
#     def __init__(self, input_file, size_threshold=100, imsize=96):
#         self.h5file = input_file
#         self.size_threshold = size_threshold
#         self.img_size = imsize
#         self.h5_table = tb.open_file(self.h5file)
        
#         self.T1 = np.array(self.h5_table.root.dl1.event.telescope.images.tel_001.cols.image)
#         self.T1_sum = np.sum(self.T1, axis=1)
#         self.T1_cond = self.T1[np.where(self.T1_sum>= size_threshold)]
        
#         self.T2 = np.array(self.h5_table.root.dl1.event.telescope.images.tel_002.cols.image)
#         self.T2_sum = np.sum(self.T2, axis=1)
#         self.T2_cond = self.T2[np.where(self.T2_sum>= size_threshold)]
        
#         self.T3 = np.array(self.h5_table.root.dl1.event.telescope.images.tel_003.cols.image)
#         self.T3_sum = np.sum(self.T3, axis=1)
#         self.T3_cond = self.T3[np.where(self.T3_sum>= size_threshold)]
        
#         self.T4 = np.array(self.h5_table.root.dl1.event.telescope.images.tel_004.cols.image)
#         self.T4_sum = np.sum(self.T4, axis=1)
#         self.T4_cond = self.T4[np.where(self.T4_sum>= size_threshold)]
        
#         self.Tall = np.concatenate([self.T1_cond, self.T2_cond, self.T3_cond, self.T4_cond], axis=0)
#         print(f'Found total samples {self.Tall.shape[0]} meeting size thresh {self.size_threshold}')
        
#     def __len__(self):
#         return(len(self.Tall))
    
    
#     def __getitem__(self, index, info_key='image'):
        
#         working_array = self.Tall[index]
        
#         working_array = working_array/np.percentile(working_array[np.where(working_array!=0)],95)
#         working_array[(np.where(np.array(working_array)<0))[0]] = 0
        
# #         print(working_array.shape)
#         return torch.unsqueeze(torch.Tensor(charge2img(working_array, imsize=self.img_size).astype('float32')), dim=0)
    
    
# class VeritasDataGenNorm(VeritasDataGen):
#     def __getitem__(self, index):
#         working_array = self.Tall[index]
#         working_array = working_array/np.sum(working_array)
#         working_array[(np.where(np.array(working_array)<0))[0]] = 0
        
#         percentile_value = np.percentile(working_array, 99)
#         working_array[np.where(working_array>percentile_value)] = percentile_value
        
#         working_array = working_array/np.max(working_array)
#         return torch.unsqueeze(torch.Tensor(charge2img(working_array, imsize=self.img_size).astype('float32')), dim=0)
    
    
        