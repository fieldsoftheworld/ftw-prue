import torch
import yaml
from box import Box


class preprocess:
    """
    Unified preprocessing helper for FTW images.

    Handles normalization and scaling for CLAY, TerraFM, DINOv3, and FTW baseline,
    including all temporal configurations: windowA, windowB, stacked, median, rgb, random_window.
    """

    def __init__(
        self,
        preprocess_type: str = "ftw",
        temporal_options: str = "stacked",
        metadata_path: str = "/u/subashk/storage/ftw-ablation/FTW-Bakeoff/ftw-baselines-2/configs/metadata.yaml",
        input_type: str = "images",
    ):
        self.input_type = input_type
        self.preprocess_type = preprocess_type.lower()
        self.temporal_options = temporal_options.lower()
        self.metadata_path = metadata_path

        if "images" in self.input_type:
            # Derive mean, std, and scaling logic
            self.mean, self.std, self.scale_first = self._get_mean_std_and_scale()

            print(f"[preprocess] Type={self.preprocess_type}  Temporal={self.temporal_options}")
            print(f"Mean: {self.mean.tolist()}")
            print(f"Std:  {self.std.tolist()}")
            print(f"Scale-first: {self.scale_first}")

    # ------------------------------------------------------------------
    def _get_mean_std_and_scale(self):
        temporal = self.temporal_options
        preprocess = self.preprocess_type

        # -------------------- CLAY --------------------
        if preprocess == "clay":
            if self.metadata_path is None:
                raise ValueError("metadata_path required for CLAY preprocessing")

            metadata = Box(yaml.safe_load(open(self.metadata_path, "r")))
            platform = "sentinel-2-l2a"
            base_bands = ["red", "green", "blue", "nir"]

            mean_4 = torch.tensor([metadata[platform].bands.mean[str(b)] for b in base_bands])
            std_4 = torch.tensor([metadata[platform].bands.std[str(b)] for b in base_bands])

            if temporal in ("windowa", "windowb", "median", "random_window"):
                mean, std = mean_4, std_4
            elif temporal == "stacked":
                mean = torch.cat([mean_4, mean_4])
                std = torch.cat([std_4, std_4])
            elif temporal == "rgb":
                mean_rgb = mean_4[:3]
                std_rgb = std_4[:3]
                mean = torch.cat([mean_rgb, mean_rgb])
                std = torch.cat([std_rgb, std_rgb])
            else:
                raise ValueError(f"Unsupported temporal option for CLAY: {temporal}")

            scale_first = False  # already reflectance-normalized

        # -------------------- TerraFM --------------------
        elif preprocess == "terrafm":
            if temporal in ("windowa", "windowb", "median", "random_window"):
                mean = torch.zeros(4)
                std = torch.ones(4)
            elif temporal == "stacked":
                mean = torch.zeros(8)
                std = torch.ones(8)
            elif temporal == "rgb":
                mean = torch.zeros(6)
                std = torch.ones(6)
            else:
                raise ValueError(f"Unsupported temporal option for TerraFM: {temporal}")

            scale_first = True  # divide by 3000

        # -------------------- DINOv3 --------------------
        elif preprocess == "dinov3":
            mean_3 = torch.tensor([0.430, 0.411, 0.296])
            std_3 = torch.tensor([0.213, 0.156, 0.143])

            if temporal in ("windowa", "windowb", "median", "random_window"):
                mean, std = mean_3, std_3
            elif temporal in ("stacked", "rgb"):
                mean = torch.cat([mean_3, mean_3])
                std = torch.cat([std_3, std_3])
            else:
                raise ValueError(f"Unsupported temporal option for DINOv3: {temporal}")

            scale_first = True  # divide by 3000 first

        # -------------------- FTW baseline --------------------
        else:
            if temporal in ("windowa", "windowb", "median", "random_window"):
                mean = torch.zeros(4)
                std = torch.ones(4)
            elif temporal == "stacked":
                mean = torch.zeros(8)
                std = torch.ones(8)
            elif temporal == "rgb":
                mean = torch.zeros(6)
                std = torch.ones(6)
            else:
                raise ValueError(f"Unsupported temporal option for FTW: {temporal}")

            scale_first = True

        return mean, std, scale_first

    # ------------------------------------------------------------------
    def __call__(self, sample):
        """Apply normalization to a sample dict."""
        if "images" not in self.input_type:
            return sample

        img = sample["image"].float()

        # 1️⃣ Scale reflectance if needed
        if self.scale_first:
            img = img / 3000.0

        # 2️⃣ Normalize if needed
        if self.preprocess_type in ["clay", "dinov3"]:
            mean = self.mean.to(img.device).view(-1, 1, 1)
            std = self.std.to(img.device).view(-1, 1, 1)
            img = (img - mean) / std

        sample["image"] = img
        return sample
