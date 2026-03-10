import os
import numpy as np
import torch
import rasterio
from pathlib import Path
from src.galileo import GalileoWrapper
from ftw_tools.settings import ALL_COUNTRIES

device = "cuda" if torch.cuda.is_available() else "cpu"

s2_band_names = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B08A", "B09", "B10", "B11", "B12"]

OURS_S2_MEAN = [
    1395.34,
    1395.34,
    1338.40,
    1343.09,
    1543.86,
    2186.20,
    2525.09,
    2410.33,
    2750.28,
    2750.28,
    2234.91,
    2234.91,
    1474.53,
]
OURS_S2_STD = [
    917.70,
    917.70,
    913.29,
    1092.68,
    1047.22,
    1048.01,
    1143.69,
    1098.97,
    1204.47,
    1204.47,
    1145.97,
    1145.97,
    980.24,
]

imputes = [
    ("B04", "B05"),
    ("B04", "B06"),
    ("B08", "B07"),
    ("B08", "B08A"),
    ("B08", "B09"),
    ("B08", "B10"),
    ("B08", "B11"),
    ("B08", "B12"),
]


def impute_bands(image_list, names_list, imputes, all_bands):
    new_images = []
    for band in all_bands:
        if band in names_list:
            new_images.append(image_list[names_list.index(band)])
        else:
            for src, tgt in imputes:
                if tgt == band and src in names_list:
                    new_images.append(image_list[names_list.index(src)])
                    break
            else:
                new_images.append(np.zeros_like(image_list[0]))
    return new_images


def process_embeddings(emb, size=(256, 256)):
    emb = torch.nn.functional.interpolate(emb.unsqueeze(0), size=size, mode="bilinear", align_corners=False).squeeze(0)
    return emb


def generate_embedding(tif_path, output_path, model):
    try:
        with rasterio.open(tif_path) as src:
            img_data = src.read().astype(np.float32)
        img_data = np.transpose(img_data, (1, 2, 0))
        h, w, c = img_data.shape

        available_band_names = ["B04", "B03", "B02", "B08"]
        image_list = [img_data[..., i] for i in range(c)]

        imputed_bands = impute_bands(image_list, available_band_names, imputes, s2_band_names)
        s2_full = np.stack(imputed_bands, axis=-1)

        s2_full = (s2_full - np.array(OURS_S2_MEAN)) / np.array(OURS_S2_STD)
        s2_full = torch.from_numpy(s2_full).float().cuda()

        with torch.no_grad():
            patch_embeddings = model(s2=s2_full.unsqueeze(0))

        b, n, d = patch_embeddings.shape
        p = int(n**0.5)
        emb_img = patch_embeddings.view(b, p, p, d).permute(0, 3, 1, 2)[0]

        emb_resized = emb_img
        # emb_resized = process_embeddings(emb_img)

        emb_fp16 = emb_resized.cpu().numpy().astype(np.float16)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, emb_fp16)

    except Exception as e:
        print(f"Failed {tif_path}: {type(e).__name__}: {e}")


def process_country(country: str, base_dir: Path, wrapper):
    country_dir = base_dir / "ftw" / country / "s2_images"
    for window in ["window_b", "window_a"]:
        in_dir = country_dir / window
        if not in_dir.exists():
            continue
        out_dir = Path(str(in_dir).replace("FTW-Dataset/ftw", "FTW-Galileo-Embeddings"))
        out_dir = Path(str(out_dir).replace("s2_images", "galileo_base"))
        os.makedirs(out_dir, exist_ok=True)

        for fname in os.listdir(in_dir):
            if not fname.endswith(".tif"):
                continue
            in_path = in_dir / fname
            out_path = out_dir / fname.replace(".tif", ".npy")
            if out_path.exists():
                continue
            generate_embedding(str(in_path), str(out_path), wrapper)


if __name__ == "__main__":
    countries = ALL_COUNTRIES

    base_dir = Path("/path/to/FTW-Dataset")
    ckpt_base_dir = Path("/path/to/baseline_models")

    wrapper = GalileoWrapper(pretrained_path=ckpt_base_dir / "base", patch_size=4, month=6, do_pool=False).to(device)

    for country in countries:
        process_country(country, base_dir, wrapper)
