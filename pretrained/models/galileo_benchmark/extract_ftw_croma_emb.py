import os
import numpy as np
import torch
import rasterio
from pathlib import Path
from src.galileo import GalileoWrapper

device = "cuda" if torch.cuda.is_available() else "cpu"

s2_band_names = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B08A", "B09", "B10", "B11", "B12"
]

OURS_S2_MEAN = [
    1395.34, 1395.34, 1338.40, 1343.09, 1543.86, 2186.20, 2525.09,
    2410.33, 2750.28, 2750.28, 2234.91, 2234.91, 1474.53,
]
OURS_S2_STD = [
    917.70, 917.70, 913.29, 1092.68, 1047.22, 1048.01, 1143.69,
    1098.97, 1204.47, 1204.47, 1145.97, 1145.97, 980.24,
]

imputes = [
    ("B04", "B05"), ("B04", "B06"), ("B08", "B07"),
    ("B08", "B08A"), ("B08", "B09"), ("B08", "B10"),
    ("B08", "B11"), ("B08", "B12")
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
                new_images.append(torch.zeros_like(image_list[0]))
    return new_images


def process_embeddings(emb, size=(256, 256)):
    emb = torch.nn.functional.interpolate(
        emb.unsqueeze(0), size=size, mode="bilinear", align_corners=False
    ).squeeze(0)
    return emb


def generate_embedding(tif_path, output_dir, model):
    try:
        # Load 4-band Sentinel-2 chip
        with rasterio.open(tif_path) as src:
            img_data = src.read().astype(np.float32)  # [C, H, W]
        img_data = np.transpose(img_data, (1, 2, 0))  # [H, W, C]
        h, w, c = img_data.shape

        available_band_names = ["B04", "B03", "B02", "B08"]  # Red, Green, Blue, NIR
        image_list = [img_data[..., i] for i in range(c)]

        # --- Impute missing Sentinel-2 bands ---
        imputed_bands = impute_bands(image_list, available_band_names, imputes, s2_band_names)
        s2_full = np.stack(imputed_bands, axis=-1)

        # --- Normalize ---
        s2_full = (s2_full - np.array(OURS_S2_MEAN)) / np.array(OURS_S2_STD)
        s2_full = torch.from_numpy(s2_full).float().cuda()

        # --- Extract patch embeddings (same as your code) ---
        with torch.no_grad():
            patch_embeddings = model(s2=s2_full.unsqueeze(0))  # [1, N, D]

        b, n, d = patch_embeddings.shape
        p = int(n ** 0.5)
        emb_img = patch_embeddings.view(b, p, p, d).permute(0, 3, 1, 2)[0]  # [D, H, W]

        # --- Interpolate to 256x256 ---
        emb_resized = emb_img
        # emb_resized = process_embeddings(emb_img)

        # --- Convert to int8 ---
        # emb_int8 = np.clip(emb_resized.cpu().numpy() * 127, -128, 127).astype(np.int8)
        emb_fp16 = emb_resized.cpu().numpy().astype(np.float16)

        # --- Save ---
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.basename(tif_path).replace(".tif", ".npy")
        np.save(os.path.join(output_dir, base), emb_fp16)

        print(f"✅ Saved embedding: {base}")

    except Exception as e:
        print(f"❌ Failed {tif_path}: {type(e).__name__}: {e}")

# img_data: [B, H, W, C]
def image_2_embedding(batch_img_data, model, device='cpu'):
    # # Load 4-band Sentinel-2 chip
    # with rasterio.open(tif_path) as src:
    #     img_data = src.read().astype(np.float32)  # [C, H, W]
    #img_data = np.transpose(img_data, (1, 2, 0))  # [H, W, C]
    b, h, w, c = batch_img_data.shape

    available_band_names = ["B04", "B03", "B02", "B08"]  # Red, Green, Blue, NIR

    batched_list = []
    for img_data in batch_img_data:
        image_list = [img_data[..., i] for i in range(c)]

        # --- Impute missing Sentinel-2 bands ---
        imputed_bands = impute_bands(image_list, available_band_names, imputes, s2_band_names)
        s2_full = torch.stack(imputed_bands, dim=-1).to(device)

        # --- Normalize ---
        s2_full = (s2_full - torch.tensor(OURS_S2_MEAN, device=device)) / torch.tensor(OURS_S2_STD, device=device)
        s2_full = s2_full.float().to(device)
        batched_list.append(s2_full)
    batched_s2_full = torch.stack(batched_list, dim=0)

    # --- Extract patch embeddings (same as your code) ---
    with torch.no_grad():
        #patch_embeddings = model(s2=s2_full.unsqueeze(0))  # [1, N, D]
        patch_embeddings = model(s2=batched_s2_full) # [B, N, D]

    b, n, d = patch_embeddings.shape
    p = int(n ** 0.5)
    emb_img = patch_embeddings.view(b, p, p, d).permute(0, 3, 1, 2) # [B, D, H, W] #[0]  # [D, H, W]

    # --- Interpolate to 256x256 ---
    emb_resized = emb_img
    # emb_resized = process_embeddings(emb_img)

    # --- Convert to int8 ---
    # emb_int8 = np.clip(emb_resized.cpu().numpy() * 127, -128, 127).astype(np.int8)
    # emb_fp16 = emb_resized.cpu().numpy().astype(np.float16)

    # --- Save ---
    # os.makedirs(output_dir, exist_ok=True)
    # base = os.path.basename(tif_path).replace(".tif", ".npy")
    # np.save(os.path.join(output_dir, base), emb_fp16)
    return emb_resized

from src.eval.baseline_models import CROMAWrapper

# Load CROMA model
def wrapper():
    return CROMAWrapper(
        weights_path=Path("/u/gmuhawenayo/projects/FTW-Galileo/galileo/data/baseline_models/croma"),
        size="base",
        modality="optical",
        do_pool=False,
        load_state=False,
    )#.to(device)

def process_country(country: str):
    base_dir = f"/u/gmuhawenayo/datasets/FTW-Dataset/ftw/{country}/s2_images"
    for window in ["window_a", "window_b"]:
        in_dir = os.path.join(base_dir, window)
        if not os.path.exists(in_dir):
            continue
        out_dir = in_dir.replace("FTW-Dataset/ftw", "FTW-CROMA-Embeddings")
        out_dir = out_dir.replace("s2_images", "croma_base")
        os.makedirs(out_dir, exist_ok=True)

        for fname in os.listdir(in_dir):
            if not fname.endswith(".tif"):
                continue
            in_path = os.path.join(in_dir, fname)
            out_path = os.path.join(out_dir, fname.replace(".tif", ".npy"))
            if os.path.exists(out_path):
                continue
            generate_embedding(in_path, out_path, wrapper)

# ==============================================================
# Parallel Execution
# ==============================================================
if __name__ == "__main__":
    countries = [
        # DONE: "rwanda", "vietnam", "spain","sweden", "portugal","slovakia","slovenia","south_africa", "kenya","latvia","lithuania","luxembourg","netherlands", "kenya","latvia","lithuania","luxembourg","netherlands", "finland","france","germany","india",

        "austria", "belgium","brazil","cambodia","corsica","croatia",
        "denmark","estonia",

        # "finland","france","germany","india",
        # "kenya","latvia","lithuania","luxembourg","netherlands",
        # "portugal","rwanda","slovakia","slovenia","south_africa",
        # "spain","sweden", "vietnam",

        # "kenya","latvia","lithuania","luxembourg","netherlands", "finland","france","germany","india",
        
        
    ]

    # with ProcessPoolExecutor(max_workers=4) as pool:
    #     pool.map(process_country, countries)
    for country in countries:
        process_country(country)