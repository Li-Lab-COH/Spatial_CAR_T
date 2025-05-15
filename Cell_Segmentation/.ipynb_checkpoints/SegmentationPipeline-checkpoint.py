#!/usr/bin/env python3
"""
segmentation_pipeline.py

Batch nuclei segmentation and Visium HD custom binning pipeline.
Run under your `spatial-nuclei` conda environment:

    python3 segmentation_pipeline.py

"""

import os
import re
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import scanpy as sc
import anndata
import tifffile
from PIL import Image
import matplotlib.pyplot as plt
from stardist.models import StarDist2D
from csbdeep.utils import normalize
from shapely.geometry import Polygon, Point
from scipy import sparse
from tensorflow.keras import backend as K
import gc

# ---------------------------------
# Configuration
# ---------------------------------
SAMPLES = [
    #"BANOSSM_SSM0015_1_PR_Whole_C1_VISHD_F07833_22WJCYLT3",
    #"BANOSSM_SSM0015_1_PR_Whole_C1_VISHD_F07834_22WJCYLT3",
    "BANOSSM_SSM0015_1_PR_Whole_C1_VISHD_F07835_22WJCYLT3_swapped",
    "BANOSSM_SSM0015_1_PR_Whole_C1_VISHD_F07836_22WJCYLT3_swapped",
    #"BANOSSM_SSM0015_1_PR_Whole_C1_VISHD_F07837_22WJCYLT3",
    # "BANOSSM_SSM0015_1_PR_Whole_C1_VISHD_F07838_22WJCYLT3",
]

HNE_TIF_PATHS = {
    "F07833": Path("/mnt/c/Users/jonan/Documents/1Work/RoseLab/Spatial/dietary_droject/data/images_for_alignments/111324-111424_Training_TMA2/hne/MycCAP_TMA2_Slide2.tif"),
    "F07834": Path("/mnt/c/Users/jonan/Documents/1Work/RoseLab/Spatial/dietary_droject/data/images_for_alignments/111324-111424_Training_TMA2/hne/MycCAP_TMA2_Slide1.tif"),
    "F07835": Path("/mnt/c/Users/jonan/Documents/1Work/RoseLab/Spatial/dietary_droject/data/images_for_alignments/120524-120624_MycCap_TMA1_1_TMA3_1/hne/MycCap_TMA1_slide1.tif"),
    "F07836": Path("/mnt/c/Users/jonan/Documents/1Work/RoseLab/Spatial/dietary_droject/data/images_for_alignments/120524-120624_MycCap_TMA1_1_TMA3_1/hne/RL_MycCap_TMA3_slide1.tif"),
    "F07837": Path("/mnt/c/Users/jonan/Documents/1Work/RoseLab/Spatial/dietary_droject/data/images_for_alignments/121124-121224_SKO_TMA1_1_n_TMA2_1/hne/12_11_2024_RL_SKOTMA1_Slide_1.tif"),
    "F07838": Path("/mnt/c/Users/jonan/Documents/1Work/RoseLab/Spatial/dietary_droject/data/images_for_alignments/121124-121224_SKO_TMA1_1_n_TMA2_1/hne/12_11_2024_RL_SKOTMA2 Slide_1.tif"),
}


BASE_DIR = Path("/mnt/c/Users/jonan/Documents/1Work/RoseLab/Spatial/dietary_droject/data/Rose_Li_VisiumHD")
SEGMENTATION_PATH = Path("/mnt/c/Users/jonan/Documents/1Work/RoseLab/Spatial/dietary_droject/data/cell_segmentation")

# StarDist parameters
MIN_PERCENTILE = 5
MAX_PERCENTILE = 95
MODEL_SCALE = 2
NMS_THRESHOLD = 0.1
PROB_THRESHOLD = 0.33
N_TILES = (80,80,1)

# QC thresholds
AREA_CUTOFF = 60
UMI_CUTOFF = 50
UMI_SPATIAL_CUTOFF = 50

# Logging setup
LOG_FILE = SEGMENTATION_PATH / "pipeline.log"
SEGMENTATION_PATH.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def load_spatial_data(sample: str):
    spatial_dir = BASE_DIR / sample / "outs" / "binned_outputs" / "square_002um"
    h5_file = spatial_dir / "filtered_feature_bc_matrix.h5"
    pq_file = spatial_dir / "spatial" / "tissue_positions.parquet"

    logging.info(f'Loading spatial data for {sample}')
    adata = sc.read_10x_h5(str(h5_file))
    adata.var_names_make_unique()

    df_pos = pd.read_parquet(str(pq_file)).set_index("barcode")
    df_pos["index"] = df_pos.index
    adata.obs =  pd.merge(adata.obs, df_pos, left_index=True, right_index=True)


    geometries = [Point(xy) for xy in zip(df_pos["pxl_col_in_fullres"],
                                          df_pos["pxl_row_in_fullres"])]
    gdf_coords = gpd.GeoDataFrame(df_pos, geometry=geometries)
    return adata, gdf_coords
#CHECKED

def segment_nuclei(sample: str):
    # Load full-resolution RGB H&E image from dictionary
    sample_id = re.search(r'F\d{5}', sample).group(0)
    img_path = HNE_TIF_PATHS[sample_id]
    logging.info(f"Loading full-res image {img_path}")
    img_np = tifffile.imread(str(img_path))  # full-res TIFF loaded as (H, W, 3)

    # Normalize intensities
    img_norm = normalize(img_np, MIN_PERCENTILE, MAX_PERCENTILE, axis=(0,1,2))

    # Run StarDist segmentation
    model = StarDist2D.from_pretrained("2D_versatile_he")
    logging.info("Running StarDist segmentation on full-res image")
    labels, polys = model.predict_instances(
        img_norm,
        scale=MODEL_SCALE,
        n_tiles=N_TILES,
        nms_thresh=NMS_THRESHOLD,
        prob_thresh=PROB_THRESHOLD,
        show_tile_progress=True
    )

    # Clear TensorFlow session to free memory
    K.clear_session()
    gc.collect()

    return img_np, labels, polys


def make_geodataframe(polys: dict, offset_id: int = 0):
    geometries = []
    for i in range(len(polys["coord"])):
        ys, xs = polys["coord"][i]
        coords = [(y, x) for x, y in zip(xs, ys)]
        geometries.append(Polygon(coords))
    gdf = gpd.GeoDataFrame(geometry=geometries)
    gdf["id"] = [f"ID_{offset_id + i + 1}" for i in range(len(gdf))]
    gdf["area"] = gdf.geometry.area
    next_offset = offset_id + len(gdf)
    return gdf, next_offset

def bin_and_sum(adata, gdf_coords, gdf):
    logging.info('Performing spatial join and summation')
    join = gpd.sjoin(gdf_coords, gdf, how="left", predicate="within")
    join["is_within_polygon"] = ~join["index_right"].isna()
    overlaps = pd.unique(join[join.duplicated(subset=["index"])].index)
    join["is_not_overlap"] = ~join["index"].isin(overlaps)

    good = join[join.is_within_polygon & join.is_not_overlap]
    mask = adata.obs_names.isin(good["index"])
    filtered = adata[mask, :].copy()
    filtered.obs = filtered.obs.join(good.set_index("index")[['geometry', 'id']], how='left')

    # Summation
    groups = filtered.obs.groupby("id", observed=True).indices
    N = len(groups)
    G = filtered.X.shape[1]
    mat = sparse.lil_matrix((N, G))
    ids = []
    for i,(poly, idxs) in enumerate(groups.items()):
        mat[i] = filtered.X[idxs].sum(0)
        ids.append(poly)
    mat = mat.tocsr()
    gf_adata = anndata.AnnData(X=mat,
                               obs=pd.DataFrame(ids, columns=["id"], index=ids),
                               var=filtered.var)
    sc.pp.calculate_qc_metrics(gf_adata, inplace=True)
    return gf_adata

def save_outputs(sample: str, img_np, labels, polys, gdf, gf_adata):
    sample_id = re.search(r'_F\d+_', sample).group(0).strip('_')
    out_root = SEGMENTATION_PATH / sample_id
    mdl_out = out_root / "model_output"
    fig_out = out_root / "figures"
    for d in [out_root, mdl_out, fig_out]:
        ensure_dir(d)

    # 1) Model outputs
    tifffile.imwrite(mdl_out / f"{sample_id}_labels.tif", labels.astype("uint16"))
    with open(mdl_out / f"{sample_id}_polys.pkl", "wb") as f:
        pickle.dump(polys, f)
    params = dict(
        model="2D_versatile_he",
        scale=MODEL_SCALE,
        nms_threshold=NMS_THRESHOLD,
        prob_threshold=PROB_THRESHOLD,
        min_percentile=MIN_PERCENTILE,
        max_percentile=MAX_PERCENTILE,
        image_shape=img_np.shape
    )
    with open(mdl_out / f"{sample_id}_params.json", "w") as f:
        json.dump(params, f, indent=4)

    # 2) Save H&E and overlay
    Image.fromarray(img_np).save(fig_out / f"{sample_id}_hne_image.png")
    plt.figure(figsize=(15,15))
    plt.imshow(img_np)
    plt.imshow(labels, cmap="jet", alpha=0.22)
    plt.axis("off")
    plt.title("StarDist Segmentation Over H&E")
    plt.savefig(fig_out / f"{sample_id}_overlay.png", dpi=1200, bbox_inches="tight", pad_inches=0)
    plt.close()

    # 3) Save gdf
    gdf.to_file(out_root / f"{sample_id}_gdf.gpkg", driver="GPKG")

    # 4) Save AnnData
    gf_adata.write(out_root / f"{sample_id}_grouped_filtered_adata.h5ad")

    # 5) QC plots: area distribution
    fig, axs = plt.subplots(1,2,figsize=(15,4))
    axs[0].hist(gdf.area, bins=50, edgecolor="black")
    axs[0].set_title("Nuclei Area")
    axs[1].hist(gdf[gdf.area<AREA_CUTOFF].area, bins=50, edgecolor="black")
    axs[1].set_title(f"Nuclei Area < {AREA_CUTOFF}")
    fig.savefig(fig_out / f"{sample_id}_nuclei_area_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 6) UMI distribution
    fig, axs = plt.subplots(1,2,figsize=(12,4))
    axs[0].boxplot(gf_adata.obs["total_counts"], vert=False)
    axs[0].set_title("Total UMI counts (all)")
    axs[1].boxplot(gf_adata.obs["total_counts"][gf_adata.obs["total_counts"]>UMI_CUTOFF], vert=False)
    axs[1].set_title(f"Total UMI > {UMI_CUTOFF}")
    fig.savefig(fig_out / f"{sample_id}_umi_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 7) Spatial UMI maps
    gdf_umi = gdf.merge(gf_adata.obs[["total_counts"]], left_on="id", right_index=True)
    for cutoff, suffix in [(None, "all"), (UMI_SPATIAL_CUTOFF, f">_{UMI_SPATIAL_CUTOFF}")]:
        df = gdf_umi if cutoff is None else gdf_umi[gdf_umi["total_counts"]>cutoff]
        fig, ax = plt.subplots(figsize=(10,10))
        df.plot(column="total_counts", cmap="inferno", legend=True, linewidth=0.1, edgecolor="black", ax=ax)
        ax.set_title(f"UMI per nucleus ({suffix})")
        ax.axis("off")
        fig.savefig(fig_out / f"{sample_id}_umi_spatial_{suffix}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    logging.info(f"Saved all outputs for {sample_id} in {out_root}")

def process_sample(sample: str, offset_id: int):
    adata, gdf_coords = load_spatial_data(sample)
    
    img_np, labels, polys = segment_nuclei(sample)
    
    gdf, new_offset = make_geodataframe(polys, offset_id)
    
    gf_adata = bin_and_sum(adata, gdf_coords, gdf)
    
    save_outputs(sample, img_np, labels, polys, gdf, gf_adata)
    return new_offset

def main():
    offset = 0
    for s in SAMPLES:
        offset = process_sample(s, offset)

if __name__ == "__main__":
    main()
