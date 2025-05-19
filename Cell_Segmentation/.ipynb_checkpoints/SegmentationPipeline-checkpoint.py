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
import sys

"""
TODOs:

- [] Set up a config file parser for the model parameters, outputs
- [] Save the ID ranges for each sample and then sub samples
- [] Remove "DIFF" comments if everything works out - These are sections that do something slightly different
        Than the original code, but seems to do the same thing
- [] Why y,x format?

"""


# ---------------------------------
# Configuration
# ---------------------------------
SAMPLES = [
    'BANOSSM_SSM0015_1_PR_Whole_C1_VISHD_F07839_22WJCYLT3',
    'BANOSSM_SSM0015_1_PR_Whole_C1_VISHD_F07840_22WJCYLT3'
]

HNE_TIF_PATHS = {
    "F07839": Path("/mnt/c/Users/jonan/Documents/1Work/RoseLab/Spatial/dietary_droject/data/images_for_alignments/121724-121924_RL_mRT_TMA4_1_TMA5_1/hne/121724_RL_mRT_TMA4_Slide_1.tif"),
    "F07840": Path("/mnt/c/Users/jonan/Documents/1Work/RoseLab/Spatial/dietary_droject/data/images_for_alignments/121724-121924_RL_mRT_TMA4_1_TMA5_1/hne/121724_RL_mRT_TMA5_Slide_1.tif")
}


BASE_DIR = Path("/mnt/c/Users/jonan/Documents/1Work/RoseLab/Spatial/CAR_T/data/Raw")
SEGMENTATION_PATH = Path("/mnt/c/Users/jonan/Documents/1Work/RoseLab/Spatial/CAR_T/data/cell_segmentation")

# StarDist parameters
MIN_PERCENTILE = 5
MAX_PERCENTILE = 95
MODEL_SCALE = 2
NMS_THRESHOLD = 0.1
PROB_THRESHOLD = 0.33
N_TILES = (80,80,1)
# StarDist “big” segmentation params (tunable to fit your GPU)
BLOCK_SIZE = 4096         # pixel width/height of each tile
MIN_OVERLAP_BIG = 128     # how much each tile overlaps its neighbor
CONTEXT = 64              # extra border to avoid edge artifacts


# QC thresholds
AREA_CUTOFF = 100
UMI_CUTOFF = 50
UMI_SPATIAL_CUTOFF = 50

# Logging setup
LOG_FILE = SEGMENTATION_PATH / "pipeline.log"
SEGMENTATION_PATH.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s]: %(message)s", #The colon appearsafter all the previous info
    handlers=[ #Loggers generate messages. Handlers decide what to do with them.
        logging.FileHandler(LOG_FILE), # Sends log message to the log file
        logging.StreamHandler() # Sends to a stream, usually sys.stderr
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

    logging.info(f'Loading parquet file for {sample}')
    df_pos = pd.read_parquet(str(pq_file)).set_index("barcode") #DIFF - .set_index is on another line df_tissue_positions['index']=df_tissue_positions.index
    df_pos["index"] = df_pos.index # To check joins
    adata.obs =  pd.merge(adata.obs, df_pos, left_index=True, right_index=True)

    # Creates a geomotries object that contains tuples of the coordiates - used later
    geometries = [Point(xy) for xy in zip(df_pos["pxl_col_in_fullres"], # DIFF - geomtry instead of geometries
                                          df_pos["pxl_row_in_fullres"])]
    gdf_coords = gpd.GeoDataFrame(df_pos, geometry=geometries)
    logging.info(f'Data loaded {sample}')
    return adata, gdf_coords
#CHECKED

def segment_nuclei(sample: str):
    # Load full-resolution RGB H&E image from dictionary
    sample_id = re.search(r'F\d{5}', sample).group(0)
    logging.info(f'Starting segmentation {sample_id}')
    img_path = HNE_TIF_PATHS[sample_id]
    
    if not img_path.exists():
        logging.critical(f"Image file not found for sample {sample_id}: {img_path}")
        sys.exit(1)
    
    
    
    logging.info(f"Loading full-res image {img_path}")
    img_np = tifffile.imread(str(img_path))  # full-res TIFF loaded as (H, W, 3)

    # Normalize intensities
    img_norm = normalize(img_np, MIN_PERCENTILE, MAX_PERCENTILE, axis=(0,1,2))

    # Load model once
    model = StarDist2D.from_pretrained("2D_versatile_he")

    logging.info(
        f"Running StarDist.predict_instances_big with block_size={BLOCK_SIZE}, "
        f"min_overlap={MIN_OVERLAP_BIG}, context={CONTEXT}"
    )
    labels, polys = model.predict_instances_big(
        img_norm,
        axes='YXC',
        block_size=BLOCK_SIZE,
        min_overlap=MIN_OVERLAP_BIG,
        context=CONTEXT,
        prob_thresh=PROB_THRESHOLD,
        nms_thresh=NMS_THRESHOLD
    )
    logging.info("Stardist (big) segmentation complete")

    # Clear TensorFlow session to free memory
    K.clear_session()
    gc.collect()

    return img_np, labels, polys


def make_geodataframe(polys: dict, sample_id: str, offset_id: int = 0):
    logging.info('Starting make_geodataframe from model output')
    geometries = []
    for nuclei in range(len(polys['coord'])):
        ## DIFF this was the original, but it's actually not that different from the vignette
        # ys, xs = polys["coord"][i] It's possible that not specifying [0] or [1] could have messed things up?
        # coords = [(y, x) for x, y in zip(xs, ys)]
        
        
        # Extracting coordinates for the current nuclei and converting them to (y, x) format
        # Pixels work in y,x while everything else works in x,y (Cartesian (geometry))
        coords = [(y, x) for x, y in zip(polys['coord'][nuclei][0], polys['coord'][nuclei][1])]
        # Creating a Polygon geometry from the coordinates
        geometries.append(Polygon(coords))
        
    gdf = gpd.GeoDataFrame(geometry=geometries)
    gdf["id"] = [f"ID_{offset_id + i + 1}" for i in range(len(gdf))]
    gdf["area"] = gdf.geometry.area
    next_offset = offset_id + len(gdf)

    # Save ID range to a txt file
    id_range_path = SEGMENTATION_PATH / f"{sample_id}_offset.txt"
    with open(id_range_path, "w") as f:
        f.write(f"{sample_id}: ID_{offset_id + 1} to ID_{next_offset}\n")

    logging.info('Created geodataframe')
    return gdf, next_offset

def bin_and_sum(adata, gdf_coords, gdf, sample_id: str):
    # adata - gene expression data associated to a barcode which is a bin
    # gdf_coords - from the parquet file, associates pixels to barcodes
    # gdf - polygon gdf from model
    logging.info('Performing spatial join and summation')
    
    # Identify which coordinates from the parquet file are in a cell nucleus from the model output
    # It adds a column "index_right" to gdf_coords (renamed to join), which refers to the matching row in gdf
    join = gpd.sjoin(gdf_coords, gdf, how="left", predicate="within")

    # Removes barcodes that are not within a nucleus
    join["is_within_polygon"] = ~join["index_right"].isna()
    
    # overlaps = pd.unique(join[join.duplicated(subset=["index"])].index) - original
    overlaps = pd.unique(join[join.duplicated(subset=["index"])]['index'])
    join["is_not_overlap"] = ~join["index"].isin(overlaps)

    # Removing polygons that overlap - True within a polyn, true that not in overlap
    # good = join[join.is_within_polygon & join.is_not_overlap]
    good = join[join['is_within_polygon'] & 
                join['is_not_overlap']
    ]


    # identify barcodes in adata that are in nuclei, non-overlapping, polygons
    mask = adata.obs_names.isin(good["index"])
    
    # filtered = adata[mask, :].copy() - original
    filtered = adata[mask, :]

    # Save barcode -> nucleus ID mapping
    barcode_nucleus_mapping = good[['index', 'id']].copy()
    barcode_nucleus_mapping.to_csv(SEGMENTATION_PATH / f"{sample_id}_barcode_to_nucleus_mapping.csv", index=False)
    
    
    # filtered.obs = filtered.obs.join(good.set_index("index")[['geometry', 'id']], how='left')
    filtered.obs = pd.merge(
        filtered.obs,
        good[['index','geometry','id','is_within_polygon','is_not_overlap']],
        left_index=True, right_index=True
    )
    logging.info('Saved barcode and id mapping' )
    
    #==============================================================================================#
    # summation
    #==============================================================================================#
    logging.info(f'Total barcodes before filtering: {adata.n_obs}')
    logging.info(f'Barcodes retained after spatial join and filtering: {filtered.n_obs}')
    
    
    groupby_object = filtered.obs.groupby(["id"], observed=True)

    # Extract the gene expression counts
    counts = filtered.X

    # Obtain the number of unique nuclei and the number of genes in the expression data
    N_groups = groupby_object.ngroups
    logging.info(f'Number of unique nuclei with barcodes assigned: {N_groups}')
    N_genes = counts.shape[1]
    logging.info(f'Number of genes: {N_genes}')

    # Initialize a sparse matrix to store the summed gene counts for each nucleus
    summed_counts = sparse.lil_matrix((N_groups, N_genes))
    
    # Lists to store the IDs of polygons and the current row index
    polygon_id = []
    row = 0

    # Iterate over each unique polygon to calculate the sum of gene counts.
    for polygons, idx_ in groupby_object.indices.items():
        summed_counts[row] = counts[idx_].sum(0)
        row += 1
        polygon_id.append(polygons)
        

    # Create and AnnData object from the summed count matrix
    summed_counts = summed_counts.tocsr()
    gf_adata = anndata.AnnData(
        X=summed_counts,
        obs=pd.DataFrame(polygon_id, columns=['id'], index=polygon_id),
        var=filtered.var
    )
    sc.pp.calculate_qc_metrics(gf_adata, inplace=True)
    # N = len(groups)
    # G = filtered.X.shape[1]
    # mat = sparse.lil_matrix((N, G))
    # ids = []
    # for i,(poly, idxs) in enumerate(groups.items()):
    #     mat[i] = filtered.X[idxs].sum(0)
    #     ids.append(poly)
    # mat = mat.tocsr()
    # gf_adata = anndata.AnnData(X=mat,
    #                            obs=pd.DataFrame(ids, columns=["id"], index=ids),
    #                            var=filtered.var)
    # sc.pp.calculate_qc_metrics(gf_adata, inplace=True)
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
        image_shape=img_np.shape,
        block_size=BLOCK_SIZE,
        min_overlap=MIN_OVERLAP_BIG,
        context=CONTEXT
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

def process_sample(sample: str, offset_id: int, sample_id: str):
    
    adata, gdf_coords = load_spatial_data(sample)
    
    img_np, labels, polys = segment_nuclei(sample)
    
    gdf, new_offset = make_geodataframe(polys, sample_id, offset_id)
    
    gf_adata = bin_and_sum(adata, gdf_coords, gdf, sample_id)
    
    save_outputs(sample, img_np, labels, polys, gdf, gf_adata)
    return new_offset

def main():
    offset = 0
    for s in SAMPLES:
        s_id = re.search(r'_F\d+_', s).group(0).strip('_')
        logging.info(f"Starting sample: {s_id}, offeset: {offset}")
        offset = process_sample(s, offset, s_id)

if __name__ == "__main__":
    main()
