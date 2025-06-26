import anndata as ad
import squidpy as sq
import cellcharter as cc
import pandas as pd
import scanpy as sc
import scvi
import numpy as np
import matplotlib.pyplot as plt
from lightning.pytorch import seed_everything
from pathlib import Path
from skimage.io import imread

import torch
torch.set_float32_matmul_precision("high")
  
seed_everything(1337)
scvi.settings.seed = 1337





# # LAPTOP
# segmentation_path = Path("/Users/janzules/Roselab/Spatial/CAR_T/data/cell_segmentation/")

# COMPUTER
segmentation_path = Path("/mnt/c/Users/jonan/Documents/1Work/RoseLab/Spatial/CAR_T/data/cell_segmentation/")
adata_file     = segmentation_path / "concatenated" / "combined_adata.h5ad"
geneList = segmentation_path / "Gene_lists"
fig_out = Path("/Users/janzules/Roselab/Spatial/CAR_T/figures/clustering_results/")
ST_sample = sc.read_h5ad(adata_file)

# ST_sample = ST_sample_org[ST_sample_org.obs['mouse'].isin([sample])].copy()
# del ST_sample_org





# C:/Users/jonan/Documents/1Work/RoseLab/Spatial/CAR_T/data/images_for_alignments/121724-121924_RL_mRT_TMA4_1_TMA5_1

# "/mnt/c/Users/jonan/Documents/1Work/RoseLab/Spatial/CAR_T/data/images_for_alignments/121724-121924_RL_mRT_TMA4_1_TMA5_1/hne/"


# # LAPTOP
# HNE_TIF_PATHS = {
#     "F07839": Path("/Users/janzules/Roselab/Spatial/dietary_project/data/images_for_alignments/121724-121924_RL_mRT_TMA4_1_TMA5_1/hne/121724_RL_mRT_TMA4_Slide_1.tif"),
#     "F07840": Path("/Users/janzules/Roselab/Spatial/dietary_project/data/images_for_alignments/121724-121924_RL_mRT_TMA4_1_TMA5_1/hne/121724_RL_mRT_TMA5_Slide_1.tif")
# }

# COMPUTER
HNE_TIF_PATHS = {
    "F07839": Path("/mnt/c/Users/jonan/Documents/1Work/RoseLab/Spatial/CAR_T/data/images_for_alignments/121724-121924_RL_mRT_TMA4_1_TMA5_1/hne/121724_RL_mRT_TMA4_Slide_1.tif"),
    "F07840": Path("/mnt/c/Users/jonan/Documents/1Work/RoseLab/Spatial/CAR_T/data/images_for_alignments/121724-121924_RL_mRT_TMA4_1_TMA5_1/hne/121724_RL_mRT_TMA5_Slide_1.tif")
}




ST_sample.obsm["spatial"] = ST_sample.obs[["cx", "cy"]].to_numpy()


# 1) Make sure your AnnData knows which “library” (TMA) each spot comes from.
#    We’ll use your existing 'TMA' column as the library_key:
ST_sample.obs['library_id'] = ST_sample.obs['TMA']

# 2) Build the minimal `adata.uns['spatial']` structure Squidpy expects:
#    For each TMA, point to your H&E image (as a numpy array) and set a scale factor.
ST_sample.uns['spatial'] = {}
for lib_id, tif_path in HNE_TIF_PATHS.items():
    img = imread(tif_path)
    ST_sample.uns['spatial'][lib_id] = {
        "images": {"hires": img},
        "scalefactors": {
            "tissue_hires_scalef": 1.0,
            "spot_diameter_fullres": 100.0,   # <- add this
            "tissue_lowres_scalef": 1.0,      # <- add this (not always needed)
        }
    }


sc.pp.filter_genes(ST_sample, min_counts=3)
sc.pp.filter_cells(ST_sample, min_counts=50)


ST_sample.layers['counts'] = ST_sample.X.copy()


sc.pp.normalize_total(ST_sample, target_sum=1e6)
sc.pp.log1p(ST_sample)





scvi.model.SCVI.setup_anndata(
    ST_sample,
    layer="counts",
    batch_key="mouse"
)

model = scvi.model.SCVI(ST_sample)


# # LAPTOP
# model.train(early_stopping=True, enable_progress_bar=True, accelerator="mps")

# COMPUTER
model.train(early_stopping=True, enable_progress_bar=True, accelerator="cuda")


ST_sample.obsm['X_scVI'] = model.get_latent_representation(ST_sample).astype(np.float32)


# sq.gr.spatial_neighbors(ST_sample, library_key='mouse', coord_type='generic', delaunay=True)
sq.gr.spatial_neighbors(ST_sample, library_key='mouse', coord_type='generic', delaunay=True, spatial_key='spatial', percentile=99)





# grabbing coords for CyPSCA_1_2

# Adjust this column name if needed
subset = ST_sample[ST_sample.obs['mouse'] == 'CyPSCA_1_2']

# Extract (cx, cy) coordinates
coords = subset.obsm['spatial']

# Compute bounding box
xmin = np.min(coords[:, 0])
xmax = np.max(coords[:, 0])
ymin = np.min(coords[:, 1])
ymax = np.max(coords[:, 1])

print(f"xmin: {xmin}, xmax: {xmax}")
print(f"ymin: {ymin}, ymax: {ymax}")



# 1. Subset AnnData for the target mouse/condition
subset = ST_sample[ST_sample.obs['mouse'] == 'CyPSCA_1_2']

# 2. Extract spatial coordinates
coords = subset.obsm['spatial']
xmin, xmax = coords[:, 0].min(), coords[:, 0].max()
ymin, ymax = coords[:, 1].min(), coords[:, 1].max()

# Optional: Add margin (e.g., 100 pixels on each side)
margin = 100
crop_coord = (
    int(xmin - margin),
    int(ymin - margin),
    int(xmax + margin),
    int(ymax + margin)
)

print("Crop coordinates:", crop_coord)



# Get the unique TMA(s) that contain this mouse
ST_sample.obs.loc[ST_sample.obs['mouse'] == 'CyPSCA_1_2', 'TMA'].unique()





sq.pl.spatial_scatter(
    ST_sample,
    shape=None,
    spatial_key='spatial',
    library_key='library_id',
    library_id=['F07840'],         # or whichever TMA sample contains CyPSCA_1_2
    crop_coord=crop_coord,
    img=True,
    img_alpha=1,
    color='condition',
    connectivity_key='spatial_connectivities',
    size=0.01,
    edges_width=0.3,
    legend_loc='right',
    title=['CyPSCA_1_2']
)



cc.gr.remove_long_links(ST_sample)


sq.pl.spatial_scatter(
    ST_sample,
    shape=None,
    spatial_key='spatial',
    library_key='library_id',
    library_id=['F07840'],         # or whichever TMA sample contains CyPSCA_1_2
    crop_coord=crop_coord,
    img=True,
    img_alpha=1,
    color='condition',
    connectivity_key='spatial_connectivities',
    size=0.01,
    edges_width=0.3,
    legend_loc='right',
    title=['CyPSCA_1_2']
)






# output = segmentation_path / "scvi_all.h5ad"


# ST_sample.write(output)





# ST_sample = sc.read_h5ad("/mnt/c/Users/jonan/Documents/1Work/RoseLab/Spatial/CAR_T/data/cell_segmentation/concatenated/scvi_all.h5ad")


# sq.gr.spatial_neighbors(ST_sample, library_key='mouse', coord_type='generic', delaunay=True, spatial_key='spatial', percentile=99)


cc.gr.aggregate_neighbors(ST_sample, n_layers=1, use_rep='X_scVI', out_key='X_cellcharter', sample_key='mouse')


# https://cellcharter.readthedocs.io/en/stable/generated/cellcharter.tl.ClusterAutoK.html?utm_source=chatgpt.com
model_params = {
    "random_state": 1337,
    "trainer_params": {
        "accelerator": "cuda",   # or "cuda" depending on PL version
        "devices": 1,          # number of GPUs to use
        "enable_progress_bar": True
    }
}


autok = cc.tl.ClusterAutoK(
    n_clusters=(2,20),
    max_runs=10,
    convergence_tol=0.001,
    model_class=cc.tl.GaussianMixture,
    model_params=model_params
)



autok.fit(ST_sample, 
          use_rep='X_cellcharter')


cc.pl.autok_stability(autok)


# ST_sample.obs['cluster_cellcharter'] = autok.predict(ST_sample, use_rep='X_cellcharter')
ST_sample.obs['cluster_cellcharter'] = autok.predict(ST_sample, use_rep='X_cellcharter', k=10) # Choose number of clusters


for tma in ST_sample.obs['TMA'].unique():
    ax = sq.pl.spatial_scatter(
        ST_sample,
        color=['cluster_cellcharter'],
        library_key='library_id',
        size=0.23,
        img=False,
        # img_res_key = "hires",
        spatial_key='spatial',
        palette='tab10',
        figsize=(5, 5),
        ncols=1,
        library_id=[tma],
        return_ax=True
    )

    # Save and close
    outpath = f"/mnt/c/Users/jonan/Documents/1Work/RoseLab/Spatial/CAR_T/CellCharterAnalysis/onefer/figures/spatial_scatter_{tma}.png"
    ax.figure.savefig(outpath, dpi=600, bbox_inches='tight')
    plt.close(ax.figure)  # <- Suppress inline display



sq.pl.spatial_scatter(
    ST_sample, 
    color=['cluster_cellcharter'], 
    library_key='library_id',  
    size=0.3, 
    img=None,
    spatial_key='spatial',
    palette='tab10',
    figsize=(5,5),
    ncols=1,
    library_id=['F07839']
)


sq.pl.spatial_scatter(
    ST_sample, 
    color=['cluster_cellcharter'], 
    library_key='library_id',  
    size=0.3, 
    img=None,
    spatial_key='spatial',
    palette='Set3',
    figsize=(5,5),
    ncols=1,
    library_id=['F07840']
)

















ST_sample.write("/mnt/c/Users/jonan/Documents/1Work/RoseLab/Spatial/CAR_T/CellCharterAnalysis/onefer/onefer.h5ad")








ST_sample.obs.columns


ST_sample.obs['condition'].unique()








from cellcharter.tl import GaussianMixture

# Make sure you use "gpu" (not "cuda") for Lightning v2.x
gm = GaussianMixture(
    n_clusters=3,
    trainer_params={
        "accelerator": "gpu",   # for PL 2.x, "gpu" is correct
        "devices": 1,           # how many GPUs to use
        "enable_progress_bar": False
    }
)



trainer = gm.trainer()   # this returns the Lightning Trainer configured for GM
print("accelerator:", trainer.accelerator)
print("device_ids:", trainer.device_ids)
print("root_device:", trainer._device_type)






import cellcharter
print(cellcharter.__version__)



import torch, lightning.pytorch as pl
print(torch.__version__, pl.__version__)


# https://cellcharter.readthedocs.io/en/stable/generated/cellcharter.tl.ClusterAutoK.html?utm_source=chatgpt.com
model_params = {
    "random_state": 12345,
    "trainer_params": {
        "accelerator": "cuda",   # or "cuda" depending on PL version
        "devices": 1,          # number of GPUs to use
        "enable_progress_bar": True
    }
}


autok = cc.tl.ClusterAutoK(
    n_clusters=(2,20),
    max_runs=10,
    convergence_tol=0.001,
    model_class=cc.tl.GaussianMixture,
    model_params=model_params
)



autok.fit(ST_sample, 
          use_rep='X_cellcharter')
