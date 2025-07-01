library(Seurat)
library(SeuratData)
library(SeuratDisk)
library(Seurat)
library(SeuratDisk)

# 1. Read your object
seurat_obj <- readRDS("~/1Work/RoseLab/Spatial/CAR_T/data/annotated.all.cells_PricemanLab.RDS")

# 2. Keep only the RNA assay
DefaultAssay(seurat_obj) <- "RNA"
seurat_obj@assays <- seurat_obj@assays["RNA"]  # drop everything but RNA

# 3. Drop ALL dimensionality reductions
#    (assign an empty list, not NULL)
seurat_obj@reductions <- list()

# 4. Optionally trim out unwanted assay slots (e.g. scale.data) to save space
seurat_obj@assays$RNA@scale.data <- NULL

# 5. Save & convert, allowing overwrite
out_h5seurat <- "~/1Work/RoseLab/Spatial/CAR_T/data/carT_reference_pruned.h5Seurat"
out_h5ad      <- "~/1Work/RoseLab/Spatial/CAR_T/data/carT_reference_pruned.h5ad"

SaveH5Seurat(
  seurat_obj,
  filename  = out_h5seurat,
  overwrite = TRUE
)

Convert(
  out_h5seurat,
  dest      = "h5ad",
  filename  = out_h5ad,
  overwrite = TRUE
)
