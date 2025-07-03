library(Seurat)
library(SeuratData)
library(SeuratDisk)
library(Seurat)
library(SeuratDisk)

# Read your object
seurat_obj <- readRDS("~/1Work/RoseLab/Spatial/CAR_T/data/annotated.all.cells_PricemanLab.RDS")

colnames(seurat_obj@meta.data)

unique(seurat_obj@meta.data$sctype_classification)

# or equivalently
unique(Idents(seurat_obj))

DimPlot(
  seurat_obj,
  group.by = "sctype_classification",
  label = TRUE,          # adds text labels at cluster centroids
  repel = TRUE           # prevents label overlap
)
