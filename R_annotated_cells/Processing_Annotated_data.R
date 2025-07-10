library(Seurat)
library(SeuratData)
library(SeuratDisk)
library(Seurat)
library(SeuratDisk)

# Read your object
seurat_obj <- readRDS("~/1Work/RoseLab/Spatial/CAR_T/data/annotated.all.cells_PricemanLab.RDS")

colnames(seurat_obj@meta.data)
table(seurat_obj$sctype_classification)


barplot(
  sort(table(seurat_obj@meta.data$sctype_classification), decreasing = TRUE),
  las = 2,             # rotation
  cex.names = 0.7,     # text size
  col = "steelblue"
)

# or equivalently
unique(Idents(seurat_obj))

DimPlot(
  seurat_obj,
  group.by = "sctype_classification",
  label = TRUE,          # adds text labels at cluster centroids
  repel = TRUE           # prevents label overlap
)
