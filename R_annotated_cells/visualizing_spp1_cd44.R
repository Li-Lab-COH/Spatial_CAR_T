library(Seurat)
# library(SeuratData)
# library(SeuratDisk)
library(dplyr)


# Read your object
base_dir <- "~/1Work/RoseLab/Spatial/CAR_T/data/sc-reference"
rds_file <- file.path(base_dir, "annotated.all.cells_PricemanLab.RDS")

seurat_obj <- readRDS(rds_file)

#-------------------------- Sanity check ------------------------------------

# Assume your object is called `seurat_obj` (change if needed)
# Quick sanity checks:
colnames(seurat_obj@meta.data) %>% head()
table(seurat_obj@meta.data$sctype_classification, useNA = "ifany")

# Plot UMAP colored by your cell-type labels
DimPlot(
  seurat_obj,
  reduction = "umap.integrated",
  group.by = "sctype_classification",
  label = TRUE,                # draws cluster labels at centroids of groups
  repel = TRUE,                # nicer label placement
  pt.size = 0.3
)


#------------------------------ Visualizing ----------------------------------

genes_available <- rownames(seurat_obj)
spp1_gene <- intersect(genes_available, c("SPP1","Spp1","spp1"))[1]
cd44_gene <- intersect(genes_available, c("CD44","Cd44","cd44"))[1]

spp1_gene; cd44_gene  # confirm matches


FeaturePlot(
  seurat_obj,
  features = c(spp1_gene, cd44_gene),
  reduction = "umap.integrated",
  pt.size = 0.3,
  order = TRUE
)





