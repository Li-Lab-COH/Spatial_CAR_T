library(Seurat)
library(SeuratData)
library(SeuratDisk)


# Read your object
# base_dir <- "~/1Work/RoseLab/Spatial/CAR_T/data/sc-reference"
base_dir <- "~/Roselab/Spatial/CAR_T/data/sc-reference/"
rds_file <- file.path(base_dir, "annotated.all.cells.RDS")

seurat_obj <- readRDS(rds_file)


### ----------------------- Counts and tables ----------------------------------

# preparing file location
counts_file <- file.path(base_dir, "counts.mtx")
features_file <- file.path(base_dir, "features.tsv")
barcodes_file <- file.path(base_dir, "barcodes.tsv")

# Grabing counts files and saving
counts <- LayerData(object = seurat_obj, layer = "counts")
Matrix::writeMM(counts, file = counts_file)

# saving gene and barcodes
write.table(data.frame(rownames(counts)), file = features_file,
            row.names = FALSE, col.names = FALSE, sep = "\t", quote = FALSE)

write.table(data.frame(colnames(counts)), file = barcodes_file,
            row.names = FALSE, col.names = FALSE, sep = "\t", quote = FALSE)
            
# saving metadata
write.csv(seurat_obj@meta.data, file.path(base_dir, "metadata.csv"))
### ---------------------- Embedding and annotations ---------------------------

cell_type_file <- file.path(base_dir, "cell_type.csv")

# pulling cell type annotations
barcodes <- colnames(seurat_obj)
cell_types <- seurat_obj@meta.data$sctype_classification
cell_type_df <- data.frame(Barcode = barcodes, cell_type = cell_types)


# saving
write.csv(cell_type_df, cell_type_file, row.names = FALSE)


### -----------------------------UMAP coords-----------------------------------

# file loc
umap_integrated_file <- file.path(base_dir, "umap_coords.csv")

umap_coords <- Embeddings(seurat_obj, reduction = "umap.integrated")

umap_df <- data.frame(Barcode = rownames(umap_coords),
                      UMAP_1 = umap_coords[,1],
                      UMAP_2 = umap_coords[,2])

write.csv(umap_df, umap_integrated_file, row.names=FALSE)




### ----------------------- Visualizing ---------------------------------------



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
DimPlot(
  seurat_obj,
  group.by = "sctype_classification",
  label = TRUE,          # adds text labels at cluster centroids
  repel = TRUE,           # prevents label overlap
  reduction = "umap.integrated"
)


Reductions(seurat_obj)

###########
# Clear all variables

rm(list= ls())


########## Sanity Checks ##############

cell_barcodes <- c(
  "56955_TGCAGATGTGAGCAGT-1",
  "56971_GTATTTCGTCGTCAGC-1",
  "56971_ACTGTGATCTACCACC-1"
)

library(Matrix)

counts <- LayerData(seurat_obj, layer = "counts")
n_top <- 10

# Ensure barcodes exist
valid_barcodes <- cell_barcodes[cell_barcodes %in% colnames(counts)]
if (length(valid_barcodes) != length(cell_barcodes)) {
  warning("Some barcodes not found in the object:\n",
          paste(setdiff(cell_barcodes, valid_barcodes), collapse = ", "))
}

for (cell_name in valid_barcodes) {
  
  vec <- counts[, cell_name]          # sparse column for this barcode
  vec_dense <- as.numeric(vec)        # convert to numeric vector
  
  # order descending
  ord <- order(vec_dense, decreasing = TRUE)
  top_idx <- ord[1:n_top]
  
  top_genes <- rownames(counts)[top_idx]
  top_vals  <- vec_dense[top_idx]
  
  cat("\n=== Cell:", cell_name, "===\n")
  
  cat("Annotations:\n")
  print(seurat_obj@meta.data[cell_name, , drop = FALSE])
  
  cat("\nTop genes:\n")
  print(data.frame(gene = top_genes, count = top_vals))
}



