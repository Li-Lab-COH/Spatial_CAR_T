library(Seurat)
# library(SeuratData)
# library(SeuratDisk)
library(dplyr)
library(ggplot2)


# Read your object
base_dir <- "~/1Work/RoseLab/Spatial/CAR_T"
fig_out_dir <- file.path(base_dir, "figures/sc_reference_data/")

rds_file <- file.path(base_dir, "data/sc-referenceannotated.all.cells_PricemanLab.RDS")

fit_out <- 
seurat_obj <- readRDS(rds_file)

#-------------------------- Sanity check ------------------------------------

# Assume your object is called `seurat_obj` (change if needed)
# Quick sanity checks:
colnames(seurat_obj@meta.data) %>% head()
table(seurat_obj@meta.data$sctype_classification, useNA = "ifany")

table(seurat_obj@meta.data$treatment)

# Plot UMAP colored by your cell-type labels
type_umap <- DimPlot(
  seurat_obj,
  reduction = "umap.integrated",
  group.by = "sctype_classification",
  label = TRUE,                # draws cluster labels at centroids of groups
  repel = TRUE,                # nicer label placement
  pt.size = 0.3
)

ggsave(
  filename = file.path(fig_out_dir, "umap_celltypes.png"),
  plot = type_umap,
  width = 8, height = 6, dpi = 300
)
#------------------------------ Visualizing ----------------------------------

genes_available <- rownames(seurat_obj)
spp1_gene <- intersect(genes_available, c("SPP1","Spp1","spp1"))[1]
cd44_gene <- intersect(genes_available, c("CD44","Cd44","cd44"))[1]
glp1r_gene <- intersect(genes_available, c("GLP1R","Glp1r","glp1r"))[1]

spp1_gene; cd44_gene; glp1r_gene  # confirm they resolved


gene_plot <- FeaturePlot(
  seurat_obj,
  features = na.omit(c(spp1_gene, cd44_gene)),
  reduction = "umap.integrated",
  pt.size = 0.3,
  order = TRUE
)

ggsave(
  filename = file.path(fig_out_dir, "spp1_cd44_genes.png"),
  plot = gene_plot,
  width = 8, height = 6, dpi = 300
)

#--------------------------- GLP1R ---------------------------------------
c("Gipr", "Gcg", "Dpp4", "Adcy", "Creb1") %in% genes_available


Gene_list_glp1_related <- c("Gipr", "Gcg", "Dpp4", "Creb1")

adcy_mouse1 <- paste0("Adcy", 1:4)
adcy_mouse2 <- paste0("Adcy", 5:9)
adcy_mouse %in% genes_available

glp1_related <- FeaturePlot(
  seurat_obj,
  features = Gene_list_glp1_related,
  reduction = "umap.integrated",
  pt.size = 0.3,
  order = TRUE
)

ggsave(
  filename = file.path(fig_out_dir, "glp1_related.png"),
  plot = glp1_related,
  width = 8, height = 6, dpi = 300
)




#------------------------ checking genes-----------------------------------
check_genes_case_insensitive <- function(gene_list, query_genes) {
  # Lowercase everything
  gene_list_lower <- tolower(gene_list)
  query_lower <- tolower(query_genes)
  
  # Check membership in lowercase space
  present <- query_lower %in% gene_list_lower
  
  # Name result with original queries
  names(present) <- query_genes
  
  return(present)
}

# Example usage:
check_genes_case_insensitive(genes_available, c("KRT5", "nkx3", "PsCa"))


find_genes_starting_with <- function(gene_list, prefix) {
  # gene_list: vector of gene names (e.g., rownames from Seurat object)
  # prefix: string to search for at the start of the gene name
  
  matches <- grep(paste0("^", prefix), gene_list, ignore.case = TRUE, value = TRUE)
  return(matches)
}

# Example usage:
# genes_available <- rownames(seurat_obj)
find_genes_starting_with(genes_available, "rm")

FeaturePlot(
  seurat_obj,
  features = "Ar",
  reduction = "umap.integrated",
  pt.size = 0.3,
  order = TRUE
)

