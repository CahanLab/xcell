# Convert a Seurat .rds file to .h5ad via SeuratDisk
# Usage: Rscript convert_seurat.R <input.rds> <output.h5seurat>
# Produces <output.h5ad> alongside the .h5seurat file

library(Seurat)
library(SeuratDisk)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("Usage: Rscript convert_seurat.R <input.rds> <output.h5seurat>")
}

input_rds <- args[1]
output_h5seurat <- args[2]

obj <- readRDS(input_rds)

# SeuratDisk only converts dimensional reductions (obj@reductions) to obsm,
# but spatial coordinates live in the images slot (obj@images) and are silently
# dropped. Extract them and inject as a reduction so they end up in obsm.
images <- Images(obj)
if (length(images) > 0 && !("spatial" %in% Reductions(obj))) {
  coords <- GetTissueCoordinates(obj, image = images[1])
  if (!is.null(coords) && nrow(coords) > 0) {
    spatial_mat <- as.matrix(coords[, 1:2, drop = FALSE])
    colnames(spatial_mat) <- c("spatial_1", "spatial_2")
    obj[["spatial"]] <- CreateDimReducObject(
      embeddings = spatial_mat,
      key = "spatial_",
      assay = DefaultAssay(obj)
    )
  }
}

SaveH5Seurat(obj, filename = output_h5seurat, overwrite = TRUE)
Convert(output_h5seurat, dest = "h5ad", overwrite = TRUE)
