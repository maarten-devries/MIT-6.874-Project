library(Seurat)
library(Signac)
library(SeuratDisk)

###Convert scRNA Suerat object to Scanpy adata
rna_result<-readRDS("/Users/yaoestelle/Downloads/MIT684/project/dbClust_0130.rds")
SaveH5Seurat(rna_result, filename = "scRNA_mit.h5Seurat")
Convert("scRNA_mit.h5Seurat", dest = "h5ad")

###Convert scATAC pseudo RNA Suerat object to Scanpy adata
atac_result<-readRDS("/Users/yaoestelle/Downloads/MIT684/project/combined_processed_0119.rds")
SaveH5Seurat(atac_result, filename = "scAtac_mit.h5Seurat")
Convert("scAtac_mit.h5Seurat", dest = "h5ad")

###Convert scATAC Bed file Suerat object to Scanpy adata
atac<-atac_result$ATAC
object <- CreateSeuratObject(counts = atac)
SaveH5Seurat(object, filename = "scAtac_bed_mit.h5Seurat")
Convert("scAtac_bed_mit.h5Seurat", dest = "h5ad")
