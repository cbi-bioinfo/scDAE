library(SingleCellExperiment)
library(scmap)

for (i in 1:10){
 filename <- paste0("group_", i)
 trainFile_X <- paste0(filename, "_train_X")
 trainFile_Y <- paste0(filename, "_train_Y_ml")
 testFile <- paste0(filename, "_test_X")

 ann <- read.csv(trainFile_Y, header = T)
 yan <- read.csv(trainFile_X, header = T)
 test <- read.csv(testFile, header = T)
 yan <- as.matrix(yan)
 yan <- t(yan)
 colnames(yan) = c(1:ncol(yan))
 test <- as.matrix(test)
 test <- t(test)
 colnames(test) = c(1:ncol(test))
 colnames(ann) = c("cell_type1")

 sce <- SingleCellExperiment(assays = list(normcounts = yan), colData = ann)
 test_sce <- SingleCellExperiment(assays = list(normcounts = test))

 logcounts(sce) <- log2(normcounts(sce) + 1)
 rowData(sce)$feature_symbol <- rownames(sce)
 isSpike(sce, "ERCC") <- grepl("^ERCC-", rownames(sce))
 sce <- sce[!duplicated(rownames(sce)), ]

 sce <- selectFeatures(sce, suppress_plot = FALSE)
 sce <- indexCell(sce)

 logcounts(test_sce) <- log2(normcounts(test_sce) + 1)
 rowData(test_sce)$feature_symbol <- rownames(test_sce)

 scmapCell_results <- scmapCell(
   test_sce, 
   list(
     yan = metadata(sce)$scmap_cell_index
   )
 )

 scmapCell_clusters <- scmapCell2Cluster(
   scmapCell_results, 
   list(
     as.character(colData(sce)$cell_type1)
   )
 )

 res <- as.data.frame(scmapCell_clusters$scmap_cluster_labs)
 colnames(res) = c("pred")
 resultFile <- paste0("scmap_result_group_", i, ".csv")
 write.csv(res, resultFile, row.names = F, quote = F)
}
