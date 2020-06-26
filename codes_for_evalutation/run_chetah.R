library(CHETAH)
library("Rtsne")

for (i in 1:10){
 filename <- paste0("group_", i)
 trainFile_X <- paste0(filename, "_train_X")
 trainFile_Y <- paste0(filename, "_train_Y_ml")
 testFile <- paste0(filename, "_test_X")

 celltypes_hn <- read.csv(trainFile_Y, header = T)
 colnames(celltypes_hn) = c("celltypes")
 counts_hn <- read.csv(trainFile_X, header = T)
 counts_hn <- as.matrix(counts_hn)
 counts_hn <- t(counts_hn)
 test <- read.csv(testFile, header = T)
 tsne_result <- Rtsne(test, dims = 2)
 sne_result_Y <- tsne_result$Y
 tsne_result_Y <- as.data.frame(sne_result_Y)

 test <- as.matrix(test)
 test <- t(test)

 colnames(counts_hn) = c(1:ncol(counts_hn))
 colnames(test) = c(1:ncol(test))

 ref <- SingleCellExperiment(assays = list(counts = counts_hn), colData = celltypes_hn)

 input <- SingleCellExperiment(assays = list(counts = test), reducedDims = SimpleList(TSNE = tsne_result_Y))

 input <- CHETAHclassifier(input = input, ref_cells = ref, thresh = 0.0)

 res <- as.data.frame(input$celltype_CHETAH)
 colnames(res) = c("pred")
 resultFile <- paste0("chetah_result_group_", i, ".csv")
 write.csv(res, resultFile, row.names = F, quote = F)
}
