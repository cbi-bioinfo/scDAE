library(scater)
library(xgboost)
library(igraph)
BREAKS=c(-1, 0, 1, 6, Inf)
nFeatures = 100

for (i in 1:10){
 filename <- paste0("group_", i)
 trainFile_X <- paste0(filename, "_train_X")
 trainFile_Y <- paste0(filename, "_train_Y_ml")
 testFile <- paste0(filename, "_test_X")

 train_data <- read.csv(trainFile_X, header = T)
 test_data <- read.csv(testFile, header = T)
 train_info <- read.csv(trainFile_Y, header = T)
 train_data <- as.matrix(train_data)
 train_data <- t(train_data)
 colnames(train_data) = c(1:ncol(train_data))
 test_data <- as.matrix(test_data)
 test_data <- t(test_data)
 colnames(test_data) = c(1:ncol(test_data))

 colnames(train_info) = c("cell_type1")
 source <- SingleCellExperiment(assays = list(counts = train_data), colData = train_info)

 target <- SingleCellExperiment(assays = list(counts = test_data))

 exprs(source) <- train_data
 exprs(target) <- test_data

 ds1 = t(exprs(source)) 
 ds2 = t(exprs(target))

 sourceCellTypes = as.factor(colData(source)[,"cell_type1"])

 source_n_cells_counts = apply(exprs(source), 1, function(x) { sum(x > 0) } )
 target_n_cells_counts = apply(exprs(target), 1, function(x) { sum(x > 0) } )

 common_genes = intersect( rownames(source)[source_n_cells_counts>10], rownames(target)[target_n_cells_counts>10] )

 remove(source_n_cells_counts, target_n_cells_counts)

 ds1 = ds1[, colnames(ds1) %in% common_genes]
 ds2 = ds2[, colnames(ds2) %in% common_genes]
 ds = rbind(ds1[,common_genes], ds2[,common_genes])
 isSource = c(rep(TRUE,nrow(ds1)), rep(FALSE,nrow(ds2)))
 remove(ds1, ds2)

 topFeaturesAvg = colnames(ds)[order(apply(ds, 2, mean), decreasing = T)]
 topFeaturesMi = names(sort(apply(ds[isSource,],2,function(x) { compare(cut(x,breaks=BREAKS),sourceCellTypes,method = "nmi") }), decreasing = T))

 selectedFeatures = union(head(topFeaturesAvg, nFeatures) , head(topFeaturesMi, nFeatures) )

 tmp = cor(ds[,selectedFeatures], method = "pearson")
 tmp[!lower.tri(tmp)] = 0
 selectedFeatures = selectedFeatures[apply(tmp,2,function(x) any(x < 0.9))]
 remove(tmp)

 dsBins = apply(ds[, selectedFeatures], 2, cut, breaks= BREAKS)
 nUniq = apply(dsBins, 2, function(x) { length(unique(x)) })
 ds = model.matrix(~ . , as.data.frame(dsBins[,nUniq>1]))
 remove(dsBins, nUniq)

 train = runif(nrow(ds[isSource,]))<0.8

 if (length(unique(sourceCellTypes)) > 2) {
   xg=xgboost(data=ds[isSource,][train, ] , 
        label=as.numeric(sourceCellTypes[train])-1,
        objective="multi:softmax", num_class=length(unique(sourceCellTypes)),
        eta=0.7 , nthread=5, nround=20, verbose=0,
        gamma=0.001, max_depth=5, min_child_weight=10)
 } else {
   xg=xgboost(data=ds[isSource,][train, ] , 
        label=as.numeric(sourceCellTypes[train])-1,
        eta=0.7 , nthread=5, nround=20, verbose=0,
        gamma=0.001, max_depth=5, min_child_weight=10)
 }

 predictedClasses = predict(xg, ds[!isSource, ])

 predictedClasses <- as.data.frame(predictedClasses)
 colnames(predictedClasses) = c("pred")
 k <- predictedClasses
 k[k == 0] = "alpha"
 k[k == 1] = "beta"
 k[k == 2] = "delta"
 k[k == 3] = "gamma"
 
 resultFile <- paste0("castle_result_group_", i, ".csv")
 write.csv(k, resultFile, row.names = F, quote = F)
}
