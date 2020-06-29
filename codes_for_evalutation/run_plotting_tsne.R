library("Rtsne")
library("rgl")
library("RColorBrewer")
library("pals")
library("ggplot2")


color_rainbow <- read.csv("color_rainbow_brain.csv")
AE_Y_encoder_results_in_class <- read.csv("AE_Y_encoder_results_in_class_brain_200610.csv")
#AE_encoder_results <- read.csv("../tsne/AE_encoder_results.csv", header=FALSE)

set.seed(1)

class_criteria <- AE_Y_encoder_results_in_class
tsne_result <- Rtsne(AE_encoder_results, dims = 2)

sne_result_Y <- tsne_result$Y
tsne_result_Y <- as.data.frame(sne_result_Y)
tsne_result_Y <- read.csv("original_brain_tsne_results_2d.csv", header = F)
tsne_result_Y_and_class <- cbind(tsne_result_Y, class_criteria)

color_rainbow<-as.matrix(color_rainbow)
color_rainbow<-as.vector(color_rainbow)

class_criteria <- AE_Y_encoder_results_in_class
colnames(class_criteria) = c("Celltype")
tsne_result_Y_2dim <- cbind(tsne_result_Y, class_criteria)

#Alpha, Beta, Delta, Gamma
ggplot(tsne_result_Y_2dim, aes(x=V1, y=V2, color = Celltype)) +
   geom_point() +
   theme_bw() +
   scale_color_manual(values=c("#3A53A3", "#2C8A57", "#F37E20", "#662506")) +
   labs(x="tSNE1", y="tSNE2", color = "Cell type") +
   theme(axis.text=element_text(size=8),axis.title=element_text(size=8), legend.title=element_text(size=12, face = "bold"), 
         legend.text=element_text(size=12))


#Brain

class_criteria <- AE_Y_encoder_results_in_class
colnames(class_criteria) = c("Celltype")
tsne_result_Y_2dim <- cbind(tsne_result_Y, class_criteria)

cl <- c("#CBAD2D", "#A1CB59", "#3A53A3", "#EA2127", "#F37E20","#2D2E72","#2C8A57","#8A461F", "#A62A2A","#EBB320",   "#81509F")
ggplot(tsne_result_Y_2dim, aes(x=V1, y=V2, color = Celltype)) +
   geom_point() +
   theme_bw() +
   scale_color_manual(values=cl) +
   labs(x="tSNE1", y="tSNE2", color = "Cell type") +
   theme(axis.text=element_text(size=8),axis.title=element_text(size=8), legend.title=element_text(size=12, face = "bold"), 
         legend.text=element_text(size=12))


# Original
class_criteria_original <- AE_Y_encoder_results_in_class_original_for_2d
tsne_result_Y_2dim_original <- cbind(tsne_result_dim2_original_data, class_criteria_original)
ggplot(tsne_result_Y_2dim_original, aes(x=V1, y=V2, color = Celltype)) +
   geom_point() +
   theme_bw() +
   scale_color_manual(values=c("#3A53A3", "#2C8A57", "#F37E20", "#662506")) +
   labs(x="tSNE1", y="tSNE2", color = "Cell type") +
   theme(axis.text=element_text(size=8),axis.title=element_text(size=8), legend.title=element_text(size=12, face = "bold"), 
         legend.text=element_text(size=12))

cl <- c("#CBAD2D", "#A1CB59", "#3A53A3", "#EA2127", "#F37E20","#2D2E72","#2C8A57","#8A461F", "#A62A2A","#EBB320",   "#81509F")
class_criteria_brain <- AE_Y_encoder_results_in_class_brain
tsne_result_Y_2dim_brain <- cbind(tsne_result_Y_2d_brain, class_criteria_brain)
ggplot(tsne_result_Y_2dim_brain, aes(x=V1, y=V2, color = class)) +
   geom_point() +
   theme_bw() +
   scale_color_manual(values=cl) +
   labs(x="tSNE1", y="tSNE2", color = "Cell type") +
   theme(axis.text=element_text(size=8),axis.title=element_text(size=8), legend.title=element_text(size=12, face = "bold"), 
         legend.text=element_text(size=12))
