library("ggplot2")


class_10cv_result2$Method <- factor(class_10cv_result2$Method, levels = c("scDAE", "CaSTLe", "scPred", "scmap", "CHETAH"))

ggplot(class_10cv_result2, aes(x = Celltype, y = Accuracy, fill = Method)) +
   geom_bar(stat = "identity", width = 0.5, position = "dodge", color = "black") +
   theme_bw() +
   scale_fill_manual(values=c("#FDB462", "#386CB0", "#7FC97F", "#662506", "#EF3B2C")) +
   coord_cartesian( ylim=c(0.5, 1.0)) +
   labs(x="Cell Type", y="Accuracy") +
   theme(axis.text=element_text(size=12),axis.title=element_text(size=12,face="bold"), legend.title=element_text(size=12, face = "bold"), 
         legend.text=element_text(size=12))

result_10cv_brain$Method <- factor(result_10cv_brain$Method, levels = c("scDAE", "scPred", "CaSTLe", "CHETAH", "scmap"))


ggplot(result_10cv_brain, aes(x = Method, y = Accuracy, fill = Method)) +
   geom_boxplot(size = 0.8) +
   scale_y_continuous(limits = c(0.97, 1.0)) +
   scale_x_discrete(limits = c("scDAE", "scPred", "CaSTLe", "CHETAH", "scmap")) +
   theme_bw() +
   theme(axis.text=element_text(size=12),axis.title=element_text(size=12,face="bold"), legend.position = "none") +
   scale_fill_manual(values=c("#A99B38", "#006CE6", "#51841C", "#E69F00", "#999999")) 