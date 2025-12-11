setwd("")

library(data.table)
library(dplyr)
library(tidyverse)
library(limma)
library(ggplot2)
library(ggpubr)
library(ggsignif)
library(pheatmap)

stad.phe = fread("TCGA-BLCA.GDC_phenotype.tsv.gz", header = TRUE, sep = '\t', data.table = FALSE)
stad.fkpm = fread("TCGA-BLCA.mirna.tsv.gz", header = TRUE, sep = '\t', data.table = FALSE)
stad.pro = fread("gencode.v22.annotation.gene.probeMap", header = TRUE, sep = '\t', data.table = FALSE)

stad.pro = stad.pro[, c(1, 2)]
dim(stad.fkpm)
sum(duplicated(stad.fkpm$miRNA_ID))

stad.fkpm <- column_to_rownames(stad.fkpm, "miRNA_ID")

rownames(stad.phe) = stad.phe$submitter_id.samples
stad.phe.t = filter(stad.phe, sample_type.samples == "Primary Tumor")
stad.phe.n = filter(stad.phe, sample_type.samples == "Solid Tissue Normal")

z1 = intersect(rownames(stad.phe.t), colnames(stad.fkpm))
z2 = intersect(rownames(stad.phe.n), colnames(stad.fkpm))

stad.t = stad.fkpm[, z1]
stad.n = stad.fkpm[, z2]

colnames(stad.n) = paste0("N", 1:ncol(stad.n))
colnames(stad.t) = paste0("T", 1:ncol(stad.t))

stad.exp = merge(stad.n, stad.t, by.x = 0, by.y = 0)
stad.exp <- column_to_rownames(stad.exp, "Row.names")

nromalized.data = normalizeBetweenArrays(stad.exp)

write.csv(stad.exp, file = "miRNA-seq-bladder.csv", row.names = TRUE)

exprSet.all.r = stad.exp[c("hsa-mir-17", "hsa-mir-1305",
                           "hsa-mir-130b", "hsa-mir-653", "hsa-mir-107"), ]

exprSet.all.r = t(exprSet.all.r)
exprSet.all.r = as.data.frame(exprSet.all.r)

exprSet.all.r$Type = c(rep("N", 19), rep("T", 412))

exprSet.as2 = exprSet.all.r[, c(1, 6)]
exprSet.as2$Gene = "hsa-mir-17"
colnames(exprSet.as2)[1] = "Relative Expression"

exprSet.h19 = exprSet.all.r[, c(2, 6)]
exprSet.h19$Gene = "hsa-mir-1305"
colnames(exprSet.h19)[1] = "Relative Expression"

exprSet.ror = exprSet.all.r[, c(3, 6)]
exprSet.ror$Gene = "hsa-mir-130b"
colnames(exprSet.ror)[1] = "Relative Expression"

exprSet.ccat = exprSet.all.r[, c(4, 6)]
exprSet.ccat$Gene = "hsa-mir-653"
colnames(exprSet.ccat)[1] = "Relative Expression"

exprSet.ha = exprSet.all.r[, c(5, 6)]
exprSet.ha$Gene = "hsa-mir-107"
colnames(exprSet.ha)[1] = "Relative Expression"

x.all = rbind(exprSet.as2, exprSet.h19, exprSet.ror, exprSet.ccat, exprSet.ha)

q <- ggboxplot(x.all, x = "Gene", y = "Relative Expression",
               color = "Type", palette = "Type",
               add = "detplot", size = 1)

q + stat_compare_means(aes(group = Type),
                       label = "p.format") +
  theme(legend.text = element_text(size = 10),
        axis.text = element_text(size = 10))

exprSet.all.r1 = t(exprSet.all.r)
exprSet.all.r1 = as.data.frame(exprSet.all.r1)
exprSet.all.r2 <- exprSet.all.r1[-nrow(exprSet.all.r1), ]

exprSet.all.matrix <- as.matrix(exprSet.all.r2)
exprSet.all.matrix <- data.matrix(exprSet.all.matrix)

write.csv(exprSet.all.matrix, file = "miRNA_差异表达矩阵.csv", row.names = TRUE)

data <- read.table(file = "miRNA_差异表达矩阵.csv",
                   header = TRUE, row.names = 1, sep = ',')

anno_col <- read.table(file = "列注释信息.csv",
                       header = TRUE, row.names = 1, sep = ',')

p <- pheatmap(data, scale = "row",
              show_rownames = TRUE,
              show_colnames = FALSE,
              cluster_rows = TRUE,
              cluster_cols = FALSE,
              annotation_col = anno_col,
              color = colorRampPalette(c("darkgreen", "white", "red"))(100))
