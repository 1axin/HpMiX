setwd("")

library(data.table)
library(dplyr)
library(tidyverse)

stad.phe = fread("TCGA-BLCA.GDC_phenotype.tsv.gz", header = TRUE, sep = '\t', data.table = FALSE)
class(stad.phe)

stad.fkpm = fread("TCGA-BLCA.htseq_fpkm.tsv.gz", header = TRUE, sep = '\t', data.table = FALSE)
class(stad.fkpm)
colnames(stad.fkpm)

stad.pro = fread("gencode.v22.annotation.gene.probeMap", header = TRUE, sep = '\t', data.table = FALSE)
View(stad.pro)
colnames(stad.pro)

stad.pro = stad.pro[, c(1,2)]
colnames(stad.pro)

stad.fkpm.pro = merge(stad.pro, stad.fkpm, by.y = "Ensembl_ID", by.x = "id")
?merge
dim(stad.fkpm.pro)

sum(duplicated(stad.fkpm.pro$gene))

stad.fkpm.pro = distinct(stad.fkpm.pro, gene, .keep_all = TRUE)
dim(stad.fkpm.pro)

stad.fkpm.pro <- column_to_rownames(stad.fkpm.pro, "gene")
View(stad.fkpm.pro)

View(stad.phe)

rownames(stad.phe) = stad.phe$submitter_id.samples
colnames(stad.phe)
table(stad.phe$sample_type.samples)

stad.phe.t = filter(stad.phe, sample_type.samples == "Primary Tumor")
stad.phe.n = filter(stad.phe, sample_type.samples == "Solid Tissue Normal")

intersect(c("a","b"), c("b","c"))

z1 = intersect(rownames(stad.phe.t), colnames(stad.fkpm.pro))
z2 = intersect(rownames(stad.phe.n), colnames(stad.fkpm.pro))

stad.t = stad.fkpm.pro[, z1]
stad.n = stad.fkpm.pro[, z2]

colnames(stad.n) = paste0("N", 1:19)
paste("xy", c(1,5,8,11,3), sep = "-")
?paste

colnames(stad.t) = paste0("T", 1:411)

stad.exp = merge(stad.n, stad.t, by.x = 0, by.y = 0)
colnames(stad.exp)
stad.exp <- column_to_rownames(stad.exp, "Row.names")

library(BiocManager)
library(limma)

nromalized.data = normalizeBetweenArrays(stad.exp)
?normalizeBetweenArrays

library(ggplot2)

exprSet.all.r = stad.exp[c("PTEN", "MDM2", "RUNX1", "TP53", "EGFR"),]
exprSet.all.r = t(exprSet.all.r)
exprSet.all.r = as.data.frame(exprSet.all.r)

x = c(rep("N",19), rep("T",411))
exprSet.all.r$Type = x

exprSet.as2 = exprSet.all.r[, c(1,6)]
exprSet.as2$Gene = rep("PTEN")
colnames(exprSet.as2)[1] = "Relative Expression"

exprSet.h19 = exprSet.all.r[, c(2,6)]
exprSet.h19$Gene = rep("MDM2")
colnames(exprSet.h19)[1] = "Relative Expression"

exprSet.ror = exprSet.all.r[, c(3,6)]
exprSet.ror$Gene = rep("RUNX1")
colnames(exprSet.ror)[1] = "Relative Expression"

exprSet.ccat = exprSet.all.r[, c(4,6)]
exprSet.ccat$Gene = rep("TP53")
colnames(exprSet.ccat)[1] = "Relative Expression"

exprSet.ha = exprSet.all.r[, c(5,6)]
exprSet.ha$Gene = rep("EGFR")
colnames(exprSet.ha)[1] = "Relative Expression"

x.all = rbind(exprSet.as2, exprSet.h19, exprSet.ror, exprSet.ccat, exprSet.ha)
colnames(x.all)

library(ggsignif)
library(ggpubr)
library(ggplot2)

q <- ggboxplot(x.all, x = "Gene", y = "Relative Expression",
               color = "Type", palette = "Type",
               add = "detplot", size = 1)

q + stat_compare_means(aes(group = Type),
                       label = 'p.format') +
  theme(legend.text = element_text(size = 10))

table(x.all$Gene)

exprSet.all.r1 = t(exprSet.all.r)
exprSet.all.r1 = as.data.frame(exprSet.all.r1)

n <- nrow(exprSet.all.r1)
exprSet.all.r2 <- exprSet.all.r1[-n, ]

library(pheatmap)

exprSet.all.matrix <- as.matrix(exprSet.all.r2)
exprSet.all.matrix <- data.matrix(exprSet.all.matrix)

write.csv(exprSet.all.matrix, file = "5个gene表达矩阵.csv", row.names = TRUE)

data <- read.table(file = '5个gene表达矩阵.csv', header = TRUE, row.names = 1, sep = ',')
head(data)

anno_col <- read.table(file = '列注释信息.csv', header = TRUE, row.names = 1, sep = ',')
anno_col1 = as.data.frame(anno_col)

p <- pheatmap(data, scale = 'row',
              show_rownames = TRUE,
              show_colnames = FALSE,
              cluster_rows = TRUE,
              cluster_cols = FALSE,
              annotation_col = anno_col,
              color = colorRampPalette(c("blue","white","red"))(30))
