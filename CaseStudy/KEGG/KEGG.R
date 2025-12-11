setwd("")

library(data.table)
library(dplyr)
library(tidyverse)
library(clusterProfiler)
library(GSEABase)
library(enrichplot)
library(ggplot2)

stad.phe <- fread("TCGA-BLCA.GDC_phenotype.tsv.gz", data.table = FALSE)
stad.fkpm <- fread("TCGA-BLCA.htseq_fpkm.tsv.gz", data.table = FALSE)
stad.pro  <- fread("gencode.v22.annotation.gene.probeMap", data.table = FALSE)

stad.pro  <- stad.pro[, c(1, 2)]
stad.fkpm.pro <- merge(stad.pro, stad.fkpm, by.x = "id", by.y = "Ensembl_ID")
stad.fkpm.pro <- distinct(stad.fkpm.pro, gene, .keep_all = TRUE)
stad.fkpm.pro <- column_to_rownames(stad.fkpm.pro, "gene")

selected_genes <- c("PTEN", "MDM2", "RUNX1", "TP53", "EGFR")
selected_genes <- selected_genes[selected_genes %in% rownames(stad.fkpm.pro)]
print("用于 KEGG 富集的基因：")
print(selected_genes)

gmt.file <- "c2.cp.kegg.v7.4.symbols.gmt"
gmt.sets <- getGmt(gmt.file)

kegg_df <- data.frame(
  term = rep(names(gmt.sets), times = sapply(gmt.sets, function(x) length(geneIds(x)))),
  gene = unlist(lapply(gmt.sets, geneIds))
)

kegg.enrich <- enricher(
  selected_genes,
  TERM2GENE = kegg_df,
  pvalueCutoff = 0.05,
  qvalueCutoff = 0.2
)

bp <- barplot(kegg.enrich, showCategory = 10, title = "KEGG Pathway Enrichment")
ggsave("KEGG_barplot.png", plot = bp, width = 8, height = 6, dpi = 300)

dp <- dotplot(kegg.enrich, showCategory = 10, title = "KEGG Pathway Enrichment")
ggsave("KEGG_dotplot.png", plot = dp, width = 8, height = 6, dpi = 300)

cnet_plot <- cnetplot(
  kegg.enrich,
  showCategory = 10,
  circular = FALSE,
  node_label = "all",
  colorEdge = TRUE
) +
  theme_bw(base_size = 12) +
  theme(
    panel.grid = element_blank(),
    axis.title = element_blank(),
    axis.text = element_blank(),
    axis.ticks = element_blank(),
    legend.position = "right",
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold")
  ) +
  scale_color_brewer(palette = "Set2") +
  ggtitle("KEGG Pathway–Gene Network")

ggsave("KEGG_cnetplot.png", plot = cnet_plot, width = 10, height = 8, dpi = 300)

cat("\nKEGG_barplot.png、KEGG_dotplot.png、KEGG_cnetplot.png 已全部生成完毕！\n")
