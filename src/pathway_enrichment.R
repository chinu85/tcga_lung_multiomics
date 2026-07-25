if (!requireNamespace("clusterProfiler", quietly = TRUE))
  BiocManager::install("clusterProfiler", update = FALSE, ask = FALSE)

library(clusterProfiler)
library(org.Hs.eg.db)
library(msigdbr)
library(fgsea)
library(ggplot2)
library(dplyr)

out_dir <- "c:/Users/shmso/UCD/Spring/internship/dataset/enrichment"
if (!dir.exists(out_dir))
  dir.create(out_dir, recursive = TRUE)

weights_path <- "c:/Users/shmso/UCD/Spring/internship/dataset/mofa_weights_r.csv"
weights      <- read.csv(weights_path, row.names = 1, check.names = FALSE)

is_exp       <- grepl("_Exp$", rownames(weights))
weights_exp  <- weights[is_exp, ]
rownames(weights_exp) <- sub("_Exp$", "", rownames(weights_exp))

h_df          <- msigdbr(species = "Homo sapiens", category = "H")
pathway2gene  <- h_df %>% select(gs_name, gene_symbol)
hallmark_list <- split(h_df$gene_symbol, h_df$gs_name)

TOP_N          <- 500
target_factors <- c("Factor1", "Factor2", "Factor3")

for (f in target_factors) {

  gene_list <- weights_exp[[f]]
  names(gene_list) <- rownames(weights_exp)
  gene_list <- sort(gene_list, decreasing = TRUE)
  gene_list <- gene_list[!is.na(names(gene_list))]
  gene_list <- gene_list[!duplicated(names(gene_list))]

  pos_genes  <- names(gene_list)[1:min(TOP_N, length(gene_list))]
  pos_entrez <- bitr(pos_genes, fromType = "SYMBOL", toType = "ENTREZID",
                     OrgDb = org.Hs.eg.db, drop = TRUE)

  if (nrow(pos_entrez) > 0) {
    go_pos <- enrichGO(gene = pos_entrez$ENTREZID,
                       OrgDb = org.Hs.eg.db, ont = "BP",
                       pAdjustMethod = "BH",
                       pvalueCutoff = 0.05, qvalueCutoff = 0.2,
                       readable = TRUE)
    if (!is.null(go_pos) && nrow(as.data.frame(go_pos)) > 0) {
      write.csv(as.data.frame(go_pos),
                file.path(out_dir, paste0(tolower(f), "_go_positive.csv")),
                row.names = FALSE)
      p <- dotplot(go_pos, showCategory = 10, title = paste(f, "GO:BP (+)"))
      ggsave(file.path(out_dir, paste0(tolower(f), "_go_positive_dotplot.png")),
             plot = p, width = 8, height = 6, dpi = 150)
    }
  }

  hall_pos <- enricher(gene = pos_genes, TERM2GENE = pathway2gene,
                       pAdjustMethod = "BH",
                       pvalueCutoff = 0.05, qvalueCutoff = 0.2)
  if (!is.null(hall_pos) && nrow(as.data.frame(hall_pos)) > 0) {
    write.csv(as.data.frame(hall_pos),
              file.path(out_dir, paste0(tolower(f), "_hallmark_positive.csv")),
              row.names = FALSE)
    p <- dotplot(hall_pos, showCategory = 10, title = paste(f, "Hallmark (+)"))
    ggsave(file.path(out_dir, paste0(tolower(f), "_hallmark_positive_dotplot.png")),
           plot = p, width = 8, height = 6, dpi = 150)
  }

  neg_genes  <- names(tail(gene_list, TOP_N))
  neg_entrez <- bitr(neg_genes, fromType = "SYMBOL", toType = "ENTREZID",
                     OrgDb = org.Hs.eg.db, drop = TRUE)

  if (nrow(neg_entrez) > 0) {
    go_neg <- enrichGO(gene = neg_entrez$ENTREZID,
                       OrgDb = org.Hs.eg.db, ont = "BP",
                       pAdjustMethod = "BH",
                       pvalueCutoff = 0.05, qvalueCutoff = 0.2,
                       readable = TRUE)
    if (!is.null(go_neg) && nrow(as.data.frame(go_neg)) > 0) {
      write.csv(as.data.frame(go_neg),
                file.path(out_dir, paste0(tolower(f), "_go_negative.csv")),
                row.names = FALSE)
      p <- dotplot(go_neg, showCategory = 10, title = paste(f, "GO:BP (-)"))
      ggsave(file.path(out_dir, paste0(tolower(f), "_go_negative_dotplot.png")),
             plot = p, width = 8, height = 6, dpi = 150)
    }
  }

  hall_neg <- enricher(gene = neg_genes, TERM2GENE = pathway2gene,
                       pAdjustMethod = "BH",
                       pvalueCutoff = 0.05, qvalueCutoff = 0.2)
  if (!is.null(hall_neg) && nrow(as.data.frame(hall_neg)) > 0) {
    write.csv(as.data.frame(hall_neg),
              file.path(out_dir, paste0(tolower(f), "_hallmark_negative.csv")),
              row.names = FALSE)
    p <- dotplot(hall_neg, showCategory = 10, title = paste(f, "Hallmark (-)"))
    ggsave(file.path(out_dir, paste0(tolower(f), "_hallmark_negative_dotplot.png")),
           plot = p, width = 8, height = 6, dpi = 150)
  }

  all_entrez <- bitr(names(gene_list), fromType = "SYMBOL", toType = "ENTREZID",
                     OrgDb = org.Hs.eg.db, drop = TRUE)
  if (nrow(all_entrez) > 0) {
    kegg_res <- enrichKEGG(gene = pos_entrez$ENTREZID, organism = "hsa",
                           pAdjustMethod = "BH",
                           pvalueCutoff = 0.05, qvalueCutoff = 0.2)
    if (!is.null(kegg_res) && nrow(as.data.frame(kegg_res)) > 0) {
      write.csv(as.data.frame(kegg_res),
                file.path(out_dir, paste0(tolower(f), "_kegg_positive.csv")),
                row.names = FALSE)
      p_kegg <- dotplot(kegg_res, showCategory = 12, title = paste(f, "KEGG"))
      ggsave(file.path(out_dir, paste0(tolower(f), "_kegg_positive_dotplot.png")),
             plot = p_kegg, width = 9, height = 7, dpi = 150)
    }
  }

  set.seed(42)
  gsea_res <- fgsea(pathways = hallmark_list, stats = gene_list,
                    minSize = 15, maxSize = 500)

  if (!is.null(gsea_res) && nrow(gsea_res) > 0) {
    gsea_res_df <- as.data.frame(gsea_res)
    gsea_res_df$leadingEdge <- sapply(gsea_res_df$leadingEdge, paste, collapse = ";")
    write.csv(gsea_res_df,
              file.path(out_dir, paste0(tolower(f), "_hallmark_gsea.csv")),
              row.names = FALSE)

    sig_gsea <- gsea_res_df %>% filter(padj < 0.05)
    if (nrow(sig_gsea) > 0) {
      p_gsea <- ggplot(sig_gsea, aes(x = reorder(pathway, NES), y = NES, fill = NES > 0)) +
        geom_col(show.legend = FALSE) +
        coord_flip() +
        theme_minimal(base_size = 11) +
        scale_fill_manual(values = c("TRUE" = "#377EB8", "FALSE" = "#E41A1C")) +
        labs(title = paste(f, "GSEA"), x = "Pathway", y = "NES")
      ggsave(file.path(out_dir, paste0(tolower(f), "_hallmark_gsea_nes.png")),
             plot = p_gsea,
             width = 9, height = min(10, max(4, nrow(sig_gsea) * 0.3)), dpi = 150)
    }
  }
}
