if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager", repos = "https://cloud.r-project.org")
if (!requireNamespace("limma", quietly = TRUE))
  BiocManager::install("limma", update = FALSE, ask = FALSE)
if (!requireNamespace("ggrepel", quietly = TRUE))
  install.packages("ggrepel", repos = "https://cloud.r-project.org")

library(limma)
library(ggplot2)
library(dplyr)
library(ggrepel)

clean_id <- function(sid) {
  sapply(sid, function(x) {
    if (is.na(x)) return(NA)
    p <- strsplit(as.character(x), "-")[[1]]
    if (length(p) >= 4) paste(p[1:4], collapse = "-") else x
  }, USE.NAMES = FALSE)
}

factors <- read.csv("c:/Users/shmso/UCD/Spring/internship/dataset/mofa_factors.csv",
                    row.names = 1, check.names = FALSE)
rownames(factors) <- sapply(rownames(factors), function(x) strsplit(x, "\\|")[[1]][1])
rownames(factors) <- clean_id(rownames(factors))

exp <- read.table("c:/Users/shmso/UCD/Spring/internship/dataset/HiSeqV2",
                  sep = "\t", header = TRUE, row.names = 1, check.names = FALSE)
colnames(exp) <- clean_id(colnames(exp))

common_samples <- intersect(rownames(factors), colnames(exp))

f2      <- factors[common_samples, "Factor2", drop = FALSE]
exp_sub <- exp[, common_samples]

row_means    <- rowMeans(exp_sub, na.rm = TRUE)
exp_filtered <- exp_sub[row_means > 1, ]

f2_median <- median(f2$Factor2, na.rm = TRUE)
group     <- ifelse(f2$Factor2 >= f2_median, "High", "Low")
group     <- factor(group, levels = c("Low", "High"))

design <- model.matrix(~ group)
fit    <- lmFit(as.matrix(exp_filtered), design)
fit    <- eBayes(fit)

de_results       <- topTable(fit, coef = "groupHigh", number = Inf, sort.by = "P")
de_results$gene  <- rownames(de_results)

write.csv(de_results,
          "c:/Users/shmso/UCD/Spring/internship/dataset/factor2_de_results.csv",
          row.names = FALSE)

weights <- read.csv("c:/Users/shmso/UCD/Spring/internship/dataset/mofa_weights_r.csv",
                    row.names = 1, check.names = FALSE)
is_exp       <- grepl("_Exp$", rownames(weights))
w2           <- weights[is_exp, "Factor2", drop = FALSE]
rownames(w2) <- sub("_Exp$", "", rownames(w2))

overlap_genes <- intersect(de_results$gene, rownames(w2))
merged        <- merge(de_results[de_results$gene %in% overlap_genes, ],
                       w2[overlap_genes, , drop = FALSE],
                       by.x = "gene", by.y = "row.names")
colnames(merged)[colnames(merged) == "Factor2"] <- "MOFA_weight"

corr_test <- cor.test(merged$logFC, merged$MOFA_weight, method = "spearman")
cat(paste("Spearman r =", round(corr_test$estimate, 3),
          "| p =", format(corr_test$p.value, scientific = TRUE, digits = 3), "\n"))

top_genes <- merged %>%
  filter(abs(logFC) > 1 & adj.P.Val < 0.05) %>%
  arrange(desc(abs(MOFA_weight))) %>%
  head(20)

p_scatter <- ggplot(merged, aes(x = MOFA_weight, y = logFC)) +
  geom_point(aes(color = adj.P.Val < 0.05 & abs(logFC) > 1), alpha = 0.4, size = 1) +
  geom_smooth(method = "lm", formula = y ~ x, color = "red", se = TRUE) +
  geom_text(data = top_genes, aes(label = gene), size = 2.5, vjust = -0.5,
            color = "black", check_overlap = TRUE) +
  scale_color_manual(values = c("FALSE" = "grey70", "TRUE" = "#377EB8"),
                     labels = c("Not significant", "Significant DE"), name = NULL) +
  theme_minimal(base_size = 11) +
  labs(title = "MOFA Weights vs DE Fold Change",
       subtitle = paste0("Spearman r = ", round(corr_test$estimate, 3)),
       x = "MOFA Weight",
       y = "limma logFC")

ggsave("c:/Users/shmso/UCD/Spring/internship/dataset/factor2_convergence_scatter.png",
       plot = p_scatter, width = 7, height = 5, dpi = 150)

de_results$sig <- ifelse(abs(de_results$logFC) > 1 & de_results$adj.P.Val < 0.05,
                         ifelse(de_results$logFC > 0, "Up (High F2)", "Down (High F2)"),
                         "NS")

top_up    <- de_results %>% filter(sig == "Up (High F2)")   %>% arrange(adj.P.Val) %>% head(15)
top_down  <- de_results %>% filter(sig == "Down (High F2)") %>% arrange(adj.P.Val) %>% head(15)
label_genes <- rbind(top_up, top_down)

p_volcano <- ggplot(de_results, aes(x = logFC, y = -log10(adj.P.Val), color = sig)) +
  geom_point(alpha = 0.5, size = 1) +
  ggrepel::geom_text_repel(data = label_genes, aes(label = gene), size = 2.5,
                            max.overlaps = 20, segment.size = 0.3) +
  scale_color_manual(values = c("Up (High F2)" = "#E41A1C",
                                "Down (High F2)" = "#377EB8",
                                "NS" = "grey70"), name = NULL) +
  geom_vline(xintercept = c(-1, 1), linetype = "dashed", color = "grey40") +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "grey40") +
  theme_minimal(base_size = 11) +
  labs(title = "Factor 2 DE",
       x = "logFC", y = "-log10(adj.P)")

ggsave("c:/Users/shmso/UCD/Spring/internship/dataset/factor2_de_volcano.png",
       plot = p_volcano, width = 7, height = 6, dpi = 150)
