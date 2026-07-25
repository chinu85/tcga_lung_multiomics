library(MOFA2)
library(ggplot2)
library(dplyr)
library(tidyr)
library(tibble)
library(pheatmap)
library(ggrepel)
library(survival)
library(survminer)

pdf(NULL)

DIR  <- "c:/Users/shmso/UCD/Spring/internship/dataset"
CLIN <- file.path(dirname(DIR), "dataset/TCGA LUNG sampleMap_LUNG_clinicalMatrix.tsv.txt")
SURV <- file.path(DIR, "LUNG_survival.txt")

clean_id <- function(x) {
  sapply(x, function(s) {
    p <- strsplit(as.character(s), "-")[[1]]
    if (length(p) >= 4) paste(p[1:4], collapse="-") else s
  }, USE.NAMES=FALSE)
}

mdl  <- load_model(file.path(DIR, "mofa_model.hdf5"))
clin <- read.delim(CLIN, sep="\t", header=TRUE, stringsAsFactors=FALSE)
surv <- read.delim(SURV, sep="\t", header=TRUE, stringsAsFactors=FALSE)

clin$sample_id <- clean_id(clin$sampleID)
surv$sample_id <- clean_id(surv$sample)

meta <- clin %>%
  select(sample_id,
         subtype       = X_cohort,
         stage         = pathologic_stage,
         gender,
         smoking       = tobacco_smoking_history,
         cancer_status = person_neoplasm_cancer_status,
         vital_status,
         response      = primary_therapy_outcome_success,
         age           = age_at_initial_pathologic_diagnosis,
         kras = KRAS, egfr = EGFR) %>%
  left_join(surv %>% select(sample_id, OS_event=OS, OS=OS.time), by="sample_id") %>%
  distinct(sample_id, .keep_all=TRUE)

stage_simple <- function(s) {
  s <- toupper(trimws(s))
  case_when(
    grepl("^STAGE I$|^STAGE IA$|^STAGE IB$", s) ~ "I",
    grepl("^STAGE III", s) ~ "III",
    grepl("^STAGE II",  s) ~ "II",
    grepl("^STAGE IV",  s) ~ "IV",
    TRUE ~ NA_character_
  )
}
meta$stage_simple <- stage_simple(meta$stage)

mofa_samples       <- samples_names(mdl)[[1]]
mofa_samples_clean <- clean_id(mofa_samples)
row_idx            <- match(mofa_samples_clean, meta$sample_id)
meta_aligned       <- meta[row_idx, ]
meta_aligned$sample <- mofa_samples
rownames(meta_aligned) <- mofa_samples

samples_metadata(mdl) <- meta_aligned

p_var <- plot_variance_explained(mdl, max_r2=15) +
  ggtitle("R2 per Factor") +
  theme_minimal(base_size=11)
print(p_var)

fac_scores <- get_factors(mdl, as.data.frame=FALSE)[[1]]
meta_num   <- meta_aligned %>%
  select(sample, subtype, stage_simple, gender, smoking,
         cancer_status, vital_status, response, age) %>%
  mutate(across(-sample, ~ as.numeric(as.factor(.)))) %>%
  as.data.frame()
rownames(meta_num) <- meta_num$sample
meta_num$sample    <- NULL
meta_num <- meta_num[rownames(fac_scores), , drop=FALSE]
meta_num <- meta_num[, apply(meta_num, 2, function(x) var(x, na.rm=TRUE) > 0 & sum(!is.na(x)) > 10), drop=FALSE]

cor_mat  <- cor(fac_scores, meta_num, use="pairwise.complete.obs", method="spearman")
cor_long <- as.data.frame(as.table(cor_mat))
colnames(cor_long) <- c("Factor", "Covariate", "Spearman_r")

p_cor <- ggplot(cor_long, aes(Covariate, Factor, fill=Spearman_r)) +
  geom_tile(color="white") +
  geom_text(aes(label=round(Spearman_r, 2)), size=2.5) +
  scale_fill_gradient2(low="#377EB8", mid="white", high="#E41A1C",
                       midpoint=0, limits=c(-1,1), name="r") +
  theme_minimal(base_size=10) +
  theme(axis.text.x=element_text(angle=40, hjust=1)) +
  labs(title="Factor Correlations", x=NULL, y=NULL)
print(p_cor)

fac_mat <- get_factors(mdl, as.data.frame=FALSE)[[1]]
pca_out <- prcomp(fac_mat, scale.=TRUE)
pca_df  <- as.data.frame(pca_out$x[, 1:4])
pca_df$sample <- rownames(pca_df)
pca_df  <- left_join(pca_df,
                     meta_aligned %>% select(sample, subtype, stage_simple, vital_status, response, smoking),
                     by="sample")
pvar <- round(summary(pca_out)$importance[2, 1:4] * 100, 1)

print(ggplot(pca_df, aes(PC1, PC2, color=subtype)) +
  geom_point(alpha=0.6, size=1.5) +
  scale_color_brewer(palette="Set1", na.value="grey70") +
  labs(title="PCA by Subtype",
       x=paste0("PC1 (", pvar[1], "%)"), y=paste0("PC2 (", pvar[2], "%)")) +
  theme_minimal(base_size=11))

print(ggplot(pca_df, aes(PC1, PC2, color=stage_simple)) +
  geom_point(alpha=0.6, size=1.5) +
  scale_color_brewer(palette="RdYlBu", na.value="grey70") +
  labs(title="PCA by Stage",
       x=paste0("PC1 (", pvar[1], "%)"), y=paste0("PC2 (", pvar[2], "%)")) +
  theme_minimal(base_size=11))

print(ggplot(pca_df, aes(PC1, PC2, color=vital_status)) +
  geom_point(alpha=0.6, size=1.5) +
  scale_color_brewer(palette="Set2", na.value="grey70") +
  labs(title="PCA by Vital Status",
       x=paste0("PC1 (", pvar[1], "%)"), y=paste0("PC2 (", pvar[2], "%)")) +
  theme_minimal(base_size=11))

for (fac in c(1, 2, 3)) {
  print(plot_top_weights(mdl, view="Expression", factor=fac, nfeatures=20) +
    ggtitle(paste0("Factor ", fac, " Weights")) +
    theme_minimal(base_size=10))
}

pivot_counts <- meta_aligned %>%
  filter(!is.na(stage_simple), !is.na(subtype)) %>%
  count(stage_simple, subtype) %>%
  pivot_wider(names_from=subtype, values_from=n, values_fill=0) %>%
  arrange(stage_simple)
pivot_counts$Total <- rowSums(pivot_counts[, -1])
print(as.data.frame(pivot_counts))

fac_wide <- get_factors(mdl, as.data.frame=FALSE)[[1]] %>%
  as.data.frame() %>%
  mutate(sample=rownames(.)) %>%
  left_join(meta_aligned %>% select(sample, stage_simple, subtype), by="sample")

pivot_fac <- fac_wide %>%
  filter(!is.na(stage_simple)) %>%
  group_by(stage_simple) %>%
  summarise(across(starts_with("Factor"),
                   list(mean=\(x) round(mean(x, na.rm=TRUE), 3)),
                   .names="{.col}_mean"),
            n=n(), .groups="drop") %>%
  select(stage_simple, n, Factor2_mean, Factor3_mean)
print(as.data.frame(pivot_fac))

plot_df <- fac_wide %>% filter(!is.na(stage_simple), !is.na(subtype))

print(ggplot(plot_df, aes(stage_simple, Factor2, fill=subtype)) +
  geom_violin(trim=FALSE, alpha=0.7, position=position_dodge(0.9)) +
  geom_boxplot(width=0.1, outlier.size=0.5, position=position_dodge(0.9)) +
  scale_fill_brewer(palette="Set1") +
  labs(title="Factor 2 by Stage/Subtype", x="Stage", y="Factor 2") +
  theme_minimal(base_size=11))

print(ggplot(plot_df, aes(stage_simple, Factor3, fill=subtype)) +
  geom_violin(trim=FALSE, alpha=0.7, position=position_dodge(0.9)) +
  geom_boxplot(width=0.1, outlier.size=0.5, position=position_dodge(0.9)) +
  scale_fill_brewer(palette="Set1") +
  labs(title="Factor 3 by Stage/Subtype", x="Stage", y="Factor 3") +
  theme_minimal(base_size=11))

heat_df <- fac_wide %>%
  filter(!is.na(stage_simple), !is.na(subtype)) %>%
  group_by(stage_simple, subtype) %>%
  summarise(across(starts_with("Factor"), \(x) mean(x, na.rm=TRUE)), .groups="drop") %>%
  unite("group", stage_simple, subtype, sep=" / ") %>%
  as.data.frame()
rownames(heat_df) <- heat_df$group
heat_df$group     <- NULL
fac_cols <- grep("^Factor", colnames(heat_df), value=TRUE)[1:6]
pheatmap(t(heat_df[, fac_cols]),
         main="Mean Factor Scores",
         cluster_rows=FALSE, cluster_cols=FALSE,
         color=colorRampPalette(c("steelblue","white","firebrick"))(50), fontsize=9)

stage_bar_df <- meta_aligned %>%
  filter(!is.na(stage_simple), !is.na(subtype)) %>%
  count(stage_simple, subtype)
print(ggplot(stage_bar_df, aes(stage_simple, n, fill=subtype)) +
  geom_col() +
  geom_text(aes(label=n), position=position_stack(vjust=0.5), size=3.5) +
  scale_fill_brewer(palette="Set1") +
  labs(title="Patients by Stage", x="Stage", y="Count") +
  theme_minimal(base_size=11))

resp_order <- c("Complete Remission/Response","Partial Remission/Response",
                "Stable Disease","Progressive Disease")
resp_df <- fac_wide %>%
  left_join(meta_aligned %>% select(sample, resp2=response), by="sample") %>%
  filter(!is.na(resp2), resp2 %in% resp_order) %>%
  mutate(resp2=factor(resp2, levels=resp_order))

print(ggplot(resp_df, aes(resp2, Factor2, fill=resp2)) +
  geom_violin(trim=FALSE, alpha=0.75) +
  geom_boxplot(width=0.1, outlier.size=0.5, fill="white") +
  scale_fill_brewer(palette="RdYlGn", direction=-1) +
  labs(title="Factor 2 by Response", x=NULL, y="Factor 2") +
  theme_minimal(base_size=11) + theme(legend.position="none") +
  scale_x_discrete(labels=c("CR","PR","SD","PD")))

print(ggplot(resp_df, aes(resp2, Factor3, fill=resp2)) +
  geom_violin(trim=FALSE, alpha=0.75) +
  geom_boxplot(width=0.1, outlier.size=0.5, fill="white") +
  scale_fill_brewer(palette="RdYlGn", direction=-1) +
  labs(title="Factor 3 by Response", x=NULL, y="Factor 3") +
  theme_minimal(base_size=11) + theme(legend.position="none") +
  scale_x_discrete(labels=c("CR","PR","SD","PD")))

cr_pd_df <- fac_wide %>%
  left_join(meta_aligned %>% select(sample, resp3=response), by="sample") %>%
  filter(resp3 %in% c("Complete Remission/Response","Progressive Disease")) %>%
  mutate(group=ifelse(resp3=="Complete Remission/Response","CR","PD")) %>%
  group_by(group) %>%
  summarise(across(starts_with("Factor"), \(x) mean(x, na.rm=TRUE)), .groups="drop") %>%
  as.data.frame()
rownames(cr_pd_df) <- cr_pd_df$group
cr_pd_df$group     <- NULL
pheatmap(t(cr_pd_df),
         main="Mean Factor Profile: CR vs PD",
         cluster_rows=FALSE, cluster_cols=FALSE,
         color=colorRampPalette(c("steelblue","white","firebrick"))(50), fontsize=9)

surv_df <- fac_wide %>%
  left_join(meta_aligned %>% select(sample, OS, OS_event), by="sample") %>%
  filter(!is.na(OS), !is.na(OS_event), !is.na(Factor2)) %>%
  mutate(f2_group=ifelse(Factor2 >= median(Factor2, na.rm=TRUE), "High Factor 2", "Low Factor 2"))

km_fit <- survfit(Surv(OS, OS_event) ~ f2_group, data=surv_df)
print(ggsurvplot(km_fit, data=surv_df,
                 pval=TRUE, conf.int=TRUE,
                 risk.table=TRUE, risk.table.height=0.3,
                 palette=c("#E41A1C","#377EB8"),
                 title="Survival: Factor 2 High vs Low",
                 xlab="Days", legend.labs=c("High F2","Low F2"),
                 ggtheme=theme_minimal(base_size=11)))

if (!interactive()) {
  dev.off()
}
