library(MOFA2)
library(ggplot2)
library(dplyr)

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
  select(sample_id, subtype_raw=X_primary_disease, stage=pathologic_stage, vital_status) %>%
  mutate(subtype = case_when(
    grepl("adenocarcinoma", subtype_raw, ignore.case=TRUE) ~ "LUAD",
    grepl("squamous", subtype_raw, ignore.case=TRUE) ~ "LUSC",
    TRUE ~ NA_character_
  )) %>%
  left_join(surv %>% select(sample_id, OS_event=OS, OS=OS.time), by="sample_id") %>%
  distinct(sample_id, .keep_all=TRUE)

stage_simple <- function(s) {
  s <- toupper(trimws(s))
  case_when(
    grepl("^STAGE I$|^STAGE IA$|^STAGE IB$", s) ~ "Stage I",
    grepl("^STAGE II",  s) ~ "Stage II",
    grepl("^STAGE III", s) ~ "Stage III",
    grepl("^STAGE IV",  s) ~ "Stage IV",
    TRUE ~ NA_character_
  )
}
meta$stage_simple <- stage_simple(meta$stage)

mofa_samples <- samples_names(mdl)[[1]]
mofa_samples_clean <- clean_id(mofa_samples)
row_idx <- match(mofa_samples_clean, meta$sample_id)
meta_aligned <- meta[row_idx, ]
meta_aligned$sample <- mofa_samples

fac_mat <- get_factors(mdl, as.data.frame=FALSE)[[1]]
pca_out <- prcomp(fac_mat, scale.=TRUE)
pca_df  <- as.data.frame(pca_out$x[, 1:4])
pca_df$sample <- rownames(pca_df)
pca_df  <- left_join(pca_df, meta_aligned %>% select(sample, subtype, stage_simple, vital_status), by="sample")
pvar <- round(summary(pca_out)$importance[2, 1:4] * 100, 1)

p_sub <- ggplot(pca_df %>% filter(!is.na(subtype)), aes(PC1, PC2, color=subtype)) +
  geom_point(alpha=0.75, size=1.8) +
  scale_color_manual(values=c("LUAD"="#1f77b4", "LUSC"="#d62728")) +
  labs(title="Latent Factor PCA by Subtype", x=paste0("PC1 (", pvar[1], "%)"), y=paste0("PC2 (", pvar[2], "%)"), color="Subtype") +
  theme_minimal(base_size=11) +
  theme(plot.title=element_text(face="bold", hjust=0.5))

p_stage <- ggplot(pca_df %>% filter(!is.na(stage_simple)), aes(PC1, PC2, color=stage_simple)) +
  geom_point(alpha=0.75, size=1.8) +
  scale_color_brewer(palette="YlOrRd") +
  labs(title="Latent Factor PCA by Pathologic Stage", x=paste0("PC1 (", pvar[1], "%)"), y=paste0("PC2 (", pvar[2], "%)"), color="Stage") +
  theme_minimal(base_size=11) +
  theme(plot.title=element_text(face="bold", hjust=0.5))

p_vital <- ggplot(pca_df %>% filter(!is.na(vital_status)), aes(PC1, PC2, color=vital_status)) +
  geom_point(alpha=0.75, size=1.8) +
  scale_color_brewer(palette="Set2") +
  labs(title="Latent Factor PCA by Vital Status", x=paste0("PC1 (", pvar[1], "%)"), y=paste0("PC2 (", pvar[2], "%)"), color="Status") +
  theme_minimal(base_size=11) +
  theme(plot.title=element_text(face="bold", hjust=0.5))

ggsave(file.path(DIR, "plot_pca_subtype.png"), plot=p_sub, width=6, height=4.5, dpi=150)
ggsave(file.path(DIR, "plot_pca_stage.png"), plot=p_stage, width=6, height=4.5, dpi=150)
ggsave(file.path(DIR, "plot_pca_vital.png"), plot=p_vital, width=6, height=4.5, dpi=150)

cat("PCA plots exported successfully with LUAD and LUSC colored separately!\n")
