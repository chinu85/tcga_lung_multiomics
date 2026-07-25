library(MOFA2)
library(ggplot2)
library(dplyr)
library(tidyr)
library(pheatmap)
library(gridExtra)

DIR <- "c:/Users/shmso/UCD/Spring/internship/dataset"
CLIN <- file.path(DIR, "TCGA LUNG sampleMap_LUNG_clinicalMatrix.tsv.txt")
SURV <- file.path(DIR, "LUNG_survival.txt")

clean_id <- function(x) {
  sapply(x, function(s) {
    if (is.na(s)) return(NA)
    p <- strsplit(as.character(s), "-")[[1]]
    if (length(p) >= 4) paste(p[1:4], collapse="-") else s
  }, USE.NAMES=FALSE)
}

factors <- read.csv(file.path(DIR, "mofa_factors.csv"), row.names=1, check.names=FALSE)
rownames(factors) <- sapply(rownames(factors), function(x) strsplit(x, "\\|")[[1]][1])
rownames(factors) <- clean_id(rownames(factors))
factors$sample_id <- rownames(factors)

clin <- read.delim(CLIN, sep="\t", header=TRUE, stringsAsFactors=FALSE)
surv <- read.delim(SURV, sep="\t", header=TRUE, stringsAsFactors=FALSE)

clin$sample_id <- clean_id(clin$sampleID)
surv$sample_id <- clean_id(surv$sample)

meta <- clin %>%
  select(sample_id,
         subtype_raw   = X_primary_disease,
         stage         = pathologic_stage,
         gender,
         smoking       = tobacco_smoking_history,
         cancer_status = person_neoplasm_cancer_status,
         vital_status,
         response      = primary_therapy_outcome_success,
         age           = age_at_initial_pathologic_diagnosis) %>%
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
    grepl("^STAGE I$|^STAGE IA$|^STAGE IB$", s) ~ "I",
    grepl("^STAGE III", s) ~ "III",
    grepl("^STAGE II",  s) ~ "II",
    grepl("^STAGE IV",  s) ~ "IV",
    TRUE ~ NA_character_
  )
}
meta$stage_simple <- stage_simple(meta$stage)

df <- merge(factors, meta, by="sample_id")

weights <- read.csv(file.path(DIR, "mofa_weights_r.csv"), row.names=1, check.names=FALSE)
is_exp <- grepl("_Exp$", rownames(weights))
weights_exp <- weights[is_exp, ]
rownames(weights_exp) <- sub("_Exp$", "", rownames(weights_exp))

# 1. Gender Boxplot
df_gender <- df %>% filter(!is.na(gender), gender != "")
df_gender_long <- df_gender %>%
  pivot_longer(cols=c("Factor1", "Factor2", "Factor3"), names_to="Factor", values_to="Score")

p_gender <- ggplot(df_gender_long, aes(x=gender, y=Score, fill=gender)) +
  geom_boxplot(outlier.size=0.8, alpha=0.8) +
  facet_wrap(~Factor, scales="free_y") +
  scale_fill_manual(values=c("FEMALE"="#E41A1C", "MALE"="#377EB8")) +
  theme_minimal(base_size=11) +
  theme(legend.position="none") +
  labs(title="Factor Scores by Gender", x="Gender", y="Score")

ggsave(file.path(DIR, "plot_gender_boxplot.png"), plot=p_gender, width=8, height=4, dpi=150)

# 2. Data Imbalance Plot (Figure 2A)
p1 <- ggplot(df %>% filter(!is.na(subtype)), aes(x=subtype, fill=subtype)) +
  geom_bar() + geom_text(stat="count", aes(label=after_stat(count)), vjust=-0.3, size=3) +
  scale_fill_manual(values=c("LUAD"="#1f77b4", "LUSC"="#d62728")) + theme_minimal(base_size=10) +
  theme(legend.position="none") + labs(title="Subtype", x=NULL, y="Count")

p2 <- ggplot(df %>% filter(!is.na(gender), gender!=""), aes(x=gender, fill=gender)) +
  geom_bar() + geom_text(stat="count", aes(label=after_stat(count)), vjust=-0.3, size=3) +
  scale_fill_manual(values=c("FEMALE"="#E41A1C", "MALE"="#377EB8")) + theme_minimal(base_size=10) +
  theme(legend.position="none") + labs(title="Gender", x=NULL, y="Count")

p3 <- ggplot(df %>% filter(!is.na(stage_simple)), aes(x=stage_simple, fill=stage_simple)) +
  geom_bar() + geom_text(stat="count", aes(label=after_stat(count)), vjust=-0.3, size=3) +
  scale_fill_brewer(palette="RdYlBu") + theme_minimal(base_size=10) +
  theme(legend.position="none") + labs(title="Stage", x=NULL, y="Count")

p4 <- ggplot(df %>% filter(!is.na(vital_status), vital_status!=""), aes(x=vital_status, fill=vital_status)) +
  geom_bar() + geom_text(stat="count", aes(label=after_stat(count)), vjust=-0.3, size=3) +
  scale_fill_brewer(palette="Set2") + theme_minimal(base_size=10) +
  theme(legend.position="none") + labs(title="Vital Status", x=NULL, y="Count")

resp_order <- c("Complete Remission/Response","Partial Remission/Response","Stable Disease","Progressive Disease")
df_resp_imb <- df %>% filter(!is.na(response), response %in% resp_order) %>%
  mutate(response_short = case_when(
    response == "Complete Remission/Response" ~ "CR",
    response == "Partial Remission/Response" ~ "PR",
    response == "Stable Disease" ~ "SD",
    response == "Progressive Disease" ~ "PD"
  ))
df_resp_imb$response_short <- factor(df_resp_imb$response_short, levels=c("CR","PR","SD","PD"))

p5 <- ggplot(df_resp_imb, aes(x=response_short, fill=response_short)) +
  geom_bar() + geom_text(stat="count", aes(label=after_stat(count)), vjust=-0.3, size=3) +
  scale_fill_brewer(palette="Dark2") + theme_minimal(base_size=10) +
  theme(legend.position="none") + labs(title="Treatment Response", x=NULL, y="Count")

p_imb <- grid.arrange(p1, p2, p3, p4, p5, ncol=3)
ggsave(file.path(DIR, "plot_data_imbalance.png"), plot=p_imb, width=10, height=6, dpi=150)

cat("Successfully generated plot_data_imbalance.png with LUAD (512) and LUSC (498) split!\n")
