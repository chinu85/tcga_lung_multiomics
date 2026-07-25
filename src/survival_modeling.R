if (!requireNamespace("timeROC", quietly = TRUE))
  install.packages("timeROC", repos = "https://cloud.r-project.org")

library(survival)
library(survminer)
library(glmnet)
library(timeROC)
library(ggplot2)
library(dplyr)

clean_sample_ids <- function(sids) {
  sapply(sids, function(sid) {
    if (is.na(sid)) return(NA)
    parts <- strsplit(as.character(sid), "-")[[1]]
    if (length(parts) >= 4) paste(parts[1:4], collapse = "-") else sid
  }, USE.NAMES = FALSE)
}

factors <- read.csv("c:/Users/shmso/UCD/Spring/internship/dataset/mofa_factors.csv",
                    row.names = 1, check.names = FALSE)
rownames(factors) <- sapply(rownames(factors), function(x) strsplit(x, "\\|")[[1]][1])
rownames(factors) <- clean_sample_ids(rownames(factors))
factors$sample_id <- rownames(factors)

clin <- read.delim("c:/Users/shmso/UCD/Spring/internship/dataset/TCGA LUNG sampleMap_LUNG_clinicalMatrix.tsv.txt",
                   sep = "\t", header = TRUE, check.names = FALSE, quote = "", fill = TRUE)
clin$sample_id <- clean_sample_ids(clin$sampleID)

surv <- read.table("c:/Users/shmso/UCD/Spring/internship/dataset/LUNG_survival.txt",
                   sep = "\t", header = TRUE, check.names = FALSE)
colnames(surv)[colnames(surv) == "sample"]  <- "sample_id"
colnames(surv)[colnames(surv) == "OS"]      <- "OS_event"
colnames(surv)[colnames(surv) == "OS.time"] <- "OS_days"
surv$sample_id <- clean_sample_ids(surv$sample_id)

cnv_raw <- read.table("c:/Users/shmso/UCD/Spring/internship/dataset/TCGA.LUNG.sampleMap_Gistic2_CopyNumber_Gistic2_all_data_by_genes.gz",
                      sep = "\t", header = TRUE, row.names = 1, check.names = FALSE)
pivot <- t(as.matrix(cnv_raw))
rownames(pivot) <- clean_sample_ids(rownames(pivot))

cnv_mean        <- rowMeans(pivot, na.rm = TRUE)
n_non_na        <- rowSums(!is.na(pivot))
cnv_var         <- rowSums((pivot - cnv_mean)^2, na.rm = TRUE) / (n_non_na - 1)
cnv_std         <- sqrt(cnv_var)
cnv_amp_frac    <- rowMeans(pivot > 0.5,  na.rm = TRUE)
cnv_del_frac    <- rowMeans(pivot < -0.5, na.rm = TRUE)
cnv_neutral_frac <- rowMeans(abs(pivot) <= 0.5, na.rm = TRUE)
cnv_total_burden <- rowMeans(abs(pivot), na.rm = TRUE)

cnv_s <- data.frame(
  sample_id        = rownames(pivot),
  cnv_mean         = cnv_mean,
  cnv_std          = cnv_std,
  cnv_amp_frac     = cnv_amp_frac,
  cnv_del_frac     = cnv_del_frac,
  cnv_neutral_frac = cnv_neutral_frac,
  cnv_total_burden = cnv_total_burden
)

df <- merge(factors, cnv_s, by = "sample_id")
df <- merge(df, clin, by = "sample_id")
df <- merge(df, surv[, c("sample_id", "OS_event", "OS_days")], by = "sample_id")
df <- df[!is.na(df$OS_days) & df$OS_days > 0, ]

stage_map <- c(
  "Stage I" = 1, "Stage IA" = 1, "Stage IB" = 1,
  "Stage II" = 2, "Stage IIA" = 2, "Stage IIB" = 2,
  "Stage III" = 3, "Stage IIIA" = 3, "Stage IIIB" = 3,
  "Stage IV" = 4
)
df$stage_num    <- stage_map[df$pathologic_stage]
df$stage_simple <- ifelse(df$stage_num == 1, "I",
                   ifelse(df$stage_num == 2, "II",
                   ifelse(df$stage_num == 3, "III",
                   ifelse(df$stage_num == 4, "IV", NA))))

df_ph_subset <- df[!is.na(df$stage_simple), ]
cox_model    <- coxph(Surv(OS_days, OS_event) ~ Factor2 + Factor3 + stage_simple, data = df_ph_subset)
print(summary(cox_model))

ph_test <- cox.zph(cox_model)
print(ph_test)

png("c:/Users/shmso/UCD/Spring/internship/dataset/plot_cox_ph_test.png",
    width = 800, height = 800, res = 120)
print(ggcoxzph(ph_test))
dev.off()

factor_cols <- colnames(factors)[colnames(factors) != "sample_id"]
cnv_cols    <- c("cnv_mean", "cnv_std", "cnv_amp_frac", "cnv_del_frac",
                 "cnv_neutral_frac", "cnv_total_burden")
x_matrix    <- as.matrix(df[, c(factor_cols, cnv_cols)])
y_surv      <- Surv(df$OS_days, df$OS_event)

set.seed(42)
cv_fit   <- cv.glmnet(x_matrix, y_surv, family = "cox", alpha = 1)
coef_min <- as.matrix(coef(cv_fit, s = "lambda.min"))
coef_1se <- as.matrix(coef(cv_fit, s = "lambda.1se"))

lasso_coefs <- data.frame(
  Feature  = rownames(coef_min),
  Coef_min = coef_min[, 1],
  Coef_1se = coef_1se[, 1]
)
write.csv(lasso_coefs,
          "c:/Users/shmso/UCD/Spring/internship/dataset/survival_lasso_coefficients.csv",
          row.names = FALSE)
print(lasso_coefs[lasso_coefs$Coef_min != 0, ])

df_roc_clean <- df[!is.na(df$stage_num), ]
cox_pred_fit <- coxph(Surv(OS_days, OS_event) ~ Factor2 + Factor3 + stage_num, data = df_roc_clean)
risk_score   <- predict(cox_pred_fit, type = "lp")

t_roc <- timeROC(T      = df_roc_clean$OS_days,
                 delta  = df_roc_clean$OS_event,
                 marker = risk_score,
                 cause  = 1,
                 times  = c(365, 1095, 1825),
                 ROC    = TRUE,
                 iid    = TRUE)
print(t_roc)

png("c:/Users/shmso/UCD/Spring/internship/dataset/plot_time_roc.png",
    width = 600, height = 600, res = 120)
plot(t_roc, time = 365,  col = "red",   title = FALSE, lwd = 2)
plot(t_roc, time = 1095, col = "blue",  add = TRUE,    lwd = 2)
plot(t_roc, time = 1825, col = "green", add = TRUE,    lwd = 2)
legend("bottomright", c(
  paste("1y AUC =", round(t_roc$AUC[1], 2)),
  paste("3y AUC =", round(t_roc$AUC[2], 2)),
  paste("5y AUC =", round(t_roc$AUC[3], 2))
), col = c("red", "blue", "green"), lwd = 2, bty = "n")
title("Time-Dependent ROC")
dev.off()
