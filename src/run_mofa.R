# Data Exploration

df_cnv <- read.table("c:/Users/shmso/UCD/Spring/internship/dataset/Gistic2_CopyNumber_Gistic2_all_data_by_genes", sep = "\t", header = TRUE, row.names = 1, check.names = FALSE)
df_exp <- read.table("c:/Users/shmso/UCD/Spring/internship/dataset/HiSeqV2", sep = "\t", header = TRUE, row.names = 1, check.names = FALSE)

cat("--- CNV Dataset ---\n")
print(dim(df_cnv))
print(head(df_cnv[, 1:5]))
cat("Missing values:", any(is.na(df_cnv)), "\n")
cat("Value range:", paste(range(as.matrix(df_cnv), na.rm = TRUE), collapse = " to "), "\n\n")

cat("--- Expression Dataset ---\n")
print(dim(df_exp))
print(head(df_exp[, 1:5]))
cat("Missing values:", any(is.na(df_exp)), "\n")
cat("Value range:", paste(range(as.matrix(df_exp), na.rm = TRUE), collapse = " to "), "\n\n")

# MOFA Integration
if (!requireNamespace("MOFA2", quietly = TRUE)) {
  if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager", repos = "https://cloud.r-project.org")
  }
  BiocManager::install("MOFA2", update = FALSE, ask = FALSE)
}
library(reticulate)
library(MOFA2)
if (!reticulate::py_module_available("mofapy2")) {
  reticulate::py_install("mofapy2", pip = TRUE)
}

common_samples <- intersect(colnames(df_cnv), colnames(df_exp))
df_cnv_sub <- df_cnv[, common_samples]
df_exp_sub <- df_exp[, common_samples]

row_variances <- function(x) {
  rowSums((x - rowMeans(x, na.rm = TRUE))^2, na.rm = TRUE) / (rowSums(!is.na(x)) - 1)
}

var_cnv <- row_variances(as.matrix(df_cnv_sub))
var_exp <- row_variances(as.matrix(df_exp_sub))

top_cnv_genes <- names(sort(var_cnv, decreasing = TRUE)[1:min(5000, length(var_cnv))])
top_exp_genes <- names(sort(var_exp, decreasing = TRUE)[1:min(5000, length(var_exp))])

df_cnv_filtered <- df_cnv_sub[top_cnv_genes, ]
df_exp_filtered <- df_exp_sub[top_exp_genes, ]

rownames(df_cnv_filtered) <- paste0(rownames(df_cnv_filtered), "_CNV")
rownames(df_exp_filtered) <- paste0(rownames(df_exp_filtered), "_Exp")

data <- list(
  CNV = as.matrix(df_cnv_filtered),
  Expression = as.matrix(df_exp_filtered)
)

mofa_obj <- create_mofa(data)

data_opts <- get_default_data_options(mofa_obj)

model_opts <- get_default_model_options(mofa_obj)
model_opts$num_factors <- 15
model_opts$likelihoods <- c("gaussian", "gaussian")

train_opts <- get_default_training_options(mofa_obj)
train_opts$maxiter <- 150
train_opts$verbose <- TRUE

mofa_obj <- prepare_mofa(
  object = mofa_obj,
  data_options = data_opts,
  model_options = model_opts,
  training_options = train_opts
)

run_mofa(mofa_obj, outfile = "c:/Users/shmso/UCD/Spring/internship/dataset/mofa_model_r.hdf5", use_basilisk = FALSE)

mofa_model <- load_model("c:/Users/shmso/UCD/Spring/internship/dataset/mofa_model_r.hdf5")

factors <- get_factors(mofa_model)[[1]]
write.csv(factors, "c:/Users/shmso/UCD/Spring/internship/dataset/mofa_factors_r.csv")

weights <- get_weights(mofa_model)
weights_all <- rbind(weights$CNV, weights$Expression)
write.csv(weights_all, "c:/Users/shmso/UCD/Spring/internship/dataset/mofa_weights_r.csv")
