# Multi-Omics Factor Analysis and Machine Learning for Clinical Subtyping and Outcome Prediction in TCGA Lung Cancer

## 📌 Project Overview & Introduction

Lung cancer is the leading cause of cancer-related deaths worldwide. Most cases are Non-Small Cell Lung Cancer (NSCLC), which primarily consists of two histological subtypes: **Lung Adenocarcinoma (LUAD)** and **Lung Squamous Cell Carcinoma (LUSC)**. Although these two subtypes originate in different lung tissues and harbor distinct genomic alterations, patients diagnosed at identical disease stages are frequently treated with similar regimens, leading to variable therapeutic responses and outcomes.

Single-omics approaches (e.g., analyzing RNA-seq expression alone) provide a partial view of cellular state while missing structural genomic driver events like Copy Number Variations (CNVs). Furthermore, public retrospective datasets often contain subtle data quality issues—such as sample barcode duplicates—that can quietly leak information between training and testing cross-validation folds, leading to overly optimistic machine learning performance metrics.

In this project, we built an end-to-end multi-omics analysis and machine learning pipeline applied to **1,010 clean, non-duplicated patient records** from the TCGA Lung Cancer cohort (**TCGA-LUNG**). Combining RNA-seq gene expression and GISTIC2 copy number variation data using **Multi-Omics Factor Analysis (MOFA+)**, we extracted 14 latent biological factors and evaluated five machine learning model families across clinical subtyping, disease staging, and overall survival prediction.

### Key Highlights & Results:
* **Data Leakage & Deduplication Audit:** Identified and removed 121 duplicate sample entries using a strict 4-field barcode filter, preventing cross-validation data leakage.
* **Histological Subtype Discrimination (AUC = 0.98):** Factor 1 separated LUAD from LUSC cleanly, driven by 3q26 focal copy number amplifications (*SOX2*, *DCUN1D1*, *ATP11B*) paired with downstream basal squamous transcriptional activation (*KRT5*, *DSG3*, *KRT14*).
* **Tumor Staging & Cell Proliferation ($r = 0.944$):** Factor 2 tracked clinical stage progression and mapped to E2F/MYC cell proliferation pathways, verified via independent *limma* differential expression ($r = 0.944, p \approx 0$).
* **Pipeline Audit & Seed Stability:** Multi-seed evaluations across five random initializations validated core factor stability (TCC > 0.90) while exposing Factor 9 as an unrepeatable initialization artifact (TCC = 0.674).

---

## 📁 Repository Contents

```
tcga_lung_multiomics_github/
├── README.md                 # Project description, intro, and run guide
├── requirements.txt          # Python dependencies
├── src/                      # Python & R code files
│   ├── tcga_lung_analysis.py # Barcode deduplication & dataset preparation
│   ├── run_mofa.R            # MOFA+ multi-omics factor integration (R)
│   ├── run_mofa.py           # MOFA+ factor integration (Python wrapper)
│   ├── mofa_stability.py     # Multi-seed factor stability audit
│   ├── ml_dl_pipeline.py     # Machine learning classifiers (LogReg, XGB, LightGBM, MLP, VAE)
│   ├── convergent_validation.R # limma differential expression validation
│   ├── pathway_enrichment.R  # clusterProfiler ORA & fgsea Hallmark pathway analysis
│   ├── survival_modeling.R   # Schoenfeld residual test & LASSO Cox survival models
│   ├── export_pca_plots.R    # Factor space PCA plotting
│   ├── generate_additional_plots.R # Demographics & subtype plotting
│   └── mofa2_downstream.R    # MOFA downstream inspection
└── outputs/                  # Key output plots & CSV summary tables
    ├── mofa_stability_plot.png
    ├── plot_data_imbalance.png
    ├── plot_gender_boxplot.png
    ├── factor2_de_volcano.png
    ├── factor2_convergence_scatter.png
    ├── plot_factor1_genes.png
    ├── plot_factor2_genes.png
    ├── plot_cox_ph_test.png
    ├── plot_time_roc.png
    ├── plot_pca_subtype.png
    ├── mofa_factors.csv
    ├── mofa_weights.csv
    └── summary_statistics.csv
```

---

## ⚡ Execution Order

1. **Preprocessing & Deduplication:** `python src/tcga_lung_analysis.py`
2. **MOFA+ Integration:** `Rscript src/run_mofa.R`
3. **Seed Stability Audit:** `python src/mofa_stability.py`
4. **Machine Learning Classifiers:** `python src/ml_dl_pipeline.py`
5. **Limma Validation:** `Rscript src/convergent_validation.R`
6. **Pathway Enrichment:** `Rscript src/pathway_enrichment.R`
7. **Survival Analysis:** `Rscript src/survival_modeling.R`
