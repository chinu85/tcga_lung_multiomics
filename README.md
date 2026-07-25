# TCGA Lung Cancer Multi-Omics Pipeline

This repository contains the core **Python** and **R** code files and output figures/tables for the TCGA Lung Cancer multi-omics factor modeling pipeline.

---

## 📁 Repository Contents

```
tcga_lung_multiomics_github/
├── README.md                 # Project description and run guide
├── requirements.txt          # Python dependencies
├── src/                      # Python & R code files
│   ├── tcga_lung_analysis.py # Barcode deduplication & sample cleanup
│   ├── run_mofa.R            # MOFA+ multi-omics factor integration (R)
│   ├── run_mofa.py           # MOFA+ factor integration (Python wrapper)
│   ├── mofa_stability.py     # Multi-seed factor stability audit
│   ├── ml_dl_pipeline.py     # ML models (Logistic Regression, XGBoost, LightGBM, MLP, VAE)
│   ├── convergent_validation.R # limma differential expression validation against Factor 2
│   ├── pathway_enrichment.R  # clusterProfiler ORA & fgsea Hallmark pathway analysis
│   ├── survival_modeling.R   # Schoenfeld residual test & LASSO Cox survival models
│   ├── export_pca_plots.R    # Factor space PCA plotting
│   └── generate_additional_plots.R # Demographics & subtype plotting
└── outputs/                  # Key generated figures & summary tables
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
