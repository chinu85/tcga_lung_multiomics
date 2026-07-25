"""
=============================================================================
TCGA LUNG – Multi-Omics Hidden Pattern Discovery
=============================================================================
Integrates:
  1. MOFA+ latent factors (mofa_factors.csv)
  2. CNV pivot table  (TCGA.LUNG.sampleMap_Gistic2_CopyNumber_Gistic2_all_data_by_genes.gz)
  3. Survival data    (downloaded from TCGA Xena Hub)
  4. Clinical matrix  (TCGA LUNG sampleMap_LUNG_clinicalMatrix.tsv.txt)

Analysis targets:
  - Gender differences in MOFA factors / CNV burden
  - Age-at-diagnosis groupings
  - Tumor status (TUMOR FREE vs WITH TUMOR)
  - Treatment response (Complete Remission, Progressive Disease, Stable Disease)
  - Vital status / survival time
  - Cancer subtype (LUAD vs LUSC)
  - Smoking history
  - Pathologic stage
  - Mutual-information ranking of clinical variables against MOFA factors
=============================================================================
"""

import os, urllib.request, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, kruskal, spearmanr
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE     = "c:/Users/shmso/UCD/Spring/internship"
DATA     = os.path.join(BASE, "dataset")
OUT      = os.path.join(BASE, "analysis_output")
os.makedirs(OUT, exist_ok=True)

PATH_FACTORS  = os.path.join(DATA, "mofa_factors.csv")
PATH_CNV_GZ   = os.path.join(DATA, "TCGA.LUNG.sampleMap_Gistic2_CopyNumber_Gistic2_all_data_by_genes.gz")
PATH_CLINICAL = os.path.join(DATA, "TCGA LUNG sampleMap_LUNG_clinicalMatrix.tsv.txt")
PATH_SURVIVAL = os.path.join(DATA, "LUNG_survival.txt")
SURVIVAL_URL  = "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/survival%2FLUNG_survival.txt"

# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

def save_fig(name, dpi=150):
    path = os.path.join(OUT, name)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"   Saved -> {path}")

def clean_sample_id(sid):
    if pd.isna(sid):
        return np.nan
    s = str(sid).strip().upper()
    parts = s.split("-")
    if len(parts) >= 4:
        return "-".join(parts[:4])
    return s

def strip_to_patient(sid):
    if pd.isna(sid):
        return np.nan
    s = str(sid).strip().upper()
    parts = s.split("-")
    if len(parts) >= 3:
        return "-".join(parts[:3])
    return s

def factor_cols(df):
    return [c for c in df.columns if c.startswith("Factor")]

# ═══════════════════════════════════════════════════════════════════
# 1.  DOWNLOAD SURVIVAL DATA
# ═══════════════════════════════════════════════════════════════════

def download_survival():
    if os.path.exists(PATH_SURVIVAL):
        print(f"[1/6] Survival file already present: {PATH_SURVIVAL}")
    else:
        print(f"[1/6] Downloading survival data ...")
        urllib.request.urlretrieve(SURVIVAL_URL, PATH_SURVIVAL)
        print(f"      Saved -> {PATH_SURVIVAL}")

# ═══════════════════════════════════════════════════════════════════
# 2.  LOAD & PIVOT CNV
# ═══════════════════════════════════════════════════════════════════

def load_cnv_pivot():
    print("[2/6] Loading and pivoting CNV data (genes x samples -> samples x genes) ...")
    df = pd.read_csv(PATH_CNV_GZ, sep="\t", index_col=0, compression="gzip")
    print(f"      Raw shape (genes x samples): {df.shape}")

    # File is genes x samples; transpose -> samples x genes
    pivot = df.T
    pivot.index = [clean_sample_id(s) for s in pivot.index]
    pivot.index.name = "sample_id"

    cnv_summary = pd.DataFrame({
        "cnv_mean":         pivot.mean(axis=1),
        "cnv_std":          pivot.std(axis=1),
        "cnv_amp_frac":     (pivot > 0.5).mean(axis=1),
        "cnv_del_frac":     (pivot < -0.5).mean(axis=1),
        "cnv_neutral_frac": (pivot.abs() <= 0.5).mean(axis=1),
        "cnv_total_burden": pivot.abs().mean(axis=1),
    })
    print(f"      Pivot shape (samples x genes): {pivot.shape}")
    print(f"      CNV summary computed for {len(cnv_summary)} samples")
    return pivot, cnv_summary

# ═══════════════════════════════════════════════════════════════════
# 3.  LOAD CLINICAL + SURVIVAL
# ═══════════════════════════════════════════════════════════════════

def load_clinical():
    print("[3/6] Loading clinical matrix ...")
    clin = pd.read_csv(PATH_CLINICAL, sep="\t", low_memory=False)
    clin.columns = [c.strip() for c in clin.columns]
    clin["sample_id"]  = clin["sampleID"].apply(clean_sample_id)
    clin["patient_id"] = clin["sampleID"].apply(strip_to_patient)

    clin["gender"]        = clin["gender"].str.upper().str.strip()
    clin["vital_status"]  = clin["vital_status"].str.upper().str.strip() if "vital_status" in clin else np.nan
    clin["cancer_status"] = clin["person_neoplasm_cancer_status"].str.upper().str.strip() if "person_neoplasm_cancer_status" in clin else np.nan
    clin["treatment_resp"]= clin["primary_therapy_outcome_success"].str.strip() if "primary_therapy_outcome_success" in clin else np.nan
    clin["followup_resp"] = clin["followup_treatment_success"].str.strip() if "followup_treatment_success" in clin else np.nan
    clin["best_response"] = clin["followup_resp"].combine_first(clin["treatment_resp"])

    clin["age"]       = pd.to_numeric(clin["age_at_initial_pathologic_diagnosis"], errors="coerce")
    clin["age_group"] = pd.cut(clin["age"],
                                bins=[0, 50, 60, 70, 80, 120],
                                labels=["<=50", "51-60", "61-70", "71-80", ">80"])
    clin["stage"]   = clin["pathologic_stage"].str.strip() if "pathologic_stage" in clin else np.nan
    clin["subtype"] = clin["_primary_disease"].str.strip() if "_primary_disease" in clin else np.nan
    clin["smoking"] = clin["tobacco_smoking_history"].astype(str).str.strip()

    print(f"      Clinical samples: {len(clin)}")
    print(f"      Gender:\n{clin['gender'].value_counts().to_string()}")
    print(f"      Cancer status:\n{clin['cancer_status'].value_counts().to_string()}")
    print(f"      Best response:\n{clin['best_response'].value_counts().to_string()}")

    print("[3/6] Loading survival data ...")
    surv = pd.read_csv(PATH_SURVIVAL, sep="\t", low_memory=False)
    surv.columns = [c.strip() for c in surv.columns]
    surv = surv.rename(columns={
        "sample":    "sample_id",
        "OS":        "OS_event",
        "OS.time":   "OS_days",
        "DSS":       "DSS_event",
        "DSS.time":  "DSS_days",
        "DFI":       "DFI_event",
        "DFI.time":  "DFI_days",
        "PFI":       "PFI_event",
        "PFI.time":  "PFI_days",
    })
    surv["sample_id"]  = surv["sample_id"].apply(clean_sample_id)
    surv["patient_id"] = surv["sample_id"].apply(strip_to_patient)

    merged = pd.merge(
        clin,
        surv[[c for c in surv.columns if c not in clin.columns or c == "sample_id"]],
        on="sample_id",
        how="left"
    )
    print(f"      After merge with survival: {len(merged)} rows")
    return merged

# ═══════════════════════════════════════════════════════════════════
# 4.  LOAD MOFA FACTORS
# ═══════════════════════════════════════════════════════════════════

def load_mofa():
    print("[4/6] Loading MOFA+ factors ...")
    factors = pd.read_csv(PATH_FACTORS, index_col=0)
    factors.index = [str(i).split("|")[0] for i in factors.index]
    factors.index = [clean_sample_id(s) for s in factors.index]
    factors.index.name = "sample_id"
    print(f"      MOFA factors shape: {factors.shape}")
    return factors

# ═══════════════════════════════════════════════════════════════════
# 5.  INTEGRATE
# ═══════════════════════════════════════════════════════════════════

def integrate(factors, cnv_summary, clinical):
    print("[5/6] Integrating all datasets ...")
    df   = factors.reset_index()
    cnv_s = cnv_summary.reset_index()
    cnv_s.columns = ["sample_id"] + list(cnv_summary.columns)

    merged = df.merge(cnv_s, on="sample_id", how="inner")
    merged = merged.merge(clinical, on="sample_id", how="inner")
    print(f"      Integrated shape: {merged.shape}")
    print(f"      Unique samples: {merged['sample_id'].nunique()}")
    return merged

# ═══════════════════════════════════════════════════════════════════
# 6.  ANALYSIS PLOTS
# ═══════════════════════════════════════════════════════════════════

def plot_gender_factors(df):
    print("   -> Gender vs MOFA factors")
    fc  = factor_cols(df)[:6]
    sub = df[df["gender"].isin(["MALE","FEMALE"])].copy()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("MOFA Latent Factors by Gender", fontsize=15, fontweight="bold")
    for i, (ax, f) in enumerate(zip(axes.flat, fc)):
        for g, col in [("MALE","#4C72B0"), ("FEMALE","#DD8452")]:
            vals = sub[sub["gender"]==g][f].dropna()
            ax.hist(vals, bins=30, alpha=0.6, color=col, label=g, density=True)
        tf = sub[sub["gender"]=="MALE"][f].dropna()
        ff = sub[sub["gender"]=="FEMALE"][f].dropna()
        _, pval = mannwhitneyu(tf, ff, alternative="two-sided") if len(tf)>1 and len(ff)>1 else (0, 1)
        ax.set_title(f"{f}  (p={pval:.3f})", fontsize=10)
        ax.set_xlabel("Factor Value"); ax.set_ylabel("Density")
        if i == 0: ax.legend()
    plt.tight_layout()
    save_fig("01_gender_mofa_factors.png")

def plot_age_factors(df):
    print("   -> Age group vs MOFA factors")
    fc    = factor_cols(df)[:4]
    sub   = df.dropna(subset=["age_group"]).copy()
    sub["age_group"] = sub["age_group"].astype(str)
    order = ["<=50", "51-60", "61-70", "71-80", ">80"]
    pal   = sns.color_palette("viridis", len(order))
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("MOFA Factors Across Age Groups", fontsize=14, fontweight="bold")
    for ax, f in zip(axes, fc):
        groups = [sub[sub["age_group"]==g][f].dropna() for g in order]
        valid  = [g for g in groups if len(g)>1]
        stat, pval = kruskal(*valid) if len(valid)>1 else (0, 1)
        sns.boxplot(data=sub, x="age_group", y=f, order=order, palette=pal, ax=ax, showfliers=False)
        ax.set_title(f"{f}\np={pval:.3f}", fontsize=9)
        ax.set_xlabel("Age Group"); ax.set_ylabel("Factor Value")
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    save_fig("02_age_mofa_factors.png")

def plot_cancer_status(df):
    print("   -> Tumor status vs MOFA factors")
    fc  = factor_cols(df)[:6]
    sub = df[df["cancer_status"].isin(["TUMOR FREE","WITH TUMOR"])].copy()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("MOFA Factors: Tumor Free vs With Tumor", fontsize=14, fontweight="bold")
    for ax, f in zip(axes.flat, fc):
        sns.violinplot(data=sub, x="cancer_status", y=f,
                       palette=["#55A868","#C44E52"], ax=ax, linewidth=0.8)
        tf = sub[sub["cancer_status"]=="TUMOR FREE"][f].dropna()
        wt = sub[sub["cancer_status"]=="WITH TUMOR"][f].dropna()
        pval = mannwhitneyu(tf, wt, alternative="two-sided")[1] if len(tf)>1 and len(wt)>1 else 1
        ax.set_title(f"{f}  p={pval:.3f}", fontsize=9)
        ax.set_xlabel(""); ax.set_ylabel("Factor Value")
        ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    save_fig("03_tumor_status_mofa.png")

def plot_treatment_response(df):
    print("   -> Treatment response vs MOFA factors")
    resp_cats = ["Complete Remission/Response","Partial Remission/Response",
                 "Stable Disease","Progressive Disease"]
    fc  = factor_cols(df)[:4]
    sub = df[df["best_response"].isin(resp_cats)].copy()
    if len(sub) < 10:
        print("     Skipped - too few samples with known response")
        return
    pal = sns.color_palette("Set2", len(resp_cats))
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("MOFA Factors by Treatment Response", fontsize=14, fontweight="bold")
    for ax, f in zip(axes, fc):
        groups = [sub[sub["best_response"]==r][f].dropna() for r in resp_cats]
        valid  = [g for g in groups if len(g)>1]
        stat, pval = kruskal(*valid) if len(valid)>1 else (0, 1)
        sns.boxplot(data=sub, x="best_response", y=f, order=resp_cats,
                    palette=pal, ax=ax, showfliers=False)
        ax.set_title(f"{f}  p={pval:.3f}", fontsize=9)
        ax.set_xlabel(""); ax.tick_params(axis="x", rotation=25)
        ax.set_ylabel("Factor Value")
    plt.tight_layout()
    save_fig("04_treatment_response_mofa.png")

def plot_cnv_burden(df):
    print("   -> CNV burden analysis")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Chromosomal Instability (CNV Burden) by Clinical Variables",
                 fontsize=13, fontweight="bold")

    sub_g = df[df["gender"].isin(["MALE","FEMALE"])].copy()
    sns.boxplot(data=sub_g, x="gender", y="cnv_total_burden",
                palette=["#4C72B0","#DD8452"], ax=axes[0], showfliers=False)
    pv = mannwhitneyu(sub_g[sub_g["gender"]=="MALE"]["cnv_total_burden"].dropna(),
                      sub_g[sub_g["gender"]=="FEMALE"]["cnv_total_burden"].dropna())[1]
    axes[0].set_title(f"Gender  (p={pv:.3f})"); axes[0].set_ylabel("Mean |CNV| per gene")

    sub_s = df[df["cancer_status"].isin(["TUMOR FREE","WITH TUMOR"])].copy()
    sns.boxplot(data=sub_s, x="cancer_status", y="cnv_total_burden",
                palette=["#55A868","#C44E52"], ax=axes[1], showfliers=False)
    pv = mannwhitneyu(sub_s[sub_s["cancer_status"]=="TUMOR FREE"]["cnv_total_burden"].dropna(),
                      sub_s[sub_s["cancer_status"]=="WITH TUMOR"]["cnv_total_burden"].dropna())[1]
    axes[1].set_title(f"Cancer Status  (p={pv:.3f})"); axes[1].set_ylabel("")
    axes[1].tick_params(axis="x", rotation=15)

    sub_t = df[df["subtype"].notna()].copy()
    sns.boxplot(data=sub_t, x="subtype", y="cnv_total_burden",
                palette="Set3", ax=axes[2], showfliers=False)
    axes[2].set_title("Cancer Subtype"); axes[2].set_ylabel("")
    axes[2].tick_params(axis="x", rotation=20)
    plt.tight_layout()
    save_fig("05_cnv_burden_clinical.png")

def plot_survival_factors(df):
    print("   -> Survival time vs MOFA factors")
    os_col = next((c for c in df.columns if "OS_days" in c or "OS.time" in c), None)
    if os_col is None:
        print("     OS column not found - skipping")
        return
    fc  = factor_cols(df)[:8]
    sub = df[[os_col] + fc].copy()
    sub[os_col] = pd.to_numeric(sub[os_col], errors="coerce")
    sub = sub.dropna(subset=[os_col])
    corrs, pvals = [], []
    for f in fc:
        vals = sub[[f, os_col]].dropna()
        if len(vals) > 5:
            r, p = spearmanr(vals[f], vals[os_col])
        else:
            r, p = 0, 1
        corrs.append(r); pvals.append(p)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#C44E52" if p < 0.05 else "#95A5A6" for p in pvals]
    bars   = ax.barh(fc, corrs, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Spearman r  (Factor <-> OS days)")
    ax.set_title("Correlation of MOFA Factors with Overall Survival\n(red = p < 0.05)",
                 fontweight="bold")
    for bar, p in zip(bars, pvals):
        x_pos = bar.get_width() + (0.005 if bar.get_width() >= 0 else -0.005)
        ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                f"p={p:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    save_fig("06_survival_mofa_correlation.png")

def plot_stage_factors(df):
    print("   -> Pathologic stage vs MOFA factors")
    stage_map = {"Stage I":"I","Stage IA":"I","Stage IB":"I",
                 "Stage II":"II","Stage IIA":"II","Stage IIB":"II",
                 "Stage III":"III","Stage IIIA":"III","Stage IIIB":"III",
                 "Stage IV":"IV"}
    df2 = df.copy()
    df2["stage_simple"] = df2["stage"].map(stage_map)
    sub = df2[df2["stage_simple"].isin(["I","II","III","IV"])].copy()
    fc  = factor_cols(df)[:4]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("MOFA Factors by Pathologic Stage", fontsize=13, fontweight="bold")
    pal = sns.color_palette("coolwarm", 4)
    for ax, f in zip(axes, fc):
        groups = [sub[sub["stage_simple"]==s][f].dropna() for s in ["I","II","III","IV"]]
        valid  = [g for g in groups if len(g)>1]
        stat, pval = kruskal(*valid) if len(valid)>1 else (0, 1)
        sns.boxplot(data=sub, x="stage_simple", y=f, order=["I","II","III","IV"],
                    palette=pal, ax=ax, showfliers=False)
        ax.set_title(f"{f}  p={pval:.3f}", fontsize=9)
        ax.set_xlabel("Stage"); ax.set_ylabel("Factor Value")
    plt.tight_layout()
    save_fig("07_stage_mofa_factors.png")

def plot_mi_heatmap(df):
    print("   -> Mutual information heatmap")
    fc   = factor_cols(df)
    cats = [c for c in ["gender","age_group","cancer_status","best_response",
                         "vital_status","stage","subtype","smoking",
                         "radiation_therapy","targeted_molecular_therapy"]
            if c in df.columns]
    mi_rows = []
    for cat in cats:
        sub = df[fc + [cat]].dropna(subset=[cat]).copy()
        if sub[cat].nunique() < 2: continue
        le = LabelEncoder()
        y  = le.fit_transform(sub[cat].astype(str))
        X  = sub[fc].fillna(sub[fc].median())
        mi = mutual_info_classif(X, y, random_state=42)
        mi_rows.append(pd.Series(mi, index=fc, name=cat))
    if not mi_rows: return
    mi_df = pd.DataFrame(mi_rows)
    fig, ax = plt.subplots(figsize=(max(12, len(fc)), max(4, len(cats))))
    sns.heatmap(mi_df, cmap="YlOrRd", annot=True, fmt=".2f", linewidths=0.3,
                cbar_kws={"label":"Mutual Information"}, ax=ax)
    ax.set_title("Mutual Information: Clinical Variables x MOFA Factors",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("MOFA Factor"); ax.set_ylabel("Clinical Variable")
    plt.tight_layout()
    save_fig("08_mutual_info_heatmap.png")

def plot_tsne(df):
    print("   -> t-SNE of MOFA latent space")
    fc  = factor_cols(df)
    sub = df[fc + ["gender","cancer_status","subtype"]].dropna(subset=fc).copy()
    X   = StandardScaler().fit_transform(sub[fc].fillna(0))
    tsne   = TSNE(n_components=2, perplexity=min(30, len(X)-1), random_state=42, max_iter=1000)
    coords = tsne.fit_transform(X)
    sub["tSNE1"] = coords[:,0]; sub["tSNE2"] = coords[:,1]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("t-SNE of MOFA Latent Space", fontsize=14, fontweight="bold")
    for ax, col in zip(axes, ["gender","cancer_status","subtype"]):
        cats = sub[col].dropna().unique()
        pal  = dict(zip(cats, sns.color_palette("tab10", len(cats))))
        for cat in cats:
            m = sub[col] == cat
            ax.scatter(sub.loc[m,"tSNE1"], sub.loc[m,"tSNE2"],
                       c=[pal[cat]], label=str(cat), s=18, alpha=0.7, edgecolors="none")
        ax.set_title(f"Colored by: {col}", fontsize=10)
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
        ax.legend(fontsize=7, markerscale=1.5, loc="best")
    plt.tight_layout()
    save_fig("09_tsne_mofa.png")

def plot_rf_importance(df):
    print("   -> Random Forest feature importance")
    fc        = factor_cols(df)
    cnv_feats = ["cnv_mean","cnv_std","cnv_amp_frac","cnv_del_frac","cnv_total_burden"]
    feats     = fc + cnv_feats
    targets   = {
        "Cancer Status":   "cancer_status",
        "Gender":          "gender",
        "Vital Status":    "vital_status",
    }
    fig, axes = plt.subplots(1, len(targets), figsize=(18, 6))
    fig.suptitle("Random Forest Feature Importances (MOFA factors + CNV metrics)",
                 fontsize=13, fontweight="bold")
    for ax, (title, tgt) in zip(axes, targets.items()):
        if tgt not in df.columns:
            ax.set_visible(False); continue
        sub = df[feats + [tgt]].dropna(subset=[tgt]).copy()
        sub[tgt] = sub[tgt].astype(str)
        if sub[tgt].nunique() < 2:
            ax.set_visible(False); continue
        le = LabelEncoder()
        y  = le.fit_transform(sub[tgt])
        X  = sub[feats].fillna(sub[feats].median())
        rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        imp = pd.Series(rf.feature_importances_, index=feats).sort_values(ascending=True)
        imp.tail(15).plot(kind="barh", ax=ax, color="#4C72B0")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Importance")
    plt.tight_layout()
    save_fig("10_rf_feature_importance.png")

def plot_smoking(df):
    print("   -> Smoking history vs MOFA factors")
    smoke_map = {"1":"Never","2":"Current","3":"Reformed(<=15yr)",
                 "4":"Reformed(>15yr)","5":"Not Documented","nan":"Unknown"}
    df2 = df.copy()
    df2["smoke_cat"] = df2["smoking"].map(smoke_map).fillna("Unknown")
    order = ["Never","Current","Reformed(<=15yr)","Reformed(>15yr)","Not Documented"]
    fc    = factor_cols(df)[:2]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Smoking History vs Top MOFA Factors", fontsize=13, fontweight="bold")
    pal = sns.color_palette("husl", len(order))
    for ax, f in zip(axes, fc):
        sub    = df2[df2["smoke_cat"].isin(order)]
        groups = [sub[sub["smoke_cat"]==s][f].dropna() for s in order]
        valid  = [g for g in groups if len(g)>1]
        stat, pval = kruskal(*valid) if len(valid)>1 else (0, 1)
        sns.boxplot(data=sub, x="smoke_cat", y=f, order=order,
                    palette=pal, ax=ax, showfliers=False)
        ax.set_title(f"{f}  p={pval:.3f}", fontsize=9)
        ax.set_xlabel(""); ax.tick_params(axis="x", rotation=30)
        ax.set_ylabel("Factor Value")
    plt.tight_layout()
    save_fig("11_smoking_mofa.png")

def summary_table(df):
    print("   -> Building summary statistics table")
    fc   = factor_cols(df)
    rows = []
    cats = {
        "Gender":        ("gender",        ["MALE","FEMALE"]),
        "Cancer Status": ("cancer_status", ["TUMOR FREE","WITH TUMOR"]),
        "Best Response": ("best_response", ["Complete Remission/Response",
                                            "Stable Disease","Progressive Disease",
                                            "Partial Remission/Response"]),
        "Vital Status":  ("vital_status",  ["LIVING","DECEASED"]),
    }
    for label, (col, vals) in cats.items():
        if col not in df.columns: continue
        for f in fc[:6]:
            for v in vals:
                g = df[df[col]==v][f].dropna()
                if len(g) > 0:
                    rows.append({"Category":label,"Value":v,"Factor":f,
                                 "N":len(g),"Mean":round(g.mean(),4),
                                 "Std":round(g.std(),4),"Median":round(g.median(),4)})
    tbl = pd.DataFrame(rows)
    path = os.path.join(OUT, "summary_statistics.csv")
    tbl.to_csv(path, index=False)
    print(f"      Saved -> {path}")
    return tbl

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*65)
    print("   TCGA LUNG - Multi-Omics Hidden Pattern Discovery")
    print("="*65 + "\n")

    download_survival()
    cnv_pivot, cnv_summary = load_cnv_pivot()
    clinical               = load_clinical()
    factors                = load_mofa()
    df                     = integrate(factors, cnv_summary, clinical)

    print("\n[6/6] Generating analysis plots ...")
    plot_gender_factors(df)
    plot_age_factors(df)
    plot_cancer_status(df)
    plot_treatment_response(df)
    plot_cnv_burden(df)
    plot_survival_factors(df)
    plot_stage_factors(df)
    plot_mi_heatmap(df)
    plot_tsne(df)
    plot_rf_importance(df)
    plot_smoking(df)
    summary_table(df)

    pivot_out = os.path.join(OUT, "cnv_pivot_sample.csv")
    cnv_pivot.iloc[:, :200].to_csv(pivot_out)
    print(f"\n   CNV pivot preview (first 200 genes) -> {pivot_out}")
    print("\n" + "="*65)
    print(f"   All outputs -> {OUT}")
    print("="*65 + "\n")

if __name__ == "__main__":
    main()
