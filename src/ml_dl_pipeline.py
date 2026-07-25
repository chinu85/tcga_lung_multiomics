
import os, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                              classification_report, ConfusionMatrixDisplay)
from sklearn.pipeline import Pipeline
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARN] xgboost not found – skipping XGB module")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("[WARN] lightgbm not found – skipping LGB module")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("[WARN] shap not found – SHAP plots skipped")

try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.statistics import logrank_test
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False
    print("[WARN] lifelines not found – survival module skipped")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE     = "c:/Users/shmso/UCD/Spring/internship"
DATA     = os.path.join(BASE, "dataset")
OUT_ROOT = os.path.join(BASE, "analysis_output", "ml_results")
os.makedirs(OUT_ROOT, exist_ok=True)

def save_fig(name, dpi=150):
    path = os.path.join(OUT_ROOT, name)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"   Saved -> {path}")


def clean_sample_id(sid):
    if pd.isna(sid): return np.nan
    s = str(sid).strip().upper()
    parts = s.split("-")
    return "-".join(parts[:4]) if len(parts) >= 4 else s

def strip_to_patient(sid):
    if pd.isna(sid): return np.nan
    s = str(sid).strip().upper()
    parts = s.split("-")
    return "-".join(parts[:3]) if len(parts) >= 3 else s

def load_all():
    print("[DATA] Loading MOFA factors ...")
    factors = pd.read_csv(os.path.join(DATA, "mofa_factors.csv"), index_col=0)
    factors.index = [str(i).split("|")[0] for i in factors.index]
    factors.index = [clean_sample_id(s) for s in factors.index]
    factors.index.name = "sample_id"
    factors = factors.reset_index()

    print("[DATA] Loading CNV summary ...")
    cnv_gz = os.path.join(DATA, "TCGA.LUNG.sampleMap_Gistic2_CopyNumber_Gistic2_all_data_by_genes.gz")
    cnv_raw = pd.read_csv(cnv_gz, sep="\t", index_col=0, compression="gzip")
    pivot   = cnv_raw.T
    pivot.index = [clean_sample_id(s) for s in pivot.index]
    pivot.index.name = "sample_id"
    cnv_s = pd.DataFrame({
        "sample_id":        pivot.index,
        "cnv_mean":         pivot.mean(axis=1).values,
        "cnv_std":          pivot.std(axis=1).values,
        "cnv_amp_frac":     (pivot > 0.5).mean(axis=1).values,
        "cnv_del_frac":     (pivot < -0.5).mean(axis=1).values,
        "cnv_neutral_frac": (pivot.abs() <= 0.5).mean(axis=1).values,
        "cnv_total_burden": pivot.abs().mean(axis=1).values,
    })

    print("[DATA] Loading clinical ...")
    clin = pd.read_csv(os.path.join(DATA, "TCGA LUNG sampleMap_LUNG_clinicalMatrix.tsv.txt"),
                       sep="\t", low_memory=False)
    clin.columns = [c.strip() for c in clin.columns]
    clin["sample_id"]  = clin["sampleID"].apply(clean_sample_id)
    clin["patient_id"] = clin["sampleID"].apply(strip_to_patient)

    print("[DATA] Loading survival ...")
    surv = pd.read_csv(os.path.join(DATA, "LUNG_survival.txt"), sep="\t", low_memory=False)
    surv = surv.rename(columns={"sample":"sample_id","OS":"OS_event","OS.time":"OS_days"})
    surv["sample_id"]  = surv["sample_id"].apply(clean_sample_id)
    surv["patient_id"] = surv["sample_id"].apply(strip_to_patient)

    df = factors.merge(cnv_s, on="sample_id", how="inner")
    df = df.merge(clin, on="sample_id", how="inner")
    df = df.merge(surv[["sample_id","OS_event","OS_days"]].dropna(),
                  on="sample_id", how="left")

    df["cancer_status_raw"] = df["person_neoplasm_cancer_status"].str.upper().str.strip()
    df["target_cancer"]  = (df["cancer_status_raw"] == "TUMOR FREE").astype(int)

    df["vital_raw"]      = df["vital_status"].str.upper().str.strip()
    df["target_vital"]   = (df["vital_raw"] == "LIVING").astype(int)

    df["resp_raw"]       = df["primary_therapy_outcome_success"].str.strip()
    df["target_response"]= (df["resp_raw"] == "Complete Remission/Response").astype(int)

    df["subtype_raw"]    = df["_primary_disease"].str.strip()
    df["target_subtype"] = (df["subtype_raw"] == "lung adenocarcinoma").astype(int)

    df["OS_days"]        = pd.to_numeric(df["OS_days"], errors="coerce")
    df["OS_event"]       = pd.to_numeric(df["OS_event"], errors="coerce")

    factor_feats = [f"Factor{i}" for i in range(1,15)]
    cnv_feats    = ["cnv_mean","cnv_std","cnv_amp_frac","cnv_del_frac",
                    "cnv_neutral_frac","cnv_total_burden"]
    omics_feats  = factor_feats + cnv_feats

    df["age_num"] = pd.to_numeric(df["age_at_initial_pathologic_diagnosis"], errors="coerce")
    df["gender_enc"] = (df["gender"].str.upper().str.strip() == "MALE").astype(float)
    stage_map = {"Stage I":1,"Stage IA":1,"Stage IB":1,"Stage II":2,
                 "Stage IIA":2,"Stage IIB":2,"Stage III":3,"Stage IIIA":3,
                 "Stage IIIB":3,"Stage IV":4}
    df["stage_enc"] = df["pathologic_stage"].map(stage_map)
    clin_feats   = ["age_num","gender_enc","stage_enc"]
    full_feats   = omics_feats + clin_feats

    print(f"[DATA] Final integrated shape: {df.shape}, unique samples: {df['sample_id'].nunique()}")
    return df, omics_feats, full_feats, factor_feats, cnv_feats

def run_gradient_boosting(df, omics_feats, full_feats):
    if not (HAS_XGB or HAS_LGB):
        print("[Module 1] Skipped – neither xgboost nor lightgbm available")
        return {}

    print("\n" + "="*60)
    print("  MODULE 1: Gradient Boosting Classifiers")
    print("="*60)

    targets = {
        "Cancer Status":      "target_cancer",
        "Vital Status":       "target_vital",
        "Treatment Response": "target_response",
        "Cancer Subtype":     "target_subtype",
    }
    feature_sets = {"omics_only": omics_feats, "omics+clinical": full_feats}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for feat_name, feats in feature_sets.items():
        for tgt_name, tgt_col in targets.items():
            sub = df[feats + [tgt_col]].dropna()
            if sub[tgt_col].nunique() < 2 or len(sub) < 50:
                continue
            X = sub[feats].fillna(sub[feats].median()).values
            y = sub[tgt_col].values
            scaler = StandardScaler()
            X_sc   = scaler.fit_transform(X)

            for model_name, use_model in [("XGBoost", HAS_XGB), ("LightGBM", HAS_LGB)]:
                if not use_model:
                    continue
                if model_name == "XGBoost":
                    clf = xgb.XGBClassifier(n_estimators=300, max_depth=4,
                                            learning_rate=0.05, subsample=0.8,
                                            use_label_encoder=False,
                                            eval_metric="logloss", random_state=42,
                                            n_jobs=-1, verbosity=0)
                else:
                    clf = lgb.LGBMClassifier(n_estimators=300, max_depth=4,
                                              learning_rate=0.05, subsample=0.8,
                                              random_state=42, n_jobs=-1,
                                              verbose=-1)
                probs = cross_val_predict(clf, X_sc, y, cv=cv, method="predict_proba")[:,1]
                auc   = roc_auc_score(y, probs)
                print(f"   [{model_name}] {feat_name} -> {tgt_name}: AUC = {auc:.4f}")
                results.append({"Model": model_name, "Features": feat_name,
                                 "Target": tgt_name, "AUC": round(auc,4)})

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("ROC Curves – Gradient Boosting (omics+clinical features)",
                 fontsize=13, fontweight="bold")

    for ax, (tgt_name, tgt_col) in zip(axes, [("Cancer Status","target_cancer"),
                                                ("Vital Status","target_vital")]):
        sub = df[full_feats + [tgt_col]].dropna()
        X   = StandardScaler().fit_transform(sub[full_feats].fillna(sub[full_feats].median()).values)
        y   = sub[tgt_col].values
        for model_name, use_model, color in [("XGBoost","XGBoost","#C44E52"),
                                               ("LightGBM","LightGBM","#4C72B0")]:
            if (model_name=="XGBoost" and not HAS_XGB) or (model_name=="LightGBM" and not HAS_LGB):
                continue
            clf = (xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                      use_label_encoder=False, eval_metric="logloss",
                                      random_state=42, verbosity=0, n_jobs=-1)
                   if model_name=="XGBoost" else
                   lgb.LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                       random_state=42, verbose=-1, n_jobs=-1))
            probs = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:,1]
            fpr, tpr, _ = roc_curve(y, probs)
            auc = roc_auc_score(y, probs)
            ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})", color=color)
        ax.plot([0,1],[0,1],"--", color="gray", alpha=0.5)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title(tgt_name); ax.legend(fontsize=9)
    plt.tight_layout()
    save_fig("m1_roc_curves.png")

    if HAS_XGB and HAS_SHAP:
        print("   -> Computing SHAP values (XGBoost, Cancer Status, omics+clinical) ...")
        sub = df[full_feats + ["target_cancer"]].dropna()
        X   = pd.DataFrame(StandardScaler().fit_transform(sub[full_feats].fillna(sub[full_feats].median())),
                            columns=full_feats)
        y   = sub["target_cancer"].values
        clf = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                  use_label_encoder=False, eval_metric="logloss",
                                  random_state=42, verbosity=0, n_jobs=-1)
        clf.fit(X, y)
        explainer  = shap.TreeExplainer(clf)
        shap_vals  = explainer.shap_values(X)
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(shap_vals, X, plot_type="bar", show=False)
        plt.title("SHAP Feature Importance – XGBoost (Cancer Status)", fontweight="bold")
        save_fig("m1_shap_bar.png")
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(shap_vals, X, show=False)
        plt.title("SHAP Beeswarm – XGBoost (Cancer Status)", fontweight="bold")
        save_fig("m1_shap_beeswarm.png")

    return results


def run_logistic_regression(df, omics_feats, full_feats):
    print("\n" + "="*60)
    print("  MODULE 2: Regularised Logistic Regression (LASSO / ElasticNet)")
    print("="*60)

    targets = {
        "Cancer Status":      "target_cancer",
        "Vital Status":       "target_vital",
        "Treatment Response": "target_response",
        "Cancer Subtype":     "target_subtype",
    }
    Cs     = np.logspace(-3, 2, 30)
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("ROC Curves – Regularised Logistic Regression\n(LASSO vs ElasticNet · omics-only vs omics+clinical)",
                 fontsize=12, fontweight="bold")

    for ax, (tgt_name, tgt_col) in zip(axes.flat, targets.items()):
        for feat_name, feats, color in [("omics-only","omics","#4C72B0"),
                                          ("omics+clinical","full","#C44E52")]:
            feat_list = omics_feats if feat_name=="omics-only" else full_feats
            sub = df[feat_list + [tgt_col]].dropna()
            if sub[tgt_col].nunique() < 2 or len(sub) < 50:
                continue
            X = StandardScaler().fit_transform(sub[feat_list].fillna(sub[feat_list].median()).values)
            y = sub[tgt_col].values
            for penalty, ls in [("l1","-"), ("elasticnet","--")]:
                kwargs = dict(solver="saga", max_iter=5000, random_state=42)
                if penalty == "elasticnet":
                    kwargs["l1_ratio"] = 0.5
                best_C, best_auc = 1.0, 0.0
                for C in Cs:
                    clf   = LogisticRegression(penalty=penalty, C=C, **kwargs)
                    probs = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:,1]
                    auc   = roc_auc_score(y, probs)
                    if auc > best_auc:
                        best_auc, best_C = auc, C
                clf   = LogisticRegression(penalty=penalty, C=best_C, **kwargs)
                probs = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:,1]
                fpr, tpr, _ = roc_curve(y, probs)
                label = f"{feat_name}/{penalty} (AUC={best_auc:.3f})"
                ax.plot(fpr, tpr, label=label, color=color, linestyle=ls, linewidth=1.5)
                print(f"   LR [{penalty}] {feat_name} -> {tgt_name}: best C={best_C:.4f}, AUC={best_auc:.4f}")
                results.append({"Model": f"LR-{penalty}", "Features": feat_name,
                                 "Target": tgt_name, "AUC": round(best_auc,4)})
        ax.plot([0,1],[0,1],"--", color="gray", alpha=0.4)
        ax.set_title(tgt_name, fontsize=10)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.legend(fontsize=7)
    plt.tight_layout()
    save_fig("m2_lr_roc_curves.png")

    sub = df[full_feats + ["target_cancer"]].dropna()
    X   = StandardScaler().fit_transform(sub[full_feats].fillna(sub[full_feats].median()).values)
    y   = sub["target_cancer"].values
    clf = LogisticRegression(penalty="l1", solver="saga", C=1.0, max_iter=5000, random_state=42)
    clf.fit(X, y)
    coef_s = pd.Series(clf.coef_[0], index=full_feats).sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#C44E52" if c < 0 else "#55A868" for c in coef_s]
    ax.barh(coef_s.index, coef_s.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("LASSO Logistic Regression Coefficients\n(Cancer Status prediction, omics+clinical)",
                 fontweight="bold")
    ax.set_xlabel("Coefficient value  (green=Tumor Free, red=With Tumor)")
    plt.tight_layout()
    save_fig("m2_lasso_coefficients.png")

    return results


class MLP(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64),   nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64,  32),                         nn.ReLU(),
            nn.Linear(32,   1),  nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

def train_mlp(X_tr, y_tr, X_val, y_val, n_in, epochs=150, patience=20):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model   = MLP(n_in).to(device)
    opt     = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCELoss()

    Xtr = torch.FloatTensor(X_tr).to(device)
    ytr = torch.FloatTensor(y_tr).to(device)
    Xvl = torch.FloatTensor(X_val).to(device)
    yvl = torch.FloatTensor(y_val).to(device)

    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True)
    best_val, best_state, wait = 1e9, None, 0
    tr_losses, vl_losses = [], []

    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            l = loss_fn(model(xb), yb)
            l.backward(); opt.step()
            ep_loss += l.item()
        tr_losses.append(ep_loss / len(loader))

        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xvl), yvl).item()
        vl_losses.append(vl)

        if vl < best_val:
            best_val, best_state, wait = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = model(Xvl).cpu().numpy()
    return probs, tr_losses, vl_losses

def run_mlp(df, omics_feats, full_feats):
    print("\n" + "="*60)
    print("  MODULE 3: PyTorch MLP Deep Classifier")
    print("="*60)

    targets = {
        "Cancer Status":      "target_cancer",
        "Vital Status":       "target_vital",
        "Treatment Response": "target_response",
        "Cancer Subtype":     "target_subtype",
    }
    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    fig_roc, axes_roc = plt.subplots(2, 2, figsize=(14, 10))
    fig_roc.suptitle("ROC Curves – PyTorch MLP (5-fold CV)\nomics-only vs omics+clinical",
                     fontsize=12, fontweight="bold")

    loss_history = {}

    for ax, (tgt_name, tgt_col) in zip(axes_roc.flat, targets.items()):
        for feat_name, feats in [("omics-only", omics_feats), ("omics+clinical", full_feats)]:
            sub = df[feats + [tgt_col]].dropna()
            if sub[tgt_col].nunique() < 2 or len(sub) < 50:
                continue
            X   = StandardScaler().fit_transform(sub[feats].fillna(sub[feats].median()).values)
            y   = sub[tgt_col].values.astype(np.float32)
            all_probs = np.zeros(len(y))

            fold_tr_losses, fold_vl_losses = [], []
            for fold, (tr_idx, vl_idx) in enumerate(cv.split(X, y)):
                probs, tr_l, vl_l = train_mlp(X[tr_idx], y[tr_idx],
                                               X[vl_idx], y[vl_idx],
                                               n_in=X.shape[1])
                all_probs[vl_idx] = probs
                fold_tr_losses.append(tr_l); fold_vl_losses.append(vl_l)

            auc = roc_auc_score(y, all_probs)
            fpr, tpr, _ = roc_curve(y, all_probs)
            color = "#4C72B0" if feat_name == "omics-only" else "#C44E52"
            ax.plot(fpr, tpr, color=color, linewidth=1.8,
                    label=f"{feat_name} (AUC={auc:.3f})")
            print(f"   MLP {feat_name} -> {tgt_name}: AUC = {auc:.4f}")
            results.append({"Model":"MLP","Features":feat_name,
                             "Target":tgt_name,"AUC":round(auc,4)})

            key = f"{tgt_name}|{feat_name}"
            loss_history[key] = (fold_tr_losses, fold_vl_losses)

        ax.plot([0,1],[0,1],"--",color="gray",alpha=0.4)
        ax.set_title(tgt_name, fontsize=10)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend(fontsize=8)
    plt.tight_layout()
    save_fig("m3_mlp_roc_curves.png")

    # Loss curves for Cancer Status (omics+clinical)
    key = "Cancer Status|omics+clinical"
    if key in loss_history:
        tr_ls, vl_ls = loss_history[key]
        fig, ax = plt.subplots(figsize=(10, 4))
        for fold_i, (tl, vl) in enumerate(zip(tr_ls, vl_ls)):
            ax.plot(tl, alpha=0.4, linestyle="--", label=f"Fold {fold_i+1} train")
            ax.plot(vl, alpha=0.8,                  label=f"Fold {fold_i+1} val")
        ax.set_xlabel("Epoch"); ax.set_ylabel("BCE Loss")
        ax.set_title("MLP Training vs Validation Loss – Cancer Status (omics+clinical)",
                     fontweight="bold")
        ax.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        save_fig("m3_mlp_loss_curves.png")

    return results


class VAE(nn.Module):
    def __init__(self, n_in, latent=16):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(),
                                  nn.Linear(64, 32),   nn.ReLU())
        self.mu     = nn.Linear(32, latent)
        self.logvar = nn.Linear(32, latent)
        self.dec = nn.Sequential(nn.Linear(latent, 32), nn.ReLU(),
                                  nn.Linear(32, 64),     nn.ReLU(),
                                  nn.Linear(64, n_in))

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        h   = self.enc(x)
        mu, lv = self.mu(h), self.logvar(h)
        z   = self.reparameterise(mu, lv)
        return self.dec(z), mu, lv

def vae_loss(recon, x, mu, logvar):
    recon_loss = nn.MSELoss(reduction="sum")(recon, x)
    kl_loss    = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + 0.01 * kl_loss

def run_vae(df, omics_feats):
    print("\n" + "="*60)
    print("  MODULE 4: Variational Autoencoder (Unsupervised)")
    print("="*60)

    sub     = df[["sample_id"] + omics_feats + ["target_cancer","target_vital","target_subtype",
                                 "gender","cancer_status_raw","subtype_raw"]].dropna(subset=omics_feats).copy()
    scaler  = StandardScaler()
    X       = scaler.fit_transform(sub[omics_feats].fillna(sub[omics_feats].median()).values)
    Xt      = torch.FloatTensor(X)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_in    = X.shape[1]
    model   = VAE(n_in, latent=8).to(device)
    opt     = optim.Adam(model.parameters(), lr=1e-3)
    loader  = DataLoader(TensorDataset(Xt), batch_size=64, shuffle=True)

    losses  = []
    print("   Training VAE (200 epochs) ...")
    for ep in range(200):
        model.train(); ep_loss = 0
        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad()
            recon, mu, lv = model(xb)
            l = vae_loss(recon, xb, mu, lv)
            l.backward(); opt.step()
            ep_loss += l.item()
        losses.append(ep_loss / len(loader))
    print("   VAE training complete")

    model.eval()
    with torch.no_grad():
        recon_all, mu_all, lv_all = model(Xt.to(device))
    recon_err = ((Xt.to(device) - recon_all) ** 2).mean(dim=1).cpu().numpy()
    mu_np     = mu_all.cpu().numpy()   # shape: (N, 8)

    # Project to 2D via PCA for visualisation
    from sklearn.decomposition import PCA
    pca2  = PCA(n_components=2, random_state=42)
    z2    = pca2.fit_transform(mu_np)
    sub   = sub.reset_index(drop=True)
    sub["z1"]        = z2[:,0]
    sub["z2"]        = z2[:,1]
    sub["recon_err"] = recon_err

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, color="#4C72B0"); ax.set_xlabel("Epoch"); ax.set_ylabel("VAE Loss")
    ax.set_title("VAE Training Loss (Reconstruction + KL)", fontweight="bold")
    plt.tight_layout(); save_fig("m4_vae_loss.png")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("VAE Latent Space (PCA 2D projection)", fontsize=13, fontweight="bold")
    for ax, (col, title) in zip(axes, [
            ("cancer_status_raw", "Cancer Status"),
            ("gender",            "Gender"),
            ("subtype_raw",       "Cancer Subtype")]):
        cats = sub[col].dropna().unique()
        pal  = dict(zip(cats, sns.color_palette("tab10", len(cats))))
        for cat in cats:
            m = sub[col] == cat
            ax.scatter(sub.loc[m,"z1"], sub.loc[m,"z2"],
                       c=[pal[cat]], label=str(cat), s=18, alpha=0.7, edgecolors="none")
        ax.set_title(title, fontsize=10); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.legend(fontsize=7, markerscale=1.5)
    plt.tight_layout(); save_fig("m4_vae_latent_space.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(data=sub[sub["cancer_status_raw"].isin(["TUMOR FREE","WITH TUMOR"])],
                x="cancer_status_raw", y="recon_err",
                palette=["#55A868","#C44E52"], ax=axes[0], showfliers=False)
    axes[0].set_title("Reconstruction Error by Cancer Status", fontweight="bold")
    axes[0].set_ylabel("Mean Squared Reconstruction Error")
    axes[0].set_xlabel("")

    sub_g = sub[sub["gender"].str.upper().str.strip().isin(["MALE","FEMALE"])]
    sub_g = sub_g.copy()
    sub_g["gender"] = sub_g["gender"].str.upper().str.strip()
    sns.boxplot(data=sub_g, x="gender", y="recon_err",
                palette=["#4C72B0","#DD8452"], ax=axes[1], showfliers=False)
    axes[1].set_title("Reconstruction Error by Gender", fontweight="bold")
    axes[1].set_ylabel("Mean Squared Reconstruction Error")
    plt.tight_layout(); save_fig("m4_vae_anomaly_detection.png")

    avail_cols = [c for c in ["sample_id","recon_err","cancer_status_raw","subtype_raw"] if c in sub.columns]
    top_anom = sub.nlargest(10, "recon_err")[avail_cols]
    print("   Top 10 highest-reconstruction-error samples (potential anomalies):")
    print(top_anom.to_string(index=False))

    return sub[["sample_id","z1","z2","recon_err"]]


def run_survival(df, omics_feats, full_feats):
    if not HAS_LIFELINES:
        print("[Module 5] Skipped – lifelines not available")
        return

    print("\n" + "="*60)
    print("  MODULE 5: Survival Analysis (Cox PH + Kaplan-Meier)")
    print("="*60)

    surv_cols = ["OS_days","OS_event"]
    sub = df[full_feats + surv_cols].dropna(subset=surv_cols).copy()
    sub["OS_days"]  = pd.to_numeric(sub["OS_days"],  errors="coerce")
    sub["OS_event"] = pd.to_numeric(sub["OS_event"], errors="coerce")
    sub = sub.dropna(subset=surv_cols)
    sub = sub[sub["OS_days"] > 0]
    print(f"   Survival-eligible samples: {len(sub)}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Kaplan-Meier Survival by MOFA Factor Quartile",
                 fontsize=13, fontweight="bold")
    kmf = KaplanMeierFitter()
    for ax, factor in zip(axes, ["Factor2","Factor3"]):
        median_val = sub[factor].median()
        grp_high   = sub[sub[factor] >= median_val]
        grp_low    = sub[sub[factor] <  median_val]
        kmf.fit(grp_high["OS_days"], grp_high["OS_event"], label=f"{factor} High (≥median)")
        kmf.plot_survival_function(ax=ax, color="#C44E52", ci_show=True)
        kmf.fit(grp_low["OS_days"],  grp_low["OS_event"],  label=f"{factor} Low (<median)")
        kmf.plot_survival_function(ax=ax, color="#4C72B0", ci_show=True)
        result = logrank_test(grp_high["OS_days"], grp_low["OS_days"],
                               event_observed_A=grp_high["OS_event"],
                               event_observed_B=grp_low["OS_event"])
        ax.set_title(f"KM by {factor}  (log-rank p={result.p_value:.4f})", fontsize=10)
        ax.set_xlabel("Days"); ax.set_ylabel("Survival Probability")
    plt.tight_layout(); save_fig("m5_kaplan_meier.png")

    cox_feats = ["Factor1","Factor2","Factor3","Factor4","Factor5",
                 "cnv_total_burden","cnv_amp_frac","cnv_del_frac"]
    cox_feats = [f for f in cox_feats if f in sub.columns]
    cox_df    = sub[cox_feats + ["OS_days","OS_event"]].dropna()
    cox_df    = cox_df.copy()
    # Standardise
    for c in cox_feats:
        cox_df[c] = (cox_df[c] - cox_df[c].mean()) / (cox_df[c].std() + 1e-8)

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(cox_df, duration_col="OS_days", event_col="OS_event")
    print("\n   Cox PH summary:")
    cph.print_summary(decimals=4)

    fig, ax = plt.subplots(figsize=(9, 5))
    cph.plot(ax=ax)
    ax.set_title("Cox PH Hazard Ratios (penalised, 95% CI)\nMOFA Factors + CNV Metrics",
                 fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout(); save_fig("m5_cox_hazard_ratios.png")

    # Concordance index
    c_idx = cph.concordance_index_
    print(f"\n   Cox PH Concordance Index (C-index): {c_idx:.4f}")

    return c_idx


def model_comparison_table(all_results):
    print("\n" + "="*60)
    print("  MODEL COMPARISON SUMMARY")
    print("="*60)
    tbl = pd.DataFrame(all_results)
    if tbl.empty:
        return
    pivot = tbl.pivot_table(index=["Model","Features"], columns="Target",
                             values="AUC", aggfunc="mean")
    print(pivot.to_string())
    path = os.path.join(OUT_ROOT, "model_comparison.csv")
    tbl.to_csv(path, index=False)
    print(f"\n   Saved -> {path}")

    # Bar chart comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    targets = tbl["Target"].unique()
    models  = (tbl["Model"] + " / " + tbl["Features"]).unique()
    x       = np.arange(len(targets))
    width   = 0.8 / len(models)
    pal     = sns.color_palette("tab10", len(models))
    for i, model_key in enumerate(models):
        vals = []
        for tgt in targets:
            row = tbl[(tbl["Model"] + " / " + tbl["Features"] == model_key) &
                      (tbl["Target"] == tgt)]["AUC"]
            vals.append(row.values[0] if len(row)>0 else 0)
        ax.bar(x + i*width, vals, width, label=model_key, color=pal[i], alpha=0.85)
    ax.set_xticks(x + width*(len(models)-1)/2)
    ax.set_xticklabels(targets, rotation=15, ha="right")
    ax.set_ylabel("AUC-ROC"); ax.set_ylim(0.4, 1.0)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Random (0.5)")
    ax.set_title("Model Comparison: AUC-ROC Across All Targets & Feature Sets",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout(); save_fig("model_comparison_auc.png")


def main():
    print("\n" + "="*60)
    print("   TCGA LUNG – ML / Deep Learning Pipeline")
    print("="*60 + "\n")

    df, omics_feats, full_feats, factor_feats, cnv_feats = load_all()

    all_results = []

    gb_results = run_gradient_boosting(df, omics_feats, full_feats)
    if isinstance(gb_results, list): all_results.extend(gb_results)

    lr_results = run_logistic_regression(df, omics_feats, full_feats)
    if isinstance(lr_results, list): all_results.extend(lr_results)

    mlp_results = run_mlp(df, omics_feats, full_feats)
    if isinstance(mlp_results, list): all_results.extend(mlp_results)

    run_vae(df, omics_feats)

    run_survival(df, omics_feats, full_feats)

    model_comparison_table(all_results)

    print("\n" + "="*60)
    print(f"   All ML outputs -> {OUT_ROOT}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
