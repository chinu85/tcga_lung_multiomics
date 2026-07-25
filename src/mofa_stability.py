import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mofapy2.run.entry_point import entry_point
import mofax as mfx

PATH_CNV = "c:/Users/shmso/UCD/Spring/internship/dataset/Gistic2_CopyNumber_Gistic2_all_data_by_genes"
PATH_EXP = "c:/Users/shmso/UCD/Spring/internship/dataset/HiSeqV2"
DIR_OUT  = "c:/Users/shmso/UCD/Spring/internship/dataset"

TOP_FEATURES = 5000
NUM_FACTORS  = 15
TRAIN_ITER   = 150
SEEDS        = [42, 100, 2026, 999, 123]


def load_and_preprocess():
    df_cnv = pd.read_csv(PATH_CNV, sep='\t', index_col=0)
    df_exp = pd.read_csv(PATH_EXP, sep='\t', index_col=0)

    common_samples = sorted(list(set(df_cnv.columns).intersection(set(df_exp.columns))))
    print(f" -> Number of overlapping samples: {len(common_samples)}")

    df_cnv_sub = df_cnv[common_samples]
    df_exp_sub = df_exp[common_samples]

    var_cnv = df_cnv_sub.var(axis=1)
    var_exp = df_exp_sub.var(axis=1)

    top_cnv_genes = var_cnv.nlargest(TOP_FEATURES).index
    top_exp_genes = var_exp.nlargest(TOP_FEATURES).index

    df_cnv_filtered = df_cnv_sub.loc[top_cnv_genes]
    df_exp_filtered = df_exp_sub.loc[top_exp_genes]

    cnv_feature_names = [f"{gene}_CNV" for gene in top_cnv_genes]
    exp_feature_names = [f"{gene}_Exp" for gene in top_exp_genes]

    cnv_matrix = df_cnv_filtered.T.values.astype(np.float32)
    exp_matrix = df_exp_filtered.T.values.astype(np.float32)

    data = [[cnv_matrix], [exp_matrix]]
    return data, common_samples, cnv_feature_names, exp_feature_names


def train_mofa(data, samples, cnv_features, exp_features, seed):
    print(f"Training MOFA+ model with seed {seed}...")
    ent = entry_point()
    data_fresh = [[data[0][0].copy()], [data[1][0].copy()]]
    ent.set_data_matrix(
        data_fresh,
        likelihoods=["gaussian", "gaussian"],
        views_names=["CNV", "Expression"],
        groups_names=["single_group"],
        samples_names=[list(samples)],
        features_names=[list(cnv_features), list(exp_features)]
    )
    ent.set_model_options(factors=NUM_FACTORS)
    ent.set_train_options(iter=TRAIN_ITER, verbose=False, dropR2=0.001, seed=seed)

    ent.build()
    ent.run()

    temp_file = os.path.join(DIR_OUT, f"mofa_temp_seed_{seed}.hdf5")
    ent.save(outfile=temp_file)

    model = mfx.mofa_model(temp_file)
    factors_df = model.get_factors(df=True)
    model.close()

    try:
        os.remove(temp_file)
    except OSError:
        pass

    return factors_df


def tucker_congruence(x, y):
    num = np.sum(x * y)
    den = np.sqrt(np.sum(x**2) * np.sum(y**2))
    if den == 0:
        return 0.0
    return num / den


def main():
    data, samples, cnv_features, exp_features = load_and_preprocess()

    runs_factors = {}
    for s in SEEDS:
        runs_factors[s] = train_mofa(data, samples, cnv_features, exp_features, s)

    ref_df      = runs_factors[SEEDS[0]]
    ref_factors = ref_df.columns.tolist()

    stability_data = {f: {"congruence": [], "correlation": []} for f in ref_factors}

    for s in SEEDS[1:]:
        run_df      = runs_factors[s]
        run_factors = run_df.columns.tolist()

        for f_ref in ref_factors:
            x = ref_df[f_ref].values
            best_cong = 0.0
            best_corr = 0.0

            for f_run in run_factors:
                y    = run_df[f_run].values
                corr = np.abs(np.corrcoef(x, y)[0, 1])
                cong = np.abs(tucker_congruence(x, y))

                if cong > best_cong:
                    best_cong = cong
                if corr > best_corr:
                    best_corr = corr

            stability_data[f_ref]["congruence"].append(best_cong)
            stability_data[f_ref]["correlation"].append(best_corr)

    summary_list = []
    for f in ref_factors:
        congs = stability_data[f]["congruence"]
        corrs = stability_data[f]["correlation"]
        summary_list.append({
            "Factor":            f,
            "Mean_Congruence":   np.mean(congs),
            "SD_Congruence":     np.std(congs),
            "Mean_Correlation":  np.mean(corrs),
            "SD_Correlation":    np.std(corrs)
        })

    summary_df = pd.DataFrame(summary_list)

    summary_csv_path = os.path.join(DIR_OUT, "mofa_stability_metrics.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nStability metrics saved to: {summary_csv_path}")
    print(summary_df.to_string(index=False))

    print("\nGenerating stability plot...")
    plt.figure(figsize=(10, 6))

    plot_df = pd.melt(summary_df, id_vars="Factor",
                      value_vars=["Mean_Congruence", "Mean_Correlation"],
                      var_name="Metric", value_name="Stability")

    plot_df["Metric"] = plot_df["Metric"].map({
        "Mean_Congruence":  "Tucker Congruence",
        "Mean_Correlation": "Pearson Correlation"
    })

    errors = []
    for _, row in plot_df.iterrows():
        f = row["Factor"]
        if row["Metric"] == "Tucker Congruence":
            errors.append(summary_df.loc[summary_df["Factor"] == f, "SD_Congruence"].values[0])
        else:
            errors.append(summary_df.loc[summary_df["Factor"] == f, "SD_Correlation"].values[0])
    plot_df["SD"] = errors

    sns.set_theme(style="whitegrid")
    ax = sns.barplot(data=plot_df, x="Factor", y="Stability", hue="Metric",
                     palette="muted", alpha=0.9)

    x_coords = sorted([p.get_x() + p.get_width() / 2.0 for p in ax.patches])

    for i, (_, row) in enumerate(plot_df.iterrows()):
        plt.errorbar(x_coords[i], row["Stability"], yerr=row["SD"],
                     fmt='none', c='black', capsize=3)

    plt.ylim(0, 1.05)
    plt.title("Factor Stability")
    plt.ylabel("Stability (Mean ± SD)")
    plt.xlabel("Factor")
    plt.legend(loc="lower left")
    plt.tight_layout()

    plot_path = os.path.join(DIR_OUT, "mofa_stability_plot.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Stability plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
