import os
import time
import numpy as np
import pandas as pd
from mofapy2.run.entry_point import entry_point
import mofax as mfx

# Configuration
PATH_CNV = "c:/Users/shmso/UCD/Spring/internship/dataset/Gistic2_CopyNumber_Gistic2_all_data_by_genes"
PATH_EXP = "c:/Users/shmso/UCD/Spring/internship/dataset/HiSeqV2"

DIR_OUT = "c:/Users/shmso/UCD/Spring/internship/dataset"
FILE_MODEL = os.path.join(DIR_OUT, "mofa_model.hdf5")
FILE_FACTORS = os.path.join(DIR_OUT, "mofa_factors.csv")
FILE_WEIGHTS = os.path.join(DIR_OUT, "mofa_weights.csv")

TOP_FEATURES = 5000   # Select top N most variable genes for each view
NUM_FACTORS = 15       # Latent dimensions
TRAIN_ITER = 150       # Number of iterations for convergence

def run_integration():
    print("=========================================")
    print("      Multi-Omics Data Integration       ")
    print("=========================================")
    
    # 1. Load data
    print("\n[1/6] Loading datasets...")
    t0 = time.time()
    
    if not os.path.exists(PATH_CNV):
        raise FileNotFoundError(f"CNV dataset not found at: {PATH_CNV}")
    if not os.path.exists(PATH_EXP):
        raise FileNotFoundError(f"Expression dataset not found at: {PATH_EXP}")
        
    df_cnv = pd.read_csv(PATH_CNV, sep='\t', index_col=0)
    print(f" -> CNV Loaded: {df_cnv.shape} in {time.time() - t0:.2f} seconds")
    
    t1 = time.time()
    df_exp = pd.read_csv(PATH_EXP, sep='\t', index_col=0)
    print(f" -> Expression Loaded: {df_exp.shape} in {time.time() - t1:.2f} seconds")
    
    # 2. Align samples
    print("\n[2/6] Aligning sample identifiers...")
    common_samples = sorted(list(set(df_cnv.columns).intersection(set(df_exp.columns))))
    print(f" -> Number of overlapping samples: {len(common_samples)}")
    
    df_cnv_sub = df_cnv[common_samples]
    df_exp_sub = df_exp[common_samples]
    
    # 3. Select highly variable features
    print(f"\n[3/6] Filtering for top {TOP_FEATURES} highly variable features per view...")
    # Compute variance for each gene across the aligned samples
    var_cnv = df_cnv_sub.var(axis=1)
    var_exp = df_exp_sub.var(axis=1)
    
    top_cnv_genes = var_cnv.nlargest(TOP_FEATURES).index
    top_exp_genes = var_exp.nlargest(TOP_FEATURES).index
    
    df_cnv_filtered = df_cnv_sub.loc[top_cnv_genes]
    df_exp_filtered = df_exp_sub.loc[top_exp_genes]
    
    print(f" -> CNV view dimensions after selection: {df_cnv_filtered.shape}")
    print(f" -> Expression view dimensions after selection: {df_exp_filtered.shape}")
    
    # Prepend suffixes to avoid duplicated feature names across views
    cnv_feature_names = [f"{gene}_CNV" for gene in top_cnv_genes]
    exp_feature_names = [f"{gene}_Exp" for gene in top_exp_genes]
    
    # 4. Format matrices for MOFA
    # Samples must be rows, features must be columns
    cnv_matrix = df_cnv_filtered.T.values.astype(np.float32)
    exp_matrix = df_exp_filtered.T.values.astype(np.float32)
    
    # Nested list format: data[view][group]
    data = [[cnv_matrix], [exp_matrix]]
    
    # 5. Initialize and run MOFA training
    print("\n[4/6] Initializing MOFA+ model...")
    ent = entry_point()
    ent.set_data_matrix(
        data,
        likelihoods=["gaussian", "gaussian"],
        views_names=["CNV", "Expression"],
        groups_names=["single_group"],
        samples_names=[common_samples],
        features_names=[cnv_feature_names, exp_feature_names]
    )
    
    # Set model structure options
    ent.set_model_options(
        factors=NUM_FACTORS
    )
    
    # Set training parameters
    ent.set_train_options(
        iter=TRAIN_ITER,
        verbose=True,
        dropR2=0.001
    )
    
    print(f"\n[5/6] Training MOFA+ model (factors={NUM_FACTORS}, iterations={TRAIN_ITER})...")
    t_train = time.time()
    ent.build()
    ent.run()
    print(f" -> MOFA+ training completed in {time.time() - t_train:.2f} seconds.")
    
    # Save the model
    print(f" -> Saving model to: {FILE_MODEL}")
    ent.save(outfile=FILE_MODEL)
    
    # 6. Extract results via mofax
    print("\n[6/6] Extracting merged factors and feature weights...")
    model = mfx.mofa_model(FILE_MODEL)
    
    # Retrieve factors (combined representation) and save
    factors_df = model.get_factors(df=True)
    factors_df.to_csv(FILE_FACTORS)
    print(f" -> Merged factors saved to: {FILE_FACTORS} (Shape: {factors_df.shape})")
    
    # Retrieve weights (feature loadings) and save
    weights_df = model.get_weights(df=True)
    weights_df.to_csv(FILE_WEIGHTS)
    print(f" -> Gene weights saved to: {FILE_WEIGHTS} (Shape: {weights_df.shape})")
    
    model.close()
    print("\n=========================================")
    print("   Integration completed successfully!   ")
    print("=========================================")

if __name__ == "__main__":
    run_integration()
