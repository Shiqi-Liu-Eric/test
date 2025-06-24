import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

folds = result_df["fold"].unique()
tminus_vals = result_df["T_minus"].unique()
alpha_cols = [col for col in result_df.columns if col.startswith(bundle)]

r2_dict = {}
r2adj_dict = {}
beta_dict = {}
pval_dict = {}

# Main training loop
for fold in folds:
    for t in tminus_vals:
        # Train: all folds except `fold` but same T_minus
        train = result_df[(result_df['fold'] != fold) & (result_df['T_minus'] == t)].copy()
        test = result_df[(result_df['fold'] == fold) & (result_df['T_minus'] == t)].copy()
        
        X_train = sm.add_constant(train[alpha_cols])
        y_train = train["actual"]
        mask = X_train.replace([np.inf, -np.inf, np.nan], np.nan).notnull().all(axis=1) & y_train.notnull()
        X_train = X_train[mask]
        y_train = y_train[mask]

        model = sm.OLS(y_train, X_train).fit()
        r2 = model.rsquared
        r2_adj = model.rsquared_adj
        r2_dict[(fold, t)] = r2
        r2adj_dict[(fold, t)] = r2_adj

        beta_dict[(fold, t)] = model.params.drop("const", errors='ignore')
        pval_dict[(fold, t)] = model.pvalues.drop("const", errors='ignore')

# === Step 1: R2 & AdjR2 table ===
r2_df = pd.DataFrame([
    {"fold": f, "T_minus": t, "R2": r2_dict[(f, t)]}
    for (f, t) in r2_dict
])
r2adj_df = pd.DataFrame([
    {"fold": f, "T_minus": t, "R2_adj": r2adj_dict[(f, t)]}
    for (f, t) in r2adj_dict
])

r2_mean = r2_df["R2"].mean()
r2adj_mean = r2adj_df["R2_adj"].mean()

print("=== R² Table ===")
display(r2_df.pivot(index="fold", columns="T_minus", values="R2").assign(mean=lambda df: df.mean(axis=1)))
print("Mean R²:", r2_mean)

print("\n=== Adjusted R² Table ===")
display(r2adj_df.pivot(index="fold", columns="T_minus", values="R2_adj").assign(mean=lambda df: df.mean(axis=1)))
print("Mean Adj R²:", r2adj_mean)

# === Step 2: Plot feature importance across folds ===
beta_df = pd.DataFrame(beta_dict).T  # index: (fold, tminus), columns: features
pval_df = pd.DataFrame(pval_dict).T

# Fold-wise feature importance plot
fold_avg = beta_df.groupby(level=0).mean()
fig, axes = plt.subplots(len(folds), 1, figsize=(10, 4 * len(folds)))
for i, fold in enumerate(folds):
    top_feats = fold_avg.loc[fold].abs().sort_values(ascending=False).head(15)
    axes[i].barh(top_feats.index[::-1], top_feats.values[::-1])
    axes[i].set_title(f"Fold {fold} - Top 15 Features (|beta|)")
plt.tight_layout()
plt.show()

# T_minus-wise feature importance plot
tminus_avg = beta_df.groupby(level=1).mean()
fig, axes = plt.subplots(len(tminus_vals), 1, figsize=(10, 4 * len(tminus_vals)))
for i, t in enumerate(tminus_vals):
    top_feats = tminus_avg.loc[t].abs().sort_values(ascending=False).head(15)
    axes[i].barh(top_feats.index[::-1], top_feats.values[::-1])
    axes[i].set_title(f"T_minus {t} - Top 15 Features (|beta|)")
plt.tight_layout()
plt.show()

# === Step 3: Which features are significant on average ===
pval_stack = pval_df.stack()
pval_summary = pval_stack.groupby(level=1).mean().sort_values()
significant_features = pval_summary[pval_summary < 0.05]

print("\n=== Significant Features on Average (p < 0.05) ===")
print(significant_features)
