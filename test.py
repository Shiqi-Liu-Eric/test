import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

folds = result_df["fold"].unique()
alpha_cols = [col for col in result_df.columns if col.startswith("alpha_")]
all_results = []
importance_by_fold = {}
importance_by_tminus = {}

for fold in folds:
    importance_by_fold[fold] = []
    print(f"\n=== fold {fold} ===")

    for t_minus in range(1, 6):
        train_df = result_df[(result_df["fold"] != fold) & (result_df["T_minus"] == t_minus)].copy()
        test_df = result_df[(result_df["fold"] == fold) & (result_df["T_minus"] == t_minus)].copy()

        if len(train_df) == 0 or len(test_df) == 0:
            continue

        # 去除 inf/-inf 和 NaN
        train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        train_df.dropna(subset=alpha_cols + ["actual"], inplace=True)
        test_df.dropna(subset=alpha_cols + ["actual"], inplace=True)

        # X, y 构造
        X_train = sm.add_constant(train_df[alpha_cols])
        y_train = train_df["actual"]
        X_test = sm.add_constant(test_df[alpha_cols])
        y_test = test_df["actual"]

        # 拟合 OLS 模型
        model = sm.OLS(y_train, X_train).fit()
        test_df["predicted"] = model.predict(X_test)

        # 保存结果
        all_results.append(test_df.copy())

        # 保存 feature importance
        importance_by_fold[fold].append(model.params)
        importance_by_tminus[(fold, t_minus)] = model.params

# 合并所有预测结果
final_df = pd.concat(all_results, axis=0)

# ========= 可视化部分 =========

# 可视化 1：按 fold 汇总的平均 feature importance
fig, axes = plt.subplots(1, len(folds), figsize=(5 * len(folds), 5))
for i, fold in enumerate(folds):
    if fold in importance_by_fold and len(importance_by_fold[fold]) > 0:
        avg_coef = pd.DataFrame(importance_by_fold[fold]).mean()
        avg_coef.plot(kind="bar", ax=axes[i], title=f"Fold {fold}")
plt.suptitle("Feature Importance by Fold")
plt.tight_layout()
plt.show()

# 可视化 2：按 T_minus 汇总的平均 feature importance
fig, axes = plt.subplots(1, 5, figsize=(25, 5))
for t_minus in range(1, 6):
    coefs = [importance_by_tminus[(f, t_minus)] for f in folds if (f, t_minus) in importance_by_tminus]
    if not coefs:
        continue
    avg_coef = pd.DataFrame(coefs).mean()
    avg_coef.plot(kind='bar', ax=axes[t_minus - 1], title=f"T_minus={t_minus}")
plt.suptitle("Feature Importance by T_minus")
plt.tight_layout()
plt.show()
