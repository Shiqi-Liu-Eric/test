import numpy as np
import pandas as pd

# ---- 前置：确保列名正确且无严重缺失 ---------------------------
cols_needed = ["fold", "T_minus", "EV_it",
               "current_weight", "target_weight"]
missing = [c for c in cols_needed if c not in result_df.columns]
if missing:
    raise ValueError(f"缺少列: {missing}")

# 把关键列中的 NaN 全部去掉，以免后续 rank / qcut 出错
clean_df = result_df.dropna(subset=cols_needed).copy()

# ================================================================
# 1)  每个 fold 的  Σ(EV_it * current_weight)
# ----------------------------------------------------------------
fold_sum_ev_curw = (
    clean_df
    .assign(prod=lambda x: x["EV_it"] * x["current_weight"])
    .groupby("fold")["prod"]
    .sum()
)  # Series, index=fold

# ================================================================
# 2)  每个 fold 在 T_minus == 4 时：
#     以 EV_it 分成 5 个分位组，按你给的公式求值，再对所有 fold 取平均
# ----------------------------------------------------------------
def _step2_single_fold(df_fold: pd.DataFrame) -> np.ndarray:
    """对单个 fold 返回长度为 5 的 ndarray；若该 fold 没数据则返回 NaN"""
    df = df_fold.query("T_minus == 4").copy()
    if df.empty:
        return np.full(5, np.nan)

    # 解决 pd.qcut 边界重复问题：先用 rank 保证唯一，再分位
    df["EV_rank"] = df["EV_it"].rank(method="first")
    df["quant"] = pd.qcut(df["EV_rank"], 5, labels=False)

    out = np.empty(5)
    for q in range(5):
        sub = df[df["quant"] == q]
        if sub.empty:
            out[q] = np.nan
            continue

        sign_match = np.sign(sub["EV_it"]) == np.sign(
            sub["target_weight"] - sub["current_weight"]
        )
        aligned = sub[sign_match]

        numer = (aligned["EV_it"] * (aligned["target_weight"] - aligned["current_weight"])).sum()
        denom = (sub["target_weight"] - sub["current_weight"]).abs().sum()
        out[q] = np.nan if denom == 0 else numer / denom

    return out

step2_by_fold = (
    clean_df.groupby("fold", group_keys=False)
            .apply(_step2_single_fold)      # 得到每个 fold 的 ndarray
)

# stack 到 2‑D，再按列取 NaN‑mean
step2_matrix = np.vstack(step2_by_fold.tolist())   # shape=(n_fold, 5)
step2_average = np.nanmean(step2_matrix, axis=0)   # 长度 5 的 ndarray


# ================================================================
# 3)  与 (2) 类似，只是换成 Σ(EV_it*current_weight)/Σ(current_weight)
# ----------------------------------------------------------------
def _step3_single_fold(df_fold: pd.DataFrame) -> np.ndarray:
    df = df_fold.query("T_minus == 4").copy()
    if df.empty:
        return np.full(5, np.nan)

    df["EV_rank"] = df["EV_it"].rank(method="first")
    df["quant"] = pd.qcut(df["EV_rank"], 5, labels=False)

    out = np.empty(5)
    for q in range(5):
        sub = df[df["quant"] == q]
        if sub.empty:
            out[q] = np.nan
            continue

        numer = (sub["EV_it"] * sub["current_weight"]).sum()
        denom = sub["current_weight"].sum()
        out[q] = np.nan if denom == 0 else numer / denom

    return out

step3_by_fold = (
    clean_df.groupby("fold", group_keys=False)
            .apply(_step3_single_fold)
)
step3_matrix = np.vstack(step3_by_fold.tolist())
step3_average = np.nanmean(step3_matrix, axis=0)   # 长度 5


# ------------------- 打印 / 返回结果 -----------------------------
print("1) 每个 fold 的 Σ(EV_it*current_weight):")
print(fold_sum_ev_curw)
print("\n2) 五个分位组的平均值 (target‑current 对齐):")
print(step2_average)
print("\n3) 五个分位组的平均值 (现有持仓贡献):")
print(step3_average)
