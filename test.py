use_stat_alpha = True  # ← 新控制变量

...

if use_stat_alpha:
    preds = []
    y_true_list = []

    dates = sorted(baskets[test_key]["dates"])
    feature_days = dates[-10:-5]   # T-9 ~ T-5
    target_days = dates[-5:]       # T-4 ~ T

    tickers_valid = [t for t in tickers_list if not dfs["EV_it"].loc[t, target_days].isnull().any()]

    for day in target_days:
        alpha_ret1 = []
        alpha_mean_wd = []
        alpha_std_wd = []
        alpha_other_ret = []

        for ticker in tickers_valid:
            # 1. 当前day的ret1
            val1 = dfs["ret1"].loc[ticker, day] if not pd.isna(dfs["ret1"].loc[ticker, day]) else 0
            alpha_ret1.append(val1)

            # 2. 对应feature_days的delta_weight_DMSS
            wds = [dfs["delta_weight_DMSS"].loc[ticker, d] if not pd.isna(dfs["delta_weight_DMSS"].loc[ticker, d]) else 0 for d in feature_days]
            mean_wd = np.mean(wds[-3:])
            std_wd = np.std(wds[-3:])
            alpha_mean_wd.append(mean_wd)
            alpha_std_wd.append(std_wd)

            # 3. 其他baskets这一天的ret1
            other_ret = []
            for k in basket_keys:
                if k == test_key or day not in baskets[k]["dates"]:
                    continue
                try:
                    val = dfs["ret1"].loc[ticker, day]
                    if not pd.isna(val): other_ret.append(val)
                except: pass
            mean_other = np.mean(other_ret) if other_ret else 0
            alpha_other_ret.append(mean_other)

        # 对每个 alpha 特征列做 z-score
        def zscore(col):
            mean, std = np.mean(col), np.std(col)
            return [(x - mean) / (std + 1e-6) for x in col]

        z1 = zscore(alpha_ret1)
        z2 = zscore(alpha_mean_wd)
        z3 = zscore(alpha_std_wd)
        z4 = zscore(alpha_other_ret)

        # 合成最终 alpha 值（加权或平均）
        alpha_final = [(a + b + c + d) / 4 for a, b, c, d in zip(z1, z2, z3, z4)]

        # 当前 day 的预测值：用 alpha_final 替代预测 EV_it
        for idx, ticker in enumerate(tickers_valid):
            if len(preds) <= idx:
                preds.append([])
                y_true_list.append(dfs["EV_it"].loc[ticker, target_days].values.tolist())
            preds[idx].append(alpha_final[idx])

    # 注意：preds 是每只股票为一个 list，长度为 5（对应 T-4~T）
    # y_true_list 是同样结构，用于后续评估
