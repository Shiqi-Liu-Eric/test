for idx_t, tminus in enumerate(tminuses):

    df = result_df[result_df["T_minus"] == tminus].copy()
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # optional shuffle

    # 将df按fold数量划分为k份
    k = 15
    fold_size = len(df) // k
    all_stats = []

    for fold_idx in range(k):
        test_idx = range(fold_idx * fold_size, (fold_idx + 1) * fold_size)
        test = df.iloc[test_idx].copy()
        train_rest = df.drop(test_idx)

        # 再从其余数据中抽取80%作为训练集
        train = train_rest.sample(frac=0.8, random_state=fold_idx).copy()

        X_train = sm.add_constant(train[alpha_cols])
        y_train = train[target]

        mask = X_train.replace([np.inf, -np.inf, np.nan], np.nan).notnull().all(axis=1) & y_train.notnull()
        X_train, y_train = X_train[mask], y_train[mask]

        model = sm.OLS(y_train, X_train).fit()
        stats_df = model.summary2().tables[1].copy()
        stats_df["T_minus"] = tminus
        stats_df["fold"] = fold_idx
        all_stats.append(stats_df)

        X_test = sm.add_constant(test[alpha_cols])
        y_pred = model.predict(X_test)

        # 可选：保存预测值或其他评估指标
        test["predicted"] = y_pred
        final_df = pd.concat([final_df, test], ignore_index=True)

        # 如有需要可画图或记录feature importance等

    # 每个 tminus 结束后的可视化/打印等
