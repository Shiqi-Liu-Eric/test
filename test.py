tminuses = result_df["T_minus"].unique()
alpha_cols = [col for col in result_df.columns if col.startswith(bundle)]
final_df = pd.DataFrame()
all_stats = []

for idx_t, tminus in enumerate(tminuses):
    df_t = result_df[result_df["T_minus"] == tminus].copy()
    folds = df_t["fold"].unique()

    for fold_idx in folds:
        test = df_t[df_t["fold"] == fold_idx].copy()
        train_pool = df_t[df_t["fold"] != fold_idx].copy()

        # 随机打乱并取 80% 用作训练
        train = train_pool.sample(frac=0.8, random_state=42).copy()

        # 训练
        X_train = sm.add_constant(train[alpha_cols])
        y_train = train[target]

        mask = X_train.replace([np.inf, -np.inf, np.nan], np.nan).notnull().all(axis=1) & y_train.notnull()
        X_train, y_train = X_train[mask], y_train[mask]

        model = sm.OLS(y_train, X_train).fit()
        stats_df = model.summary2().tables[1].copy()
        stats_df["T_minus"] = tminus
        stats_df["fold"] = fold_idx
        all_stats.append(stats_df)

        # 测试并预测
        X_test = sm.add_constant(test[alpha_cols])
        y_pred = model.predict(X_test)
        test["predicted"] = y_pred

        final_df = pd.concat([final_df, test], ignore_index=True)

        # 可选：打印模型表现
        if tminus == 1:
            print(f"\n=== T_minus {tminus - 1} Fold {fold_idx} ===")
            print("Training size:", X_train.shape, "Testing size:", X_test.shape)
            print("R^2:", model.rsquared, "Adj R^2:", model.rsquared_adj)

        # 可选：重要特征画图
        p_vals = model.pvalues.drop("const", errors="ignore")
        p_vals = p_vals.replace(0, 1e-10)
        lr_importances = 1 / p_vals
        indices = np.argsort(lr_importances)[-15:]
        top_features = p_vals.index[indices]
        top_importances = lr_importances[indices]

        if idx_t == 0:  # 例如只画第一组的图
            ax = axes[fold_idx]
            ax.barh(top_features[::-1], np.log10(top_importances[::-1]), label="LR", alpha=0.6)
            ax.set_title(f"T_minus {tminus-1} Fold {fold_idx}")
            ax.set_xlabel("Importance")
            ax.set_ylabel("Feature")
