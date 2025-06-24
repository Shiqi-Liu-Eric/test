tminuses = result_df["T_minus"].unique()
alpha_cols = [col for col in result_df.columns if col.startswith(bundle)]
all_stats = []
final_df = pd.DataFrame()

num_tminus = len(tminuses)
cols = 3
rows = (num_tminus + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
axes = axes.flatten()

for idx, tminus in enumerate(tminuses):
    print(f"\n=== T_minus {tminus} ===")
    
    df = result_df[result_df["T_minus"] == tminus].copy()
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    
    split = int(0.8 * len(df))
    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()

    if lr_test:
        X_train = sm.add_constant(train[alpha_cols])
        y_train = train["actual"]
        
        mask = X_train.replace([np.inf, -np.inf, np.nan], np.nan).notnull().all(axis=1) & y_train.notnull()
        X_train, y_train = X_train[mask], y_train[mask]

        model = sm.OLS(y_train, X_train).fit()
        stats_df = model.summary2().tables[1].copy()
        stats_df["T_minus"] = tminus
        all_stats.append(stats_df)

        X_test = sm.add_constant(test[alpha_cols])
        y_pred = model.predict(X_test)

        p_values = model.pvalues.drop("const", errors='ignore')
        p_values = p_values.replace(0, 1e-10)
        lr_importances = 1 / p_values
        indices = np.argsort(lr_importances)[-15:]
        top_features = lr_importances.index[indices]
        top_importances = lr_importances.values[indices]

        print("Training size:", X_train.shape, "Testing size:", X_test.shape)
        print("T_minus:", tminus, "with R^2:", model.rsquared)
        print("T_minus:", tminus, "with Adj R^2:", model.rsquared_adj)

        ax = axes[idx]
        ax.barh(top_features[::-1], np.log10(top_importances[::-1]), label="LR", alpha=0.6)
        ax.set_title(f"Top Log10(inv(p-value)) (LR) - T_minus {tminus}")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")

    if xgb_test: ...
    if xgb_torch_test: ...
    if mlp_test:
        test["predicted"] = y_pred
        final_df = pd.concat([final_df, test], ignore_index=True)

for j in range(idx + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# 汇总统计
if lr_test:
    all_stats_df = pd.concat(all_stats)
    summary_stats = all_stats_df.groupby(all_stats_df.index).agg({
        "t": "mean",
        "P>|t|": "mean"
    }).sort_values("P>|t|")

    print()
    print("Statistics in Average")
    print(summary_stats)
    print("\nSignificant Features:")
    print(summary_stats[summary_stats["P>|t|"] < 0.05])
