# Helper function to get closest available date from columns
def get_valid_date(target_date, columns):
    return columns[columns.get_indexer([target_date], method="nearest")[0]]

# Updated plotting function with valid date fallback
def plot_group_weighted_returns_v2(dfs, map_df, rebalance_dates, window_size=50, plot_list=["Add", "Delete", "Up Weight", "Down Weight"], scale=True):
    close_df = dfs["close"]
    weight_df = dfs["weight_ftse_globalallcap"]
    event_action_df = dfs["event_action"]
    country_df = dfs["country_ftse_globalallcap"].copy()

    reference_idx = -400 if country_df.shape[1] > 400 else -1
    ref_date = country_df.columns[reference_idx]
    all_countries = country_df[ref_date].value_counts().nlargest(12).index.tolist()
    country_df = country_df.applymap(lambda x: x if x in all_countries else "Others")

    all_industries = sorted(map_df["industry_sector"].unique())
    n_rows, n_cols = len(all_countries), len(all_industries)
    fig = plt.figure(figsize=(4 * n_cols, 2.5 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, wspace=0.1, hspace=0.1)

    global_min, global_max = np.inf, -np.inf
    cache_result = {}

    for r_date in rebalance_dates:
        T = pd.to_datetime(r_date)
        if T not in close_df.columns:
            T = get_valid_date(T, close_df.columns)
        T_idx = close_df.columns.get_loc(T)

        # valid tickers: those with non-null weight on T-5 BDays (valid column fallback)
        T_minus_5 = get_valid_date(T - BDay(5), weight_df.columns)
        valid_tickers = weight_df.loc[:, T_minus_5].dropna().index

        for i, country in enumerate(all_countries):
            for j, industry in enumerate(all_industries):
                tickers_in_group = [
                    t for t in valid_tickers
                    if country_df.loc[t, T] == country and map_df.loc[t, "industry_sector"] == industry
                ]
                if not tickers_in_group:
                    continue

                result = {}
                for grp in plot_list + ["All"]:
                    tickers_grp = tickers_in_group if grp == "All" else [
                        t for t in tickers_in_group
                        if event_action_df.loc[t, get_valid_date(T - BDay(5), event_action_df.columns)] == grp
                    ]
                    if not tickers_grp:
                        continue

                    weights_date = get_valid_date(T - BDay(80), weight_df.columns) if grp == "Delete" else get_valid_date(T, weight_df.columns)
                    weights = weight_df.loc[tickers_grp, weights_date]

                    base_prices_date = close_df.columns[T_idx - window_size]
                    base_prices = close_df.loc[tickers_grp, base_prices_date]

                    ret_series = []
                    for k in range(window_size):
                        c = close_df.columns[T_idx - window_size + k]
                        rel_ret = (close_df.loc[tickers_grp, c] / base_prices - 1)
                        weighted_ret = (rel_ret * weights).sum()
                        ret_series.append(weighted_ret)

                    result[grp] = ret_series
                    if scale:
                        global_min = min(global_min, min(ret_series))
                        global_max = max(global_max, max(ret_series))

                cache_result[(i, j)] = result

    # Plotting
    for i, country in enumerate(all_countries):
        for j, industry in enumerate(all_industries):
            ax = fig.add_subplot(gs[i, j])
            if (i, j) not in cache_result:
                ax.axis("off")
                continue

            result = cache_result[(i, j)]
            for grp, series in result.items():
                ax.plot(range(window_size), series, label=grp, linewidth=1)

            # T-50 (0), T-20, T
            ax.axvline(x=0, linestyle=":", color="black", alpha=0.4)
            ax.axvline(x=window_size - 20, linestyle="--", color="gray", alpha=0.5)
            ax.axvline(x=window_size - 1, linestyle=":", color="black", alpha=0.4)

            ax.set_xticks([0, window_size - 20, window_size - 1])
            ax.set_xticklabels(["", "", ""])
            ax.tick_params(labelsize=6)
            if scale:
                ax.set_ylim(global_min, global_max)

            if j == 0:
                ax.set_ylabel(country, fontsize=9, rotation=0, labelpad=25, va='center')
            if i == 0:
                ax.set_title(industry, fontsize=9)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(plot_list) + 1)
    fig.suptitle("Group Weighted Returns by Country and Industry", fontsize=14)
    plt.show()


# Rerun after fixing the fallback date logic
plot_group_weighted_returns_v2(dfs, map_df, rebalance_dates, window_size=50, scale=True)
