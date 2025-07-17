# Prepare plotting function

def plot_group_weighted_returns(dfs, map_df, rebalance_dates, window_size=50, plot_list=["Add", "Delete", "Up Weight", "Down Weight"]):
    close_df = dfs["close"]
    weight_df = dfs["weight_ftse_globalallcap"]
    event_action_df = dfs["event_action"]
    country_df = dfs["country_ftse_globalallcap"].copy()
    
    all_industries = sorted(map_df["industry_sector"].unique())
    all_countries = country_df.iloc[:, -1].value_counts().nlargest(12).index.tolist()
    country_df = country_df.applymap(lambda x: x if x in all_countries else "Others")

    n_rows, n_cols = len(all_countries), len(all_industries)
    fig = plt.figure(figsize=(4 * n_cols, 2.5 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, wspace=0.4, hspace=0.6)

    for r_date in rebalance_dates:
        T = pd.to_datetime(r_date)
        T_idx = close_df.columns.get_loc(T)
        valid_tickers = weight_df.iloc[:, T_idx - 5].dropna().index
        group_weights = weight_df.copy()

        for country_i, country in enumerate(all_countries):
            for industry_j, industry in enumerate(all_industries):
                ax = fig.add_subplot(gs[country_i, industry_j])
                tickers_in_group = [
                    t for t in valid_tickers
                    if country_df.loc[t, T] == country and map_df.loc[t, "industry_sector"] == industry
                ]
                if not tickers_in_group:
                    continue
                
                for grp in plot_list + ["All"]:
                    if grp != "All":
                        tickers_grp = [t for t in tickers_in_group if event_action_df.loc[t, T - pd.Timedelta(days=5)] == grp]
                    else:
                        tickers_grp = tickers_in_group

                    if not tickers_grp:
                        continue

                    weights = group_weights.loc[tickers_grp, T if grp != "Delete" else T - pd.Timedelta(days=80)]
                    base_close = close_df.loc[tickers_grp, close_df.columns[T_idx - window_size]]
                    ret_series = []
                    for i in range(window_size):
                        c = close_df.columns[T_idx - window_size + i]
                        returns = close_df.loc[tickers_grp, c] - base_close
                        weighted_return = (returns * weights).sum()
                        ret_series.append(weighted_return)

                    ax.plot(range(window_size), ret_series, label=grp)

                ax.axvline(x=window_size - 20, linestyle="--", color="gray", alpha=0.5)
                ax.set_xticks([0, window_size - 20, window_size - 1])
                ax.set_xticklabels([f"T-{window_size}", "T-20", "T"])
                ax.set_title(f"{country} - {industry}", fontsize=8)
                ax.tick_params(labelsize=6)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(plot_list) + 1)
    fig.suptitle("Group Weighted Returns by Country and Industry", fontsize=14)
    plt.show()

# Call the function for plotting
plot_group_weighted_returns(dfs, map_df, rebalance_dates, window_size=50)
