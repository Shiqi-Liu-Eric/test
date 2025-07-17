from pandas.tseries.offsets import BDay

def print_group_counts(dfs, map_df, rebalance_dates, window_size=50, plot_list=["Add", "Delete", "Up Weight", "Down Weight"]):
    close_df = dfs["close"]
    weight_df = dfs["weight_ftse_globalallcap"]
    event_action_df = dfs["event_action"]
    country_df = dfs["country_ftse_globalallcap"].copy()

    reference_idx = -400 if country_df.shape[1] > 400 else -1
    ref_date = country_df.columns[reference_idx]
    top_countries = country_df[ref_date].value_counts().nlargest(12).index.tolist()
    country_df = country_df.applymap(lambda x: x if x in top_countries else "Others")

    all_countries = top_countries
    all_industries = sorted(map_df["industry_sector"].unique())

    for r_date in rebalance_dates:
        print(f"\n=== Rebalance Date: {r_date} ===")
        T = pd.to_datetime(r_date)
        if T not in close_df.columns:
            continue
        T_idx = close_df.columns.get_loc(T)
        T_minus_5 = close_df.columns[close_df.columns.get_indexer([T - BDay(5)], method="nearest")[0]]
        valid_tickers = weight_df.loc[:, T_minus_5].dropna().index

        for country in all_countries:
            for industry in all_industries:
                tickers_in_group = [
                    t for t in valid_tickers
                    if country_df.loc[t, T] == country and map_df.loc[t, "industry_sector"] == industry
                ]
                if not tickers_in_group:
                    continue

                print(f"\nGroup ({country}, {industry}):")
                for grp in plot_list + ["All"]:
                    if grp == "All":
                        count = len(tickers_in_group)
                    else:
                        action_day = event_action_df.columns[event_action_df.columns.get_indexer([T - BDay(5)], method="nearest")[0]]
                        count = sum(event_action_df.loc[t, action_day] == grp for t in tickers_in_group)
                    print(f"  {grp}: {count} tickers")
