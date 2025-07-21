returns = close_df.loc[valid_tickers_k, date].pct_change().where(lambda x: ((close_df.loc[valid_tickers_k, date] != 0) & close_df.loc[valid_tickers_k, date].notna()).all(axis=1), 0).fillna(0)
