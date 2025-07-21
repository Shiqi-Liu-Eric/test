returns = (close_df.loc[valid_tickers_k, date] / close_df.loc[valid_tickers_k, date - pd.offsets.BDay(1)]).where(
    (close_df.loc[valid_tickers_k, date] != 0) & (close_df.loc[valid_tickers_k, date - pd.offsets.BDay(1)] != 0) & 
    close_df.loc[valid_tickers_k, date].notna() & close_df.loc[valid_tickers_k, date - pd.offsets.BDay(1)].notna(), 1) - 1
