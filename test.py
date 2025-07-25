def step2_calc(df):
    df = df[df['T_minus'] == 4].copy()

    if df['EV_it'].isna().all() or df['EV_it'].nunique() < 5:
        return [np.nan] * 5

    try:
        df['EV_quantile'] = pd.qcut(df['EV_it'], 5, labels=False, duplicates='drop')
    except ValueError:
        return [np.nan] * 5

    results = []
    for q in range(5):
        q_df = df[df['EV_quantile'] == q]
        if q_df.empty:
            results.append(np.nan)
            continue
        delta_weight = q_df['target_weight'] - q_df['current_weight']
        same_direction = np.sign(q_df['EV_it']) == np.sign(delta_weight)
        numerator = (q_df['EV_it'] * delta_weight)[same_direction].sum()
        denominator = delta_weight.abs().sum()
        value = numerator / denominator if denominator != 0 else 0
        results.append(value)
    return results
