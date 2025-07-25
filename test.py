step1 = (
    result_df
    .groupby('fold')
    .apply(lambda df: (df['EV_it'] * df['current_weight']).sum())
    .rename('EV_times_current_weight_sum')
    .reset_index()
)

import numpy as np
import pandas as pd

def step2_calc(df):
    df = df[df['T_minus'] == 4].copy()
    df['EV_quantile'] = pd.qcut(df['EV_it'], 5, labels=False)

    results = []
    for q in range(5):
        q_df = df[df['EV_quantile'] == q]
        delta_weight = q_df['target_weight'] - q_df['current_weight']
        same_direction = np.sign(q_df['EV_it']) == np.sign(delta_weight)
        numerator = (q_df['EV_it'] * delta_weight)[same_direction].sum()
        denominator = delta_weight.abs().sum()
        value = numerator / denominator if denominator != 0 else 0
        results.append(value)
    return results

step2_results = result_df.groupby('fold').apply(step2_calc)

# 转换为 DataFrame 并对五个分位做平均
step2_avg = pd.DataFrame(step2_results.tolist()).mean().rename({i: f'q{i+1}' for i in range(5)})

def step3_calc(df):
    df = df[df['T_minus'] == 4].copy()
    df['EV_quantile'] = pd.qcut(df['EV_it'], 5, labels=False)

    results = []
    for q in range(5):
        q_df = df[df['EV_quantile'] == q]
        numerator = (q_df['EV_it'] * q_df['current_weight']).sum()
        denominator = q_df['current_weight'].sum()
        value = numerator / denominator if denominator != 0 else 0
        results.append(value)
    return results

step3_results = result_df.groupby('fold').apply(step3_calc)
step3_avg = pd.DataFrame(step3_results.tolist()).mean().rename({i: f'q{i+1}' for i in range(5)})

print("Step 1: EV_it * current_weight per fold")
print(step1)

print("\nStep 2: Average directional gain ratio over folds (5 quantiles)")
print(step2_avg)

print("\nStep 3: Average EV_it weighted by current_weight over folds (5 quantiles)")
print(step3_avg)
