import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 初始化结果列表
line_y = []
box_data = []
x_ticks = []

# 遍历每个 T_minus
for t in sorted(final_df["T_minus"].unique()):
    if t == 1:
        continue
    sub_df = final_df[final_df["T_minus"] == t].copy()
    sub_df = sub_df.fillna(0)

    # 计算信号项
    signal = np.sign(sub_df["predicted"]) * (sub_df["target_weight"] - sub_df["current_weight"]) * sub_df["EV_it"]
    
    # 累加结果用于折线图
    line_y.append(signal.sum())
    
    # boxplot数据
    box_data.append(signal)
    x_ticks.append(f"T-{t}")

# ---- 画图部分 ----

# 折线图
plt.figure(figsize=(10, 5))
plt.plot(x_ticks, line_y, marker='o')
plt.title("Aggregate Signal Impact vs T_minus")
plt.ylabel("Sum of signal * weight_diff * EV_it")
plt.xlabel("T_minus")
plt.grid(True)
plt.tight_layout()
plt.show()

# Boxplot（去掉离群值）
plt.figure(figsize=(10, 5))
plt.boxplot(box_data, labels=x_ticks, showfliers=False)
plt.title("Distribution of signal * weight_diff * EV_it (per T_minus)")
plt.ylabel("Signal Value")
plt.xlabel("T_minus")
plt.grid(True)
plt.tight_layout()
plt.show()
