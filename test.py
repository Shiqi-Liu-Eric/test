import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 提取 alpha_ 开头的列
alpha_cols = [col for col in result_df.columns if col.startswith('alpha_')]
alpha_data = result_df[alpha_cols].dropna()  # 也可用 fillna(0) 替代

# 标准化
scaler = StandardScaler()
alpha_scaled = scaler.fit_transform(alpha_data)

# PCA
pca = PCA()
pca_result = pca.fit_transform(alpha_scaled)

# 图1: PCA分析图
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: 前10个主成分的解释方差比
axes[0].bar(range(1, 11), pca.explained_variance_ratio_[:10])
axes[0].set_title("Explained Variance Ratio of First 10 Principal Components")
axes[0].set_xlabel("Principal Component")
axes[0].set_ylabel("Explained Variance Ratio")

# Subplot 2: PC1 vs PC2
axes[1].scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
axes[1].set_title("PCA - PC1 vs PC2")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")

plt.tight_layout()
plt.show()

# 图2: alpha_相关性热图
plt.figure(figsize=(12, 10))
corr_matrix = alpha_data.corr()
sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=False)
plt.title("Correlation Matrix of Alpha Features")
plt.show()
