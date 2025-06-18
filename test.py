import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 选定特征列
X = result_df[alpha_cols].copy()
y = result_df["actual"].copy()
w = result_df["target_weight"].copy()

# 清理与填充
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.mean(), inplace=True)
y.fillna(0, inplace=True)
w.fillna(0, inplace=True)

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练测试集
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X_scaled, y, w, test_size=0.2, random_state=42
)

# 转为 tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
w_train = torch.tensor(w_train.values, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
w_test = torch.tensor(w_test.values, dtype=torch.float32).view(-1, 1)

class WeightedMLP(nn.Module):
    def __init__(self, input_dim):
        super(WeightedMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

def weighted_mse_loss(pred, target, weight):
    return torch.mean(weight * (pred - target) ** 2)

model = WeightedMLP(X_train.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    y_pred = model(X_train)
    loss = weighted_mse_loss(y_pred, y_train, w_train)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

model.eval()
with torch.no_grad():
    y_pred_test = model(X_test)
    test_loss = weighted_mse_loss(y_pred_test, y_test, w_test)
    print(f"Test Weighted Loss: {test_loss.item():.6f}")


# ====================
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 准备数据
X = result_df[alpha_cols].copy()
y = result_df["actual"].copy()
w = result_df["target_weight"].copy()

X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.mean(), inplace=True)
y.fillna(0, inplace=True)
w.fillna(0, inplace=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X_scaled, y, w, test_size=0.2, random_state=42
)

# 转为 DMatrix（必须显式传入权重）
dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
dtest = xgb.DMatrix(X_test, label=y_test, weight=w_test)

# 定义自定义损失函数：加权 MSE
def weighted_mse_obj(preds, dtrain):
    labels = dtrain.get_label()
    weights = dtrain.get_weight()
    grad = 2 * weights * (preds - labels)
    hess = 2 * weights
    return grad, hess

params = {
    'max_depth': 6,
    'eta': 0.03,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'reg:squarederror',  # 仍然设置但被 override
    'eval_metric': 'rmse'
}

bst = xgb.train(
    params,
    dtrain,
    num_boost_round=200,
    obj=weighted_mse_obj,  # ← 关键在这里
    evals=[(dtrain, 'train'), (dtest, 'test')],
    early_stopping_rounds=20
)

y_pred = bst.predict(dtest)
rmse = np.sqrt(np.average((y_pred - y_test) ** 2, weights=w_test))
print(f"Test Weighted RMSE: {rmse:.6f}")
