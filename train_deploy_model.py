# -*- coding: utf-8 -*-
"""训练最终部署用 MLP 模型（调优配置，3 种子 ensemble），保存模型 + scaler + 元数据"""
import os, json, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

BASE = r"C:\Users\Lixiao\Desktop\悦悦课题\纳米毒性QSAR论文"
DEPLOY = os.path.join(BASE, "model_deploy")
os.makedirs(DEPLOY, exist_ok=True)
MAT = ["size_nm", "electronegativity", "ionic_radius_pm", "oxidation_state", "atomic_number", "atomic_mass"]

# ===== 统一口径数据构造（dropna 4 → size_nm 填充 → 筛选 nanoreg1）=====
df = pd.read_csv(os.path.join(BASE, "data", "enanomapper", "clean_viability.csv"), encoding="utf-8-sig")
df = df[df["composition"] != "mineral"].copy()
df = df.dropna(subset=["electronegativity", "value_pct", "dose_ugml", "cell_std"]).copy()
df["size_nm"] = df.groupby("composition")["size_nm"].transform(lambda x: x.fillna(x.median()))
df["size_nm"] = df["size_nm"].fillna(df["size_nm"].median())
df["log_dose"] = np.log10(df["dose_ugml"].astype(float) + 1.0)
df["toxic"] = (df["value_pct"].astype(float) < 50).astype(int)
n1 = df[df["instance"] == "nanoreg1"].copy()

cells = sorted(n1["cell_std"].unique())
X_mat = n1[MAT].astype(float).values
X_dose = n1[["log_dose"]].astype(float).values
X_cell = pd.get_dummies(n1["cell_std"]).astype(int)[cells].values  # 固定列顺序
X = np.hstack([X_mat, X_dose, X_cell]).astype(np.float32)
y = n1["toxic"].values.astype(np.float32)
print(f"训练数据: n={len(X)} 特征={X.shape[1]} 有毒={y.mean()*100:.1f}%")

# ===== 材料描述符（app 预置）=====
mat_desc = {}
for comp in sorted(n1["composition"].unique()):
    row = n1[n1["composition"] == comp].iloc[0]
    mat_desc[comp] = {m: float(row[m]) for m in MAT}

# ===== MLP 模型（与 verify_tune_dl 调优配置一致）=====
class MLP(nn.Module):
    def __init__(self, n_feat, hidden=(128, 64), dropout=0.4, batchnorm=True):
        super().__init__()
        layers = []; prev = n_feat
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU()); layers.append(nn.Dropout(dropout)); prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)

# StandardScaler 在全部数据上 fit
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

def train_one(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = MLP(X.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    Xt = torch.tensor(Xs); yt = torch.tensor(y)
    pos_weight = torch.tensor([(yt == 0).sum() / max((yt == 1).sum(), 1)])
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for _ in range(150):
        model.train(); opt.zero_grad(); lossf(model(Xt), yt).backward(); opt.step()
    model.eval()
    return model

# 3 种子 ensemble
seeds = [0, 1, 2]
models = [train_one(s) for s in seeds]
print("3 种子训练完成")

# 保存模型权重（list of state_dict）
state_dicts = [m.state_dict() for m in models]
torch.save({"models": state_dicts, "n_feat": X.shape[1]}, os.path.join(DEPLOY, "model.pt"))

# 保存 scaler
with open(os.path.join(DEPLOY, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# 保存元数据
metadata = {
    "materials": mat_desc,
    "cell_lines": cells,
    "feature_order": MAT + ["log_dose"] + [f"cell_{c}" for c in cells],
    "n_features": X.shape[1],
    "mat_features": MAT,
    "model_config": {"hidden": [128, 64], "dropout": 0.4, "batchnorm": True,
                     "epochs": 150, "lr": 1e-3, "weight_decay": 1e-3, "seeds": seeds},
    "threshold": 0.5,
    "positive_class": "cytotoxic (viability < 50%)",
    "training_n": int(len(X)),
    "toxic_rate": float(y.mean()),
}
with open(os.path.join(DEPLOY, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

# 验证：在训练数据上的 AUC
from sklearn.metrics import roc_auc_score
with torch.no_grad():
    probs = []
    for m in models:
        probs.append(torch.sigmoid(m(torch.tensor(Xs))).numpy())
    prob = np.mean(probs, axis=0)
auc = roc_auc_score(y, prob)
print(f"训练集 AUC (3种子 ensemble): {auc:.3f}")
print(f"已保存到 {DEPLOY}:")
for f in os.listdir(DEPLOY):
    print(f"  {f} ({os.path.getsize(os.path.join(DEPLOY, f))//1024} KB)")
