# -*- coding: utf-8 -*-
"""本地测试模型预测逻辑（不启动 streamlit）"""
import os, json, pickle
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
md = json.load(open(os.path.join(HERE, "metadata.json"), encoding="utf-8"))
with open(os.path.join(HERE, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
ckpt = torch.load(os.path.join(HERE, "model.pt"), map_location="cpu")

class MLP(nn.Module):
    def __init__(self, n_feat, hidden=(128, 64), dropout=0.4, batchnorm=True):
        super().__init__()
        layers = []; prev = n_feat
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if batchnorm: layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU()); layers.append(nn.Dropout(dropout)); prev = h
        layers.append(nn.Linear(prev, 1)); self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(-1)

models = []
for sd in ckpt["models"]:
    m = MLP(ckpt["n_feat"]); m.load_state_dict(sd); m.eval()
    models.append(m)

MAT = md["mat_features"]; CELLS = md["cell_lines"]

def predict(comp, cell, dose):
    desc = md["materials"][comp]
    x = [desc[m] for m in MAT] + [np.log10(dose+1.0)] + [1.0 if c==cell else 0.0 for c in CELLS]
    x = np.array(x, dtype=np.float32).reshape(1, -1)
    xs = scaler.transform(x)
    with torch.no_grad():
        p = float(np.mean([torch.sigmoid(m(torch.tensor(xs))).item() for m in models]))
    return p

print("=== 预测测试（3 种子 ensemble）===")
for comp, cell, dose in [("Ag", "U937", 50), ("CeO2", "U937", 50),
                          ("TiO2", "A549", 10), ("ZnO", "U937", 100),
                          ("CeO2", "A549", 10), ("SiO2", "THP-1", 25)]:
    p = predict(comp, cell, dose)
    verdict = "TOXIC" if p >= 0.5 else "non-toxic"
    print(f"  {comp:6s} in {cell:9s} @ {dose:>5.0f} µg/mL → p={p:.3f}  [{verdict}]")
print("\n测试完成")
