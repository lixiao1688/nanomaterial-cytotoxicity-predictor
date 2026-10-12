# -*- coding: utf-8 -*-
"""纳米材料细胞毒性预测 web app（streamlit）
模型：MLP 调优（3 种子 ensemble），特征 = 6 材料描述符 + log10(剂量+1) + 14 细胞系 one-hot
"""
import os, json, pickle
import numpy as np
import torch
import torch.nn as nn
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))

# ===== 加载模型与元数据 =====
@st.cache_resource
def load_assets():
    md = json.load(open(os.path.join(HERE, "metadata.json"), encoding="utf-8"))
    with open(os.path.join(HERE, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    ckpt = torch.load(os.path.join(HERE, "model.pt"), map_location="cpu")
    return md, scaler, ckpt

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

md, scaler, ckpt = load_assets()
models = []
for sd in ckpt["models"]:
    m = MLP(ckpt["n_feat"], hidden=(128, 64), dropout=0.4, batchnorm=True)
    m.load_state_dict(sd)
    m.eval()
    models.append(m)

MAT = md["mat_features"]
MATERIALS = md["materials"]
CELLS = md["cell_lines"]

# ===== UI =====
st.set_page_config(page_title="NanoCytotoxicity Predictor", page_icon="🧪", layout="centered")
st.title("Nanomaterial Cytotoxicity Predictor")
st.caption("Multi-cell-line screening tool. Predicts whether a nanomaterial reduces cell viability below 50% "
           "in a given cell line, from the model described in the accompanying manuscript.")

st.markdown("### Input")
col1, col2 = st.columns(2)
with col1:
    mat_choice = st.selectbox("Material", list(MATERIALS.keys()),
                              help="Six nanomedicine-relevant material classes.")
    # 是否自定义材料描述符
    custom = st.checkbox("Edit material descriptors manually")
with col2:
    cell_choice = st.selectbox("Cell line", CELLS)
    dose = st.number_input("Exposure concentration (µg/mL)", min_value=0.0, value=50.0, step=1.0)

desc = dict(MATERIALS[mat_choice])
if custom:
    st.markdown("**Material descriptors** (elemental tier)")
    c1, c2, c3 = st.columns(3)
    with c1:
        desc["size_nm"] = st.number_input("Particle size (nm)", value=float(desc["size_nm"]), step=1.0)
        desc["electronegativity"] = st.number_input("Electronegativity", value=float(desc["electronegativity"]), step=0.01)
    with c2:
        desc["ionic_radius_pm"] = st.number_input("Ionic radius (pm)", value=float(desc["ionic_radius_pm"]), step=1.0)
        desc["oxidation_state"] = st.number_input("Oxidation state", value=float(desc["oxidation_state"]), step=1.0)
    with c3:
        desc["atomic_number"] = st.number_input("Atomic number", value=float(desc["atomic_number"]), step=1.0)
        desc["atomic_mass"] = st.number_input("Atomic mass", value=float(desc["atomic_mass"]), step=0.01)

# ===== 特征构造（与训练一致）=====
def build_features(desc, cell, dose):
    x_mat = [desc[m] for m in MAT]
    x_dose = [np.log10(dose + 1.0)]
    x_cell = [1.0 if c == cell else 0.0 for c in CELLS]
    return np.array(x_mat + x_dose + x_cell, dtype=np.float32).reshape(1, -1)

if st.button("Predict cytotoxicity", type="primary"):
    x = build_features(desc, cell_choice, dose)
    xs = scaler.transform(x)
    xt = torch.tensor(xs)
    with torch.no_grad():
        probs = [torch.sigmoid(m(xt)).item() for m in models]
    p = float(np.mean(probs))
    toxic = p >= md["threshold"]

    st.markdown("### Result")
    if toxic:
        st.error(f"**Predicted cytotoxic** (viability < 50%)\n\nProbability = {p:.3f}")
    else:
        st.success(f"**Predicted non-cytotoxic** (viability ≥ 50%)\n\nProbability = {p:.3f}")

    st.progress(min(max(p, 0.0), 1.0))
    st.caption(f"Probabilities across 3 seeds: " + ", ".join(f"{q:.3f}" for q in probs))

    # 提示：其他细胞系参考
    st.markdown("#### Per-cell-line comparison (same material and dose)")
    rows = []
    for c in CELLS:
        xc = build_features(desc, c, dose)
        xsc = scaler.transform(xc)
        with torch.no_grad():
            pc = float(np.mean([torch.sigmoid(m(torch.tensor(xsc))).item() for m in models]))
        rows.append((c, pc))
    rows.sort(key=lambda r: -r[1])
    import pandas as pd
    df_show = pd.DataFrame(rows, columns=["Cell line", "Cytotoxicity probability"])
    st.dataframe(df_show, use_container_width=True, height=360)

st.markdown("---")
st.caption("Disclaimer: this tool is a screening aid and does not replace experimental or regulatory "
           "toxicity assessment. Predictions are conditioned on the six material classes and fourteen "
           "cell lines of the training set.")
