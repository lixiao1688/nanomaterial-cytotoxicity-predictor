# Nanomaterial Cytotoxicity Predictor

A multi-cell-line machine-learning tool that predicts whether a nanomaterial reduces cell viability below 50% in a specific cell line. It accompanies the manuscript *"Cell-Line Type Dominates Machine-Learning Prediction of Nanomaterial Cytotoxicity: A Multi-Cell-Line Screening Framework for Nanomedicine"*.

## Model

- **Architecture**: multilayer perceptron (MLP, hidden layers 128–64, BatchNorm, ReLU, Dropout 0.4), a three-seed ensemble.
- **Training**: 2,655 dose–response records from the EU eNanoMapper `nanoreg1` project (six materials, fourteen cell lines); binary target = viability < 50%.
- **Features (21)**: particle size and five elemental descriptors (electronegativity, ionic radius, oxidation state, atomic number, atomic mass) + log10(dose + 1) + cell-line one-hot encoding.
- **Reported performance**: leave-one-material-out pooled AUC 0.856 (see manuscript Table 2).

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the printed local URL in a browser.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit web interface |
| `model.pt` | Trained MLP weights (3-seed ensemble) |
| `scaler.pkl` | Feature standardizer (fit on the training set) |
| `metadata.json` | Material descriptors, cell-line list, feature order |
| `train_deploy_model.py` | Script that reproduces training and exports the artifacts above |

## Usage

1. Select a material (or edit its elemental descriptors manually).
2. Select a target cell line.
3. Enter the exposure concentration in µg/mL.
4. Click **Predict cytotoxicity**. The tool returns the predicted probability that viability falls below 50%, the binary call, and a per-cell-line comparison for the same material and dose.

## Data

Training data were retrieved from the public [eNanoMapper](https://enanomapper.adma.ai/) nanosafety knowledge base.

## Disclaimer

This tool is a screening aid. It does not replace experimental or regulatory toxicity assessment. Predictions are conditioned on the six material classes and fourteen cell lines of the training set.

## Citation

Please cite the accompanying manuscript if you use this tool.
