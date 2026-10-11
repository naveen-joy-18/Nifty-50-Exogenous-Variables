# Nifty 50 Multimodal Exogenous Hybrid Forecasting Project

> **Research Objective:**  
> *"To Incorporate relevant exogenous variables (e.g., economic indicators, geopolitical events, macroeconomic factors) into the hybrid model to enhance prediction accuracy."*

---

## 🚀 Project Overview

This repository contains the complete, end-to-end research project for predicting daily movements of the **Nifty 50 Index** using a novel multimodal deep learning architecture: **`ExoVision-XAttn-BiLSTM`**.

The framework synchronizes **240 daily Nifty candlestick chart images** (spanning 2025 and 2026) with multi-scale **endogenous technical indicators** and key **exogenous macroeconomic and geopolitical drivers**:
- **Brent Crude Oil (`BZ=F`)**: Critical energy import and inflationary pressure driver.
- **USD/INR Exchange Rate (`INR=X`)**: Currency risk and foreign institutional capital flow barometer.
- **US 10-Year Treasury Yield (`^TNX`)**: Global sovereign risk-free discount rate.
- **Gold Spot (`GC=F`)**: Safe-haven and geopolitical tension index.
- **S&P 500 (`^GSPC`)**: Global equity overnight contagion driver.
- **India VIX (`^INDIAVIX`)**: Domestic market fear and option implied volatility.
- **Macro Stress Composite**: Non-linear composite stress index.

All figures and charts produced by this project are formatted in **pure Scalable Vector Graphics (`.svg`)**.

---

## 📁 Repository Structure

```
nifty daily/
├── 2025/                           # Candlestick chart images for 2025 (July - Dec)
├── 2026/                           # Candlestick chart images for 2026 (Jan - Aug)
├── src/
│   ├── data_loader.py              # Multimodal Dataset with in-memory caching & yfinance ingestion
│   ├── feature_engineering.py      # Technical indicators & macroeconomic exogenous transformations
│   ├── models.py                   # PyTorch models: Proposed ExoVision-XAttn-BiLSTM & 7 Baselines
│   ├── train_and_evaluate.py       # Training loops, multi-task Huber+BCE loss, metrics & trading backtest
│   └── svg_visualizer.py           # 10 Publication-grade SVG figure generators
├── results/
│   ├── benchmark_results.csv       # Comparative performance across all 8 models
│   └── ablation_results.csv        # Systematic ablation study metrics
├── run_full_experiment.py          # Master execution script orchestrating full pipeline
└── README.md                       # Project documentation
```

---

## 📊 Key Experimental Findings

### Model Benchmarks (Out-of-Sample Test Set: 2026)

| Model Architecture | Modalities | RMSE (Pts) ↓ | MAE (Pts) ↓ | MAPE (%) ↓ | $R^2$ Score ↑ | Directional Accuracy (%) ↑ | Sharpe Ratio ↑ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ARIMAX** | Numerical + Exo | $170.76$ | $128.08$ | $0.534\%$ | $0.8042$ | $43.75\%$ | $-0.88$ |
| **Random Forest** | Numerical + Exo | $140.81$ | $103.20$ | $0.430\%$ | $0.8669$ | $56.25\%$ | $-1.42$ |
| **XGBoost** | Numerical + Exo | $151.66$ | $114.77$ | $0.478\%$ | $0.8456$ | $56.25\%$ | $-2.75$ |
| **Pure Vision CNN** | Candlestick Images | $120.47$ | $99.27$ | $0.414\%$ | $0.9025$ | $68.75\%$ | $+2.95$ |
| **Pure BiLSTM** | Endogenous Only | $133.89$ | $95.33$ | $0.397\%$ | $0.8796$ | $68.75\%$ | $+1.66$ |
| **Vanilla CNN-BiLSTM (No Exo)** | Vision + Endogenous | $125.69$ | $93.16$ | $0.388\%$ | $0.8939$ | $68.75\%$ | $+3.66$ |
| **Early Concat Multimodal** | Vis + Endo + Exo | $121.09$ | $91.01$ | $0.379\%$ | $0.9015$ | $70.83\%$ | $+1.84$ |
| **Proposed ExoVision-XAttn** | **Vis + Endo + Exo + XAttn** | **$120.30$** | **$91.86$** | **$0.382\%$** | **$0.9028$** | **$75.00\%$** | **$+5.53$** |


## ⚡ How to Run

To re-run the entire pipeline and re-generate all results and SVG figures:

```bash
# Execute master research runner
python run_full_experiment.py
```