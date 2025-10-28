# Stock Price Prediction Model

**Author:** Gokul Akash S  
**Purpose:** Educational/analysis repo demonstrating 4 approaches to regression on stock price data.

---

## Quick Summary
This repo contains code and datasets used to:
- Pre-process historical stock price data
- Engineer common features (returns, moving averages, volatility, etc.)
- Train and evaluate 4 modeling approaches:
  1. Manual Regression (baseline)
  2. Scikit-learn models
  3. TensorFlow Neural Network
  4. XGBoost

---

## Files in this repo
- `Linear Regression - Manual.py` — manual gradient-descent regression (uses monthly_data.csv).
- `Scikit-learn.py` — scikit-learn experiments (uses monthly_data.csv/daily_data.csv).
- `Tensorflow.py` — neural-network implementation (uses `daily_data.csv`).
- `XGBoost.py` — XGBoost implementation (uses `daily_data.csv`).
- `monthly_data.csv` — smaller monthly dataset included for quick runs/demos.
- `daily_data.csv` — full daily dataset (~4,700+ rows) included for the main experiments.

---

## Important notes
- This project is for educational/analysis purposes and **Does Not** provide trading advice or production-ready forecasts.
- This repository is public for **viewing only** (no license provided). See `NOTICE.txt`.
