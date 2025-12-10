# Hull Tactical Market Prediction 📈

![Competition](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle&logoColor=white)
![Rank](https://img.shields.io/badge/Rank-1281%2F3382%20(Top%2038%25)-success)
![Score](https://img.shields.io/badge/Score-8.205-blue)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success)

A machine learning solution for tactical asset allocation and market timing, achieving **top 38% ranking** in the Kaggle Hull Tactical Market Prediction competition.

---

## 🎯 Problem Statement

**Challenge**: Predict S&P 500 excess returns and build a betting strategy that outperforms the market while staying within a **120% volatility constraint**.

This competition challenges the **Efficient Market Hypothesis (EMH)** - the idea that all information is already priced in, making market timing impossible. With modern machine learning, can we uncover patterns that traditional theory says shouldn't exist?

**Key Objectives**:
- Predict daily excess returns of the S&P 500
- Design a dynamic position sizing strategy (0-2x leverage)
- Outperform buy-and-hold while managing risk
- Stay within 120% of market volatility

**Why This Matters**: Most investors fail to beat the S&P 500, which has long been cited as proof of market efficiency. But markets are noisy, behavioral, and full of patterns. This competition tests whether data science can find repeatable edges that academic theory claims are impossible.

**Real-world Impact**: Successful models from this competition could be deployed in live tactical asset allocation strategies, helping reshape how investors understand and navigate financial markets.

---

## 🚀 Solution Highlights

My approach combines **ensemble learning**, **regime detection**, and **adaptive risk management**:

### 1️⃣ **Multi-Model Ensemble**
- **LightGBM** + **CatBoost** + **XGBoost** (equal-weighted)
- 460 estimators per model with conservative hyperparameters
- Reduces model-specific biases and improves robustness

### 2️⃣ **Market Regime Detection**
```
Bull Market   → Aggressive positioning (1280x base multiplier)
Bear Market   → Defensive positioning (1020x base multiplier)  
Neutral       → Moderate positioning (1160x base multiplier)
```
- Detects regimes using 30-day rolling returns and moving average crossovers
- Adapts strategy to current market conditions

### 3️⃣ **Advanced Feature Engineering** (60+ features)
- **Momentum indicators**: Multi-timeframe lags (1, 2, 5 days), rolling means
- **Technical indicators**: RSI, MACD, Bollinger Bands, ROC
- **Volatility features**: Rolling standard deviations, volatility regime ratios
- **Cross-sectional features**: Ratio and interaction terms

### 4️⃣ **Dynamic Risk Management**
- **Volatility targeting**: Scales positions to maintain 1.2% target volatility
- **Position limits**: Clips signals between 0-2x (no shorting)
- **Adaptive multipliers**: Adjusts based on recent 20-day volatility

---

## 📊 Results

| Metric | Value |
|--------|-------|
| **Final Score** | 8.205 |
| **Leaderboard Rank** | 1281 / 3382 |
| **Percentile** | Top 38% |
| **Competition Period** | Sept 16 - Dec 15, 2025 |
| **Evaluation Metric** | Modified Sharpe Ratio* |

*The competition uses a variant of the Sharpe ratio that penalizes strategies exceeding 120% market volatility or failing to beat market returns.

**Key Achievement**: Built a regime-aware ensemble system that balances return generation with strict volatility constraints, successfully challenging the Efficient Market Hypothesis assumption that market timing is impossible.

---

## 🏗️ Technical Architecture

```
Raw Market Data (M1, M2, P1, P2, S1, S2, I1, I2)
         ↓
Feature Engineering (60+ features)
         ↓
    Preprocessing
    ├── StandardScaler
    └── SimpleImputer (median)
         ↓
   Ensemble Models
   ├── LightGBM Regressor
   ├── CatBoost Regressor  
   └── XGBoost Regressor
         ↓
   Average Predictions
         ↓
  Regime Detection (Bull/Bear/Neutral)
         ↓
  Adaptive Multiplier Calculation
         ↓
  Position Signal (0.0 - 2.0)
```

---

## 🛠️ Tech Stack

**ML Frameworks:**
- `scikit-learn` - Preprocessing & baseline models
- `LightGBM` - Gradient boosting (primary)
- `CatBoost` - Gradient boosting with categorical handling
- `XGBoost` - Gradient boosting for ensemble diversity

**Data Processing:**
- `pandas` - Data manipulation
- `polars` - High-performance data loading
- `numpy` - Numerical computations

**Others:**
- `kaggle-evaluation` - Competition inference server

---

## 📦 Installation & Usage

### Prerequisites
```bash
Python 3.11+
```

### Setup
```bash
# Clone repository
git clone https://github.com/yourusername/hull-tactical-prediction.git
cd hull-tactical-prediction

# Install dependencies
pip install -r requirements.txt

# Download competition data from Kaggle
# Place in: data/train.csv
```

### Quick Start
```python
from src.models import train_ensemble
from src.features import create_features

# Load and prepare data
train_df = pd.read_csv('data/train.csv')
train_df = create_features(train_df)

# Train ensemble
models = train_ensemble(train_df)

# Make predictions
signal = predict(test_df)  # Returns value between 0.0-2.0
```

---

## 📁 Project Structure

```
hull-tactical-prediction/
├── README.md                          # You are here
├── requirements.txt                   # Dependencies
├── notebooks/
│   └── 04_final_solution_v38.ipynb   # Competition submission
├── src/
│   ├── features.py                   # Feature engineering
│   ├── models.py                     # Model training
│   ├── regime_detection.py           # Market regime logic
│   └── utils.py                      # Helper functions
├── data/
│   └── README.md                     # Data download instructions
├── docs/
│   ├── methodology.md                # Detailed approach
│   └── results.md                    # Performance analysis
└── images/
    └── (visualizations)
```

---

## 💡 What I Learned

1. **Market efficiency isn't absolute** - ML can find exploitable patterns that traditional finance theory overlooks
2. **Regime-aware modeling** significantly improves tactical allocation strategies
3. **Ensemble methods** reduce overfitting in financial time series
4. **Volatility targeting** is crucial for meeting competition constraints (120% limit)
5. **Feature engineering** from domain knowledge (technical indicators) beats pure statistical features
6. **Conservative hyperparameters** prevent overfitting in noisy financial data

---

## 🔮 Future Improvements

- [ ] Implement **walk-forward validation** for more realistic backtesting
- [ ] Add **alternative regime detection** methods (HMM, clustering)
- [ ] Incorporate **macroeconomic features** (VIX, yield curves)
- [ ] Test **dynamic ensemble weighting** based on recent performance
- [ ] Explore **deep learning** approaches (LSTM, Transformers)
- [ ] Add **transaction cost modeling** for real-world applicability

---

## 📧 Contact

Open to opportunities in **Data Science** and **Quantitative Finance**!

- **GitHub**: [@karankumar02-12](https://github.com/yourusername)
- **LinkedIn**: [Karan Kumar](https://linkedin.com/in/yourprofile)
- **Email**: karan.kumar021299@gmail.com

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Kaggle** for hosting the competition
- **Hull Tactical** for the problem formulation
- Open-source ML community for excellent tools

---

⭐ **If you found this helpful, please star the repository!**
