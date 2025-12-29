# Competition Data

This folder should contain the Hull Tactical Market Prediction competition data.

## How to Get the Data

### Option 1: Download from Kaggle (Recommended)

1. Visit the competition page: [Hull Tactical Market Prediction](https://www.kaggle.com/competitions/hull-tactical-market-prediction)
2. Accept the competition rules
3. Go to the "Data" tab
4. Download `train.csv` and `test.csv`
5. Place files in this directory

### Option 2: Using Kaggle API
```bash
# Install Kaggle API
pip install kaggle

# Download competition data
kaggle competitions download -c hull-tactical-market-prediction

# Unzip
unzip hull-tactical-market-prediction.zip -d data/
```

## Data Files

Once downloaded, this folder should contain:

- `train.csv` - Training data with historical features and returns
- `test.csv` - Test data for predictions (if available)

## Data Description

The dataset includes:

### Features
- **M1, M2, M3** - Market indicators
- **P1, P2** - Price-related features  
- **S1, S2** - Sector indicators
- **I1, I2** - Additional market information

### Target
- **market_forward_excess_returns** - Excess returns over risk-free rate (what we predict)

### Metadata
- **date_id** - Date identifier
- **forward_returns** - Raw forward returns
- **risk_free_rate** - Risk-free rate for the period

## Note

⚠️ **Data files are not included in this repository** due to:
- Large file sizes (GitHub limits)
- Competition rules (data redistribution)
- Privacy considerations

Please download the data yourself following the instructions above.
