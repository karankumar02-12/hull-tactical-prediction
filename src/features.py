"""
Feature Engineering Module for Hull Tactical Market Prediction

This module contains functions to create technical and momentum-based features
from raw market data for tactical asset allocation.
"""

import pandas as pd
import numpy as np


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create comprehensive feature set from raw market data.
    
    Features include:
    - Lagged values (1, 2, 5 days)
    - Rolling statistics (mean, std)
    - Momentum indicators
    - Technical indicators (RSI, MACD, Bollinger Bands)
    - Volatility features
    - Cross-sectional ratios
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with columns: M1, M2, M3, P1, P2, S1, S2, I1, I2
        
    Returns
    -------
    pd.DataFrame
        DataFrame with original columns plus 60+ engineered features
        
    Examples
    --------
    >>> train_df = pd.read_csv('train.csv')
    >>> train_df = create_features(train_df)
    >>> print(f"Features created: {len(train_df.columns)}")
    """
    df = df.copy()
    
    # Key columns from competition data
    key_cols = ['M1', 'M2', 'M3', 'P1', 'P2', 'S1', 'S2', 'I1', 'I2']
    
    # ========================================================================
    # 1. LAGGED FEATURES - Capture temporal dependencies
    # ========================================================================
    for col in key_cols:
        if col in df.columns:
            df[f'{col}_lag1'] = df[col].shift(1)
            df[f'{col}_lag2'] = df[col].shift(2)
            df[f'{col}_lag5'] = df[col].shift(5)
    
    # ========================================================================
    # 2. ROLLING STATISTICS - Smooth out noise
    # ========================================================================
    for col in ['M1', 'M2', 'P1', 'S1']:
        if col in df.columns:
            df[f'{col}_roll5_mean'] = df[col].rolling(5).mean()
            df[f'{col}_roll10_mean'] = df[col].rolling(10).mean()
            df[f'{col}_roll20_mean'] = df[col].rolling(20).mean()
            df[f'{col}_roll5_std'] = df[col].rolling(5).std()
            df[f'{col}_roll10_std'] = df[col].rolling(10).std()
    
    # ========================================================================
    # 3. MOMENTUM FEATURES - Capture trends
    # ========================================================================
    if 'M1' in df.columns and 'M2' in df.columns:
        df['momentum_M'] = df['M1'] - df['M2']
        df['M1_M2_ratio'] = df['M1'] / (df['M2'].abs() + 0.001)
    
    if 'P1' in df.columns and 'P2' in df.columns:
        df['momentum_P'] = df['P1'] - df['P2']
        df['P1_P2_ratio'] = df['P1'] / (df['P2'].abs() + 0.001)
    
    # ========================================================================
    # 4. VOLATILITY FEATURES - Measure risk/uncertainty
    # ========================================================================
    for col in ['M1', 'P1', 'S1']:
        if col in df.columns:
            df[f'{col}_vol10'] = df[col].rolling(10).std()
            df[f'{col}_vol20'] = df[col].rolling(20).std()
    
    # ========================================================================
    # 5. RSI (Relative Strength Index) - Overbought/oversold indicator
    # ========================================================================
    if 'M1' in df.columns:
        delta = df['M1'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / (loss + 0.001)
        df['M1_rsi'] = 100 - (100 / (1 + rs))
    
    # ========================================================================
    # 6. CROSS-SECTIONAL RATIOS - Inter-feature relationships
    # ========================================================================
    if 'M1' in df.columns and 'P1' in df.columns:
        df['M1_P1_ratio'] = df['M1'] / (df['P1'].abs() + 0.001)
    
    # ========================================================================
    # 7. MACD (Moving Average Convergence Divergence) - Trend strength
    # ========================================================================
    if 'M1' in df.columns:
        ema12 = df['M1'].ewm(span=12, adjust=False).mean()
        ema26 = df['M1'].ewm(span=26, adjust=False).mean()
        df['M1_macd'] = ema12 - ema26
    
    # ========================================================================
    # 8. ROC (Rate of Change) - Momentum velocity
    # ========================================================================
    if 'M1' in df.columns:
        df['M1_roc10'] = (df['M1'] - df['M1'].shift(10)) / (df['M1'].shift(10).abs() + 0.001)
    
    # ========================================================================
    # 9. BOLLINGER BANDS POSITION - Price relative to bands
    # ========================================================================
    if 'P1' in df.columns:
        ma20 = df['P1'].rolling(20).mean()
        std20 = df['P1'].rolling(20).std()
        df['P1_bb_pos'] = (df['P1'] - ma20) / (std20 + 0.001)
    
    # ========================================================================
    # 10. EMA CROSSOVER - Fast vs slow moving average
    # ========================================================================
    if 'M1' in df.columns:
        ema5 = df['M1'].ewm(span=5, adjust=False).mean()
        ema20 = df['M1'].ewm(span=20, adjust=False).mean()
        df['M1_ema_cross'] = (ema5 - ema20) / (ema20.abs() + 0.001)
    
    # ========================================================================
    # 11. VOLATILITY REGIME - Short vs long term volatility
    # ========================================================================
    if 'M1' in df.columns:
        vol5 = df['M1'].rolling(5).std()
        vol20 = df['M1'].rolling(20).std()
        df['M1_vol_regime'] = vol5 / (vol20 + 0.001)
    
    # ========================================================================
    # 12. MOMENTUM ACCELERATION - Second derivative of price
    # ========================================================================
    if 'P1' in df.columns:
        mom = df['P1'].diff()
        df['P1_mom_accel'] = mom.diff()
    
    return df


def get_feature_names(exclude_cols: list = None) -> list:
    """
    Get list of feature column names (excluding target and metadata).
    
    Parameters
    ----------
    exclude_cols : list, optional
        Additional columns to exclude
        
    Returns
    -------
    list
        List of feature column names
    """
    if exclude_cols is None:
        exclude_cols = []
    
    default_exclude = [
        'date_id', 
        'forward_returns', 
        'risk_free_rate', 
        'market_forward_excess_returns'
    ]
    
    return default_exclude + exclude_cols


def validate_features(df: pd.DataFrame, feature_cols: list) -> bool:
    """
    Validate that all required features exist in dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    feature_cols : list
        List of required feature names
        
    Returns
    -------
    bool
        True if all features exist, False otherwise
    """
    missing_features = set(feature_cols) - set(df.columns)
    
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        return False
    
    return True
