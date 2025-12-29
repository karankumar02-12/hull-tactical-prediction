"""
Utility Functions for Hull Tactical Market Prediction

This module contains helper functions for data loading, preprocessing,
signal generation, and evaluation.
"""

import numpy as np
import pandas as pd
import polars as pl
from typing import Tuple, Optional


def load_data(filepath: str, use_polars: bool = False) -> pd.DataFrame:
    """
    Load competition data from CSV file.
    
    Parameters
    ----------
    filepath : str
        Path to CSV file
    use_polars : bool
        Whether to use Polars for faster loading
        
    Returns
    -------
    pd.DataFrame
        Loaded dataframe
    """
    if use_polars:
        df_pl = pl.read_csv(filepath)
        return df_pl.to_pandas()
    else:
        return pd.read_csv(filepath)


def clip_signal(signal: float, min_val: float = 0.0, max_val: float = 2.0) -> float:
    """
    Clip position signal to valid range.
    
    Competition rules:
    - Minimum position: 0.0 (no shorting)
    - Maximum position: 2.0 (2x leverage)
    
    Parameters
    ----------
    signal : float
        Raw position signal
    min_val : float
        Minimum allowed value
    max_val : float
        Maximum allowed value
        
    Returns
    -------
    float
        Clipped signal
    """
    return float(np.clip(signal, min_val, max_val))


def generate_signal(raw_prediction: float,
                   adaptive_multiplier: float,
                   min_signal: float = 0.0,
                   max_signal: float = 2.0) -> float:
    """
    Generate final position signal from raw prediction and multiplier.
    
    Formula: signal = raw_prediction * adaptive_multiplier + 1.0
    
    Parameters
    ----------
    raw_prediction : float
        Model's raw excess return prediction
    adaptive_multiplier : float
        Regime and volatility-adjusted multiplier
    min_signal : float
        Minimum position (default: 0.0)
    max_signal : float
        Maximum position (default: 2.0)
        
    Returns
    -------
    float
        Position signal between min_signal and max_signal
        
    Examples
    --------
    >>> pred = 0.0005
    >>> mult = 1200.0
    >>> signal = generate_signal(pred, mult)
    >>> print(f"Position: {signal:.2f}x")
    """
    # Convert prediction to position
    signal = raw_prediction * adaptive_multiplier + 1.0
    
    # Clip to valid range
    signal = clip_signal(signal, min_signal, max_signal)
    
    return signal


def calculate_returns(signals: np.ndarray, 
                     actual_returns: np.ndarray) -> np.ndarray:
    """
    Calculate strategy returns given signals and actual market returns.
    
    Parameters
    ----------
    signals : np.ndarray
        Array of position signals (0-2x)
    actual_returns : np.ndarray
        Array of actual market excess returns
        
    Returns
    -------
    np.ndarray
        Strategy returns
    """
    return signals * actual_returns


def calculate_sharpe_ratio(returns: np.ndarray, 
                          annualization_factor: float = 252) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of strategy returns
    annualization_factor : float
        Number of periods per year (252 for daily data)
        
    Returns
    -------
    float
        Annualized Sharpe ratio
    """
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    if std_return == 0:
        return 0.0
    
    sharpe = (mean_return / std_return) * np.sqrt(annualization_factor)
    return sharpe


def calculate_max_drawdown(cumulative_returns: np.ndarray) -> float:
    """
    Calculate maximum drawdown from cumulative returns.
    
    Parameters
    ----------
    cumulative_returns : np.ndarray
        Cumulative return series
        
    Returns
    -------
    float
        Maximum drawdown (negative value)
    """
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    max_dd = np.min(drawdown)
    
    return max_dd


def calculate_volatility(returns: np.ndarray, 
                        annualization_factor: float = 252) -> float:
    """
    Calculate annualized volatility.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of returns
    annualization_factor : float
        Number of periods per year
        
    Returns
    -------
    float
        Annualized volatility
    """
    return np.std(returns) * np.sqrt(annualization_factor)


def get_performance_summary(signals: np.ndarray,
                           actual_returns: np.ndarray) -> dict:
    """
    Calculate comprehensive performance metrics.
    
    Parameters
    ----------
    signals : np.ndarray
        Position signals
    actual_returns : np.ndarray
        Actual market returns
        
    Returns
    -------
    dict
        Dictionary of performance metrics
    """
    strategy_returns = calculate_returns(signals, actual_returns)
    cumulative_returns = np.cumprod(1 + strategy_returns)
    
    metrics = {
        'total_return': cumulative_returns[-1] - 1,
        'annualized_return': np.mean(strategy_returns) * 252,
        'annualized_volatility': calculate_volatility(strategy_returns),
        'sharpe_ratio': calculate_sharpe_ratio(strategy_returns),
        'max_drawdown': calculate_max_drawdown(cumulative_returns),
        'avg_position': np.mean(signals),
        'max_position': np.max(signals),
        'min_position': np.min(signals),
    }
    
    return metrics


def print_performance_summary(metrics: dict) -> None:
    """
    Pretty print performance metrics.
    
    Parameters
    ----------
    metrics : dict
        Performance metrics dictionary
    """
    print("=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Total Return:         {metrics['total_return']:>10.2%}")
    print(f"Annualized Return:    {metrics['annualized_return']:>10.2%}")
    print(f"Annualized Volatility:{metrics['annualized_volatility']:>10.2%}")
    print(f"Sharpe Ratio:         {metrics['sharpe_ratio']:>10.2f}")
    print(f"Max Drawdown:         {metrics['max_drawdown']:>10.2%}")
    print("-" * 60)
    print(f"Avg Position:         {metrics['avg_position']:>10.2f}x")
    print(f"Max Position:         {metrics['max_position']:>10.2f}x")
    print(f"Min Position:         {metrics['min_position']:>10.2f}x")
    print("=" * 60)


def validate_data(df: pd.DataFrame, required_cols: list) -> bool:
    """
    Validate that dataframe contains required columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    required_cols : list
        List of required column names
        
    Returns
    -------
    bool
        True if valid, False otherwise
    """
    missing_cols = set(required_cols) - set(df.columns)
    
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        return False
    
    return True


def get_recent_history(df: pd.DataFrame, 
                       target_col: str,
                       n_days: int = 60) -> np.ndarray:
    """
    Get recent historical returns for regime detection.
    
    Parameters
    ----------
    df : pd.DataFrame
        Historical data
    target_col : str
        Column name for returns
    n_days : int
        Number of recent days to retrieve
        
    Returns
    -------
    np.ndarray
        Recent return history
    """
    return df[target_col].tail(n_days).values
