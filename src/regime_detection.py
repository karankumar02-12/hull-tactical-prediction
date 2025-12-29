"""
Market Regime Detection Module for Hull Tactical Market Prediction

This module identifies market regimes (Bull/Bear/Neutral) to adapt
position sizing and risk management strategies.
"""

import numpy as np
from typing import Literal


def detect_regime(returns_history: np.ndarray) -> Literal['bull', 'bear', 'neutral']:
    """
    Detect current market regime based on recent return history.
    
    Uses a combination of:
    - Mean return over last 30 days
    - Short-term (10-day) vs long-term (30-day) moving average crossover
    
    Parameters
    ----------
    returns_history : np.ndarray
        Array of historical excess returns (recommended: last 60 days)
        
    Returns
    -------
    str
        Market regime: 'bull', 'bear', or 'neutral'
        
    Examples
    --------
    >>> returns = np.array([0.001, 0.002, -0.001, 0.003, ...])
    >>> regime = detect_regime(returns)
    >>> print(f"Current regime: {regime}")
    
    Notes
    -----
    Bull Market Criteria:
    - Mean return > 0.0005 (0.05%)
    - Short MA > Long MA (upward momentum)
    
    Bear Market Criteria:
    - Mean return < -0.0005 (-0.05%)
    - Short MA < Long MA (downward momentum)
    
    Neutral Market:
    - Everything else (sideways/choppy)
    """
    # Focus on recent 30 days for regime assessment
    recent_returns = returns_history[-30:] if len(returns_history) >= 30 else returns_history
    
    # Calculate key metrics
    mean_return = np.mean(recent_returns)
    short_ma = np.mean(returns_history[-10:]) if len(returns_history) >= 10 else np.mean(returns_history)
    long_ma = np.mean(returns_history[-30:]) if len(returns_history) >= 30 else np.mean(returns_history)
    
    # Regime detection logic
    if mean_return > 0.0005 and short_ma > long_ma:
        return 'bull'
    elif mean_return < -0.0005 and short_ma < long_ma:
        return 'bear'
    else:
        return 'neutral'


def get_base_multiplier(regime: str) -> float:
    """
    Get base position multiplier for given market regime.
    
    Parameters
    ----------
    regime : str
        Market regime ('bull', 'bear', or 'neutral')
        
    Returns
    -------
    float
        Base multiplier for position sizing
        
    Examples
    --------
    >>> regime = detect_regime(returns_history)
    >>> multiplier = get_base_multiplier(regime)
    >>> print(f"{regime} market → {multiplier}x base position")
    
    Notes
    -----
    Multiplier Strategy:
    - Bull: 1280x (aggressive - capture upside)
    - Bear: 1020x (defensive - limit downside)
    - Neutral: 1160x (moderate - balanced approach)
    """
    regime_multipliers = {
        'bull': 1280.0,    # Aggressive positioning
        'bear': 1020.0,    # Conservative positioning
        'neutral': 1160.0  # Moderate positioning
    }
    
    return regime_multipliers.get(regime, 1160.0)


def calculate_volatility_scalar(recent_returns: np.ndarray, 
                                target_volatility: float = 0.0120) -> float:
    """
    Calculate volatility-based position scaling factor.
    
    Adjusts position size to target a specific volatility level,
    implementing basic volatility targeting for risk management.
    
    Parameters
    ----------
    recent_returns : np.ndarray
        Recent return history (recommended: last 20 days)
    target_volatility : float
        Target volatility level (default: 1.2% or 0.0120)
        
    Returns
    -------
    float
        Volatility scalar (clipped between 0.5 and 1.5)
        
    Examples
    --------
    >>> recent = returns_history[-20:]
    >>> scalar = calculate_volatility_scalar(recent)
    >>> adjusted_mult = base_multiplier * scalar
    
    Notes
    -----
    - If recent volatility > target → scale DOWN (scalar < 1.0)
    - If recent volatility < target → scale UP (scalar > 1.0)
    - Clipped to [0.5, 1.5] to prevent extreme adjustments
    """
    # Calculate recent volatility
    recent_vol = np.std(recent_returns) if len(recent_returns) > 0 else 0.01
    
    # Volatility targeting: scale inversely with realized vol
    vol_scalar = target_volatility / (recent_vol + 0.001)
    
    # Clip to reasonable range
    vol_scalar = np.clip(vol_scalar, 0.5, 1.5)
    
    return vol_scalar


def calculate_adaptive_multiplier(returns_history: np.ndarray,
                                  target_volatility: float = 0.0120) -> float:
    """
    Calculate adaptive position multiplier combining regime and volatility.
    
    This is the main function that combines:
    1. Regime detection (bull/bear/neutral)
    2. Base multiplier selection
    3. Volatility-based scaling
    
    Parameters
    ----------
    returns_history : np.ndarray
        Historical excess returns (recommended: last 60 days)
    target_volatility : float
        Target volatility for position sizing
        
    Returns
    -------
    float
        Adaptive multiplier for position sizing
        
    Examples
    --------
    >>> returns = get_recent_returns(60)
    >>> multiplier = calculate_adaptive_multiplier(returns)
    >>> position = raw_prediction * multiplier + 1.0
    """
    # Step 1: Detect current regime
    regime = detect_regime(returns_history)
    
    # Step 2: Get base multiplier for regime
    base_mult = get_base_multiplier(regime)
    
    # Step 3: Calculate volatility scalar
    recent_returns = returns_history[-20:] if len(returns_history) >= 20 else returns_history
    vol_scalar = calculate_volatility_scalar(recent_returns, target_volatility)
    
    # Step 4: Combine regime and volatility adjustments
    adaptive_mult = base_mult * vol_scalar
    
    return adaptive_mult


def get_regime_stats(returns_history: np.ndarray) -> dict:
    """
    Get detailed statistics about current market regime.
    
    Useful for debugging and understanding regime detection.
    
    Parameters
    ----------
    returns_history : np.ndarray
        Historical returns
        
    Returns
    -------
    dict
        Dictionary with regime statistics
    """
    regime = detect_regime(returns_history)
    base_mult = get_base_multiplier(regime)
    
    recent_returns = returns_history[-30:] if len(returns_history) >= 30 else returns_history
    mean_return = np.mean(recent_returns)
    volatility = np.std(recent_returns)
    
    short_ma = np.mean(returns_history[-10:]) if len(returns_history) >= 10 else np.mean(returns_history)
    long_ma = np.mean(returns_history[-30:]) if len(returns_history) >= 30 else np.mean(returns_history)
    
    return {
        'regime': regime,
        'base_multiplier': base_mult,
        'mean_return_30d': mean_return,
        'volatility_30d': volatility,
        'short_ma': short_ma,
        'long_ma': long_ma,
        'ma_cross': 'bullish' if short_ma > long_ma else 'bearish'
    }
