"""
Model Training Module for Hull Tactical Market Prediction

This module contains functions for training ensemble models and making predictions.
Uses LightGBM, CatBoost, and XGBoost in an equal-weighted ensemble.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
import lightgbm as lgb
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


class EnsembleModel:
    """
    Ensemble of gradient boosting models for market return prediction.
    
    Combines LightGBM, CatBoost, and XGBoost with equal weighting.
    Conservative hyperparameters prevent overfitting on noisy financial data.
    
    Attributes
    ----------
    model_lgb : LGBMRegressor
        LightGBM model instance
    model_cat : CatBoostRegressor
        CatBoost model instance
    model_xgb : XGBRegressor
        XGBoost model instance
    scaler : StandardScaler
        Feature scaling transformer
    imputer : SimpleImputer
        Missing value imputation
    feature_cols : list
        List of feature column names
    """
    
    def __init__(self):
        """Initialize ensemble models with conservative hyperparameters."""
        # LightGBM - Fast and efficient
        self.model_lgb = lgb.LGBMRegressor(
            n_estimators=460,
            learning_rate=0.03,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1,
            n_jobs=-1
        )
        
        # CatBoost - Handles categorical features well
        self.model_cat = CatBoostRegressor(
            iterations=460,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3,
            random_state=42,
            verbose=0
        )
        
        # XGBoost - Robust to overfitting
        self.model_xgb = XGBRegressor(
            n_estimators=460,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_cols = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'EnsembleModel':
        """
        Train all models in the ensemble.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable (excess returns)
            
        Returns
        -------
        self : EnsembleModel
            Fitted ensemble model
        """
        self.feature_cols = X.columns.tolist()
        
        # Preprocess features
        X_scaled = self.scaler.fit_transform(X)
        X_filled = self.imputer.fit_transform(X_scaled)
        
        # Train all models
        print("Training LightGBM...")
        self.model_lgb.fit(X_filled, y)
        
        print("Training CatBoost...")
        self.model_cat.fit(X_filled, y)
        
        print("Training XGBoost...")
        self.model_xgb.fit(X_filled, y)
        
        print("✓ Ensemble training complete!")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using ensemble average.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
            
        Returns
        -------
        np.ndarray
            Predicted excess returns (ensemble average)
        """
        # Ensure features match training
        for col in self.feature_cols:
            if col not in X.columns:
                X[col] = 0
        
        X = X[self.feature_cols]
        
        # Preprocess
        X_scaled = self.scaler.transform(X)
        X_filled = self.imputer.transform(X_scaled)
        
        # Get predictions from each model
        pred_lgb = self.model_lgb.predict(X_filled)
        pred_cat = self.model_cat.predict(X_filled)
        pred_xgb = self.model_xgb.predict(X_filled)
        
        # Equal-weighted ensemble
        ensemble_pred = (pred_lgb + pred_cat + pred_xgb) / 3.0
        
        return ensemble_pred
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from LightGBM model.
        
        Parameters
        ----------
        top_n : int
            Number of top features to return
            
        Returns
        -------
        pd.DataFrame
            Feature importance dataframe
        """
        importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model_lgb.feature_importances_
        })
        
        return importance.sort_values('importance', ascending=False).head(top_n)


def train_ensemble(train_df: pd.DataFrame, 
                   target_col: str = 'market_forward_excess_returns',
                   exclude_cols: list = None) -> EnsembleModel:
    """
    Train ensemble model on training data.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataframe with features and target
    target_col : str
        Name of target column
    exclude_cols : list, optional
        Additional columns to exclude from features
        
    Returns
    -------
    EnsembleModel
        Trained ensemble model
        
    Examples
    --------
    >>> train_df = pd.read_csv('train.csv')
    >>> train_df = create_features(train_df)
    >>> model = train_ensemble(train_df)
    """
    if exclude_cols is None:
        exclude_cols = []
    
    # Define columns to exclude
    default_exclude = ['date_id', 'forward_returns', 'risk_free_rate', target_col]
    all_exclude = default_exclude + exclude_cols
    
    # Get feature columns
    feature_cols = [col for col in train_df.columns if col not in all_exclude]
    
    # Split features and target
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    
    # Remove rows with missing target
    mask = ~y_train.isna()
    X_train = X_train[mask]
    y_train = y_train[mask]
    
    print(f"Training on {len(X_train)} samples with {len(feature_cols)} features")
    
    # Train ensemble
    ensemble = EnsembleModel()
    ensemble.fit(X_train, y_train)
    
    return ensemble


def calculate_model_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics for model predictions.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
        
    Returns
    -------
    dict
        Dictionary of metric names and values
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Correlation
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Correlation': corr
    }
