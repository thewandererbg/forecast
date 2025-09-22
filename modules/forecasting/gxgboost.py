import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, cast
import logging
import warnings
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from .forecast import BaseForecaster
from ..data_preparation import MarketplaceKey, CampaignInfo
from ..validation import ValidationConfig, ForecastValidator
from ..campaign import SpikeDetector, apply_campaign_factors

logger = logging.getLogger(__name__)


class GlobalXGBoostForecaster(BaseForecaster):
    """Global XGBoost forecasting implementation with feature engineering and validation"""

    def __init__(
        self,
        forecast_horizon: int = 90,
        target_column: str = "sum_quantity",
        validation_config: Optional[ValidationConfig] = None,
    ):
        super().__init__(forecast_horizon, target_column, validation_config)

        # XGBoost parameters
        self.xgb_params = {
            "objective": "reg:squarederror",
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
        }

        # Feature engineering parameters
        self.lag_features = [1, 7, 14, 30]  # Days to lag
        self.rolling_windows = [7, 14, 30]  # Rolling window sizes

        # Initialize spike detector
        self.spike_detector = SpikeDetector()

        # Label encoders for categorical features
        self.label_encoders = {}

        # Global model (trained once on all data)
        self.global_model = None
        self.feature_cols = None
        self.is_trained = False

    def get_model_name(self) -> str:
        return "XGBoost-Global"

    def create_features(self, ts_data: pd.DataFrame, country: str, marketplace: str) -> pd.DataFrame:
        """Create features for XGBoost model"""
        df = ts_data.copy()
        df = df.reset_index(drop=True)

        # Ensure datetime index
        df["ds"] = pd.to_datetime(df["ds"])
        df = df.sort_values("ds").reset_index(drop=True)

        # Time-based features
        df["year"] = df["ds"].dt.year
        df["month"] = df["ds"].dt.month
        df["day"] = df["ds"].dt.day
        df["dayofweek"] = df["ds"].dt.dayofweek
        df["dayofyear"] = df["ds"].dt.dayofyear
        df["week"] = df["ds"].dt.isocalendar().week.astype("int64")
        df["quarter"] = df["ds"].dt.quarter
        df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
        df["is_month_start"] = df["ds"].dt.is_month_start.astype(int)
        df["is_month_end"] = df["ds"].dt.is_month_end.astype(int)

        # Categorical features
        df["country"] = country
        df["marketplace"] = marketplace

        # Market-specific lag and rolling features
        target_col = "y"
        for lag in self.lag_features:
            df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

        # Market-specific rolling statistics
        for window in self.rolling_windows:
            df[f"{target_col}_rolling_mean_{window}"] = df[target_col].rolling(window=window, min_periods=1).mean()
            df[f"{target_col}_rolling_std_{window}"] = df[target_col].rolling(window=window, min_periods=1).std()
            df[f"{target_col}_rolling_max_{window}"] = df[target_col].rolling(window=window, min_periods=1).max()
            df[f"{target_col}_rolling_min_{window}"] = df[target_col].rolling(window=window, min_periods=1).min()

        # Growth features
        df[f"{target_col}_pct_change_1"] = df[target_col].pct_change(1)
        df[f"{target_col}_pct_change_7"] = df[target_col].pct_change(7)
        df[f"{target_col}_diff_1"] = df[target_col].diff(1)
        df[f"{target_col}_diff_7"] = df[target_col].diff(7)

        # Cyclical encoding for time features
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
        df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
        df["dayofyear_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
        df["dayofyear_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)

        return df

    def prepare_all_market_data(self, all_market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Prepare and combine data from all markets for global training"""
        logger.info("Preparing global training dataset from all markets")

        all_features = []

        for market_key, ts_data in all_market_data.items():
            if len(ts_data) == 0:
                logger.warning(f"No data for {market_key} - skipping")
                continue

            # Parse market key (assuming format "country-marketplace")
            try:
                country, marketplace = market_key.split("-", 1)
            except ValueError:
                logger.warning(f"Invalid market key format: {market_key} - skipping")
                continue

            # Detect spikes and create dampened series
            spike_indices, _ = self.spike_detector.detect_spikes_adaptive(ts_data, market_key)
            dampened_data = self.spike_detector.create_spike_dampened_series(ts_data, spike_indices)

            # Create features for this market
            market_features = self.create_features(dampened_data, country, marketplace)
            market_features["market_key"] = market_key  # Add market identifier

            all_features.append(market_features)

        if not all_features:
            logger.error("No valid market data found for global training")
            return pd.DataFrame()

        # Combine all market data
        combined_data = pd.concat(all_features, ignore_index=True)
        logger.info(f"Combined dataset shape: {combined_data.shape}")
        logger.info(f"Markets included: {combined_data['market_key'].nunique()}")

        return combined_data

    def prepare_training_data(self, df: pd.DataFrame) -> tuple:
        """Prepare features and target for training"""
        # Remove rows with NaN values (due to lag features)
        df_clean = df.dropna().copy()

        # Clean infinite values and extreme outliers
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        df_clean = df_clean.dropna()

        # Cap extreme values to prevent numerical issues
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != "y":  # Don't cap the target variable
                q99 = df_clean[col].quantile(0.99)
                q01 = df_clean[col].quantile(0.01)
                df_clean[col] = df_clean[col].clip(lower=q01, upper=q99)

        # Feature columns (exclude ds, y, and market_key)
        exclude_cols = ["ds", "y", "market_key"]
        feature_cols = [col for col in df_clean.columns if col not in exclude_cols]

        # Encode categorical features
        for col in ["country", "marketplace"]:
            if col in feature_cols:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_clean[col] = self.label_encoders[col].fit_transform(df_clean[col])
                else:
                    # Handle unseen categories
                    unique_values = self.label_encoders[col].classes_
                    df_clean[col] = df_clean[col].map(
                        lambda x: self.label_encoders[col].transform([x])[0] if x in unique_values else -1
                    )

        X = df_clean[feature_cols]
        y = df_clean["y"]

        return X, y, feature_cols

    def train_global_model(self, all_market_data: Dict[str, pd.DataFrame]):
        """Train one global model on all market data"""
        logger.info("Training global XGBoost model on all markets")

        # Prepare combined dataset
        combined_data = self.prepare_all_market_data(all_market_data)

        if len(combined_data) == 0:
            logger.error("No training data available for global model")
            return

        # Prepare training data
        X_train, y_train, feature_cols = self.prepare_training_data(combined_data)

        if len(X_train) == 0:
            logger.error("No training data after feature engineering")
            return

        # Store feature columns for later use
        self.feature_cols = feature_cols

        # Train global model
        logger.info(f"Training on {len(X_train)} samples with {len(feature_cols)} features")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.global_model = xgb.XGBRegressor(**self.xgb_params)
            self.global_model.fit(X_train, y_train)

        self.is_trained = True
        logger.info("Global model training completed")

    def create_future_features(
        self, last_date: pd.Timestamp, country: str, marketplace: str, historical_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Create features for future dates"""
        # Generate future dates
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=self.forecast_horizon, freq="D")

        # Create base future dataframe
        future_df = pd.DataFrame({"ds": future_dates})

        # Add time-based features
        future_df["year"] = future_df["ds"].dt.year
        future_df["month"] = future_df["ds"].dt.month
        future_df["day"] = future_df["ds"].dt.day
        future_df["dayofweek"] = future_df["ds"].dt.dayofweek
        future_df["dayofyear"] = future_df["ds"].dt.dayofyear
        future_df["week"] = future_df["ds"].dt.isocalendar().week.astype("int64")
        future_df["quarter"] = future_df["ds"].dt.quarter
        future_df["is_weekend"] = (future_df["dayofweek"] >= 5).astype(int)
        future_df["is_month_start"] = future_df["ds"].dt.is_month_start.astype(int)
        future_df["is_month_end"] = future_df["ds"].dt.is_month_end.astype(int)

        # Categorical features
        future_df["country"] = country
        future_df["marketplace"] = marketplace

        # Cyclical encoding
        future_df["month_sin"] = np.sin(2 * np.pi * future_df["month"] / 12)
        future_df["month_cos"] = np.cos(2 * np.pi * future_df["month"] / 12)
        future_df["dayofweek_sin"] = np.sin(2 * np.pi * future_df["dayofweek"] / 7)
        future_df["dayofweek_cos"] = np.cos(2 * np.pi * future_df["dayofweek"] / 7)
        future_df["dayofyear_sin"] = np.sin(2 * np.pi * future_df["dayofyear"] / 365)
        future_df["dayofyear_cos"] = np.cos(2 * np.pi * future_df["dayofyear"] / 365)

        # For lag and rolling features, use historical data from this specific market
        last_values = historical_data["y"].tail(max(self.lag_features + self.rolling_windows))

        # Initialize lag features with last known values
        for lag in self.lag_features:
            if lag <= len(last_values):
                future_df[f"y_lag_{lag}"] = last_values.iloc[-lag] if lag <= len(last_values) else last_values.mean()
            else:
                future_df[f"y_lag_{lag}"] = last_values.mean()

        # Initialize rolling features with last known statistics
        for window in self.rolling_windows:
            window_data = last_values.tail(window) if len(last_values) >= window else last_values
            future_df[f"y_rolling_mean_{window}"] = window_data.mean()
            future_df[f"y_rolling_std_{window}"] = window_data.std() if len(window_data) > 1 else 0
            future_df[f"y_rolling_max_{window}"] = window_data.max()
            future_df[f"y_rolling_min_{window}"] = window_data.min()

        # Initialize growth features with recent trends
        future_df[f"y_pct_change_1"] = 0  # Assume no change initially
        future_df[f"y_pct_change_7"] = 0
        future_df[f"y_diff_1"] = 0
        future_df[f"y_diff_7"] = 0

        return future_df

    def calculate_confidence_intervals(self, model, X_train, y_train, predictions):
        """Calculate confidence intervals for predictions"""
        error_rate = 0.6
        ci = error_rate * predictions

        # Simple percentage-based intervals
        lower_bounds = predictions - ci
        upper_bounds = predictions + ci

        return np.maximum(lower_bounds, 0), upper_bounds

    def forecast_marketplace(
        self,
        country: str,
        marketplace: str,
        ts_data: pd.DataFrame,
        campaign_calendar: Dict[str, CampaignInfo],
        _skip_validation: bool = False,
    ) -> Dict[str, Union[pd.DataFrame, Dict[str, float], None]]:
        """Generate forecast for a marketplace using the global XGBoost model"""
        logger.debug(f"Forecasting {country}-{marketplace} (Global XGBoost + Campaigns)")

        if not self.is_trained or self.global_model is None:
            logger.error("Global model is not trained. Call train_global_model() first.")
            return {"forecast": pd.DataFrame(), "validation_metrics": None}

        if len(ts_data) == 0:
            logger.warning(f"  No data for {country}-{marketplace} - skipping")
            return {"forecast": pd.DataFrame(), "validation_metrics": None}

        # Perform validation using the generic validator
        validation_metrics = None
        if not _skip_validation:
            validation_metrics = ForecastValidator.validate_forecast_model(
                ts_data=ts_data,
                config=self.validation_config,
                forecaster_instance=self,
                country=country,
                marketplace=marketplace,
                campaign_calendar=campaign_calendar,
            )

        # Detect spikes and create dampened series for main forecast
        spike_indices, _ = self.spike_detector.detect_spikes_adaptive(ts_data, f"{country}-{marketplace}")
        dampened_data = self.spike_detector.create_spike_dampened_series(ts_data, spike_indices)

        # Create future features
        last_date = cast(pd.Timestamp, ts_data["ds"].max())
        future_features = self.create_future_features(last_date, country, marketplace, dampened_data)

        # Encode categorical features in future data using existing encoders
        for col in ["country", "marketplace"]:
            if col in self.label_encoders:
                unique_values = self.label_encoders[col].classes_
                future_features[col] = future_features[col].map(
                    lambda x: self.label_encoders[col].transform([x])[0] if x in unique_values else -1
                )

        # Ensure same feature order as training
        X_future = cast(pd.DataFrame, future_features[self.feature_cols])

        # Make predictions using global model
        logger.debug(f"  Using global XGBoost model for prediction")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            predictions = self.global_model.predict(X_future)
            predictions = np.maximum(predictions, 0)

        # Calculate confidence intervals (using dummy training data for now)
        # In practice, you might want to store training residuals during global training
        lower_bounds, upper_bounds = self.calculate_confidence_intervals(self.global_model, None, None, predictions)

        # Create forecast dataframe
        forecast_start = pd.to_datetime(self.forecast_start_date)
        forecast_dates = pd.date_range(start=forecast_start, periods=self.forecast_horizon, freq="D")

        forecast_period = pd.DataFrame(
            {
                "ds": forecast_dates,
                "yhat": predictions,
                "yhat_lower": lower_bounds,
                "yhat_upper": upper_bounds,
            }
        )

        # Apply additional campaign factors
        forecast_period = apply_campaign_factors(
            forecast_period=forecast_period,
            campaign_calendar=campaign_calendar,
            country=country,
            marketplace=marketplace,
        )

        # Prepare output
        result = pd.DataFrame(
            {
                "ds": forecast_period["ds"],
                "y_pred": forecast_period["yhat"].clip(lower=0),
                "y_pred_lower": forecast_period["yhat_lower"].clip(lower=0),
                "y_pred_upper": forecast_period["yhat_upper"].clip(lower=0),
                "country": country,
                "marketplace": marketplace,
            }
        )

        logger.debug(f"  Global XGBoost forecast total: {result['y_pred'].sum():.0f}")
        logger.debug(
            f"  Confidence interval width (avg): {(result['y_pred_upper'] - result['y_pred_lower']).mean():.1f}"
        )

        return {"forecast": result, "validation_metrics": validation_metrics}

    def supports_global_training(self) -> bool:
        """This forecaster supports global training"""
        return True
