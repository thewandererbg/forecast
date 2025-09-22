import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, cast
from dataclasses import dataclass
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)

from .forecast import BaseForecaster, ValidationConfig, ForecastValidator
from ..types import CampaignInfo
from ..campaign import SpikeDetector


@dataclass
class KNNFeatures:
    """Feature configuration for k-NN forecasting"""

    day_of_week: bool = True
    month: bool = True
    month_day: bool = True
    week_of_year: bool = True
    is_campaign_day: bool = True
    campaign_intensity: bool = True
    rolling_avg_7d: bool = True
    lag_7: bool = True


class KNNForecaster(BaseForecaster):
    """Fixed k-NN forecasting with campaign awareness"""

    def __init__(
        self,
        forecast_horizon: int = 90,
        target_column: str = "sum_quantity",
        validation_config: Optional[ValidationConfig] = None,
        k_neighbors: int = 7,
        feature_config: Optional[KNNFeatures] = None,
        feature_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__(forecast_horizon, target_column, validation_config)
        self.k_neighbors = k_neighbors
        self.feature_config = feature_config or KNNFeatures()

        # Feature weights emphasizing temporal matching
        self.feature_weights = feature_weights or {
            "day_of_week": 1.0,
            "month": 1.2,
            "month_day": 10.0,  # VERY HIGH weight for exact date matching (Nov 11 = Nov 11)
            "week_of_year": 2.0,  # Medium weight for same week matching
            "is_campaign_day": 8.0,  # Very high for campaign matching
            "campaign_intensity": 3.0,  # High for similar campaign intensity
            "rolling_avg_7d": 1.0,  # Lower weight for trends
            "lag_7": 1.0,  # Lower weight for lags
        }

        self.scaler = StandardScaler()
        self.spike_detector = SpikeDetector()

    def get_model_name(self) -> str:
        return f"k-NN (k={self.k_neighbors})"

    def _add_campaign_features(self, df: pd.DataFrame, campaign_calendar: Dict[str, CampaignInfo]) -> pd.DataFrame:
        """Add campaign features - simple month-day matching"""
        df = df.copy()
        df["ds"] = pd.to_datetime(df["ds"])

        # Create month-day string for matching
        df["month_day"] = df["ds"].dt.strftime("%m-%d")

        # Initialize campaign features
        df["is_campaign_day"] = 0
        df["campaign_intensity"] = 0.0

        # Mark campaign days
        for month_day, campaign_info in campaign_calendar.items():
            mask = df["month_day"] == month_day
            df.loc[mask, "is_campaign_day"] = 1
            df.loc[mask, "campaign_intensity"] = campaign_info["avg_uplift"]

        return df.drop("month_day", axis=1)

    def _build_features(self, ts_data: pd.DataFrame, campaign_calendar: Dict[str, CampaignInfo]) -> pd.DataFrame:
        """Build feature matrix with proper year-over-year temporal features"""
        df = ts_data.copy()
        df["ds"] = pd.to_datetime(df["ds"])

        # Add campaign features
        df = self._add_campaign_features(df, campaign_calendar)

        # Add temporal features with weights applied during construction
        if self.feature_config.day_of_week:
            df["day_of_week"] = df["ds"].dt.dayofweek * self.feature_weights.get("day_of_week", 1.0)

        if self.feature_config.month:
            df["month"] = df["ds"].dt.month * self.feature_weights.get("month", 1.0)

        # NEW: Year-over-year features for better seasonal matching
        if self.feature_config.month_day:
            # Create leap-year safe month-day key: Nov 11 = 1111 always
            month_day_key = df["ds"].dt.month * 100 + df["ds"].dt.day
            # Normalize by max possible value (Dec 31 = 1231)
            df["month_day"] = (month_day_key / 1231.0) * self.feature_weights.get("month_day", 1.0)

        if self.feature_config.week_of_year:
            # Convert to week of year (1-52), normalize by dividing by 52
            week_of_year = df["ds"].dt.isocalendar().week / 52.0
            df["week_of_year"] = week_of_year * self.feature_weights.get("week_of_year", 1.0)

        # Apply weights to campaign features
        if self.feature_config.is_campaign_day:
            df["is_campaign_day"] = df["is_campaign_day"] * self.feature_weights.get("is_campaign_day", 1.0)
        if self.feature_config.campaign_intensity:
            df["campaign_intensity"] = df["campaign_intensity"] * self.feature_weights.get("campaign_intensity", 1.0)

        # Add trend features with weights
        if self.feature_config.rolling_avg_7d:
            df["rolling_avg_7d"] = df["y"].rolling(window=7, min_periods=1).mean() * self.feature_weights.get(
                "rolling_avg_7d", 1.0
            )

        # Add lag features with weights
        if self.feature_config.lag_7:
            df["lag_7"] = df["y"].shift(7) * self.feature_weights.get("lag_7", 1.0)

        # Drop NaN rows
        df = df.dropna()
        return df

    def _get_feature_columns(self) -> List[str]:
        """Get feature column names"""
        features = []
        if self.feature_config.day_of_week:
            features.append("day_of_week")
        if self.feature_config.month:
            features.append("month")
        if self.feature_config.month_day:
            features.append("month_day")
        if self.feature_config.week_of_year:
            features.append("week_of_year")
        if self.feature_config.is_campaign_day:
            features.append("is_campaign_day")
        if self.feature_config.campaign_intensity:
            features.append("campaign_intensity")
        if self.feature_config.rolling_avg_7d:
            features.append("rolling_avg_7d")
        if self.feature_config.lag_7:
            features.append("lag_7")
        return features

    def _calculate_confidence_intervals(
        self, prediction: float, neighbor_values: np.ndarray, weights: np.ndarray
    ) -> Tuple[float, float]:
        """Calculate confidence intervals based on neighbor variance"""
        if len(neighbor_values) < 2:
            return prediction * 0.9, prediction * 1.1

        # Weighted variance calculation
        weighted_mean = np.sum(neighbor_values * weights)
        weighted_var = np.sum(weights * (neighbor_values - weighted_mean) ** 2)
        weighted_std = np.sqrt(weighted_var)

        # Use 1.96 * std for ~95% confidence interval
        margin = 1.96 * weighted_std
        return max(0, prediction - margin), prediction + margin

    def forecast_marketplace(
        self,
        country: str,
        marketplace: str,
        ts_data: pd.DataFrame,
        campaign_calendar: Dict[str, CampaignInfo],
        _skip_validation: bool = False,
    ) -> Dict[str, Union[pd.DataFrame, Dict[str, float], None]]:
        """Generate k-NN forecast with fixed logic"""
        logger.debug(f"Forecasting {country}-{marketplace} (k-NN)")

        if len(ts_data) == 0:
            return {"forecast": pd.DataFrame(), "validation_metrics": None}

        # Validation
        validation_metrics = None
        if not _skip_validation:
            validation_metrics = ForecastValidator.validate_forecast_model(
                ts_data, self.validation_config, self, country, marketplace, campaign_calendar
            )

        # Build features
        feature_data = self._build_features(ts_data, campaign_calendar)
        feature_cols = self._get_feature_columns()

        # More flexible data requirement
        min_required = max(3, min(self.k_neighbors, len(feature_data) // 2))
        if len(feature_data) < min_required:
            logger.warning(f"Insufficient data for k-NN: {len(feature_data)} < {min_required}")
            return {"forecast": pd.DataFrame(), "validation_metrics": validation_metrics}

        # Prepare training data - StandardScaler handles the weighted features
        X_train = self.scaler.fit_transform(feature_data[feature_cols])
        y_train = feature_data["y"].values

        # Generate future dates and features
        last_date = pd.to_datetime(feature_data["ds"].iloc[-1])
        future_dates = pd.date_range(last_date + timedelta(days=1), periods=self.forecast_horizon, freq="D")

        forecasts = []

        # Initialize prediction history for lag features
        prediction_history = list(feature_data["y"].tail(14).values)  # Keep 14 days for safety
        rolling_window = list(feature_data["y"].tail(7).values)  # Keep 7 days for rolling avg

        for i, future_date in enumerate(future_dates):
            # Build future features
            future_features = {}

            if self.feature_config.day_of_week:
                future_features["day_of_week"] = future_date.dayofweek * self.feature_weights.get("day_of_week", 1.0)
            if self.feature_config.month:
                future_features["month"] = future_date.month * self.feature_weights.get("month", 1.0)

            # NEW: Add leap-year safe temporal features
            if self.feature_config.month_day:
                # Create month-day key that's consistent across years: Nov 11 = 1111 always
                month_day_key = future_date.month * 100 + future_date.day
                future_features["month_day"] = (month_day_key / 1231.0) * self.feature_weights.get("month_day", 1.0)

            if self.feature_config.week_of_year:
                week_of_year = future_date.isocalendar().week / 52.0
                future_features["week_of_year"] = week_of_year * self.feature_weights.get("week_of_year", 1.0)

            # Campaign features
            month_day = future_date.strftime("%m-%d")
            if month_day in campaign_calendar:
                future_features["is_campaign_day"] = 1 * self.feature_weights.get("is_campaign_day", 1.0)
                future_features["campaign_intensity"] = campaign_calendar[month_day][
                    "avg_uplift"
                ] * self.feature_weights.get("campaign_intensity", 1.0)
            else:
                future_features["is_campaign_day"] = 0
                future_features["campaign_intensity"] = 0.0

            # FIXED: Rolling average that updates with each prediction
            if self.feature_config.rolling_avg_7d:
                current_rolling_avg = (
                    np.mean(rolling_window[-7:]) if len(rolling_window) >= 7 else np.mean(rolling_window)
                )
                future_features["rolling_avg_7d"] = current_rolling_avg * self.feature_weights.get(
                    "rolling_avg_7d", 1.0
                )

            # FIXED: Proper lag feature using prediction history
            if self.feature_config.lag_7:
                if len(prediction_history) >= 7:
                    lag_value = prediction_history[-(7)]  # 7 days ago
                else:
                    lag_value = prediction_history[-1] if prediction_history else 0
                future_features["lag_7"] = lag_value * self.feature_weights.get("lag_7", 1.0)

            # Make prediction
            X_query = np.array([[future_features[col] for col in feature_cols]])
            X_query_scaled = self.scaler.transform(X_query)

            # FIXED: Simple euclidean distance (weights already applied in features)
            distances = euclidean_distances(X_query_scaled, X_train)[0]

            # Handle case where we have fewer samples than k
            actual_k = min(self.k_neighbors, len(distances))
            nearest_idx = np.argsort(distances)[:actual_k]

            # Weighted prediction with distance-based weights
            neighbor_distances = distances[nearest_idx]
            weights = 1.0 / (neighbor_distances + 1e-8)  # Add small epsilon to avoid division by zero
            weights /= weights.sum()

            neighbor_values = cast(np.ndarray, y_train[nearest_idx])
            prediction = np.sum(neighbor_values * weights)
            prediction = max(0, prediction)  # Ensure non-negative

            # FIXED: Proper confidence intervals
            lower, upper = self._calculate_confidence_intervals(prediction, neighbor_values, weights)

            # Debug logging - show temporal feature matching
            if i < 5 or month_day in campaign_calendar:  # Log first few days or campaign days
                nearest_dates = [feature_data.iloc[idx]["ds"].strftime("%Y-%m-%d") for idx in nearest_idx[:3]]
                nearest_month_days = [
                    f"{pd.to_datetime(feature_data.iloc[idx]['ds']).month:02d}-{pd.to_datetime(feature_data.iloc[idx]['ds']).day:02d}"
                    for idx in nearest_idx[:3]
                ]
                current_month_day_str = f"{future_date.month:02d}-{future_date.day:02d}"
                logger.info(
                    f"  {future_date.strftime('%Y-%m-%d')} ({current_month_day_str}) -> nearest: {nearest_dates} ({nearest_month_days}) -> pred: {prediction:.1f}"
                )

            forecasts.append(
                {
                    "ds": future_date,
                    "y_pred": prediction,
                    "y_pred_lower": lower,
                    "y_pred_upper": upper,
                    "country": country,
                    "marketplace": marketplace,
                }
            )

            # FIXED: Update prediction history for future lag features
            prediction_history.append(prediction)
            rolling_window.append(prediction)

            # Keep windows manageable
            if len(prediction_history) > 30:
                prediction_history = prediction_history[-30:]
            if len(rolling_window) > 14:
                rolling_window = rolling_window[-14:]

        forecast_df = pd.DataFrame(forecasts)
        logger.debug(f"k-NN forecast total: {forecast_df['y_pred'].sum():.0f}")

        return {"forecast": forecast_df, "validation_metrics": validation_metrics}
