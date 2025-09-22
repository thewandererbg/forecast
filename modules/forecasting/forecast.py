"""
Forecasting Module

Contains forecasting methods with a common interface for easy extensibility.
Currently implements Prophet, with support for adding other methods in the future.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, cast
import logging
import warnings
from tqdm import tqdm

# Suppress Prophet warnings
warnings.filterwarnings("ignore")

from ..data_preparation import MarketplaceKey, CampaignInfo
from ..validation import ValidationConfig, ForecastValidator
from ..campaign import SpikeDetector, apply_campaign_factors


logger = logging.getLogger(__name__)


class BaseForecaster(ABC):
    """Abstract base class for all forecasting methods"""

    def __init__(
        self,
        forecast_horizon: int = 90,
        target_column: str = "sum_quantity",
        validation_config: Optional[ValidationConfig] = None,
    ):
        self.forecast_horizon = forecast_horizon
        self.target_column = target_column
        self.forecast_start_date = ""
        self.validation_config = validation_config or ValidationConfig()

    @abstractmethod
    def forecast_marketplace(
        self,
        country: str,
        marketplace: str,
        ts_data: pd.DataFrame,
        campaign_calendar: Dict[str, CampaignInfo],
    ) -> Dict[str, Union[pd.DataFrame, Dict[str, float], None]]:
        """
        Generate forecast for a marketplace

        Returns:
            Dictionary with:
            - 'forecast': pd.DataFrame with forecast results
            - 'validation_metrics': Dict[str, float] with MAPE, MAE, directional_accuracy or None
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the name of the forecasting model"""
        pass

    def train_global_model(self, all_market_data: Dict[str, pd.DataFrame]) -> None:
        """
        Train global model (optional - override in subclasses that support global training)
        Default implementation does nothing for individual market models
        """
        pass

    def supports_global_training(self) -> bool:
        """
        Check if this forecaster supports global training
        Override in subclasses that support global training
        """
        return False


class ForecastingEngine:
    """Main forecasting engine that coordinates different forecasting methods"""

    def __init__(self, data_preparation, forecaster: BaseForecaster):
        self.data_preparation = data_preparation
        self.forecaster = forecaster

        # Set forecast start date from data preparation
        self.forecaster.forecast_start_date = data_preparation.forecast_start_date

    def _prepare_global_training_data(self, marketplace_combinations: List[MarketplaceKey]) -> Dict[str, pd.DataFrame]:
        """Prepare all market data for global training"""
        all_market_data = {}

        for country, marketplace in marketplace_combinations:
            try:
                ts_data = self.data_preparation.create_marketplace_timeseries(country, marketplace)
                if len(ts_data) > 0:
                    market_key = f"{country}-{marketplace}"
                    all_market_data[market_key] = ts_data
            except Exception as e:
                logger.warning(f"Could not prepare data for {country}-{marketplace}: {e}")

        return all_market_data

    def generate_marketplace_forecasts(self, marketplace_combinations: List[MarketplaceKey]) -> List[pd.DataFrame]:
        """Generate marketplace-level forecasts for all combinations"""
        logger.debug("\n" + "=" * 60)
        logger.debug(f"GENERATING MARKETPLACE FORECASTS WITH {self.forecaster.get_model_name()}")
        logger.debug("=" * 60)

        # Check if this is a global model that needs training
        if hasattr(self.forecaster, "supports_global_training") and self.forecaster.supports_global_training():
            # Check if it needs training (only for models that track training state)
            needs_training = not getattr(self.forecaster, "is_trained", False)

            if needs_training:
                logger.info("Training global model on all market data...")
                all_market_data = self._prepare_global_training_data(marketplace_combinations)
                self.forecaster.train_global_model(all_market_data)
                logger.info("Global model training completed")

        marketplace_forecasts = []
        validation_summary = []
        successful_forecasts = 0
        skipped_forecasts = 0

        for country, marketplace in tqdm(marketplace_combinations, desc="Generating marketplace forecasts"):
            try:
                # Get time series data
                ts_data = self.data_preparation.create_marketplace_timeseries(country, marketplace)

                # Get campaign calendar
                campaign_calendar = self.data_preparation.get_campaign_calendar(country, marketplace)

                # Generate forecast (now returns dict with forecast + validation_metrics)
                result = self.forecaster.forecast_marketplace(country, marketplace, ts_data, campaign_calendar)

                # Extract forecast DataFrame and validation metrics
                marketplace_forecast = cast(pd.DataFrame, result["forecast"])
                validation_metrics = cast(dict, result["validation_metrics"])

                if len(marketplace_forecast) > 0:
                    marketplace_forecasts.append(marketplace_forecast)
                    successful_forecasts += 1

                    # Collect validation metrics if available
                    if validation_metrics:
                        validation_summary.append(
                            {
                                "country": country,
                                "marketplace": marketplace,
                                "MAPE": validation_metrics["MAPE"],
                                "MAE": validation_metrics["MAE"],
                                "directional_accuracy": validation_metrics["directional_accuracy"],
                            }
                        )
                else:
                    skipped_forecasts += 1

            except Exception as e:
                logger.error(f"  Error forecasting {country}-{marketplace}: {e}")
                skipped_forecasts += 1

        # Log validation summary if available
        if validation_summary:
            logger.info(f"\n" + "=" * 40)
            logger.info("VALIDATION SUMMARY")
            logger.info("=" * 40)

            # Calculate average metrics
            avg_mape = sum(m["MAPE"] for m in validation_summary) / len(validation_summary)
            avg_mae = sum(m["MAE"] for m in validation_summary) / len(validation_summary)
            avg_direction = sum(m["directional_accuracy"] for m in validation_summary) / len(validation_summary)

            logger.info(f"Average MAPE: {avg_mape:.1f}%")
            logger.info(f"Average MAE: {avg_mae:.1f}")
            logger.info(f"Average Directional Accuracy: {avg_direction:.1f}%")

            # Log individual results
            logger.info("\nIndividual marketplace validation:")
            for metrics in validation_summary:
                logger.info(
                    f"  {metrics['country']}-{metrics['marketplace']}: "
                    f"MAPE={metrics['MAPE']:.1f}%, MAE={metrics['MAE']:.1f}, "
                    f"Direction={metrics['directional_accuracy']:.1f}%"
                )

        logger.debug(f"\nMarketplace forecasting results:")
        logger.debug(f"  Successful {self.forecaster.get_model_name()} forecasts: {successful_forecasts} marketplaces")
        if validation_summary:
            logger.debug(f"  Validated forecasts: {len(validation_summary)} marketplaces")
        logger.debug(f"  Skipped marketplaces: {skipped_forecasts}")

        if not marketplace_forecasts:
            raise ValueError("No successful marketplace forecasts generated")

        return marketplace_forecasts


class MovingAverageForecaster(BaseForecaster):
    """Moving Average forecasting implementation using same day of week"""

    def __init__(self, forecast_horizon: int = 90, target_column: str = "sum_quantity"):
        super().__init__(forecast_horizon, target_column)
        self.lookback_weeks = 4 * 6

        # Initialize spike detector
        self.spike_detector = SpikeDetector()

    def get_model_name(self) -> str:
        return "MovingAverage"

    def calculate_weekday_baselines(self, ts_data: pd.DataFrame) -> Dict[int, List[float]]:
        """Calculate initial weekday values for moving average (last 12 occurrences)"""
        if len(ts_data) == 0:
            return {}

        # Detect spikes and create dampened series
        spike_indices, _ = self.spike_detector.detect_spikes_adaptive(ts_data, "baseline_calculation")
        dampened_data = self.spike_detector.create_spike_dampened_series(ts_data, spike_indices)

        # Add weekday column
        dampened_data["weekday"] = dampened_data["ds"].dt.dayofweek

        baselines = {}

        for weekday in range(7):  # 0=Monday, 6=Sunday
            weekday_data = cast(pd.DataFrame, dampened_data[dampened_data["weekday"] == weekday])

            if len(weekday_data) == 0:
                baselines[weekday] = []
            else:
                # Get last 12 occurrences for this weekday
                recent_data = weekday_data.tail(12)
                values = recent_data["y"].values.tolist()
                baselines[weekday] = values

        return baselines

    def generate_baseline_forecast(
        self, forecast_start: pd.Timestamp, baselines: Dict[int, List[float]]
    ) -> pd.DataFrame:
        """Generate baseline forecast using dynamic moving averages"""
        forecast_dates = pd.date_range(start=forecast_start, periods=self.forecast_horizon, freq="D")

        # Create a copy of baselines to modify during forecasting
        moving_baselines = {weekday: values.copy() for weekday, values in baselines.items()}

        forecast_data = []

        for date in forecast_dates:
            weekday = date.dayofweek
            weekday_values = moving_baselines.get(weekday, [])

            if len(weekday_values) == 0:
                mean_val = 0
                std_val = 0
            else:
                values = np.array(weekday_values, dtype=float)
                mean_val = cast(float, np.mean(values))
                std_val = cast(float, np.std(values) if len(values) > 1 else np.mean(values) * 0.1)

            # 95% confidence interval (1.96 * std)
            lower = max(0, mean_val - 1.96 * std_val)
            upper = mean_val + 1.96 * std_val

            forecast_data.append(
                {"ds": date, "yhat": mean_val, "yhat_lower": lower, "yhat_upper": upper, "weekday": weekday}
            )

            # Update moving baseline: add current forecast and keep last 12 values
            if weekday in moving_baselines:
                moving_baselines[weekday].append(mean_val)
                if len(moving_baselines[weekday]) > 12:
                    moving_baselines[weekday] = moving_baselines[weekday][-12:]
            else:
                moving_baselines[weekday] = [mean_val]

        return pd.DataFrame(forecast_data)

    def forecast_marketplace(
        self,
        country: str,
        marketplace: str,
        ts_data: pd.DataFrame,
        campaign_calendar: Dict[str, CampaignInfo],
        _skip_validation: bool = False,
    ) -> Dict[str, Union[pd.DataFrame, Dict[str, float], None]]:
        """Generate forecast for a marketplace using Moving Average + campaign calendar"""
        logger.debug(f"Forecasting {country}-{marketplace} (MovingAverage + Campaigns)")

        if len(ts_data) == 0:
            logger.warning(f"  No data for {country}-{marketplace} - skipping")
            return {"forecast": pd.DataFrame(), "validation_metrics": None}

        # Calculate weekday baselines
        baselines = self.calculate_weekday_baselines(ts_data)

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

        # Generate baseline forecast
        forecast_start = pd.to_datetime(self.forecast_start_date)
        forecast_period = self.generate_baseline_forecast(forecast_start, baselines)

        # Apply campaign factors
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

        logger.debug(f"  MovingAverage forecast total: {result['y_pred'].sum():.0f}")
        return {"forecast": result, "validation_metrics": validation_metrics}
