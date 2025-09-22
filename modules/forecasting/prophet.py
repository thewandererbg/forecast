import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, cast
import logging
import warnings
from prophet import Prophet

from .forecast import BaseForecaster
from ..data_preparation import MarketplaceKey, CampaignInfo
from ..validation import ValidationConfig, ForecastValidator
from ..campaign import SpikeDetector, apply_campaign_factors


logger = logging.getLogger(__name__)


class ProphetForecaster(BaseForecaster):
    """Facebook Prophet forecasting implementation with validation"""

    def __init__(
        self,
        forecast_horizon: int = 90,
        target_column: str = "sum_quantity",
        validation_config: Optional[ValidationConfig] = None,
    ):
        super().__init__(forecast_horizon, target_column, validation_config)

        # Prophet parameters optimized for ecommerce
        self.prophet_params = {
            "seasonality_mode": "multiplicative",
            "yearly_seasonality": False,
            "weekly_seasonality": True,
            "daily_seasonality": False,
            "changepoint_prior_scale": 0.05,
            "seasonality_prior_scale": 10.0,
            "interval_width": 0.95,
            "growth": "linear",
        }

        # Initialize spike detector
        self.spike_detector = SpikeDetector()

    def get_model_name(self) -> str:
        return "Prophet"

    def forecast_marketplace(
        self,
        country: str,
        marketplace: str,
        ts_data: pd.DataFrame,
        campaign_calendar: Dict[str, CampaignInfo],
        _skip_validation: bool = False,
    ) -> Dict[str, Union[pd.DataFrame, Dict[str, float], None]]:
        """Generate forecast for a marketplace using Prophet + campaign calendar"""
        logger.debug(f"Forecasting {country}-{marketplace} (Prophet + Campaigns)")

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

        # Initialize and fit Prophet
        prophet_params = self.prophet_params.copy()

        # Suppress Prophet logs
        logging.getLogger("prophet").setLevel(logging.WARNING)
        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

        model = Prophet(**prophet_params)
        model.fit(dampened_data)

        # Generate forecast
        future = model.make_future_dataframe(periods=self.forecast_horizon)
        forecast = model.predict(future)

        # Extract forecast period only
        forecast_start = pd.to_datetime(self.forecast_start_date)
        forecast_period = cast(pd.DataFrame, forecast[forecast["ds"] >= forecast_start].copy())

        # Apply additional campaign factors
        forecast_period = apply_campaign_factors(
            forecast_period=forecast_period,
            campaign_calendar=campaign_calendar,
            country=country,
            marketplace=marketplace,
        )

        # Prepare output with confidence intervals
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

        logger.debug(f"  Prophet forecast total: {result['y_pred'].sum():.0f}")

        return {"forecast": result, "validation_metrics": validation_metrics}
