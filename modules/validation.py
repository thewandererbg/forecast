"""
Validation module for forecasting models

This module provides generic validation functionality that works with any forecasting model
by testing the complete pipeline output that users will see in production.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any, cast
import pandas as pd
import numpy as np
import logging

from streamlit.dataframe_util import Data

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for forecast validation"""

    enable_validation: bool = True
    validation_days: int = 90
    validation_min_data_months: int = 6


class ForecastValidator:
    """Generic forecast validation utilities"""

    @staticmethod
    def validate_forecast_model(
        ts_data: pd.DataFrame,
        config: ValidationConfig,
        forecaster_instance: Any,
        country: str,
        marketplace: str,
        campaign_calendar: Dict[str, Any],
    ) -> Optional[Dict[str, float]]:
        """
        Generic validation method for any forecasting model

        This method tests the complete forecasting pipeline by:
        1. Splitting data into train/validation periods
        2. Training the model on historical data
        3. Generating forecasts for the validation period
        4. Comparing predictions with actual values

        Args:
            ts_data: Time series data with 'ds' and target column
            target_column: Name of the target column to forecast
            config: ValidationConfig with validation parameters
            forecaster_instance: The forecaster object with forecast_marketplace method
            country: Country identifier for logging
            marketplace: Marketplace identifier for logging
            campaign_calendar: Campaign calendar for the marketplace

        Returns:
            Dictionary with MAPE, MAE, directional_accuracy or None if validation disabled/insufficient data
        """

        if not config.enable_validation:
            return None

        # Check if we have enough data for validation
        months_of_data = (ts_data["ds"].max() - ts_data["ds"].min()).days / 30.44  # Average days per month

        if months_of_data < config.validation_min_data_months:
            logger.warning(
                f"  Insufficient data for validation: {months_of_data:.1f} months (need {config.validation_min_data_months})"
            )
            return None

        logger.debug(f"  Running validation on last {config.validation_days} days...")

        # Calculate validation split date
        latest_date = ts_data["ds"].max()
        if latest_date < pd.Timestamp("2025-08-10"):
            return None

        validation_start = latest_date - pd.DateOffset(days=config.validation_days)

        # Split data
        train_data = cast(pd.DataFrame, ts_data[ts_data["ds"] < validation_start].copy())
        validation_data = cast(pd.DataFrame, ts_data[ts_data["ds"] >= validation_start].copy())

        if len(train_data) == 0 or len(validation_data) == 0:
            logger.warning(
                f"  Insufficient data for validation: {len(train_data)} train, {len(validation_data)} validation days"
            )
            return {"MAPE": float("inf"), "MAE": float("inf"), "directional_accuracy": 0.0}

        try:
            # Temporarily set forecast start date to validation start
            original_forecast_start = forecaster_instance.forecast_start_date
            forecaster_instance.forecast_start_date = validation_start.strftime("%Y-%m-%d")

            # Use the forecaster's complete forecast_marketplace method on training data
            # This tests the exact same pipeline that users will see in production
            forecast_result = forecaster_instance.forecast_marketplace(
                country, marketplace, train_data, campaign_calendar, _skip_validation=True
            )

            # Restore original forecast start date
            forecaster_instance.forecast_start_date = original_forecast_start

            # Extract forecast DataFrame (handle both dict and DataFrame returns)
            if isinstance(forecast_result, dict):
                forecast_df = forecast_result.get("forecast", pd.DataFrame())
            else:
                forecast_df = forecast_result

            if forecast_df.empty:
                logger.warning(f"  No forecast generated for validation period")
                return {"MAPE": float("inf"), "MAE": float("inf"), "directional_accuracy": 0.0}

            # Align validation data with forecast predictions by date
            # Ensure datetime columns are properly converted
            validation_dates = cast(pd.Series, validation_data["ds"].dt.strftime("%Y-%m-%d"))
            forecast_dates = cast(pd.Series, forecast_df["ds"].dt.strftime("%Y-%m-%d"))

            # Find common dates and dates not in campaign
            common_dates = set(validation_dates) & set(forecast_dates)

            if not common_dates:
                logger.warning(f"  No overlapping dates between validation data and forecast")
                return {"MAPE": float("inf"), "MAE": float("inf"), "directional_accuracy": 0.0}

            # Filter to common dates and sort
            val_mask = validation_dates.isin(common_dates)
            pred_mask = forecast_dates.isin(common_dates)

            validation_subset = cast(pd.DataFrame, validation_data[val_mask]).sort_values("ds")
            forecast_subset = cast(pd.DataFrame, forecast_df[pred_mask]).sort_values("ds")

            # Extract actual and predicted values
            actual_values = cast(np.ndarray, validation_subset["y"].values)
            predicted_values = cast(np.ndarray, forecast_subset["y_pred"].values)

            # Calculate and return validation metrics
            metrics = ForecastValidator.calculate_validation_metrics(actual_values, predicted_values)

            logger.debug(
                f"  Validation metrics - MAPE: {metrics['MAPE']:.1f}%, MAE: {metrics['MAE']:.1f}, Directional Accuracy: {metrics['directional_accuracy']:.1f}%"
            )

            ForecastValidator.export_validation_data(validation_subset, forecast_subset, metrics, country, marketplace)

            return metrics

        except Exception as e:
            logger.error(f"  Validation failed for {country}-{marketplace}: {str(e)}")
            return {"MAPE": float("inf"), "MAE": float("inf"), "directional_accuracy": 0.0}

    @staticmethod
    def calculate_validation_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """
        Calculate standard validation metrics

        Args:
            actual: Array of actual values
            predicted: Array of predicted values

        Returns:
            Dictionary with MAPE, MAE, and directional_accuracy
        """

        # Ensure arrays are same length
        min_length = min(len(actual), len(predicted))
        actual = actual[:min_length]
        predicted = predicted[:min_length]

        # MAPE (Mean Absolute Percentage Error)
        non_zero_mask = actual != 0
        if np.sum(non_zero_mask) > 0:
            mape = np.mean(np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask])) * 100
        else:
            mape = float("inf")

        # MAE (Mean Absolute Error)
        mae = np.mean(np.abs(actual - predicted))

        # Directional Accuracy (percentage of days where prediction and actual move in same direction)
        if len(actual) > 1:
            actual_direction = np.diff(actual) > 0
            predicted_direction = np.diff(predicted) > 0
            directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
        else:
            directional_accuracy = 0.0

        return {
            "MAPE": round(float(mape), 2),
            "MAE": round(float(mae), 2),
            "directional_accuracy": round(float(directional_accuracy), 2),
        }

    @staticmethod
    def log_validation_summary(validation_results: list[Dict[str, Any]]) -> None:
        """
        Log a summary of validation results across multiple marketplaces

        Args:
            validation_results: List of validation result dictionaries with country, marketplace, and metrics
        """
        if not validation_results:
            return

        logger.debug(f"\n" + "=" * 40)
        logger.debug("VALIDATION SUMMARY")
        logger.debug("=" * 40)

        # Calculate average metrics
        valid_results = [r for r in validation_results if r.get("MAPE") != float("inf")]

        if valid_results:
            avg_mape = sum(r["MAPE"] for r in valid_results) / len(valid_results)
            avg_mae = sum(r["MAE"] for r in valid_results) / len(valid_results)
            avg_direction = sum(r["directional_accuracy"] for r in valid_results) / len(valid_results)

            logger.debug(f"Average MAPE: {avg_mape:.1f}%")
            logger.debug(f"Average MAE: {avg_mae:.1f}")
            logger.debug(f"Average Directional Accuracy: {avg_direction:.1f}%")
            logger.debug(f"Validated marketplaces: {len(valid_results)}/{len(validation_results)}")
        else:
            logger.warning("No valid validation results to summarize")

        # Log individual results
        logger.debug("\nIndividual marketplace validation:")
        for result in validation_results:
            if result.get("MAPE") == float("inf"):
                logger.warning(f"  {result['country']}-{result['marketplace']}: FAILED")
            else:
                logger.debug(
                    f"  {result['country']}-{result['marketplace']}: "
                    f"MAPE={result['MAPE']:.1f}%, MAE={result['MAE']:.1f}, "
                    f"Direction={result['directional_accuracy']:.1f}%"
                )

    @staticmethod
    def export_validation_data(
        validation_subset: pd.DataFrame,
        forecast_subset: pd.DataFrame,
        metrics: Dict[str, float],
        country: str,
        marketplace: str,
    ) -> None:
        """Export validation data to master CSV file"""

        # Create output directory if it doesn't exist
        output_dir = Path(".")

        # Prepare validation data
        validation_data = pd.DataFrame(
            {
                "date": validation_subset["ds"].values,
                "actual": validation_subset["y"].values,
                "predicted": forecast_subset["y_pred"].values,
                "country": country,
                "marketplace": marketplace,
                "MAPE": metrics["MAPE"],
                "MAE": metrics["MAE"],
                "directional_accuracy": metrics["directional_accuracy"],
            }
        )

        # Append to master validation file
        master_file = output_dir / "validation_results.csv"
        if master_file.exists():
            validation_data.to_csv(master_file, mode="a", header=False, index=False)
        else:
            validation_data.to_csv(master_file, index=False)
