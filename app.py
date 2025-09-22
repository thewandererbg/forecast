"""
Top-Down Forecasting Pipeline - Refactored Main Script

This script orchestrates the complete forecasting pipeline using modular components:
1. Data Preparation: Load, clean, and prepare data with brand ratios and campaign calendars
2. Forecasting: Generate marketplace-level forecasts (currently Prophet, extensible to other methods)
3. Results Export: Breakdown to brand level and export all results

The modular design allows easy addition of new forecasting methods in the future.
"""

import traceback
import logging
from pathlib import Path
import click
from modules import (
    DataPreparation,
    ProphetForecaster,
    ForecastingEngine,
    ResultsExporter,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--model",
    "-m",
    type=click.Choice(["prophet", "moving_average", "arima", "xgboost", "knn"], case_sensitive=False),
    default="arima",
    help="Forecasting model to use",
)
def main(model: str) -> None:
    """
    Run the top-down forecasting pipeline with specified model.

    Country-Marketplace → Brand Breakdown
    """

    # Fixed parameters for now
    data_file = "data.csv"
    target_column = "sum_quantity"
    forecast_horizon = 90

    try:
        logger.info("=" * 80)
        logger.info("TOP-DOWN FORECASTING PIPELINE WITH MODULAR DESIGN")
        logger.info("   Country-Marketplace → Brand Breakdown")
        logger.info("=" * 80)
        logger.info(f"Target metric: {target_column}")
        logger.info(f"Forecast horizon: {forecast_horizon} days")
        logger.info(f"Forecaster: {model}")

        # Step 1: Initialize Data Preparation
        logger.info("\n" + "=" * 60)
        logger.info("STEP 1: DATA PREPARATION")
        logger.info("=" * 60)

        data_prep = DataPreparation(data_file=data_file, target_column=target_column)
        marketplace_combinations = data_prep.prepare_all_data()

        # Step 2: Initialize Forecasting Engine
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: FORECASTING")
        logger.info("=" * 60)

        # Select forecaster based on type
        if model.lower() == "prophet":
            forecaster = ProphetForecaster(forecast_horizon=forecast_horizon, target_column=target_column)
        elif model.lower() == "moving_average":
            from modules.forecasting import MovingAverageForecaster

            forecaster = MovingAverageForecaster(forecast_horizon=forecast_horizon, target_column=target_column)
        elif model.lower() == "arima":
            from modules.forecasting import AutoArimaForecaster

            forecaster = AutoArimaForecaster(forecast_horizon=forecast_horizon, target_column=target_column)
        elif model.lower() == "xgboost":
            from modules.forecasting import XGBoostForecaster

            forecaster = XGBoostForecaster(forecast_horizon=forecast_horizon, target_column=target_column)
        elif model.lower() == "knn":
            from modules.forecasting import KNNForecaster

            forecaster = KNNForecaster(forecast_horizon=forecast_horizon, target_column=target_column)
        else:
            raise ValueError(f"Unknown forecaster type: {model}")

        # Initialize forecasting engine
        forecasting_engine = ForecastingEngine(data_prep, forecaster)

        # Generate marketplace forecasts
        marketplace_forecasts = forecasting_engine.generate_marketplace_forecasts(marketplace_combinations)

        # Step 3: Export Results
        logger.info("\n" + "=" * 60)
        logger.info("STEP 3: RESULTS EXPORT")
        logger.info("=" * 60)

        results_exporter = ResultsExporter(data_prep, target_column=target_column)

        # Breakdown to brand level
        brand_forecasts = results_exporter.breakdown_to_brands(marketplace_forecasts)

        if not brand_forecasts:
            raise ValueError("No successful brand forecasts generated")

        # Create final dataset
        final_data = results_exporter.create_final_dataset(brand_forecasts)

        # Export all results
        results_exporter.export_all_results(final_data)

        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"✓ Data prepared: {len(marketplace_combinations)} marketplaces")
        logger.info(f"✓ Forecasts generated using {forecaster.get_model_name()}")
        logger.info(f"✓ Results exported: {len(final_data):,} total rows")
        logger.info(f"✓ Files created: forecast.csv, campaign_calendars.csv, brand_ratios_and_prices.csv")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
