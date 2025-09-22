"""
Export Results Module

Handles breakdown from marketplace to brand level and exports all results.
"""

import pandas as pd
import numpy as np
from typing import List, cast
import logging

logger = logging.getLogger(__name__)


class ResultsExporter:
    """Handles breakdown to brand level and result exports"""

    def __init__(self, data_preparation, target_column: str = "sum_quantity", revenue_column: str = "sum_s_net"):
        self.data_preparation = data_preparation
        self.target_column = target_column
        self.revenue_column = revenue_column

    def breakdown_to_brands(self, marketplace_forecasts: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """Breakdown marketplace forecasts to brand level using historical ratios"""
        logger.debug("Breaking down marketplace forecasts to brand level...")

        brand_forecasts = []
        for marketplace_forecast in marketplace_forecasts:
            if len(marketplace_forecast) == 0:
                continue

            country = str(marketplace_forecast.iloc[0]["country"])
            marketplace = str(marketplace_forecast.iloc[0]["marketplace"])

            # Get brand ratios and prices for this marketplace
            brand_ratios = self.data_preparation.get_brand_ratios(country, marketplace)
            brand_prices = self.data_preparation.get_brand_prices(country, marketplace)

            if not brand_ratios:
                logger.warning(f"  No brand ratios for {country}-{marketplace}")
                continue

            logger.debug(f"  Breaking down {country}-{marketplace} to {len(brand_ratios)} brands")

            # Create brand-level forecasts
            current_brand_forecasts = []
            for brand, ratio in brand_ratios.items():
                brand_forecast = marketplace_forecast.copy()
                brand_forecast["brand"] = brand
                brand_forecast["fk_brand_used_id"] = brand
                brand_forecast["brand_ratio"] = ratio

                # Get brand price (default to 0 if not found)
                brand_price = brand_prices.get(brand, 0.0)
                brand_forecast["brand_price"] = brand_price

                # Calculate quantity forecasts
                brand_forecast["y_pred"] = brand_forecast["y_pred"] * ratio
                brand_forecast["y_pred_lower"] = brand_forecast["y_pred_lower"] * ratio
                brand_forecast["y_pred_upper"] = brand_forecast["y_pred_upper"] * ratio

                # Calculate revenue forecasts using brand price
                brand_forecast["revenue_pred"] = brand_forecast["y_pred"] * brand_price
                brand_forecast["revenue_pred_lower"] = brand_forecast["y_pred_lower"] * brand_price
                brand_forecast["revenue_pred_upper"] = brand_forecast["y_pred_upper"] * brand_price

                current_brand_forecasts.append(brand_forecast)

                brand_forecasts.append(
                    brand_forecast[
                        [
                            "ds",
                            "y_pred",
                            "y_pred_lower",
                            "y_pred_upper",
                            "revenue_pred",
                            "revenue_pred_lower",
                            "revenue_pred_upper",
                            "country",
                            "marketplace",
                            "brand",
                            "fk_brand_used_id",
                            "brand_ratio",
                            "brand_price",
                        ]
                    ]
                )

            # Log breakdown results
            self._log_breakdown_results(marketplace_forecast, current_brand_forecasts, country, marketplace)

        return brand_forecasts

    def _log_breakdown_results(
        self,
        marketplace_forecast: pd.DataFrame,
        brand_forecasts: List[pd.DataFrame],
        country: str,
        marketplace: str,
    ) -> None:
        """Log breakdown results for verification"""
        marketplace_total = marketplace_forecast["y_pred"].sum()
        total_brand_forecast = sum(bf["y_pred"].sum() for bf in brand_forecasts)
        total_brand_revenue = sum(bf["revenue_pred"].sum() for bf in brand_forecasts)

        logger.debug(f"    Marketplace total: {marketplace_total:.0f}")
        logger.debug(
            f"    Brand sum: {total_brand_forecast:.0f} (ratio: {total_brand_forecast / marketplace_total:.3f})"
        )
        logger.debug(f"    Brand revenue sum: ${total_brand_revenue:,.0f}")

    def create_final_dataset(self, brand_forecasts: List[pd.DataFrame]) -> pd.DataFrame:
        """Combine original data with brand forecasts"""
        logger.debug("Creating final dataset...")

        # Prepare forecast data
        forecast_df = self._prepare_forecast_data(brand_forecasts)

        # Prepare original data
        original_data = self._prepare_original_data()

        # Combine datasets
        combined_data = (
            pd.concat([original_data, forecast_df], ignore_index=True) if len(forecast_df) > 0 else original_data
        )

        # Sort data
        combined_data = combined_data.sort_values(["country", "marketplace", "fk_brand_used_id", "day"]).reset_index(
            drop=True
        )

        logger.debug(f"Final dataset: {len(combined_data):,} rows")
        logger.debug(f"  Original: {len(original_data):,}")
        logger.debug(f"  Forecast: {len(forecast_df):,}")

        return combined_data

    def _prepare_forecast_data(self, brand_forecasts: List[pd.DataFrame]) -> pd.DataFrame:
        """Prepare forecast data for final dataset"""
        if not brand_forecasts:
            logger.warning("No forecasts generated")
            return pd.DataFrame()

        forecast_df = pd.concat(brand_forecasts, ignore_index=True)

        # Format dates and add metadata
        forecast_df["day"] = forecast_df["ds"].dt.strftime("%Y-%m-%d")
        forecast_df["month"] = forecast_df["ds"].dt.strftime("%Y_%m")
        forecast_df["is_forecast"] = True

        # Map quantity columns
        forecast_df[self.target_column] = forecast_df["y_pred"]
        forecast_df[f"{self.target_column}_lower"] = forecast_df["y_pred_lower"]
        forecast_df[f"{self.target_column}_upper"] = forecast_df["y_pred_upper"]

        # Map revenue columns
        forecast_df[self.revenue_column] = forecast_df["revenue_pred"]
        forecast_df[f"{self.revenue_column}_lower"] = forecast_df["revenue_pred_lower"]
        forecast_df[f"{self.revenue_column}_upper"] = forecast_df["revenue_pred_upper"]

        # Get metadata for brands
        metadata_lookup = self._create_metadata_lookup()

        # Merge with metadata
        forecast_with_metadata = forecast_df.merge(
            metadata_lookup,
            left_on=["country", "marketplace", "brand"],
            right_on=["country", "marketplace", "fk_brand_used_id"],
            how="left",
        )

        # Fill missing metadata
        forecast_with_metadata["seller"] = forecast_with_metadata["seller"].fillna("Unknown")
        forecast_with_metadata["seller name"] = forecast_with_metadata["seller name"].fillna("Unknown")

        # Select final columns
        return cast(
            pd.DataFrame,
            forecast_with_metadata[
                [
                    "country",
                    "marketplace",
                    "seller",
                    "seller name",
                    "brand",
                    "day",
                    "month",
                    self.target_column,
                    f"{self.target_column}_lower",
                    f"{self.target_column}_upper",
                    self.revenue_column,
                    f"{self.revenue_column}_lower",
                    f"{self.revenue_column}_upper",
                    "is_forecast",
                    "brand_ratio",
                    "brand_price",
                ]
            ],
        ).rename(columns={"brand": "fk_brand_used_id"})

    def _create_metadata_lookup(self) -> pd.DataFrame:
        """Create metadata lookup table for brands"""
        return (
            self.data_preparation.training_data.groupby(["country", "marketplace", "fk_brand_used_id"])
            .agg({"seller": "first", "seller name": "first"})
            .reset_index()
        )

    def _prepare_original_data(self) -> pd.DataFrame:
        """Prepare original data for final dataset"""
        original_data = self.data_preparation.raw_data.copy()
        original_data["day"] = original_data["day"].astype(str)
        original_data["is_forecast"] = False

        # Add placeholder columns to match forecast structure
        original_data["brand_ratio"] = np.nan
        original_data["brand_price"] = np.nan
        original_data[f"{self.target_column}_lower"] = np.nan
        original_data[f"{self.target_column}_upper"] = np.nan
        original_data[f"{self.revenue_column}_lower"] = np.nan
        original_data[f"{self.revenue_column}_upper"] = np.nan

        return original_data

    def export_main_results(self, final_data: pd.DataFrame, output_file: str = "forecast.csv") -> None:
        """Export main forecast results"""
        final_data.to_csv(output_file, index=False)
        logger.debug(f"\nResults exported to '{output_file}'")

    def export_campaign_calendars(self, output_file: str = "campaign_calendars.csv") -> None:
        """Export learned campaign calendars to CSV for review"""
        calendar_records = []

        for (
            country,
            marketplace,
        ), calendar in self.data_preparation.campaign_calendars.items():
            for month_day, campaign_info in calendar.items():
                calendar_records.append(
                    {
                        "country": country,
                        "marketplace": marketplace,
                        "campaign_date": month_day,
                        "avg_uplift": campaign_info["avg_uplift"],
                        "max_uplift": campaign_info["max_uplift"],
                        "min_uplift": campaign_info["min_uplift"],
                        "occurrences": campaign_info["occurrences"],
                        "consistency": campaign_info["consistency"],
                        "historical_dates": ", ".join([d.strftime("%Y-%m-%d") for d in campaign_info["dates"]]),
                    }
                )

        if calendar_records:
            calendar_df = pd.DataFrame(calendar_records)
            calendar_df = calendar_df.sort_values(["country", "marketplace", "campaign_date"])
            calendar_df.to_csv(output_file, index=False)
            logger.debug(f"Campaign calendars exported to '{output_file}'")
            logger.debug(f"   Total campaign patterns discovered: {len(calendar_records)}")
        else:
            logger.warning("No campaign patterns discovered to export")

    def export_brand_ratios_and_prices(self, output_file: str = "brand_ratios_and_prices.csv") -> None:
        """Export brand ratios and prices to CSV for review"""
        ratio_records = []

        for (
            country,
            marketplace,
        ), ratios in self.data_preparation.brand_ratios.items():
            # Get corresponding prices
            prices = self.data_preparation.brand_prices.get((country, marketplace), {})

            for brand, ratio in ratios.items():
                brand_price = prices.get(brand, 0.0)
                ratio_records.append(
                    {
                        "country": country,
                        "marketplace": marketplace,
                        "fk_brand_used_id": brand,
                        "brand_ratio": ratio,
                        "percentage": f"{ratio:.3%}",
                        "avg_selling_price": brand_price,
                    }
                )

        if ratio_records:
            ratio_df = pd.DataFrame(ratio_records)
            ratio_df = ratio_df.sort_values(["country", "marketplace", "brand_ratio"], ascending=[True, True, False])
            ratio_df.to_csv(output_file, index=False)
            logger.debug(f"Brand ratios exported to '{output_file}'")
            logger.debug(f"   Total brand ratios calculated: {len(ratio_records)}")
        else:
            logger.warning("No brand ratios calculated to export")

    def export_all_results(self, final_data: pd.DataFrame) -> None:
        """Export all results including main forecast, campaign calendars, brand ratios, and brand prices"""
        logger.debug("\n" + "=" * 60)
        logger.debug("EXPORTING RESULTS")
        logger.debug("=" * 60)

        # Export main results
        self.export_main_results(final_data)

        # Export supporting data
        self.export_campaign_calendars()
        self.export_brand_ratios_and_prices()

        logger.debug("\nAll exports completed successfully!")
