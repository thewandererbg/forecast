"""
Data Preparation Module

Handles data loading, cleaning, and preparation for forecasting.
Calculates brand ratios, campaign calendars, and historical maximums.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, TypedDict, Union, Optional, Any, cast
import logging
from tqdm import tqdm

from modules.campaign import CampaignCalendarBuilder

from .utils import DataValidator
from .types import CampaignInfo, MarketplaceKey, BrandKey, BrandRatios, BrandPrices

logger = logging.getLogger(__name__)


class DataPreparation:
    """Handles all data preparation tasks for the forecasting pipeline"""

    def __init__(
        self,
        data_file: Union[str, Path] = "data.csv",
        target_column: str = "sum_quantity",
        revenue_column: str = "sum_s_net",
    ) -> None:
        self.data_file: Path = Path(data_file)
        self.target_column: str = target_column
        self.revenue_column: str = revenue_column

        # Data storage
        self.raw_data: pd.DataFrame = pd.DataFrame()
        self.training_data: pd.DataFrame = pd.DataFrame()

        # Calculated data
        self.brand_ratios: Dict[MarketplaceKey, BrandRatios] = {}
        self.brand_prices: Dict[MarketplaceKey, BrandPrices] = {}
        self.campaign_calendars: Dict[MarketplaceKey, Dict[str, CampaignInfo]] = {}
        self.forecast_start_date: str = ""
        self.marketplace_combinations: List[MarketplaceKey] = []

        # Configuration
        self.brand_ratio_window: int = 30  # Last 30 days for brand ratios

        # Initialize utilities
        self.validator: DataValidator = DataValidator()
        self.campaign_calendar_builder: CampaignCalendarBuilder = CampaignCalendarBuilder()

    def load_and_prepare_data(self) -> None:
        """Load and prepare data for forecasting"""
        logger.debug("Loading and preparing data...")
        # Remove old csv files except data.csv
        for file in self.data_file.parent.glob("*.csv"):
            if file.name != self.data_file.name:
                file.unlink()

        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")

        # Load raw data
        self.raw_data = pd.read_csv(self.data_file)
        self.validator.validate_data(self.raw_data, self.target_column)

        # Prepare data
        self.raw_data = self.validator.prepare_datetime_column(self.raw_data)
        self.raw_data = self.validator.clean_data_columns(self.raw_data, self.target_column)

        # Auto-detect forecast start date
        last_date = cast(pd.Timestamp, self.raw_data["ds"].max())
        self.forecast_start_date: str = cast(pd.Timestamp, last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        logger.debug(f"Auto-detected forecast start date: {self.forecast_start_date}")

        # Use all data as training data
        self.training_data = self.raw_data.copy()
        logger.debug(f"Training data loaded: {len(self.training_data):,} rows")
        logger.debug(f"Date range: {self.training_data['ds'].min()} to {self.training_data['ds'].max()}")

        # Generate all possible country-marketplace combinations
        self.marketplace_combinations = self.get_marketplace_combinations()

    def get_marketplace_combinations(self) -> List[MarketplaceKey]:
        """Get all unique country-marketplace combinations"""
        combinations: pd.DataFrame = cast(
            pd.DataFrame,
            self.training_data[["country", "marketplace"]].drop_duplicates(),
        )
        combo_list: List[MarketplaceKey] = [
            (str(row["country"]), str(row["marketplace"])) for _, row in combinations.iterrows()
        ]
        logger.debug(f"Found {len(combo_list)} country-marketplace combinations")
        return combo_list

    def create_marketplace_timeseries(self, country: str, marketplace: str) -> pd.DataFrame:
        """Create aggregated time series for a country-marketplace combination"""
        combo_data: pd.DataFrame = cast(
            pd.DataFrame,
            self.training_data[
                (self.training_data["country"] == country) & (self.training_data["marketplace"] == marketplace)
            ],
        )

        if len(combo_data) == 0:
            return pd.DataFrame()

        # Aggregate by date across all brands and sellers
        ts_data: pd.DataFrame = combo_data.groupby("ds")[self.target_column].sum().reset_index()

        # Create complete date range and fill missing dates with 0
        if len(ts_data) > 0:
            date_range: pd.DatetimeIndex = pd.date_range(
                start=pd.to_datetime(ts_data["ds"].min()),
                end=pd.to_datetime(ts_data["ds"].max()),
                freq="D",
            )
            complete_ts: pd.DataFrame = pd.DataFrame({"ds": date_range})
            ts_data = complete_ts.merge(ts_data, on="ds", how="left")
            ts_data[self.target_column] = ts_data[self.target_column].fillna(0)

        # Rename columns for Prophet
        return cast(pd.DataFrame, ts_data[["ds", self.target_column]]).rename(columns={self.target_column: "y"})

    def calculate_brand_metrics(self, country: str, marketplace: str) -> Tuple[BrandRatios, BrandPrices]:
        """Calculate brand share ratios and average selling prices from last 30 days before forecast start date"""
        logger.debug(f"Calculating brand metrics for {country}-{marketplace}")

        marketplace_data: pd.DataFrame = cast(
            pd.DataFrame,
            self.training_data[
                (self.training_data["country"] == country) & (self.training_data["marketplace"] == marketplace)
            ],
        )

        if len(marketplace_data) == 0:
            logger.warning(f"  No data for {country}-{marketplace}")
            return {}, {}

        # Define the 30-day window before forecast start
        forecast_start: pd.Timestamp = pd.to_datetime(self.forecast_start_date)
        ratio_start: pd.Timestamp = cast(pd.Timestamp, forecast_start - pd.Timedelta(days=self.brand_ratio_window))

        recent_data: pd.DataFrame = cast(
            pd.DataFrame,
            marketplace_data[(marketplace_data["ds"] >= ratio_start) & (marketplace_data["ds"] < forecast_start)],
        )

        if len(recent_data) == 0:
            logger.debug(f"  No recent data for metrics calculation")
            return {}, {}

        return self._calculate_ratios_and_prices(recent_data)

    def _calculate_ratios_and_prices(self, recent_data: pd.DataFrame) -> Tuple[BrandRatios, BrandPrices]:
        """Calculate and normalize brand ratios and average selling prices"""
        # Check for sales in last 2 weeks first
        forecast_start = cast(pd.Timestamp, pd.to_datetime(self.forecast_start_date))
        last_n_days_start = cast(pd.Timestamp, forecast_start - pd.Timedelta(weeks=2))
        last_n_days_data = cast(pd.DataFrame, recent_data[recent_data["ds"] >= last_n_days_start])
        last_n_days_volume: float = float(last_n_days_data[self.target_column].sum())

        if last_n_days_volume == 0:
            logger.debug(f"  Zero total volume in last 2 weeks")
            return {}, {}

        # Group by brand and calculate totals
        brand_agg = (
            recent_data.groupby("fk_brand_used_id")
            .agg({self.target_column: "sum", self.revenue_column: "sum"})
            .reset_index()
        )

        total_marketplace_volume: float = float(brand_agg[self.target_column].sum())

        brand_ratios: BrandRatios = {}
        brand_prices: BrandPrices = {}

        for _, row in brand_agg.iterrows():
            brand = str(row["fk_brand_used_id"])
            volume = float(row[self.target_column])
            revenue = float(row[self.revenue_column])

            # Calculate ratio
            ratio = volume / total_marketplace_volume
            brand_ratios[brand] = ratio

            # Calculate weighted average selling price
            avg_price = revenue / volume if volume > 0 else 0.0
            brand_prices[brand] = avg_price

            logger.debug(f"    {brand}: {ratio:.3%} ({volume:,.0f} units) @ ${avg_price:.2f}")

        logger.debug(f"  Total brands: {len(brand_ratios)}")
        return brand_ratios, brand_prices

    def get_campaign_calendar(self, country: str, marketplace: str) -> Dict[str, CampaignInfo]:
        """Get or build campaign calendar for a specific (country, marketplace) combination"""
        key: MarketplaceKey = (country, marketplace)
        if key not in self.campaign_calendars:
            # Check if this country has been built
            country_built = any(k[0] == country for k in self.campaign_calendars.keys())

            if not country_built:
                self._build_country_campaigns(country)

        return self.campaign_calendars.get(key, {})

    def _build_country_campaigns(self, country: str) -> None:
        """Build campaign calendars for entire country"""
        # Get all marketplaces for this country
        marketplaces = list(set(mp for c, mp in self.marketplace_combinations if c == country))

        # Prepare timeseries data
        marketplace_data = {}
        for marketplace in marketplaces:
            ts_data = self.create_marketplace_timeseries(country, marketplace)
            if ts_data is not None and not ts_data.empty:
                marketplace_data[marketplace] = ts_data

        if marketplace_data:
            # Use the updated builder (now country-first)
            country_calendars = self.campaign_calendar_builder.build_campaign_calendar(country, marketplace_data)

            # Store results
            for marketplace, calendar in country_calendars.items():
                self.campaign_calendars[(country, marketplace)] = calendar

    def get_brand_ratios(self, country: str, marketplace: str) -> BrandRatios:
        """Get or calculate brand ratios for a specific (country, marketplace) combination"""
        key: MarketplaceKey = (country, marketplace)
        if key not in self.brand_ratios:
            self.brand_ratios[key], self.brand_prices[key] = self.calculate_brand_metrics(country, marketplace)
        return self.brand_ratios[key]

    def get_brand_prices(self, country: str, marketplace: str) -> BrandPrices:
        """Get or calculate brand prices for a specific (country, marketplace) combination"""
        key: MarketplaceKey = (country, marketplace)
        if key not in self.brand_prices:
            self.brand_ratios[key], self.brand_prices[key] = self.calculate_brand_metrics(country, marketplace)
        return self.brand_prices[key]

    def build_all_calendars_and_metrics(self) -> None:
        """Build campaign calendars and brand metrics for all marketplaces"""
        logger.debug("\n" + "=" * 60)
        logger.debug("BUILDING CAMPAIGN CALENDARS & BRAND METRICS")
        logger.debug("=" * 60)

        # Build campaigns by country
        countries = set(country for country, _ in self.marketplace_combinations)
        for country in tqdm(countries, desc="Building country campaigns"):
            self._build_country_campaigns(country)

        # Build brand metrics individually
        for country, marketplace in tqdm(self.marketplace_combinations, desc="Building brand metrics"):
            self.calculate_brand_metrics(country, marketplace)

    def prepare_all_data(self) -> List[MarketplaceKey]:
        """Execute all data preparation steps"""
        # Step 1: Load and prepare data
        self.load_and_prepare_data()

        # Step 2: Build campaign calendars and brand metrics
        self.build_all_calendars_and_metrics()

        return self.marketplace_combinations
