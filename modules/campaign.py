import math
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, TypedDict, Union, Optional, Any, cast
import logging
from tqdm import tqdm
from collections import defaultdict

from .utils import DataValidator
from .types import CampaignInfo, MarketplaceKey, BrandKey, BrandRatios

logger = logging.getLogger(__name__)


class CampaignCalendarBuilder:
    def __init__(self) -> None:
        self.spike_detector: SpikeDetector = SpikeDetector()

    def build_campaign_calendar(
        self, country: str, marketplace_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, CampaignInfo]]:
        """Build campaign calendars for all marketplaces in a country with enhanced cross-marketplace logic"""
        logger.debug(f"Building campaign calendar for country {country} with {len(marketplace_data)} marketplaces")

        # Step 1: Collect patterns from all marketplaces
        all_marketplace_patterns = {}
        country_pattern_counts = defaultdict(int)

        for marketplace, ts_data in marketplace_data.items():
            logger.debug(f"Processing {country}-{marketplace}")

            spike_indices, _ = self.spike_detector.detect_spikes_adaptive(ts_data, type="adaptive")
            if not spike_indices:
                logger.debug(f"  No spikes detected for {country}-{marketplace}")
                all_marketplace_patterns[marketplace] = {}
                continue

            marketplace_patterns = self._process_campaign_patterns(ts_data, spike_indices)
            all_marketplace_patterns[marketplace] = marketplace_patterns

            # Count occurrences across country
            for month_day in marketplace_patterns.keys():
                country_pattern_counts[month_day] += len(marketplace_patterns[month_day]["dates"])

        # Step 2: Filter patterns using enhanced logic
        result = {}
        for marketplace, patterns in all_marketplace_patterns.items():
            filtered_calendar = self._filter_significant_patterns_enhanced(
                patterns, country_pattern_counts, f"{country}-{marketplace}"
            )
            result[marketplace] = filtered_calendar

        return result

    def _process_campaign_patterns(
        self, ts_data: pd.DataFrame, spike_indices: List[int]
    ) -> Dict[str, Dict[str, List[Any]]]:
        """Process spike patterns into campaign calendar"""
        campaign_patterns: Dict[str, Dict[str, List[Any]]] = {}

        for idx in spike_indices:
            spike_date_value = ts_data.iloc[idx]["ds"]
            if hasattr(spike_date_value, "iloc"):
                spike_date_value = spike_date_value.iloc[0]
            spike_date: pd.Timestamp = pd.to_datetime(spike_date_value)
            month_day: str = f"{spike_date.month:02d}-{spike_date.day:02d}"
            spike_value: float = float(ts_data.iloc[idx]["y"])

            # Calculate base value using surrounding non-spike days
            surrounding_values: List[float] = _get_surrounding_values(ts_data, idx, spike_indices)

            if surrounding_values:
                base_value: float = float(np.mean(surrounding_values))
                if base_value > 0:
                    raw_uplift: float = spike_value / base_value
                    # Apply log-based dampening to reduce uplift for small base values
                    reference_base: float = 200.0
                    dampen_factor: float = min(1.0, math.log(1 + base_value) / math.log(1 + reference_base))
                    uplift_factor: float = raw_uplift * dampen_factor

                    self._store_campaign_pattern(
                        campaign_patterns,
                        month_day,
                        uplift_factor,
                        spike_date,
                        spike_value,
                        base_value,
                    )

        return campaign_patterns

    def _store_campaign_pattern(
        self,
        patterns: Dict[str, Dict[str, List[Any]]],
        month_day: str,
        uplift_factor: float,
        spike_date: pd.Timestamp,
        spike_value: float,
        base_value: float,
    ) -> None:
        """Store campaign pattern data"""
        if month_day not in patterns:
            patterns[month_day] = {
                "uplifts": [],
                "dates": [],
                "spike_values": [],
                "base_values": [],
            }

        patterns[month_day]["uplifts"].append(uplift_factor)
        patterns[month_day]["dates"].append(spike_date)
        patterns[month_day]["spike_values"].append(spike_value)
        patterns[month_day]["base_values"].append(base_value)

    def _filter_significant_patterns_enhanced(
        self,
        campaign_patterns: Dict[str, Dict[str, List[Any]]],
        country_pattern_counts: Dict[str, int],
        marketplace_name: str,
    ) -> Dict[str, CampaignInfo]:
        """Enhanced filtering: include patterns with ≥2 occurrences in marketplace OR 1 in marketplace + ≥2 in country"""
        campaign_calendar: Dict[str, CampaignInfo] = {}

        for month_day, pattern_data in campaign_patterns.items():
            marketplace_occurrences: int = len(pattern_data["uplifts"])
            country_total_occurrences: int = country_pattern_counts[month_day]

            # Enhanced logic:
            # Include if: ≥2 occurrences in same marketplace OR (1 in marketplace AND ≥2 total in country)
            should_include = (
                marketplace_occurrences >= 2
                or (marketplace_occurrences == 1 and country_total_occurrences >= 2)
                or month_day in ("09-09", "10-10", "11-11", "12-12")
            )

            if should_include:
                uplifts: List[float] = pattern_data["uplifts"]
                dates: List[Any] = pattern_data["dates"]

                # Calculate weighted average uplift based on year ranking
                years = [d.year for d in dates]
                unique_years = sorted(set(years), reverse=True)  # Most recent first
                year_weights = {year: len(unique_years) - i for i, year in enumerate(unique_years)}

                weights = [year_weights[d.year] for d in dates]
                weighted_avg_uplift = float(np.average(uplifts, weights=weights))

                campaign_calendar[month_day] = CampaignInfo(
                    avg_uplift=weighted_avg_uplift,
                    max_uplift=float(np.max(uplifts)),
                    min_uplift=float(np.min(uplifts)),
                    occurrences=marketplace_occurrences,
                    dates=pattern_data["dates"],
                    consistency=float(1.0 / (1.0 + np.std(uplifts))),
                )

                # Enhanced logging
                date_list: List[str] = [d.strftime("%Y-%m-%d") for d in pattern_data["dates"]]
                avg_uplift: float = campaign_calendar[month_day]["avg_uplift"]
                inclusion_reason = (
                    f"≥2 in marketplace"
                    if marketplace_occurrences >= 2
                    else f"1 in marketplace + {country_total_occurrences} total in country"
                )
                logger.debug(
                    f"  {marketplace_name} {month_day}: {avg_uplift:.1f}x avg uplift ({marketplace_occurrences} times: {date_list}) - Included: {inclusion_reason}"
                )

        logger.debug(f"  {marketplace_name}: Found {len(campaign_calendar)} campaign patterns")
        return campaign_calendar


class SpikeDetector:
    """Detects spikes in time series data"""

    def __init__(self, window: int = 20, base_threshold: float = 4.0):
        self.window = window
        self.base_threshold = base_threshold

    def detect_spikes_adaptive(self, ts_data: pd.DataFrame, type: str) -> Tuple[List[int], pd.Series]:
        """Detect spikes using adaptive threshold based on rolling statistics"""
        if len(ts_data) < self.window:
            return [], pd.Series(dtype=float)

        y_values = pd.Series(ts_data["y"].copy())

        # Calculate rolling statistics
        rolling_median = y_values.rolling(
            window=self.window,
            center=True,
            min_periods=max(10, self.window // 3),
        ).median()

        rolling_mad = y_values.rolling(window=self.window, center=True, min_periods=max(10, self.window // 3)).apply(
            lambda x: np.median(np.abs(x - np.median(x)))
        )

        # Forward/backward fill for edge cases, but don't use fillna(0)
        rolling_median = rolling_median.bfill().ffill()
        rolling_mad = rolling_mad.bfill().ffill()

        # Handle remaining NaN values by using global statistics
        global_median = y_values.median()
        global_mad = np.median(np.abs(y_values - global_median))

        rolling_median = rolling_median.fillna(global_median)
        rolling_mad = rolling_mad.fillna(global_mad)

        # Much more conservative approach - use percentile-based thresholds
        # Calculate what constitutes "normal" variation
        rolling_std = (
            y_values.rolling(window=self.window, center=True, min_periods=max(10, self.window // 3))
            .std()
            .bfill()
            .ffill()
            .fillna(y_values.std())
        )

        # Use a fixed multiplier approach - simpler and more predictable
        spike_threshold = rolling_median + self.base_threshold * rolling_std

        # Additional safety checks
        valid_detection_mask = (
            (~rolling_median.isna())
            & (~rolling_std.isna())
            & (rolling_std > 1e-6)  # Avoid detection in flat regions
            & (y_values > rolling_median + 2 * rolling_std)  # Must be 2+ standard deviations
        )

        # Meaningful spike detection: both relative AND absolute thresholds
        absolute_increase = y_values - rolling_median
        relative_increase = absolute_increase / (rolling_median + 1e-6)  # Avoid division by zero
        meaningful_spike = (
            (relative_increase > 1)  # 100% increase
            & (absolute_increase > 100)  # AND at least 100 units
            & valid_detection_mask
        )

        special_spike_indices = self._detect_special_date(ts_data)
        is_spike = meaningful_spike

        for idx in special_spike_indices:
            if idx < len(is_spike):
                is_spike.iloc[idx] = True
                spike_threshold.iloc[idx] = y_values.iloc[idx]

        spike_indices = cast(pd.Index, y_values.index[is_spike]).tolist()

        # Set threshold to NaN where detection isn't valid
        spike_threshold[~valid_detection_mask] = np.nan

        return spike_indices, spike_threshold

    def _detect_special_date(self, ts_data: pd.DataFrame) -> List[int]:
        """Detect 6/5, 6/6 and same-day-same-month patterns"""
        dates = pd.to_datetime(ts_data["ds"])
        same_mask = (dates.dt.day == dates.dt.month) & (dates.dt.month <= 12)
        consecutive_mask = (dates.dt.day == dates.dt.month - 1) & (dates.dt.month >= 2)
        combined_mask = same_mask | consecutive_mask
        return ts_data.index[combined_mask].tolist()

    def create_spike_dampened_series(self, ts_data: pd.DataFrame, spike_indices: List[int]) -> pd.DataFrame:
        """Remove spike data points entirely from the time series"""
        if not spike_indices:
            return ts_data

        # Simply drop the rows with spike indices
        cleaned_data = ts_data.drop(index=spike_indices).reset_index(drop=True)
        return cleaned_data


def _get_surrounding_values(ts_data: pd.DataFrame, spike_idx: int, spike_indices: List[int]) -> List[float]:
    """Get surrounding non-spike values for baseline calculation"""
    start_idx: int = max(0, spike_idx - 60)  # 30-day window
    end_idx: int = min(len(ts_data), spike_idx + 60)

    return [float(ts_data.iloc[i]["y"]) for i in range(start_idx, end_idx) if i not in spike_indices and i != spike_idx]


def apply_campaign_factors(
    forecast_period: pd.DataFrame,
    campaign_calendar: Dict[str, CampaignInfo],
    country: str,
    marketplace: str,
    max_uplift: float = 20.0,
) -> pd.DataFrame:
    """
    Apply additional campaign factors to forecast data

    Args:
        forecast_period: DataFrame with forecast data (ds, yhat, yhat_lower, yhat_upper)
        campaign_calendar: Dictionary of campaign information by date (MM-DD format)
        country: Country identifier for logging
        marketplace: Marketplace identifier for logging
        max_uplift: Maximum allowed uplift factor

    Returns:
        DataFrame with campaign adjustments applied
    """
    if not campaign_calendar or forecast_period.empty:
        return forecast_period

    applied_campaigns = []
    adjusted_forecast = forecast_period.copy()

    for idx, row in adjusted_forecast.iterrows():
        forecast_date = cast(pd.Timestamp, row["ds"])
        month_day = f"{forecast_date.month:02d}-{forecast_date.day:02d}"

        if month_day in campaign_calendar:
            campaign_info = campaign_calendar[month_day]
            occurrences = campaign_info["occurrences"]

            # Conservative uplift application
            uplift_factor = min(max_uplift, campaign_info["avg_uplift"])

            # Apply factor if meaningful
            if uplift_factor >= 1.5:
                current_position = cast(int, adjusted_forecast.index.get_loc(idx))
                start_idx = max(0, current_position - 30)

                # Filter out campaign days from baseline calculation
                baseline_days = []
                baseline_interval_days = []
                for i in range(start_idx, current_position):
                    day_date = cast(pd.Timestamp, adjusted_forecast.iloc[i]["ds"]).strftime("%m-%d")
                    if day_date not in campaign_calendar:
                        baseline_days.append(adjusted_forecast.iloc[i]["yhat"])
                        baseline_interval_days.append(
                            adjusted_forecast.loc[idx, "yhat_upper"] - adjusted_forecast.loc[idx, "yhat_lower"]
                        )

                if baseline_days:  # Ensure we have baseline days
                    average_prev_nday = sum(baseline_days) / len(baseline_days)
                    average_prev_interval = sum(baseline_interval_days) / len(baseline_interval_days)

                    # Calculate interval width and update all values
                    new_yhat = uplift_factor * average_prev_nday
                    new_interval = average_prev_interval * uplift_factor

                    adjusted_forecast.loc[idx, "yhat"] = new_yhat
                    adjusted_forecast.loc[idx, "yhat_lower"] = new_yhat - new_interval / 2
                    adjusted_forecast.loc[idx, "yhat_upper"] = new_yhat + new_interval / 2

                    applied_campaigns.append(
                        {
                            "date": month_day,
                            "factor": uplift_factor,
                            "occurrences": occurrences,
                            "average_prev_nday": average_prev_nday,
                        }
                    )

    # Log applied additional factors
    if applied_campaigns:
        logger.debug(f"  Applied campaign factors for {country}-{marketplace}:")
        for campaign in applied_campaigns:
            logger.debug(f"    {campaign['date']}: +{campaign['factor']:.1f}x additional")

    return adjusted_forecast
