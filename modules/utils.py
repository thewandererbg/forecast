"""
Utility functions for the forecasting pipeline
"""

import math
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, cast
import logging

from .types import CampaignInfo

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates data integrity for forecasting pipeline"""

    @staticmethod
    def validate_data(data: pd.DataFrame, target_column: str) -> None:
        """Validate that required columns exist"""
        required_columns: List[str] = [
            "country",
            "marketplace",
            "fk_brand_used_id",
            "day",
            target_column,
        ]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    @staticmethod
    def prepare_datetime_column(data: pd.DataFrame) -> pd.DataFrame:
        """Convert day column to datetime and handle invalid dates"""
        data = data.copy()
        data["ds"] = pd.to_datetime(data["day"], errors="coerce")

        invalid_dates_mask = data["ds"].isna()
        if invalid_dates_mask.sum() > 0:
            logger.warning(f"Removing {invalid_dates_mask.sum()} rows with invalid dates")
            data = pd.DataFrame(data[~invalid_dates_mask])

        return data

    @staticmethod
    def clean_data_columns(data: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Clean and standardize data columns"""
        data = data.copy()
        data["fk_brand_used_id"] = data["fk_brand_used_id"].fillna("unknown")
        data[target_column] = pd.to_numeric(data[target_column], errors="coerce")
        data[target_column] = data[target_column].fillna(0)
        return data
