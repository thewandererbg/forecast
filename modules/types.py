from typing import Dict, List, Tuple, TypedDict, Union, Optional, Any, cast

import pandas as pd

# Type aliases
MarketplaceKey = Tuple[str, str]  # (country, marketplace)
BrandKey = Tuple[str, str, str]  # (country, marketplace, brand)
BrandRatios = Dict[str, float]  # brand_id -> ratio in country, marketplace
BrandPrices = Dict[str, float]  # brand_id -> average_selling_price in country, marketplace


# Proper TypedDict for CampaignInfo
class CampaignInfo(TypedDict):
    avg_uplift: float
    max_uplift: float
    min_uplift: float
    occurrences: int
    dates: List[pd.Timestamp]
    consistency: float
