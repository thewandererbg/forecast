# Forecasting Pipeline

A modular forecasting system that generates marketplace-level predictions and breaks them down to brand level.

## Overview

**Pipeline Flow:** Country-Marketplace → Brand Breakdown

The system:
1. Loads and prepares data with brand ratios and campaign calendars
2. Generates marketplace-level forecasts using configurable models
3. Breaks down forecasts to brand level and exports results

## How to Run the Application

### Using pip
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py -m arima
```

### Using uv
```bash
uv run app.py -m arima
```

## Available Models

- `arima` - Auto ARIMA (default)
- `prophet` - Facebook Prophet
- `moving_average` - Simple moving average
- `xgboost` - XGBoost regressor
- `knn` - K-Nearest Neighbors

## Requirements

- Input file: `data.csv`
- Target column: `sum_quantity`
- Forecast horizon: 90 days

## Output Files

- `forecast.csv` - Main forecast results
- `campaign_calendars.csv` - Campaign calendar data
- `brand_ratios.csv` - Brand ratio breakdowns

## Module Structure

- **DataPreparation** - Data loading and preprocessing
- **ForecastingEngine** - Coordinates forecast generation
- **Forecasters** - Individual model implementations
- **ResultsExporter** - Handles output and brand breakdowns
