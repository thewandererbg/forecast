import io
import math
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union, cast
import warnings

warnings.filterwarnings("ignore")

# Configure Streamlit page
st.set_page_config(
    page_title="📈 E-commerce Forecast Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


class ForecastDashboard:
    """
    E-commerce Forecast Visualization Dashboard using Streamlit and Plotly
    """

    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.metric_names = {
            "sum_quantity": "Units Sold",
            "sum_s_net": "Revenue",
            # "sum_s_seller_selling": "Seller Selling Amount",
        }
        self.aggregation_options = {"daily": "Daily", "weekly": "Weekly", "monthly": "Monthly", "yearly": "Yearly"}

    # @st.cache_data
    def load_forecast_data(_self, filepath: str = "forecast.csv") -> pd.DataFrame:
        """
        Load and preprocess forecast data with caching

        Args:
            filepath: Path to the forecast CSV file

        Returns:
            Processed DataFrame
        """
        try:
            # Load the data
            df = pd.read_csv(filepath)

            # Basic data validation
            required_columns = [
                "day",
                "country",
                "marketplace",
                "seller name",
                "fk_brand_used_id",
                "sum_quantity",
                "sum_s_net",
                "sum_s_seller_selling",
                "is_forecast",
            ]

            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                st.error(f"❌ Missing required columns: {missing_cols}")
                st.stop()

            # Convert and validate data types
            df["day"] = pd.to_datetime(df["day"], errors="coerce")
            df = df.dropna(subset=["day"])  # Remove rows with invalid dates

            # Convert boolean forecast flag
            df["is_forecast"] = df["is_forecast"].fillna(False).astype(bool)

            # Convert numeric columns
            numeric_cols = ["sum_quantity", "sum_s_net", "sum_s_seller_selling"]
            for col in numeric_cols:
                df[col] = cast(pd.Series, pd.to_numeric(df[col], errors="coerce")).fillna(0)

            # Handle confidence interval columns if they exist
            confidence_cols = []
            for metric in numeric_cols:
                lower_col = f"{metric}_lower"
                upper_col = f"{metric}_upper"
                if lower_col in df.columns:
                    df[lower_col] = cast(pd.Series, pd.to_numeric(df[lower_col], errors="coerce")).fillna(0)
                    confidence_cols.append(lower_col)
                if upper_col in df.columns:
                    df[upper_col] = cast(pd.Series, pd.to_numeric(df[upper_col], errors="coerce")).fillna(0)
                    confidence_cols.append(upper_col)

            # Clean string columns
            string_cols = ["country", "marketplace", "seller name", "fk_brand_used_id"]
            for col in string_cols:
                if col in df.columns:
                    df[col] = df[col].astype(str).fillna("Unknown")

            # Sort by date
            df = df.sort_values("day").reset_index(drop=True)

            return df

        except FileNotFoundError:
            st.error(
                f"❌ File '{filepath}' not found. Please ensure the forecast file exists in the current directory."
            )
            st.stop()
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            st.stop()

    def aggregate_data_by_time_period(self, df: pd.DataFrame, aggregation: str, metric: str) -> pd.DataFrame:
        """
        Aggregate data by specified time period

        Args:
            df: Input DataFrame
            aggregation: Time aggregation level ('daily', 'weekly', 'monthly', 'yearly')
            metric: Metric column to aggregate

        Returns:
            Aggregated DataFrame
        """
        if df.empty:
            return df

        # Create a copy to avoid modifying original data
        agg_df = df.copy()

        # Create aggregation period column
        if aggregation == "daily":
            agg_df["period"] = agg_df["day"]
            period_format = "%Y-%m-%d"
        elif aggregation == "weekly":
            # Use Monday as start of week
            agg_df["period"] = agg_df["day"].dt.to_period("W-MON").dt.start_time
            period_format = "%Y-W%U"
        elif aggregation == "monthly":
            agg_df["period"] = agg_df["day"].dt.to_period("M").dt.start_time
            period_format = "%Y-%m"
        elif aggregation == "yearly":
            agg_df["period"] = agg_df["day"].dt.to_period("Y").dt.start_time
            period_format = "%Y"
        else:
            # Default to daily
            agg_df["period"] = agg_df["day"]
            period_format = "%Y-%m-%d"

        # Determine aggregation columns
        agg_columns = {metric: "sum"}

        # Add confidence interval columns if they exist
        lower_col = f"{metric}_lower"
        upper_col = f"{metric}_upper"
        if lower_col in agg_df.columns:
            agg_columns[lower_col] = "sum"
        if upper_col in agg_df.columns:
            agg_columns[upper_col] = "sum"

        # Group by period and forecast flag, then aggregate
        try:
            aggregated = cast(pd.DataFrame, agg_df.groupby(["period", "is_forecast"], as_index=False).agg(agg_columns))

            # Rename period back to day for consistency
            aggregated = aggregated.rename(columns={"period": "day"})

            # Sort by date
            aggregated = aggregated.sort_values("day").reset_index(drop=True)

            return aggregated

        except Exception as e:
            st.error(f"Error in time aggregation: {e}")
            return pd.DataFrame()

    def create_filters_sidebar(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Create interactive filters in the sidebar

        Args:
            df: Input DataFrame

        Returns:
            Dictionary containing all filter selections
        """
        st.sidebar.title("🎛️ Dashboard Filters")

        # Time aggregation selection
        st.sidebar.subheader("⏱️ Time Aggregation")
        selected_aggregation = st.sidebar.selectbox(
            "Aggregate Data By:",
            options=list(self.aggregation_options.keys()),
            format_func=lambda x: self.aggregation_options[x],
            index=1,  # Default to daily
            key="aggregation_filter",
        )

        # Date range filter
        st.sidebar.subheader("📅 Time Period")
        min_date = df["day"].min().date()
        max_date = df["day"].max().date()

        # Create two columns for date inputs
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input(
                "From:",
                value=min_date,
                min_value=min_date,
                max_value=max_date,
                key="start_date",
            )
        with col2:
            end_date = st.date_input(
                "To:",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
                key="end_date",
            )

        # filters
        st.sidebar.subheader("🔍 Filters")
        all_countries = sorted(df["country"].unique())
        selected_countries = st.sidebar.multiselect(
            "Select Countries:", options=all_countries, default=[], key="country_filter"
        )

        # Filter data for next level
        country_filtered = cast(pd.DataFrame, df[df["country"].isin(selected_countries)] if selected_countries else df)

        # Marketplace filter
        all_marketplaces = sorted(country_filtered["marketplace"].unique())
        selected_marketplaces = st.sidebar.multiselect(
            "Select Marketplaces:",
            options=all_marketplaces,
            default=[],
            key="marketplace_filter",
        )

        # Filter data for next level
        marketplace_filtered = cast(
            pd.DataFrame,
            country_filtered[country_filtered["marketplace"].isin(selected_marketplaces)]
            if selected_marketplaces
            else country_filtered,
        )

        # Brand filter
        all_brands = sorted(marketplace_filtered["fk_brand_used_id"].unique())
        # Default to first 5 brands to avoid overwhelming UI
        selected_brands = st.sidebar.multiselect("Select Brands:", options=all_brands, default=[], key="brand_filter")

        # Metric selection
        st.sidebar.subheader("📊 Metrics")
        selected_metric = st.sidebar.selectbox(
            "Select Metric to Visualize:",
            options=list(self.metric_names.keys()),
            format_func=lambda x: self.metric_names[x],
            key="metric_filter",
        )

        # Forecast options
        st.sidebar.subheader("🔮 Forecast Options")
        show_confidence = st.sidebar.checkbox("Show Confidence Intervals", value=True, key="show_confidence")

        self.create_data_download_section(df)

        return {
            "countries": selected_countries,
            "marketplaces": selected_marketplaces,
            "brands": selected_brands,
            "metric": selected_metric,
            "metric_name": self.metric_names[selected_metric],
            "aggregation": selected_aggregation,
            "aggregation_name": self.aggregation_options[selected_aggregation],
            "start_date": start_date,
            "end_date": end_date,
            "show_confidence": show_confidence,
        }

    def create_data_download_section(self, df: pd.DataFrame) -> None:
        """
        Create data download section in sidebar for full dataset

        Args:
            df: Full DataFrame to download
        """
        st.sidebar.subheader("💾 Data Export")

        if df.empty:
            st.sidebar.warning("No data to download")
            return

        # Display download info
        st.sidebar.write(f"**Total Records:** {len(df):,}")

        @st.cache_data
        def convert_to_csv(data):
            return data.to_csv(index=False)

        # CSV Download
        csv_data = convert_to_csv(df)
        st.sidebar.download_button(
            label="📄 CSV",
            data=csv_data,
            file_name="forecast_data.csv",
            mime="text/csv",
            use_container_width=False,
            help="Download as CSV file",
        )

    def apply_filters_to_data(self, df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply all selected filters to the dataset

        Args:
            df: Input DataFrame
            filters: Filter selections dictionary

        Returns:
            Filtered DataFrame
        """
        filtered_df = df.copy()

        # Apply categorical filters
        if filters["countries"]:
            filtered_df = cast(pd.DataFrame, filtered_df[filtered_df["country"].isin(filters["countries"])])
        if filters["marketplaces"]:
            filtered_df = cast(pd.DataFrame, filtered_df[filtered_df["marketplace"].isin(filters["marketplaces"])])
        if filters["brands"]:
            filtered_df = cast(pd.DataFrame, filtered_df[filtered_df["fk_brand_used_id"].isin(filters["brands"])])

        # Apply date range filter
        start_ts = pd.Timestamp(filters["start_date"])
        end_ts = pd.Timestamp(filters["end_date"])
        filtered_df = cast(pd.DataFrame, filtered_df[(filtered_df["day"] >= start_ts) & (filtered_df["day"] <= end_ts)])

        return filtered_df

    def create_time_series_chart(self, df: pd.DataFrame, filters: Dict[str, Any]) -> None:
        """
        Create the main time series visualization

        Args:
            df: Filtered DataFrame
            filters: Filter selections
        """
        if df.empty:
            st.warning("⚠️ No data to display")
            return

        st.subheader(f"📈 {filters['metric_name']} Time Series ({filters['aggregation_name']})")

        try:
            # Ensure we have the confidence interval columns, create them if missing
            lower_col = f"{filters['metric']}_lower"
            upper_col = f"{filters['metric']}_upper"

            if lower_col not in df.columns:
                df[lower_col] = 0.0
            if upper_col not in df.columns:
                df[upper_col] = 0.0

            Z_SCORE = 1.96  # For 95% CI
            if filters["aggregation"] == "daily":
                # Convert bounds to standard deviations
                df["std"] = (df[upper_col] - df[lower_col]) / (2 * Z_SCORE)

                daily_agg = cast(
                    pd.DataFrame,
                    df.groupby(["day", "is_forecast"], as_index=False).agg(
                        {filters["metric"]: "sum", "std": lambda x: np.sqrt((x**2).sum())}
                    ),
                )

                # Convert back to 80% confidence intervals
                daily_agg[lower_col] = np.maximum(0, daily_agg[filters["metric"]] - Z_SCORE * daily_agg["std"])
                daily_agg[upper_col] = daily_agg[filters["metric"]] + Z_SCORE * daily_agg["std"]

            else:
                # First, round dates to period boundaries
                df_copy = df.copy()
                df_copy["std"] = (df_copy[upper_col] - df_copy[lower_col]) / (2 * Z_SCORE)

                if filters["aggregation"] == "yearly":
                    df_copy["day"] = df_copy["day"].dt.to_period("Y").dt.start_time
                elif filters["aggregation"] == "monthly":
                    df_copy["day"] = df_copy["day"].dt.to_period("M").dt.start_time
                elif filters["aggregation"] == "weekly":
                    df_copy["day"] = df_copy["day"].dt.to_period("W").dt.start_time

                # Aggregate using variance-based approach
                daily_agg = cast(
                    pd.DataFrame,
                    df_copy.groupby("day", as_index=False).agg(
                        {
                            filters["metric"]: "sum",
                            "std": lambda x: np.sqrt((x**2).sum()),  # Sum variances, then sqrt
                            "is_forecast": "any",
                        }
                    ),
                )

                # Convert back to confidence intervals
                daily_agg[lower_col] = np.maximum(daily_agg[filters["metric"]] - Z_SCORE * daily_agg["std"], 0)
                daily_agg[upper_col] = daily_agg[filters["metric"]] + Z_SCORE * daily_agg["std"]
                daily_agg = daily_agg.drop(columns=["std"])

            # Split historical and forecast with explicit copies
            historical = cast(pd.DataFrame, daily_agg[daily_agg["is_forecast"] == False]).copy()
            forecast = cast(pd.DataFrame, daily_agg[daily_agg["is_forecast"] == True]).copy()

            # Sort by date to ensure proper ordering
            if not historical.empty:
                historical = historical.sort_values("day").reset_index(drop=True)
            if not forecast.empty:
                forecast = forecast.sort_values("day").reset_index(drop=True)

            # Create the plot
            fig = go.Figure()

            # Add historical data line
            if not historical.empty:
                # Format hover text based on aggregation
                hover_template = (
                    f"<b>Historical Data</b><br>Period: %{{x}}<br>{filters['metric_name']}: %{{y:,.0f}}<extra></extra>"
                )

                fig.add_trace(
                    go.Scatter(
                        x=historical["day"],
                        y=historical[filters["metric"]],
                        mode="lines+markers" if filters["aggregation"] != "daily" else "lines",
                        name="Historical",
                        line=dict(color="#2E86AB", width=2),
                        marker=dict(size=4) if filters["aggregation"] != "daily" else None,
                        hovertemplate=hover_template,
                    )
                )

            # Add forecast data line
            if not forecast.empty:
                hover_template = (
                    f"<b>Forecast</b><br>Period: %{{x}}<br>{filters['metric_name']}: %{{y:,.0f}}<extra></extra>"
                )

                fig.add_trace(
                    go.Scatter(
                        x=forecast["day"],
                        y=forecast[filters["metric"]],
                        mode="lines+markers" if filters["aggregation"] != "daily" else "lines",
                        name="Forecast",
                        line=dict(color="#F18F01", width=2, dash="dash"),
                        marker=dict(size=4) if filters["aggregation"] != "daily" else None,
                        hovertemplate=hover_template,
                    )
                )

                # Add confidence intervals if requested and available
                if filters["show_confidence"] and not forecast.empty:
                    lower_col = f"{filters['metric']}_lower"
                    upper_col = f"{filters['metric']}_upper"

                    if lower_col in forecast.columns and upper_col in forecast.columns:
                        # Create confidence interval traces
                        forecast_sorted = forecast.sort_values("day").reset_index(drop=True)

                        # Upper bound (invisible line)
                        fig.add_trace(
                            go.Scatter(
                                x=forecast_sorted["day"],
                                y=forecast_sorted[upper_col],
                                mode="lines",
                                line=dict(width=0, color="rgba(241, 143, 1, 0)"),
                                showlegend=False,
                                hoverinfo="skip",
                                name="upper_bound",
                            )
                        )

                        # Lower bound with fill to upper
                        fig.add_trace(
                            go.Scatter(
                                x=forecast_sorted["day"],
                                y=forecast_sorted[lower_col],
                                mode="lines",
                                line=dict(width=0, color="rgba(241, 143, 1, 0)"),
                                fill="tonexty",
                                fillcolor="rgba(241, 143, 1, 0.2)",
                                name="Confidence Interval",
                                hovertemplate=(
                                    f"<b>Confidence Interval</b><br>Period: %{{x}}<br>Lower: %{{y:,.0f}}<extra></extra>"
                                ),
                            )
                        )

            # Update layout with appropriate title
            chart_title = f"{filters['metric_name']} - Historical vs Forecast ({filters['aggregation_name']})"

            fig.update_layout(
                title=chart_title,
                xaxis_title="Date",
                yaxis_title=filters["metric_name"],
                hovermode="x unified",
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )

            # # Show aggregation summary
            # total_periods = len(daily_agg)
            # hist_periods = len(historical)
            # forecast_periods = len(forecast)

            # col1, col2, col3 = st.columns(3)
            # with col1:
            #     st.metric("Total Periods", total_periods)
            # with col2:
            #     st.metric("Historical Periods", hist_periods)
            # with col3:
            #     st.metric("Forecast Periods", forecast_periods)

            st.plotly_chart(fig, width="stretch")

        except Exception as e:
            st.error(f"❌ Error creating time series chart: {str(e)}")
            # Show the error details in expander for debugging
            with st.expander("🔍 Error Details (for debugging)"):
                st.code(str(e))
                st.write("Data types:")
                st.write(df.dtypes)
                if not df.empty:
                    st.write("Sample data:")
                    st.write(df.head())

    def create_sales_table_by_location(self, df: pd.DataFrame, filters: Dict[str, Any]) -> None:
        """
        Create a pivot table showing sales by country, marketplace and time periods

        Args:
            df: Filtered DataFrame
            filters: Filter selections including aggregation and metric
        """
        if df.empty:
            st.warning("⚠️ No data to display")
            return

        st.subheader(f"📊 {filters['metric_name']} by Location & Time ({filters['aggregation_name']})")

        try:
            # Create a copy for processing
            df_pivot = df.copy()

            # Aggregate data by time period and format as 'yyyy-mm-dd'
            if filters["aggregation"] == "daily":
                df_pivot["time_period"] = df_pivot["day"].dt.strftime("%Y-%m-%d")
            elif filters["aggregation"] == "weekly":
                df_pivot["time_period"] = df_pivot["day"].dt.to_period("W").dt.start_time.dt.strftime("%Y-%m-%d")
            elif filters["aggregation"] == "monthly":
                df_pivot["time_period"] = df_pivot["day"].dt.to_period("M").dt.start_time.dt.strftime("%Y-%m-%d")
            elif filters["aggregation"] == "yearly":
                df_pivot["time_period"] = df_pivot["day"].dt.to_period("Y").dt.start_time.dt.strftime("%Y-%m-%d")

            # Aggregate the data by country, marketplace, and time period
            pivot_data = df_pivot.groupby(["country", "marketplace", "time_period"], as_index=False)[
                filters["metric"]
            ].sum()

            # Create pivot table
            pivot_table = pivot_data.pivot_table(
                index=["country", "marketplace"],
                columns="time_period",
                values=filters["metric"],
                fill_value=0,
                aggfunc="sum",
            )

            # Sort columns by date (newest first)
            pivot_table = pivot_table.reindex(sorted(pivot_table.columns, reverse=True), axis=1)

            # Add total column for each country
            pivot_table["Total"] = pivot_table.sum(axis=1)

            # Round and format numbers
            pivot_table = pivot_table.round(0).astype(int)

            # Display the pivot table
            st.dataframe(pivot_table, width="stretch")

        except Exception as e:
            st.error(f"❌ Error creating location table: {str(e)}")
            with st.expander("🔍 Error Details (for debugging)"):
                st.code(str(e))
                st.write("Data types:")
                st.write(df.dtypes)
                if not df.empty:
                    st.write("Sample data:")
                    st.write(df.head())

    def create_share_charts(self, df: pd.DataFrame, filters: Dict[str, Any]) -> None:
        """
        Create pie charts showing share by country and marketplace with time series below

        Args:
            df: Filtered DataFrame
            filters: Filter selections including metric and aggregation
        """
        if df.empty:
            st.warning("⚠️ No data to display")
            return

        st.subheader(f"📊 {filters['metric_name']} Share Analysis")

        try:
            # Prepare time-aggregated data
            df_time = df.copy()
            if filters["aggregation"] == "yearly":
                df_time["day"] = df_time["day"].dt.to_period("Y").dt.start_time
            elif filters["aggregation"] == "monthly":
                df_time["day"] = df_time["day"].dt.to_period("M").dt.start_time
            elif filters["aggregation"] == "weekly":
                df_time["day"] = df_time["day"].dt.to_period("W").dt.start_time
            # daily aggregation doesn't need transformation

            # Create three columns for side-by-side charts
            col1, col2, col3 = st.columns(3)

            # Country share analysis
            with col1:
                # Country donut chart
                country_data = df.groupby("country")[filters["metric"]].sum().reset_index()
                country_data = country_data.sort_values(filters["metric"], ascending=False)

                fig_country = go.Figure(
                    data=[
                        go.Pie(
                            labels=country_data["country"],
                            values=country_data[filters["metric"]],
                            hole=0.3,
                            textinfo="label+percent",
                            textposition="auto",
                            hovertemplate="<b>%{label}</b><br>Value: %{value:,.0f}<br>Share: %{percent}<extra></extra>",
                        )
                    ]
                )

                fig_country.update_layout(
                    title=f"{filters['metric_name']} Share by Country",
                    height=400,
                    showlegend=True,
                    legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05),
                )

                st.plotly_chart(fig_country, use_container_width=True)

                # Country time series bar chart
                country_time_data = df_time.groupby(["day", "country"])[filters["metric"]].sum().reset_index()

                # Calculate percentage share for each day
                daily_totals = country_time_data.groupby("day")[filters["metric"]].sum().reset_index()
                daily_totals.columns = ["day", "total"]
                country_time_data = country_time_data.merge(daily_totals, on="day")
                country_time_data["percentage"] = (
                    country_time_data[filters["metric"]] / country_time_data["total"]
                ) * 100

                fig_country_time = go.Figure()

                for country in country_data["country"]:
                    country_subset = country_time_data[country_time_data["country"] == country]
                    fig_country_time.add_trace(
                        go.Bar(
                            x=country_subset["day"],
                            y=country_subset["percentage"],
                            name=country,
                            hovertemplate=f"<b>{country}</b><br>Date: %{{x}}<br>Share: %{{y:.1f}}%<br>Value: %{{customdata:,.0f}}<extra></extra>",
                            customdata=country_subset[filters["metric"]],
                        )
                    )

                fig_country_time.update_layout(
                    title=f"Country Share Over Time ({filters['aggregation'].title()})",
                    xaxis_title="Date",
                    yaxis_title="Share (%)",
                    height=500,
                    barmode="stack",
                    showlegend=False,
                    yaxis=dict(range=[0, 100]),
                )

                st.plotly_chart(fig_country_time, use_container_width=True)

            # Marketplace share analysis
            with col2:
                # Marketplace donut chart
                marketplace_data = df.groupby("marketplace")[filters["metric"]].sum().reset_index()
                marketplace_data = marketplace_data.sort_values(filters["metric"], ascending=False)

                fig_marketplace = go.Figure(
                    data=[
                        go.Pie(
                            labels=marketplace_data["marketplace"],
                            values=marketplace_data[filters["metric"]],
                            hole=0.3,
                            textinfo="label+percent",
                            textposition="auto",
                            hovertemplate="<b>%{label}</b><br>Value: %{value:,.0f}<br>Share: %{percent}<extra></extra>",
                        )
                    ]
                )

                fig_marketplace.update_layout(
                    title=f"{filters['metric_name']} Share by Marketplace",
                    height=400,
                    showlegend=True,
                    legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05),
                )

                st.plotly_chart(fig_marketplace, use_container_width=True)

                # Marketplace time series bar chart
                marketplace_time_data = df_time.groupby(["day", "marketplace"])[filters["metric"]].sum().reset_index()

                # Calculate percentage share for each day
                marketplace_time_data = marketplace_time_data.merge(daily_totals, on="day")
                marketplace_time_data["percentage"] = (
                    marketplace_time_data[filters["metric"]] / marketplace_time_data["total"]
                ) * 100

                fig_marketplace_time = go.Figure()

                for marketplace in marketplace_data["marketplace"]:
                    marketplace_subset = marketplace_time_data[marketplace_time_data["marketplace"] == marketplace]
                    fig_marketplace_time.add_trace(
                        go.Bar(
                            x=marketplace_subset["day"],
                            y=marketplace_subset["percentage"],
                            name=marketplace,
                            hovertemplate=f"<b>{marketplace}</b><br>Date: %{{x}}<br>Share: %{{y:.1f}}%<br>Value: %{{customdata:,.0f}}<extra></extra>",
                            customdata=marketplace_subset[filters["metric"]],
                        )
                    )

                fig_marketplace_time.update_layout(
                    title=f"Marketplace Share Over Time ({filters['aggregation'].title()})",
                    xaxis_title="Date",
                    yaxis_title="Share (%)",
                    height=500,
                    barmode="stack",
                    showlegend=False,
                    yaxis=dict(range=[0, 100]),
                )

                st.plotly_chart(fig_marketplace_time, use_container_width=True)

            # Brand share analysis
            with col3:
                # Check if brand column exists
                if "fk_brand_used_id" in df.columns:
                    # Brand donut chart
                    brand_data = df.groupby("fk_brand_used_id")[filters["metric"]].sum().reset_index()
                    brand_data = brand_data.sort_values(filters["metric"], ascending=False)

                    fig_brand = go.Figure(
                        data=[
                            go.Pie(
                                labels=brand_data["fk_brand_used_id"],
                                values=brand_data[filters["metric"]],
                                hole=0.3,
                                textinfo="label+percent",
                                textposition="auto",
                                hovertemplate="<b>%{label}</b><br>Value: %{value:,.0f}<br>Share: %{percent}<extra></extra>",
                            )
                        ]
                    )

                    fig_brand.update_layout(
                        title=f"{filters['metric_name']} Share by Brand",
                        height=400,
                        showlegend=True,
                        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05),
                    )

                    st.plotly_chart(fig_brand, use_container_width=True)

                    # Brand time series bar chart
                    brand_time_data = (
                        df_time.groupby(["day", "fk_brand_used_id"])[filters["metric"]].sum().reset_index()
                    )

                    # Calculate percentage share for each day
                    brand_time_data = brand_time_data.merge(daily_totals, on="day")
                    brand_time_data["percentage"] = (
                        brand_time_data[filters["metric"]] / brand_time_data["total"]
                    ) * 100

                    fig_brand_time = go.Figure()

                    for brand in brand_data["fk_brand_used_id"]:
                        brand_subset = brand_time_data[brand_time_data["fk_brand_used_id"] == brand]
                        fig_brand_time.add_trace(
                            go.Bar(
                                x=brand_subset["day"],
                                y=brand_subset["percentage"],
                                name=brand,
                                hovertemplate=f"<b>{brand}</b><br>Date: %{{x}}<br>Share: %{{y:.1f}}%<br>Value: %{{customdata:,.0f}}<extra></extra>",
                                customdata=brand_subset[filters["metric"]],
                            )
                        )

                    fig_brand_time.update_layout(
                        title=f"Brand Share Over Time ({filters['aggregation'].title()})",
                        xaxis_title="Date",
                        yaxis_title="Share (%)",
                        height=500,
                        barmode="stack",
                        showlegend=False,
                        yaxis=dict(range=[0, 100]),
                    )

                    st.plotly_chart(fig_brand_time, use_container_width=True)
                else:
                    st.warning("⚠️ Brand data not available")

        except Exception as e:
            st.error(f"❌ Error creating share charts: {str(e)}")
            with st.expander("🔍 Error Details (for debugging)"):
                st.code(str(e))
                st.write("Data types:")
                st.write(df.dtypes)
                if not df.empty:
                    st.write("Sample data:")
                    st.write(df.head())

    def create_country_performance_heatmaps(self, df: pd.DataFrame, filters: Dict[str, Any]) -> None:
        """
        Create side-by-side heatmaps showing performance matrix of:
        1. Country vs Marketplace
        2. Country vs Brand
        3. Marketplace vs Brand

        Args:
            df: Filtered DataFrame
            filters: Filter selections including metric
        """
        if df.empty:
            st.warning("⚠️ No data to display")
            return

        st.subheader(f"🔥 {filters['metric_name']} Performance Heatmaps")

        # Create three columns for side-by-side display
        col1, col2, col3 = st.columns(3)

        # Helper function to create individual heatmap
        def create_heatmap(data_df, index_col, column_col, title_suffix, index_label=None, column_label=None):
            try:
                # Use provided labels or default to title_suffix
                idx_label = (
                    index_label or title_suffix.split(" vs ")[0] if " vs " in title_suffix else index_col.title()
                )
                col_label = (
                    column_label or title_suffix.split(" vs ")[1]
                    if " vs " in title_suffix
                    else column_col.replace("fk_brand_used_id", "Brand").replace("_", " ").title()
                )

                # Aggregate data
                heatmap_data = data_df.groupby([index_col, column_col], as_index=False)[filters["metric"]].sum()

                # Create pivot table
                heatmap_pivot = heatmap_data.pivot_table(
                    index=index_col, columns=column_col, values=filters["metric"], fill_value=0, aggfunc="sum"
                )

                # Sort by total value (descending)
                index_totals = heatmap_pivot.sum(axis=1).sort_values(ascending=False)
                column_totals = heatmap_pivot.sum(axis=0).sort_values(ascending=False)

                heatmap_pivot = heatmap_pivot.reindex(index_totals.index)
                heatmap_pivot = heatmap_pivot.reindex(column_totals.index, axis=1)

                # Create heatmap
                fig = go.Figure(
                    data=go.Heatmap(
                        z=heatmap_pivot.values,
                        x=heatmap_pivot.columns,
                        y=heatmap_pivot.index,
                        colorscale="RdYlBu_r",
                        hoverongaps=False,
                        hovertemplate=(
                            f"<b>{idx_label}:</b> %{{y}}<br>"
                            f"<b>{col_label}:</b> %{{x}}<br>"
                            f"<b>{filters['metric_name']}:</b> %{{z:,.0f}}<extra></extra>"
                        ),
                        colorbar=dict(
                            title=filters["metric_name"],
                            tickmode="linear",
                            tick0=0,
                            dtick=heatmap_pivot.values.max() / 5 if heatmap_pivot.values.max() > 0 else 1,
                        ),
                    )
                )

                # Update layout
                fig.update_layout(
                    title=f"{idx_label} vs {col_label}",
                    xaxis_title=col_label,
                    yaxis_title=idx_label,
                    height=max(400, len(heatmap_pivot.index) * 30),
                    xaxis=dict(side="bottom"),
                    yaxis=dict(autorange="reversed"),
                    margin=dict(l=50, r=50, t=50, b=50),
                )

                # Add text annotations for smaller heatmaps
                if heatmap_pivot.size <= 50:
                    annotations = []
                    for i, country in enumerate(heatmap_pivot.index):
                        for j, col_val in enumerate(heatmap_pivot.columns):
                            value = heatmap_pivot.iloc[i, j]
                            if value > 0:
                                annotations.append(
                                    dict(
                                        x=col_val,
                                        y=country,
                                        text=f"{value:,.0f}",
                                        showarrow=False,
                                        font=dict(
                                            color="white" if value > heatmap_pivot.values.max() * 0.5 else "black",
                                            size=10,
                                        ),
                                    )
                                )
                    fig.update_layout(annotations=annotations)

                return fig, heatmap_data

            except Exception as e:
                st.error(f"❌ Error creating {title_suffix.lower()} heatmap: {str(e)}")
                return None, None

        # Create country-marketplace heatmap
        with col1:
            st.markdown("### 🏪 Country vs Marketplace")
            fig1, data1 = create_heatmap(df, "country", "marketplace", "Country vs Marketplace")
            if fig1:
                st.plotly_chart(fig1, use_container_width=True)

        # Create country-brand heatmap
        with col2:
            st.markdown("### 🏷️ Country vs Brand")
            fig2, data2 = create_heatmap(df, "country", "fk_brand_used_id", "Country vs Brand")
            if fig2:
                st.plotly_chart(fig2, use_container_width=True)

        # Create marketplace-brand heatmap
        with col3:
            st.markdown("### 🛍️ Marketplace vs Brand")
            fig3, data3 = create_heatmap(df, "marketplace", "fk_brand_used_id", "Marketplace vs Brand")
            if fig3:
                st.plotly_chart(fig3, use_container_width=True)

    def display_dashboard_key_metrics(self, df: pd.DataFrame, filters: Dict[str, Any]) -> None:
        """
        Display key metrics at the top of the dashboard

        Args:
            df: Filtered DataFrame
            filters: Filter selections including metric
        """
        try:
            if df.empty:
                st.warning("⚠️ No data available for metrics")
                return

            st.markdown("### 📊 Key Metrics Overview")

            # First row: 3 columns
            row1_col1, row1_col2, row1_col3 = st.columns(3)
            # Second row: 4 columns
            row2_col1, row2_col2, row2_col3 = st.columns(3)

            with row1_col1:
                unique_countries = int(df["country"].nunique())
                st.metric("🌍 Countries", unique_countries)

            with row1_col2:
                unique_marketplaces = int(df["marketplace"].nunique())
                st.metric("🏪 Marketplaces", unique_marketplaces)

            with row1_col3:
                unique_brands = int(df["fk_brand_used_id"].nunique())
                st.metric("🏷️ Brands", unique_brands)

            with row2_col1:
                total_quantity = df["sum_quantity"].sum()
                st.metric(
                    "📦 Total Quantity",
                    f"{total_quantity / 1_000_000:.1f}M"
                    if total_quantity >= 1_000_000
                    else f"{total_quantity / 1_000:.1f}K"
                    if total_quantity >= 1_000
                    else f"{total_quantity:,.0f}",
                )

            with row2_col2:
                total_revenue = df["sum_s_net"].sum()
                st.metric(
                    "💰 Total Revenue",
                    f"{total_revenue / 1_000_000:.1f}M"
                    if total_revenue >= 1_000_000
                    else f"{total_revenue / 1_000:.1f}K"
                    if total_revenue >= 1_000
                    else f"{total_revenue:,.0f}",
                )

            # Get current aggregation level for time-based metrics
            current_aggregation = filters.get("aggregation", "daily")
            aggregation_map = {"daily": "Daily", "weekly": "Weekly", "monthly": "Monthly", "yearly": "Yearly"}
            aggregation_label = aggregation_map.get(current_aggregation, "Daily")

            with row2_col3:
                # Calculate average sales based on aggregation level
                if current_aggregation == "daily":
                    avg_sales = df.groupby("day")["sum_s_net"].sum().mean()
                elif current_aggregation == "weekly":
                    df_temp = df.copy()
                    df_temp["week"] = pd.to_datetime(df_temp["day"]).dt.isocalendar().week
                    avg_sales = df_temp.groupby("week")["sum_s_net"].sum().mean()
                elif current_aggregation == "monthly":
                    df_temp = df.copy()
                    df_temp["month"] = pd.to_datetime(df_temp["day"]).dt.to_period("M")
                    avg_sales = df_temp.groupby("month")["sum_s_net"].sum().mean()
                else:  # yearly
                    df_temp = df.copy()
                    df_temp["year"] = pd.to_datetime(df_temp["day"]).dt.year
                    avg_sales = df_temp.groupby("year")["sum_s_net"].sum().mean()

                st.metric(
                    f"📊 Avg {aggregation_label} Sales",
                    f"{avg_sales / 1_000_000:.1f}M"
                    if avg_sales >= 1_000_000
                    else f"{avg_sales / 1_000:.1f}K"
                    if avg_sales >= 1_000
                    else f"{avg_sales:,.0f}",
                )

            st.markdown("---")

        except Exception as e:
            st.error(f"❌ Error displaying key metrics: {str(e)}")
            with st.expander("🔍 Error Details"):
                st.code(str(e))
                st.write("**Data types:**")
                st.write(df.dtypes)
                if not df.empty:
                    st.write("**Sample data:**")
                    st.write(df.head())

    def create_country_performance_map(self, df: pd.DataFrame, filters: Dict[str, Any]) -> None:
        """
        Create a geographic map with bubble markers showing metric performance by country.

        Features:
        - Bubble size = Total metric value per country
        - Hover details = Marketplace breakdown with percentages
        - Clean, interactive visualization

        Args:
            df: Filtered DataFrame
            filters: Filter selections including metric
        """
        if df.empty:
            st.warning("⚠️ No data to display")
            return

        st.subheader(f"🌍 {filters['metric_name']} by Country & Marketplace")

        try:
            # Aggregate data by country and marketplace
            country_marketplace_data = df.groupby(["country", "marketplace"], as_index=False)[filters["metric"]].sum()

            # Calculate total per country
            country_totals = cast(
                pd.DataFrame, country_marketplace_data.groupby("country")[filters["metric"]].sum().reset_index()
            )
            country_totals.columns = ["country", "total_metric"]

            # Southeast Asia country coordinates
            country_coords = {
                "MY": {"lat": 4.2105, "lon": 101.9758, "name": "Malaysia"},
                "TH": {"lat": 15.8700, "lon": 100.9925, "name": "Thailand"},
                "SG": {"lat": 1.3521, "lon": 103.8198, "name": "Singapore"},
                "PH": {"lat": 12.8797, "lon": 121.7740, "name": "Philippines"},
            }

            # Get countries with data and valid coordinates
            countries_with_data = country_totals["country"].unique()
            valid_countries = [country for country in countries_with_data if country in country_coords]

            if not valid_countries:
                st.warning("⚠️ No countries with valid coordinates found")
                return

            # Calculate bubble sizes (normalize between 20-80 pixels)
            max_total = country_totals["total_metric"].max()
            min_total = country_totals["total_metric"].min()

            def calculate_bubble_size(value):
                if max_total == min_total:
                    return 50
                # Scale between 20 and 80 pixels
                normalized = (value - min_total) / (max_total - min_total)
                return 20 + (normalized * 60)

            # Prepare data for each country bubble
            bubble_data = []
            for country in valid_countries:
                country_data = cast(
                    pd.DataFrame, country_marketplace_data[country_marketplace_data["country"] == country]
                )
                total_value = cast(
                    pd.Series, country_totals[country_totals["country"] == country]["total_metric"]
                ).iloc[0]

                # Calculate marketplace percentages
                marketplace_breakdown = []
                for _, row in country_data.iterrows():
                    percentage = (row[filters["metric"]] / total_value) * 100
                    marketplace_breakdown.append(
                        f"{row['marketplace']}: {row[filters['metric']]:,.0f} ({percentage:.1f}%)"
                    )

                bubble_data.append(
                    {
                        "country": country,
                        "lat": country_coords[country]["lat"],
                        "lon": country_coords[country]["lon"],
                        "name": country_coords[country]["name"],
                        "total": total_value,
                        "size": calculate_bubble_size(total_value),
                        "breakdown": "<br>".join(marketplace_breakdown),
                    }
                )

            # Create the map
            fig = go.Figure()

            # Add bubble markers
            fig.add_trace(
                go.Scattergeo(
                    lon=[item["lon"] for item in bubble_data],
                    lat=[item["lat"] for item in bubble_data],
                    mode="markers",
                    marker=dict(
                        size=[item["size"] for item in bubble_data],
                        color="rgba(31, 119, 180, 0.7)",  # Blue with transparency
                        line=dict(width=2, color="rgba(31, 119, 180, 1)"),
                        sizemode="diameter",
                    ),
                    text=[item["name"] for item in bubble_data],
                    customdata=[[item["breakdown"], item["total"]] for item in bubble_data],
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        + f"Total {filters['metric_name']}: %{{customdata[1]:,.0f}}<br>"
                        + "<b>Marketplace Breakdown:</b><br>"
                        + "%{customdata[0]}<br>"
                        + "<extra></extra>"
                    ),
                    showlegend=False,
                )
            )

            # Add country labels
            fig.add_trace(
                go.Scattergeo(
                    lon=[item["lon"] for item in bubble_data],
                    lat=[item["lat"] - 1.5 for item in bubble_data],  # Offset labels below bubbles
                    mode="text",
                    text=[item["name"] for item in bubble_data],
                    textfont=dict(size=12, color="rgba(33, 37, 41, 0.8)"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            # Update geo layout
            fig.update_geos(
                projection_type="natural earth",
                showland=True,
                landcolor="rgba(248, 249, 250, 1)",
                coastlinecolor="rgba(108, 117, 125, 0.3)",
                coastlinewidth=1,
                showocean=True,
                oceancolor="rgba(173, 216, 230, 0.3)",
                showlakes=True,
                lakecolor="rgba(173, 216, 230, 0.3)",
                showcountries=True,
                countrycolor="rgba(108, 117, 125, 0.2)",
                countrywidth=0.5,
                showframe=False,
                lonaxis=dict(range=[92, 128]),
                lataxis=dict(range=[-8, 22]),
            )

            # Update layout
            fig.update_layout(
                height=900,
                margin=dict(l=20, r=20, t=40, b=20),
                font=dict(family="Inter, Arial, sans-serif", size=12),
                showlegend=False,
            )

            # Add legend explaining bubble sizes
            st.caption(
                f"💡 Bubble size represents total {filters['metric_name']} per country. Hover for marketplace breakdown."
            )

            # Display the map
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        except Exception as e:
            st.error(f"❌ Error creating country performance map: {str(e)}")
            st.info("💡 Make sure your data contains 'country' and 'marketplace' columns")

    def run_dashboard(self) -> None:
        """
        Main function to run the complete dashboard
        """
        # App header
        st.title("📈 E-commerce Forecast Dashboard")
        st.markdown(
            "*Visualize historical data and forecasts across multiple dimensions with flexible time aggregation*"
        )
        st.markdown("---")

        # Load data
        with st.spinner("🔄 Loading forecast data..."):
            self.data = self.load_forecast_data()

        # st.success(f"✅ Loaded {len(self.data):,} records from forecast data")

        # Create filters
        filters = self.create_filters_sidebar(self.data)

        # Apply filters
        filtered_data = self.apply_filters_to_data(self.data, filters)

        # Show filter results
        # st.info(f"📊 Showing {len(filtered_data):,} records after filtering")

        # Main dashboard sections
        if not filtered_data.empty:
            # Time series chart
            self.display_dashboard_key_metrics(filtered_data, filters)
            self.create_time_series_chart(filtered_data, filters)
            self.create_share_charts(filtered_data, filters)
            self.create_country_performance_heatmaps(filtered_data, filters)
            self.create_country_performance_map(filtered_data, filters)
            self.create_sales_table_by_location(filtered_data, filters)
        else:
            st.error("❌ No data matches the selected filters. Please adjust your selection.")

        # Footer
        st.markdown("---")
        st.markdown("*Built with Streamlit & Plotly*")


def main():
    """
    Entry point for the Streamlit application
    """
    dashboard = ForecastDashboard()
    dashboard.run_dashboard()


if __name__ == "__main__":
    main()
