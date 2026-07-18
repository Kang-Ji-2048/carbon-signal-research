"""
Interactive Climate Finance Dashboard built with Dash and Plotly.

Visualises global climate investment flows by sector, region, and
instrument type from the unified ETL dataset.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# --- Colour palette ---
SECTOR_COLORS = {
    "Renewable Energy": "#2ecc71",
    "Energy Efficiency": "#3498db",
    "Sustainable Transport": "#9b59b6",
    "Climate Adaptation": "#e67e22",
    "Forestry & Land Use": "#27ae60",
    "Water & Waste Management": "#1abc9c",
    "Cross-cutting": "#95a5a6",
}

REGION_COLORS = {
    "Sub-Saharan Africa": "#e74c3c",
    "South Asia": "#f39c12",
    "East Asia & Pacific": "#2ecc71",
    "Latin America & Caribbean": "#3498db",
    "Europe & Central Asia": "#9b59b6",
    "Middle East & North Africa": "#1abc9c",
}


def load_data() -> dict[str, pd.DataFrame]:
    """Load pre-aggregated views from processed CSVs."""
    views = {}
    for name in [
        "by_year_sector",
        "by_year_region",
        "by_year_instrument",
        "by_year_region_sector",
        "by_source",
    ]:
        path = PROCESSED_DIR / f"{name}.csv"
        views[name] = pd.read_csv(path)

    views["detail"] = pd.read_csv(PROCESSED_DIR / "climate_finance_clean.csv")
    return views


def create_app(views: dict[str, pd.DataFrame]) -> Dash:
    """Build and return the Dash application."""

    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.FLATLY],
        title="Climate Finance Dashboard",
    )

    detail = views["detail"]
    # Convert to native Python ints to avoid numpy.int64 JSON serialisation errors
    all_years = sorted(int(y) for y in detail["year"].unique())
    all_regions = sorted(str(r) for r in detail["region"].unique())
    all_sectors = sorted(str(s) for s in detail["sector"].unique())

    # ---- Layout ----
    app.layout = dbc.Container(
        fluid=True,
        className="py-3",
        children=[
            # Header
            dbc.Row(
                dbc.Col(
                    html.Div([
                        html.H1(
                            "Global Climate Finance Dashboard",
                            className="text-primary mb-1",
                        ),
                        html.P(
                            "Unified view of OECD, IRENA & World Bank climate investment data",
                            className="text-muted",
                        ),
                    ]),
                    width=12,
                ),
                className="mb-3",
            ),

            # Filters
            dbc.Row(
                [
                    dbc.Col([
                        html.Label("Year Range", className="fw-bold"),
                        dcc.RangeSlider(
                            id="year-slider",
                            min=int(min(all_years)),
                            max=int(max(all_years)),
                            value=[int(min(all_years)), int(max(all_years))],
                            marks={int(y): str(y) for y in all_years[::2]},
                            step=1,
                        ),
                    ], md=4),
                    dbc.Col([
                        html.Label("Region", className="fw-bold"),
                        dcc.Dropdown(
                            id="region-filter",
                            options=[{"label": r, "value": r} for r in all_regions],
                            value=[],
                            multi=True,
                            placeholder="All regions",
                        ),
                    ], md=4),
                    dbc.Col([
                        html.Label("Sector", className="fw-bold"),
                        dcc.Dropdown(
                            id="sector-filter",
                            options=[{"label": s, "value": s} for s in all_sectors],
                            value=[],
                            multi=True,
                            placeholder="All sectors",
                        ),
                    ], md=4),
                ],
                className="mb-4",
            ),

            # KPI cards
            dbc.Row(id="kpi-cards", className="mb-4"),

            # Row 1: trend + sector breakdown
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="trend-chart"), md=7),
                    dbc.Col(dcc.Graph(id="sector-pie"), md=5),
                ],
                className="mb-4",
            ),

            # Row 2: region bar + instrument sunburst
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="region-bar"), md=6),
                    dbc.Col(dcc.Graph(id="instrument-chart"), md=6),
                ],
                className="mb-4",
            ),

            # Row 3: heatmap + source comparison
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="heatmap"), md=7),
                    dbc.Col(dcc.Graph(id="source-chart"), md=5),
                ],
                className="mb-4",
            ),

            # Footer
            dbc.Row(
                dbc.Col(
                    html.P(
                        "Data sources: OECD DAC, IRENA, World Bank Climate Finance | "
                        "Built with Python, Pandas, Plotly & Dash",
                        className="text-muted text-center small mt-3",
                    )
                )
            ),
        ],
    )

    # ---- Callbacks ----
    @callback(
        Output("kpi-cards", "children"),
        Output("trend-chart", "figure"),
        Output("sector-pie", "figure"),
        Output("region-bar", "figure"),
        Output("instrument-chart", "figure"),
        Output("heatmap", "figure"),
        Output("source-chart", "figure"),
        Input("year-slider", "value"),
        Input("region-filter", "value"),
        Input("sector-filter", "value"),
    )
    def update_all(year_range, regions, sectors):
        df = detail.copy()

        # Apply filters
        df = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]
        if regions:
            df = df[df["region"].isin(regions)]
        if sectors:
            df = df[df["sector"].isin(sectors)]

        # KPIs
        total = df["amount_usd_mn"].sum()
        n_countries = df["country"].nunique()
        n_flows = len(df)
        avg_flow = df["amount_usd_mn"].mean() if len(df) > 0 else 0

        kpis = [
            _kpi_card("Total Investment", f"${total / 1000:,.1f}B", "USD Billions"),
            _kpi_card("Countries", f"{n_countries}", "Unique recipients"),
            _kpi_card("Finance Flows", f"{n_flows:,}", "Individual transactions"),
            _kpi_card("Avg Flow Size", f"${avg_flow:,.1f}M", "USD Millions"),
        ]

        # 1. Trend line by sector
        trend_data = (
            df.groupby(["year", "sector"], as_index=False)["amount_usd_mn"].sum()
        )
        fig_trend = px.area(
            trend_data,
            x="year",
            y="amount_usd_mn",
            color="sector",
            color_discrete_map=SECTOR_COLORS,
            labels={"amount_usd_mn": "USD Millions", "year": "Year", "sector": "Sector"},
            title="Climate Finance Trends by Sector",
        )
        fig_trend.update_layout(
            legend=dict(orientation="h", y=-0.2),
            margin=dict(t=40, b=80),
        )

        # 2. Sector pie
        sector_totals = df.groupby("sector", as_index=False)["amount_usd_mn"].sum()
        fig_pie = px.pie(
            sector_totals,
            values="amount_usd_mn",
            names="sector",
            color="sector",
            color_discrete_map=SECTOR_COLORS,
            title="Sector Distribution",
            hole=0.4,
        )
        fig_pie.update_layout(margin=dict(t=40, b=40))

        # 3. Region grouped bar
        region_data = (
            df.groupby(["year", "region"], as_index=False)["amount_usd_mn"].sum()
        )
        fig_region = px.bar(
            region_data,
            x="year",
            y="amount_usd_mn",
            color="region",
            color_discrete_map=REGION_COLORS,
            barmode="group",
            labels={"amount_usd_mn": "USD Millions", "year": "Year", "region": "Region"},
            title="Investment by Region Over Time",
        )
        fig_region.update_layout(
            legend=dict(orientation="h", y=-0.25),
            margin=dict(t=40, b=80),
        )

        # 4. Instrument type treemap
        inst_data = (
            df.groupby(["instrument_type", "sector"], as_index=False)["amount_usd_mn"].sum()
        )
        fig_inst = px.treemap(
            inst_data,
            path=["instrument_type", "sector"],
            values="amount_usd_mn",
            color="instrument_type",
            title="Finance by Instrument Type & Sector",
        )
        fig_inst.update_layout(margin=dict(t=40, b=20))

        # 5. Heatmap: region x sector
        heatmap_data = (
            df.groupby(["region", "sector"], as_index=False)["amount_usd_mn"]
            .sum()
            .pivot(index="region", columns="sector", values="amount_usd_mn")
            .fillna(0)
        )
        fig_heatmap = px.imshow(
            heatmap_data,
            labels=dict(x="Sector", y="Region", color="USD Millions"),
            title="Investment Heatmap: Region x Sector",
            color_continuous_scale="YlGn",
            aspect="auto",
        )
        fig_heatmap.update_layout(margin=dict(t=40, b=40))

        # 6. Source comparison
        source_data = (
            df.groupby(["year", "source"], as_index=False)["amount_usd_mn"].sum()
        )
        fig_source = px.line(
            source_data,
            x="year",
            y="amount_usd_mn",
            color="source",
            markers=True,
            labels={"amount_usd_mn": "USD Millions", "year": "Year", "source": "Source"},
            title="Data Source Comparison",
        )
        fig_source.update_layout(margin=dict(t=40, b=40))

        return kpis, fig_trend, fig_pie, fig_region, fig_inst, fig_heatmap, fig_source

    return app


def _kpi_card(title: str, value: str, subtitle: str) -> dbc.Col:
    """Create a styled KPI card."""
    return dbc.Col(
        dbc.Card(
            dbc.CardBody([
                html.H6(title, className="text-muted mb-1"),
                html.H3(value, className="text-primary mb-1"),
                html.Small(subtitle, className="text-muted"),
            ]),
            className="shadow-sm",
        ),
        md=3,
    )
