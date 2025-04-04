import dash
from dash import dcc, html, Input, Output, ctx, no_update
import pandas as pd
import numpy as np
import plotly.express as px
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

# Load Data
file_path = "C:\\Users\\dell\\git_hub_repo\\-warehouse-and-retail-sales\\Warehouse_and_Retail_Sales.csv"
df = pd.read_csv(file_path)

# Data Preprocessing
df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH']].assign(DAY=1))
valid_min = df['DATE'].min().timestamp()
valid_max = df['DATE'].max().timestamp()

# Handle missing values for dropdowns
df['SUPPLIER'] = df['SUPPLIER'].fillna('Unknown Supplier')
df['ITEM TYPE'] = df['ITEM TYPE'].fillna('Unknown Item Type')

# Initialize the Dash App
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the Layout
app.layout = dbc.Container(
    fluid=True,
    children=[
        dbc.Row([
            dbc.Col(
                html.H1(
                    "📊 Warehouse & Retail Sales Dashboard",
                    style={'textAlign': 'center', 'color': '#007bff', 'fontSize': '36px'}
                ),
                width=12
            )
        ], className="mb-4"),

        # Filters
        dbc.Row([
            dbc.Col(
                dcc.Dropdown(
                    id="year_filter",
                    options=[{"label": "All", "value": "All"}] + 
                            [{"label": str(y), "value": y} for y in sorted(df['YEAR'].unique())],
                    value="All",
                    clearable=False,
                    style={'color': 'black'}
                ),
                width=3
            ),
            dbc.Col(
                dcc.Dropdown(
                    id="supplier_filter",
                    options=[{"label": s, "value": s} for s in sorted(df['SUPPLIER'].unique())],
                    multi=True,
                    placeholder="Select Suppliers...",
                    style={'color': 'black'}
                ),
                width=3
            ),
            dbc.Col(
                dcc.Dropdown(
                    id="item_type_filter",
                    options=[{"label": i, "value": i} for i in sorted(df['ITEM TYPE'].unique())],
                    multi=True,
                    placeholder="Select Item Types...",
                    style={'color': 'black'}
                ),
                width=3
            ),
            dbc.Col(
                html.Button(
                    "Download Data",
                    id="download_btn",
                    style={'backgroundColor': '#007bff', 'color': 'white', 'border': 'none', 'padding': '8px 12px'}
                ),
                width=3
            ),
        ], className="g-2 mb-4"),

        # Time Slider
        dbc.Row([
            dbc.Col(
                dcc.Slider(
                    id='month_slider',
                    min=valid_min,
                    max=valid_max,
                    step=2592000,  # Approx 1 month
                    value=valid_min,
                    marks={int(ts): pd.to_datetime(ts, unit='s').strftime('%Y-%m') for ts in np.linspace(valid_min, valid_max, num=12)},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
                width=12
            )
        ], className="mb-4"),

        # KPIs
        dbc.Row([
            dbc.Col(
                html.Div(id="kpi_display", style={'display': 'flex', 'justifyContent': 'space-around'}),
                width=12
            )
        ], className="mb-4"),

        # Charts
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(dcc.Graph(id="sales_trend_chart")),
                    className="shadow-sm p-3 bg-white rounded"
                ),
                width=6
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(dcc.Graph(id="supplier_comparison_chart")),
                    className="shadow-sm p-3 bg-white rounded"
                ),
                width=6
            ),
        ], className="g-2 mb-4"),

        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(dcc.Graph(id="item_type_sales_chart")),
                    className="shadow-sm p-3 bg-white rounded"
                ),
                width=6
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(dcc.Graph(id="heatmap_chart")),
                    className="shadow-sm p-3 bg-white rounded"
                ),
                width=6
            ),
        ], className="g-2 mb-4"),

        # Download Component
        dcc.Download(id="download_data")
    ],
    style={'backgroundColor': '#f8f9fa', 'color': 'black', 'padding': '20px'}
)

def create_empty_fig():
    fig = px.scatter(title="No Data Available")
    fig.update_layout(
        annotations=[dict(
            text="No Data Available",
            showarrow=False,
            x=0.5, y=0.5,
            font=dict(size=20)
        )]
    )
    return fig

@app.callback(
    [
        Output("kpi_display", "children"),
        Output("sales_trend_chart", "figure"),
        Output("supplier_comparison_chart", "figure"),
        Output("item_type_sales_chart", "figure"),
        Output("heatmap_chart", "figure"),
        Output("download_data", "data")
    ],
    [
        Input("year_filter", "value"),
        Input("supplier_filter", "value"),
        Input("item_type_filter", "value"),
        Input("month_slider", "value"),
        Input("download_btn", "n_clicks")
    ]
)
def update_dashboard(selected_year, selected_suppliers, selected_item_types, selected_date, download_click):
    try:
        filtered_df = df.copy()
        empty_fig = create_empty_fig()

        # Apply filters
        if selected_year != "All":
            filtered_df = filtered_df[filtered_df["YEAR"] == selected_year]
        
        if selected_suppliers:
            filtered_df = filtered_df[filtered_df["SUPPLIER"].isin(selected_suppliers)]
        
        if selected_item_types:
            filtered_df = filtered_df[filtered_df["ITEM TYPE"].isin(selected_item_types)]

        # Date filtering
        selected_date = pd.to_datetime(selected_date, unit='s').replace(day=1)
        filtered_df = filtered_df[
            (filtered_df['DATE'] >= selected_date) & 
            (filtered_df['DATE'] < (selected_date + pd.offsets.MonthEnd(1)))
        ]

        if filtered_df.empty:
            return (
                [dbc.Card(
                    dbc.CardBody([html.H5("No Data Available", className="card-title")]),
                    className="bg-light text-center m-2 shadow-sm"
                )] * 3,
                empty_fig,
                empty_fig,
                empty_fig,
                empty_fig,
                None
            )

        # KPIs
        numeric_cols = ['RETAIL SALES', 'WAREHOUSE SALES', 'RETAIL TRANSFERS']
        kpi_values = filtered_df[numeric_cols].sum()

        kpi_elements = [
            dbc.Card(
                dbc.CardBody([
                    html.H5(f"Total Retail Sales: ${kpi_values[0]:,.2f}", className="card-title"),
                ]),
                className="bg-light text-center m-2 shadow-sm"
            ),
            dbc.Card(
                dbc.CardBody([
                    html.H5(f"Total Warehouse Sales: ${kpi_values[1]:,.2f}", className="card-title"),
                ]),
                className="bg-light text-center m-2 shadow-sm"
            ),
            dbc.Card(
                dbc.CardBody([
                    html.H5(f"Total Transfers: ${kpi_values[2]:,.2f}", className="card-title"),
                ]),
                className="bg-light text-center m-2 shadow-sm"
            )
        ]

        # Sales Trend Chart
        trend_data = filtered_df.groupby("DATE")[numeric_cols[:2]].sum().reset_index()
        trend_fig = px.line(
            trend_data,
            x="DATE", 
            y=["RETAIL SALES", "WAREHOUSE SALES"],
            labels={"value": "Sales", "variable": "Category"},
            title="📈 Sales Trend Over Time"
        )

        # Supplier Comparison Chart
        supplier_data = filtered_df.groupby("SUPPLIER")['RETAIL SALES'].sum().nlargest(10).reset_index()
        supplier_fig = px.bar(
            supplier_data,
            x="RETAIL SALES", 
            y="SUPPLIER",
            orientation="h",
            title="🏭 Top 10 Suppliers by Retail Sales"
        )

        # Item Type Sales Chart
        item_type_data = filtered_df.groupby("ITEM TYPE")['RETAIL SALES'].sum().reset_index()
        item_type_fig = px.pie(
            item_type_data,
            values="RETAIL SALES", 
            names="ITEM TYPE",
            title="🏷️ Sales by Item Type"
        )

        # Sales Heatmap
        heatmap_data = filtered_df.pivot_table(
            index="MONTH", 
            columns="YEAR", 
            values="RETAIL SALES", 
            aggfunc="sum"
        )
        heatmap_fig = px.imshow(
            heatmap_data,
            labels=dict(x="Year", y="Month", color="Sales"),
            title="🌡️ Sales Heatmap"
        ) if not heatmap_data.empty else empty_fig

        # Handle CSV Download
        if ctx.triggered_id == "download_btn":
            return (
                kpi_elements,
                trend_fig,
                supplier_fig,
                item_type_fig,
                heatmap_fig,
                dcc.send_data_frame(filtered_df.to_csv, "filtered_data.csv", index=False)
            )

        return kpi_elements, trend_fig, supplier_fig, item_type_fig, heatmap_fig, None

    except Exception as e:
        print(f"Error: {str(e)}")
        return no_update, no_update, no_update, no_update, no_update, no_update

if __name__ == "__main__":
    app.run(debug=True)