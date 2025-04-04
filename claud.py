import dash
from dash import dcc, html, Input, Output, State, ctx
import pandas as pd
import numpy as np
import plotly.express as px
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import traceback  # Add this for better error tracking

# Load Data
file_path = "C:\\Users\\dell\\git_hub_repo\\-warehouse-and-retail-sales\\Warehouse_and_Retail_Sales.csv"
df = pd.read_csv(file_path)

# Data Preprocessing
df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH']].assign(DAY=1))
valid_min = df['DATE'].min().timestamp()
valid_max = df['DATE'].max().timestamp()

# Handle missing values
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

        # Debug Info (can be removed in production)
        dbc.Row([
            dbc.Col(
                html.Div(id="debug_info", style={'color': 'red', 'fontSize': '12px'}),
                width=12
            )
        ], className="mb-2"),

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
                    options=[{"label": "All", "value": "All"}] + 
                            [{"label": s, "value": s} for s in sorted(df['SUPPLIER'].unique())],
                    value="All",
                    style={'color': 'black'}
                ),
                width=3
            ),
            dbc.Col(
                dcc.Dropdown(
                    id="item_type_filter",
                    options=[{"label": "All", "value": "All"}] + 
                            [{"label": i, "value": i} for i in sorted(df['ITEM TYPE'].unique())],
                    value="All",
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

        # Date Range Selector (replaces slider for more flexibility)
        dbc.Row([
            dbc.Col(
                dcc.DatePickerRange(
                    id='date_range',
                    min_date_allowed=df['DATE'].min().date(),
                    max_date_allowed=df['DATE'].max().date(),
                    start_date=df['DATE'].min().date(),
                    end_date=df['DATE'].max().date(),
                    display_format='YYYY-MM'
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

def create_empty_fig(message="No Data Available"):
    fig = px.scatter(title=message)
    fig.update_layout(
        annotations=[dict(
            text=message,
            showarrow=False,
            x=0.5,
            y=0.5,
            font=dict(size=20)
        )]
    )
    return fig

# Add a callback for debugging info
@app.callback(
    Output("debug_info", "children"),
    [
        Input("year_filter", "value"),
        Input("supplier_filter", "value"),
        Input("item_type_filter", "value"),
        Input("date_range", "start_date"),
        Input("date_range", "end_date")
    ]
)
def update_debug_info(year, supplier, item_type, start_date, end_date):
    return f"Filters: Year={year}, Supplier={supplier}, Item={item_type}, Date Range={start_date} to {end_date}"

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
        Input("date_range", "start_date"),
        Input("date_range", "end_date"),
        Input("download_btn", "n_clicks")
    ],
    [State("download_data", "data")]  # Add state to prevent download on every update
)
def update_dashboard(selected_year, selected_supplier, selected_item_type, 
                    start_date, end_date, download_click, prev_download):
    try:
        # Start with a copy of the full dataset
        filtered_df = df.copy()
        
        # Apply year filter
        if selected_year != "All":
            filtered_df = filtered_df[filtered_df["YEAR"] == int(selected_year)]
        
        # Apply supplier filter - modified to handle single selection
        if selected_supplier != "All":
            filtered_df = filtered_df[filtered_df["SUPPLIER"] == selected_supplier]
        
        # Apply item type filter - modified to handle single selection
        if selected_item_type != "All":
            filtered_df = filtered_df[filtered_df["ITEM TYPE"] == selected_item_type]
        
        # Apply date range filter
        if start_date and end_date:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            filtered_df = filtered_df[
                (filtered_df['DATE'] >= start_date) & 
                (filtered_df['DATE'] <= end_date)
            ]
        
        # Check if data is available after filtering
        if filtered_df.empty:
            empty_message = "No data matches the selected filters"
            return (
                [dbc.Card(
                    dbc.CardBody([html.H5(empty_message, className="card-title")]),
                    className="bg-light text-center m-2 shadow-sm"
                )] * 3,
                create_empty_fig(empty_message),
                create_empty_fig(empty_message),
                create_empty_fig(empty_message),
                create_empty_fig(empty_message),
                None  # No download if no data
            )

        # KPIs
        numeric_cols = ['RETAIL SALES', 'WAREHOUSE SALES', 'RETAIL TRANSFERS']
        kpi_values = filtered_df[numeric_cols].sum()

        kpi_elements = [
            dbc.Card(
                dbc.CardBody([
                    html.H5(f"Total Retail Sales: ${kpi_values.iloc[0]:,.2f}", className="card-title"),
                ]),
                className="bg-light text-center m-2 shadow-sm"
            ),
            dbc.Card(
                dbc.CardBody([
                    html.H5(f"Total Warehouse Sales: ${kpi_values.iloc[1]:,.2f}", className="card-title"),
                ]),
                className="bg-light text-center m-2 shadow-sm"
            ),
            dbc.Card(
                dbc.CardBody([
                    html.H5(f"Total Transfers: ${kpi_values.iloc[2]:,.2f}", className="card-title"),
                ]),
                className="bg-light text-center m-2 shadow-sm"
            )
        ]

        # Charts
        # Improved error handling for chart creation
        try:
            trend_data = filtered_df.groupby("DATE")[["RETAIL SALES", "WAREHOUSE SALES"]].sum().reset_index()
            trend_fig = px.line(
                trend_data,
                x="DATE", 
                y=["RETAIL SALES", "WAREHOUSE SALES"],
                labels={"value": "Sales", "variable": "Category"},
                title="📈 Sales Trend Over Time"
            )
        except Exception:
            trend_fig = create_empty_fig("Could not create Sales Trend chart")

        try:
            if len(filtered_df['SUPPLIER'].unique()) > 1:
                supplier_data = filtered_df.groupby("SUPPLIER")['RETAIL SALES'].sum().nlargest(10).reset_index()
                supplier_fig = px.bar(
                    supplier_data,
                    x="RETAIL SALES", 
                    y="SUPPLIER",
                    orientation="h",
                    title="🏭 Top 10 Suppliers by Retail Sales"
                )
            else:
                supplier_fig = create_empty_fig("Need multiple suppliers for comparison")
        except Exception:
            supplier_fig = create_empty_fig("Could not create Supplier chart")

        try:
            if len(filtered_df['ITEM TYPE'].unique()) > 1:
                item_type_data = filtered_df.groupby("ITEM TYPE")['RETAIL SALES'].sum().reset_index()
                item_type_fig = px.pie(
                    item_type_data,
                    values="RETAIL SALES", 
                    names="ITEM TYPE",
                    title="🏷️ Sales by Item Type"
                )
            else:
                item_type_fig = create_empty_fig("Need multiple item types for pie chart")
        except Exception:
            item_type_fig = create_empty_fig("Could not create Item Type chart")

        try:
            # Only create heatmap if we have data across multiple months and years
            if len(filtered_df['YEAR'].unique()) > 1 and len(filtered_df['MONTH'].unique()) > 1:
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
                )
            else:
                heatmap_fig = create_empty_fig("Need multiple years and months for heatmap")
        except Exception:
            heatmap_fig = create_empty_fig("Could not create Heatmap chart")

        # Handle CSV Download - only trigger when button is clicked
        if ctx.triggered_id == "download_btn" and download_click:
            return (
                kpi_elements,
                trend_fig,
                supplier_fig,
                item_type_fig,
                heatmap_fig,
                dcc.send_data_frame(filtered_df.to_csv, "filtered_data.csv", index=False)
            )

        # Regular update without download
        return kpi_elements, trend_fig, supplier_fig, item_type_fig, heatmap_fig, None

    except Exception as e:
        # Improved error reporting
        error_msg = f"Error in dashboard update: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        
        # Return informative error visualization instead of no_update
        error_fig = create_empty_fig(f"Error: {str(e)}")
        
        return (
            [dbc.Card(
                dbc.CardBody([html.H5(f"Error: {str(e)}", className="card-title")]),
                className="bg-danger text-white text-center m-2 shadow-sm"
            )] * 3,
            error_fig,
            error_fig,
            error_fig,
            error_fig,
            None
        )

if __name__ == "__main__":
    app.run(debug=True)