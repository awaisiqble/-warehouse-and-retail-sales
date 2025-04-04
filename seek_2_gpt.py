# -*- coding: utf-8 -*-
import dash
from dash import dcc, html, Input, Output, State, dash_table
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import traceback
from datetime import date, timedelta
from functools import lru_cache
from sklearn.linear_model import LinearRegression

# --- Configuration ---
FILE_PATH = "C:\\Users\\dell\\git_hub_repo\\-warehouse-and-retail-sales\\Warehouse_and_Retail_Sales.csv"

# --- Load Data with Flexible Validation ---
try:
    df = pd.read_csv(FILE_PATH)
    print("Data loaded successfully.")
    
    # Validation logic
    required_columns = ['RETAIL SALES', 'SUPPLIER']
    date_columns = []
    
    if 'DATE' in df.columns:
        date_columns = ['DATE']
    else:
        if {'YEAR', 'MONTH'}.issubset(df.columns):
            date_columns = ['YEAR', 'MONTH']
        else:
            raise ValueError("Missing date columns: Either 'DATE' or both 'YEAR' and 'MONTH' are required")
    
    critical_columns = required_columns + date_columns
    missing = [col for col in critical_columns if col not in df.columns]
    
    if missing:
        raise ValueError(f"Missing critical columns: {missing}")

except FileNotFoundError:
    print(f"Error: File not found at {FILE_PATH}")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --- Data Preprocessing ---
print("Preprocessing data...")

# Create DATE column if needed
if 'DATE' not in df.columns:
    df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH']].assign(DAY=1))
else:
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')

# Extract date components
df['YEAR'] = df['DATE'].dt.year
df['MONTH'] = df['DATE'].dt.month

# Clean text columns
text_cols = ['SUPPLIER', 'ITEM TYPE']
for col in text_cols:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.upper()
            .replace({'NAN': 'UNKNOWN', '': 'UNKNOWN'})
        )

# Handle numeric columns
for col in ['RETAIL SALES', 'WAREHOUSE SALES', 'RETAIL TRANSFERS']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Final cleanup
df = df.dropna(subset=['DATE', 'RETAIL SALES'])
min_date = df['DATE'].min().date()
max_date = df['DATE'].max().date()

# --- Dash App Setup ---
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server

# Minimal security headers
@server.after_request
def set_secure_headers(response):
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response

# Optional custom index string for SEO/meta
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        <div id="app-entry">
            {%app_entry%}
        </div>
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# --- Helper Functions ---
def create_empty_fig(message="No data available"):
    fig = go.Figure()
    fig.add_annotation(
        text=message, x=0.5, y=0.5,
        showarrow=False, font=dict(size=16, color="#6c757d")
    )
    fig.update_layout(
        xaxis_visible=False, yaxis_visible=False,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=30, b=30, l=30, r=30)
    )
    return fig

@lru_cache(maxsize=32)
def get_filtered_data(selected_year, selected_supplier, selected_item_type, start_date, end_date):
    filtered_df = df.copy()
    
    # Filter by year
    if selected_year and selected_year != "All":
        filtered_df = filtered_df[filtered_df["YEAR"] == int(selected_year)]
    
    # Filter by supplier
    if selected_supplier and selected_supplier != "All":
        filtered_df = filtered_df[filtered_df["SUPPLIER"] == selected_supplier]
    
    # Filter by item type
    if "ITEM TYPE" in filtered_df.columns and selected_item_type and selected_item_type != "All":
        filtered_df = filtered_df[filtered_df["ITEM TYPE"] == selected_item_type]
    
    # Filter by date range
    if start_date and end_date:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        filtered_df = filtered_df[
            (filtered_df['DATE'] >= start_dt) & 
            (filtered_df['DATE'] <= end_dt)
        ]
    
    return filtered_df

def forecast_sales(filtered_df, periods=6):
    """
    Create a simple forecast using linear regression.
    periods: number of future periods (months) to forecast.
    """
    if filtered_df.empty:
        return create_empty_fig("No forecast due to insufficient data.")

    # Resample monthly and sum retail sales
    ts_data = filtered_df.resample('MS', on='DATE')['RETAIL SALES'].sum().reset_index()
    ts_data.sort_values('DATE', inplace=True)
    
    # Prepare training data
    ts_data['TIME'] = np.arange(len(ts_data))
    X = ts_data[['TIME']]
    y = ts_data['RETAIL SALES']
    
    # Fit model
    model = LinearRegression()
    model.fit(X, y)
    
    # Forecast future periods
    last_time = ts_data['TIME'].iloc[-1]
    future_times = np.arange(last_time + 1, last_time + periods + 1).reshape(-1, 1)
    forecast_values = model.predict(future_times)
    
    # Create DataFrame for forecast
    last_date = ts_data['DATE'].iloc[-1]
    future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, periods + 1)]
    forecast_df = pd.DataFrame({
        'DATE': future_dates,
        'FORECAST SALES': forecast_values
    })
    
    # Plot actuals and forecast
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts_data['DATE'], y=ts_data['RETAIL SALES'],
        mode='lines+markers', name='Actual Sales'
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df['DATE'], y=forecast_df['FORECAST SALES'],
        mode='lines+markers', name='Forecast Sales',
        line=dict(dash='dash')
    ))
    fig.update_layout(
        title="Retail Sales Forecast",
        xaxis_title="Date",
        yaxis_title="Sales ($)",
        template="plotly_white"
    )
    return fig

# --- Dashboard Layout ---
app.layout = dbc.Container(
    fluid=True,
    style={'backgroundColor': '#f8f9fa', 'padding': '15px'},
    children=[
        dbc.Row(
            dbc.Col(
                html.H1("Sales Dashboard", className="text-primary text-center mb-4",
                        style={'fontSize': '2rem', 'fontWeight': 'bold'}),
                width=12
            )
        ),
        # --- Filters with Tooltips ---
        dbc.Row([
            dbc.Col(
                dbc.Tooltip("Filter by year", target="year_filter"),
                width=0
            ),
            dbc.Col(
                dcc.Dropdown(
                    id="year_filter",
                    options=[{"label": "All Years", "value": "All"}] + [
                        {"label": y, "value": y} for y in sorted(df['YEAR'].unique())
                    ],
                    value="All",
                    clearable=False,
                    placeholder="Select Year"
                ),
                md=3
            ),
            dbc.Col(
                dbc.Tooltip("Filter by supplier", target="supplier_filter"),
                width=0
            ),
            dbc.Col(
                dcc.Dropdown(
                    id="supplier_filter",
                    options=[{"label": "All Suppliers", "value": "All"}] + [
                        {"label": s, "value": s} for s in sorted(df['SUPPLIER'].unique())
                    ],
                    value="All",
                    clearable=False,
                    placeholder="Select Supplier"
                ),
                md=3
            ),
            dbc.Col(
                dbc.Tooltip("Filter by item type", target="item_type_filter"),
                width=0
            ),
            dbc.Col(
                # Only show if ITEM TYPE exists
                dcc.Dropdown(
                    id="item_type_filter",
                    options=[{"label": "All Item Types", "value": "All"}] + [
                        {"label": t, "value": t} for t in sorted(df['ITEM TYPE'].unique())
                    ] if "ITEM TYPE" in df.columns else [],
                    value="All" if "ITEM TYPE" in df.columns else None,
                    clearable=False,
                    placeholder="Select Item Type"
                ) if "ITEM TYPE" in df.columns else html.Div(""),
                md=3
            ),
            dbc.Col(
                dcc.DatePickerRange(
                    id='date_range',
                    min_date_allowed=min_date,
                    max_date_allowed=max_date,
                    start_date=min_date,
                    end_date=max_date,
                    display_format='YYYY-MM-DD'
                ),
                md=3
            )
        ], className="mb-4 g-2"),
        
        # --- Charts ---
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("Retail Sales Trend"),
                    dbc.CardBody(
                        dcc.Graph(id="sales_trend_chart", config={"displayModeBar": True})
                    )
                ], className="shadow-sm"),
                md=6
            ),
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("Monthly Sales Heatmap"),
                    dbc.CardBody(
                        dcc.Graph(id="heatmap_chart", config={"displayModeBar": True})
                    )
                ], className="shadow-sm"),
                md=6
            ),
        ], className="mb-4 g-3"),
        
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("Top Suppliers"),
                    dbc.CardBody(
                        dcc.Graph(id="top_suppliers_chart", config={"displayModeBar": True})
                    )
                ], className="shadow-sm"),
                md=6
            ),
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("Sales by Item Type"),
                    dbc.CardBody(
                        dcc.Graph(id="item_type_chart", config={"displayModeBar": True})
                    )
                ], className="shadow-sm"),
                md=6
            ),
        ], className="mb-4 g-3"),
        
        # --- Forecast Sales (with Loading Indicator) ---
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("Forecast Sales"),
                    dbc.CardBody(
                        dcc.Loading(
                            id="loading_forecast",
                            type="default",
                            children=dcc.Graph(id="forecast_chart", config={"displayModeBar": True})
                        )
                    )
                ], className="shadow-sm"),
                md=12
            )
        ], className="mb-4 g-3"),
        
        # --- Filtered Data Table & Download Section ---
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("Filtered Data"),
                    dbc.CardBody([
                        dash_table.DataTable(
                            id='data_table',
                            columns=[{"name": i, "id": i} for i in df.columns],
                            data=[],  
                            page_size=10,
                            style_table={'overflowX': 'auto'},
                            style_header={'backgroundColor': '#f1f1f1', 'fontWeight': 'bold'},
                            style_cell={'textAlign': 'left', 'minWidth': '100px', 'padding': '5px'}
                        ),
                        dbc.Button("Download CSV", id="download_button", color="primary", className="mt-2"),
                        dcc.Download(id="download_data")
                    ])
                ], className="shadow-sm"),
                width=12
            )
        ], className="mb-4 g-3")
    ]
)

# --- Callbacks for Charts ---
@app.callback(
    [
        Output("sales_trend_chart", "figure"),
        Output("heatmap_chart", "figure"),
        Output("top_suppliers_chart", "figure"),
        Output("item_type_chart", "figure"),
        Output("forecast_chart", "figure")
    ],
    [
        Input("year_filter", "value"),
        Input("supplier_filter", "value"),
        Input("item_type_filter", "value"),
        Input("date_range", "start_date"),
        Input("date_range", "end_date")
    ]
)
def update_charts(selected_year, selected_supplier, selected_item_type, start_date, end_date):
    try:
        filtered_df = get_filtered_data(selected_year, selected_supplier, selected_item_type, start_date, end_date)
        
        # Retail Sales Trend (Line Chart)
        if filtered_df.empty:
            trend_fig = create_empty_fig("No data available for the selected filters.")
        else:
            trend_data = (
                filtered_df
                .resample('MS', on='DATE')['RETAIL SALES']
                .sum()
                .reset_index()
            )
            trend_fig = px.line(
                trend_data,
                x='DATE', y='RETAIL SALES',
                title='Retail Sales Trend',
                labels={'RETAIL SALES': 'Sales ($)', 'DATE': 'Date'}
            )
        
        # Monthly Sales Heatmap
        if filtered_df.empty:
            heatmap_fig = create_empty_fig("No data available for the selected filters.")
        else:
            month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                           'Jul','Aug','Sep','Oct','Nov','Dec']
            heatmap_data = filtered_df.pivot_table(
                index='MONTH', columns='YEAR',
                values='RETAIL SALES', aggfunc='sum'
            ).fillna(0)
            heatmap_data = heatmap_data.reindex(range(1,13), fill_value=0)
            heatmap_data.index = [month_names[m-1] for m in heatmap_data.index]
            heatmap_fig = px.imshow(
                heatmap_data,
                labels=dict(x="Year", y="Month", color="Sales ($)"),
                color_continuous_scale=px.colors.sequential.Blues,
                title='Monthly Sales Distribution'
            )
        
        # Top Suppliers (Bar Chart)
        if filtered_df.empty:
            suppliers_fig = create_empty_fig("No data available for the selected filters.")
        else:
            top_suppliers = (
                filtered_df
                .groupby('SUPPLIER', as_index=False)['RETAIL SALES']
                .sum()
                .sort_values(by='RETAIL SALES', ascending=False)
                .head(5)
            )
            suppliers_fig = px.bar(
                top_suppliers,
                x='SUPPLIER', y='RETAIL SALES',
                title='Top Suppliers (by Retail Sales)',
                labels={'SUPPLIER': 'Supplier', 'RETAIL SALES': 'Sales ($)'}
            )
        
        # Sales by Item Type (Pie Chart)
        if "ITEM TYPE" in filtered_df.columns:
            if selected_item_type == "All":
                item_type_data = (
                    filtered_df
                    .groupby('ITEM TYPE', as_index=False)['RETAIL SALES']
                    .sum()
                )
                item_type_fig = px.pie(
                    item_type_data,
                    names='ITEM TYPE', values='RETAIL SALES',
                    title='Sales by Item Type'
                )
            else:
                item_type_fig = create_empty_fig("Item breakdown not available when a single item type is selected.")
        else:
            item_type_fig = create_empty_fig("ITEM TYPE column not found.")
        
        # Forecast Sales (Advanced Analytics)
        forecast_fig = forecast_sales(filtered_df, periods=6)
        
        return trend_fig, heatmap_fig, suppliers_fig, item_type_fig, forecast_fig

    except Exception as e:
        print(f"Chart Error: {traceback.format_exc()}")
        empty = create_empty_fig("Error generating chart.")
        return empty, empty, empty, empty, empty

# --- Callback for Data Table ---
@app.callback(
    Output("data_table", "data"),
    [
        Input("year_filter", "value"),
        Input("supplier_filter", "value"),
        Input("item_type_filter", "value"),
        Input("date_range", "start_date"),
        Input("date_range", "end_date")
    ]
)
def update_table(selected_year, selected_supplier, selected_item_type, start_date, end_date):
    filtered_df = get_filtered_data(selected_year, selected_supplier, selected_item_type, start_date, end_date)
    return filtered_df.to_dict("records")

# --- Callback for CSV Download ---
@app.callback(
    Output("download_data", "data"),
    Input("download_button", "n_clicks"),
    [
        State("year_filter", "value"),
        State("supplier_filter", "value"),
        State("item_type_filter", "value"),
        State("date_range", "start_date"),
        State("date_range", "end_date")
    ],
    prevent_initial_call=True,
)
def download_csv(n_clicks, selected_year, selected_supplier, selected_item_type, start_date, end_date):
    filtered_df = get_filtered_data(selected_year, selected_supplier, selected_item_type, start_date, end_date)
    return dcc.send_data_frame(filtered_df.to_csv, "filtered_sales_data.csv", index=False)

# --- Run App ---
if __name__ == "__main__":
    app.run(debug=True)
