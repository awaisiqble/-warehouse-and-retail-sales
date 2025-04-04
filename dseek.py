# -*- coding: utf-8 -*-
import dash
from dash import dcc, html, Input, Output, State, ctx, dash_table
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import traceback
from datetime import date

# --- Configuration ---
FILE_PATH = "C:\\Users\\dell\\git_hub_repo\\-warehouse-and-retail-sales\\Warehouse_and_Retail_Sales.csv"

# --- Load Data ---
try:
    df = pd.read_csv(FILE_PATH)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: File not found at {FILE_PATH}")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --- Data Preprocessing ---
print("Preprocessing data...")
# Handle DATE/YEAR/MONTH columns
if 'YEAR' in df.columns and 'MONTH' in df.columns:
    df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH']].assign(DAY=1))
else:
    if 'DATE' in df.columns:
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    else:
        print("Error: DATE column cannot be created.")
        exit()

# Extract YEAR/MONTH from DATE if missing
if 'DATE' in df.columns:
    if 'YEAR' not in df.columns:
        df['YEAR'] = df['DATE'].dt.year
    if 'MONTH' not in df.columns:
        df['MONTH'] = df['DATE'].dt.month

# Clean text columns
for col in ['SUPPLIER', 'ITEM TYPE']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.upper()
        df[col] = df[col].replace('NAN', 'UNKNOWN')
    else:
        df[col] = 'UNKNOWN'

# Handle numeric columns
numeric_cols = ['RETAIL SALES', 'WAREHOUSE SALES', 'RETAIL TRANSFERS']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    else:
        df[col] = 0

# Final cleanups
df = df.dropna(subset=['DATE', 'RETAIL SALES'])
min_date = df['DATE'].min().date()
max_date = df['DATE'].max().date()

# --- Initialize Dash App ---
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server

# --- Helper Functions ---
def create_empty_fig(message="No data"):
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False, font_size=16)
    fig.update_layout(xaxis_visible=False, yaxis_visible=False)
    return fig

# --- App Layout ---
app.layout = dbc.Container(
    fluid=True,
    children=[
        dbc.Row(dbc.Col(html.H1("📊 Sales Dashboard", className="text-primary mb-4"))),
        
        # Filters
        dbc.Row([
            dbc.Col(dcc.Dropdown(
                id="year_filter", options=[{'label': y, 'value': y} for y in df['YEAR'].unique()],
                placeholder="Select Year", multi=False
            ), md=3),
            dbc.Col(dcc.Dropdown(
                id="supplier_filter", options=[{'label': s, 'value': s} for s in df['SUPPLIER'].unique()],
                placeholder="Select Supplier", multi=False
            ), md=3),
            dbc.Col(dcc.DatePickerRange(
                id='date_range',
                min_date_allowed=min_date,
                max_date_allowed=max_date,
                start_date=min_date,
                end_date=max_date
            ), md=6)
        ], className="mb-4"),

        # Charts
        dbc.Row([
            dbc.Col(dcc.Graph(id="sales_trend"), md=6),
            dbc.Col(dcc.Graph(id="heatmap"), md=6)
        ], className="mb-4"),

        dbc.Row([
            dbc.Col(dcc.Graph(id="supplier_chart"), md=6),
            dbc.Col(dcc.Graph(id="item_chart"), md=6)
        ])
    ]
)

# --- Callbacks ---
@app.callback(
    [Output("sales_trend", "figure"),
     Output("heatmap", "figure"),
     Output("supplier_chart", "figure"),
     Output("item_chart", "figure")],
    [Input("year_filter", "value"),
     Input("supplier_filter", "value"),
     Input("date_range", "start_date"),
     Input("date_range", "end_date")]
)
def update_charts(year, supplier, start_date, end_date):
    filtered_df = df.copy()
    
    # Apply filters
    if year:
        filtered_df = filtered_df[filtered_df['YEAR'] == year]
    if supplier:
        filtered_df = filtered_df[filtered_df['SUPPLIER'] == supplier]
    if start_date and end_date:
        filtered_df = filtered_df[(filtered_df['DATE'] >= start_date) & 
                                 (filtered_df['DATE'] <= end_date)]

    # Sales Trend
    trend_fig = px.line(
        filtered_df.groupby('DATE')['RETAIL SALES'].sum().reset_index(),
        x='DATE', y='RETAIL SALES',
        title='Retail Sales Trend'
    )

    # Heatmap
    heatmap_data = filtered_df.pivot_table(
        index='MONTH', columns='YEAR',
        values='RETAIL SALES', aggfunc='sum'
    ).fillna(0)
    heatmap_fig = px.imshow(
        heatmap_data,
        labels=dict(x="Year", y="Month", color="Sales"),
        title='Monthly Sales Heatmap'
    )

    # Supplier Chart
    supplier_fig = px.bar(
        filtered_df.groupby('SUPPLIER')['RETAIL SALES'].sum().nlargest(10).reset_index(),
        x='RETAIL SALES', y='SUPPLIER',
        title='Top Suppliers'
    )

    # Item Chart
    item_fig = px.pie(
        filtered_df.groupby('ITEM TYPE')['RETAIL SALES'].sum().reset_index(),
        names='ITEM TYPE', values='RETAIL SALES',
        title='Sales by Item Type'
    )

    return trend_fig, heatmap_fig, supplier_fig, item_fig

# --- Run App ---
if __name__ == '__main__':
    app.run(debug=True)