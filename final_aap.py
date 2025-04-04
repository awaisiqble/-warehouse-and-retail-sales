import dash
from dash import dcc, html, Input, Output, ctx, no_update
import pandas as pd
import numpy as np
import plotly.express as px
import dash_bootstrap_components as dbc  # For better styling

# Load Data
file_path = file_path = "C:\\Users\\dell\\git_hub_repo\\-warehouse-and-retail-sales\\Warehouse_and_Retail_Sales.csv"
df = pd.read_csv(file_path)

# Data Preprocessing
df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH']].assign(DAY=1))  # Create Date column

# Ensure DATE is treated as a string to avoid sum() issues
df['DATE_STR'] = df['DATE'].astype(str)

# Ensure valid Unix timestamps for the slider
valid_min = df['DATE'].min().timestamp()
valid_max = df['DATE'].max().timestamp()

# Dash App
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(style={'backgroundColor': '#f8f9fa', 'color': 'black', 'padding': '20px'}, children=[
    html.H1("📊 Warehouse & Retail Sales Dashboard", style={'textAlign': 'center', 'color': '#007bff', 'fontSize': '36px'}),

    # Filters
    html.Div([
        dbc.Row([
            dbc.Col(dcc.Dropdown(
                id="year_filter",
                options=[{"label": "All", "value": "All"}] + 
                [{"label": str(year), "value": str(year)} for year in sorted(df['YEAR'].unique())],
                value="All",
                placeholder="Select Year",
                clearable=True,
            ), width=4),
            
            dbc.Col(dcc.Slider(
                id="month_slider",
                min=valid_min,
                max=valid_max,
                value=valid_max,  # Default to latest month
                marks={int(ts): pd.to_datetime(ts, unit='s').strftime('%b %Y') for ts in df['DATE'].astype('int64') // 10**9},
                step=None,
                tooltip={"placement": "bottom", "always_visible": True},
            ), width=8),
        ])
    ], style={'marginBottom': '20px'}),

    # Graph
    dcc.Graph(id="sales_graph")
])

# Callback to update graph
@app.callback(
    Output("sales_graph", "figure"),
    [Input("year_filter", "value"),
     Input("month_slider", "value")]
)
def update_graph(selected_year, selected_date):
    # Convert timestamp to datetime
    selected_date = pd.to_datetime(selected_date, unit='s')

    # Filter data based on selection
    filtered_df = df.copy()
    if selected_year != "All":
        filtered_df = filtered_df[filtered_df['YEAR'] == int(selected_year)]
    
    filtered_df = filtered_df[filtered_df['DATE'] <= selected_date]

    # Debugging output
    print(f"Filtered DataFrame Shape: {filtered_df.shape}")
    print(f"Filtered DataFrame Columns: {filtered_df.columns}")

    # Group by SUPPLIER and sum numeric columns, excluding DATE
    numeric_cols = ['RETAIL SALES', 'RETAIL TRANSFERS', 'WAREHOUSE SALES']
    filtered_df = filtered_df.groupby(['SUPPLIER'])[numeric_cols].sum().reset_index()

    # Create graph
    fig = px.bar(filtered_df, x="SUPPLIER", y="RETAIL SALES", title="Retail Sales by Supplier")

    return fig


# Run the app
if __name__ == "__main__":
    app.run(debug=True)
