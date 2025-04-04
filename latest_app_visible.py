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

# --- Data Loading and Preprocessing ---
FILE_PATH = "C:\\Users\\dell\\git_hub_repo\\-warehouse-and-retail-sales\\Warehouse_and_Retail_Sales.csv"

try:
    df = pd.read_csv(FILE_PATH)
    
    # Validation logic
    required_columns = ['RETAIL SALES', 'SUPPLIER']
    date_columns = []
    
    if 'DATE' in df.columns:
        date_columns = ['DATE']
    else:
        if {'YEAR', 'MONTH'}.issubset(df.columns):
            date_columns = ['YEAR', 'MONTH']
        else:
            raise ValueError("Missing date columns")
    
    critical_columns = required_columns + date_columns
    missing = [col for col in critical_columns if col not in df.columns]
    
    if missing:
        raise ValueError(f"Missing columns: {missing}")

except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --- Data Preprocessing ---
if 'DATE' not in df.columns:
    df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH']].assign(DAY=1))
else:
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')

df['YEAR'] = df['DATE'].dt.year
df['MONTH'] = df['DATE'].dt.month

text_cols = ['SUPPLIER', 'ITEM TYPE']
for col in text_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.upper().replace({'NAN': 'UNKNOWN', '': 'UNKNOWN'})

numeric_cols = ['RETAIL SALES', 'WAREHOUSE SALES', 'RETAIL TRANSFERS']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

df = df.dropna(subset=['DATE', 'RETAIL SALES'])
min_date = df['DATE'].min().date()
max_date = df['DATE'].max().date()

# --- Dash App Setup ---
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)
server = app.server

# --- UI Components ---
def create_card(title, chart_id, height=400):
    return dbc.Card(
        [
            dbc.CardHeader(
                html.H4(title, className="mb-0 fw-bold", style={"color": "#7DF9FF"})
            ),
            dbc.CardBody(
                dcc.Loading(
                    id=f"loading-{chart_id}",
                    type="circle",
                    children=dcc.Graph(
                        id=chart_id,
                        config={"displayModeBar": True, "displaylogo": False},
                        style={"height": f"{height}px"}
                    )
                )
            )
        ],
        className="shadow-lg mb-4 border-0",
        style={"backgroundColor": "rgba(25,25,25,0.9)"}
    )

def create_filter_dropdown(label, options, id):
    return dbc.Col(
        dbc.Card(
            [
                dbc.CardBody([
                    html.Label(label, className="mb-2 fw-bold", style={"color": "#39FF14"}),
                    dcc.Dropdown(
                        id=id,
                        options=[{"label": f"All {label}s", "value": "All"}] + options,
                        value="All",
                        clearable=False,
                        className="dbc-dark",
                        style={
                            "color": "#39FF14",
                            "backgroundColor": "#2a2a2a",
                            "border": "1px solid #495057"
                        }
                    )
                ])
            ],
            className="border-0 shadow-sm",
            style={"backgroundColor": "rgba(40,40,40,0.9)"}
        ),
        md=3, className="mb-3"
    )

# --- Layout ---
app.layout = dbc.Container(
    fluid=True,
    className="dbc-dark bg-dark",
    style={"minHeight": "100vh", "padding": "20px"},
    children=[
        dbc.Row(
            dbc.Col(
                html.Div([
                    html.H1("Sales Analytics Dashboard", 
                           className="display-3 fw-bold mb-2", 
                           style={"color": "#39FF14"}),
                    html.P("Interactive Sales Analysis Platform", 
                          className="fs-4",
                          style={"color": "#7DF9FF", "opacity": "0.9"})
                ], className="text-center mb-5"),
                width=12
            )
        ),
        
        dbc.Row([
            create_filter_dropdown("Year", [{"label": y, "value": y} for y in sorted(df['YEAR'].unique())], "year_filter"),
            create_filter_dropdown("Supplier", [{"label": s, "value": s} for s in sorted(df['SUPPLIER'].unique())], "supplier_filter"),
            create_filter_dropdown("Item Type", [{"label": t, "value": t} for t in sorted(df['ITEM TYPE'].unique())], "item_type_filter") if "ITEM TYPE" in df.columns else dbc.Col(),
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.Label("Date Range", className="mb-2 fw-bold", style={"color": "#39FF14"}),
                        dcc.DatePickerRange(
                            id='date_range',
                            min_date_allowed=min_date,
                            max_date_allowed=max_date,
                            start_date=min_date,
                            end_date=max_date,
                            className="dbc-dark",
                            style={
                                "backgroundColor": "#2a2a2a",
                                "color": "#39FF14",
                                "border": "1px solid #495057"
                            }
                        )
                    ])
                ], className="border-0 shadow-sm", style={"backgroundColor": "rgba(40,40,40,0.9)"}),
                md=3, className="mb-3"
            )
        ], className="g-4 mb-4"),
        
        dbc.Row([
            dbc.Col(create_card("Sales Trend", "sales_trend_chart"), md=6),
            dbc.Col(create_card("Monthly Heatmap", "heatmap_chart"), md=6),
            dbc.Col(create_card("Supplier Performance", "top_suppliers_chart"), md=4),
            dbc.Col(create_card("Product Mix", "item_type_chart"), md=4),
            dbc.Col(create_card("6-Month Forecast", "forecast_chart"), md=4)
        ], className="g-4"),
        
        dbc.Row(
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader(
                        html.H4("Transaction Data", className="mb-0 fw-bold", style={"color": "#39FF14"})
                    ),
                    dbc.CardBody([
                        dash_table.DataTable(
                            id='data_table',
                            columns=[{"name": i, "id": i} for i in df.columns],
                            page_size=10,
                            filter_action="native",
                            sort_action="native",
                            style_table={"overflowX": "auto"},
                            style_header={
                                "backgroundColor": "rgba(40,40,40,0.9)",
                                "color": "#7DF9FF",
                                "fontWeight": "bold"
                            },
                            style_cell={
                                "backgroundColor": "rgba(25,25,25,0.9)",
                                "color": "white",
                                "border": "1px solid #495057"
                            }
                        ),
                        dbc.Button("Export CSV", 
                                  id="download_button", 
                                  color="primary",
                                  className="mt-3 float-end",
                                  style={"color": "#39FF14"})
                    ])
                ], className="shadow-lg border-0", style={"backgroundColor": "rgba(25,25,25,0.9)"})
            )
        )
    ]
)

# --- Callbacks ---
@app.callback(
    [
        Output("sales_trend_chart", "figure"),
        Output("heatmap_chart", "figure"),
        Output("top_suppliers_chart", "figure"),
        Output("item_type_chart", "figure"),
        Output("forecast_chart", "figure"),
        Output("data_table", "data")
    ],
    [
        Input("year_filter", "value"),
        Input("supplier_filter", "value"),
        Input("item_type_filter", "value"),
        Input("date_range", "start_date"),
        Input("date_range", "end_date")
    ]
)
def update_all(selected_year, selected_supplier, selected_item_type, start_date, end_date):
    filtered_df = get_filtered_data(selected_year, selected_supplier, selected_item_type, start_date, end_date)
    figures = [create_empty_fig()] * 5
    
    try:
        # Sales Trend
        trend_data = filtered_df.resample('MS', on='DATE')['RETAIL SALES'].sum().reset_index()
        figures[0] = dark_figure_template(px.line(trend_data, x='DATE', y='RETAIL SALES'))
        
        # Heatmap
        heatmap_data = filtered_df.pivot_table(index='MONTH', columns='YEAR', values='RETAIL SALES', aggfunc='sum').fillna(0)
        figures[1] = dark_figure_template(px.imshow(heatmap_data, labels=dict(x="Year", y="Month")))
        
        # Top Suppliers
        top_suppliers = filtered_df.groupby('SUPPLIER')['RETAIL SALES'].sum().nlargest(5).reset_index()
        figures[2] = dark_figure_template(px.bar(top_suppliers, x='SUPPLIER', y='RETAIL SALES'))
        
        # Item Type
        if "ITEM TYPE" in filtered_df.columns:
            item_data = filtered_df.groupby('ITEM TYPE')['RETAIL SALES'].sum().reset_index()
            figures[3] = dark_figure_template(px.pie(item_data, names='ITEM TYPE', values='RETAIL SALES'))
        
        # Forecast
        figures[4] = forecast_sales(filtered_df)
        
    except Exception as e:
        traceback.print_exc()
    
    return (*figures, filtered_df.to_dict('records'))

@app.callback(
    Output("download_data", "data"),
    Input("download_button", "n_clicks"),
    [
        State("year_filter", "value"),
        State("supplier_filter", "value"),
        State("item_type_filter", "value"),
        State("date_range", "start_date"),
        State("date_range", "end_date")
    ]
)
def download_data(n_clicks, year, supplier, item_type, start_date, end_date):
    filtered_df = get_filtered_data(year, supplier, item_type, start_date, end_date)
    return dcc.send_data_frame(filtered_df.to_csv, "filtered_sales.csv")

# --- Helper Functions ---
def dark_figure_template(fig):
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={"color": "#adb5bd"},
        title_font={"color": "#7DF9FF"},
        margin={"t": 40, "b": 40}
    )
    return fig

def create_empty_fig(message="No data available"):
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False, font={"size": 16, "color": "#6c757d"})
    fig.update_layout(xaxis_visible=False, yaxis_visible=False)
    return fig

@lru_cache(maxsize=32)
def get_filtered_data(year, supplier, item_type, start_date, end_date):
    filtered = df.copy()
    
    if year != "All":
        filtered = filtered[filtered["YEAR"] == int(year)]
    if supplier != "All":
        filtered = filtered[filtered["SUPPLIER"] == supplier]
    if item_type != "All" and "ITEM TYPE" in filtered.columns:
        filtered = filtered[filtered["ITEM TYPE"] == item_type]
    
    if start_date and end_date:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        filtered = filtered[(filtered['DATE'] >= start_dt) & (filtered['DATE'] <= end_dt)]
    
    return filtered

def forecast_sales(data, periods=6):
    if data.empty:
        return create_empty_fig("Insufficient data for forecast")
    
    ts_data = data.resample('MS', on='DATE')['RETAIL SALES'].sum().reset_index()
    ts_data['TIME'] = np.arange(len(ts_data))
    
    model = LinearRegression()
    model.fit(ts_data[['TIME']], ts_data['RETAIL SALES'])
    
    last_date = ts_data['DATE'].iloc[-1]
    future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, periods+1)]
    forecast = model.predict(np.arange(len(ts_data), len(ts_data)+periods).reshape(-1,1))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts_data['DATE'], y=ts_data['RETAIL SALES'], name="Historical"))
    fig.add_trace(go.Scatter(x=future_dates, y=forecast, name="Forecast", line=dict(dash='dash')))
    return dark_figure_template(fig)

if __name__ == "__main__":
    app.run(debug=True)