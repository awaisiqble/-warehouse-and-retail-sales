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

if 'DATE' not in df.columns:
    df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH']].assign(DAY=1))
else:
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')

df['YEAR'] = df['DATE'].dt.year
df['MONTH'] = df['DATE'].dt.month

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

for col in ['RETAIL SALES', 'WAREHOUSE SALES', 'RETAIL TRANSFERS']:
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
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server

# --- UI Components with Proper Indentation ---
def create_card(title, chart_id, height=400):
    return dbc.Card(
        [
            dbc.CardHeader(
                html.H4(title, className="card-title mb-0"),
                className="bg-dark border-bottom"
            ),
            dbc.CardBody(
                dcc.Loading(
                    id=f"loading-{chart_id}",
                    type="circle",
                    children=dcc.Graph(
                        id=chart_id,
                        config={
                            "displayModeBar": True,
                            "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                            "displaylogo": False
                        },
                        style={"height": f"{height}px"}
                    )
                )
            ),
            dbc.CardFooter(className="bg-dark border-top")
        ],
        className="shadow-lg mb-4 border-0",
        style={"backgroundColor": "rgba(25,25,25,0.9)"}
    )

def create_filter_dropdown(label, options, id, value="All"):
    return dbc.Col(
        dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.Label(label, className="text-light mb-2"),
                        dcc.Dropdown(
                            id=id,
                            options=[{"label": f"All {label}s", "value": "All"}] + options,
                            value=value,
                            clearable=False,
                            className="dbc-dark"
                        )
                    ]
                )
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
                html.Div(
                    [
                        html.H1("Sales Analytics Dashboard", 
                               className="text-light mb-2 display-4"),
                        html.P("Interactive Sales Analysis Platform", 
                              className="text-secondary lead")
                    ],
                    className="text-center mb-5"
                ),
                width=12
            )
        ),
        
        dbc.Row(
            [
                create_filter_dropdown(
                    "Year",
                    [{"label": y, "value": y} for y in sorted(df['YEAR'].unique())],
                    "year_filter"
                ),
                create_filter_dropdown(
                    "Supplier",
                    [{"label": s, "value": s} for s in sorted(df['SUPPLIER'].unique())],
                    "supplier_filter"
                ),
                create_filter_dropdown(
                    "Item Type",
                    [{"label": t, "value": t} for t in sorted(df['ITEM TYPE'].unique())] 
                    if "ITEM TYPE" in df.columns else [],
                    "item_type_filter"
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.Label("Date Range", className="text-light mb-2"),
                                    dcc.DatePickerRange(
                                        id='date_range',
                                        min_date_allowed=min_date,
                                        max_date_allowed=max_date,
                                        start_date=min_date,
                                        end_date=max_date,
                                        className="dbc-dark",
                                        style={"border": "1px solid #495057"}
                                    )
                                ]
                            )
                        ],
                        className="border-0 shadow-sm",
                        style={"backgroundColor": "rgba(40,40,40,0.9)"}
                    ),
                    md=3, className="mb-3"
                )
            ],
            className="g-4 mb-4"
        ),
        
        dbc.Row(
            [
                dbc.Col(create_card("Sales Trend", "sales_trend_chart"), md=6),
                dbc.Col(create_card("Monthly Heatmap", "heatmap_chart"), md=6),
                dbc.Col(create_card("Supplier Performance", "top_suppliers_chart"), md=4),
                dbc.Col(create_card("Product Mix", "item_type_chart"), md=4),
                dbc.Col(create_card("6-Month Forecast", "forecast_chart"), md=4)
            ],
            className="g-4"
        ),
        
        dbc.Row(
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader(
                            html.H4("Transaction Data", className="mb-0"),
                            className="bg-dark border-bottom"
                        ),
                        dbc.CardBody(
                            [
                                dash_table.DataTable(
                                    id='data_table',
                                    columns=[{"name": i, "id": i} for i in df.columns],
                                    data=[],
                                    page_size=10,
                                    filter_action="native",
                                    sort_action="native",
                                    style_table={
                                        "overflowX": "auto",
                                        "backgroundColor": "rgba(25,25,25,0.9)"
                                    },
                                    style_header={
                                        "backgroundColor": "rgba(40,40,40,0.9)",
                                        "color": "white",
                                        "fontWeight": "bold"
                                    },
                                    style_cell={
                                        "backgroundColor": "rgba(25,25,25,0.9)",
                                        "color": "white",
                                        "border": "1px solid #495057"
                                    },
                                    style_filter={
                                        "backgroundColor": "rgba(40,40,40,0.9)",
                                        "border": "1px solid #495057"
                                    }
                                ),
                                dbc.Button(
                                    "Export Data",
                                    id="download_button",
                                    color="primary",
                                    className="mt-3 float-end",
                                    outline=True
                                ),
                                dcc.Download(id="download_data")
                            ]
                        )
                    ],
                    className="shadow-lg border-0",
                    style={"backgroundColor": "rgba(25,25,25,0.9)"}
                ),
                className="mb-4"
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
def update_all_charts(selected_year, selected_supplier, selected_item_type, start_date, end_date):
    filtered_df = get_filtered_data(selected_year, selected_supplier, selected_item_type, start_date, end_date)
    
    figures = [create_empty_fig()] * 5
    table_data = []

    try:
        # Sales Trend
        trend_data = filtered_df.resample('MS', on='DATE')['RETAIL SALES'].sum().reset_index()
        fig1 = px.line(trend_data, x='DATE', y='RETAIL SALES', title='Retail Sales Trend')
        figures[0] = dark_figure_template(fig1)

        # Heatmap
        month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        heatmap_data = filtered_df.pivot_table(index='MONTH', columns='YEAR', 
                                            values='RETAIL SALES', aggfunc='sum').fillna(0)
        heatmap_data = heatmap_data.reindex(range(1,13), fill_value=0)
        heatmap_data.index = [month_names[m-1] for m in heatmap_data.index]
        fig2 = px.imshow(heatmap_data, labels=dict(x="Year", y="Month", color="Sales ($)"))
        figures[1] = dark_figure_template(fig2)

        # Top Suppliers
        top_suppliers = filtered_df.groupby('SUPPLIER')['RETAIL SALES'].sum().nlargest(5).reset_index()
        fig3 = px.bar(top_suppliers, x='SUPPLIER', y='RETAIL SALES', title='Top Suppliers')
        figures[2] = dark_figure_template(fig3)

        # Item Type
        if "ITEM TYPE" in filtered_df.columns:
            item_data = filtered_df.groupby('ITEM TYPE')['RETAIL SALES'].sum().reset_index()
            fig4 = px.pie(item_data, names='ITEM TYPE', values='RETAIL SALES', title='Sales by Item Type')
            figures[3] = dark_figure_template(fig4)

        # Forecast
        forecast_fig = forecast_sales(filtered_df)
        figures[4] = dark_figure_template(forecast_fig)

        table_data = filtered_df.to_dict('records')

    except Exception as e:
        print(f"Error updating charts: {str(e)}")
        traceback.print_exc()

    return (*figures, table_data)

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
def download_csv(n_clicks, year, supplier, item_type, start_date, end_date):
    filtered_df = get_filtered_data(year, supplier, item_type, start_date, end_date)
    return dcc.send_data_frame(filtered_df.to_csv, "filtered_data.csv")

# --- Helper Functions ---
def dark_figure_template(fig):
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={"color": "#adb5bd"},
        xaxis={"gridcolor": "#495057"},
        yaxis={"gridcolor": "#495057"},
        margin={"t": 40, "b": 40}
    )
    return fig

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
    
    if selected_year and selected_year != "All":
        filtered_df = filtered_df[filtered_df["YEAR"] == int(selected_year)]
    
    if selected_supplier and selected_supplier != "All":
        filtered_df = filtered_df[filtered_df["SUPPLIER"] == selected_supplier]
    
    if "ITEM TYPE" in filtered_df.columns and selected_item_type and selected_item_type != "All":
        filtered_df = filtered_df[filtered_df["ITEM TYPE"] == selected_item_type]
    
    if start_date and end_date:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        filtered_df = filtered_df[
            (filtered_df['DATE'] >= start_dt) & 
            (filtered_df['DATE'] <= end_dt)
        ]
    
    return filtered_df

def forecast_sales(filtered_df, periods=6):
    if filtered_df.empty:
        return create_empty_fig("No forecast due to insufficient data.")

    ts_data = filtered_df.resample('MS', on='DATE')['RETAIL SALES'].sum().reset_index()
    ts_data.sort_values('DATE', inplace=True)
    
    ts_data['TIME'] = np.arange(len(ts_data))
    X = ts_data[['TIME']]
    y = ts_data['RETAIL SALES']
    
    model = LinearRegression()
    model.fit(X, y)
    
    last_time = ts_data['TIME'].iloc[-1]
    future_times = np.arange(last_time + 1, last_time + periods + 1).reshape(-1, 1)
    forecast_values = model.predict(future_times)
    
    last_date = ts_data['DATE'].iloc[-1]
    future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, periods + 1)]
    forecast_df = pd.DataFrame({
        'DATE': future_dates,
        'FORECAST SALES': forecast_values
    })
    
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

# --- Run App ---
if __name__ == "__main__":
    app.run(debug=True)