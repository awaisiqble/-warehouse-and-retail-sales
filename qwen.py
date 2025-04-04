import dash
from dash import dcc, html, Input, Output, ctx, no_update, callback_context
import pandas as pd
import numpy as np
import plotly.express as px
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

# Load Data
df = pd.read_csv("C:\\Users\\dell\\git_hub_repo\\-warehouse-and-retail-sales\\Warehouse_and_Retail_Sales.csv")

# Data Preprocessing
df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH']].assign(DAY=1))
df['SUPPLIER'] = df['SUPPLIER'].str.strip().str.upper().fillna('UNKNOWN SUPPLIER')
df['ITEM TYPE'] = df['ITEM TYPE'].str.strip().str.upper().fillna('UNKNOWN ITEM TYPE')

# Initialize App
app = dash.Dash(__name__, 
               external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
               meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}])

# App Layout
app.layout = dbc.Container(fluid=True, children=[
    # Navbar
    dbc.NavbarSimple(
        brand=html.Div([
            html.I(className="fas fa-chart-line fa-lg me-2", style={'color': '#007bff'}),
            "Sales Analytics Dashboard"
        ]),
        color="#f8f9fa",
        dark=False,
        className="mb-4 shadow-sm",
        children=[
            dbc.Button("Refresh", id="refresh-btn", outline=True, color="primary", className="me-2"),
            dbc.Button("Reset Filters", id="reset-btn", outline=True, color="danger")
        ]
    ),
    
    # Filter Row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Filters", className="bg-primary text-white"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(dcc.Dropdown(
                            id="year_filter",
                            placeholder="Select Year",
                            className="filter-dropdown"
                        ), width=6, sm=6, md=3),
                        
                        dbc.Col(dcc.Dropdown(
                            id="supplier_filter",
                            placeholder="Select Suppliers",
                            multi=True,
                            className="filter-dropdown"
                        ), width=6, sm=6, md=3),
                        
                        dbc.Col(dcc.Dropdown(
                            id="item_type_filter",
                            placeholder="Select Item Types",
                            multi=True,
                            className="filter-dropdown"
                        ), width=6, sm=6, md=3),
                        
                        dbc.Col(dbc.Button("Apply Filters", id="apply_filters", 
                                          color="success", outline=True, 
                                          className="w-100 mt-2 mt-md-0"), 
                                width=6, sm=6, md=3)
                    ])
                ])
            ], className="mb-4")
        ], width=12)
    ]),
    
    # Date Slider
    dbc.Row([
        dbc.Col(dcc.Slider(
            id='month_slider',
            marks=None,
            tooltip={"placement": "bottom", "always_visible": True},
            className="custom-slider"
        ), width=12)
    ], className="mb-4"),
    
    # KPIs Row
    dbc.Row(id="kpi-row", className="g-2 mb-4"),
    
    # Charts Row
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader([
                html.I(className="fas fa-chart-line me-2"),
                "Sales Trend"
            ]),
            dbc.CardBody(dcc.Graph(id="sales_trend_chart", config={'displayModeBar': False}))
        ]), width=12, lg=6),
        
        dbc.Col(dbc.Card([
            dbc.CardHeader([
                html.I(className="fas fa-users me-2"),
                "Supplier Performance"
            ]),
            dbc.CardBody(dcc.Graph(id="supplier_comparison_chart", config={'displayModeBar': False}))
        ]), width=12, lg=6)
    ], className="g-2 mb-4"),
    
    # Download and Footer
    dbc.Row([
        dbc.Col([
            dbc.Button("Export Data", id="download_btn", color="primary", 
                      className="w-100", outline=True),
            dcc.Download(id="download_data")
        ], width=12),
        dbc.Col(html.Div([
            "© 2024 Sales Analytics Team",
            html.Br(),
            html.Small("Powered by Dash", className="text-muted")
        ]), width=12, className="text-center mt-4")
    ])
], style={'padding': '20px'})

# Initialize Callbacks
@app.callback(
    [Output('year_filter', 'options'),
     Output('year_filter', 'value')],
    [Input('refresh-btn', 'n_clicks')]
)
def update_year_options(n_clicks):
    options = [{'label': 'All Years', 'value': 'All'}] + \
              [{'label': str(y), 'value': y} for y in sorted(df['YEAR'].unique())]
    return options, 'All'

@app.callback(
    [Output('supplier_filter', 'options'),
     Output('item_type_filter', 'options')],
    [Input('year_filter', 'value')]
)
def update_dependent_filters(selected_year):
    filtered = df if selected_year == 'All' else df[df['YEAR'] == selected_year]
    
    supplier_options = [{'label': s, 'value': s} for s in sorted(filtered['SUPPLIER'].unique())]
    item_options = [{'label': i, 'value': i} for i in sorted(filtered['ITEM TYPE'].unique())]
    
    return supplier_options, item_options

@app.callback(
    [Output('month_slider', 'min'),
     Output('month_slider', 'max'),
     Output('month_slider', 'marks'),
     Output('month_slider', 'value')],
    [Input('year_filter', 'value')]
)
def update_slider(selected_year):
    filtered = df if selected_year == 'All' else df[df['YEAR'] == selected_year]
    
    min_date = filtered['DATE'].min()
    max_date = filtered['DATE'].max()
    
    dates = pd.date_range(start=min_date, end=max_date, freq='MS')
    marks = {int(d.timestamp()): d.strftime('%b %Y') for d in dates}
    
    return min_date.timestamp(), max_date.timestamp(), marks, max_date.timestamp()

@app.callback(
    [Output("kpi-row", "children"),
     Output("sales_trend_chart", "figure"),
     Output("supplier_comparison_chart", "figure"),
     Output("download_data", "data")],
    [Input("apply_filters", "n_clicks"),
     Input("download_btn", "n_clicks")],
    [dash.dependencies.State("year_filter", "value"),
     dash.dependencies.State("supplier_filter", "value"),
     dash.dependencies.State("item_type_filter", "value"),
     dash.dependencies.State("month_slider", "value")]
)
def update_dashboard(n_apply, n_download, year, suppliers, item_types, date_val):
    filtered_df = df.copy()
    
    # Apply Filters
    if year != 'All':
        filtered_df = filtered_df[filtered_df['YEAR'] == year]
    if suppliers:
        filtered_df = filtered_df[filtered_df['SUPPLIER'].isin(suppliers)]
    if item_types:
        filtered_df = filtered_df[filtered_df['ITEM TYPE'].isin(item_types)]
    if date_val:
        target_month = pd.to_datetime(date_val, unit='s').to_period('M')
        filtered_df = filtered_df[filtered_df['DATE'].dt.to_period('M') <= target_month]
    
    # Handle empty data
    if filtered_df.empty:
        return (
            [dbc.Col(dbc.Card("No Data Available", className="text-center p-4"), width=12)]*3,
            create_empty_fig(),
            create_empty_fig(),
            None
        )
    
    # Calculate KPIs
    retail_sales = filtered_df['RETAIL SALES'].sum()
    warehouse_sales = filtered_df['WAREHOUSE SALES'].sum()
    transfers = filtered_df['RETAIL TRANSFERS'].sum()
    
    # Create KPI Cards
    kpi_cards = [
        create_kpi_card("Retail Sales", retail_sales, "#007bff", "fas fa-store"),
        create_kpi_card("Warehouse Sales", warehouse_sales, "#28a745", "fas fa-warehouse"),
        create_kpi_card("Transfers", transfers, "#6c757d", "fas fa-exchange-alt")
    ]
    
    # Create Charts
    trend_fig = create_trend_chart(filtered_df)
    supplier_fig = create_supplier_chart(filtered_df)
    
    # Handle Download
    if ctx.triggered_id == "download_btn":
        return (
            kpi_cards,
            trend_fig,
            supplier_fig,
            dcc.send_data_frame(filtered_df.to_csv, "filtered_sales_data.csv", index=False)
        )

    return kpi_cards, trend_fig, supplier_fig, None

def create_kpi_card(title, value, color, icon):
    return dbc.Col(dbc.Card([
        dbc.CardBody([
            html.Div([
                html.I(className=f"{icon} fa-2x", style={'color': color}),
                html.H5(title, className="mt-2"),
                html.H3(f"${value:,.0f}", style={'color': color})
            ], className="text-center")
        ])
    ], className="shadow-sm h-100"), width=12, sm=6, md=4, lg=4)

def create_trend_chart(data):
    trend_data = data.groupby('DATE')[['RETAIL SALES', 'WAREHOUSE SALES']].sum().reset_index()
    fig = px.line(trend_data, x='DATE', y=['RETAIL SALES', 'WAREHOUSE SALES'],
                 title="Sales Performance Over Time")
    fig.update_layout(
        legend_title_text='Sales Type',
        xaxis_title="Date",
        yaxis_title="Sales Amount",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def create_supplier_chart(data):
    supplier_data = data.groupby('SUPPLIER')['RETAIL SALES'].sum().nlargest(10).reset_index()
    fig = px.bar(supplier_data, x='RETAIL SALES', y='SUPPLIER',
                orientation='h', title="Top Performing Suppliers")
    fig.update_layout(
        yaxis={'categoryorder':'total ascending'},
        xaxis_title="Total Sales",
        yaxis_title="Supplier",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def create_empty_fig():
    fig = px.scatter()
    fig.add_annotation(text="No Data Available",
                      xref="paper", yref="paper",
                      showarrow=False,
                      font=dict(size=16, color="#6c757d"))
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False)
    )
    return fig

@app.callback(
    [Output('year_filter', 'value'),
     Output('supplier_filter', 'value'),
     Output('item_type_filter', 'value'),
     Output('month_slider', 'value')],
    [Input('reset-btn', 'n_clicks')]
)
def reset_filters(n_clicks):
    if n_clicks:
        return 'All', [], [], df['DATE'].max().timestamp()
    return no_update, no_update, no_update, no_update

if __name__ == "__main__":
    app.run(debug=False, dev_tools_ui=False)