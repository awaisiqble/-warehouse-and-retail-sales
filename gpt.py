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

# Clean string columns
df['SUPPLIER'] = df['SUPPLIER'].fillna('Unknown Supplier').str.strip()
df['ITEM TYPE'] = df['ITEM TYPE'].fillna('Unknown Item Type').str.strip()

# Initialize the Dash App
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout
app.layout = dbc.Container(
    fluid=True,
    children=[
        dbc.Row([
            dbc.Col(
                html.H1("📊 Warehouse & Retail Sales Dashboard", style={'textAlign': 'center', 'color': '#007bff'}),
                width=12
            )
        ], className="mb-4"),

        dbc.Row([
            dbc.Col(
                dcc.Dropdown(
                    id="year_filter",
                    options=[{"label": "All", "value": "All"}] + [{"label": str(y), "value": y} for y in sorted(df['YEAR'].unique())],
                    value="All",
                    clearable=False,
                    style={'color': 'black'}
                ), width=3
            ),
            dbc.Col(
                dcc.Dropdown(
                    id="supplier_filter",
                    options=[{"label": s, "value": s} for s in sorted(df['SUPPLIER'].unique())],
                    multi=True,
                    placeholder="Select Suppliers...",
                    style={'color': 'black'}
                ), width=3
            ),
            dbc.Col(
                dcc.Dropdown(
                    id="item_type_filter",
                    options=[{"label": i, "value": i} for i in sorted(df['ITEM TYPE'].unique())],
                    multi=True,
                    placeholder="Select Item Types...",
                    style={'color': 'black'}
                ), width=3
            ),
            dbc.Col(
                html.Button("Download Data", id="download_btn", style={'backgroundColor': '#007bff', 'color': 'white'}),
                width=3
            ),
        ], className="g-2 mb-4"),

        dbc.Row([
            dbc.Col(
                dcc.DatePickerRange(
                    id='date_range',
                    min_date_allowed=df['DATE'].min(),
                    max_date_allowed=df['DATE'].max(),
                    start_date=df['DATE'].min(),
                    end_date=df['DATE'].max(),
                    display_format='YYYY-MM-DD',
                    style={"width": "100%"}
                ),
                width=12
            )
        ], className="mb-4"),

        dbc.Row([
            dbc.Col(
                html.Div(id="kpi_display", style={'display': 'flex', 'justifyContent': 'space-around'}),
                width=12
            )
        ], className="mb-4"),

        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="sales_trend_chart"))), width=6),
            dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="supplier_comparison_chart"))), width=6),
        ], className="g-2 mb-4"),

        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="item_type_sales_chart"))), width=6),
            dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="heatmap_chart"))), width=6),
        ], className="g-2 mb-4"),

        dcc.Download(id="download_data")
    ],
    style={'padding': '20px'}
)

def create_empty_fig():
    fig = px.scatter(title="No Data Available")
    fig.update_layout(
        annotations=[dict(
            text="No Data Available",
            showarrow=False,
            x=0.5,
            y=0.5,
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
        Input("date_range", "start_date"),
        Input("date_range", "end_date"),
        Input("download_btn", "n_clicks")
    ]
)
def update_dashboard(selected_year, selected_suppliers, selected_item_types, start_date, end_date, download_click):
    try:
        filtered_df = df.copy()

        # Debug info
        print("Selected Year:", selected_year)
        print("Selected Suppliers:", selected_suppliers)
        print("Selected Item Types:", selected_item_types)
        print("Date Range:", start_date, "to", end_date)

        # Apply filters
        if selected_year != "All":
            filtered_df = filtered_df[filtered_df["YEAR"] == int(selected_year)]
        
        if selected_suppliers:
            filtered_df = filtered_df[filtered_df["SUPPLIER"].isin(selected_suppliers)]
        
        if selected_item_types:
            filtered_df = filtered_df[filtered_df["ITEM TYPE"].isin(selected_item_types)]
        
        if start_date and end_date:
            filtered_df = filtered_df[
                (filtered_df['DATE'] >= pd.to_datetime(start_date)) &
                (filtered_df['DATE'] <= pd.to_datetime(end_date))
            ]

        print("Filtered Rows Count:", len(filtered_df))

        if filtered_df.empty:
            return (
                [dbc.Card(dbc.CardBody([html.H5("No Data Available", className="card-title")]), className="bg-light text-center m-2 shadow-sm")] * 3,
                create_empty_fig(),
                create_empty_fig(),
                create_empty_fig(),
                create_empty_fig(),
                None
            )

        # KPIs
        kpi_values = filtered_df[['RETAIL SALES', 'WAREHOUSE SALES', 'RETAIL TRANSFERS']].sum()

        kpi_elements = [
            dbc.Card(dbc.CardBody([html.H5(f"Total Retail Sales: ${kpi_values['RETAIL SALES']:,.2f}")]), className="bg-light text-center m-2 shadow-sm"),
            dbc.Card(dbc.CardBody([html.H5(f"Total Warehouse Sales: ${kpi_values['WAREHOUSE SALES']:,.2f}")]), className="bg-light text-center m-2 shadow-sm"),
            dbc.Card(dbc.CardBody([html.H5(f"Total Transfers: ${kpi_values['RETAIL TRANSFERS']:,.2f}")]), className="bg-light text-center m-2 shadow-sm")
        ]

        # Charts
        trend_data = filtered_df.groupby("DATE")[["RETAIL SALES", "WAREHOUSE SALES"]].sum().reset_index()
        trend_fig = px.line(trend_data, x="DATE", y=["RETAIL SALES", "WAREHOUSE SALES"], title="📈 Sales Trend Over Time")

        supplier_data = filtered_df.groupby("SUPPLIER")['RETAIL SALES'].sum().nlargest(10).reset_index()
        supplier_fig = px.bar(supplier_data, x="RETAIL SALES", y="SUPPLIER", orientation="h", title="🏭 Top 10 Suppliers by Retail Sales")

        item_type_data = filtered_df.groupby("ITEM TYPE")['RETAIL SALES'].sum().reset_index()
        item_type_fig = px.pie(item_type_data, values="RETAIL SALES", names="ITEM TYPE", title="🏷️ Sales by Item Type")

        heatmap_data = filtered_df.pivot_table(index="MONTH", columns="YEAR", values="RETAIL SALES", aggfunc="sum")
        heatmap_fig = px.imshow(heatmap_data, labels=dict(x="Year", y="Month", color="Sales"), title="🌡️ Sales Heatmap") if not heatmap_data.empty else create_empty_fig()

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
