 
from flask import Flask
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

# Load Data
file_path = r"C:\Users\dell\git hub repo\-warehouse-and-retail-sales\Cleaned_Warehouse_and_Retail_Sales.csv"
df = pd.read_csv(file_path)
df["DATE"] = pd.to_datetime(df["YEAR"].astype(str) + "-" + df["MONTH"].astype(str) + "-01")

# Initialize Flask Server
server = Flask(__name__)

# Initialize Dash App
app = dash.Dash(__name__, server=server)

# Layout of the Dashboard
app.layout = html.Div([
    html.H1("📊 Retail Sales Dashboard", style={'textAlign': 'center'}),

    # Filters
    html.Div([
        html.Label("Select Year:"),
        dcc.Dropdown(id='year_filter', options=[{'label': y, 'value': y} for y in sorted(df['YEAR'].unique())], 
                     value=df['YEAR'].max(), clearable=False),

        html.Label("Select Supplier:"),
        dcc.Dropdown(id='supplier_filter', options=[{'label': s, 'value': s} for s in df['SUPPLIER'].unique()], 
                     value=df['SUPPLIER'].unique()[0], clearable=True),
    ], style={'width': '48%', 'display': 'inline-block'}),

    # Sales Trends Line Chart
    dcc.Graph(id='sales_trend'),

    # Sales by Item Type Pie Chart
    dcc.Graph(id='sales_by_item'),

    # Retail vs Warehouse Sales Comparison
    dcc.Graph(id='retail_vs_warehouse'),

    # Top Suppliers Bar Chart
    dcc.Graph(id='top_suppliers'),
])

# Callbacks for interactivity
@app.callback(
    Output('sales_trend', 'figure'),
    Output('sales_by_item', 'figure'),
    Output('retail_vs_warehouse', 'figure'),
    Output('top_suppliers', 'figure'),
    Input('year_filter', 'value'),
    Input('supplier_filter', 'value')
)
def update_charts(selected_year, selected_supplier):
    filtered_df = df[(df["YEAR"] == selected_year)]
    if selected_supplier:
        filtered_df = filtered_df[filtered_df["SUPPLIER"] == selected_supplier]

    fig_trend = px.line(filtered_df, x="DATE", y="RETAIL SALES", title="📈 Sales Trends Over Time")
    fig_pie = px.pie(filtered_df, names="ITEM TYPE", values="RETAIL SALES", title="🛒 Sales by Item Type")
    fig_bar = px.bar(filtered_df, x="ITEM TYPE", y=["RETAIL SALES", "WAREHOUSE SALES"], 
                     title="🏪 Retail vs. Warehouse Sales", barmode="group")
    top_suppliers = df.groupby("SUPPLIER")["RETAIL SALES"].sum().reset_index().nlargest(10, "RETAIL SALES")
    fig_top_suppliers = px.bar(top_suppliers, x="SUPPLIER", y="RETAIL SALES", title="🏆 Top 10 Suppliers by Sales")

    return fig_trend, fig_pie, fig_bar, fig_top_suppliers

# Run the application
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8080, debug=True)
