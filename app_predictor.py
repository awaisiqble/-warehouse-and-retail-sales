 
from flask import Flask
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
import pickle

# Load Data
file_path = r"C:\Users\dell\git hub repo\-warehouse-and-retail-sales\Cleaned_Warehouse_and_Retail_Sales.csv"
df = pd.read_csv(file_path)
df["DATE"] = pd.to_datetime(df["YEAR"].astype(str) + "-" + df["MONTH"].astype(str) + "-01")

# Load Machine Learning Model
model_path = "retail_sales_model.pkl"
with open(model_path, "rb") as f:
    sales_model = pickle.load(f)

# Initialize Flask Server
server = Flask(__name__)

# Initialize Dash App
app = dash.Dash(__name__, server=server)

# Layout of the Dashboard
app.layout = html.Div([
    html.H1("📊 Retail Sales Predictor Dashboard", style={'textAlign': 'center'}),

    # Filters for Data Visualization
    html.Div([
        html.Label("Select Year:"),
        dcc.Dropdown(id='year_filter', options=[{'label': y, 'value': y} for y in sorted(df['YEAR'].unique())], 
                     value=df['YEAR'].max(), clearable=False),

        html.Label("Select Supplier:"),
        dcc.Dropdown(id='supplier_filter', options=[{'label': s, 'value': s} for s in df['SUPPLIER'].unique()], 
                     value=df['SUPPLIER'].unique()[0], clearable=True),
    ], style={'width': '48%', 'display': 'inline-block'}),

    # Prediction Section
    html.Div([
        html.H3("🔮 Predict Retail Sales", style={'textAlign': 'center'}),
        
        html.Label("Enter Item Type:"),
        dcc.Input(id='item_type', type='text', value="General Merchandise", style={'margin-bottom': '10px'}),

        html.Label("Enter Warehouse Sales:"),
        dcc.Input(id='warehouse_sales', type='number', value=10000, style={'margin-bottom': '10px'}),

        html.Label("Select Year & Month:"),
        dcc.DatePickerSingle(id='prediction_date', date="2024-06-01"),

        html.Button("Predict Sales", id='predict_button', n_clicks=0, style={'margin-top': '10px'}),

        html.H4(id='prediction_output', style={'color': 'blue', 'margin-top': '10px'})
    ], style={'width': '48%', 'display': 'inline-block', 'padding': '20px', 'border': '1px solid black'}),

    # Sales Trends Line Chart
    dcc.Graph(id='sales_trend'),

    # Sales by Item Type Pie Chart
    dcc.Graph(id='sales_by_item'),

    # Retail vs Warehouse Sales Comparison
    dcc.Graph(id='retail_vs_warehouse'),

    # Top Suppliers Bar Chart
    dcc.Graph(id='top_suppliers'),
])

# Callback for Predictions
@app.callback(
    Output('prediction_output', 'children'),
    Input('predict_button', 'n_clicks'),
    State('item_type', 'value'),
    State('warehouse_sales', 'value'),
    State('prediction_date', 'date')
)
def predict_sales(n_clicks, item_type, warehouse_sales, prediction_date):
    if n_clicks > 0:
        # Example transformation (ensure this matches how the model was trained)
        input_data = [[warehouse_sales]]  # Modify based on model needs

        # Make Prediction
        predicted_sales = sales_model.predict(input_data)[0]

        return f"Predicted Retail Sales: ${predicted_sales:,.2f}"
    return ""

# Callbacks for Charts
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
    app.run(debug=True, host='0.0.0.0', port=8080)
