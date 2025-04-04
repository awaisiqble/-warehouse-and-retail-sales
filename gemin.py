# -*- coding: utf-8 -*-
import dash
from dash import dcc, html, Input, Output, State, ctx, dash_table
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go # Import graph_objects for create_empty_fig
import dash_bootstrap_components as dbc
import traceback
from datetime import date

# --- Configuration ---
# !!! IMPORTANT: Update this path to your actual file location !!!
FILE_PATH = "C:\\Users\\dell\\git_hub_repo\\-warehouse-and-retail-sales\\Warehouse_and_Retail_Sales.csv"
# FILE_PATH = "Warehouse_and_Retail_Sales.csv" # Use this if the file is in the same directory

# --- Load Data ---
try:
    df = pd.read_csv(FILE_PATH)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: File not found at {FILE_PATH}")
    print("Please update the FILE_PATH variable in the script.")
    exit() # Exit if the file isn't found
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --- Data Preprocessing ---
print("Preprocessing data...")
# Convert YEAR and MONTH to a proper DATE column if they exist
if 'YEAR' in df.columns and 'MONTH' in df.columns:
    df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH']].assign(DAY=1))
else:
    print("Warning: YEAR or MONTH column missing. Cannot create DATE column.")
    # Handle case where DATE might already exist or needs different creation logic
    if 'DATE' not in df.columns:
        print("Error: DATE column cannot be created and does not exist.")
        # exit() # Or create a dummy date if appropriate for testing
        df['DATE'] = pd.NaT # Assign Not-a-Time value

# Clean and standardize text columns (handle potential whitespace and case issues)
for col in ['SUPPLIER', 'ITEM TYPE']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.upper()
        df[col] = df[col].replace('NAN', 'UNKNOWN')
    else:
        print(f"Warning: Column '{col}' not found in CSV. Creating it as 'UNKNOWN'.")
        df[col] = 'UNKNOWN'

# Fill remaining NaNs - apply AFTER initial cleaning
df['SUPPLIER'] = df['SUPPLIER'].fillna('UNKNOWN SUPPLIER')
df['ITEM TYPE'] = df['ITEM TYPE'].fillna('UNKNOWN ITEM TYPE')

# Ensure numeric columns are numeric, coerce errors to NaN
numeric_cols = ['RETAIL SALES', 'WAREHOUSE SALES', 'RETAIL TRANSFERS']
for col in numeric_cols:
     if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
     else:
        print(f"Warning: Numeric column '{col}' not found in CSV. Creating it with 0.")
        df[col] = 0

# Drop rows where essential sales figures are non-numeric/missing *after* coercion
if 'RETAIL SALES' in df.columns:
    initial_rows = len(df)
    df = df.dropna(subset=['RETAIL SALES'])
    print(f"Dropped {initial_rows - len(df)} rows due to missing RETAIL SALES.")
else:
    print("Warning: 'RETAIL SALES' column not found. Cannot drop rows based on it.")

# Get unique sorted lists for dropdowns *after* cleaning and dropping NA
all_years = sorted(df['YEAR'].unique()) if 'YEAR' in df.columns else []
all_suppliers = sorted(df['SUPPLIER'].unique()) if 'SUPPLIER' in df.columns else []
all_item_types = sorted(df['ITEM TYPE'].unique()) if 'ITEM TYPE' in df.columns else []

# Get date range for picker
# Ensure DATE column is datetime before finding min/max
if 'DATE' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['DATE']):
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
df = df.dropna(subset=['DATE']) # Drop rows where date conversion failed

min_date = df['DATE'].min().date() if 'DATE' in df.columns and not df['DATE'].empty else date.today()
max_date = df['DATE'].max().date() if 'DATE' in df.columns and not df['DATE'].empty else date.today()

print("Preprocessing complete.")
print(f"Date Range: {min_date} to {max_date}")
print(f"Unique Years: {len(all_years)}")
print(f"Unique Suppliers: {len(all_suppliers)}")
print(f"Unique Item Types: {len(all_item_types)}")


# --- Initialize the Dash App ---
# Added meta_tags for responsiveness
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server # Expose server for deployment

# --- Helper Functions ---
def create_empty_fig(message="No data for selected filters"):
    """Creates a Plotly figure indicating no data."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        align='center',
        showarrow=False,
        xref='paper',
        yref='paper',
        x=0.5,
        y=0.5,
        font=dict(size=16, color='#6c757d') # Use a subtle color
    )
    fig.update_layout(
        xaxis={'visible': False},
        yaxis={'visible': False},
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=30, b=30, l=30, r=30) # Add some margin
    )
    return fig

def create_error_fig(error_message="An error occurred"):
     """Creates a Plotly figure indicating an error."""
     return create_empty_fig(f"⚠️ Error: {error_message}")

# --- Define the Layout ---
app.layout = dbc.Container(
    fluid=True,
    style={'backgroundColor': '#f8f9fa', 'padding': '15px'}, # Reduced padding slightly
    children=[
        # -- Header --
        dbc.Row(
            dbc.Col(
                html.H1("📊 Warehouse & Retail Sales Dashboard",
                        className="text-primary text-center mb-4",
                        style={'fontSize': '2rem', 'fontWeight': 'bold'}), # Slightly smaller H1
                width=12
            )
        ),

        # -- Filters Row 1 --
        dbc.Row([
            dbc.Col(dcc.Dropdown(
                id="year_filter",
                options=[{"label": "All Years", "value": "All"}] + [{"label": y, "value": y} for y in all_years],
                value="All", placeholder="Select Year", clearable=True
            ), width=12, md=6, lg=3, className="mb-2"), # Adjusted grid for better spacing
            dbc.Col(dcc.Dropdown(
                id="supplier_filter",
                options=[{"label": "All Suppliers", "value": "All"}] + [{"label": s, "value": s} for s in all_suppliers],
                value="All", placeholder="Select Supplier", clearable=True
            ), width=12, md=6, lg=3, className="mb-2"),
            dbc.Col(dcc.Dropdown(
                id="item_type_filter", # Options will be updated dynamically
                placeholder="Select Item Type", clearable=True,
                value="All"
            ), width=12, md=6, lg=3, className="mb-2"),
             dbc.Col(dcc.DatePickerRange(
                id='date_range',
                min_date_allowed=min_date,
                max_date_allowed=max_date,
                start_date=min_date,
                end_date=max_date,
                display_format='YYYY-MM-DD',
                className="w-100" # Ensure date picker takes full col width
            ), width=12, md=6, lg=3, className="mb-2")
        ], className="mb-3 align-items-center g-2"), # Added g-2 for gutters

        # -- Filters Row 2 (Buttons) --
        dbc.Row([
            dbc.Col(html.Button(html.I(className="bi bi-arrow-clockwise me-1"), id="reset_btn", n_clicks=0, className="btn btn-secondary btn-sm me-2", title="Reset Filters"), width="auto"), # Added icon and tooltip
            dbc.Col(html.Button(html.I(className="bi bi-download me-1"), id="download_btn", n_clicks=0, className="btn btn-primary btn-sm", title="Download Filtered Data"), width="auto"), # Added icon and tooltip
        ], className="mb-4 justify-content-end"), # Align buttons to the right

        # -- KPIs Row --
        dbc.Row(id="kpi_display", className="mb-4 g-3"), # Use g-3 for gutters/spacing

        # -- Charts Row 1 --
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader("📈 Sales Trend Over Time"),
                dbc.CardBody(dcc.Loading(dcc.Graph(id="sales_trend_chart", figure=create_empty_fig("Loading...")))) # Add initial figure
            ], className="shadow-sm h-100"), lg=6, className="mb-3"),
            dbc.Col(dbc.Card([
                dbc.CardHeader("🏭 Top 10 Suppliers by Retail Sales"),
                dbc.CardBody(dcc.Loading(dcc.Graph(id="supplier_comparison_chart", figure=create_empty_fig("Loading..."))))
            ], className="shadow-sm h-100"), lg=6, className="mb-3"),
        ], className="g-3"),

        # -- Charts Row 2 --
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader("🏷️ Retail Sales Distribution by Item Type"),
                dbc.CardBody(dcc.Loading(dcc.Graph(id="item_type_sales_chart", figure=create_empty_fig("Loading..."))))
            ], className="shadow-sm h-100"), lg=6, className="mb-3"),
            dbc.Col(dbc.Card([
                dbc.CardHeader("🌡️ Monthly Retail Sales Heatmap"),
                dbc.CardBody(dcc.Loading(dcc.Graph(id="heatmap_chart", figure=create_empty_fig("Loading..."))))
            ], className="shadow-sm h-100"), lg=6, className="mb-3"),
        ], className="g-3"),

         # -- Data Table Row --
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader("📋 Filtered Data View"),
                dbc.CardBody(dcc.Loading(html.Div(id='data_table_div', children="Loading data table..."))) # Wrap table in Div for loading
             ], className="shadow-sm"), width=12)
        ], className="mb-4"),

        # -- Hidden Divs --
        dcc.Download(id="download_data"),
    ]
)

# --- Callbacks ---

# Callback to dynamically update Item Type dropdown based on Supplier
@app.callback(
    Output("item_type_filter", "options"),
    Output("item_type_filter", "value"),
    Input("year_filter", "value"),
    Input("supplier_filter", "value"),
    Input("date_range", "start_date"),
    Input("date_range", "end_date"),
    Input("reset_btn", "n_clicks"), # Listen to reset button
    State("item_type_filter", "value"), # Keep current value if possible
    prevent_initial_call=True
)
def update_item_type_dropdown(selected_year, selected_supplier, start_date, end_date, reset_clicks, current_item_value):
    triggered_id = ctx.triggered_id
    if triggered_id == 'reset_btn':
         return [{"label": "All Item Types", "value": "All"}] + [{"label": i, "value": i} for i in all_item_types], "All"

    filtered_options_df = df.copy()

    if selected_year != "All" and selected_year is not None:
        try:
            filtered_options_df = filtered_options_df[filtered_options_df["YEAR"] == int(selected_year)]
        except ValueError:
             print(f"Warning: Could not convert selected year '{selected_year}' to int.")
             pass

    if start_date and end_date:
        try:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            if not pd.api.types.is_datetime64_any_dtype(filtered_options_df['DATE']): # Ensure date column is datetime
                filtered_options_df['DATE'] = pd.to_datetime(filtered_options_df['DATE'], errors='coerce')
            filtered_options_df = filtered_options_df.dropna(subset=['DATE']) # Drop rows if date conversion failed
            filtered_options_df = filtered_options_df[(filtered_options_df['DATE'] >= start_dt) & (filtered_options_df['DATE'] <= end_dt)]
        except ValueError:
             print(f"Warning: Invalid date format encountered in item type filter: {start_date} or {end_date}")
             pass

    if selected_supplier != "All" and selected_supplier is not None:
        filtered_options_df = filtered_options_df[filtered_options_df["SUPPLIER"] == selected_supplier]
        available_items = sorted(filtered_options_df['ITEM TYPE'].unique())
        options = [{"label": i, "value": i} for i in available_items]
        new_value = current_item_value if current_item_value in available_items else "All"
        # Handle case where supplier name might be long
        supplier_label = (selected_supplier[:20] + '...') if len(selected_supplier) > 23 else selected_supplier
        return [{"label": f"All Types for {supplier_label}", "value": "All"}] + options, new_value
    else:
        available_items = sorted(filtered_options_df['ITEM TYPE'].unique())
        options = [{"label": i, "value": i} for i in available_items]
        new_value = current_item_value if current_item_value in available_items else "All"
        return [{"label": "All Item Types", "value": "All"}] + options, new_value


# Main callback to update KPIs, Charts, Table, and handle Download
@app.callback(
    Output("kpi_display", "children"),
    Output("sales_trend_chart", "figure"),
    Output("supplier_comparison_chart", "figure"),
    Output("item_type_sales_chart", "figure"),
    Output("heatmap_chart", "figure"),
    Output("data_table_div", "children"), # Output for the data table
    Output("download_data", "data"),
    # Reset filters on reset button click
    Output("year_filter", "value", allow_duplicate=True),
    Output("supplier_filter", "value", allow_duplicate=True),
    Output("item_type_filter", "value", allow_duplicate=True),
    Output("date_range", "start_date", allow_duplicate=True),
    Output("date_range", "end_date", allow_duplicate=True),
    # Inputs
    Input("year_filter", "value"),
    Input("supplier_filter", "value"),
    Input("item_type_filter", "value"),
    Input("date_range", "start_date"),
    Input("date_range", "end_date"),
    Input("download_btn", "n_clicks"),
    Input("reset_btn", "n_clicks"),
    prevent_initial_call=True
)
def update_dashboard(selected_year, selected_supplier, selected_item_type,
                     start_date, end_date, download_clicks, reset_clicks):
    try:
        triggered_id = ctx.triggered_id
        print(f"Callback triggered by: {triggered_id}")

        # --- Handle Reset ---
        # Check n_clicks > 0 to avoid triggering on initial load if button exists
        if triggered_id == 'reset_btn' and reset_clicks > 0:
            print("Resetting filters...")
            empty_kpis = [
                dbc.Col(dbc.Card(dbc.CardBody("Select filters to view KPIs"), className="text-center border-0 bg-light shadow-sm h-100"), lg=4)
            ] * 3
            no_data_fig = create_empty_fig("Filters Reset. Select new options.")
            no_data_table = html.Div("Filters Reset.", style={'textAlign': 'center', 'padding': '20px'})

            return (
                empty_kpis,
                no_data_fig, no_data_fig, no_data_fig, no_data_fig,
                no_data_table,
                None, # No download
                "All", "All", "All", # Reset dropdowns
                min_date, max_date # Reset dates
            )

        # --- Filtering ---
        filtered_df = df.copy()
        print(f"Initial rows: {len(filtered_df)}")

        # Apply Year Filter
        if selected_year != "All" and selected_year is not None:
            try:
                filtered_df = filtered_df[filtered_df["YEAR"] == int(selected_year)]
                print(f"Rows after Year filter ({selected_year}): {len(filtered_df)}")
            except ValueError:
                print(f"Warning: Invalid year value for filtering: {selected_year}")
                pass

        # Apply Supplier Filter
        if selected_supplier != "All" and selected_supplier is not None:
            filtered_df = filtered_df[filtered_df["SUPPLIER"] == selected_supplier]
            print(f"Rows after Supplier filter ({selected_supplier}): {len(filtered_df)}")

        # Apply Item Type Filter
        if selected_item_type != "All" and selected_item_type is not None:
            filtered_df = filtered_df[filtered_df["ITEM TYPE"] == selected_item_type]
            print(f"Rows after Item Type filter ({selected_item_type}): {len(filtered_df)}")

        # Apply Date Range Filter
        if start_date and end_date:
            try:
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                # Ensure DATE column is datetime type before comparison
                if not pd.api.types.is_datetime64_any_dtype(filtered_df['DATE']):
                    filtered_df['DATE'] = pd.to_datetime(filtered_df['DATE'], errors='coerce')
                # Drop rows again if date conversion failed within this filter step
                filtered_df = filtered_df.dropna(subset=['DATE'])

                filtered_df = filtered_df[(filtered_df['DATE'] >= start_dt) & (filtered_df['DATE'] <= end_dt)]
                print(f"Rows after Date filter ({start_date} - {end_date}): {len(filtered_df)}")
            except Exception as e:
                print(f"Date filtering error: {e}")
                # Return error figure for charts if date filtering fails catastrophically?
                # For now, just prints error and continues with potentially unfiltered data by date.

        # --- Handle No Data ---
        if filtered_df.empty:
            print("No data matches the selected filters.")
            empty_message = "No data matches the selected filters"
            no_data_fig = create_empty_fig(empty_message)
            kpi_placeholders = [
                dbc.Col(dbc.Card(dbc.CardBody([
                     html.Div("-$--.--", className="fs-4 fw-bold text-muted"),
                     html.Div("No Data Available")
                ]), className="text-center border-0 bg-light shadow-sm h-100"), lg=4)
             ] * 3
            no_data_table = html.Div(empty_message, style={'textAlign': 'center', 'padding': '20px', 'color': '#6c757d'})

            return (
                kpi_placeholders,
                no_data_fig, no_data_fig, no_data_fig, no_data_fig,
                no_data_table,
                None, # No download
                dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
            )

        # --- Calculate KPIs ---
        total_retail_sales = filtered_df['RETAIL SALES'].sum()
        total_warehouse_sales = filtered_df['WAREHOUSE SALES'].sum()
        total_transfers = filtered_df['RETAIL TRANSFERS'].sum()

        kpi_elements = [
            dbc.Col(dbc.Card(dbc.CardBody([
                html.I(className="bi bi-cart-check-fill fs-2 text-success"),
                html.Div(f"${total_retail_sales:,.0f}", className="fs-4 fw-bold mt-2"), # Format as integer
                html.Div("Total Retail Sales")
            ]), className="text-center border-0 bg-white shadow-sm h-100"), lg=4),
             dbc.Col(dbc.Card(dbc.CardBody([
                html.I(className="bi bi-house-gear-fill fs-2 text-info"),
                html.Div(f"${total_warehouse_sales:,.0f}", className="fs-4 fw-bold mt-2"), # Format as integer
                html.Div("Total Warehouse Sales")
            ]), className="text-center border-0 bg-white shadow-sm h-100"), lg=4),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.I(className="bi bi-truck fs-2 text-warning"),
                 html.Div(f"${total_transfers:,.0f}", className="fs-4 fw-bold mt-2"), # Format as integer
                html.Div("Total Retail Transfers")
            ]), className="text-center border-0 bg-white shadow-sm h-100"), lg=4)
        ]

        # --- Generate Charts ---
        chart_margin = dict(t=30, b=30, l=40, r=20) # Consistent margins
        # Sales Trend
        try:
            if not pd.api.types.is_datetime64_any_dtype(filtered_df['DATE']):
                filtered_df['DATE'] = pd.to_datetime(filtered_df['DATE'], errors='coerce')
            trend_data = filtered_df.dropna(subset=['DATE']).groupby(pd.Grouper(key='DATE', freq='MS'))[["RETAIL SALES", "WAREHOUSE SALES"]].sum().reset_index()
            if not trend_data.empty:
                trend_fig = px.line(trend_data, x="DATE", y=["RETAIL SALES", "WAREHOUSE SALES"],
                                    labels={"value": "Total Sales ($)", "variable": "Channel", "DATE": "Month"}, markers=False)
                trend_fig.update_layout(legend_title_text='Sales Channel', margin=chart_margin, hovermode="x unified")
            else:
                 trend_fig = create_empty_fig("No trend data")
        except Exception as e:
            print(f"Error creating trend chart: {e}\n{traceback.format_exc()}")
            trend_fig = create_error_fig("Trend Chart Error")

        # Supplier Comparison (Top 10)
        try:
            supplier_data = filtered_df.groupby("SUPPLIER")['RETAIL SALES'].sum().nlargest(10).reset_index().sort_values(by='RETAIL SALES', ascending=True)
            if not supplier_data.empty:
                supplier_fig = px.bar(supplier_data, y="SUPPLIER", x="RETAIL SALES", orientation="h",
                                      labels={"RETAIL SALES": "Total Retail Sales ($)", "SUPPLIER": "Supplier"}, text='RETAIL SALES')
                supplier_fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside') # Format text
                supplier_fig.update_layout(yaxis={'categoryorder':'total ascending', 'tickfont': {'size': 10}}, margin=chart_margin)
            else:
                supplier_fig = create_empty_fig("No supplier data")
        except Exception as e:
            print(f"Error creating supplier chart: {e}\n{traceback.format_exc()}")
            supplier_fig = create_error_fig("Supplier Chart Error")


        # Item Type Distribution (Pie Chart)
        try:
            item_type_data = filtered_df.groupby("ITEM TYPE")['RETAIL SALES'].sum().reset_index()
            if not item_type_data.empty and item_type_data['RETAIL SALES'].sum() > 0:
                threshold = 0.02
                total_sales_all_items = item_type_data['RETAIL SALES'].sum()
                item_type_data['Percentage'] = item_type_data['RETAIL SALES'] / total_sales_all_items
                small_slices = item_type_data[item_type_data['Percentage'] < threshold]
                main_slices = item_type_data[item_type_data['Percentage'] >= threshold]
                if not small_slices.empty:
                     other_sum = small_slices['RETAIL SALES'].sum()
                     other_row = pd.DataFrame([{'ITEM TYPE': f'OTHER ({len(small_slices)} types)', 'RETAIL SALES': other_sum}])
                     item_type_data_agg = pd.concat([main_slices[['ITEM TYPE', 'RETAIL SALES']], other_row], ignore_index=True)
                else:
                     item_type_data_agg = main_slices[['ITEM TYPE', 'RETAIL SALES']]

                item_type_fig = px.pie(item_type_data_agg, values="RETAIL SALES", names="ITEM TYPE", hole=0.4)
                item_type_fig.update_traces(textposition='inside', textinfo='percent', hoverinfo='label+value+percent') # Show % inside, detail on hover
                item_type_fig.update_layout(showlegend=True, legend_title_text='Item Type', legend={'font': {'size': 10}}, margin=chart_margin)
            else:
                 item_type_fig = create_empty_fig("No item data")
        except Exception as e:
             print(f"Error creating item type chart: {e}\n{traceback.format_exc()}")
             item_type_fig = create_error_fig("Item Type Chart Error")

        # Sales Heatmap
        try:
            if 'MONTH' in filtered_df.columns and 'YEAR' in filtered_df.columns:
                heatmap_data = filtered_df.pivot_table(index="MONTH", columns="YEAR", values="RETAIL SALES", aggfunc="sum", fill_value=0)
                heatmap_data = heatmap_data.reindex(range(1, 13), fill_value=0)
                month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                             7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
                heatmap_data.index = heatmap_data.index.map(month_map)
                # Ensure columns are sorted numerically if they are years
                heatmap_data = heatmap_data.reindex(columns=sorted([col for col in heatmap_data.columns if isinstance(col, (int, float))]))

                if not heatmap_data.empty:
                    heatmap_fig = px.imshow(heatmap_data,
                                            labels=dict(x="Year", y="Month", color="Sales ($)"),
                                            aspect="auto", text_auto='.0f', # Format as int
                                            color_continuous_scale=px.colors.sequential.Blues)
                    heatmap_fig.update_traces(hovertemplate="Month: %{y}<br>Year: %{x}<br>Sales: $%{z:,.0f}<extra></extra>")
                    heatmap_fig.update_layout(margin=chart_margin)
                else:
                     heatmap_fig = create_empty_fig("No heatmap data")
            else:
                heatmap_fig = create_empty_fig("MONTH or YEAR column missing")
        except Exception as e:
             print(f"Error creating heatmap: {e}\n{traceback.format_exc()}")
             heatmap_fig = create_error_fig("Heatmap Error")

         # --- Create Data Table ---
        try:
            cols_to_display = [
                 'DATE', 'YEAR', 'MONTH', 'SUPPLIER', 'ITEM TYPE',
                 'RETAIL SALES', 'WAREHOUSE SALES', 'RETAIL TRANSFERS'
             ]
            cols_present = [col for col in cols_to_display if col in filtered_df.columns] # *** THIS IS THE CORRECTED INDENTATION ***
            table_df = filtered_df[cols_present].copy()

            rename_map = {
                 'DATE': 'Date', 'YEAR': 'Year', 'MONTH': 'Month',
                 'SUPPLIER': 'Supplier', 'ITEM TYPE': 'Item Type',
                 'RETAIL SALES': 'Retail Sales ($)',
                 'WAREHOUSE SALES': 'Warehouse Sales ($)',
                 'RETAIL TRANSFERS': 'Retail Transfers ($)'
             }
            cols_to_rename = {k: v for k, v in rename_map.items() if k in table_df.columns}
            table_df = table_df.rename(columns=cols_to_rename)

            if 'Date' in table_df.columns and pd.api.types.is_datetime64_any_dtype(table_df['Date']):
                 table_df['Date'] = table_df['Date'].dt.strftime('%Y-%m-%d')

            # Define formatting for numeric columns
            num_format = dash_table.Format.Format(precision=0, scheme=dash_table.Format.Scheme.fixed).group(True).symbol(dash_table.Format.Symbol.yes).symbol_prefix('$')
            table_columns = []
            for col in table_df.columns:
                col_def = {"name": col, "id": col}
                if '($)' in col: # Apply formatting to columns ending in ($)
                    col_def["type"] = "numeric"
                    col_def["format"] = num_format
                table_columns.append(col_def)


            data_table_component = dash_table.DataTable(
                 id='filtered_table',
                 columns=table_columns,
                 data=table_df.to_dict('records'),
                 page_size=10,
                 style_table={'overflowX': 'auto', 'minWidth': '100%'},
                 style_cell={
                     'textAlign': 'left',
                     'padding': '8px',
                     'fontSize': '13px',
                     'fontFamily': 'Arial, sans-serif',
                     'whiteSpace': 'normal',
                     'height': 'auto',
                     'border': '1px solid #eee'
                 },
                 style_header={
                     'backgroundColor': 'rgb(220, 220, 220)',
                     'fontWeight': 'bold',
                     'border': '1px solid #ccc'
                 },
                 style_data_conditional=[
                    {'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'}
                 ],
                 filter_action="native",
                 sort_action="native",
                 sort_mode="multi",
                 export_format='csv',
                 export_headers='display',
             )
        except Exception as e:
             print(f"Error creating data table: {e}\n{traceback.format_exc()}")
             data_table_component = html.Div(f"Error generating data table: {e}", style={'color': 'red', 'padding': '20px'})

        # --- Handle Download ---
        download_content = None
        if triggered_id == "download_btn":
            print("Download triggered...")
            try:
                # Prepare df for download
                download_df = filtered_df[cols_present].copy()
                download_df = download_df.rename(columns=cols_to_rename)
                if 'Date' in download_df.columns and pd.api.types.is_datetime64_any_dtype(download_df['Date']):
                     download_df['Date'] = download_df['Date'].dt.strftime('%Y-%m-%d')
                # Generate filename based on filters
                file_suffix = ""
                if selected_year != "All": file_suffix += f"_Y{selected_year}"
                if selected_supplier != "All": file_suffix += f"_S{selected_supplier[:10].replace(' ','')}" # Shorten supplier name
                if selected_item_type != "All": file_suffix += f"_I{selected_item_type[:10].replace(' ','')}" # Shorten item type
                filename = f"filtered_sales_data{file_suffix}.csv"

                download_content = dcc.send_data_frame(download_df.to_csv, filename, index=False, encoding='utf-8')
                print(f"Prepared download: {filename}")
            except Exception as e:
                print(f"Error preparing download data: {e}\n{traceback.format_exc()}")
                # Optionally notify the user download failed

        # --- Return Outputs ---
        return (
            kpi_elements,
            trend_fig,
            supplier_fig,
            item_type_fig,
            heatmap_fig,
            data_table_component,
            download_content,
            dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
        )

    except Exception as e:
        error_msg = f"Dashboard Update Error: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        error_fig = create_error_fig(f"Details: {str(e)}")
        error_kpis = [
             dbc.Col(dbc.Card(dbc.CardBody(f"Error generating KPI"), className="text-center bg-danger text-white"), lg=4)
         ] * 3
        error_table = html.Div(f"Error displaying table: {str(e)}", style={'color': 'red', 'padding': '20px'})
        return (
            error_kpis,
            error_fig, error_fig, error_fig, error_fig,
            error_table,
            None, # No download
            dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
        )


# --- Run the App ---
if __name__ == "__main__":
    print("Starting Dash server...")
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css">
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''
    app.run(debug=True)