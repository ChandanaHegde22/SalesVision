import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np

from src.forecast import prepare_daily_sales, make_future_forecast
from src.dax_engine import (
    eval_calculated_column,
    eval_measure,
    eval_dax_table,
    divide,
    dax_if
)
from src.theme_manager import inject_theme_css, apply_plotly_theme, THEMES

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_DIR, "data", "sales_data.csv")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "sales_forecast_model.pkl")
METRICS_PATH = os.path.join(ROOT_DIR, "models", "metrics.txt")

# Set up page configurations
st.set_page_config(page_title="SalesVision BI Workspace", page_icon="📊", layout="wide")

# Check if model exists
model_available = os.path.exists(MODEL_PATH)
if model_available:
    @st.cache_resource
    def load_model():
        return joblib.load(MODEL_PATH)
    model = load_model()
else:
    model = None

# Initialize state
if "theme_name" not in st.session_state:
    st.session_state["theme_name"] = "Cyberpunk Dark"

if "custom_columns" not in st.session_state:
    st.session_state["custom_columns"] = []

if "custom_measures" not in st.session_state:
    st.session_state["custom_measures"] = []

if "custom_tables" not in st.session_state:
    st.session_state["custom_tables"] = []

if "custom_tables_data" not in st.session_state:
    st.session_state["custom_tables_data"] = {}

if "pinned_visuals" not in st.session_state:
    # Initialize with default pinned charts for the standard dataset
    st.session_state["pinned_visuals"] = [
        {
            "id": "viz_1",
            "title": "Daily Revenue Trend",
            "type": "Line Chart",
            "x": "date",
            "y": "revenue",
            "agg": "Sum",
            "color": "None",
            "show_grid": True
        },
        {
            "id": "viz_2",
            "title": "Revenue by Category",
            "type": "Vertical Bar Chart",
            "x": "category",
            "y": "revenue",
            "agg": "Sum",
            "color": "category",
            "show_grid": True
        },
        {
            "id": "viz_3",
            "title": "Revenue Share by Region",
            "type": "Pie Chart",
            "x": "region",
            "y": "revenue",
            "agg": "Sum",
            "color": "None",
            "show_grid": True
        }
    ]

# Inject the visual styling CSS
inject_theme_css(st.session_state["theme_name"])

# Load base dataset
@st.cache_data
def load_default_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        df["date"] = pd.to_datetime(df["date"])
        return df
    else:
        # Create a tiny fallback DataFrame in case generate_data hasn't run
        st.warning("Default sales dataset not found. Generating simple in-memory dataset.")
        dates = pd.date_range(start="2025-01-01", end="2025-01-10", freq="D")
        data = {
            "date": pd.to_datetime(dates),
            "category": ["Electronics", "Fashion"] * 5,
            "region": ["North", "South"] * 5,
            "quantity": [2, 5, 3, 4, 1, 8, 2, 3, 5, 2],
            "price": [1000.0, 500.0] * 5,
            "discount": [10, 0, 5, 0, 10, 20, 0, 5, 0, 0],
            "revenue": [1800.0, 2500.0, 2850.0, 2000.0, 900.0, 3200.0, 2000.0, 1425.0, 5000.0, 1000.0],
            "profit": [300.0, 400.0, 500.0, 300.0, 100.0, 600.0, 300.0, 200.0, 800.0, 150.0],
            "promotion": [1, 0] * 5,
            "holiday": [0] * 10
        }
        return pd.DataFrame(data)

if "base_df" not in st.session_state:
    st.session_state["base_df"] = load_default_data()
    st.session_state["active_df_name"] = "Default Sales Dataset"

# Rebuilder for the active DataFrame applying custom columns
def rebuild_active_df():
    df = st.session_state["base_df"].copy()
    errors = []
    for col in st.session_state["custom_columns"]:
        try:
            # Handle dates specifically if the column is date
            res = eval_calculated_column(df, col["formula"])
            # If the calculated column yields date strings, parse them
            if col["name"].lower() == "date" or "date" in col["name"].lower():
                try:
                    res = pd.to_datetime(res)
                except:
                    pass
            df[col["name"]] = res
        except Exception as e:
            errors.append(f"Column '{col['name']}': {str(e)}")
    st.session_state["active_df"] = df
    return errors

# Build the initial active DataFrame
rebuild_errors = rebuild_active_df()
active_df = st.session_state["active_df"]

# Helper to generate plotly charts from definitions
def generate_plotly_fig(df, viz_def, theme_name):
    chart_type = viz_def["type"]
    x_col = viz_def["x"]
    y_col = viz_def["y"]
    agg = viz_def.get("agg", "Sum")
    color_col = viz_def.get("color", "None")
    if color_col == "None":
        color_col = None
    title = viz_def.get("title", f"{y_col} by {x_col}")
    show_grid = viz_def.get("show_grid", True)
    
    plot_df = df.copy()
    
    # Pre-process columns
    if x_col in plot_df.columns and pd.api.types.is_datetime64_any_dtype(plot_df[x_col]):
        plot_df[x_col] = pd.to_datetime(plot_df[x_col])
        
    # Perform aggregation if needed
    if y_col in plot_df.columns and agg != "None":
        groupby_cols = [x_col]
        if color_col and color_col in plot_df.columns:
            groupby_cols.append(color_col)
            
        if agg == "Sum":
            plot_df = plot_df.groupby(groupby_cols, as_index=False)[y_col].sum()
        elif agg == "Average":
            plot_df = plot_df.groupby(groupby_cols, as_index=False)[y_col].mean()
        elif agg == "Min":
            plot_df = plot_df.groupby(groupby_cols, as_index=False)[y_col].min()
        elif agg == "Max":
            plot_df = plot_df.groupby(groupby_cols, as_index=False)[y_col].max()
        elif agg == "Count":
            plot_df = plot_df.groupby(groupby_cols, as_index=False)[y_col].count()
            
    # Sort bar charts
    if chart_type in ("Vertical Bar Chart", "Horizontal Bar Chart") and y_col in plot_df.columns:
        plot_df = plot_df.sort_values(y_col, ascending=False)
        
    # Create figure
    if chart_type == "Vertical Bar Chart":
        fig = px.bar(plot_df, x=x_col, y=y_col, color=color_col, title=title, barmode="group")
    elif chart_type == "Horizontal Bar Chart":
        fig = px.bar(plot_df, x=y_col, y=x_col, color=color_col, title=title, orientation="h", barmode="group")
    elif chart_type == "Line Chart":
        fig = px.line(plot_df, x=x_col, y=y_col, color=color_col, title=title, markers=True)
    elif chart_type == "Area Chart":
        fig = px.area(plot_df, x=x_col, y=y_col, color=color_col, title=title)
    elif chart_type == "Pie Chart":
        fig = px.pie(plot_df, names=x_col, values=y_col, title=title)
    elif chart_type == "Donut Chart":
        fig = px.pie(plot_df, names=x_col, values=y_col, hole=0.4, title=title)
    elif chart_type == "Scatter Plot":
        fig = px.scatter(plot_df, x=x_col, y=y_col, color=color_col, title=title)
    elif chart_type == "Treemap":
        fig = px.treemap(plot_df, path=[x_col], values=y_col, title=title)
    elif chart_type == "Histogram":
        fig = px.histogram(plot_df, x=x_col, y=y_col, title=title)
    elif chart_type == "Box Plot":
        fig = px.box(plot_df, x=x_col, y=y_col, color=color_col, title=title)
    else:
        fig = px.scatter(title="Unsupported visual type")
        
    fig = apply_plotly_theme(fig, theme_name)
    if not show_grid:
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        
    return fig

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.markdown("# 📈 SalesVision Workspace")
    st.markdown("---")
    
    # Dataset Selector/Uploader Section
    st.markdown("### 📂 Dataset Connection")
    st.info(f"**Active Table**: {st.session_state['active_df_name']}\n\n**Rows**: {len(active_df):,} | **Columns**: {len(active_df.columns)}")
    
    uploaded_file = st.file_uploader("Upload custom dataset (CSV)", type=["csv"])
    if uploaded_file is not None:
        try:
            # Reset states for custom data to prevent column mismatch crashes
            df_new = pd.read_csv(uploaded_file)
            
            # Auto convert date-like columns to datetimes
            for col in df_new.columns:
                if "date" in col.lower():
                    try:
                        df_new[col] = pd.to_datetime(df_new[col])
                    except:
                        pass
                        
            st.session_state["base_df"] = df_new
            st.session_state["active_df_name"] = uploaded_file.name
            st.session_state["custom_columns"] = []
            st.session_state["custom_measures"] = []
            st.session_state["custom_tables"] = []
            st.session_state["pinned_visuals"] = [] # Clear default charts as they don't apply to new schema
            st.session_state["custom_tables_data"] = {}
            st.rerun()
        except Exception as e:
            st.error(f"Error parsing CSV: {str(e)}")

    if st.button("Reset to Default Sales Data", use_container_width=True):
        if os.path.exists(DATA_PATH):
            del st.session_state["base_df"]
            if "active_df" in st.session_state:
                del st.session_state["active_df"]
            st.session_state["active_df_name"] = "Default Sales Dataset"
            st.session_state["custom_columns"] = []
            st.session_state["custom_measures"] = []
            st.session_state["custom_tables"] = []
            st.session_state["custom_tables_data"] = {}
            # Reset defaults
            st.session_state["pinned_visuals"] = [
                {
                    "id": "viz_1",
                    "title": "Daily Revenue Trend",
                    "type": "Line Chart",
                    "x": "date",
                    "y": "revenue",
                    "agg": "Sum",
                    "color": "None",
                    "show_grid": True
                },
                {
                    "id": "viz_2",
                    "title": "Revenue by Category",
                    "type": "Vertical Bar Chart",
                    "x": "category",
                    "y": "revenue",
                    "agg": "Sum",
                    "color": "category",
                    "show_grid": True
                },
                {
                    "id": "viz_3",
                    "title": "Revenue Share by Region",
                    "type": "Pie Chart",
                    "x": "region",
                    "y": "revenue",
                    "agg": "Sum",
                    "color": "None",
                    "show_grid": True
                }
            ]
            st.rerun()

    st.markdown("---")
    
    # Theme configuration
    st.markdown("### 🎨 Visual Theme")
    selected_theme = st.selectbox(
        "Theme Palette", 
        options=list(THEMES.keys()), 
        index=list(THEMES.keys()).index(st.session_state["theme_name"])
    )
    if selected_theme != st.session_state["theme_name"]:
        st.session_state["theme_name"] = selected_theme
        st.rerun()
        
    st.markdown("---")
    
    # Dynamic Filtering
    filtered_df = active_df.copy()
    cat_cols = []
    
    # Priority cols for filtering
    for c in active_df.columns:
        if c.lower() in ("region", "category", "store", "promotion", "holiday"):
            cat_cols.append(c)
            
    # Auto-detect other categoricals
    for c in active_df.columns:
        if c not in cat_cols:
            if active_df[c].dtype == "object" or isinstance(active_df[c].dtype, pd.CategoricalDtype):
                if 1 < active_df[c].nunique() <= 15:
                    cat_cols.append(c)
                    
    cat_cols = cat_cols[:4]  # Limit filters to keep sidebar clean
    
    filters = {}
    if cat_cols:
        st.markdown("### 🔍 Report Canvas Filters")
        for col in cat_cols:
            unique_vals = sorted(active_df[col].dropna().unique().tolist())
            selected_vals = st.multiselect(
                f"Filter by {col.replace('_', ' ').title()}",
                options=unique_vals,
                default=unique_vals,
                key=f"filter_{col}"
            )
            filters[col] = selected_vals
            
        # Apply filters
        for col, vals in filters.items():
            if vals:
                filtered_df = filtered_df[filtered_df[col].isin(vals)]

# Notify about calculated column errors
if rebuild_errors:
    for err in rebuild_errors:
        st.error(err)

# -------------------- MAIN PAGE --------------------
st.title("📊 SalesVision BI & Forecasting Canvas")
st.write("A fully customizable, Power BI-inspired analytics workspace. Upload datasets, run DAX-like calculations, and design charts.")

# Main navigation tabs
tab_dashboard, tab_dax, tab_chart_builder, tab_forecasting = st.tabs([
    "🖥️ Report Dashboard", 
    "⚡ DAX & Calculations", 
    "🎨 Visualizations Creator", 
    "📈 Time Series Forecasting"
])

# -------------------- TAB 1: REPORT DASHBOARD --------------------
with tab_dashboard:
    # 1. KPI cards rendering
    # We want to show cards for standard KPI measures: Revenue, Profit, Quantity, Discount
    # Or automatically display any metrics present in the active dataset
    kpi_cols = st.columns(4)
    
    # Figure out what numeric columns exist in filtered dataset
    num_cols = [c for c in filtered_df.columns if pd.api.types.is_numeric_dtype(filtered_df[c]) and c.lower() not in ("holiday", "promotion")]
    
    # Priority numeric columns to display in KPIs
    priority_num = ["revenue", "profit", "quantity", "discount", "price", "sales", "amount", "cost"]
    kpi_candidates = []
    
    for p in priority_num:
        for c in num_cols:
            if p in c.lower() and c not in kpi_candidates:
                kpi_candidates.append(c)
                
    for c in num_cols:
        if c not in kpi_candidates:
            kpi_candidates.append(c)
            
    kpi_candidates = kpi_candidates[:4]
    
    # Render KPIs
    for idx, col_name in enumerate(kpi_candidates):
        with kpi_cols[idx % 4]:
            # Check if it is percentage-like (discount) or currency/integer
            if "discount" in col_name.lower() or "rate" in col_name.lower():
                val = filtered_df[col_name].mean()
                st.metric(f"Avg {col_name.replace('_', ' ').title()}", f"{val:.1f}%")
            elif "price" in col_name.lower() or "cost" in col_name.lower():
                val = filtered_df[col_name].mean()
                st.metric(f"Avg {col_name.replace('_', ' ').title()}", f"₹{val:,.2f}")
            elif "profit" in col_name.lower() or "revenue" in col_name.lower() or "sales" in col_name.lower() or "amount" in col_name.lower():
                val = filtered_df[col_name].sum()
                st.metric(f"Total {col_name.replace('_', ' ').title()}", f"₹{val:,.0f}")
            else:
                val = filtered_df[col_name].sum()
                st.metric(f"Total {col_name.replace('_', ' ').title()}", f"{val:,.0f}")
                
    # Also evaluate user measures dynamically on filtered dataframe and display them in KPI pane if they exist
    if st.session_state["custom_measures"]:
        st.markdown("### 🏷️ Active Custom Measures")
        measure_cols = st.columns(min(len(st.session_state["custom_measures"]), 4))
        for idx, m in enumerate(st.session_state["custom_measures"]):
            with measure_cols[idx % 4]:
                try:
                    m_val = eval_measure(filtered_df, m["formula"])
                    if isinstance(m_val, (int, float, np.integer, np.floating)):
                        st.metric(m["name"], f"{m_val:,.2f}")
                    else:
                        st.metric(m["name"], str(m_val))
                except Exception as e:
                    st.metric(m["name"], "Error", help=str(e))
                    
    st.markdown("---")
    
    # 2. Render pinned visuals
    pinned_viz = st.session_state["pinned_visuals"]
    if not pinned_viz:
        st.info("No charts pinned to the dashboard canvas yet. Use the **Visualizations Creator** tab to build and pin charts.")
    else:
        st.markdown("### 📊 Custom Dashboard Layout")
        
        # Render in a 2-column grid
        grid_cols = st.columns(2)
        
        # Create a copy of list to avoid issues when deleting during iteration
        for idx, viz in enumerate(pinned_viz.copy()):
            with grid_cols[idx % 2]:
                with st.container(border=True):
                    # Title & Delete controls in a sub-column
                    title_col, btn_col = st.columns([5, 1])
                    title_col.markdown(f"#### {viz['title']}")
                    if btn_col.button("🗑️", key=f"delete_viz_{viz['id']}", help="Remove visual from dashboard"):
                        st.session_state["pinned_visuals"] = [v for v in st.session_state["pinned_visuals"] if v["id"] != viz["id"]]
                        st.rerun()
                        
                    # Generate and display the plot
                    try:
                        fig = generate_plotly_fig(filtered_df, viz, st.session_state["theme_name"])
                        st.plotly_chart(fig, use_container_width=True, key=f"chart_render_{viz['id']}")
                    except Exception as e:
                        st.error(f"Error rendering chart: {str(e)}")

# -------------------- TAB 2: DAX & CALCULATIONS --------------------
with tab_dax:
    st.header("⚡ DAX Calculated Columns, Measures & Tables")
    st.write("Write formulas in DAX-like formats referencing columns as `[ColumnName]`.")
    
    # Sub-tabs for Columns, Measures, and Tables
    dax_sub_cols, dax_sub_meas, dax_sub_tbl = st.tabs(["🆕 Calculated Columns", "🔢 Aggregated Measures", "📂 Calculated Tables"])
    
    # Calculated Columns Sub-tab
    with dax_sub_cols:
        col_form, col_list = st.columns([1, 1])
        
        with col_form:
            st.markdown("### Create Calculated Column")
            st.write("Adds a new column calculated row-by-row on the dataset.")
            st.code("[new_column] = [quantity] * [price]\n[margin] = IF([revenue] > 0, [profit] / [revenue], 0)")
            
            with st.form("calc_col_form"):
                new_col_name = st.text_input("New Column Name (e.g. total_cost)").strip()
                col_formula = st.text_input("Formula (e.g. [price] * [quantity] * (1 - [discount]/100))").strip()
                submit_col = st.form_submit_button("Add Calculated Column", type="primary")
                
                if submit_col:
                    if not new_col_name or not col_formula:
                        st.error("Please fill in both name and formula.")
                    elif new_col_name in st.session_state["base_df"].columns:
                        st.error(f"Column '{new_col_name}' already exists in the base dataset.")
                    elif any(c["name"] == new_col_name for c in st.session_state["custom_columns"]):
                        st.error(f"Calculated column '{new_col_name}' already exists.")
                    else:
                        try:
                            # Test evaluate
                            test_res = eval_calculated_column(active_df, col_formula)
                            
                            # Add to state and rebuild
                            st.session_state["custom_columns"].append({
                                "name": new_col_name,
                                "formula": col_formula
                            })
                            st.success(f"Column '{new_col_name}' added successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Calculation Error: {str(e)}")
                            
        with col_list:
            st.markdown("### Custom Columns List")
            if not st.session_state["custom_columns"]:
                st.info("No custom calculated columns added yet.")
            else:
                for idx, col in enumerate(st.session_state["custom_columns"]):
                    col_row_1, col_row_2 = st.columns([5, 1])
                    col_row_1.markdown(f"**`{col['name']}`** = `{col['formula']}`")
                    if col_row_2.button("Delete", key=f"delete_col_{idx}"):
                        st.session_state["custom_columns"].pop(idx)
                        st.rerun()
                        
        st.markdown("---")
        st.markdown("### 🔍 Active Dataset Preview")
        st.dataframe(active_df.head(10), use_container_width=True)
        
    # Measures Sub-tab
    with dax_sub_meas:
        meas_form, meas_list = st.columns([1, 1])
        
        with meas_form:
            st.markdown("### Create Aggregated Measure")
            st.write("Measures aggregate values across rows to return a single summary value.")
            st.code("Total Sales = SUM([revenue])\nProfit Margin = DIVIDE(SUM([profit]), SUM([revenue]))\nUnique Products = DISTINCTCOUNT([product_name])")
            
            with st.form("calc_meas_form"):
                new_meas_name = st.text_input("Measure Name (e.g. Sales Margin)").strip()
                meas_formula = st.text_input("Formula (e.g. DIVIDE(SUM([profit]), SUM([revenue])))").strip()
                submit_meas = st.form_submit_button("Add Aggregated Measure", type="primary")
                
                if submit_meas:
                    if not new_meas_name or not meas_formula:
                        st.error("Please fill in both name and formula.")
                    elif any(m["name"] == new_meas_name for m in st.session_state["custom_measures"]):
                        st.error(f"Measure '{new_meas_name}' already exists.")
                    else:
                        try:
                            # Test evaluate
                            test_val = eval_measure(active_df, meas_formula)
                            
                            st.session_state["custom_measures"].append({
                                "name": new_meas_name,
                                "formula": meas_formula
                            })
                            st.success(f"Measure '{new_meas_name}' added! Current value: {test_val}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Calculation Error: {str(e)}")
                            
        with meas_list:
            st.markdown("### Custom Measures List")
            if not st.session_state["custom_measures"]:
                st.info("No custom measures added yet.")
            else:
                for idx, m in enumerate(st.session_state["custom_measures"]):
                    m_row_1, m_row_2 = st.columns([5, 1])
                    try:
                        # Show formula and current value
                        curr_val = eval_measure(filtered_df, m["formula"])
                        val_str = f"{curr_val:,.2f}" if isinstance(curr_val, (int, float, np.integer, np.floating)) else str(curr_val)
                        m_row_1.markdown(f"**`{m['name']}`** = `{m['formula']}`  \n↳ *Current value (filtered):* **{val_str}**")
                    except Exception as err:
                        m_row_1.markdown(f"**`{m['name']}`** = `{m['formula']}`  \n↳ *Current value:* **Error: {str(err)}**")
                        
                    if m_row_2.button("Delete", key=f"delete_meas_{idx}"):
                        st.session_state["custom_measures"].pop(idx)
                        st.rerun()
                        
    # Calculated Tables Sub-tab
    with dax_sub_tbl:
        tbl_form, tbl_view = st.columns([1, 2])
        
        with tbl_form:
            st.markdown("### Create Calculated Table")
            st.write("Generates a completely new table from the active dataset using DAX group/filter logic.")
            st.markdown("**Supported Functions**:")
            st.code("SUMMARIZE(df, [category], [region], \"Sales\", SUM([revenue]))\nFILTER(df, [revenue] > 500)")
            
            with st.form("calc_tbl_form"):
                new_tbl_name = st.text_input("New Table Name (e.g. Category_Summary)").strip()
                tbl_formula = st.text_input("Table Expression (e.g. SUMMARIZE(df, [category], \"Total Revenue\", SUM([revenue]))").strip()
                submit_tbl = st.form_submit_button("Build Calculated Table", type="primary")
                
                if submit_tbl:
                    if not new_tbl_name or not tbl_formula:
                        st.error("Please fill in both name and formula.")
                    else:
                        try:
                            # Evaluate Table
                            tbl_df = eval_dax_table(active_df, tbl_formula)
                            
                            # Save in states
                            st.session_state["custom_tables"].append({
                                "name": new_tbl_name,
                                "formula": tbl_formula
                            })
                            st.session_state["custom_tables_data"][new_tbl_name] = tbl_df
                            st.success(f"Calculated table '{new_tbl_name}' created successfully with {len(tbl_df)} rows!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Table Creation Error: {str(e)}")
                            
        with tbl_view:
            st.markdown("### Active Calculated Tables")
            if not st.session_state["custom_tables"]:
                st.info("No custom calculated tables built yet.")
            else:
                table_names = [t["name"] for t in st.session_state["custom_tables"]]
                selected_tbl_name = st.selectbox("View Table", options=table_names)
                
                # Retrieve details
                tbl_def = next(t for t in st.session_state["custom_tables"] if t["name"] == selected_tbl_name)
                
                # Check if data exists in cache, if not rebuild it
                if selected_tbl_name not in st.session_state["custom_tables_data"]:
                    try:
                        st.session_state["custom_tables_data"][selected_tbl_name] = eval_dax_table(active_df, tbl_def["formula"])
                    except Exception as e:
                        st.error(f"Error rebuilding table: {str(e)}")
                        
                if selected_tbl_name in st.session_state["custom_tables_data"]:
                    show_tbl_df = st.session_state["custom_tables_data"][selected_tbl_name]
                    
                    st.markdown(f"**Expression**: `{tbl_def['formula']}`")
                    st.write(f"Row count: {len(show_tbl_df)}")
                    
                    st.dataframe(show_tbl_df, use_container_width=True)
                    
                    # Download CSV option
                    csv_data = show_tbl_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Table as CSV",
                        data=csv_data,
                        file_name=f"{selected_tbl_name}.csv",
                        mime="text/csv"
                    )
                    
                    if st.button("Delete Table", key=f"delete_tbl_{selected_tbl_name}"):
                        st.session_state["custom_tables"] = [t for t in st.session_state["custom_tables"] if t["name"] != selected_tbl_name]
                        if selected_tbl_name in st.session_state["custom_tables_data"]:
                            del st.session_state["custom_tables_data"][selected_tbl_name]
                        st.rerun()

# -------------------- TAB 3: VISUALIZATIONS CREATOR --------------------
with tab_chart_builder:
    st.header("🎨 Interactive Visualizations Creator")
    st.write("Select chart variables, preview results, and pin them directly to the main Report Canvas.")
    
    param_col, preview_col = st.columns([1, 2])
    
    with param_col:
        st.markdown("### Chart Variables")
        viz_type = st.selectbox(
            "Visualization Type",
            options=[
                "Vertical Bar Chart",
                "Horizontal Bar Chart",
                "Line Chart",
                "Area Chart",
                "Pie Chart",
                "Donut Chart",
                "Scatter Plot",
                "Treemap",
                "Histogram",
                "Box Plot"
            ]
        )
        
        # Field choices
        all_cols = sorted(active_df.columns.tolist())
        num_fields = sorted([c for c in active_df.columns if pd.api.types.is_numeric_dtype(active_df[c])])
        
        # Determine logical defaults
        def_x = "date" if "date" in all_cols else all_cols[0]
        def_y = "revenue" if "revenue" in num_fields else (num_fields[0] if num_fields else all_cols[0])
        
        x_field = st.selectbox("Axis/Dimension (X-Axis / Legend Name)", options=all_cols, index=all_cols.index(def_x) if def_x in all_cols else 0)
        y_field = st.selectbox("Value Column (Y-Axis / Segment Value)", options=num_fields if num_fields else all_cols, index=num_fields.index(def_y) if def_y in num_fields else 0)
        
        agg_opt = st.selectbox(
            "Aggregation Mode",
            options=["Sum", "Average", "Min", "Max", "Count", "None"],
            index=0
        )
        
        color_choices = ["None"] + sorted([c for c in all_cols if c != x_field])
        color_field = st.selectbox(
            "Group By / Color (Legend)",
            options=color_choices,
            index=0
        )
        
        with st.expander("Customize Visual Style"):
            chart_title = st.text_input("Chart Title", value=f"{agg_opt} of {y_field} by {x_field}")
            show_gridlines = st.checkbox("Show Gridlines", value=True)
            
        # Visual Definition Dictionary
        viz_definition = {
            "id": f"custom_viz_{len(st.session_state['pinned_visuals']) + 1}_{pd.Timestamp.now().microsecond}",
            "title": chart_title,
            "type": viz_type,
            "x": x_field,
            "y": y_field,
            "agg": agg_opt,
            "color": color_field,
            "show_grid": show_gridlines
        }
        
        pin_btn = st.button("Pin Visual to Dashboard", type="primary", use_container_width=True)
        if pin_btn:
            st.session_state["pinned_visuals"].append(viz_definition)
            st.success("Visual pinned successfully to Tab 1: **Report Dashboard**!")
            
    with preview_col:
        st.markdown("### Chart Preview")
        try:
            fig = generate_plotly_fig(filtered_df, viz_definition, st.session_state["theme_name"])
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering preview chart: {str(e)}")

# -------------------- TAB 4: TIME SERIES FORECASTING --------------------
with tab_forecasting:
    st.header("📈 Machine Learning Sales Forecasting")
    
    # Check if forecasting schema is compatible
    required_cols = {"date", "revenue", "quantity", "profit", "promotion", "holiday"}
    active_cols_lower = {col.lower() for col in active_df.columns}
    has_forecasting_schema = required_cols.issubset(active_cols_lower)
    
    if not has_forecasting_schema:
        st.warning("⚠️ Machine Learning Forecasting is disabled for this dataset.")
        st.write("The forecasting model is trained specifically on the default Sales schema. To enable forecasting, please ensure your uploaded CSV contains these exact columns:")
        st.code(", ".join(required_cols))
        st.write("Alternatively, use the **Visualizations Creator** to plot trendlines for your custom data.")
    elif not model_available:
        st.error("Model files not found. Run: `python src/train_model.py` in the terminal to train the forecasting model.")
    else:
        st.success("✅ Dataset schema is compatible with the Random Forest forecasting model!")
        
        # Forecast length selector
        forecast_days = st.selectbox("Forecast Horizon (Days)", [7, 30, 60, 90], index=1)
        
        # Map case-insensitive column names for compatibility
        map_df = active_df.copy()
        col_mappings = {}
        for target in required_cols:
            for source in map_df.columns:
                if source.lower() == target:
                    col_mappings[source] = target
        map_df = map_df.rename(columns=col_mappings)
        
        # Run forecast
        try:
            with st.spinner("Generating forecasts..."):
                daily_df = prepare_daily_sales(map_df)
                forecast_df = make_future_forecast(model, daily_df, forecast_days)
                
            col_chart, col_table = st.columns([2, 1])
            
            with col_chart:
                fig_forecast = px.line(
                    forecast_df, 
                    x="date", 
                    y="predicted_revenue", 
                    title=f"Next {forecast_days} Days Predicted Revenue Forecast"
                )
                fig_forecast = apply_plotly_theme(fig_forecast, st.session_state["theme_name"])
                st.plotly_chart(fig_forecast, use_container_width=True)
                
            with col_table:
                st.markdown("#### Forecasted Revenue Grid")
                st.dataframe(forecast_df, use_container_width=True, height=350)
                
                forecast_total = forecast_df["predicted_revenue"].sum()
                st.markdown(f"**Total Expected Revenue**: ₹{forecast_total:,.2f}")
                
            # Model metrics
            if os.path.exists(METRICS_PATH):
                with st.expander("📊 Model Training Metrics"):
                    with open(METRICS_PATH, "r") as f:
                        st.code(f.read())
        except Exception as e:
            st.error(f"Error executing forecast pipeline: {str(e)}")
