import streamlit as st

THEMES = {
    "Classic Power BI": {
        "bg_color": "#ffffff",
        "card_bg": "#f3f2f1",
        "text_color": "#252423",
        "accent_color": "#118dff",
        "plotly_theme": "plotly",
        "colors": ["#118DFF", "#12239E", "#E66C37", "#118DFF", "#6B007B", "#E044A7", "#744EC2", "#D9B300", "#D64554"]
    },
    "Cyberpunk Dark": {
        "bg_color": "#0d0e15",
        "card_bg": "#161925",
        "text_color": "#e0e6ed",
        "accent_color": "#ff007f",
        "plotly_theme": "plotly_dark",
        "colors": ["#ff007f", "#00f0ff", "#ab00ff", "#ffd700", "#ff5f00", "#00ff66", "#ff003c", "#007fff", "#ff00ff"]
    },
    "Emerald Forest": {
        "bg_color": "#0f1710",
        "card_bg": "#172b1d",
        "text_color": "#ecfdf5",
        "accent_color": "#10b981",
        "plotly_theme": "plotly_dark",
        "colors": ["#10b981", "#34d399", "#059669", "#f59e0b", "#10b981", "#6ee7b7", "#047857", "#d97706", "#fcd34d"]
    },
    "Ocean Breeze": {
        "bg_color": "#0b132b",
        "card_bg": "#1c2541",
        "text_color": "#edf2f4",
        "accent_color": "#48cae4",
        "plotly_theme": "plotly_dark",
        "colors": ["#00b4d8", "#90e0ef", "#0077b6", "#03045e", "#48cae4", "#0096c7", "#caf0f8", "#023e8a", "#00a896"]
    },
    "Sunset Warmth": {
        "bg_color": "#120c14",
        "card_bg": "#211526",
        "text_color": "#ffe8e8",
        "accent_color": "#ff6b6b",
        "plotly_theme": "plotly_dark",
        "colors": ["#ff6b6b", "#fec89a", "#ff8787", "#ffd166", "#f07167", "#f4a261", "#e76f51", "#ffb703", "#e63946"]
    }
}

def inject_theme_css(theme_name: str = "Cyberpunk Dark"):
    """
    Injects custom CSS to style the Streamlit interface matching the selected theme.
    """
    theme = THEMES.get(theme_name, THEMES["Cyberpunk Dark"])
    
    bg = theme["bg_color"]
    card = theme["card_bg"]
    text = theme["text_color"]
    accent = theme["accent_color"]
    
    css = f"""
    <style>
    /* Styling main content and container boxes */
    .stApp {{
        background-color: {bg} !important;
        color: {text} !important;
    }}
    
    /* Modern metrics styling with glassmorphism card feel */
    div[data-testid="metric-container"] {{
        background: {card} !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        padding: 1.5rem !important;
        border-radius: 12px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2) !important;
        transition: all 0.3s ease-in-out !important;
    }}
    
    div[data-testid="metric-container"]:hover {{
        border-color: {accent}55 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.3) !important;
    }}
    
    /* Title gradient styles */
    h1 {{
        background: linear-gradient(45deg, {accent}, #ffffff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        letter-spacing: -0.5px !important;
        margin-bottom: 1.5rem !important;
    }}
    
    /* Subheaders styled nicely */
    h2, h3, h4 {{
        color: {text} !important;
        font-weight: 600 !important;
        letter-spacing: -0.3px !important;
    }}
    
    /* Card blocks for custom layouts */
    .dashboard-card {{
        background: {card} !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        padding: 1.5rem !important;
        border-radius: 16px !important;
        box-shadow: 0 10px 30px 0 rgba(0, 0, 0, 0.15) !important;
        margin-bottom: 1.5rem !important;
    }}
    
    /* File uploader styling */
    section[data-testid="stFileUploader"] {{
        background-color: {card} !important;
        border: 2px dashed {accent}33 !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }}
    
    /* Style tabs container */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px !important;
        background-color: transparent !important;
    }}

    .stTabs [data-baseweb="tab"] {{
        background-color: {card}dd !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 8px 8px 0px 0px !important;
        color: {text}aa !important;
        padding: 10px 20px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }}

    .stTabs [data-baseweb="tab"]:hover {{
        color: {text} !important;
        border-color: {accent}55 !important;
    }}

    .stTabs [aria-selected="true"] {{
        background-color: {card} !important;
        color: {accent} !important;
        border-bottom: 2px solid {accent} !important;
        font-weight: 700 !important;
    }}
    
    /* Clean button designs */
    .stButton>button {{
        background-color: {card} !important;
        color: {text} !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }}
    
    .stButton>button:hover {{
        border-color: {accent} !important;
        color: {accent} !important;
        box-shadow: 0 4px 15px 0 {accent}22 !important;
    }}
    
    /* Primary action buttons */
    .stButton>button[kind="primary"] {{
        background: linear-gradient(135deg, {accent}, {accent}cc) !important;
        color: white !important;
        border: none !important;
    }}
    
    .stButton>button[kind="primary"]:hover {{
        box-shadow: 0 4px 20px 0 {accent}44 !important;
        transform: translateY(-1px) !important;
    }}
    
    /* Style tables and dataframes */
    [data-testid="stDataFrame"] {{
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        overflow: hidden !important;
    }}
    
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def apply_plotly_theme(fig, theme_name: str = "Cyberpunk Dark"):
    """
    Applies the appropriate template and color palette to a Plotly figure.
    """
    theme = THEMES.get(theme_name, THEMES["Cyberpunk Dark"])
    fig.update_layout(
        template=theme["plotly_theme"],
        colorway=theme["colors"],
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=theme["text_color"]),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    
    # Grid lines settings
    grid_color = "rgba(255, 255, 255, 0.05)" if theme["plotly_theme"] == "plotly_dark" else "rgba(0, 0, 0, 0.05)"
    fig.update_xaxes(gridcolor=grid_color, zerolinecolor=grid_color)
    fig.update_yaxes(gridcolor=grid_color, zerolinecolor=grid_color)
    
    return fig
