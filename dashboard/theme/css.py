"""
Global CSS generator for the NeuralStream HUD.
Produces a themed stylesheet string that can be injected via
`st.markdown(..., unsafe_allow_html=True)`.
"""

from dashboard.theme.palette import Palette


def generate_css(palette: Palette) -> str:
    """
    Generate the complete HUD stylesheet for a given palette.

    Sections:
        1. Root variables
        2. Streamlit overrides (body, sidebar, tabs)
        3. Metric cards
        4. Buttons (primary, secondary)
        5. Badges / status pills
        6. Card containers (glassmorphic)
        7. Animations (pulse, scan, glow)
        8. Scrollbar styling
    """
    p = palette
    accent_seq = " ".join(p.chart_sequence)

    return f"""
<style>
/* ================= ROOT VARIABLES ================= */
:root {{
    --bg-root: {p.bg_root};
    --bg-surface: {p.bg_surface};
    --bg-elevated: {p.bg_elevated};
    --text-primary: {p.text_primary};
    --text-secondary: {p.text_secondary};
    --text-muted: {p.text_muted};
    --accent: {p.accent_primary};
    --accent-2: {p.accent_secondary};
    --success: {p.success};
    --warning: {p.warning};
    --danger: {p.danger};
    --info: {p.info};
    --border: {p.border};
    --border-active: {p.border_active};
    --glow: {p.glow_primary};
    --glow-2: {p.glow_secondary};
    --grid: {p.grid_line};
    --mono: 'JetBrains Mono', 'Fira Code', 'SF Mono', monospace;
    --sans: 'Inter', system-ui, -apple-system, sans-serif;
}}

/* ================= GLOBAL ================= */
html, body, [class*="css"], .stApp {{
    background: var(--bg-root) !important;
    color: var(--text-primary);
    font-family: var(--sans);
    -webkit-font-smoothing: antialiased;
}}

.stApp {{
    background:
        radial-gradient(circle at 15% 10%, rgba(56, 189, 248, 0.05) 0%, transparent 45%),
        radial-gradient(circle at 85% 90%, rgba(167, 139, 250, 0.05) 0%, transparent 45%),
        var(--bg-root) !important;
}}

/* Prevent layout collapse */
.block-container {{
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
    max-width: 100% !important;
}}

h1, h2, h3, h4, h5, h6 {{
    color: var(--text-primary) !important;
    font-weight: 700;
    letter-spacing: -0.01em;
}}

h1 {{ font-size: 2rem !important; letter-spacing: -0.02em; }}
h2 {{ font-size: 1.4rem !important; color: {p.accent_primary} !important; }}
h3 {{ font-size: 1.15rem !important; }}

a {{ color: var(--accent) !important; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}

code, pre {{
    font-family: var(--mono) !important;
    background: var(--bg-elevated) !important;
    color: var(--accent);
    border-radius: 6px;
}}

/* ================= SIDEBAR ================= */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg,
        rgba(10, 15, 30, 0.98) 0%,
        rgba(5, 8, 16, 0.98) 100%) !important;
    border-right: 1px solid var(--border);
    box-shadow: 4px 0 30px rgba(0,0,0,0.5);
}}

section[data-testid="stSidebar"] h2 {{
    color: var(--accent) !important;
    font-size: 1.25rem !important;
}}

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
    color: var(--text-secondary);
}}

/* ================= METRIC CARDS ================= */
div[data-testid="stMetric"] {{
    background: linear-gradient(135deg,
        rgba(20, 28, 46, 0.85) 0%,
        rgba(10, 15, 30, 0.6) 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 18px !important;
    box-shadow:
        0 4px 24px rgba(0, 0, 0, 0.5),
        inset 0 1px 0 rgba(255, 255, 255, 0.04);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    transition: all 0.25s ease;
    position: relative;
    overflow: hidden;
}}

div[data-testid="stMetric"]::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg,
        transparent,
        var(--glow) 50%,
        transparent);
    opacity: 0.6;
}}

div[data-testid="stMetric"]:hover {{
    border-color: var(--border-active);
    box-shadow:
        0 6px 32px rgba(0, 0, 0, 0.6),
        0 0 24px var(--glow),
        inset 0 1px 0 rgba(255, 255, 255, 0.08);
    transform: translateY(-1px);
}}

div[data-testid="stMetricLabel"] {{
    color: var(--text-muted) !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 4px;
}}

div[data-testid="stMetricValue"] {{
    color: var(--accent) !important;
    font-family: var(--mono);
    font-weight: 700 !important;
    font-size: 1.65rem !important;
    line-height: 1.1;
    text-shadow: 0 0 20px var(--glow);
}}

div[data-testid="stMetricDelta"] {{
    font-family: var(--mono);
    font-size: 0.75rem !important;
}}

/* ================= BUTTONS ================= */
.stButton > button {{
    background: linear-gradient(135deg,
        rgba(20, 28, 46, 0.9) 0%,
        rgba(10, 15, 30, 0.9) 100%);
    color: var(--text-primary) !important;
    border: 1px solid var(--border);
    border-radius: 10px;
    font-weight: 600;
    letter-spacing: 0.02em;
    padding: 10px 16px;
    transition: all 0.2s ease;
    font-family: var(--sans);
}}

.stButton > button:hover {{
    border-color: var(--border-active);
    color: var(--accent) !important;
    box-shadow:
        0 0 20px var(--glow),
        inset 0 0 12px rgba(56, 189, 248, 0.05);
    transform: translateY(-1px);
}}

.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg,
        {p.accent_primary} 0%,
        {p.accent_secondary} 100%);
    color: #050810 !important;
    border: none;
    font-weight: 700;
    box-shadow: 0 4px 20px var(--glow);
}}

.stButton > button[kind="primary"]:hover {{
    box-shadow: 0 6px 28px var(--glow), 0 0 40px var(--glow-2);
    transform: translateY(-2px);
}}

.stButton > button:active {{
    transform: translateY(0);
    box-shadow: 0 2px 8px var(--glow) !important;
}}

/* ================= TABS ================= */
.stTabs [data-baseweb="tab-list"] {{
    gap: 6px;
    background: transparent;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0;
}}

.stTabs [data-baseweb="tab"] {{
    background: rgba(20, 28, 46, 0.4);
    border: 1px solid transparent;
    border-bottom: none;
    border-radius: 10px 10px 0 0;
    color: var(--text-secondary);
    padding: 10px 18px !important;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.02em;
    transition: all 0.2s ease;
    height: auto !important;
}}

.stTabs [data-baseweb="tab"]:hover {{
    background: rgba(20, 28, 46, 0.7);
    color: var(--text-primary);
}}

.stTabs [data-baseweb="tab"][aria-selected="true"] {{
    background: linear-gradient(180deg,
        rgba(56, 189, 248, 0.18) 0%,
        rgba(56, 189, 248, 0.04) 100%);
    color: var(--accent) !important;
    border: 1px solid var(--border-active);
    border-bottom: none;
    box-shadow: 0 -4px 20px var(--glow);
}}

.stTabs [data-baseweb="tab-highlight"] {{
    background: var(--accent) !important;
    height: 2px;
}}

/* ================= INPUTS / SELECT / RADIO ================= */
.stTextInput input,
.stNumberInput input,
.stSelectbox > div > div,
.stTextArea textarea {{
    background: var(--bg-elevated) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: var(--mono);
}}

.stTextInput input:focus,
.stNumberInput input:focus,
.stTextArea textarea:focus {{
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px var(--glow) !important;
    outline: none !important;
}}

.stRadio > div {{
    background: transparent;
    gap: 6px;
}}

/* Radio buttons rendered as pill cards */
.stRadio > div > label {{
    background: rgba(20, 28, 46, 0.5);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 14px !important;
    transition: all 0.2s ease;
    cursor: pointer;
}}

.stRadio > div > label:hover {{
    border-color: var(--border-active);
    background: rgba(20, 28, 46, 0.85);
}}

.stRadio > div > label[data-checked="true"] {{
    background: rgba(56, 189, 248, 0.12);
    border-color: var(--accent);
    box-shadow: inset 0 0 12px var(--glow);
}}

/* ================= FILE UPLOADER ================= */
[data-testid="stFileUploader"] {{
    border: 1px dashed var(--border-active);
    border-radius: 10px;
    background: rgba(20, 28, 46, 0.4);
    padding: 4px;
}}

[data-testid="stFileUploader"]:hover {{
    background: rgba(20, 28, 46, 0.7);
    box-shadow: 0 0 20px var(--glow);
}}

/* ================= CONTAINERS & EXPANDERS ================= */
div[data-testid="stExpander"] {{
    background: rgba(20, 28, 46, 0.5);
    border: 1px solid var(--border);
    border-radius: 10px;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
}}

div[data-testid="stExpander"]:hover {{
    border-color: var(--border-active);
}}

/* ================= ALERTS / SUCCESS / WARNING ================= */
div[data-testid="stAlert"] {{
    background: rgba(20, 28, 46, 0.7);
    border-radius: 10px;
    border: 1px solid var(--border);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
}}

/* ================= CUSTOM HUD CLASSES ================= */

/* Glassmorphic card container */
.hud-card {{
    background: linear-gradient(135deg,
        rgba(20, 28, 46, 0.85) 0%,
        rgba(10, 15, 30, 0.6) 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 20px;
    margin: 8px 0;
    box-shadow:
        0 4px 24px rgba(0, 0, 0, 0.5),
        inset 0 1px 0 rgba(255, 255, 255, 0.04);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    position: relative;
    overflow: hidden;
}}

.hud-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg,
        transparent,
        var(--accent) 50%,
        transparent);
    opacity: 0.5;
}}

.hud-card-title {{
    color: var(--text-muted);
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin-bottom: 8px;
}}

.hud-card-value {{
    color: var(--accent);
    font-family: var(--mono);
    font-size: 1.6rem;
    font-weight: 700;
    line-height: 1.1;
    text-shadow: 0 0 20px var(--glow);
}}

.hud-card-sub {{
    color: var(--text-secondary);
    font-size: 0.8rem;
    margin-top: 4px;
    font-family: var(--mono);
}}

/* Status badges */
.hud-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-family: var(--mono);
    border: 1px solid;
    backdrop-filter: blur(8px);
}}

.hud-badge.active {{
    background: rgba(16, 185, 129, 0.14);
    color: {p.success};
    border-color: {p.success};
    box-shadow: 0 0 16px rgba(16, 185, 129, 0.35);
}}

.hud-badge.paused {{
    background: rgba(245, 158, 11, 0.14);
    color: {p.warning};
    border-color: {p.warning};
    box-shadow: 0 0 16px rgba(245, 158, 11, 0.3);
}}

.hud-badge.critical {{
    background: rgba(239, 68, 68, 0.16);
    color: {p.danger};
    border-color: {p.danger};
    box-shadow: 0 0 20px rgba(239, 68, 68, 0.4);
    animation: pulse-danger 1.5s ease-in-out infinite;
}}

.hud-badge.info {{
    background: rgba(6, 182, 212, 0.14);
    color: {p.info};
    border-color: {p.info};
    box-shadow: 0 0 14px rgba(6, 182, 212, 0.3);
}}

.hud-badge.neutral {{
    background: rgba(148, 163, 184, 0.14);
    color: var(--text-secondary);
    border-color: var(--text-muted);
}}

/* Divider */
.hud-divider {{
    height: 1px;
    background: linear-gradient(90deg,
        transparent,
        var(--border-active),
        transparent);
    margin: 16px 0;
}}

/* Section header */
.hud-section {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 18px 0 12px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
}}

.hud-section-icon {{
    font-size: 1.15rem;
    filter: drop-shadow(0 0 8px var(--glow));
}}

.hud-section-title {{
    font-size: 0.95rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--accent);
}}

/* Kbd-style chips */
.hud-chip {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    font-family: var(--mono);
    font-size: 0.7rem;
    color: var(--text-secondary);
}}

/* ================= ANIMATIONS ================= */
@keyframes pulse-danger {{
    0%, 100% {{ box-shadow: 0 0 20px rgba(239, 68, 68, 0.4); }}
    50%      {{ box-shadow: 0 0 32px rgba(239, 68, 68, 0.8); }}
}}

@keyframes pulse-accent {{
    0%, 100% {{ box-shadow: 0 0 12px var(--glow); }}
    50%      {{ box-shadow: 0 0 28px var(--glow-2); }}
}}

@keyframes scan-line {{
    0%   {{ transform: translateY(-100%); }}
    100% {{ transform: translateY(100vh); }}
}}

@keyframes fade-in {{
    from {{ opacity: 0; transform: translateY(6px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}

.hud-fade-in {{
    animation: fade-in 0.4s ease both;
}}

/* ================= SCROLLBAR ================= */
::-webkit-scrollbar {{
    width: 10px;
    height: 10px;
}}

::-webkit-scrollbar-track {{
    background: var(--bg-root);
}}

::-webkit-scrollbar-thumb {{
    background: linear-gradient(180deg,
        var(--accent), var(--accent-2));
    border-radius: 5px;
    border: 2px solid var(--bg-root);
}}

::-webkit-scrollbar-thumb:hover {{
    background: var(--accent);
    box-shadow: 0 0 12px var(--glow);
}}

/* ================= PLOTLY ================= */
.js-plotly-plot .plotly .modebar {{
    background: rgba(20, 28, 46, 0.7) !important;
    border-radius: 6px;
    padding: 4px !important;
}}

.js-plotly-plot .plotly .modebar-btn svg {{
    fill: var(--text-secondary) !important;
}}

.js-plotly-plot .plotly .modebar-btn:hover svg {{
    fill: var(--accent) !important;
}}

/* Remove Streamlit default decorations */
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent !important; }}
div[data-testid="stDecoration"] {{ display: none; }}
</style>
"""


def inject_theme(st_module, palette: Palette) -> None:
    """
    Inject the theme CSS into Streamlit.

    Args:
        st_module: Reference to the `streamlit` module
        palette: Palette to use
    """
    st_module.markdown(generate_css(palette), unsafe_allow_html=True)