"""
Simulador Interactivo de Biorreactores 3D (Tema Claro & Pastel)
================================================================
Arquitectura modular en Streamlit, NumPy y Plotly con 4 pestañas:
1. Filtración (Mallas): Spin-Filter 3D animado (10 s) + presets de línea celular,
   tipo de malla, retrolavado, crecimiento (1, 3, 7 días) y perfusión (VVD).
2. Térmico 3D: T(r, z, t) animado (10 s) + acoplamiento Viscosidad(T) y alerta Tg/Tm.
3. Dinámica de Fluidos: Impulsores vs. Solo filtro rotatorio + vectores 3D (go.Cone)
   + curva Np-Re.
4. Bioproceso, Materiales y Costos: Catálogo de líneas celulares, medios/nutrientes,
   polímeros biocompatibles (Tg/Tm), calculadora de costos, proveedores y prototipo rápido.
"""

from datetime import datetime
import io
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# =====================================================================
# 0. CONFIGURACIÓN DE PÁGINA Y ESTILOS PASTEL (TEMA CLARO)
# =====================================================================
st.set_page_config(
    page_title="Simulador de Biorreactores 3D",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
    div[data-testid="stMetric"] {
        background-color: #f3f6fc;
        border: 1px solid #dbe4ff;
        padding: 10px 14px;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }
    div[data-testid="stMetricLabel"] { color: #475569; font-weight: 600; }
    div[data-testid="stMetricValue"] { color: #1e293b; }
    .pastel-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #748ffc;
        padding: 12px 16px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

RHO_FLUID = 1005.0   # kg/m³ (medio acuoso a 37 °C)
ANIM_SECONDS = 10.0  # duración de animación en pantalla
N_FRAMES = 50        # cuadros por animación (200 ms c/u)
FRAME_MS = int(ANIM_SECONDS * 1000 / N_FRAMES)

# Paleta pastel compartida para gráficas sobre fondo claro
PASTEL = {
    "mesh": "#74c0fc",
    "mesh_dark": "#4dabf7",
    "wall": "#ced4da",
    "bulk": "#20c997",      # menta pastel legible
    "fouled": "#ffa94d",    # durazno pastel
    "permeated": "#ff6b6b", # coral pastel
    "rib": "#f06595",       # rosa pastel
    "impeller": "#fcc419",  # amarillo suave
    "star": "#fa5252",
}

# =====================================================================
# 1. CATÁLOGOS DE BIOINGENIERÍA (LÍNEAS CELULARES, MALLAS Y POLÍMEROS)
# =====================================================================
CELL_LINES: Dict[str, Dict[str, Any]] = {
    "CHO-K1 (Mamífero — Suspensión)": {
        "diam_um": 15.0,
        "mass_ng": 2.5,
        "shape": "Esférica",
        "culture_type": "Suspensión libre (ideal para Spin-Filter)",
        "mu_max_h": 0.032,        # t_duplicación ~21.7 h
        "t_opt_c": 37.0,
        "shear_limit_pa": 1.5,
        "q_met_pw": 90.0,
        "medium": "CD-CHO / DMEM-F12 (libre de suero)",
        "nutrients": "Glucosa (4–6 g/L), L-Glutamina (4 mM), aminoácidos esenciales, pluronic F-68",
        "product": "Anticuerpos monoclonales (mAbs) y proteínas recombinantes",
        "extraction": "Permeado libre de células -> Cromatografía de afinidad (Proteína A) + Ultrafiltración",
        "suppliers": "ATCC (CCL-61), ECACC | Medio: Gibco / Sigma-Aldrich",
        "cost_cell_usd": "$480 – $650 USD (vial criopreservado)",
        "cost_vial_usd": 565.0,
        "cost_medium_usd_l": 45.0,
    },
    "HEK293 (Humana — Vacunas/Vectores)": {
        "diam_um": 14.0,
        "mass_ng": 2.1,
        "shape": "Esférica / Epitelial adaptada",
        "culture_type": "Suspensión o Adherente en microacarreadores",
        "mu_max_h": 0.030,
        "t_opt_c": 37.0,
        "shear_limit_pa": 1.3,
        "q_met_pw": 85.0,
        "medium": "FreeStyle 293 / Expi293 / DMEM",
        "nutrients": "Glucosa, L-Glutamina, Piruvato de sodio, vitaminas B",
        "product": "Vacunas virales, vectores lentivirales/AAV y proteínas",
        "extraction": "Cosecha del sobrenadante o lisis suave -> Clarificación + Cromatografía de intercambio iónico",
        "suppliers": "ATCC (CRL-1573), DSMZ | Medio: Thermo Fisher (Gibco)",
        "cost_cell_usd": "$420 – $600 USD",
        "cost_vial_usd": 510.0,
        "cost_medium_usd_l": 55.0,
    },
    "Hibridoma Sp2/0 (Anticuerpos)": {
        "diam_um": 13.0,
        "mass_ng": 1.8,
        "shape": "Esférica",
        "culture_type": "Suspensión libre",
        "mu_max_h": 0.036,
        "t_opt_c": 37.0,
        "shear_limit_pa": 1.2,
        "q_met_pw": 100.0,
        "medium": "RPMI-1640 + 10% SFB o Hybridoma-SFM",
        "nutrients": "Glucosa, Glutamina, lípidos, insulina/transferrina",
        "product": "Anticuerpos monoclonales terapéuticos y de diagnóstico",
        "extraction": "Perfusión continua -> Precipitación / Proteína G",
        "suppliers": "ATCC (CRL-1581), ECACC | Medio: Corning / Cytiva",
        "cost_cell_usd": "$350 – $520 USD",
        "cost_vial_usd": 435.0,
        "cost_medium_usd_l": 38.0,
    },
    "Saccharomyces / Pichia (Levadura)": {
        "diam_um": 6.0,
        "mass_ng": 0.045,
        "shape": "Ovoide (con pared celular rígida)",
        "culture_type": "Suspensión de alta densidad",
        "mu_max_h": 0.25,
        "t_opt_c": 30.0,
        "shear_limit_pa": 15.0,
        "q_met_pw": 18.0,
        "medium": "YPD / Medio salino definido con glicerol o metanol",
        "nutrients": "Dextrosa o Glicerol (10–20 g/L), peptona, extracto de levadura, fosfatos",
        "product": "Proteínas recombinantes, enzimas industriales, vacunas (VLP)",
        "extraction": "Secreción al medio -> Microfiltración + Diálisis / Ultrafiltración TFF",
        "suppliers": "ATCC, DSMZ | Reactivos: Sigma-Aldrich / BD Difco",
        "cost_cell_usd": "$150 – $280 USD",
        "cost_vial_usd": 215.0,
        "cost_medium_usd_l": 8.0,
    },
    "Escherichia coli (Bacteria — Bacilo)": {
        "diam_um": 1.8,
        "mass_ng": 0.001,
        "shape": "Bacilo (cilíndrica alargada)",
        "culture_type": "Suspensión libre de muy rápido crecimiento",
        "mu_max_h": 0.65,
        "t_opt_c": 37.0,
        "shear_limit_pa": 25.0,
        "q_met_pw": 3.0,
        "medium": "Caldo LB (Luria-Bertani) o Medio Mínimo M9",
        "nutrients": "Triptona, extracto de levadura, NaCl, glucosa, sales de amonio",
        "product": "Insulina, plásmidos, enzimas; (en Streptomyces: antibióticos)",
        "extraction": "Lisis celular (homogeneización) -> Centrifugación + Purificación de cuerpos de inclusión",
        "suppliers": "ATCC, Addgene, NEB | Medio: Merck / Sigma-Aldrich",
        "cost_cell_usd": "$80 – $190 USD",
        "cost_vial_usd": 135.0,
        "cost_medium_usd_l": 3.5,
    },
}

MESH_MATERIALS: Dict[str, Dict[str, Any]] = {
    "Acero Inoxidable 316L — Tejido Plano": {
        "open_area": 0.34,
        "pore_range": "5 – 40 µm",
        "tg_c": None,
        "tm_c": 1400,
        "extrudable": "No (tejido metálico por alambre)",
        "autoclave": "Sí (121 °C, reutilizable indefinidamente)",
        "supplier": "McMaster-Carr / W.S. Tyler (~$45 – $110 USD/m²)",
        "cost_unit_usd": 75.0,
    },
    "Acero Inoxidable 316L — Tejido Holandés": {
        "open_area": 0.18,
        "pore_range": "2 – 20 µm",
        "tg_c": None,
        "tm_c": 1400,
        "extrudable": "No (alta resistencia mecánica)",
        "autoclave": "Sí (121 °C, soporta alto ΔP)",
        "supplier": "Bopp / McMaster-Carr (~$95 – $220 USD/m²)",
        "cost_unit_usd": 155.0,
    },
    "PVDF (Polímero biocompatible)": {
        "open_area": 0.45,
        "pore_range": "0.2 – 15 µm",
        "tg_c": -35,
        "tm_c": 170,
        "extrudable": "Sí (fibra hueca o lámina porosa)",
        "autoclave": "Sí (hasta 125 °C)",
        "supplier": "Millipore / Sigma-Aldrich (~$60 – $150 USD)",
        "cost_unit_usd": 105.0,
    },
    "PEEK (Polímero alta temperatura)": {
        "open_area": 0.30,
        "pore_range": "5 – 30 µm",
        "tg_c": 143,
        "tm_c": 343,
        "extrudable": "Sí (filamento longitudinal / moldeo)",
        "autoclave": "Sí (121–134 °C sin deformarse)",
        "supplier": "Goodfellow / Victrex (~$120 – $260 USD)",
        "cost_unit_usd": 190.0,
    },
    "PCL (Policaprolactona — Biopolímero)": {
        "open_area": 0.38,
        "pore_range": "10 – 40 µm",
        "tg_c": -60,
        "tm_c": 60,
        "extrudable": "Sí (funde a 60 °C, ideal prototipos longitudinales)",
        "autoclave": "No (se funde > 60 °C; esterilizar con etanol 70% / UV)",
        "supplier": "Sigma-Aldrich (~$50 – $90 USD / 100 g)",
        "cost_unit_usd": 65.0,
    },
    "PLA (Ácido Poliláctico — Impresión 3D)": {
        "open_area": 0.35,
        "pore_range": "15 – 40 µm",
        "tg_c": 60,
        "tm_c": 170,
        "extrudable": "Sí (extrusión longitudinal e impresión 3D)",
        "autoclave": "No (se ablanda > 60 °C; usar UV / etanol)",
        "supplier": "Filamento grado médico / NatureWorks (~$25 – $60 USD/kg)",
        "cost_unit_usd": 35.0,
    },
}


# =====================================================================
# 2. UTILIDADES COMUNES (ANIMACIÓN PLOTLY PASTEL + REPORTES MD)
# =====================================================================
def add_animation_controls(
    fig: go.Figure,
    frames: Sequence[go.Frame],
    labels: Sequence[str],
    slider_prefix: str,
) -> go.Figure:
    """Agrega frames + botones ▶ Reproducir / ⏸ Pausa / ↺ Inicio + slider temporal."""
    fig.frames = list(frames)
    names = [f.name for f in frames]
    play_args = dict(
        frame=dict(duration=FRAME_MS, redraw=True),
        transition=dict(duration=0),
        fromcurrent=True,
        mode="immediate",
    )
    jump = lambda name: [  # noqa: E731
        [name],
        dict(frame=dict(duration=0, redraw=True), transition=dict(duration=0), mode="immediate"),
    ]
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.0,
                y=0.0,
                xanchor="left",
                yanchor="top",
                pad=dict(t=42, r=6),
                showactive=False,
                bgcolor="#f1f5f9",
                bordercolor="#cbd5e1",
                font=dict(color="#1e293b"),
                buttons=[
                    dict(label="▶ Reproducir", method="animate", args=[None, play_args]),
                    dict(
                        label="⏸ Pausa",
                        method="animate",
                        args=[
                            [None],
                            dict(frame=dict(duration=0, redraw=False), transition=dict(duration=0), mode="immediate"),
                        ],
                    ),
                    dict(label="↺ Inicio", method="animate", args=jump(names[0])),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                x=0.35,
                len=0.63,
                y=0.0,
                pad=dict(t=42, b=0),
                bgcolor="#e2e8f0",
                activebgcolor="#748ffc",
                currentvalue=dict(prefix=slider_prefix, font=dict(size=14, color="#1e293b")),
                ticklen=0,
                font=dict(color="rgba(0,0,0,0)"),
                steps=[dict(method="animate", label=lab, args=jump(name)) for name, lab in zip(names, labels)],
            )
        ],
    )
    return fig


def make_report(
    title: str,
    module: str,
    params_rows: List[Tuple[str, str, str]],
    kpi_rows: List[Tuple[str, str, str]],
    equations: str,
    diagnostics: List[str],
) -> str:
    """Genera una memoria de cálculo en Markdown (.md)."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    table = lambda rows: "\n".join(f"| {a} | `{b}` | {c} |" for a, b, c in rows)  # noqa: E731
    diag = "\n".join(f"* {d}" for d in diagnostics)
    return f"""# Memoria de Cálculo: {title}

**Fecha de generación:** {ts}  
**Módulo:** {module}  
**Software:** Simulador Computacional de Biorreactores (Python / NumPy / Plotly)

---

## 1. Parámetros de Entrada

| Parámetro | Valor | Unidad |
| :--- | :---: | :---: |
{table(params_rows)}

---

## 2. Resultados e Indicadores Clave (KPIs)

| Indicador | Resultado | Unidad |
| :--- | :---: | :---: |
{table(kpi_rows)}

---

## 3. Marco Matemático

{equations}

---

## 4. Diagnóstico Automático

{diag}

---
*Reporte generado automáticamente por el Simulador de Biorreactores 3D.*
"""


def _form_defaults(key: str, defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Inicializa en session_state los parámetros de un módulo."""
    if key not in st.session_state:
        st.session_state[key] = dict(defaults)
    return st.session_state[key]


def _light_scene(
    xr: List[float],
    yr: List[float],
    zr: List[float],
    z_title: str = "Altura Z (mm)",
    xy_unit: str = "mm",
) -> Dict[str, Any]:
    """Configuración de ejes claros y pasteles para escenas 3D."""
    axis_style = dict(
        backgroundcolor="#f8fafc",
        gridcolor="#e2e8f0",
        zerolinecolor="#cbd5e1",
        showbackground=True,
    )
    return dict(
        xaxis=dict(title=f"X ({xy_unit})", range=xr, **axis_style),
        yaxis=dict(title=f"Y ({xy_unit})", range=yr, **axis_style),
        zaxis=dict(title=z_title, range=zr, **axis_style),
    )


def _build_3d_arrow_traces(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    max_arrow_len: float,
    vmax: float,
    colorscale: List[Any],
    show_colorbar: bool = True,
    name: str = "Vectores 3D",
    line_width: float = 3.6,
    head_ratio: float = 0.30,
    opacity: float = 0.92,
) -> Tuple[go.Scatter3d, go.Cone]:
    """Construye flechas 3D reales (eje Scatter3d + punta pequeña Cone con anchor='tip')."""
    xa = np.asarray(x, dtype=float).ravel()
    ya = np.asarray(y, dtype=float).ravel()
    za = np.asarray(z, dtype=float).ravel()
    ua = np.asarray(u, dtype=float).ravel()
    va = np.asarray(v, dtype=float).ravel()
    wa = np.asarray(w, dtype=float).ravel()

    mag = np.sqrt(ua**2 + va**2 + wa**2)
    vmax_s = max(float(vmax), 1e-9)
    mask = mag > 1e-9

    if not np.any(mask):
        empty_line = go.Scatter3d(
            x=[], y=[], z=[], mode="lines",
            line=dict(color=[], colorscale=colorscale, cmin=0.0, cmax=vmax_s, width=line_width),
            name=f"{name} (ejes)", hoverinfo="skip", showlegend=False,
        )
        empty_cone = go.Cone(
            x=[], y=[], z=[], u=[], v=[], w=[], colorscale=colorscale,
            sizemode="raw", sizeref=1.0, cmin=0.0, cmax=vmax_s, anchor="tip",
            showscale=show_colorbar, name=name,
        )
        return empty_line, empty_cone

    xa, ya, za = xa[mask], ya[mask], za[mask]
    ua, va, wa, mag = ua[mask], va[mask], wa[mask], mag[mask]
    dx = ua / mag
    dy = va / mag
    dz = wa / mag

    lengths = max_arrow_len * np.clip(mag / vmax_s, 0.14, 1.0)
    xt = xa + lengths * dx
    yt = ya + lengths * dy
    zt = za + lengths * dz

    n = len(xa)
    sx = np.empty((n, 3), dtype=object)
    sy = np.empty((n, 3), dtype=object)
    sz = np.empty((n, 3), dtype=object)
    sc = np.empty((n, 3), dtype=float)
    sx[:, 0], sx[:, 1], sx[:, 2] = xa, xt, None
    sy[:, 0], sy[:, 1], sy[:, 2] = ya, yt, None
    sz[:, 0], sz[:, 1], sz[:, 2] = za, zt, None
    sc[:, 0], sc[:, 1], sc[:, 2] = mag, mag, 0.0

    shaft_trace = go.Scatter3d(
        x=sx.ravel().tolist(),
        y=sy.ravel().tolist(),
        z=sz.ravel().tolist(),
        mode="lines",
        line=dict(color=sc.ravel().tolist(), colorscale=colorscale, cmin=0.0, cmax=vmax_s, width=line_width),
        opacity=opacity,
        name=f"{name} (ejes)",
        hoverinfo="skip",
        showlegend=False,
    )

    cone_kwargs: Dict[str, Any] = {}
    if show_colorbar:
        cone_kwargs["colorbar"] = dict(title="|v| (m/s)", thickness=14, len=0.72)

    head_trace = go.Cone(
        x=xt, y=yt, z=zt,
        u=ua, v=va, w=wa,
        colorscale=colorscale,
        sizemode="raw",
        sizeref=(head_ratio * max_arrow_len) / vmax_s,
        cmin=0.0,
        cmax=vmax_s,
        anchor="tip",
        showscale=show_colorbar,
        opacity=opacity,
        name=name,
        hovertemplate=(
            "x=%{x:.1f} · y=%{y:.1f} · z=%{z:.1f}<br>"
            "vx=%{u:.3f} · vy=%{v:.3f} · vz=%{w:.3f} m/s<extra>Vector</extra>"
        ),
        **cone_kwargs,
    )
    return shaft_trace, head_trace


def _quiver_2d_coords(
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    max_len: float,
    ref_mag: float | None = None,
    head_ratio: float = 0.34,
    head_angle: float = 0.48,
) -> Tuple[List[float | None], List[float | None]]:
    """Genera coordenadas (qx, qy) separadas por None para dibujar flechas 2D en NumPy puro."""
    xa = np.asarray(x, dtype=float).ravel()
    ya = np.asarray(y, dtype=float).ravel()
    ua = np.asarray(u, dtype=float).ravel()
    va = np.asarray(v, dtype=float).ravel()

    mag = np.hypot(ua, va)
    if len(mag) == 0 or float(np.max(mag)) < 1e-12:
        return [], []

    scale_ref = float(ref_mag) if (ref_mag is not None and ref_mag > 1e-12) else float(np.max(mag))
    mask = mag > 1e-9
    if not np.any(mask):
        return [], []

    xa, ya, ua, va, mag = xa[mask], ya[mask], ua[mask], va[mask], mag[mask]
    lengths = max_len * np.clip(mag / scale_ref, 0.12, 1.0)
    ang = np.arctan2(va, ua)

    x1 = xa + lengths * np.cos(ang)
    y1 = ya + lengths * np.sin(ang)
    hl = head_ratio * lengths
    hx1 = x1 - hl * np.cos(ang - head_angle)
    hy1 = y1 - hl * np.sin(ang - head_angle)
    hx2 = x1 - hl * np.cos(ang + head_angle)
    hy2 = y1 - hl * np.sin(ang + head_angle)

    n = len(xa)
    qx_arr = np.empty((n, 7), dtype=object)
    qy_arr = np.empty((n, 7), dtype=object)
    qx_arr[:, 0] = xa
    qx_arr[:, 1] = x1
    qx_arr[:, 2] = None
    qx_arr[:, 3] = hx1
    qx_arr[:, 4] = x1
    qx_arr[:, 5] = hx2
    qx_arr[:, 6] = None

    qy_arr[:, 0] = ya
    qy_arr[:, 1] = y1
    qy_arr[:, 2] = None
    qy_arr[:, 3] = hy1
    qy_arr[:, 4] = y1
    qy_arr[:, 5] = hy2
    qy_arr[:, 6] = None

    return qx_arr.ravel().tolist(), qy_arr.ravel().tolist()


# =====================================================================
# 3. MÓDULO 1 — FILTRACIÓN ROTATORIA + CULTIVO EN PERFUSIÓN
# =====================================================================
FILT_DEFAULTS = dict(
    cell_line="CHO-K1 (Mamífero — Suspensión)",
    mesh_material="Acero Inoxidable 316L — Tejido Plano",
    rpm=180,
    pore_size_um=10.0,
    cell_diameter_um=15.0,
    cell_density_ml=5.0e6,
    horizon_days=3,
    seed_density_ml=1000.0,
    perfusion_vvd=1.5,
    backflush=False,
)


@st.cache_data(show_spinner=False)
def run_filtration_solver(
    cell_line: str,
    mesh_material: str,
    rpm: float,
    pore_size_um: float,
    cell_diameter_um: float,
    cell_density_ml: float,
    horizon_days: int,
    seed_density_ml: float,
    perfusion_vvd: float,
    backflush: bool,
    duration_s: float = ANIM_SECONDS,
    n_frames: int = N_FRAMES,
    seed: int = 7,
) -> Dict[str, Any]:
    """
    Integra la cinemática 3D de partículas frente a la malla rotatoria (0 -> 10 s)
    y calcula la evolución macroscópica del cultivo (1, 3 o 7 días) con perfusión.
    """
    rng = np.random.default_rng(seed)
    cell_info = CELL_LINES[cell_line]
    mesh_info = MESH_MATERIALS[mesh_material]

    r_mesh, r_vessel, height = 40.0, 120.0, 240.0  # mm
    sub = 4
    dt = duration_s / (n_frames * sub)

    n = int(np.clip(120 + 28 * cell_density_ml / 1e6, 150, 1400))
    diam = np.clip(rng.normal(cell_diameter_um, 0.12 * cell_diameter_um, n), 0.8, None)
    r = np.sqrt(rng.uniform((r_mesh * 1.3) ** 2, (r_vessel * 0.95) ** 2, n))
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    z = rng.uniform(12.0, height - 12.0, n)
    state = np.zeros(n, dtype=int)  # 0 suspensión, 1 retenida, 2 permeada
    pass_target = rng.uniform(0.1, 0.7, n) * r_mesh

    rpm_n = rpm / 600.0
    f_rev = 0.04 + 0.40 * rpm_n
    w_vis = 2.0 * np.pi * f_rev
    capacity = 260.0 * (mesh_info["open_area"] / 0.34)

    # -----------------------------------------------------------------
    # >>> INYECTAR ECUACIONES PROPIAS AQUÍ <<<
    # -----------------------------------------------------------------
    v_suc = 12.0 * (0.7 + 0.35 * perfusion_vvd)
    v_lift = 19.0 * rpm_n**1.5 * (diam / 15.0)
    detach_rate = (0.02 + 0.9 * rpm_n**2) * (2.6 if backflush else 1.0)
    # -----------------------------------------------------------------

    xs, ys, zs, sts, counts = [], [], [], [], []
    for k in range(n_frames + 1):
        xs.append(r * np.cos(theta))
        ys.append(r * np.sin(theta))
        zs.append(z.copy())
        sts.append(state.copy())
        counts.append([int((state == 0).sum()), int((state == 1).sum()), int((state == 2).sum())])
        if k == n_frames:
            break
        for _ in range(sub):
            bulk = state == 0
            nb = int(bulk.sum())
            if nb:
                rb = r[bulk]
                dr = (-v_suc * np.sqrt(r_mesh / rb) + v_lift[bulk]) * dt
                dr += 3.0 * np.sqrt(dt) * rng.standard_normal(nb)
                r[bulk] = np.clip(rb + dr, r_mesh, r_vessel * 0.97)
                theta[bulk] += w_vis * (r_mesh / r[bulk]) ** 1.5 * dt
                z[bulk] = np.clip(z[bulk] + 4.0 * np.sqrt(dt) * rng.standard_normal(nb), 8.0, height - 8.0)

            contact = (state == 0) & (r <= r_mesh * 1.03)
            if contact.any():
                frac = min(1.0, (state == 1).sum() / max(capacity, 1.0))
                perm_prob = min(0.95, 0.85 * (mesh_info["open_area"] / 0.34)) * (1.0 - frac)
                can_pass = (diam < pore_size_um) & (rng.random(n) < perm_prob)
                passing = contact & can_pass
                sticking = contact & ~can_pass
                state[passing] = 2
                r[passing] = r_mesh * 0.92
                state[sticking] = 1
                r[sticking] = r_mesh * 1.03 + rng.uniform(0.0, 5.0, int(sticking.sum())) * (0.5 + frac)

            stuck = state == 1
            theta[stuck] += w_vis * dt
            # Pulso de retrolavado a mitad de simulación o limpieza continua por RPM
            pulse_boost = 0.35 if (backflush and k in (22, 23, 38, 39)) else 0.0
            detach = stuck & (rng.random(n) < np.clip(1.0 - np.exp(-detach_rate * dt) + pulse_boost, 0.0, 0.95))
            state[detach] = 0
            r[detach] = r_mesh * 1.14

            perm = state == 2
            r[perm] += (pass_target[perm] - r[perm]) * min(1.0, 4.0 * dt)
            theta[perm] += w_vis * 1.5 * dt
            z[perm] = np.maximum(z[perm] - 30.0 * dt, 2.0)

    counts_arr = np.array(counts)
    t = np.linspace(0.0, duration_s, n_frames + 1)

    # Hidrodinámica de la malla
    omega = rpm * 2.0 * np.pi / 60.0
    mu = 1.1e-3
    v_tip = omega * 0.04
    delta_bl = max(1e-4, 0.005 / np.sqrt(1.0 + rpm / 30.0))
    shear = mu * v_tip / delta_bl
    re_rot = RHO_FLUID * omega * 0.04**2 / mu
    retention = float((n - counts_arr[-1, 2]) / n * 100.0)
    fouling = float(min(100.0, counts_arr[-1, 1] / max(capacity, 1.0) * 100.0))

    # Modelo macroscópico de cultivo (1, 3 o 7 días) con crecimiento logístico + perfusión
    hours = np.linspace(0.0, horizon_days * 24.0, 80)
    dt_h = hours[1] - hours[0]
    mu_max = cell_info["mu_max_h"]
    # Penalización por daño de corte si supera el límite de la línea celular
    shear_penalty = max(0.2, 1.0 - 0.35 * max(0.0, shear / cell_info["shear_limit_pa"] - 1.0))
    x_cap = 2.5e7 * (0.4 + 0.6 * min(perfusion_vvd, 3.0))  # capacidad máxima sostenida por perfusión
    x_curve = np.zeros_like(hours)
    s_curve = np.zeros_like(hours)  # glucosa g/L
    x_curve[0] = seed_density_ml
    s_curve[0] = 4.5
    d_bleed = (1.0 - retention / 100.0) * perfusion_vvd / 24.0  # pérdida celular por permeado

    for i in range(1, len(hours)):
        monod = s_curve[i - 1] / (0.2 + s_curve[i - 1])
        mu_eff = mu_max * monod * shear_penalty * (1.0 - x_curve[i - 1] / x_cap)
        dx = (mu_eff - d_bleed) * x_curve[i - 1]
        x_curve[i] = max(100.0, x_curve[i - 1] + dx * dt_h)
        # Balance de glucosa con alimentación por perfusión (S_feed = 4.5 g/L)
        ds = (perfusion_vvd / 24.0) * (4.5 - s_curve[i - 1]) - 1.5e-10 * x_curve[i - 1]
        s_curve[i] = float(np.clip(s_curve[i - 1] + ds * dt_h, 0.05, 5.0))

    # Producción acumulada estimada de proteína/vacuna (mg/L de reactor)
    protein_mgl = float(np.trapezoid(x_curve, hours) * 1.2e-8 * (retention / 100.0))

    return {
        "x": np.array(xs),
        "y": np.array(ys),
        "z": np.array(zs),
        "states": np.array(sts),
        "diam": diam,
        "t": t,
        "counts": counts_arr,
        "w_vis": float(w_vis),
        "n_particles": n,
        "geometry": dict(r_mesh=r_mesh, r_vessel=r_vessel, height=height),
        "vel_field": dict(
            omega=float(omega),
            v_suc_ms=float(v_suc / 1000.0),
            r_mesh_mm=float(r_mesh),
            r_vessel_mm=float(r_vessel),
            height_mm=float(height),
        ),
        "culture": dict(hours=hours, days=hours / 24.0, cells_ml=x_curve, glucose_gl=s_curve),
        "kpis": dict(
            retention_efficiency=retention,
            fouling_index=fouling,
            shear_stress_pa=float(shear),
            shear_limit_pa=float(cell_info["shear_limit_pa"]),
            re_rot=float(re_rot),
            v_tip_ms=float(v_tip),
            bulk=int(counts_arr[-1, 0]),
            fouled=int(counts_arr[-1, 1]),
            permeated=int(counts_arr[-1, 2]),
            final_cells_ml=float(x_curve[-1]),
            protein_mgl=protein_mgl,
        ),
    }


def build_filtration_figure(sim: Dict[str, Any], params: Dict[str, Any], show_vessel_wall: bool = True) -> go.Figure:
    """Figura 3D animada en tonos pastel claros."""
    g = sim["geometry"]
    r_mesh, r_vessel, h = g["r_mesh"], g["r_vessel"], g["height"]
    diam, sts = sim["diam"], sim["states"]
    sizes = np.clip(diam * 0.55, 3.0, 14.0)
    n_frames = len(sim["t"])

    th = np.linspace(0, 2 * np.pi, 45)
    zz = np.linspace(0, h, 25)
    thm, zm = np.meshgrid(th, zz)

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=r_mesh * np.cos(thm), y=r_mesh * np.sin(thm), z=zm,
            name="Malla rotatoria", showscale=False, opacity=0.35,
            colorscale=[[0, PASTEL["mesh"]], [1, PASTEL["mesh_dark"]]], hoverinfo="name",
            contours=dict(
                x=dict(show=True, color="rgba(30,41,59,0.18)", width=1),
                y=dict(show=True, color="rgba(30,41,59,0.18)", width=1),
                z=dict(show=True, color="rgba(30,41,59,0.22)", width=1),
            ),
        )
    )
    fig.add_trace(
        go.Surface(
            x=r_vessel * np.cos(thm), y=r_vessel * np.sin(thm), z=zm,
            name="Pared del biorreactor", showscale=False, opacity=0.10,
            colorscale=[[0, PASTEL["wall"]], [1, "#e9ecef"]], hoverinfo="skip", visible=show_vessel_wall,
        )
    )
    for zref, label, size, color, sym in (
        (h * 0.95, f"Poro {params['pore_size_um']:.1f} µm", params["pore_size_um"], "#475569", "circle-open"),
        (h * 0.82, f"Célula {params['cell_diameter_um']:.1f} µm", params["cell_diameter_um"], PASTEL["bulk"], "circle"),
    ):
        fig.add_trace(
            go.Scatter3d(
                x=[-r_vessel * 1.25], y=[0], z=[zref], mode="markers+text", text=[label],
                textposition="middle right", textfont=dict(color="#1e293b", size=12),
                marker=dict(size=float(np.clip(size * 0.55, 3, 24)), color=color, symbol=sym, line=dict(width=3, color=color)),
                name=label, showlegend=False, hoverinfo="skip",
            )
        )

    styles = [
        ("En suspensión", PASTEL["bulk"], "circle", 0.88),
        ("Retenida en malla (fouling)", PASTEL["fouled"], "diamond", 0.95),
        ("Permeada (interior filtro)", PASTEL["permeated"], "circle", 0.92),
    ]

    def particle_trace(k: int, s: int) -> go.Scatter3d:
        m = sts[k] == s
        name, color, symbol, op = styles[s]
        return go.Scatter3d(
            x=sim["x"][k][m], y=sim["y"][k][m], z=sim["z"][k][m], mode="markers", name=name,
            marker=dict(size=sizes[m], color=color, symbol=symbol, opacity=op,
                        line=dict(width=0.6, color="rgba(30,41,59,0.35)")),
            hovertemplate=f"<b>{name}</b><br>X %{{x:.0f}} · Y %{{y:.0f}} · Z %{{z:.0f}} mm<extra></extra>",
        )

    def rib_trace(k: int) -> go.Scatter3d:
        a = sim["w_vis"] * sim["t"][k]
        xr, yr, zr = [], [], []
        for aa in (a, a + np.pi):
            xr += [r_mesh * np.cos(aa)] * 2 + [None]
            yr += [r_mesh * np.sin(aa)] * 2 + [None]
            zr += [0, h, None]
        return go.Scatter3d(x=xr, y=yr, z=zr, mode="lines", name="Marca de giro",
                            line=dict(color=PASTEL["rib"], width=8), hoverinfo="skip")

    for s in range(3):
        fig.add_trace(particle_trace(0, s))
    fig.add_trace(rib_trace(0))

    frames = [
        go.Frame(data=[particle_trace(k, 0), particle_trace(k, 1), particle_trace(k, 2), rib_trace(k)],
                 traces=[4, 5, 6, 7], name=str(k))
        for k in range(n_frames)
    ]
    scene = _light_scene([-190, 130], [-130, 130], [0, h])
    scene["aspectmode"] = "manual"
    scene["aspectratio"] = dict(x=1.2, y=0.9, z=1.0)
    scene["camera"] = dict(eye=dict(x=1.5, y=1.5, z=0.9))
    fig.update_layout(
        template="plotly_white", height=640, margin=dict(l=0, r=0, t=35, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0.0, bgcolor="rgba(255,255,255,0.8)"),
        scene=scene,
    )
    return add_animation_controls(fig, frames, [f"{t:.1f} s" for t in sim["t"]], "Tiempo: ")


def build_filtration_charts(sim: Dict[str, Any], horizon_days: int) -> Tuple[go.Figure, go.Figure]:
    """Gráficas 2D: poblaciones en malla (10 s) y crecimiento celular + glucosa (1/3/7 días)."""
    c, t = sim["counts"], sim["t"]
    f1 = go.Figure()
    for i, (name, color) in enumerate(
        [("En suspensión", PASTEL["bulk"]), ("Retenidas (fouling)", PASTEL["fouled"]), ("Permeadas", PASTEL["permeated"])]
    ):
        f1.add_trace(go.Scatter(x=t, y=c[:, i], name=name, mode="lines", line=dict(color=color, width=3)))
    f1.update_layout(
        template="plotly_white", height=300, margin=dict(l=10, r=10, t=35, b=10),
        title="Dinámica en malla (10 s)", xaxis_title="Tiempo (s)", yaxis_title="N.º de células",
        legend=dict(orientation="h", y=-0.32),
    )

    cu = sim["culture"]
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=cu["days"], y=cu["cells_ml"], name="Células (cel/mL)",
                            line=dict(color="#748ffc", width=3)))
    f2.add_trace(go.Scatter(x=cu["days"], y=cu["glucose_gl"], name="Glucosa (g/L)",
                            line=dict(color="#63e6be", width=2.5, dash="dash"), yaxis="y2"))
    f2.update_layout(
        template="plotly_white", height=300, margin=dict(l=10, r=10, t=35, b=10),
        title=f"Crecimiento celular y nutriente ({horizon_days} día(s))",
        xaxis_title="Tiempo de cultivo (días)",
        yaxis=dict(title="Concentración (cel/mL)", type="log"),
        yaxis2=dict(title="Glucosa (g/L)", overlaying="y", side="right", range=[0, 5.2]),
        legend=dict(orientation="h", y=-0.32),
    )
    return f1, f2


def _filtration_velocity_transient(
    r_mm: np.ndarray | float,
    theta: np.ndarray | float,
    z_mm: np.ndarray | float,
    t_s: float,
    vf: Dict[str, float],
    w_vis: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Campo de velocidades transitorio alrededor de la malla rotatoria (v_r, v_z, v_theta, f_arr)."""
    omega = vf["omega"]
    v_suc_ms = vf["v_suc_ms"]
    r_mesh = vf["r_mesh_mm"]
    height = vf["height_mm"]
    r_mesh_m = r_mesh / 1000.0

    r_arr = np.asarray(r_mm, dtype=float)
    th_arr = np.asarray(theta, dtype=float)
    z_arr = np.asarray(z_mm, dtype=float)
    r_safe = np.maximum(r_arr, 1e-6)

    f_arr = float(0.25 + 0.75 * (1.0 - np.exp(-float(t_s) / 1.8)))
    g_rib = 1.0 + 0.10 * np.exp(-((r_arr - r_mesh) / (0.35 * r_mesh)) ** 2) * np.cos(4.0 * (th_arr - w_vis * t_s))

    vt = np.where(
        r_arr >= r_mesh,
        f_arr * g_rib * omega * r_mesh_m * (r_mesh / r_safe) ** 1.5,
        f_arr * omega * (r_arr / 1000.0),
    )
    vr = np.where(
        r_arr >= r_mesh,
        -f_arr * v_suc_ms * np.sqrt(r_mesh / r_safe),
        -f_arr * v_suc_ms * (r_arr / r_mesh),
    )
    vz = np.where(
        r_arr >= r_mesh,
        0.015 * f_arr * np.cos(2.0 * np.pi * z_arr / height),
        -0.025 * f_arr,
    )
    return vr, vz, vt, f_arr


@st.cache_data(show_spinner=False)
def run_filtration_vector_animation(
    vf: Dict[str, float],
    w_vis: float,
    duration_s: float = ANIM_SECONDS,
    n_frames: int = N_FRAMES,
) -> Dict[str, Any]:
    """Integra 128 flechas 3D del campo vectorial alrededor de la malla durante 10 s."""
    r_mesh = vf["r_mesh_mm"]
    r_vessel = vf["r_vessel_mm"]
    height = vf["height_mm"]

    rg, thg, zg = np.meshgrid(
        np.linspace(0.45 * r_mesh, 0.90 * r_vessel, 4),
        np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False),
        np.linspace(0.10 * height, 0.90 * height, 4),
        indexing="ij",
    )
    r_cur = rg.ravel().copy()
    th_cur = thg.ravel().copy()
    z_cur = zg.ravel().copy()

    sub = 3
    dt_anim = duration_s / (n_frames * sub)
    dt_adv = (2.0 / duration_s) * dt_anim

    xs, ys, zs, vxs, vys, vzs, spds, f_arrs, angs = [], [], [], [], [], [], [], [], []
    t_grid = np.linspace(0.0, duration_s, n_frames + 1)

    for k in range(n_frames + 1):
        t_k = float(t_grid[k])
        vr, vz, vt, f_arr = _filtration_velocity_transient(r_cur, th_cur, z_cur, t_k, vf, w_vis)
        xs.append(r_cur * np.cos(th_cur))
        ys.append(r_cur * np.sin(th_cur))
        zs.append(z_cur.copy())
        vxs.append(vr * np.cos(th_cur) - vt * np.sin(th_cur))
        vys.append(vr * np.sin(th_cur) + vt * np.cos(th_cur))
        vzs.append(vz)
        spds.append(np.sqrt(vr**2 + vz**2 + vt**2))
        f_arrs.append(f_arr)
        angs.append(float(w_vis * t_k))

        if k == n_frames:
            break
        for s_idx in range(sub):
            t_sub = t_k + s_idx * dt_anim
            vr1, vz1, vt1, _ = _filtration_velocity_transient(r_cur, th_cur, z_cur, t_sub, vf, w_vis)
            # Convertir velocidades (m/s) a mm/s para posiciones en mm
            rm = np.clip(r_cur + 0.5 * dt_adv * (vr1 * 1000.0), 3.0, 0.96 * r_vessel)
            thm = th_cur + 0.5 * dt_adv * (vt1 * 1000.0) / np.maximum(r_cur, 4.0)
            zm = np.clip(z_cur + 0.5 * dt_adv * (vz1 * 1000.0), 4.0, height - 4.0)

            vr2, vz2, vt2, _ = _filtration_velocity_transient(rm, thm, zm, t_sub + 0.5 * dt_anim, vf, w_vis)
            th_cur = (th_cur + dt_adv * (vt2 * 1000.0) / np.maximum(rm, 4.0)) % (2.0 * np.pi)
            r_cur = np.clip(r_cur + dt_adv * (vr2 * 1000.0), 3.0, 0.96 * r_vessel)
            z_cur = np.clip(z_cur + dt_adv * (vz2 * 1000.0), 4.0, height - 4.0)

            # Reciclar flechas que llegan al fondo del colector interno para mantener poblado el campo
            reseed = (r_cur < 0.35 * r_mesh) & (z_cur <= 8.0)
            if np.any(reseed):
                r_cur[reseed] = 0.88 * r_vessel
                z_cur[reseed] = 0.88 * height

    spd_arr = np.array(spds)
    return {
        "x": np.array(xs),
        "y": np.array(ys),
        "z": np.array(zs),
        "vx": np.array(vxs),
        "vy": np.array(vys),
        "vz": np.array(vzs),
        "speed": spd_arr,
        "t": t_grid,
        "f_arr": np.array(f_arrs),
        "angle_mesh": np.array(angs),
        "vmax_global": float(np.max(spd_arr)) or 1.0,
    }


def build_filtration_vector_3d_figure(
    sim: Dict[str, Any],
    vec_anim: Dict[str, Any],
    paused_frame: int | None = None,
) -> go.Figure:
    """Construye el campo vectorial 3D animado (10 s) o congelado en paused_frame alrededor de la malla."""
    g = sim["geometry"]
    r_mesh, r_vessel, h = g["r_mesh"], g["r_vessel"], g["height"]
    vmax = vec_anim["vmax_global"]
    pastel_vel = [[0.0, "#74c0fc"], [0.35, "#63e6be"], [0.7, "#ffe066"], [1.0, "#ff6b6b"]]

    th = np.linspace(0, 2 * np.pi, 45)
    zz = np.linspace(0, h, 20)
    thm, zm = np.meshgrid(th, zz)

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=r_mesh * np.cos(thm), y=r_mesh * np.sin(thm), z=zm,
            name="Malla rotatoria", showscale=False, opacity=0.30,
            colorscale=[[0, PASTEL["mesh"]], [1, PASTEL["mesh_dark"]]], hoverinfo="name",
        )
    )
    fig.add_trace(
        go.Surface(
            x=r_vessel * np.cos(thm), y=r_vessel * np.sin(thm), z=zm,
            name="Pared del tanque", showscale=False, opacity=0.08,
            colorscale=[[0, PASTEL["wall"]], [1, "#f1f5f9"]], hoverinfo="skip",
        )
    )

    def rib_trace(k: int) -> go.Scatter3d:
        a = vec_anim["angle_mesh"][k]
        xr, yr, zr = [], [], []
        for aa in (a, a + np.pi):
            xr += [r_mesh * np.cos(aa)] * 2 + [None]
            yr += [r_mesh * np.sin(aa)] * 2 + [None]
            zr += [0, h, None]
        return go.Scatter3d(
            x=xr, y=yr, z=zr, mode="lines", name="Marca de giro",
            line=dict(color=PASTEL["rib"], width=7), hoverinfo="skip",
        )

    def arrow_traces(k: int) -> Tuple[go.Scatter3d, go.Cone]:
        return _build_3d_arrow_traces(
            vec_anim["x"][k], vec_anim["y"][k], vec_anim["z"][k],
            vec_anim["vx"][k], vec_anim["vy"][k], vec_anim["vz"][k],
            max_arrow_len=0.25 * r_vessel,
            vmax=vmax,
            colorscale=pastel_vel,
            show_colorbar=True,
            name="Vectores 3D",
            line_width=3.5,
            head_ratio=0.30,
        )

    k0 = 0 if paused_frame is None else int(np.clip(paused_frame, 0, len(vec_anim["t"]) - 1))
    s0, h0 = arrow_traces(k0)
    fig.add_trace(rib_trace(k0))  # 2
    fig.add_trace(s0)             # 3
    fig.add_trace(h0)             # 4

    scene = _light_scene([-r_vessel * 1.08, r_vessel * 1.08], [-r_vessel * 1.08, r_vessel * 1.08], [0, h])
    scene["aspectmode"] = "manual"
    scene["aspectratio"] = dict(x=1.0, y=1.0, z=1.05)
    scene["camera"] = dict(eye=dict(x=1.55, y=1.45, z=0.85))

    if paused_frame is None:
        frames = [
            go.Frame(data=[rib_trace(k), *arrow_traces(k)], traces=[2, 3, 4], name=str(k))
            for k in range(len(vec_anim["t"]))
        ]
        fig.update_layout(
            template="plotly_white", height=480, margin=dict(l=0, r=0, t=40, b=0),
            title="Simulación 3D de Vectores de Velocidad (10 s)", showlegend=False, scene=scene,
        )
        return add_animation_controls(fig, frames, [f"{t:.1f} s" for t in vec_anim["t"]], "Tiempo: ")

    t_k = float(vec_anim["t"][k0])
    fig.update_layout(
        template="plotly_white", height=480, margin=dict(l=0, r=0, t=40, b=0),
        title=f"Vista 3D Pausada — t = {t_k:.1f} s", showlegend=False, scene=scene,
    )
    return fig


def build_filtration_vector_figures(sim: Dict[str, Any], t_s: float = 10.0) -> Tuple[go.Figure, go.Figure]:
    """Construye el corte horizontal 2D del campo vectorial (v_θ + v_r) y perfiles radiales en t = t_s."""
    vf = sim["vel_field"]
    r_mesh = vf["r_mesh_mm"]
    r_vessel = vf["r_vessel_mm"]
    w_vis = sim["w_vis"]
    z_mid = vf["height_mm"] * 0.5

    # 1. Corte horizontal 2D (plano x-y)
    grid = np.linspace(-r_vessel, r_vessel, 61)
    gx, gy = np.meshgrid(grid, grid)
    gr = np.hypot(gx, gy)
    gth = np.arctan2(gy, gx)
    vr_g, vz_g, vt_g, _ = _filtration_velocity_transient(gr, gth, np.full_like(gr, z_mid), t_s, vf, w_vis)
    vmag_g = np.where(gr <= r_vessel, np.sqrt(vt_g**2 + vr_g**2 + vz_g**2), np.nan)

    # Referencia máxima en t = 10 s para que se note la aceleración al mover el slider de tiempo
    vr_ref, vz_ref, vt_ref, _ = _filtration_velocity_transient(gr, gth, np.full_like(gr, z_mid), ANIM_SECONDS, vf, w_vis)
    vmax_ref = float(np.nanmax(np.where(gr <= r_vessel, np.sqrt(vt_ref**2 + vr_ref**2 + vz_ref**2), np.nan))) or 1.0

    pastel_vel = [[0.0, "#e7f5ff"], [0.3, "#74c0fc"], [0.55, "#63e6be"], [0.8, "#ffe066"], [1.0, "#ff6b6b"]]
    fig_xy = go.Figure()
    fig_xy.add_trace(
        go.Heatmap(
            x=grid, y=grid, z=vmag_g,
            zmin=0.0, zmax=vmax_ref * 1.05,
            colorscale=pastel_vel,
            colorbar=dict(title="|v| (m/s)", thickness=14, len=0.82),
            hovertemplate="x=%{x:.1f} mm · y=%{y:.1f} mm<br>|v|=%{z:.3f} m/s<extra></extra>",
        )
    )

    q_pts = np.linspace(-r_vessel * 0.92, r_vessel * 0.92, 15)
    qx_g, qy_g = np.meshgrid(q_pts, q_pts)
    qr_g = np.hypot(qx_g, qy_g)
    q_mask = (qr_g >= 6.0) & (qr_g <= r_vessel * 0.94)
    qx_v, qy_v, qr_v = qx_g[q_mask], qy_g[q_mask], qr_g[q_mask]
    qth_v = np.arctan2(qy_v, qx_v)
    vr_q, _, vt_q, _ = _filtration_velocity_transient(qr_v, qth_v, np.full_like(qr_v, z_mid), t_s, vf, w_vis)
    cos_t = qx_v / qr_v
    sin_t = qy_v / qr_v
    vx_q = vr_q * cos_t - vt_q * sin_t
    vy_q = vr_q * sin_t + vt_q * cos_t

    arrow_x, arrow_y = _quiver_2d_coords(qx_v, qy_v, vx_q, vy_q, max_len=r_vessel * 0.13, ref_mag=vmax_ref)
    fig_xy.add_trace(
        go.Scatter(
            x=arrow_x, y=arrow_y, mode="lines",
            line=dict(color="#1e293b", width=1.6),
            name="Vectores (vx, vy)", hoverinfo="skip",
        )
    )

    th_c = np.linspace(0, 2 * np.pi, 100)
    fig_xy.add_trace(
        go.Scatter(
            x=r_mesh * np.cos(th_c), y=r_mesh * np.sin(th_c), mode="lines",
            line=dict(color=PASTEL["mesh_dark"], width=3), name="Malla rotatoria", hoverinfo="name",
        )
    )
    fig_xy.add_trace(
        go.Scatter(
            x=r_vessel * np.cos(th_c), y=r_vessel * np.sin(th_c), mode="lines",
            line=dict(color="#64748b", width=2), name="Pared tanque", hoverinfo="name",
        )
    )
    fig_xy.update_layout(
        template="plotly_white", height=480, margin=dict(l=10, r=10, t=40, b=15),
        title=f"Corte horizontal 2D (t = {t_s:.1f} s): Vórtice + Succión radial",
        xaxis=dict(title="X (mm)", range=[-r_vessel * 1.06, r_vessel * 1.06]),
        yaxis=dict(title="Y (mm)", range=[-r_vessel * 1.06, r_vessel * 1.06], scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", y=-0.18),
    )

    # 2. Perfiles radiales v_theta(r) y |v_r(r)| en t = t_s
    r_prof = np.linspace(1.0, r_vessel, 120)
    vr_p, _, vt_p, _ = _filtration_velocity_transient(r_prof, np.zeros_like(r_prof), np.full_like(r_prof, z_mid), t_s, vf, w_vis)
    fig_prof = go.Figure()
    fig_prof.add_trace(
        go.Scatter(x=r_prof, y=vt_p, mode="lines", name="Velocidad tangencial v_θ (m/s)",
                   line=dict(color="#748ffc", width=3))
    )
    fig_prof.add_trace(
        go.Scatter(x=r_prof, y=np.abs(vr_p), mode="lines", name="Velocidad de succión |v_r| (m/s)",
                   line=dict(color="#ff8787", width=3, dash="dash"))
    )
    fig_prof.add_vline(x=r_mesh, line=dict(color=PASTEL["mesh_dark"], width=2, dash="dot"),
                       annotation_text="Malla rotatoria (R_m)")
    fig_prof.update_layout(
        template="plotly_white", height=320, margin=dict(l=10, r=10, t=35, b=10),
        title=f"Perfiles radiales de velocidad (t = {t_s:.1f} s)",
        xaxis_title="Radio r (mm)", yaxis_title="Velocidad (m/s)",
        legend=dict(orientation="h", y=-0.28),
    )
    return fig_xy, fig_prof


@st.cache_data(show_spinner=False)
def generate_filtration_snapshot_png(
    sim: Dict[str, Any],
    params: Dict[str, Any],
    vec_anim: Dict[str, Any],
    frame_idx: int,
) -> bytes:
    """Genera la ficha técnica PNG en alta resolución (3D + 2D + parámetros + ecuación vectorial)."""
    k_idx = int(np.clip(frame_idx, 0, len(vec_anim["t"]) - 1))
    t_k = float(vec_anim["t"][k_idx])
    f_arr_k = float(vec_anim["f_arr"][k_idx])
    vf = sim["vel_field"]
    kpis = sim["kpis"]
    r_mesh, r_vessel, h_mm = vf["r_mesh_mm"], vf["r_vessel_mm"], vf["height_mm"]
    vmax_3d = vec_anim["vmax_global"]

    pastel_cmap = LinearSegmentedColormap.from_list(
        "pastel_vel_3d", ["#74c0fc", "#63e6be", "#ffe066", "#ff6b6b"]
    )
    pastel_cmap_2d = LinearSegmentedColormap.from_list(
        "pastel_vel_2d", ["#e7f5ff", "#74c0fc", "#63e6be", "#ffe066", "#ff6b6b"]
    )

    fig = plt.figure(figsize=(14.5, 9.6), dpi=150, facecolor="#fcfcfd")
    try:
        gs = GridSpec(3, 2, figure=fig, height_ratios=[0.11, 0.56, 0.33], hspace=0.28, wspace=0.18)

        # Banda superior
        ax_head = fig.add_subplot(gs[0, :])
        ax_head.axis("off")
        ax_head.add_patch(
            plt.Rectangle((0.0, 0.05), 1.0, 0.90, facecolor="#edf2ff", edgecolor="#bac8ff", linewidth=1.5, transform=ax_head.transAxes)
        )
        ax_head.text(
            0.02, 0.64,
            "REPORTE INSTANTÁNEO — CAMPO VECTORIAL DE VELOCIDADES (FILTRACIÓN ROTATORIA)",
            fontsize=13, fontweight="bold", color="#1e293b", va="center", transform=ax_head.transAxes,
        )
        ax_head.text(
            0.02, 0.25,
            f"Línea: {params['cell_line']}  |  Malla: {params['mesh_material']}  |  "
            f"Filtro: {params['rpm']} RPM  |  Perfusión: {params['perfusion_vvd']:.1f} VVD  |  Instante: t = {t_k:.1f} s / 10.0 s",
            fontsize=9.6, color="#334155", va="center", transform=ax_head.transAxes,
        )

        # Panel 3D
        ax3d = fig.add_subplot(gs[1, 0], projection="3d", facecolor="#f8fafc")
        th_c = np.linspace(0, 2 * np.pi, 40)
        z_c = np.linspace(0, h_mm, 12)
        thm, zm = np.meshgrid(th_c, z_c)
        ax3d.plot_surface(r_mesh * np.cos(thm), r_mesh * np.sin(thm), zm, color="#74c0fc", alpha=0.22, edgecolor="none")
        ax3d.plot_wireframe(r_vessel * np.cos(thm), r_vessel * np.sin(thm), zm, color="#cbd5e1", alpha=0.35, rstride=3, cstride=4, linewidth=0.6)
        a_m = float(vec_anim["angle_mesh"][k_idx])
        for aa in (a_m, a_m + np.pi):
            ax3d.plot([r_mesh * np.cos(aa), r_mesh * np.cos(aa)], [r_mesh * np.sin(aa), r_mesh * np.sin(aa)], [0, h_mm], color="#f06595", linewidth=2.5)

        xk, yk, zk = vec_anim["x"][k_idx], vec_anim["y"][k_idx], vec_anim["z"][k_idx]
        ux, uy, uz = vec_anim["vx"][k_idx], vec_anim["vy"][k_idx], vec_anim["vz"][k_idx]
        spdk = vec_anim["speed"][k_idx]
        scale_3d = (0.22 * r_vessel) / max(vmax_3d, 1e-6)
        n_bins = 8
        edges = np.linspace(0.0, vmax_3d + 1e-9, n_bins + 1)
        for b in range(n_bins):
            m = (spdk >= edges[b]) & (spdk <= edges[b + 1])
            if np.any(m):
                c_val = pastel_cmap((b + 0.5) / n_bins)
                ax3d.quiver(
                    xk[m], yk[m], zk[m], ux[m] * scale_3d, uy[m] * scale_3d, uz[m] * scale_3d,
                    color=c_val, arrow_length_ratio=0.32, linewidth=1.2, normalize=False,
                )
        ax3d.set_xlim(-r_vessel * 1.05, r_vessel * 1.05)
        ax3d.set_ylim(-r_vessel * 1.05, r_vessel * 1.05)
        ax3d.set_zlim(0, h_mm)
        ax3d.set_xlabel("X (mm)", fontsize=8.5)
        ax3d.set_ylabel("Y (mm)", fontsize=8.5)
        ax3d.set_zlabel("Z (mm)", fontsize=8.5)
        ax3d.set_title(f"Campo Vectorial 3D — t = {t_k:.1f} s", fontsize=11, fontweight="bold", color="#1e293b", pad=8)
        sm3d = ScalarMappable(norm=Normalize(0.0, vmax_3d), cmap=pastel_cmap)
        sm3d.set_array([])
        cb3d = fig.colorbar(sm3d, ax=ax3d, shrink=0.68, pad=0.14)
        cb3d.set_label("|v| (m/s)", fontsize=8.5)

        # Panel 2D
        ax2d = fig.add_subplot(gs[1, 1], facecolor="#f8fafc")
        z_mid = h_mm * 0.5
        g2 = np.linspace(-r_vessel, r_vessel, 81)
        gx, gy = np.meshgrid(g2, g2)
        gr = np.hypot(gx, gy)
        gth = np.arctan2(gy, gx)
        vr_g, vz_g, vt_g, _ = _filtration_velocity_transient(gr, gth, np.full_like(gr, z_mid), t_k, vf, sim["w_vis"])
        vmag_g = np.where(gr <= r_vessel, np.sqrt(vt_g**2 + vr_g**2 + vz_g**2), np.nan)
        vr_r, vz_r, vt_r, _ = _filtration_velocity_transient(gr, gth, np.full_like(gr, z_mid), ANIM_SECONDS, vf, sim["w_vis"])
        vmax_2d = float(np.nanmax(np.where(gr <= r_vessel, np.sqrt(vt_r**2 + vr_r**2 + vz_r**2), np.nan))) or 1.0

        im = ax2d.imshow(
            vmag_g, extent=[-r_vessel, r_vessel, -r_vessel, r_vessel],
            origin="lower", cmap=pastel_cmap_2d, vmin=0.0, vmax=vmax_2d * 1.05,
        )
        qp = np.linspace(-r_vessel * 0.90, r_vessel * 0.90, 14)
        qx, qy = np.meshgrid(qp, qp)
        qr = np.hypot(qx, qy)
        qm = (qr >= 6.0) & (qr <= r_vessel * 0.93)
        vr_q, _, vt_q, _ = _filtration_velocity_transient(qr[qm], np.arctan2(qy[qm], qx[qm]), np.full_like(qr[qm], z_mid), t_k, vf, sim["w_vis"])
        cos_t = qx[qm] / qr[qm]
        sin_t = qy[qm] / qr[qm]
        ax2d.quiver(
            qx[qm], qy[qm], vr_q * cos_t - vt_q * sin_t, vr_q * sin_t + vt_q * cos_t,
            color="#1e293b", scale=vmax_2d * 10.5, width=0.004,
        )
        th_ring = np.linspace(0, 2 * np.pi, 120)
        ax2d.plot(r_mesh * np.cos(th_ring), r_mesh * np.sin(th_ring), color="#4dabf7", linewidth=2.2, label="Malla")
        ax2d.plot(r_vessel * np.cos(th_ring), r_vessel * np.sin(th_ring), color="#64748b", linewidth=1.6, label="Tanque")
        ax2d.set_aspect("equal")
        ax2d.set_xlim(-r_vessel * 1.06, r_vessel * 1.06)
        ax2d.set_ylim(-r_vessel * 1.06, r_vessel * 1.06)
        ax2d.set_xlabel("X (mm)", fontsize=8.5)
        ax2d.set_ylabel("Y (mm)", fontsize=8.5)
        ax2d.set_title(f"Corte Horizontal 2D — t = {t_k:.1f} s", fontsize=11, fontweight="bold", color="#1e293b", pad=8)
        cb2d = fig.colorbar(im, ax=ax2d, shrink=0.78, pad=0.04)
        cb2d.set_label("|v| (m/s)", fontsize=8.5)

        # Tarjeta inferior izquierda: Parámetros y KPIs
        ax_p = fig.add_subplot(gs[2, 0])
        ax_p.axis("off")
        ax_p.add_patch(
            plt.Rectangle((0.0, 0.0), 1.0, 1.0, facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=1.2, transform=ax_p.transAxes)
        )
        ax_p.text(0.03, 0.88, "PARÁMETROS DE CONFIGURACIÓN Y RESULTADOS", fontsize=10, fontweight="bold", color="#1e293b", transform=ax_p.transAxes)
        param_lines = [
            f"• Línea celular: {params['cell_line']}",
            f"• Material malla: {params['mesh_material']}",
            f"• Velocidad filtro: {params['rpm']} RPM   |   Poro / Célula: {params['pore_size_um']:.1f} / {params['cell_diameter_um']:.1f} µm",
            f"• Perfusión: {params['perfusion_vvd']:.1f} VVD   |   Retrolavado: {'Sí' if params['backflush'] else 'No'}",
            f"• Retención: {kpis['retention_efficiency']:.1f}%   |   Fouling: {kpis['fouling_index']:.0f}%   |   Corte τ_max: {kpis['shear_stress_pa']:.2f} Pa",
            f"• Re rotacional: {kpis['re_rot']:,.0f}   |   |v| máx en t={t_k:.1f} s: {float(np.max(spdk)):.3f} m/s",
        ]
        ax_p.text(0.03, 0.77, "\n".join(param_lines), fontsize=8.8, color="#334155", va="top", linespacing=1.48, transform=ax_p.transAxes)

        # Tarjeta inferior derecha: Ecuación Vectorial
        ax_eq = fig.add_subplot(gs[2, 1])
        ax_eq.axis("off")
        ax_eq.add_patch(
            plt.Rectangle((0.0, 0.0), 1.0, 1.0, facecolor="#f8fafc", edgecolor="#748ffc", linewidth=1.4, transform=ax_eq.transAxes)
        )
        ax_eq.text(0.03, 0.88, "ECUACIÓN VECTORIAL DEL CAMPO DE VELOCIDADES", fontsize=10, fontweight="bold", color="#1e293b", transform=ax_eq.transAxes)
        eq_lines = [
            r"$\mathbf{v}(r,\theta,z,t) = v_r\,\hat{e}_r + v_\theta\,\hat{e}_\theta + v_z\,\hat{e}_z$",
            r"$v_x = v_r\cos\theta - v_\theta\sin\theta,\quad v_y = v_r\sin\theta + v_\theta\cos\theta$",
            r"$v_\theta = f(t)\,g(\theta,t)\,\omega R_m (R_m/r)^{1.5}\ (r \geq R_m),\quad v_r = -f(t)\,v_{\mathrm{suc}}\sqrt{R_m/r}$",
            r"$f(t) = 0.25 + 0.75(1 - e^{-t/1.8}) = %.3f$" % (f_arr_k,),
            r"$\omega = %.2f\ \mathrm{rad/s},\quad R_m = %.0f\ \mathrm{mm},\quad v_{\mathrm{suc}} = %.4f\ \mathrm{m/s}$"
            % (vf["omega"], r_mesh, vf["v_suc_ms"]),
        ]
        y_pos = 0.76
        for line in eq_lines:
            ax_eq.text(0.03, y_pos, line, fontsize=8.2, color="#1e293b", va="top", transform=ax_eq.transAxes)
            y_pos -= 0.15

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        return buf.getvalue()
    finally:
        plt.close(fig)


def filtration_report(p: Dict[str, Any], sim: Dict[str, Any]) -> str:
    k = sim["kpis"]
    cinfo = CELL_LINES[p["cell_line"]]
    diags = [
        (f"⚠️ **Alerta de corte:** τ = {k['shear_stress_pa']:.2f} Pa supera el límite de {p['cell_line']} ({k['shear_limit_pa']} Pa)."
         if k["shear_stress_pa"] > k["shear_limit_pa"] else f"✅ **Corte seguro:** τ = {k['shear_stress_pa']:.2f} Pa (límite {k['shear_limit_pa']} Pa)."),
        (f"⚠️ **Fouling alto ({k['fouling_index']:.0f} %):** aumente RPM o active retrolavado."
         if k["fouling_index"] > 45 else f"✅ **Fouling controlado ({k['fouling_index']:.0f} %).**"),
        (f"⚠️ **Pérdida celular:** retención {k['retention_efficiency']:.1f} %; reduzca el tamaño de poro."
         if k["retention_efficiency"] < 85 else f"✅ **Retención adecuada ({k['retention_efficiency']:.1f} %).**"),
    ]
    return make_report(
        "Filtración Rotatoria y Cultivo en Perfusión", "Filtración Rotatoria 3D",
        [("Línea celular", p["cell_line"], "-"),
         ("Tipo de malla / polímero", p["mesh_material"], "-"),
         ("Velocidad del filtro", f"{p['rpm']}", "RPM"),
         ("Tamaño de poro / Diámetro celular", f"{p['pore_size_um']:.1f} / {p['cell_diameter_um']:.1f}", "µm"),
         ("Horizonte de cultivo / Inóculo", f"{p['horizon_days']} días / {p['seed_density_ml']:,.0f}", "cel/mL"),
         ("Tasa de perfusión", f"{p['perfusion_vvd']:.1f}", "VVD (vol/día)"),
         ("Retrolavado (Backflush)", "Activo" if p["backflush"] else "Inactivo", "-")],
        [("Eficiencia de retención", f"{k['retention_efficiency']:.1f}", "%"),
         ("Índice de fouling", f"{k['fouling_index']:.0f}", "%"),
         ("Esfuerzo de corte", f"{k['shear_stress_pa']:.2f}", "Pa"),
         ("Densidad celular final", f"{k['final_cells_ml']:.2e}", "cel/mL"),
         ("Producto estimado", f"{k['protein_mgl']:.1f}", "mg/L"),
         ("Producto biológico objetivo", cinfo["product"], "-")],
        r"""1. **Velocidad tangencial y corte:** $v_{tip}=\omega R_m$, $\;\tau_w=\mu v_{tip}/\delta_{BL}$
2. **Crecimiento en perfusión (Monod):** $\frac{dX}{dt} = \mu_{max}\frac{S}{K_s+S}\left(1-\frac{X}{X_{cap}}\right)X - D_{bleed}X$""",
        diags,
    )


def render_filtration_tab() -> None:
    p = _form_defaults("filt_params", FILT_DEFAULTS)
    col_c, col_v = st.columns([1.05, 2.55], gap="large")

    with col_c:
        st.subheader("⚙️ Parámetros de Filtración y Cultivo")
        # Selector rápido fuera del form para mostrar ficha biológica inmediata
        cell_names = list(CELL_LINES)
        mesh_names = list(MESH_MATERIALS)

        with st.form("filt_form"):
            cell_line = st.selectbox("Línea celular", cell_names, index=cell_names.index(p["cell_line"]), key="f_cl")
            mesh_mat = st.selectbox("Material de malla / polímero", mesh_names, index=mesh_names.index(p["mesh_material"]), key="f_mm")
            use_preset_diam = st.checkbox("Usar diámetro típico de la línea celular elegida", value=True, key="f_pdiam")

            rpm = st.slider("Velocidad del filtro (RPM)", 10, 600, int(p["rpm"]), 10, key="f_rpm")
            pore = st.slider("Tamaño de poro (µm)", 0.5, 40.0, float(p["pore_size_um"]), 0.5, key="f_pore")
            cdiam = st.slider("Diámetro celular manual (µm)", 1.0, 30.0, float(p["cell_diameter_um"]), 0.5, key="f_cd")
            dens = st.slider("Densidad actual en malla 3D (×10⁶ cel/mL)", 0.5, 50.0, float(p["cell_density_ml"] / 1e6), 0.5, key="f_dens")

            st.markdown("**Simulación de Crecimiento y Perfusión**")
            horizon = st.radio("Duración del cultivo", [1, 3, 7], index=[1, 3, 7].index(p["horizon_days"]),
                               format_func=lambda d: f"{d} día{'s' if d > 1 else ''}", horizontal=True, key="f_hor")
            seed_d = st.select_slider("Inóculo inicial (cel/mL)", options=[1000, 10000, 100000, 500000, 1000000],
                                      value=int(p["seed_density_ml"]), key="f_seed")
            vvd = st.slider("Tasa de perfusión diaria (VVD)", 0.2, 4.0, float(p["perfusion_vvd"]), 0.2, key="f_vvd",
                            help="Volúmenes de reactor renovados al día.")
            backflush = st.checkbox("Activar pulso de retrolavado (liberar células de malla)", value=bool(p["backflush"]), key="f_bf")

            if st.form_submit_button("🚀 Correr Simulación", width="stretch", type="primary"):
                final_diam = CELL_LINES[cell_line]["diam_um"] if use_preset_diam else cdiam
                st.session_state["filt_params"] = dict(
                    cell_line=cell_line, mesh_material=mesh_mat, rpm=rpm, pore_size_um=pore,
                    cell_diameter_um=final_diam, cell_density_ml=dens * 1e6, horizon_days=horizon,
                    seed_density_ml=float(seed_d), perfusion_vvd=vvd, backflush=backflush,
                )

        p = st.session_state["filt_params"]
        sim = run_filtration_solver(**p)
        cinfo = CELL_LINES[p["cell_line"]]
        minfo = MESH_MATERIALS[p["mesh_material"]]
        with st.expander("🧬 Ficha rápida: Línea celular y malla seleccionadas"):
            st.markdown(
                f"- **Forma / Masa:** {cinfo['shape']} · `{cinfo['mass_ng']}` ng/célula\n"
                f"- **Producto:** {cinfo['product']}\n"
                f"- **Medio:** {cinfo['medium']}\n"
                f"- **Malla:** Área abierta `{int(minfo['open_area']*100)}%` · Autoclave: {minfo['autoclave']}"
            )
        st.download_button("📥 Descargar Memoria de Cálculo (.md)", filtration_report(p, sim).encode("utf-8"),
                           file_name=f"memoria_filtracion_{p['rpm']}rpm.md", mime="text/markdown",
                           width="stretch", key="dl_filt")

    k = sim["kpis"]
    with col_v:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Retención final", f"{k['retention_efficiency']:.1f} %",
                  "Óptima" if k["retention_efficiency"] >= 85 else "Pierde células",
                  delta_color="normal" if k["retention_efficiency"] >= 85 else "inverse")
        c2.metric("Fouling final", f"{k['fouling_index']:.0f} %",
                  "Colmatación" if k["fouling_index"] > 45 else "Barrido estable",
                  delta_color="inverse" if k["fouling_index"] > 45 else "normal")
        c3.metric("Esfuerzo de corte", f"{k['shear_stress_pa']:.2f} Pa",
                  "Riesgo celular" if k["shear_stress_pa"] > k["shear_limit_pa"] else "Biocompatible",
                  delta_color="inverse" if k["shear_stress_pa"] > k["shear_limit_pa"] else "normal")
        c4.metric(f"Cultivo ({p['horizon_days']} d)", f"{k['final_cells_ml']:.1e}",
                  f"~{k['protein_mgl']:.0f} mg/L prod.", delta_color="off")

        show_wall = st.checkbox("Mostrar pared del tanque", True, key="f_wall")
        st.plotly_chart(build_filtration_figure(sim, p, show_wall), width="stretch", key="filt_3d",
                        config={"displaylogo": False})
        ca, cb = st.columns(2)
        f1, f2 = build_filtration_charts(sim, p["horizon_days"])
        ca.plotly_chart(f1, width="stretch", key="filt_counts", config={"displaylogo": False})
        cb.plotly_chart(f2, width="stretch", key="filt_growth", config={"displaylogo": False})

        st.divider()
        st.subheader("🧭 Campo Vectorial Fluidodinámico alrededor de la Malla")
        st.caption(
            "Simulación animada 3D (10 s) de las flechas de velocidad del vórtice de Taylor-Couette ($v_\\theta$) "
            "acoplado con la succión radial ($v_r$) y el corte horizontal 2D."
        )
        st.session_state.setdefault("filt_vec_paused", False)
        filt_vec_anim = run_filtration_vector_animation(sim["vel_field"], sim["w_vis"])

        if not st.session_state["filt_vec_paused"]:
            if st.button("⏸ Pausar en un segundo y Descargar Imagen (.png)", key="btn_filt_pause", type="secondary"):
                st.session_state["filt_vec_paused"] = True
                st.rerun()
            fv1, fv2 = st.columns([1.05, 1], gap="medium")
            fv1.plotly_chart(
                build_filtration_vector_3d_figure(sim, filt_vec_anim, paused_frame=None),
                width="stretch", key="filt_vec_3d", config={"displaylogo": False},
            )
            fig_fxy, fig_fprof = build_filtration_vector_figures(sim, t_s=ANIM_SECONDS)
            fv2.plotly_chart(fig_fxy, width="stretch", key="filt_vec_xy", config={"displaylogo": False})
            st.plotly_chart(fig_fprof, width="stretch", key="filt_vec_prof", config={"displaylogo": False})
        else:
            pbar1, pbar2 = st.columns([1, 2.2])
            with pbar1:
                if st.button("▶ Reanudar Simulación 3D (10 s)", key="btn_filt_resume", type="primary"):
                    st.session_state["filt_vec_paused"] = False
                    st.rerun()
            with pbar2:
                filt_t = st.slider(
                    "Instante pausado t (s)",
                    min_value=0.0,
                    max_value=float(ANIM_SECONDS),
                    value=4.0,
                    step=0.2,
                    key="filt_pause_t",
                )
            f_idx = int(np.clip(round(filt_t / (ANIM_SECONDS / N_FRAMES)), 0, N_FRAMES))
            fv1, fv2 = st.columns([1.05, 1], gap="medium")
            fv1.plotly_chart(
                build_filtration_vector_3d_figure(sim, filt_vec_anim, paused_frame=f_idx),
                width="stretch", key="filt_vec_3d", config={"displaylogo": False},
            )
            fig_fxy, fig_fprof = build_filtration_vector_figures(sim, t_s=filt_t)
            fv2.plotly_chart(fig_fxy, width="stretch", key="filt_vec_xy", config={"displaylogo": False})
            st.plotly_chart(fig_fprof, width="stretch", key="filt_vec_prof", config={"displaylogo": False})

            st.markdown("**Ecuación Vectorial en el instante pausado:**")
            st.latex(
                r"\mathbf{v}(r,\theta,z,t) = v_r\,\hat{e}_r + v_\theta\,\hat{e}_\theta + v_z\,\hat{e}_z,\qquad "
                r"v_\theta(r,t) = f(t)\,g(\theta,t)\,\omega R_m\left(\frac{R_m}{r}\right)^{1.5},\quad "
                r"v_r(r,t) = -f(t)\,v_{\mathrm{suc}}\sqrt{\frac{R_m}{r}}"
            )
            filt_png = generate_filtration_snapshot_png(sim, p, filt_vec_anim, f_idx)
            st.download_button(
                label=f"📥 Descargar Imagen de Velocidades en Malla (.png) — t = {filt_t:.1f} s",
                data=filt_png,
                file_name=f"campo_velocidades_filtracion_{p['rpm']}rpm_t{filt_t:.1f}s.png",
                mime="image/png",
                type="primary",
                width="stretch",
                key="dl_filt_vec_png",
            )
            with st.expander("👁️ Vista previa de la imagen PNG descargable"):
                st.image(filt_png, width="stretch")


# =====================================================================
# 4. MÓDULO 2 — TRANSFERENCIA DE CALOR 3D + VISCOSIDAD(T) + POLÍMEROS
# =====================================================================
THERM_DEFAULTS = dict(
    t_init_c=25.0, t_jacket_c=39.0, u_wm2k=150.0, rpm=100,
    cell_density_ml=20.0e6, q_cell_pw=90.0, sim_minutes=40,
    polymer_check="PCL (Policaprolactona — Biopolímero)",
)


def viscosity_vs_temp_mpas(temp_c: np.ndarray | float, mu_37: float = 0.75) -> np.ndarray | float:
    """Ley de Andrade / Arrhenius para la viscosidad de medio acuoso (mPa·s) vs temperatura (°C)."""
    t_k = np.asarray(temp_c) + 273.15
    return mu_37 * np.exp(1950.0 * (1.0 / t_k - 1.0 / 310.15))


@st.cache_data(show_spinner=False)
def run_thermal_solver(
    t_init_c: float, t_jacket_c: float, u_wm2k: float, rpm: float, cell_density_ml: float,
    q_cell_pw: float, sim_minutes: float, polymer_check: str, n_frames: int = N_FRAMES,
) -> Dict[str, Any]:
    """Resuelve T(r, z, t) y calcula el efecto de la temperatura sobre la viscosidad del caldo."""
    R, H = 0.10, 0.20
    nr, nz = 12, 24
    dr, dz = R / nr, H / nz
    rc = (np.arange(nr) + 0.5) * dr
    zc = (np.arange(nz) + 0.5) * dz
    rf = np.arange(nr + 1) * dr
    rho_cp = 4.0e6

    # -----------------------------------------------------------------
    # >>> INYECTAR ECUACIONES PROPIAS AQUÍ <<<
    # -----------------------------------------------------------------
    alpha = 1.4e-7 + 4.0e-6 * (rpm / 100.0)
    q_src = cell_density_ml * 1e6 * q_cell_pw * 1e-12 / rho_cp
    sim_s = sim_minutes * 60.0
    dt = min(2.0, 0.2 * dr * dr / alpha)
    n_steps = int(np.ceil(sim_s / dt))
    dt = sim_s / n_steps
    snap_steps = set(np.linspace(0, n_steps, n_frames + 1).round().astype(int).tolist())

    T = np.full((nr, nz), float(t_init_c))
    w = rc[:, None] * np.ones((1, nz))
    vol = np.pi * R**2 * H
    frames, tm, tmax, tmin, qj = [], [], [], [], []
    for step in range(n_steps + 1):
        if step in snap_steps:
            frames.append(T.copy())
            tm.append(float((T * w).sum() / w.sum()))
            tmax.append(float(T.max()))
            tmin.append(float(T.min()))
            qj.append(float(u_wm2k * 2 * np.pi * R * H * (T[-1].mean() - t_jacket_c)))
        if step == n_steps:
            break
        flux = np.zeros((nr + 1, nz))
        flux[1:nr] = alpha * (T[1:] - T[:-1]) / dr * rf[1:nr, None]
        flux[nr] = -(u_wm2k / rho_cp) * (T[-1] - t_jacket_c) * R
        d_r = (flux[1:] - flux[:-1]) / (rc[:, None] * dr)
        tp = np.pad(T, ((0, 0), (1, 1)), mode="edge")
        d_z = alpha * (tp[:, 2:] - 2 * T + tp[:, :-2]) / dz**2
        T = T + dt * (d_r + d_z + q_src)
    # -----------------------------------------------------------------

    frames_arr = np.array(frames)
    times = np.linspace(0, sim_minutes, len(frames))
    tm_a = np.array(tm)
    mu_t = viscosity_vs_temp_mpas(tm_a)

    lo = min(t_init_c, t_jacket_c, frames_arr.min()) - 0.5
    hi = max(t_init_c, t_jacket_c, frames_arr.max()) + 0.5
    if hi - lo < 2.0:
        lo, hi = lo - 1.0, hi + 1.0

    return {
        "frames": frames_arr, "times": times, "tmean": tm_a, "tmax": np.array(tmax), "tmin": np.array(tmin),
        "mu_mpas": mu_t, "qjacket": np.array(qj), "rc": rc, "zc": zc, "R": R, "H": H,
        "cmin": float(lo), "cmax": float(hi),
        "kpis": dict(
            t_mean=float(tm_a[-1]), t_max=float(tmax[-1]), t_min=float(tmin[-1]),
            delta_t=float(tmax[-1] - tmin[-1]), q_met_w=float(q_src * rho_cp * vol),
            mu_init=float(mu_t[0]), mu_final=float(mu_t[-1]), alpha=float(alpha),
        ),
    }


def _ext(T: np.ndarray) -> np.ndarray:
    Te = np.concatenate([T[:1], T, T[-1:]], axis=0)
    return np.concatenate([Te[:, :1], Te, Te[:, -1:]], axis=1)


def build_thermal_figure(sim: Dict[str, Any], params: Dict[str, Any]) -> go.Figure:
    """Corte axial 3D con escala pastel térmica sobre fondo blanco."""
    R, H = sim["R"] * 100, sim["H"] * 100
    rp = np.concatenate([[0], sim["rc"] * 100, [R]])
    zp = np.concatenate([[0], sim["zc"] * 100, [H]])
    y = np.concatenate([-rp[::-1][:-1], rp])

    def section(T: np.ndarray) -> np.ndarray:
        Te = _ext(T)
        return np.concatenate([Te[::-1][:-1], Te], axis=0).T

    th = np.linspace(np.pi / 2, 3 * np.pi / 2, 41)

    def wall(T: np.ndarray) -> np.ndarray:
        return np.tile(_ext(T)[-1, :][:, None], (1, len(th)))

    T0 = sim["frames"][0]
    pastel_thermal = [[0.0, "#74c0fc"], [0.45, "#a5d8ff"], [0.65, "#ffe066"], [1.0, "#ff8787"]]

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=np.zeros((len(zp), len(y))), y=np.tile(y, (len(zp), 1)), z=np.tile(zp[:, None], (1, len(y))),
        surfacecolor=section(T0), coloraxis="coloraxis", name="Corte axial",
        hovertemplate="T=%{surfacecolor:.2f} °C<extra>Corte</extra>"))
    fig.add_trace(go.Surface(
        x=R * np.cos(th)[None, :] * np.ones((len(zp), 1)), y=R * np.sin(th)[None, :] * np.ones((len(zp), 1)),
        z=np.tile(zp[:, None], (1, len(th))), surfacecolor=wall(T0), coloraxis="coloraxis", opacity=0.62,
        name="Pared interna", hovertemplate="T pared=%{surfacecolor:.2f} °C<extra></extra>"))

    tt = np.linspace(0, 2 * np.pi, 80)
    jx, jy, jz = [], [], []
    for zl in np.linspace(0, H, 7):
        jx += list(R * 1.07 * np.cos(tt)) + [None]
        jy += list(R * 1.07 * np.sin(tt)) + [None]
        jz += [zl] * len(tt) + [None]
    fig.add_trace(go.Scatter3d(
        x=jx, y=jy, z=jz, mode="lines", name=f"Chaqueta {params['t_jacket_c']:.0f} °C",
        line=dict(color=np.full(len(jx), params["t_jacket_c"]), coloraxis="coloraxis", width=8), hoverinfo="name"))

    frames = [
        go.Frame(data=[go.Surface(surfacecolor=section(Tk)), go.Surface(surfacecolor=wall(Tk))],
                 traces=[0, 1], name=str(k))
        for k, Tk in enumerate(sim["frames"])
    ]
    scene = _light_scene([-R * 1.2, R * 1.2], [-R * 1.2, R * 1.2], [0, H], z_title="z (cm)", xy_unit="cm")
    scene["aspectmode"] = "data"
    scene["camera"] = dict(eye=dict(x=2.0, y=1.1, z=0.8))
    fig.update_layout(
        template="plotly_white", height=620, margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", y=1.02, bgcolor="rgba(255,255,255,0.8)"),
        coloraxis=dict(colorscale=pastel_thermal, cmin=sim["cmin"], cmax=sim["cmax"],
                       colorbar=dict(title="T (°C)", thickness=15, len=0.7)),
        scene=scene,
    )
    return add_animation_controls(fig, frames, [f"{t:.1f} min" for t in sim["times"]], "Tiempo: ")


def build_thermal_charts(sim: Dict[str, Any], params: Dict[str, Any]) -> Tuple[go.Figure, go.Figure]:
    t = sim["times"]
    f1 = go.Figure()
    f1.add_trace(go.Scatter(x=t, y=sim["tmax"], name="T máx", line=dict(color="#ff8787", width=2, dash="dot")))
    f1.add_trace(go.Scatter(x=t, y=sim["tmean"], name="T media", line=dict(color="#ffa94d", width=4)))
    f1.add_trace(go.Scatter(x=t, y=sim["tmin"], name="T mín", line=dict(color="#74c0fc", width=2, dash="dot")))
    f1.add_hline(y=37.0, line=dict(color="#20c997", dash="dash"), annotation_text="Óptimo 37 °C")
    f1.update_layout(template="plotly_white", height=300, margin=dict(l=10, r=10, t=35, b=10),
                     title="Temperatura vs tiempo", xaxis_title="Tiempo (min)", yaxis_title="T (°C)",
                     legend=dict(orientation="h", y=-0.3))

    t_range = np.linspace(5.0, 50.0, 60)
    mu_curve = viscosity_vs_temp_mpas(t_range)
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=t_range, y=mu_curve, name="Viscosidad μ(T)", line=dict(color="#748ffc", width=3)))
    f2.add_trace(go.Scatter(x=[sim["tmean"][-1]], y=[sim["mu_mpas"][-1]], mode="markers", name="Estado actual",
                            marker=dict(size=12, color="#ff6b6b")))
    f2.update_layout(template="plotly_white", height=300, margin=dict(l=10, r=10, t=35, b=10),
                     title="Efecto de Temperatura en Viscosidad μ(T)", xaxis_title="Temperatura (°C)",
                     yaxis_title="Viscosidad (mPa·s)", legend=dict(orientation="h", y=-0.3))
    return f1, f2


def render_thermal_tab() -> None:
    p = _form_defaults("therm_params", THERM_DEFAULTS)
    col_c, col_v = st.columns([1.05, 2.55], gap="large")
    with col_c:
        st.subheader("⚙️ Parámetros Térmicos")
        poly_names = list(MESH_MATERIALS)
        with st.form("therm_form"):
            ti = st.slider("T inicial del medio (°C)", 4.0, 45.0, float(p["t_init_c"]), 0.5, key="t_ti")
            tj = st.slider("T de la chaqueta (°C)", 4.0, 50.0, float(p["t_jacket_c"]), 0.5, key="t_tj")
            u = st.slider("Coef. global U (W/m²·K)", 20.0, 600.0, float(p["u_wm2k"]), 10.0, key="t_u")
            rpm = st.slider("Agitación (RPM)", 20, 600, int(p["rpm"]), 10, key="t_rpm")
            dens = st.slider("Densidad celular (×10⁶ cel/mL)", 1.0, 100.0, float(p["cell_density_ml"] / 1e6), 1.0, key="t_dens")
            q = st.slider("Calor metabólico (pW/célula)", 10.0, 300.0, float(p["q_cell_pw"]), 10.0, key="t_q")
            mins = st.slider("Tiempo simulado (min)", 5, 120, int(p["sim_minutes"]), 5, key="t_min")
            poly = st.selectbox("Polímero / Malla interna (verificar Tg y Tm)", poly_names,
                                index=poly_names.index(p["polymer_check"]), key="t_poly")
            if st.form_submit_button("🚀 Correr Simulación", width="stretch", type="primary"):
                st.session_state["therm_params"] = dict(
                    t_init_c=ti, t_jacket_c=tj, u_wm2k=u, rpm=rpm, cell_density_ml=dens * 1e6,
                    q_cell_pw=q, sim_minutes=mins, polymer_check=poly,
                )
        p = st.session_state["therm_params"]
        sim = run_thermal_solver(**p)
        minfo = MESH_MATERIALS[p["polymer_check"]]
        if minfo["tg_c"] is not None:
            st.info(f"**{p['polymer_check']}:** $T_g = {minfo['tg_c']}\\text{{ °C}}$ · $T_m = {minfo['tm_c']}\\text{{ °C}}$ · Autoclave: {minfo['autoclave']}")
        report_t = make_report(
            "Transferencia de Calor 3D y Viscosidad", "Térmico 3D",
            [("T inicial / T chaqueta", f"{p['t_init_c']} / {p['t_jacket_c']}", "°C"),
             ("Agitación", f"{p['rpm']}", "RPM"), ("Material evaluado", p["polymer_check"], "-")],
            [("T media final", f"{sim['kpis']['t_mean']:.2f}", "°C"),
             ("Gradiente ΔT", f"{sim['kpis']['delta_t']:.2f}", "°C"),
             ("Viscosidad μ(T) final", f"{sim['kpis']['mu_final']:.2f}", "mPa·s")],
            r"1. **Energía:** $\rho c_p \partial T/\partial t = \nabla\cdot(k_{ef}\nabla T)+q_{met}$" + "\n" +
            r"2. **Viscosidad (Andrade):** $\mu(T)=\mu_{37}\exp[B(1/T - 1/310.15)]$",
            [f"Viscosidad cambia de {sim['kpis']['mu_init']:.2f} a {sim['kpis']['mu_final']:.2f} mPa·s."],
        )
        st.download_button("📥 Descargar Memoria de Cálculo (.md)", report_t.encode("utf-8"),
                           file_name="memoria_termico_3d.md", mime="text/markdown", width="stretch", key="dl_therm")

    k = sim["kpis"]
    with col_v:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("T media final", f"{k['t_mean']:.2f} °C")
        c2.metric("Gradiente ΔT", f"{k['delta_t']:.2f} °C", "Homogéneo" if k["delta_t"] <= 1 else "Gradiente alto",
                  delta_color="normal" if k["delta_t"] <= 1 else "inverse")
        c3.metric("Viscosidad μ(T)", f"{k['mu_final']:.2f} mPa·s", f"Inicio: {k['mu_init']:.2f}", delta_color="off")
        c4.metric("Calor metabólico", f"{k['q_met_w']:.1f} W")

        st.plotly_chart(build_thermal_figure(sim, p), width="stretch", key="therm_3d", config={"displaylogo": False})
        ca, cb = st.columns(2)
        f1, f2 = build_thermal_charts(sim, p)
        ca.plotly_chart(f1, width="stretch", key="therm_ts", config={"displaylogo": False})
        cb.plotly_chart(f2, width="stretch", key="therm_visc", config={"displaylogo": False})


# =====================================================================
# 5. MÓDULO 3 — DINÁMICA DE FLUIDOS (CON HÉLICE VS. SOLO FILTRO)
# =====================================================================
IMPELLERS: Dict[str, Dict[str, Any]] = {
    "Turbina Rushton (radial)": dict(np_t=5.0, kp=70.0, nq=0.72, vcf=0.80, double=True, blades=6, kind="rushton"),
    "Palas inclinadas 45° (axial)": dict(np_t=1.3, kp=50.0, nq=0.75, vcf=0.55, double=False, blades=4, kind="pbt"),
    "Hélice marina (axial)": dict(np_t=0.35, kp=40.0, nq=0.55, vcf=0.70, double=False, blades=3, kind="marine"),
    "Sin hélice (Solo filtro rotatorio)": dict(np_t=0.12, kp=25.0, nq=0.10, vcf=0.10, double=True, blades=0, kind="spin_only"),
}
FLUID_DEFAULTS = dict(impeller="Turbina Rushton (radial)", tank_d_cm=20.0, d_over_t=0.33, rpm=150, mu_mpas=1.0, n_index=1.0)


def _fluid_velocity(r: np.ndarray, z: np.ndarray, pr: Dict[str, float]):
    R, H, zi = pr["R"], pr["H"], pr["zi"]
    rho = np.clip(r / R, 0, 1)
    if pr["double"]:
        up = z >= zi
        z0 = np.where(up, zi, 0.0)
        dzl = np.where(up, H - zi, zi)
        sign = np.where(up, -1.0, 1.0)
    else:
        z0, dzl, sign = 0.0, H, -1.0
    s = np.clip((z - z0) / dzl, 0, 1)
    amp = sign * pr["vc"]
    vz = amp * (1 - 2 * rho**2) * np.sin(np.pi * s)
    vr = -amp * 0.5 * R * rho * (1 - rho**2) * np.pi * np.cos(np.pi * s) / dzl
    r_imp = pr["d_imp"] / 2
    vt_prof = np.where(r < r_imp, r / r_imp, r_imp / np.maximum(r, 1e-6))
    vt = pr["vth"] * vt_prof * (0.6 + 0.4 * np.exp(-(((z - zi) / (0.3 * H)) ** 2)))
    return vr, vz, vt


def _fluid_velocity_transient(
    r: np.ndarray | float,
    theta: np.ndarray | float,
    z: np.ndarray | float,
    t_s: float,
    pr: Dict[str, float],
    blades: int,
    w_vis: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Campo de velocidades transitorio con rampa de arranque e interacción por paso de palas."""
    r_arr = np.asarray(r, dtype=float)
    th_arr = np.asarray(theta, dtype=float)
    z_arr = np.asarray(z, dtype=float)

    vr0, vz0, vt0 = _fluid_velocity(r_arr, z_arr, pr)
    f_arr = float(0.25 + 0.75 * (1.0 - np.exp(-float(t_s) / 1.8)))
    amp_b = 0.06 if blades == 0 else 0.15
    nb = max(int(blades), 2)
    g_wake = 1.0 + amp_b * np.exp(-((z_arr - pr["zi"]) / (0.28 * pr["H"])) ** 2) * np.cos(nb * (th_arr - w_vis * t_s))
    factor = f_arr * g_wake
    return vr0 * factor, vz0 * factor, vt0 * factor, f_arr


@st.cache_data(show_spinner=False)
def run_fluid_vector_animation(
    pr: Dict[str, float],
    kind: str,
    blades: int,
    rpm: float,
    density: str = "Media",
    duration_s: float = ANIM_SECONDS,
    n_frames: int = N_FRAMES,
) -> Dict[str, Any]:
    """Integra trayectorias de flechas 3D del campo vectorial durante 10 s (t = 0 -> 10 s)."""
    res_map = {"Baja": (3, 6, 4), "Media": (4, 8, 4), "Alta": (4, 10, 5)}
    nr, nth, nz = res_map.get(density, (4, 8, 4))

    R_m, H_m = pr["R"], pr["H"]
    rg, thg, zg = np.meshgrid(
        np.linspace(0.18 * R_m, 0.90 * R_m, nr),
        np.linspace(0.0, 2.0 * np.pi, nth, endpoint=False),
        np.linspace(0.10 * H_m, 0.90 * H_m, nz),
        indexing="ij",
    )
    r_cur = rg.ravel().copy()
    th_cur = thg.ravel().copy()
    z_cur = zg.ravel().copy()

    rpm_vis = 0.05 + 0.30 * min(rpm, 400) / 400.0
    w_vis = 2.0 * np.pi * rpm_vis

    sub = 3
    dt_anim = duration_s / (n_frames * sub)
    dt_adv = (2.0 / duration_s) * dt_anim

    xs, ys, zs, vxs, vys, vzs, spds, f_arrs, angs = [], [], [], [], [], [], [], [], []
    t_grid = np.linspace(0.0, duration_s, n_frames + 1)

    for k in range(n_frames + 1):
        t_k = float(t_grid[k])
        vr, vz, vt, f_arr = _fluid_velocity_transient(r_cur, th_cur, z_cur, t_k, pr, blades, w_vis)
        xs.append(r_cur * np.cos(th_cur) * 100.0)
        ys.append(r_cur * np.sin(th_cur) * 100.0)
        zs.append(z_cur * 100.0)
        vxs.append(vr * np.cos(th_cur) - vt * np.sin(th_cur))
        vys.append(vr * np.sin(th_cur) + vt * np.cos(th_cur))
        vzs.append(vz)
        spds.append(np.sqrt(vr**2 + vz**2 + vt**2))
        f_arrs.append(f_arr)
        angs.append(float(w_vis * t_k))

        if k == n_frames:
            break
        for s_idx in range(sub):
            t_sub = t_k + s_idx * dt_anim
            vr1, vz1, vt1, _ = _fluid_velocity_transient(r_cur, th_cur, z_cur, t_sub, pr, blades, w_vis)
            rm = np.clip(r_cur + 0.5 * dt_adv * vr1, 0.04 * R_m, 0.96 * R_m)
            thm = th_cur + 0.5 * dt_adv * vt1 / np.maximum(r_cur, 0.004)
            zm = np.clip(z_cur + 0.5 * dt_adv * vz1, 0.03 * H_m, 0.97 * H_m)

            vr2, vz2, vt2, _ = _fluid_velocity_transient(rm, thm, zm, t_sub + 0.5 * dt_anim, pr, blades, w_vis)
            th_cur = (th_cur + dt_adv * vt2 / np.maximum(rm, 0.004)) % (2.0 * np.pi)
            r_cur = np.clip(r_cur + dt_adv * vr2, 0.04 * R_m, 0.96 * R_m)
            z_cur = np.clip(z_cur + dt_adv * vz2, 0.03 * H_m, 0.97 * H_m)

    spd_arr = np.array(spds)
    return {
        "x": np.array(xs),
        "y": np.array(ys),
        "z": np.array(zs),
        "vx": np.array(vxs),
        "vy": np.array(vys),
        "vz": np.array(vzs),
        "speed": spd_arr,
        "t": t_grid,
        "f_arr": np.array(f_arrs),
        "angle_imp": np.array(angs),
        "w_vis": float(w_vis),
        "vmax_global": float(np.max(spd_arr)) or 1.0,
    }


def _impeller_lines(kind: str, n_blades: int, d: float, zi: float, angle: float):
    xs, ys, zs = [], [], []

    def add(pts):
        for rr, aa, zz in pts:
            xs.append(rr * np.cos(aa))
            ys.append(rr * np.sin(aa))
            zs.append(zz)
        xs.append(None)
        ys.append(None)
        zs.append(None)

    if kind == "spin_only":
        # Cilindro central rotatorio (sin aspas)
        for zz in (zi * 0.4, zi * 1.4, zi * 2.3):
            add([(0.5 * d, a + angle, zz) for a in np.linspace(0, 2 * np.pi, 32)])
        for b in range(4):
            a = angle + b * np.pi / 2
            add([(0.5 * d, a, zi * 0.4), (0.5 * d, a, zi * 2.3)])
    elif kind == "rushton":
        add([(0.375 * d, a, zi) for a in np.linspace(0, 2 * np.pi, 40)])
        for b in range(n_blades):
            a = angle + b * 2 * np.pi / n_blades
            add([(0.375 * d, a, zi - 0.1 * d), (0.5 * d, a, zi - 0.1 * d), (0.5 * d, a, zi + 0.1 * d),
                 (0.375 * d, a, zi + 0.1 * d), (0.375 * d, a, zi - 0.1 * d)])
    else:
        for b in range(n_blades):
            a = angle + b * 2 * np.pi / n_blades
            add([(0.08 * d, a, zi), (0.5 * d, a - 0.18, zi - 0.09 * d), (0.5 * d, a + 0.18, zi + 0.09 * d),
                 (0.08 * d, a, zi)])
    return xs, ys, zs


def _compute_vortex_traces(
    pr: Dict[str, float],
    n_steps: int = 120,
    t_trace: float = 4.5,
) -> Dict[str, np.ndarray]:
    """Integra 14 líneas de corriente 3D (trazas de vórtices) mediante RK2."""
    R, H, zi = pr["R"], pr["H"], pr["zi"]
    if pr["double"]:
        # 7 semillas en vórtice superior + 7 en vórtice inferior
        r_s = np.array([
            0.35, 0.55, 0.78, 0.45, 0.70, 0.60, 0.85,
            0.35, 0.55, 0.78, 0.45, 0.70, 0.60, 0.85,
        ]) * R
        z_s = np.array([
            0.55, 0.65, 0.75, 0.82, 0.58, 0.70, 0.84,
            0.10, 0.16, 0.22, 0.26, 0.14, 0.20, 0.28,
        ]) * H
    else:
        r_s = np.array([0.30, 0.48, 0.68, 0.84, 0.36, 0.58, 0.78, 0.42, 0.64, 0.82, 0.28, 0.52, 0.72, 0.88]) * R
        z_s = np.array([0.18, 0.28, 0.42, 0.55, 0.68, 0.80, 0.88, 0.35, 0.52, 0.72, 0.76, 0.60, 0.22, 0.46]) * H

    th_s = np.linspace(0.0, 2.0 * np.pi, len(r_s), endpoint=False)
    r = r_s.copy()
    z = z_s.copy()
    th = th_s.copy()

    n_tr = len(r)
    dt = t_trace / n_steps
    xs = np.zeros((n_tr, n_steps + 1))
    ys = np.zeros((n_tr, n_steps + 1))
    zs = np.zeros((n_tr, n_steps + 1))
    rs = np.zeros((n_tr, n_steps + 1))
    vxs = np.zeros((n_tr, n_steps + 1))
    vys = np.zeros((n_tr, n_steps + 1))
    vzs = np.zeros((n_tr, n_steps + 1))
    spds = np.zeros((n_tr, n_steps + 1))

    for k in range(n_steps + 1):
        vr, vz, vt = _fluid_velocity(r, z, pr)
        xs[:, k] = r * np.cos(th) * 100.0
        ys[:, k] = r * np.sin(th) * 100.0
        zs[:, k] = z * 100.0
        rs[:, k] = r * 100.0
        vxs[:, k] = vr * np.cos(th) - vt * np.sin(th)
        vys[:, k] = vr * np.sin(th) + vt * np.cos(th)
        vzs[:, k] = vz
        spds[:, k] = np.sqrt(vr**2 + vz**2 + vt**2)
        if k == n_steps:
            break
        rm = np.clip(r + 0.5 * dt * vr, 0.002, 0.985 * R)
        zm = np.clip(z + 0.5 * dt * vz, 0.02 * H, 0.98 * H)
        vr2, vz2, vt2 = _fluid_velocity(rm, zm, pr)
        th = th + dt * vt2 / np.maximum(rm, 0.004)
        r = np.clip(r + dt * vr2, 0.002, 0.985 * R)
        z = np.clip(z + dt * vz2, 0.02 * H, 0.98 * H)

    return {
        "t_s": np.linspace(0.0, t_trace, n_steps + 1),
        "x_cm": xs,
        "y_cm": ys,
        "z_cm": zs,
        "r_cm": rs,
        "vx": vxs,
        "vy": vys,
        "vz": vzs,
        "speed": spds,
    }


@st.cache_data(show_spinner=False)
def run_fluid_solver(
    impeller: str, tank_d_cm: float, d_over_t: float, rpm: float, mu_mpas: float, n_index: float,
    n_frames: int = N_FRAMES, t_phys: float = 2.0, n_tracers: int = 450, seed: int = 3,
) -> Dict[str, Any]:
    cfg = IMPELLERS[impeller]
    T = tank_d_cm / 100.0
    D = T * d_over_t
    H = T
    N = rpm / 60.0
    vol = np.pi * T**2 / 4.0 * H
    ks = 11.0

    gamma = ks * N
    mu = (mu_mpas / 1000.0) * gamma ** (n_index - 1.0)
    re = RHO_FLUID * N * D**2 / mu
    npw = cfg["np_t"] + cfg["kp"] / re
    power = npw * RHO_FLUID * N**3 * D**5
    eps = power / (RHO_FLUID * vol)
    nu = mu / RHO_FLUID
    eta = (nu**3 / eps) ** 0.25
    u_tip = np.pi * N * D
    q_pump = cfg["nq"] * N * D**3
    t_mix = 5.9 * T ** (2 / 3) * eps ** (-1 / 3) * (D / T) ** (-1 / 3)
    if re < 1e4:
        t_mix *= (1e4 / re) ** 0.5
    regime = "Laminar" if re < 10 else ("Transición" if re < 1e4 else "Turbulento")

    f_c = re / (re + 300.0)
    pr = dict(R=T / 2, H=H, zi=H / 3, d_imp=D, double=cfg["double"], vc=cfg["vcf"] * u_tip * f_c,
              vth=0.45 * u_tip)

    rng = np.random.default_rng(seed)
    r = pr["R"] * np.sqrt(rng.uniform(0.02, 0.9, n_tracers))
    z = rng.uniform(0.04 * H, 0.96 * H, n_tracers)
    th = rng.uniform(0, 2 * np.pi, n_tracers)
    sub = 4
    dt = t_phys / (n_frames * sub)
    px, py, pz, vx_a, vy_a, vz_a, spd = [], [], [], [], [], [], []
    for k in range(n_frames + 1):
        vr, vz, vt = _fluid_velocity(r, z, pr)
        px.append(r * np.cos(th) * 100)
        py.append(r * np.sin(th) * 100)
        pz.append(z * 100)
        vx_a.append(vr * np.cos(th) - vt * np.sin(th))
        vy_a.append(vr * np.sin(th) + vt * np.cos(th))
        vz_a.append(vz)
        spd.append(np.sqrt(vr**2 + vz**2 + vt**2))
        if k == n_frames:
            break
        for _ in range(sub):
            vr1, vz1, vt1 = _fluid_velocity(r, z, pr)
            rm = np.clip(r + 0.5 * dt * vr1, 0.002, 0.985 * pr["R"])
            zm = np.clip(z + 0.5 * dt * vz1, 0.02 * H, 0.98 * H)
            vr2, vz2, vt2 = _fluid_velocity(rm, zm, pr)
            th = th + dt * vt2 / np.maximum(rm, 0.004)
            r = np.clip(r + dt * vr2, 0.002, 0.985 * pr["R"])
            z = np.clip(z + dt * vz2, 0.02 * H, 0.98 * H)

    re_grid = np.logspace(0, 6, 120)
    curves = {n: c["np_t"] + c["kp"] / re_grid for n, c in IMPELLERS.items()}
    vortex_traces = _compute_vortex_traces(pr)
    return {
        "x": np.array(px), "y": np.array(py), "z": np.array(pz),
        "vx": np.array(vx_a), "vy": np.array(vy_a), "vz": np.array(vz_a),
        "speed": np.array(spd), "t_phys": np.linspace(0, t_phys, n_frames + 1),
        "cfg": cfg, "geom": dict(T=T, D=D, H=H, zi=H / 3),
        "vel_params": pr,
        "vortex_traces": vortex_traces,
        "re_grid": re_grid, "np_curves": curves,
        "kpis": dict(re=float(re), regime=regime, np=float(npw), power_w=float(power), p_over_v=float(power / vol),
                     u_tip=float(u_tip), shear=float(gamma), eps=float(eps), eta_um=float(eta * 1e6),
                     t_mix=float(t_mix), q_pump_ls=float(q_pump * 1000), circ=float(f_c),
                     v_max_ms=float(np.max(spd)), v_circ_max_ms=float(pr["vc"]), v_theta_max_ms=float(pr["vth"])),
    }


def build_fluid_figure(
    sim: Dict[str, Any],
    params: Dict[str, Any],
    show_vectors: bool = False,
    show_vortex_traces: bool = True,
) -> go.Figure:
    g = sim["geom"]
    Rc, Hc, zi, d = g["T"] * 50, g["H"] * 100, g["zi"] * 100, g["D"] * 100
    cfg = sim["cfg"]
    th = np.linspace(0, 2 * np.pi, 50)
    zz = np.linspace(0, Hc, 20)
    thm, zm = np.meshgrid(th, zz)
    rpm_vis = 0.05 + 0.30 * min(params["rpm"], 400) / 400.0
    times = np.arange(len(sim["t_phys"])) * FRAME_MS / 1000.0
    vmax = float(sim["speed"].max()) or 1.0
    pastel_vel = [[0.0, "#74c0fc"], [0.35, "#63e6be"], [0.7, "#ffe066"], [1.0, "#ff6b6b"]]

    fig = go.Figure()
    fig.add_trace(go.Surface(x=Rc * np.cos(thm), y=Rc * np.sin(thm), z=zm, showscale=False, opacity=0.10,
                             colorscale=[[0, PASTEL["wall"]], [1, "#f1f5f9"]], hoverinfo="skip", name="Tanque"))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[Hc * 1.02, zi], mode="lines",
                               line=dict(color="#94a3b8", width=6), name="Eje", hoverinfo="skip"))

    def imp(k: int) -> go.Scatter3d:
        xs, ys, zs = _impeller_lines(cfg["kind"], cfg["blades"], d, zi, 2 * np.pi * rpm_vis * times[k])
        col = PASTEL["mesh_dark"] if cfg["kind"] == "spin_only" else PASTEL["impeller"]
        return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=col, width=7),
                            name="Impulsor / Filtro", hoverinfo="skip")

    def tracers(k: int) -> go.Scatter3d:
        return go.Scatter3d(
            x=sim["x"][k], y=sim["y"][k], z=sim["z"][k], mode="markers", name="Trazadores",
            marker=dict(size=3.8, color=sim["speed"][k], colorscale=pastel_vel, cmin=0, cmax=vmax, opacity=0.92,
                        colorbar=dict(title="|v| (m/s)", thickness=15, len=0.7)),
            hovertemplate="|v| = %{marker.color:.2f} m/s<extra></extra>")

    fig.add_trace(imp(0))      # 2
    fig.add_trace(tracers(0))  # 3

    if show_vortex_traces:
        vt_data = sim["vortex_traces"]
        n_tr, n_pts = vt_data["x_cm"].shape
        vx_l = np.empty((n_tr, n_pts + 1), dtype=object)
        vy_l = np.empty((n_tr, n_pts + 1), dtype=object)
        vz_l = np.empty((n_tr, n_pts + 1), dtype=object)
        vc_l = np.empty((n_tr, n_pts + 1), dtype=float)
        vx_l[:, :-1], vx_l[:, -1] = vt_data["x_cm"], None
        vy_l[:, :-1], vy_l[:, -1] = vt_data["y_cm"], None
        vz_l[:, :-1], vz_l[:, -1] = vt_data["z_cm"], None
        vc_l[:, :-1], vc_l[:, -1] = vt_data["speed"], 0.0
        fig.add_trace(go.Scatter3d(
            x=vx_l.ravel().tolist(),
            y=vy_l.ravel().tolist(),
            z=vz_l.ravel().tolist(),
            mode="lines",
            line=dict(color=vc_l.ravel().tolist(), colorscale=pastel_vel, cmin=0.0, cmax=vmax, width=3.5),
            opacity=0.78,
            name="Trazas de vórtices",
            hoverinfo="skip",
        ))

    if show_vectors:
        pr = sim["vel_params"]
        R_m, H_m = pr["R"], pr["H"]
        rg, thg, zg = np.meshgrid(
            np.array([0.30, 0.60, 0.88]) * R_m,
            np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False),
            np.linspace(0.15 * H_m, 0.85 * H_m, 4),
            indexing="ij",
        )
        vr_g, vz_g, vt_g = _fluid_velocity(rg.ravel(), zg.ravel(), pr)
        th_flat = thg.ravel()
        shaft_tr, head_tr = _build_3d_arrow_traces(
            rg.ravel() * np.cos(th_flat) * 100.0,
            rg.ravel() * np.sin(th_flat) * 100.0,
            zg.ravel() * 100.0,
            vr_g * np.cos(th_flat) - vt_g * np.sin(th_flat),
            vr_g * np.sin(th_flat) + vt_g * np.cos(th_flat),
            vz_g,
            max_arrow_len=0.22 * Rc,
            vmax=vmax,
            colorscale=pastel_vel,
            show_colorbar=False,
            name="Flechas de flujo",
            line_width=3.0,
            head_ratio=0.28,
            opacity=0.82,
        )
        fig.add_trace(shaft_tr)
        fig.add_trace(head_tr)

    frames = [go.Frame(data=[imp(k), tracers(k)], traces=[2, 3], name=str(k)) for k in range(len(times))]
    scene = _light_scene([-Rc * 1.05, Rc * 1.05], [-Rc * 1.05, Rc * 1.05], [0, Hc], z_title="z (cm)", xy_unit="cm")
    scene["aspectmode"] = "data"
    scene["camera"] = dict(eye=dict(x=1.7, y=1.3, z=0.7))
    fig.update_layout(template="plotly_white", height=620, margin=dict(l=0, r=0, t=30, b=0), showlegend=False, scene=scene)
    return add_animation_controls(fig, frames, [f"{t:.2f} s" for t in sim["t_phys"]], "Tiempo físico: ")


def build_np_re_figure(sim: Dict[str, Any], impeller: str) -> go.Figure:
    k = sim["kpis"]
    fig = go.Figure()
    for name, curve in sim["np_curves"].items():
        fig.add_trace(go.Scatter(x=sim["re_grid"], y=curve, name=name,
                                 line=dict(width=4 if name == impeller else 1.5, dash="solid" if name == impeller else "dot")))
    fig.add_trace(go.Scatter(x=[k["re"]], y=[k["np"]], mode="markers", name="Punto actual",
                             marker=dict(size=15, color=PASTEL["star"], symbol="star", line=dict(width=1, color="#1e293b"))))
    fig.update_layout(template="plotly_white", height=320, margin=dict(l=10, r=10, t=35, b=10),
                      title="Número de potencia vs Reynolds", xaxis_title="Re", yaxis_title="Np",
                      xaxis_type="log", yaxis_type="log", legend=dict(orientation="h", y=-0.35))
    return fig


def build_fluid_vector_3d_figure(
    sim: Dict[str, Any],
    params: Dict[str, Any],
    vec_anim: Dict[str, Any],
    paused_frame: int | None = None,
) -> go.Figure:
    """Construye la simulación animada 3D (10 s) o congelada de flechas de velocidad."""
    g = sim["geom"]
    Rc, Hc, zi, d = g["T"] * 50, g["H"] * 100, g["zi"] * 100, g["D"] * 100
    cfg = sim["cfg"]
    vmax = vec_anim["vmax_global"]

    pastel_vel = [[0.0, "#74c0fc"], [0.35, "#63e6be"], [0.7, "#ffe066"], [1.0, "#ff6b6b"]]
    th_s = np.linspace(0, 2 * np.pi, 45)
    zz_s = np.linspace(0, Hc, 16)
    thm, zm = np.meshgrid(th_s, zz_s)

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=Rc * np.cos(thm), y=Rc * np.sin(thm), z=zm, showscale=False, opacity=0.08,
        colorscale=[[0, PASTEL["wall"]], [1, "#f1f5f9"]], hoverinfo="skip", name="Tanque",
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[Hc * 1.02, zi], mode="lines",
        line=dict(color="#94a3b8", width=5), name="Eje", hoverinfo="skip",
    ))

    def imp_trace(k: int) -> go.Scatter3d:
        ix, iy, iz = _impeller_lines(cfg["kind"], cfg["blades"], d, zi, float(vec_anim["angle_imp"][k]))
        col_imp = PASTEL["mesh_dark"] if cfg["kind"] == "spin_only" else PASTEL["impeller"]
        return go.Scatter3d(
            x=ix, y=iy, z=iz, mode="lines",
            line=dict(color=col_imp, width=6), name="Impulsor / Filtro", hoverinfo="skip",
        )

    def arrow_traces(k: int) -> Tuple[go.Scatter3d, go.Cone]:
        return _build_3d_arrow_traces(
            vec_anim["x"][k], vec_anim["y"][k], vec_anim["z"][k],
            vec_anim["vx"][k], vec_anim["vy"][k], vec_anim["vz"][k],
            max_arrow_len=0.28 * Rc,
            vmax=vmax,
            colorscale=pastel_vel,
            show_colorbar=True,
            name="Vectores 3D",
            line_width=3.6,
            head_ratio=0.30,
        )

    k0 = 0 if paused_frame is None else int(np.clip(paused_frame, 0, len(vec_anim["t"]) - 1))
    s0, h0 = arrow_traces(k0)
    fig.add_trace(imp_trace(k0))  # 2
    fig.add_trace(s0)             # 3
    fig.add_trace(h0)             # 4

    scene = _light_scene([-Rc * 1.05, Rc * 1.05], [-Rc * 1.05, Rc * 1.05], [0, Hc], z_title="z (cm)", xy_unit="cm")
    scene["aspectmode"] = "data"
    scene["camera"] = dict(eye=dict(x=1.65, y=1.35, z=0.75))

    if paused_frame is None:
        frames = [
            go.Frame(data=[imp_trace(k), *arrow_traces(k)], traces=[2, 3, 4], name=str(k))
            for k in range(len(vec_anim["t"]))
        ]
        fig.update_layout(
            template="plotly_white", height=520, margin=dict(l=0, r=0, t=40, b=0),
            title="Simulación 3D de Vectores de Velocidad (10 s)", showlegend=False, scene=scene,
        )
        return add_animation_controls(fig, frames, [f"{t:.1f} s" for t in vec_anim["t"]], "Tiempo: ")

    t_k = float(vec_anim["t"][k0])
    fig.update_layout(
        template="plotly_white", height=520, margin=dict(l=0, r=0, t=40, b=0),
        title=f"Vista 3D Pausada — t = {t_k:.1f} s", showlegend=False, scene=scene,
    )
    return fig


def build_fluid_vector_2d_figure(
    sim: Dict[str, Any],
    params: Dict[str, Any],
    plane: str = "Meridional (r–z)",
    z_slice_cm: float | None = None,
    t_s: float = 10.0,
) -> go.Figure:
    """Construye un mapa de calor + flechas quiver 2D en corte meridional (r-z) u horizontal (x-y) en t = t_s."""
    pr = sim["vel_params"]
    g = sim["geom"]
    R_m, H_m = pr["R"], pr["H"]
    Rc, Hc, zi, d = g["T"] * 50, g["H"] * 100, g["zi"] * 100, g["D"] * 100
    cfg = sim["cfg"]
    blades = cfg["blades"]
    rpm_vis = 0.05 + 0.30 * min(params["rpm"], 400) / 400.0
    w_vis = 2.0 * np.pi * rpm_vis
    z_max_hm = sim["kpis"]["v_max_ms"] * 1.15
    pastel_vel = [[0.0, "#e7f5ff"], [0.3, "#74c0fc"], [0.55, "#63e6be"], [0.8, "#ffe066"], [1.0, "#ff6b6b"]]

    fig = go.Figure()
    if plane == "Meridional (r–z)":
        xg = np.linspace(-Rc, Rc, 61)
        zg = np.linspace(0.0, Hc, 45)
        xx, zz = np.meshgrid(xg, zg)
        th_hm = np.where(xx >= 0.0, 0.0, np.pi)
        vr_h, vz_h, vt_h, _ = _fluid_velocity_transient(np.abs(xx) / 100.0, th_hm, zz / 100.0, t_s, pr, blades, w_vis)
        vmag_h = np.sqrt(vr_h**2 + vz_h**2 + vt_h**2)
        fig.add_trace(go.Heatmap(
            x=xg, y=zg, z=vmag_h, colorscale=pastel_vel,
            zmin=0.0, zmax=z_max_hm,
            colorbar=dict(title="|v| (m/s)", thickness=14, len=0.8),
            hovertemplate="x=%{x:.1f} cm · z=%{y:.1f} cm<br>|v|=%{z:.3f} m/s<extra></extra>",
        ))

        qx = np.linspace(-0.92 * Rc, 0.92 * Rc, 14)
        qz = np.linspace(0.07 * Hc, 0.93 * Hc, 11)
        qxx, qzz = np.meshgrid(qx, qz)
        q_mask = np.abs(qxx) >= 0.06 * Rc
        qxv, qzv = qxx[q_mask], qzz[q_mask]
        th_q = np.where(qxv >= 0.0, 0.0, np.pi)
        vr_q, vz_q, vt_q, _ = _fluid_velocity_transient(np.abs(qxv) / 100.0, th_q, qzv / 100.0, t_s, pr, blades, w_vis)
        u_q = vr_q * np.sign(qxv)
        v_q = vz_q
        ref_rz = float(sim["kpis"]["v_circ_max_ms"]) or 1.0
        ax, ay = _quiver_2d_coords(qxv, qzv, u_q, v_q, max_len=Rc * 0.14, ref_mag=ref_rz)
        fig.add_trace(go.Scatter(
            x=ax, y=ay, mode="lines", line=dict(color="#1e293b", width=1.6),
            name="Flujo (vr, vz)", hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=qxv, y=qzv, mode="markers", marker=dict(size=6, opacity=0.0),
            customdata=np.stack([np.abs(qxv), vr_q, vz_q, vt_q, np.sqrt(vr_q**2 + vz_q**2 + vt_q**2)], axis=-1),
            hovertemplate=(
                "r=%{customdata[0]:.1f} cm · z=%{y:.1f} cm<br>"
                "vr=%{customdata[1]:.3f} · vz=%{customdata[2]:.3f} · vθ=%{customdata[3]:.3f} m/s<br>"
                "|v|=%{customdata[4]:.3f} m/s<extra>Nodo</extra>"
            ),
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=[-Rc, Rc, Rc, -Rc, -Rc], y=[0, 0, Hc, Hc, 0], mode="lines",
            line=dict(color="#64748b", width=2), name="Tanque", hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=[0, 0], y=[Hc, zi], mode="lines",
            line=dict(color="#475569", width=3), name="Eje", hoverinfo="skip",
        ))
        if cfg["kind"] == "spin_only":
            r_c = 0.5 * d
            fig.add_trace(go.Scatter(
                x=[-r_c, r_c, r_c, -r_c, -r_c], y=[zi * 0.4, zi * 0.4, zi * 2.3, zi * 2.3, zi * 0.4],
                mode="lines", line=dict(color=PASTEL["mesh_dark"], width=3), name="Filtro rotatorio", hoverinfo="skip",
            ))
        else:
            r_c = 0.5 * d
            fig.add_trace(go.Scatter(
                x=[-r_c, r_c], y=[zi, zi], mode="lines",
                line=dict(color=PASTEL["impeller"], width=5), name="Impulsor", hoverinfo="skip",
            ))

        fig.update_layout(
            template="plotly_white", height=520, margin=dict(l=10, r=10, t=40, b=15),
            title=f"Corte 2D Meridional (r–z) — t = {t_s:.1f} s",
            xaxis=dict(title="Posición radial x (cm)", range=[-Rc * 1.06, Rc * 1.06]),
            yaxis=dict(title="Altura z (cm)", range=[0, Hc], scaleanchor="x", scaleratio=1),
            showlegend=False,
        )
    else:
        z_cm = z_slice_cm if z_slice_cm is not None else zi
        z_m = z_cm / 100.0
        grid = np.linspace(-Rc, Rc, 61)
        xx, yy = np.meshgrid(grid, grid)
        rr_m = np.hypot(xx, yy) / 100.0
        th_hm = np.arctan2(yy, xx)
        vr_h, vz_h, vt_h, _ = _fluid_velocity_transient(rr_m, th_hm, np.full_like(rr_m, z_m), t_s, pr, blades, w_vis)
        vmag_h = np.where(rr_m <= R_m, np.sqrt(vr_h**2 + vz_h**2 + vt_h**2), np.nan)
        fig.add_trace(go.Heatmap(
            x=grid, y=grid, z=vmag_h, colorscale=pastel_vel,
            zmin=0.0, zmax=z_max_hm,
            colorbar=dict(title="|v| (m/s)", thickness=14, len=0.8),
            hovertemplate="x=%{x:.1f} cm · y=%{y:.1f} cm<br>|v|=%{z:.3f} m/s<extra></extra>",
        ))

        q_pts = np.linspace(-0.92 * Rc, 0.92 * Rc, 15)
        qxx, qyy = np.meshgrid(q_pts, q_pts)
        qrr_m = np.hypot(qxx, qyy) / 100.0
        q_mask = (qrr_m >= 0.08 * R_m) & (qrr_m <= 0.94 * R_m)
        qxv, qyv, qrv = qxx[q_mask], qyy[q_mask], qrr_m[q_mask]
        th_q = np.arctan2(qyv, qxv)
        vr_q, vz_q, vt_q, _ = _fluid_velocity_transient(qrv, th_q, np.full_like(qrv, z_m), t_s, pr, blades, w_vis)
        r_safe = np.maximum(qrv * 100.0, 1e-6)
        cos_t = qxv / r_safe
        sin_t = qyv / r_safe
        vx_q = vr_q * cos_t - vt_q * sin_t
        vy_q = vr_q * sin_t + vt_q * cos_t

        ref_xy = float(sim["kpis"]["v_max_ms"]) or 1.0
        ax, ay = _quiver_2d_coords(qxv, qyv, vx_q, vy_q, max_len=Rc * 0.13, ref_mag=ref_xy)
        fig.add_trace(go.Scatter(
            x=ax, y=ay, mode="lines", line=dict(color="#1e293b", width=1.6),
            name="Vectores (vx, vy)", hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=qxv, y=qyv, mode="markers", marker=dict(size=6, opacity=0.0),
            customdata=np.stack([qrv * 100.0, vr_q, vz_q, vt_q, np.sqrt(vr_q**2 + vz_q**2 + vt_q**2)], axis=-1),
            hovertemplate=(
                "r=%{customdata[0]:.1f} cm<br>"
                "vr=%{customdata[1]:.3f} · vz=%{customdata[2]:.3f} · vθ=%{customdata[3]:.3f} m/s<br>"
                "|v|=%{customdata[4]:.3f} m/s<extra>Nodo</extra>"
            ),
            showlegend=False,
        ))
        th_c = np.linspace(0, 2 * np.pi, 100)
        fig.add_trace(go.Scatter(
            x=Rc * np.cos(th_c), y=Rc * np.sin(th_c), mode="lines",
            line=dict(color="#64748b", width=2), name="Pared", hoverinfo="skip",
        ))
        col_imp = PASTEL["mesh_dark"] if cfg["kind"] == "spin_only" else PASTEL["impeller"]
        fig.add_trace(go.Scatter(
            x=(d / 2.0) * np.cos(th_c), y=(d / 2.0) * np.sin(th_c), mode="lines",
            line=dict(color=col_imp, width=3, dash="dash"), name="Impulsor", hoverinfo="skip",
        ))

        fig.update_layout(
            template="plotly_white", height=520, margin=dict(l=10, r=10, t=40, b=15),
            title=f"Corte 2D Horizontal (x–y) a z = {z_cm:.1f} cm — t = {t_s:.1f} s",
            xaxis=dict(title="x (cm)", range=[-Rc * 1.06, Rc * 1.06]),
            yaxis=dict(title="y (cm)", range=[-Rc * 1.06, Rc * 1.06], scaleanchor="x", scaleratio=1),
            showlegend=False,
        )
    return fig


@st.cache_data(show_spinner=False)
def generate_fluid_snapshot_png(
    sim: Dict[str, Any],
    params: Dict[str, Any],
    vec_anim: Dict[str, Any],
    frame_idx: int,
    plane: str,
    z_slice_cm: float | None,
) -> bytes:
    """Genera la ficha técnica PNG en alta resolución (3D + 2D + parámetros + ecuación vectorial)."""
    k_idx = int(np.clip(frame_idx, 0, len(vec_anim["t"]) - 1))
    t_k = float(vec_anim["t"][k_idx])
    f_arr_k = float(vec_anim["f_arr"][k_idx])
    pr = sim["vel_params"]
    g = sim["geom"]
    kpis = sim["kpis"]
    cfg = sim["cfg"]
    blades = cfg["blades"]
    w_vis = vec_anim["w_vis"]
    R_m, H_m = pr["R"], pr["H"]
    Rc, Hc, zi, d = g["T"] * 50, g["H"] * 100, g["zi"] * 100, g["D"] * 100
    vmax_3d = vec_anim["vmax_global"]

    pastel_cmap = LinearSegmentedColormap.from_list(
        "pastel_vel_3d", ["#74c0fc", "#63e6be", "#ffe066", "#ff6b6b"]
    )
    pastel_cmap_2d = LinearSegmentedColormap.from_list(
        "pastel_vel_2d", ["#e7f5ff", "#74c0fc", "#63e6be", "#ffe066", "#ff6b6b"]
    )

    fig = plt.figure(figsize=(14.5, 9.6), dpi=150, facecolor="#fcfcfd")
    try:
        gs = GridSpec(3, 2, figure=fig, height_ratios=[0.11, 0.56, 0.33], hspace=0.28, wspace=0.18)

        # Banda superior
        ax_head = fig.add_subplot(gs[0, :])
        ax_head.axis("off")
        ax_head.add_patch(
            plt.Rectangle((0.0, 0.05), 1.0, 0.90, facecolor="#edf2ff", edgecolor="#bac8ff", linewidth=1.5, transform=ax_head.transAxes)
        )
        ax_head.text(
            0.02, 0.64,
            "REPORTE INSTANTÁNEO DEL CAMPO VECTORIAL DE VELOCIDADES (DINÁMICA DE FLUIDOS)",
            fontsize=13, fontweight="bold", color="#1e293b", va="center", transform=ax_head.transAxes,
        )
        ax_head.text(
            0.02, 0.25,
            f"Configuración: {params['impeller']}  |  Velocidad: {params['rpm']} RPM  |  "
            f"Régimen: {kpis['regime']} (Re = {kpis['re']:,.0f})  |  Instante: t = {t_k:.1f} s / 10.0 s",
            fontsize=9.6, color="#334155", va="center", transform=ax_head.transAxes,
        )

        # Panel 3D
        ax3d = fig.add_subplot(gs[1, 0], projection="3d", facecolor="#f8fafc")
        th_c = np.linspace(0, 2 * np.pi, 36)
        z_c = np.linspace(0, Hc, 10)
        thm, zm = np.meshgrid(th_c, z_c)
        ax3d.plot_wireframe(Rc * np.cos(thm), Rc * np.sin(thm), zm, color="#cbd5e1", alpha=0.35, rstride=3, cstride=4, linewidth=0.6)
        ax3d.plot([0, 0], [0, 0], [Hc * 1.02, zi], color="#94a3b8", linewidth=2.2)

        ix, iy, iz = _impeller_lines(cfg["kind"], blades, d, zi, float(vec_anim["angle_imp"][k_idx]))
        col_imp = "#4dabf7" if cfg["kind"] == "spin_only" else "#fcc419"
        # Separar segmentos con None
        seg_x, seg_y, seg_z = [], [], []
        for xi, yi, zii in zip(ix, iy, iz):
            if xi is None:
                if seg_x:
                    ax3d.plot(seg_x, seg_y, seg_z, color=col_imp, linewidth=2.2)
                    seg_x, seg_y, seg_z = [], [], []
            else:
                seg_x.append(xi)
                seg_y.append(yi)
                seg_z.append(zii)

        xk, yk, zk = vec_anim["x"][k_idx], vec_anim["y"][k_idx], vec_anim["z"][k_idx]
        ux, uy, uz = vec_anim["vx"][k_idx], vec_anim["vy"][k_idx], vec_anim["vz"][k_idx]
        spdk = vec_anim["speed"][k_idx]
        scale_3d = (0.24 * Rc) / max(vmax_3d, 1e-6)
        n_bins = 8
        edges = np.linspace(0.0, vmax_3d + 1e-9, n_bins + 1)
        for b in range(n_bins):
            m = (spdk >= edges[b]) & (spdk <= edges[b + 1])
            if np.any(m):
                c_val = pastel_cmap((b + 0.5) / n_bins)
                ax3d.quiver(
                    xk[m], yk[m], zk[m], ux[m] * scale_3d, uy[m] * scale_3d, uz[m] * scale_3d,
                    color=c_val, arrow_length_ratio=0.32, linewidth=1.2, normalize=False,
                )
        ax3d.set_xlim(-Rc * 1.05, Rc * 1.05)
        ax3d.set_ylim(-Rc * 1.05, Rc * 1.05)
        ax3d.set_zlim(0, Hc)
        ax3d.set_xlabel("X (cm)", fontsize=8.5)
        ax3d.set_ylabel("Y (cm)", fontsize=8.5)
        ax3d.set_zlabel("Z (cm)", fontsize=8.5)
        ax3d.set_title(f"Campo Vectorial 3D — t = {t_k:.1f} s", fontsize=11, fontweight="bold", color="#1e293b", pad=8)
        sm3d = ScalarMappable(norm=Normalize(0.0, vmax_3d), cmap=pastel_cmap)
        sm3d.set_array([])
        cb3d = fig.colorbar(sm3d, ax=ax3d, shrink=0.68, pad=0.14)
        cb3d.set_label("|v| (m/s)", fontsize=8.5)

        # Panel 2D
        ax2d = fig.add_subplot(gs[1, 1], facecolor="#f8fafc")
        vmax_2d = kpis["v_max_ms"] * 1.15 or 1.0
        if plane == "Meridional (r–z)":
            xg = np.linspace(-Rc, Rc, 81)
            zg = np.linspace(0.0, Hc, 61)
            xx, zz = np.meshgrid(xg, zg)
            th_h = np.where(xx >= 0.0, 0.0, np.pi)
            vr_h, vz_h, vt_h, _ = _fluid_velocity_transient(np.abs(xx) / 100.0, th_h, zz / 100.0, t_k, pr, blades, w_vis)
            vmag_h = np.sqrt(vr_h**2 + vz_h**2 + vt_h**2)
            im = ax2d.imshow(
                vmag_h, extent=[-Rc, Rc, 0.0, Hc],
                origin="lower", cmap=pastel_cmap_2d, vmin=0.0, vmax=vmax_2d,
            )
            qx = np.linspace(-0.90 * Rc, 0.90 * Rc, 13)
            qz = np.linspace(0.08 * Hc, 0.92 * Hc, 10)
            qxx, qzz = np.meshgrid(qx, qz)
            qm = np.abs(qxx) >= 0.06 * Rc
            vr_q, vz_q, _, _ = _fluid_velocity_transient(
                np.abs(qxx[qm]) / 100.0, np.where(qxx[qm] >= 0.0, 0.0, np.pi), qzz[qm] / 100.0, t_k, pr, blades, w_vis
            )
            ax2d.quiver(
                qxx[qm], qzz[qm], vr_q * np.sign(qxx[qm]), vz_q,
                color="#1e293b", scale=max(kpis["v_circ_max_ms"], 1e-4) * 11.0, width=0.004,
            )
            ax2d.plot([-Rc, Rc, Rc, -Rc, -Rc], [0, 0, Hc, Hc, 0], color="#64748b", linewidth=1.6)
            ax2d.plot([0, 0], [Hc, zi], color="#475569", linewidth=2.0)
            r_c = 0.5 * d
            if cfg["kind"] == "spin_only":
                ax2d.plot([-r_c, r_c, r_c, -r_c, -r_c], [zi * 0.4, zi * 0.4, zi * 2.3, zi * 2.3, zi * 0.4], color="#4dabf7", linewidth=2.2)
            else:
                ax2d.plot([-r_c, r_c], [zi, zi], color="#fcc419", linewidth=3.5)
            ax2d.set_xlim(-Rc * 1.06, Rc * 1.06)
            ax2d.set_ylim(0.0, Hc * 1.03)
            ax2d.set_xlabel("Posición radial x (cm)", fontsize=8.5)
            ax2d.set_ylabel("Altura z (cm)", fontsize=8.5)
            ax2d.set_title(f"Corte 2D Meridional (r–z) — t = {t_k:.1f} s", fontsize=11, fontweight="bold", color="#1e293b", pad=8)
        else:
            z_cm = z_slice_cm if z_slice_cm is not None else zi
            z_m = z_cm / 100.0
            grid = np.linspace(-Rc, Rc, 81)
            xx, yy = np.meshgrid(grid, grid)
            rr_m = np.hypot(xx, yy) / 100.0
            vr_h, vz_h, vt_h, _ = _fluid_velocity_transient(rr_m, np.arctan2(yy, xx), np.full_like(rr_m, z_m), t_k, pr, blades, w_vis)
            vmag_h = np.where(rr_m <= R_m, np.sqrt(vr_h**2 + vz_h**2 + vt_h**2), np.nan)
            im = ax2d.imshow(
                vmag_h, extent=[-Rc, Rc, -Rc, Rc],
                origin="lower", cmap=pastel_cmap_2d, vmin=0.0, vmax=vmax_2d,
            )
            qp = np.linspace(-0.90 * Rc, 0.90 * Rc, 14)
            qxx, qyy = np.meshgrid(qp, qp)
            qrr_m = np.hypot(qxx, qyy) / 100.0
            qm = (qrr_m >= 0.08 * R_m) & (qrr_m <= 0.93 * R_m)
            qxv, qyv, qrv = qxx[qm], qyy[qm], qrr_m[qm]
            vr_q, _, vt_q, _ = _fluid_velocity_transient(qrv, np.arctan2(qyv, qxv), np.full_like(qrv, z_m), t_k, pr, blades, w_vis)
            rs = np.maximum(qrv * 100.0, 1e-6)
            ax2d.quiver(
                qxv, qyv, vr_q * (qxv / rs) - vt_q * (qyv / rs), vr_q * (qyv / rs) + vt_q * (qxv / rs),
                color="#1e293b", scale=max(kpis["v_max_ms"], 1e-4) * 11.0, width=0.004,
            )
            th_r = np.linspace(0, 2 * np.pi, 120)
            ax2d.plot(Rc * np.cos(th_r), Rc * np.sin(th_r), color="#64748b", linewidth=1.6)
            ax2d.plot((d / 2.0) * np.cos(th_r), (d / 2.0) * np.sin(th_r), color=col_imp, linewidth=2.2, linestyle="--")
            ax2d.set_xlim(-Rc * 1.06, Rc * 1.06)
            ax2d.set_ylim(-Rc * 1.06, Rc * 1.06)
            ax2d.set_xlabel("x (cm)", fontsize=8.5)
            ax2d.set_ylabel("y (cm)", fontsize=8.5)
            ax2d.set_title(f"Corte 2D Horizontal (x–y, z={z_cm:.1f} cm) — t = {t_k:.1f} s", fontsize=10.5, fontweight="bold", color="#1e293b", pad=8)

        ax2d.set_aspect("equal")
        cb2d = fig.colorbar(im, ax=ax2d, shrink=0.78, pad=0.04)
        cb2d.set_label("|v| (m/s)", fontsize=8.5)

        # Tarjeta inferior izquierda: Parámetros y KPIs
        ax_p = fig.add_subplot(gs[2, 0])
        ax_p.axis("off")
        ax_p.add_patch(
            plt.Rectangle((0.0, 0.0), 1.0, 1.0, facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=1.2, transform=ax_p.transAxes)
        )
        ax_p.text(0.03, 0.88, "PARÁMETROS DE CONFIGURACIÓN Y RESULTADOS", fontsize=10, fontweight="bold", color="#1e293b", transform=ax_p.transAxes)
        plane_label = plane if plane == "Meridional (r–z)" else f"{plane} (z={z_slice_cm:.1f} cm)" if z_slice_cm is not None else plane
        param_lines = [
            f"• Impulsor / Modo: {params['impeller']}",
            f"• Tanque T: {params['tank_d_cm']:.1f} cm   |   D/T: {params['d_over_t']:.2f}   |   Velocidad: {params['rpm']} RPM",
            f"• Viscosidad: {params['mu_mpas']:.1f} mPa·s   |   Índice de flujo n: {params['n_index']:.2f}",
            f"• Corte 2D: {plane_label}   |   Escala Kolmogorov η_K: {kpis['eta_um']:.0f} µm",
            f"• Potencia P: {kpis['power_w']:.2f} W ({kpis['p_over_v']:.0f} W/m³)   |   Tiempo de mezcla: {kpis['t_mix']:.1f} s",
            f"• |v| máx en t={t_k:.1f} s: {float(np.max(spdk)):.3f} m/s   |   Factor arranque f(t): {f_arr_k:.3f}",
        ]
        ax_p.text(0.03, 0.77, "\n".join(param_lines), fontsize=8.8, color="#334155", va="top", linespacing=1.48, transform=ax_p.transAxes)

        # Tarjeta inferior derecha: Ecuación Vectorial
        ax_eq = fig.add_subplot(gs[2, 1])
        ax_eq.axis("off")
        ax_eq.add_patch(
            plt.Rectangle((0.0, 0.0), 1.0, 1.0, facecolor="#f8fafc", edgecolor="#748ffc", linewidth=1.4, transform=ax_eq.transAxes)
        )
        ax_eq.text(0.03, 0.88, "ECUACIÓN VECTORIAL DEL CAMPO DE VELOCIDADES", fontsize=10, fontweight="bold", color="#1e293b", transform=ax_eq.transAxes)
        eq_lines = [
            r"$\mathbf{v}(r,\theta,z,t) = v_r\,\hat{e}_r + v_\theta\,\hat{e}_\theta + v_z\,\hat{e}_z,\ \ v_x=v_r\cos\theta-v_\theta\sin\theta,\ \ v_y=v_r\sin\theta+v_\theta\cos\theta$",
            r"$v_z = \pm f(t)\,g\,v_c[1 - 2(r/R)^2]\sin(\pi s),\ \ v_r = \mp f(t)\,g\,v_c\frac{\pi R}{2 L_z}\frac{r}{R}[1-(r/R)^2]\cos(\pi s)$",
            r"$v_\theta = f(t)\,g\,v_{\theta,0}\,\psi(r)[0.6 + 0.4\,e^{-((z-z_i)/(0.3H))^2}],\ \ f(t)=0.25+0.75(1-e^{-t/1.8})$",
            r"$v_c=%.3f\ \mathrm{m/s},\ \ v_{\theta,0}=%.3f\ \mathrm{m/s},\ \ R=%.2f\ \mathrm{m},\ \ H=%.2f\ \mathrm{m},\ \ z_i=%.3f\ \mathrm{m},\ \ f(t)=%.3f$"
            % (pr["vc"], pr["vth"], R_m, H_m, pr["zi"], f_arr_k),
        ]
        y_pos = 0.74
        for line in eq_lines:
            ax_eq.text(0.03, y_pos, line, fontsize=7.6, color="#1e293b", va="top", transform=ax_eq.transAxes)
            y_pos -= 0.175

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        return buf.getvalue()
    finally:
        plt.close(fig)


def generate_vortex_traces_csv(sim: Dict[str, Any], params: Dict[str, Any]) -> bytes:
    """Genera el archivo CSV con las coordenadas y velocidades de todas las trazas de vórtices."""
    vt = sim["vortex_traces"]
    n_tr, n_pts = vt["x_cm"].shape
    lines = ["trace_id,step,t_s,x_cm,y_cm,z_cm,r_cm,vx_ms,vy_ms,vz_ms,v_mag_ms,impeller,rpm,mu_mpas"]
    imp_label = params["impeller"].replace(",", ";")
    rpm_v = params["rpm"]
    mu_v = params["mu_mpas"]
    for i in range(n_tr):
        for s in range(n_pts):
            lines.append(
                f"{i + 1},{s},{vt['t_s'][s]:.4f},"
                f"{vt['x_cm'][i, s]:.4f},{vt['y_cm'][i, s]:.4f},{vt['z_cm'][i, s]:.4f},{vt['r_cm'][i, s]:.4f},"
                f"{vt['vx'][i, s]:.5f},{vt['vy'][i, s]:.5f},{vt['vz'][i, s]:.5f},{vt['speed'][i, s]:.5f},"
                f"\"{imp_label}\",{rpm_v},{mu_v}"
            )
    return "\n".join(lines).encode("utf-8")


@st.cache_data(show_spinner=False)
def generate_vortex_traces_png(sim: Dict[str, Any], params: Dict[str, Any]) -> bytes:
    """Genera la imagen PNG con las trazas de vórtices 3D, corte meridional (r, z) y métricas."""
    vt = sim["vortex_traces"]
    g = sim["geom"]
    kpis = sim["kpis"]
    cfg = sim["cfg"]
    Rc, Hc, zi, d = g["T"] * 50, g["H"] * 100, g["zi"] * 100, g["D"] * 100
    vmax = float(np.max(vt["speed"])) or 1.0

    pastel_cmap = LinearSegmentedColormap.from_list(
        "pastel_vortex", ["#74c0fc", "#63e6be", "#ffe066", "#ff6b6b"]
    )
    fig = plt.figure(figsize=(14.0, 8.8), dpi=150, facecolor="#fcfcfd")
    try:
        gs = GridSpec(3, 2, figure=fig, height_ratios=[0.11, 0.60, 0.29], hspace=0.28, wspace=0.18)

        ax_head = fig.add_subplot(gs[0, :])
        ax_head.axis("off")
        ax_head.add_patch(
            plt.Rectangle((0.0, 0.05), 1.0, 0.90, facecolor="#edf2ff", edgecolor="#bac8ff", linewidth=1.5, transform=ax_head.transAxes)
        )
        ax_head.text(
            0.02, 0.64,
            f"TRAZAS DE VÓRTICES 3D Y LÍNEAS DE CORRIENTE — {params['impeller'].upper()}",
            fontsize=12.5, fontweight="bold", color="#1e293b", va="center", transform=ax_head.transAxes,
        )
        ax_head.text(
            0.02, 0.25,
            f"Tanque T = {params['tank_d_cm']:.1f} cm  |  D/T = {params['d_over_t']:.2f}  |  {params['rpm']} RPM  |  "
            f"Viscosidad: {params['mu_mpas']:.1f} mPa·s (n = {params['n_index']:.2f})  |  Re = {kpis['re']:,.0f} ({kpis['regime']})",
            fontsize=9.5, color="#334155", va="center", transform=ax_head.transAxes,
        )

        # Panel 3D de trazas
        ax3d = fig.add_subplot(gs[1, 0], projection="3d", facecolor="#f8fafc")
        th_c = np.linspace(0, 2 * np.pi, 36)
        z_c = np.linspace(0, Hc, 10)
        thm, zm = np.meshgrid(th_c, z_c)
        ax3d.plot_wireframe(Rc * np.cos(thm), Rc * np.sin(thm), zm, color="#cbd5e1", alpha=0.35, rstride=3, cstride=4, linewidth=0.6)
        ax3d.plot([0, 0], [0, 0], [Hc * 1.02, zi], color="#94a3b8", linewidth=2.2)

        ix, iy, iz = _impeller_lines(cfg["kind"], cfg["blades"], d, zi, 0.0)
        col_imp = "#4dabf7" if cfg["kind"] == "spin_only" else "#fcc419"
        sx, sy, sz = [], [], []
        for xi, yi, zii in zip(ix, iy, iz):
            if xi is None:
                if sx:
                    ax3d.plot(sx, sy, sz, color=col_imp, linewidth=2.2)
                    sx, sy, sz = [], [], []
            else:
                sx.append(xi)
                sy.append(yi)
                sz.append(zii)

        n_tr, n_pts = vt["x_cm"].shape
        for i in range(n_tr):
            for s in range(0, n_pts - 1, 2):
                s_end = min(s + 2, n_pts - 1)
                v_seg = 0.5 * (vt["speed"][i, s] + vt["speed"][i, s_end])
                ax3d.plot(
                    vt["x_cm"][i, s:s_end + 1], vt["y_cm"][i, s:s_end + 1], vt["z_cm"][i, s:s_end + 1],
                    color=pastel_cmap(np.clip(v_seg / vmax, 0.0, 1.0)), linewidth=1.6, alpha=0.90,
                )

        ax3d.set_xlim(-Rc * 1.05, Rc * 1.05)
        ax3d.set_ylim(-Rc * 1.05, Rc * 1.05)
        ax3d.set_zlim(0, Hc)
        ax3d.set_xlabel("X (cm)", fontsize=8.5)
        ax3d.set_ylabel("Y (cm)", fontsize=8.5)
        ax3d.set_zlabel("Z (cm)", fontsize=8.5)
        ax3d.set_title("Líneas de Corriente Helicoidales 3D", fontsize=11, fontweight="bold", color="#1e293b", pad=8)
        sm3d = ScalarMappable(norm=Normalize(0.0, vmax), cmap=pastel_cmap)
        sm3d.set_array([])
        cb3d = fig.colorbar(sm3d, ax=ax3d, shrink=0.68, pad=0.14)
        cb3d.set_label("|v| (m/s)", fontsize=8.5)

        # Panel 2D Meridional (r, z)
        ax2d = fig.add_subplot(gs[1, 1], facecolor="#f8fafc")
        for i in range(n_tr):
            side = 1.0 if i % 2 == 0 else -1.0
            r_plot = side * vt["r_cm"][i]
            z_plot = vt["z_cm"][i]
            for s in range(0, n_pts - 1, 2):
                s_end = min(s + 2, n_pts - 1)
                v_seg = 0.5 * (vt["speed"][i, s] + vt["speed"][i, s_end])
                ax2d.plot(
                    r_plot[s:s_end + 1], z_plot[s:s_end + 1],
                    color=pastel_cmap(np.clip(v_seg / vmax, 0.0, 1.0)), linewidth=1.8, alpha=0.90,
                )
            # Flecha direccional en mitad de la traza
            mid = n_pts // 2
            if mid + 1 < n_pts:
                ax2d.annotate(
                    "", xy=(r_plot[mid + 1], z_plot[mid + 1]), xytext=(r_plot[mid], z_plot[mid]),
                    arrowprops=dict(arrowstyle="->", color="#1e293b", lw=1.3),
                )

        ax2d.plot([-Rc, Rc, Rc, -Rc, -Rc], [0, 0, Hc, Hc, 0], color="#64748b", linewidth=1.6)
        ax2d.plot([0, 0], [Hc, zi], color="#475569", linewidth=2.0)
        r_c = 0.5 * d
        if cfg["kind"] == "spin_only":
            ax2d.plot([-r_c, r_c, r_c, -r_c, -r_c], [zi * 0.4, zi * 0.4, zi * 2.3, zi * 2.3, zi * 0.4], color="#4dabf7", linewidth=2.2)
        else:
            ax2d.plot([-r_c, r_c], [zi, zi], color="#fcc419", linewidth=3.5)
        ax2d.set_aspect("equal")
        ax2d.set_xlim(-Rc * 1.06, Rc * 1.06)
        ax2d.set_ylim(0.0, Hc * 1.03)
        ax2d.set_xlabel("Posición radial (±r, cm)", fontsize=8.5)
        ax2d.set_ylabel("Altura z (cm)", fontsize=8.5)
        ax2d.set_title("Proyección Meridional de Vórtices (r–z)", fontsize=11, fontweight="bold", color="#1e293b", pad=8)
        cb2d = fig.colorbar(sm3d, ax=ax2d, shrink=0.78, pad=0.04)
        cb2d.set_label("|v| (m/s)", fontsize=8.5)

        # Tarjeta inferior resumen
        ax_bot = fig.add_subplot(gs[2, :])
        ax_bot.axis("off")
        ax_bot.add_patch(
            plt.Rectangle((0.0, 0.0), 1.0, 1.0, facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=1.2, transform=ax_bot.transAxes)
        )
        ax_bot.text(0.02, 0.85, "RESUMEN HIDRODINÁMICO DE LAS TRAZAS DE VÓRTICES", fontsize=10, fontweight="bold", color="#1e293b", transform=ax_bot.transAxes)
        info_txt = (
            f"• Régimen de flujo: {kpis['regime']} (Re = {kpis['re']:,.0f})   |   Número de potencia Np: {kpis['np']:.2f}   |   "
            f"Potencia: {kpis['power_w']:.2f} W ({kpis['p_over_v']:.0f} W/m³)\n"
            f"• Caudal de bombeo Q: {kpis['q_pump_ls']:.2f} L/s   |   Tiempo de mezcla θ_95: {kpis['t_mix']:.1f} s   |   "
            f"Eficiencia de circulación: {kpis['circ'] * 100:.0f}%\n"
            f"• Velocidad máxima de circulación (r–z): {kpis['v_circ_max_ms']:.3f} m/s   |   "
            f"Velocidad tangencial máxima (remolino): {kpis['v_theta_max_ms']:.3f} m/s   |   "
            f"Escala de Kolmogorov η_K: {kpis['eta_um']:.0f} µm\n"
            f"• Trazas integradas: {n_tr} líneas de corriente RK2 durante {vt['t_s'][-1]:.1f} s ({n_pts} puntos por línea)"
        )
        ax_bot.text(0.02, 0.68, info_txt, fontsize=9.0, color="#334155", va="top", linespacing=1.55, transform=ax_bot.transAxes)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        return buf.getvalue()
    finally:
        plt.close(fig)


def render_fluid_tab() -> None:
    p = _form_defaults("fluid_params", FLUID_DEFAULTS)
    col_c, col_v = st.columns([1.05, 2.55], gap="large")
    with col_c:
        st.subheader("⚙️ Parámetros Hidrodinámicos")
        with st.form("fluid_form"):
            names = list(IMPELLERS)
            imp = st.selectbox("Configuración de impulsor", names, index=names.index(p["impeller"]), key="fl_imp",
                               help="Elige 'Sin hélice' para comparar el flujo de vórtice puro del filtro rotatorio.")
            tank = st.slider("Diámetro del tanque (cm)", 10.0, 50.0, float(p["tank_d_cm"]), 1.0, key="fl_t")
            dt_ = st.slider("Relación D impulsor / D tanque", 0.20, 0.60, float(p["d_over_t"]), 0.01, key="fl_dt")
            rpm = st.slider("Velocidad (RPM)", 30, 400, int(p["rpm"]), 10, key="fl_rpm")
            mu = st.select_slider("Viscosidad del caldo (mPa·s)", options=[1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0],
                                  value=float(p["mu_mpas"]), key="fl_mu")
            nn = st.slider("Índice de flujo n (1=newtoniano, <1 pseudoplástico)", 0.3, 1.0, float(p["n_index"]), 0.05, key="fl_n")
            if st.form_submit_button("🚀 Correr Simulación", width="stretch", type="primary"):
                st.session_state["fluid_params"] = dict(impeller=imp, tank_d_cm=tank, d_over_t=dt_, rpm=rpm,
                                                        mu_mpas=mu, n_index=nn)
        p = st.session_state["fluid_params"]
        sim = run_fluid_solver(**p)
        report_fl = make_report(
            "Dinámica de Fluidos en Tanque Agitado", "Dinámica de Fluidos",
            [("Configuración", p["impeller"], "-"), ("Diámetro tanque / D/T", f"{p['tank_d_cm']} cm / {p['d_over_t']}", "-"),
             ("Agitación", f"{p['rpm']}", "RPM"), ("Viscosidad / índice n", f"{p['mu_mpas']} mPa·s / {p['n_index']}", "-")],
            [("Reynolds", f"{sim['kpis']['re']:.0f} ({sim['kpis']['regime']})", "-"),
             ("Potencia", f"{sim['kpis']['power_w']:.2f}", "W"),
             ("Escala Kolmogorov", f"{sim['kpis']['eta_um']:.1f}", "µm"),
             ("Tiempo de mezcla", f"{sim['kpis']['t_mix']:.1f}", "s")],
            r"1. $Re=\rho N D^2/\mu_{ap}$ · $P=N_p\rho N^3 D^5$ · $\eta_K=(\nu^3/\varepsilon)^{1/4}$",
            [f"Régimen {sim['kpis']['regime']} con circulación del {sim['kpis']['circ']*100:.0f}%."],
        )
        st.download_button("📥 Descargar Memoria de Cálculo (.md)", report_fl.encode("utf-8"),
                           file_name="memoria_dinamica_fluidos.md", mime="text/markdown", width="stretch", key="dl_fluid")

    k = sim["kpis"]
    with col_v:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Reynolds", f"{k['re']:,.0f}", k["regime"], delta_color="off")
        c2.metric("Potencia", f"{k['power_w']:.2f} W", f"{k['p_over_v']:.0f} W/m³", delta_color="off")
        c3.metric("Escala Kolmogorov", f"{k['eta_um']:.0f} µm", "Riesgo células" if k["eta_um"] < 30 else "Seguro",
                  delta_color="inverse" if k["eta_um"] < 30 else "normal")
        c4.metric("Tiempo de mezcla", f"{k['t_mix']:.1f} s", "Lento" if k["t_mix"] > 30 else "Rápido",
                  delta_color="inverse" if k["t_mix"] > 30 else "normal")

        fv_c1, fv_c2 = st.columns(2)
        show_vortex = fv_c1.checkbox("Mostrar trazas de vórtices 3D (líneas de corriente)", value=True, key="fl_vortex")
        show_vec = fv_c2.checkbox("Superponer flechas de velocidad 3D", value=False, key="fl_vec")
        st.plotly_chart(
            build_fluid_figure(sim, p, show_vectors=show_vec, show_vortex_traces=show_vortex),
            width="stretch", key="fluid_3d", config={"displaylogo": False},
        )
        st.session_state.setdefault("show_vortex_png", False)
        vd_col1, vd_col2 = st.columns(2, gap="medium")
        with vd_col1:
            st.download_button(
                "📥 Descargar Traza de Vórtices (.csv)",
                data=generate_vortex_traces_csv(sim, p),
                file_name=f"trazas_vortices_{p['rpm']}rpm.csv",
                mime="text/csv",
                width="stretch",
                key="dl_vortex_csv",
            )
        with vd_col2:
            if not st.session_state["show_vortex_png"]:
                if st.button("🖼️ Generar Imagen de Trazas de Vórtices (.png)", key="btn_gen_vortex_png", width="stretch"):
                    st.session_state["show_vortex_png"] = True
                    st.rerun()
            else:
                vpng = generate_vortex_traces_png(sim, p)
                st.download_button(
                    "📥 Descargar Imagen de Trazas (.png)",
                    data=vpng,
                    file_name=f"trazas_vortices_{p['rpm']}rpm.png",
                    mime="image/png",
                    type="primary",
                    width="stretch",
                    key="dl_vortex_png",
                )
        if st.session_state["show_vortex_png"]:
            with st.expander("👁️ Vista previa de trazas de vórtices (.png)"):
                st.image(generate_vortex_traces_png(sim, p), width="stretch")
        ca, cb = st.columns([1.35, 1])
        ca.plotly_chart(build_np_re_figure(sim, p["impeller"]), width="stretch", key="fluid_np", config={"displaylogo": False})
        with cb:
            st.markdown("##### Análisis de Vórtices e Indicadores")
            st.markdown(
                f"- **Velocidad de punta:** `{k['u_tip']:.2f} m/s`\n"
                f"- **Caudal de bombeo:** `{k['q_pump_ls']:.2f} L/s`\n"
                f"- **Eficiencia circulación:** `{k['circ']*100:.0f} %`\n"
                f"- **Nota sobre vórtices:** Sin hélice (solo filtro rotatorio) el flujo axial cae casi a cero y el tiempo de mezcla aumenta drásticamente frente a un impulsor axial o radial."
            )

        st.divider()
        st.subheader("🧭 Campo Vectorial de Velocidades (Simulación 10 s y Captura)")
        st.caption(
            "Simulación animada 3D de 10 s con flechas de velocidad advectadas por los vórtices + corte 2D. "
            "Usa el botón de pausa para congelar un segundo exacto, ver la ecuación vectorial y descargar el reporte PNG."
        )
        vk1, vk2, vk3 = st.columns(3)
        vk1.metric("|v| máxima", f"{k['v_max_ms']:.3f} m/s")
        vk2.metric("v circulación (r–z) máx", f"{k['v_circ_max_ms']:.3f} m/s")
        vk3.metric("v tangencial (remolino) máx", f"{k['v_theta_max_ms']:.3f} m/s")

        st.session_state.setdefault("fl_vec_paused", False)
        ctrl1, ctrl2, ctrl3 = st.columns([1.1, 1.3, 1.2])
        with ctrl1:
            vdens = st.select_slider(
                "Densidad de vectores 3D",
                options=["Baja", "Media", "Alta"],
                value="Media",
                key="fl_vdens",
            )
        with ctrl2:
            vplane = st.radio(
                "Corte 2D del campo vectorial",
                options=["Meridional (r–z)", "Horizontal (x–y)"],
                horizontal=True,
                key="fl_vplane",
            )
        with ctrl3:
            z_slice = None
            if vplane == "Horizontal (x–y)":
                max_z = max(2.0, float(p["tank_d_cm"]) - 1.0)
                def_z = float(np.clip(round(p["tank_d_cm"] / 3.0, 1), 1.0, max_z))
                z_slice = st.slider(
                    "Altura de corte z (cm)",
                    min_value=1.0,
                    max_value=max_z,
                    value=def_z,
                    step=0.5,
                    key="fl_vz",
                )

        vec_anim = run_fluid_vector_animation(sim["vel_params"], sim["cfg"]["kind"], sim["cfg"]["blades"], p["rpm"], vdens)

        if not st.session_state["fl_vec_paused"]:
            if st.button("⏸ Pausar en un segundo y Descargar Imagen (.png)", key="btn_fl_pause", type="secondary"):
                st.session_state["fl_vec_paused"] = True
                st.rerun()

            vcol1, vcol2 = st.columns(2, gap="medium")
            vcol1.plotly_chart(
                build_fluid_vector_3d_figure(sim, p, vec_anim, paused_frame=None),
                width="stretch",
                key="fluid_vec_3d",
                config={"displaylogo": False},
            )
            vcol2.plotly_chart(
                build_fluid_vector_2d_figure(sim, p, vplane, z_slice, t_s=ANIM_SECONDS),
                width="stretch",
                key="fluid_vec_2d",
                config={"displaylogo": False},
            )
        else:
            pbar1, pbar2 = st.columns([1, 2.2])
            with pbar1:
                if st.button("▶ Reanudar Simulación 3D (10 s)", key="btn_fl_resume", type="primary"):
                    st.session_state["fl_vec_paused"] = False
                    st.rerun()
            with pbar2:
                pause_t = st.slider(
                    "Instante pausado t (s)",
                    min_value=0.0,
                    max_value=float(ANIM_SECONDS),
                    value=4.0,
                    step=0.2,
                    key="fl_pause_t",
                )
            frame_idx = int(np.clip(round(pause_t / (ANIM_SECONDS / N_FRAMES)), 0, N_FRAMES))

            vcol1, vcol2 = st.columns(2, gap="medium")
            vcol1.plotly_chart(
                build_fluid_vector_3d_figure(sim, p, vec_anim, paused_frame=frame_idx),
                width="stretch",
                key="fluid_vec_3d",
                config={"displaylogo": False},
            )
            vcol2.plotly_chart(
                build_fluid_vector_2d_figure(sim, p, vplane, z_slice, t_s=pause_t),
                width="stretch",
                key="fluid_vec_2d",
                config={"displaylogo": False},
            )

            st.markdown("**Ecuación Vectorial en el instante pausado:**")
            st.latex(
                r"\mathbf{v}(r,\theta,z,t) = v_r\,\hat{e}_r + v_\theta\,\hat{e}_\theta + v_z\,\hat{e}_z,\qquad "
                r"v_z = \pm f(t)\,g\,v_c\!\left[1-2(r/R)^2\right]\sin(\pi s),\quad "
                r"f(t) = 0.25 + 0.75\left(1 - e^{-t/1.8}\right)"
            )
            png_bytes = generate_fluid_snapshot_png(sim, p, vec_anim, frame_idx, vplane, z_slice)
            st.download_button(
                label=f"📥 Descargar Imagen del Campo de Velocidades (.png) — t = {pause_t:.1f} s",
                data=png_bytes,
                file_name=f"campo_velocidades_{p['rpm']}rpm_t{pause_t:.1f}s.png",
                mime="image/png",
                type="primary",
                width="stretch",
                key="dl_fluid_vec_png",
            )
            with st.expander("👁️ Vista previa de la imagen PNG descargable"):
                st.image(png_bytes, width="stretch")


# =====================================================================
# 6. MÓDULO 4 — BIOPROCESO, MATERIALES, COSTOS Y PROTOTIPO RÁPIDO
# =====================================================================
def _sorted_cell_keys(reverse: bool = False) -> List[str]:
    return sorted(
        CELL_LINES.keys(),
        key=lambda k: (CELL_LINES[k]["cost_medium_usd_l"], CELL_LINES[k]["cost_vial_usd"]),
        reverse=reverse,
    )


def _sorted_mesh_keys(reverse: bool = False) -> List[str]:
    return sorted(
        MESH_MATERIALS.keys(),
        key=lambda k: MESH_MATERIALS[k]["cost_unit_usd"],
        reverse=reverse,
    )


COST_DEFAULTS = dict(
    cell_line=_sorted_cell_keys()[0],
    mesh_material=_sorted_mesh_keys()[0],
    vol_l=1.0,
    days=3,
    vvd=1.5,
    include_vial=True,
    consumables_usd=40.0,
    sort_order="Menor a mayor costo",
)


def render_bioprocess_tab() -> None:
    st.subheader("🧬 Catálogo Biológico, Materiales Poliméricos y Economía del Experimento")
    st.caption("Opciones ordenadas por costo. Ajusta los parámetros de la calculadora y presiona Calcular Costos.")

    sorted_cells = _sorted_cell_keys(reverse=False)
    sorted_meshes = _sorted_mesh_keys(reverse=False)

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown("#### 1. Líneas Celulares, Medios y Extracción")
        sel_cell = st.selectbox(
            "Inspeccionar línea celular (menor a mayor costo de medio):",
            sorted_cells,
            format_func=lambda k: f"{k} — Medio ${CELL_LINES[k]['cost_medium_usd_l']:.1f}/L · Vial ~${CELL_LINES[k]['cost_vial_usd']:.0f}",
            key="cat_cell",
        )
        ci = CELL_LINES[sel_cell]
        st.markdown(
            f"""<div class="pastel-card">
            <b>Masa y Forma:</b> {ci['mass_ng']} ng/célula · {ci['shape']} · Ø {ci['diam_um']} µm<br>
            <b>Tipo de cultivo:</b> {ci['culture_type']}<br>
            <b>Tasa crecimiento máx (μ_max):</b> {ci['mu_max_h']} h⁻¹ · T óptima: {ci['t_opt_c']} °C<br>
            <b>Medio recomendado:</b> {ci['medium']} (<b>${ci['cost_medium_usd_l']:.1f} USD/L</b>)<br>
            <b>Nutrientes específicos:</b> {ci['nutrients']}<br>
            <b>Producto obtenido:</b> {ci['product']}<br>
            <b>Extracción / Cosecha:</b> {ci['extraction']}<br>
            <b>Proveedores y Costo vial:</b> {ci['suppliers']} · <b>{ci['cost_cell_usd']}</b>
            </div>""",
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown("#### 2. Mallas de Acero y Polímeros Biocompatibles ($T_g$ y $T_m$)")
        sel_mat = st.selectbox(
            "Inspeccionar material de malla o matriz (menor a mayor costo):",
            sorted_meshes,
            format_func=lambda k: f"{k} — ~${MESH_MATERIALS[k]['cost_unit_usd']:.0f} USD",
            key="cat_mat",
        )
        mi = MESH_MATERIALS[sel_mat]
        tg_txt = f"{mi['tg_c']} °C" if mi["tg_c"] is not None else "N/A (Metal)"
        st.markdown(
            f"""<div class="pastel-card">
            <b>Costo módulo/lote ref.:</b> <b>${mi['cost_unit_usd']:.0f} USD</b><br>
            <b>Rango de poro / tejido:</b> {mi['pore_range']} · Área abierta: {int(mi['open_area']*100)} %<br>
            <b>Temperatura transición vítrea (Tg):</b> {tg_txt} · <b>Fusión (Tm):</b> {mi['tm_c']} °C<br>
            <b>Extrusión / Fundición longitudinal:</b> {mi['extrudable']}<br>
            <b>Esterilización:</b> {mi['autoclave']}<br>
            <b>Proveedores y costo ref.:</b> {mi['supplier']}
            </div>""",
            unsafe_allow_html=True,
        )

    st.divider()
    cp = _form_defaults("cost_params", COST_DEFAULTS)
    rev_sort = cp["sort_order"] == "Mayor a menor costo"
    form_cells = _sorted_cell_keys(reverse=rev_sort)
    form_meshes = _sorted_mesh_keys(reverse=rev_sort)

    col_c1, col_c2 = st.columns([1.05, 1.35], gap="large")
    with col_c1:
        st.markdown("#### 3. Calculadora de Costos del Experimento")
        with st.form("cost_form"):
            sort_ord = st.radio(
                "Ordenar opciones y comparativa por costo:",
                ["Menor a mayor costo", "Mayor a menor costo"],
                index=0 if cp["sort_order"] == "Menor a mayor costo" else 1,
                horizontal=True,
                key="cost_sort_ord",
            )
            c_sel = st.selectbox(
                "Línea celular del ensayo",
                form_cells,
                index=form_cells.index(cp["cell_line"]) if cp["cell_line"] in form_cells else 0,
                format_func=lambda k: f"{k} (${CELL_LINES[k]['cost_medium_usd_l']:.1f}/L · Vial ~${CELL_LINES[k]['cost_vial_usd']:.0f})",
                key="cost_cell_sel",
            )
            m_sel = st.selectbox(
                "Material de malla / filtro",
                form_meshes,
                index=form_meshes.index(cp["mesh_material"]) if cp["mesh_material"] in form_meshes else 0,
                format_func=lambda k: f"{k} (~${MESH_MATERIALS[k]['cost_unit_usd']:.0f} USD)",
                key="cost_mesh_sel",
            )
            vol_l = st.slider("Volumen útil del biorreactor (Litros)", 0.25, 10.0, float(cp["vol_l"]), 0.25, key="cost_vol")
            days_c = st.radio("Duración del ensayo", [1, 3, 7], index=[1, 3, 7].index(cp["days"]), horizontal=True, key="cost_days")
            vvd_c = st.slider("Perfusión (VVD)", 0.0, 4.0, float(cp["vvd"]), 0.25, key="cost_vvd")
            inc_vial = st.checkbox("Incluir compra de vial celular inicial", value=bool(cp["include_vial"]), key="cost_vial")
            extras = st.number_input("Consumibles / extras (mangueras, jeringas, etc. USD)", 0.0, 2000.0, float(cp["consumables_usd"]), 5.0, key="cost_extras")

            if st.form_submit_button("🧮 Calcular Costos", type="primary", width="stretch"):
                st.session_state["cost_params"] = dict(
                    cell_line=c_sel,
                    mesh_material=m_sel,
                    vol_l=vol_l,
                    days=days_c,
                    vvd=vvd_c,
                    include_vial=inc_vial,
                    consumables_usd=float(extras),
                    sort_order=sort_ord,
                )

    cp = st.session_state["cost_params"]
    rev_final = cp["sort_order"] == "Mayor a menor costo"
    ci_calc = CELL_LINES[cp["cell_line"]]
    mi_calc = MESH_MATERIALS[cp["mesh_material"]]
    total_liters = cp["vol_l"] * (1.0 + cp["vvd"] * cp["days"])
    cost_medium = total_liters * ci_calc["cost_medium_usd_l"]
    cost_vial = ci_calc["cost_vial_usd"] if cp["include_vial"] else 0.0
    cost_mesh = mi_calc["cost_unit_usd"]
    cost_total = cost_medium + cost_vial + cost_mesh + cp["consumables_usd"]

    with col_c2:
        st.markdown("#### Desglose y Comparativa Ordenada por Costo")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Medio total", f"{total_liters:.1f} L")
        m2.metric("Costo medio", f"${cost_medium:,.0f} USD", f"${ci_calc['cost_medium_usd_l']:.1f}/L", delta_color="off")
        m3.metric("Vial + Malla + Extras", f"${cost_vial + cost_mesh + cp['consumables_usd']:,.0f} USD")
        m4.metric("Costo Total Estimado", f"${cost_total:,.0f} USD", f"${cost_total / cp['days']:,.0f}/día", delta_color="off")

        comp_rows = []
        for cname, cdata in CELL_LINES.items():
            cm = total_liters * cdata["cost_medium_usd_l"]
            cv = cdata["cost_vial_usd"] if cp["include_vial"] else 0.0
            ct = cm + cv + cost_mesh + cp["consumables_usd"]
            comp_rows.append(
                dict(
                    linea=cname,
                    medio_usd=cm,
                    vial_usd=cv,
                    malla_extras_usd=cost_mesh + cp["consumables_usd"],
                    total_usd=ct,
                )
            )
        comp_rows.sort(key=lambda r: r["total_usd"], reverse=rev_final)

        fig_cost = go.Figure()
        labels_c = [r["linea"].split(" (")[0] for r in comp_rows]
        fig_cost.add_trace(go.Bar(y=labels_c, x=[r["medio_usd"] for r in comp_rows], name="Medio cultivo", orientation="h", marker_color="#74c0fc"))
        fig_cost.add_trace(go.Bar(y=labels_c, x=[r["vial_usd"] for r in comp_rows], name="Vial celular", orientation="h", marker_color="#ffa94d"))
        fig_cost.add_trace(go.Bar(y=labels_c, x=[r["malla_extras_usd"] for r in comp_rows], name="Malla + Extras", orientation="h", marker_color="#63e6be"))
        fig_cost.update_layout(
            barmode="stack",
            template="plotly_white",
            height=265,
            margin=dict(l=10, r=10, t=35, b=10),
            title=f"Comparativa ordenada ({cp['sort_order']}) — {cp['days']} día(s), {cp['vol_l']} L, {cp['vvd']} VVD",
            xaxis_title="Costo estimado (USD)",
            yaxis=dict(autorange="reversed"),
            legend=dict(orientation="h", y=-0.28),
        )
        st.plotly_chart(fig_cost, width="stretch", key="cost_bar_chart", config={"displaylogo": False})

        st.dataframe(
            [
                {
                    "Línea Celular": r["linea"],
                    "Medio (USD)": f"${r['medio_usd']:,.1f}",
                    "Vial (USD)": f"${r['vial_usd']:,.0f}",
                    "Malla+Extras (USD)": f"${r['malla_extras_usd']:,.0f}",
                    "Total (USD)": f"${r['total_usd']:,.1f}",
                }
                for r in comp_rows
            ],
            width="stretch",
            hide_index=True,
        )

    st.divider()
    st.markdown("#### 4. Experimentación Rápida y Bajo Costo (Prototipado)")
    st.markdown(
        """
        - **Bombas económicas:** Jeringas de 50 mL montadas en *syringe pump* impreso en 3D o minibombas peristálticas de 12 V con manguera de silicón grado alimenticio.
        - **Materiales reciclables / pruebas rápidas:** Frascos de vidrio borosilicato para pruebas de agitación con imán/hélice impresa en PLA/PETG; agua + glicerina para simular viscosidades de cultivo sin gastar células reales.
        - **Ecuaciones usadas en la app:** Modelo de Monod (crecimiento), Stokes/Saffman (corte y sustentación en malla), Fourier/Newton (calor 3D), Andrade (viscosidad vs. T) y Metzner–Otto (fluido no newtoniano).
        """
    )


# =====================================================================
# 7. INTERFAZ PRINCIPAL
# =====================================================================
def main() -> None:
    st.title("🧬 Simulador Computacional de Biorreactores 3D")
    st.caption(
        "Plataforma interactiva en tema claro: filtración rotatoria, transferencia de calor 3D, "
        "dinámica de fluidos y diseño de bioproceso. Cada simulación animada dura ~10 s."
    )
    tab_f, tab_t, tab_d, tab_b = st.tabs([
        "🌀 Filtración (Mallas)",
        "🌡️ Térmico 3D",
        "🌊 Dinámica de Fluidos",
        "🧪 Bioproceso, Materiales y Costos",
    ])
    with tab_f:
        render_filtration_tab()
    with tab_t:
        render_thermal_tab()
    with tab_d:
        render_fluid_tab()
    with tab_b:
        render_bioprocess_tab()


if __name__ == "__main__":
    main()
