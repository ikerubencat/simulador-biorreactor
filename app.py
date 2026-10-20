"""
Simulador Interactivo de Biorreactores 3D
=========================================
Arquitectura base en Streamlit, NumPy y Plotly con tres módulos animados
(cada simulación se reproduce en ~10 s, paso a paso, con botón ▶ / ⏸ y slider de tiempo):

1. Filtración Rotatoria (Spin-Filter): cinemática de células frente a una malla rotatoria.
2. Térmico 3D: calentamiento/enfriamiento por chaqueta + calor metabólico (T(r, z, t)).
3. Dinámica de Fluidos: tanque agitado (impulsor, Re, Np, potencia, mezcla, trazadores).

Cada módulo sigue el mismo patrón desacoplado:
    run_<modulo>_solver(...)   -> diccionario con arrays (aquí inyectas tus ecuaciones)
    build_<modulo>_figure(...) -> figura Plotly animada
    render_<modulo>_tab()      -> interfaz Streamlit (form + KPIs + gráficas + reporte .md)
"""

from datetime import datetime
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# =====================================================================
# 0. CONFIGURACIÓN DE PÁGINA Y ESTILOS
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
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    div[data-testid="stMetric"] {
        background-color: rgba(28, 131, 225, 0.08);
        border: 1px solid rgba(28, 131, 225, 0.25);
        padding: 10px 14px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

RHO_FLUID = 1005.0   # kg/m³  (medio de cultivo acuoso, 37 °C)
ANIM_SECONDS = 10.0  # duración de la animación en pantalla
N_FRAMES = 50        # cuadros por animación (-> 200 ms por cuadro)
FRAME_MS = int(ANIM_SECONDS * 1000 / N_FRAMES)


# =====================================================================
# 1. UTILIDADES COMUNES (animación Plotly y reportes Markdown)
# =====================================================================
def add_animation_controls(
    fig: go.Figure,
    frames: Sequence[go.Frame],
    labels: Sequence[str],
    slider_prefix: str,
) -> go.Figure:
    """Agrega frames + botones ▶ Reproducir / ⏸ Pausa / ↺ Inicio + slider de tiempo."""
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
                pad=dict(t=45, r=6),
                showactive=False,
                buttons=[
                    dict(label="▶ Reproducir", method="animate", args=[None, play_args]),
                    dict(
                        label="⏸ Pausa",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                transition=dict(duration=0),
                                mode="immediate",
                            ),
                        ],
                    ),
                    dict(label="↺ Inicio", method="animate", args=jump(names[0])),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                x=0.34,
                len=0.64,
                y=0.0,
                pad=dict(t=45, b=0),
                currentvalue=dict(prefix=slider_prefix, font=dict(size=15)),
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
    """Genera una memoria de cálculo en Markdown (.md) con tablas, ecuaciones y diagnóstico."""
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
    """Inicializa en session_state los parámetros de un módulo (persisten entre reruns)."""
    if key not in st.session_state:
        st.session_state[key] = dict(defaults)
    return st.session_state[key]


# =====================================================================
# 2. MÓDULO 1 — FILTRACIÓN ROTATORIA (SPIN-FILTER)
# =====================================================================
FILT_DEFAULTS = dict(rpm=180, pore_size_um=10.0, cell_diameter_um=14.5, cell_density_ml=5.0e6)


@st.cache_data(show_spinner=False)
def run_filtration_solver(
    rpm: float,
    pore_size_um: float,
    cell_diameter_um: float,
    cell_density_ml: float,
    duration_s: float = ANIM_SECONDS,
    n_frames: int = N_FRAMES,
    seed: int = 7,
) -> Dict[str, Any]:
    """
    Integra en el tiempo (0 -> duration_s) la trayectoria de cada célula frente a la malla.

    Física (modelo reducido, coordenadas cilíndricas en mm):
      * Succión de permeado: arrastra las células hacia la malla (v_in).
      * Sustentación rotatoria: empuja las células lejos de la malla (v_lift ~ RPM^1.5 · d).
      * Contacto con la malla: si d_célula < d_poro pasa (permeado), si no queda retenida.
        La torta (fouling) bloquea poros y reduce la permeación.
      * Limpieza dinámica: células retenidas se desprenden con tasa ~ RPM².
      * Número de partículas ~ densidad celular.

    NOTA PARA EL DESARROLLADOR:
        Sustituye el bloque marcado con `>>> INYECTAR ECUACIONES PROPIAS AQUÍ <<<`
        manteniendo las llaves del diccionario de retorno.
    """
    rng = np.random.default_rng(seed)
    r_mesh, r_vessel, height = 40.0, 120.0, 240.0  # mm
    sub = 4
    dt = duration_s / (n_frames * sub)

    n = int(np.clip(120 + 28 * cell_density_ml / 1e6, 150, 1500))
    diam = np.clip(rng.normal(cell_diameter_um, 0.12 * cell_diameter_um, n), 1.0, None)
    r = np.sqrt(rng.uniform((r_mesh * 1.3) ** 2, (r_vessel * 0.95) ** 2, n))
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    z = rng.uniform(12.0, height - 12.0, n)
    state = np.zeros(n, dtype=int)  # 0 suspensión, 1 retenida, 2 permeada
    pass_target = rng.uniform(0.1, 0.7, n) * r_mesh

    rpm_n = rpm / 600.0
    f_rev = 0.04 + 0.40 * rpm_n               # rev/s mostradas (cámara lenta visual)
    w_vis = 2.0 * np.pi * f_rev
    capacity = 250.0                           # partículas que saturan la malla (100 % fouling)

    # -----------------------------------------------------------------
    # >>> INYECTAR ECUACIONES PROPIAS AQUÍ <<<
    # -----------------------------------------------------------------
    v_suc = 14.0                                          # mm/s succión de permeado
    v_lift = 19.0 * rpm_n**1.5 * (diam / 15.0)            # mm/s sustentación (array por célula)
    detach_rate = 0.02 + 0.9 * rpm_n**2                   # 1/s desprendimiento de la torta
    # -----------------------------------------------------------------

    xs, ys, zs, sts, counts = [], [], [], [], []

    def snapshot() -> None:
        xs.append(r * np.cos(theta))
        ys.append(r * np.sin(theta))
        zs.append(z.copy())
        sts.append(state.copy())
        counts.append([(state == 0).sum(), (state == 1).sum(), (state == 2).sum()])

    for k in range(n_frames + 1):
        snapshot()
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
                frac = min(1.0, (state == 1).sum() / capacity)
                can_pass = (diam < pore_size_um) & (rng.random(n) < 0.9 * (1.0 - frac))
                passing = contact & can_pass
                sticking = contact & ~can_pass
                state[passing] = 2
                r[passing] = r_mesh * 0.92
                state[sticking] = 1
                r[sticking] = r_mesh * 1.03 + rng.uniform(0.0, 5.0, int(sticking.sum())) * (0.5 + frac)

            stuck = state == 1
            theta[stuck] += w_vis * dt
            detach = stuck & (rng.random(n) < 1.0 - np.exp(-detach_rate * dt))
            state[detach] = 0
            r[detach] = r_mesh * 1.12

            perm = state == 2
            r[perm] += (pass_target[perm] - r[perm]) * min(1.0, 4.0 * dt)
            theta[perm] += w_vis * 1.5 * dt
            z[perm] = np.maximum(z[perm] - 30.0 * dt, 2.0)

    counts_arr = np.array(counts)
    t = np.linspace(0.0, duration_s, n_frames + 1)

    # KPIs globales (estado final) + hidrodinámica analítica de la malla
    omega = rpm * 2.0 * np.pi / 60.0
    mu = 1.1e-3
    v_tip = omega * 0.04
    delta_bl = max(1e-4, 0.005 / np.sqrt(1.0 + rpm / 30.0))
    shear = mu * v_tip / delta_bl
    re_rot = RHO_FLUID * omega * 0.04**2 / mu
    retention = float((n - counts_arr[-1, 2]) / n * 100.0)
    fouling = float(min(100.0, counts_arr[-1, 1] / capacity * 100.0))

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
        "kpis": dict(
            retention_efficiency=retention,
            fouling_index=fouling,
            shear_stress_pa=float(shear),
            re_rot=float(re_rot),
            v_tip_ms=float(v_tip),
            bulk=int(counts_arr[-1, 0]),
            fouled=int(counts_arr[-1, 1]),
            permeated=int(counts_arr[-1, 2]),
        ),
    }


def build_filtration_figure(sim: Dict[str, Any], params: Dict[str, Any], show_vessel_wall: bool = True) -> go.Figure:
    """Figura 3D animada: malla rotatoria + células coloreadas por estado + marca de giro."""
    g = sim["geometry"]
    r_mesh, r_vessel, h = g["r_mesh"], g["r_vessel"], g["height"]
    diam, sts = sim["diam"], sim["states"]
    sizes = np.clip(diam * 0.55, 3.0, 14.0)
    n_frames = len(sim["t"])

    th = np.linspace(0, 2 * np.pi, 45)
    zz = np.linspace(0, h, 25)
    thm, zm = np.meshgrid(th, zz)

    fig = go.Figure()
    fig.add_trace(  # 0 malla
        go.Surface(
            x=r_mesh * np.cos(thm), y=r_mesh * np.sin(thm), z=zm,
            name="Malla rotatoria", showscale=False, opacity=0.30,
            colorscale=[[0, "#00b4d8"], [1, "#0077b6"]], hoverinfo="name",
            contours=dict(
                x=dict(show=True, color="rgba(255,255,255,0.3)", width=1),
                y=dict(show=True, color="rgba(255,255,255,0.3)", width=1),
                z=dict(show=True, color="rgba(255,255,255,0.4)", width=1),
            ),
        )
    )
    fig.add_trace(  # 1 pared
        go.Surface(
            x=r_vessel * np.cos(thm), y=r_vessel * np.sin(thm), z=zm,
            name="Pared del biorreactor", showscale=False, opacity=0.08,
            colorscale=[[0, "#adb5bd"], [1, "#dee2e6"]], hoverinfo="skip", visible=show_vessel_wall,
        )
    )
    # 2-3 referencias de escala (poro vs célula) fuera del recipiente
    for zref, label, size, color in (
        (h * 0.95, f"Poro {params['pore_size_um']:.1f} µm", params["pore_size_um"], "white"),
        (h * 0.82, f"Célula {params['cell_diameter_um']:.1f} µm", params["cell_diameter_um"], "#2ec4b6"),
    ):
        fig.add_trace(
            go.Scatter3d(
                x=[-r_vessel * 1.25], y=[0], z=[zref], mode="markers+text", text=[label],
                textposition="middle right", textfont=dict(color="white", size=12),
                marker=dict(size=float(np.clip(size * 0.55, 3, 24)), color=color,
                            symbol="circle-open" if color == "white" else "circle",
                            line=dict(width=3, color=color)),
                name=label, showlegend=False, hoverinfo="skip",
            )
        )

    styles = [
        ("En suspensión", "#2ec4b6", "circle", 0.85),
        ("Retenida en malla (fouling)", "#ff9f1c", "diamond", 0.95),
        ("Permeada (interior filtro)", "#e71d36", "circle", 0.9),
    ]

    def particle_trace(k: int, s: int) -> go.Scatter3d:
        m = sts[k] == s
        name, color, symbol, op = styles[s]
        return go.Scatter3d(
            x=sim["x"][k][m], y=sim["y"][k][m], z=sim["z"][k][m], mode="markers", name=name,
            marker=dict(size=sizes[m], color=color, symbol=symbol, opacity=op,
                        line=dict(width=0.5, color="rgba(255,255,255,0.4)")),
            hovertemplate=f"<b>{name}</b><br>X %{{x:.0f}} · Y %{{y:.0f}} · Z %{{z:.0f}} mm<extra></extra>",
        )

    def rib_trace(k: int) -> go.Scatter3d:
        a = sim["w_vis"] * sim["t"][k]
        xr, yr, zr = [], [], []
        for aa in (a, a + np.pi):
            xr += [r_mesh * np.cos(aa)] * 2 + [None]
            yr += [r_mesh * np.sin(aa)] * 2 + [None]
            zr += [0, h, None]
        return go.Scatter3d(x=xr, y=yr, z=zr, mode="lines", name="Marca de giro del filtro",
                            line=dict(color="#ff4d6d", width=9), hoverinfo="skip")

    for s in range(3):  # 4,5,6
        fig.add_trace(particle_trace(0, s))
    fig.add_trace(rib_trace(0))  # 7

    frames = [
        go.Frame(
            data=[particle_trace(k, 0), particle_trace(k, 1), particle_trace(k, 2), rib_trace(k)],
            traces=[4, 5, 6, 7],
            name=str(k),
        )
        for k in range(n_frames)
    ]
    fig.update_layout(
        template="plotly_dark", height=680, margin=dict(l=0, r=0, t=35, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0.0, bgcolor="rgba(0,0,0,0.4)"),
        scene=dict(
            xaxis=dict(title="X (mm)", range=[-190, 130]), yaxis=dict(title="Y (mm)", range=[-130, 130]),
            zaxis=dict(title="Altura Z (mm)", range=[0, h]), aspectmode="manual", aspectratio=dict(x=1.2, y=0.9, z=1.0),
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.9)),
        ),
    )
    return add_animation_controls(fig, frames, [f"{t:.1f} s" for t in sim["t"]], "Tiempo: ")


def build_filtration_counts_figure(sim: Dict[str, Any]) -> go.Figure:
    """Evolución temporal de cada población celular."""
    c, t = sim["counts"], sim["t"]
    fig = go.Figure()
    for i, (name, color) in enumerate(
        [("En suspensión", "#2ec4b6"), ("Retenidas en malla", "#ff9f1c"), ("Permeadas", "#e71d36")]
    ):
        fig.add_trace(go.Scatter(x=t, y=c[:, i], name=name, mode="lines", line=dict(color=color, width=3)))
    fig.update_layout(
        template="plotly_dark", height=300, margin=dict(l=10, r=10, t=30, b=10),
        title="Evolución de las poblaciones celulares", xaxis_title="Tiempo (s)", yaxis_title="N.º de células",
        legend=dict(orientation="h", y=-0.35),
    )
    return fig


def filtration_report(p: Dict[str, Any], sim: Dict[str, Any]) -> str:
    k = sim["kpis"]
    ratio = p["cell_diameter_um"] / max(p["pore_size_um"], 0.1)
    diags = [
        (f"⚠️ **Alerta de corte:** τ = {k['shear_stress_pa']:.2f} Pa supera ~1.5 Pa (umbral típico células de mamífero)."
         if k["shear_stress_pa"] > 1.5 else f"✅ **Corte seguro:** τ = {k['shear_stress_pa']:.2f} Pa."),
        (f"⚠️ **Fouling alto ({k['fouling_index']:.0f} %):** aumente RPM o el tamaño de poro."
         if k["fouling_index"] > 45 else f"✅ **Fouling controlado ({k['fouling_index']:.0f} %).**"),
        (f"⚠️ **Pérdida de células por permeado:** retención {k['retention_efficiency']:.1f} %; reduzca el poro."
         if k["retention_efficiency"] < 85 else f"✅ **Retención adecuada ({k['retention_efficiency']:.1f} %).**"),
    ]
    return make_report(
        "Filtración Rotatoria en Biorreactor", "Filtración Rotatoria 3D (Spin-Filter)",
        [("Velocidad del filtro (RPM)", f"{p['rpm']}", "RPM"),
         ("Tamaño de poro $d_p$", f"{p['pore_size_um']:.1f}", "µm"),
         ("Diámetro celular $d_c$", f"{p['cell_diameter_um']:.1f}", "µm"),
         ("Relación $d_c/d_p$", f"{ratio:.2f}", "-"),
         ("Densidad celular", f"{p['cell_density_ml']:.2e}", "cel/mL"),
         ("Duración simulada", f"{sim['t'][-1]:.0f}", "s"),
         ("Partículas simuladas", f"{sim['n_particles']}", "-")],
        [("Eficiencia de retención", f"{k['retention_efficiency']:.2f}", "%"),
         ("Índice de fouling", f"{k['fouling_index']:.2f}", "%"),
         ("Esfuerzo de corte en malla", f"{k['shear_stress_pa']:.3f}", "Pa"),
         ("Velocidad tangencial", f"{k['v_tip_ms']:.3f}", "m/s"),
         ("Reynolds rotacional", f"{k['re_rot']:.0f}", "-"),
         ("Células en suspensión / retenidas / permeadas",
          f"{k['bulk']} / {k['fouled']} / {k['permeated']}", "partículas")],
        r"""1. **Velocidad tangencial y Reynolds:** $v_\theta=\omega R_m$, $\;Re_\omega=\rho\,\omega R_m^2/\mu$
2. **Corte en la malla:** $\tau_w\approx\mu\,v_{tip}/\delta_{BL}$
3. **Criterio de paso:** una célula permea si $d_c<d_p$ y el poro no está bloqueado (probabilidad $\propto 1-\text{fouling}$).
4. **Balance radial:** $\dot r = -v_{suc}\sqrt{R_m/r} + v_{lift}(\text{RPM},d_c)$; desprendimiento con tasa $\propto \text{RPM}^2$.""",
        diags,
    )


def render_filtration_tab() -> None:
    p = _form_defaults("filt_params", FILT_DEFAULTS)
    col_c, col_v = st.columns([1, 2.6], gap="large")

    with col_c:
        st.subheader("⚙️ Parámetros")
        st.caption("Ajusta y pulsa **Correr Simulación**. Luego presiona ▶ en la gráfica 3D (dura 10 s).")
        with st.form("filt_form"):
            rpm = st.slider("Velocidad del filtro (RPM)", 10, 600, int(p["rpm"]), 10, key="f_rpm",
                            help="Bajas RPM: las células se pegan a la malla. Altas RPM: la rotación las aleja.")
            pore = st.slider("Tamaño de poro (µm)", 1.0, 40.0, float(p["pore_size_um"]), 0.5, key="f_pore",
                             help="Células más grandes que el poro quedan retenidas.")
            cdiam = st.slider("Diámetro medio celular (µm)", 2.0, 30.0, float(p["cell_diameter_um"]), 0.5, key="f_cd")
            dens = st.slider("Densidad celular (×10⁶ cel/mL)", 0.5, 50.0, float(p["cell_density_ml"] / 1e6), 0.5,
                             key="f_dens", help="Más densidad = más partículas = torta más rápida.")
            if st.form_submit_button("🚀 Correr Simulación", width="stretch", type="primary"):
                st.session_state["filt_params"] = dict(rpm=rpm, pore_size_um=pore, cell_diameter_um=cdiam,
                                                       cell_density_ml=dens * 1e6)

        p = st.session_state["filt_params"]
        sim = run_filtration_solver(**p)
        st.divider()
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
                  "Daño celular" if k["shear_stress_pa"] > 1.5 else "Biocompatible",
                  delta_color="inverse" if k["shear_stress_pa"] > 1.5 else "normal")
        c4.metric("Reynolds rotacional", f"{k['re_rot']:,.0f}", f"V_tip {k['v_tip_ms']:.2f} m/s", delta_color="off")

        show_wall = st.checkbox("Mostrar pared del tanque", True, key="f_wall")
        st.plotly_chart(build_filtration_figure(sim, p, show_wall), width="stretch", key="filt_3d",
                        config={"displaylogo": False})
        st.caption(
            "🔴 Poro de referencia (círculo blanco) vs 🟢 célula promedio: si la célula es mayor, queda retenida. "
            "🟠 retenidas · 🟢 en suspensión · 🔴 permeadas. La marca roja del cilindro muestra el sentido y velocidad de giro."
        )
        st.plotly_chart(build_filtration_counts_figure(sim), width="stretch", key="filt_counts",
                        config={"displaylogo": False})


# =====================================================================
# 3. MÓDULO 2 — TRANSFERENCIA DE CALOR 3D
# =====================================================================
THERM_DEFAULTS = dict(t_init_c=25.0, t_jacket_c=40.0, u_wm2k=150.0, rpm=100, cell_density_ml=20.0e6,
                      q_cell_pw=100.0, sim_minutes=40)


@st.cache_data(show_spinner=False)
def run_thermal_solver(
    t_init_c: float, t_jacket_c: float, u_wm2k: float, rpm: float, cell_density_ml: float,
    q_cell_pw: float, sim_minutes: float, n_frames: int = N_FRAMES,
) -> Dict[str, Any]:
    """
    Campo T(r, z, t) en un biorreactor cilíndrico (diferencias finitas axisimétricas, explícito).

        ∂T/∂t = (1/r)∂/∂r(r α_ef ∂T/∂r) + ∂/∂z(α_ef ∂T/∂z) + q_met/(ρ cp)
        Pared (r=R):  -ρ cp α_ef ∂T/∂r = U (T - T_chaqueta)      Tapa y fondo: adiabáticos

    α_ef = α_molecular + α_turbulenta(RPM)  ->  más agitación, temperatura más uniforme.

    NOTA PARA EL DESARROLLADOR: reemplaza el bucle temporal marcado por tu solver 3D.
    """
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
    alpha = 1.4e-7 + 4.0e-6 * (rpm / 100.0)                       # m²/s difusividad efectiva
    q_src = cell_density_ml * 1e6 * q_cell_pw * 1e-12 / rho_cp   # K/s calor metabólico
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
    target = 37.0
    reached = None
    if abs(t_init_c - target) < 0.05:
        reached = 0.0
    else:
        hit = np.where((tm_a - target) * np.sign(target - t_init_c) >= 0)[0]
        reached = float(times[hit[0]]) if len(hit) else None

    lo = min(t_init_c, t_jacket_c, frames_arr.min()) - 0.5
    hi = max(t_init_c, t_jacket_c, frames_arr.max()) + 0.5
    if hi - lo < 2.0:
        lo, hi = lo - 1.0, hi + 1.0
    return {
        "frames": frames_arr, "times": times, "tmean": tm_a, "tmax": np.array(tmax), "tmin": np.array(tmin),
        "qjacket": np.array(qj), "rc": rc, "zc": zc, "R": R, "H": H, "cmin": float(lo), "cmax": float(hi),
        "kpis": dict(
            t_mean=float(tm_a[-1]), t_max=float(tmax[-1]), t_min=float(tmin[-1]),
            delta_t=float(tmax[-1] - tmin[-1]), q_met_w=float(q_src * rho_cp * vol),
            q_jacket_w=float(qj[-1]), alpha=float(alpha), reached_target_min=reached,
            tau_min=float(rho_cp * R / (2 * max(u_wm2k, 1e-6)) / 60.0),
        ),
    }


def _ext(T: np.ndarray) -> np.ndarray:
    """Extiende T a los bordes (r=0, r=R, z=0, z=H) para dibujar superficies completas."""
    Te = np.concatenate([T[:1], T, T[-1:]], axis=0)
    return np.concatenate([Te[:, :1], Te, Te[:, -1:]], axis=1)


def build_thermal_figure(sim: Dict[str, Any], params: Dict[str, Any]) -> go.Figure:
    """Corte axial (plano x=0) + media pared coloreados por T(r, z, t) y anillos de la chaqueta."""
    R, H = sim["R"] * 100, sim["H"] * 100  # cm
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
    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=np.zeros((len(zp), len(y))), y=np.tile(y, (len(zp), 1)), z=np.tile(zp[:, None], (1, len(y))),
        surfacecolor=section(T0), coloraxis="coloraxis", name="Corte axial", hovertemplate="T=%{surfacecolor:.2f} °C<extra>Corte</extra>"))
    fig.add_trace(go.Surface(
        x=R * np.cos(th)[None, :] * np.ones((len(zp), 1)), y=R * np.sin(th)[None, :] * np.ones((len(zp), 1)),
        z=np.tile(zp[:, None], (1, len(th))), surfacecolor=wall(T0), coloraxis="coloraxis", opacity=0.6,
        name="Pared interna", hovertemplate="T pared=%{surfacecolor:.2f} °C<extra></extra>"))
    # anillos de chaqueta (color = T_chaqueta)
    tt = np.linspace(0, 2 * np.pi, 80)
    jx, jy, jz = [], [], []
    for zl in np.linspace(0, H, 7):
        jx += list(R * 1.07 * np.cos(tt)) + [None]
        jy += list(R * 1.07 * np.sin(tt)) + [None]
        jz += [zl] * len(tt) + [None]
    fig.add_trace(go.Scatter3d(
        x=jx, y=jy, z=jz, mode="lines", name=f"Chaqueta T_j = {params['t_jacket_c']:.0f} °C",
        line=dict(color=np.full(len(jx), params["t_jacket_c"]), coloraxis="coloraxis", width=8), hoverinfo="name"))

    frames = [
        go.Frame(
            data=[go.Surface(surfacecolor=section(Tk)), go.Surface(surfacecolor=wall(Tk))],
            traces=[0, 1], name=str(k))
        for k, Tk in enumerate(sim["frames"])
    ]
    fig.update_layout(
        template="plotly_dark", height=660, margin=dict(l=0, r=0, t=30, b=0), showlegend=True,
        legend=dict(orientation="h", y=1.02, bgcolor="rgba(0,0,0,0.4)"),
        coloraxis=dict(colorscale="RdYlBu_r", cmin=sim["cmin"], cmax=sim["cmax"],
                       colorbar=dict(title="T (°C)", thickness=16, len=0.7)),
        scene=dict(xaxis=dict(title="x (cm)", range=[-R * 1.2, R * 1.2]), yaxis=dict(title="y (cm)", range=[-R * 1.2, R * 1.2]),
                   zaxis=dict(title="z (cm)", range=[0, H]), aspectmode="data", camera=dict(eye=dict(x=2.0, y=1.1, z=0.8))),
    )
    return add_animation_controls(fig, frames, [f"{t:.1f} min" for t in sim["times"]], "Tiempo simulado: ")


def build_thermal_charts(sim: Dict[str, Any], params: Dict[str, Any]) -> Tuple[go.Figure, go.Figure]:
    t = sim["times"]
    f1 = go.Figure()
    f1.add_trace(go.Scatter(x=t, y=sim["tmax"], name="T máx", line=dict(color="#ff6b6b", width=2, dash="dot")))
    f1.add_trace(go.Scatter(x=t, y=sim["tmean"], name="T media", line=dict(color="#ffd166", width=4)))
    f1.add_trace(go.Scatter(x=t, y=sim["tmin"], name="T mín", line=dict(color="#4dabf7", width=2, dash="dot")))
    f1.add_hline(y=params["t_jacket_c"], line=dict(color="#ffffff", dash="dash"), annotation_text="T chaqueta")
    f1.add_hline(y=37.0, line=dict(color="#2ec4b6", dash="dash"), annotation_text="Consigna 37 °C")
    f1.update_layout(template="plotly_dark", height=320, margin=dict(l=10, r=10, t=35, b=10),
                     title="Temperatura del caldo vs tiempo", xaxis_title="Tiempo (min)", yaxis_title="T (°C)",
                     legend=dict(orientation="h", y=-0.3))
    f2 = go.Figure()
    mid = sim["frames"].shape[2] // 2
    last = len(sim["frames"]) - 1
    rr = np.concatenate([[0], sim["rc"] * 100, [sim["R"] * 100]])
    for idx, col in zip([0, last // 4, last // 2, last], ["#4dabf7", "#74c0fc", "#ffd166", "#ff6b6b"]):
        prof = _ext(sim["frames"][idx])[:, mid + 1]
        f2.add_trace(go.Scatter(x=rr, y=prof, name=f"t = {sim['times'][idx]:.0f} min", line=dict(color=col, width=3)))
    f2.update_layout(template="plotly_dark", height=320, margin=dict(l=10, r=10, t=35, b=10),
                     title="Perfil radial a media altura", xaxis_title="Radio (cm)  (0 = eje, derecha = pared)",
                     yaxis_title="T (°C)", legend=dict(orientation="h", y=-0.3))
    return f1, f2


def thermal_report(p: Dict[str, Any], sim: Dict[str, Any]) -> str:
    k = sim["kpis"]
    diags = [
        (f"⚠️ **Gradiente radial alto ({k['delta_t']:.2f} °C):** aumente la agitación; riesgo de zonas calientes/frías."
         if k["delta_t"] > 1.0 else f"✅ **Temperatura homogénea:** ΔT = {k['delta_t']:.2f} °C."),
        (f"⚠️ **Sobrecalentamiento:** T máx = {k['t_max']:.1f} °C > 38.5 °C (riesgo para células de mamífero)."
         if k["t_max"] > 38.5 else f"✅ **T máxima {k['t_max']:.1f} °C** dentro del rango de cultivo."),
        ("ℹ️ La consigna de 37 °C " + (f"se alcanza a los {k['reached_target_min']:.1f} min." if k["reached_target_min"] is not None
                                       else "no se alcanza en el tiempo simulado.")),
    ]
    return make_report(
        "Transferencia de Calor 3D en Biorreactor", "Térmico 3D (T(r, z, t))",
        [("T inicial del medio", f"{p['t_init_c']:.1f}", "°C"), ("T chaqueta", f"{p['t_jacket_c']:.1f}", "°C"),
         ("Coef. global U", f"{p['u_wm2k']:.0f}", "W/m²K"), ("Agitación", f"{p['rpm']}", "RPM"),
         ("Densidad celular", f"{p['cell_density_ml']:.2e}", "cel/mL"), ("Calor por célula", f"{p['q_cell_pw']:.0f}", "pW/cel"),
         ("Tiempo simulado", f"{p['sim_minutes']}", "min")],
        [("T media final", f"{k['t_mean']:.2f}", "°C"), ("T máx / T mín final", f"{k['t_max']:.2f} / {k['t_min']:.2f}", "°C"),
         ("Gradiente ΔT", f"{k['delta_t']:.3f}", "°C"), ("Calor metabólico total", f"{k['q_met_w']:.2f}", "W"),
         ("Potencia por la chaqueta (final)", f"{k['q_jacket_w']:.2f}", "W (− = aporta calor)"),
         ("Difusividad efectiva", f"{k['alpha']:.2e}", "m²/s"), ("Constante de tiempo de pared", f"{k['tau_min']:.1f}", "min")],
        r"""1. **Ecuación de energía axisimétrica:** $\rho c_p \dfrac{\partial T}{\partial t}=\dfrac{1}{r}\dfrac{\partial}{\partial r}\!\left(r\,k_{ef}\dfrac{\partial T}{\partial r}\right)+\dfrac{\partial}{\partial z}\!\left(k_{ef}\dfrac{\partial T}{\partial z}\right)+q_{met}$
2. **Condición en pared:** $-k_{ef}\,\partial T/\partial r|_{R}=U\,(T-T_j)$
3. **Calor metabólico volumétrico:** $q_{met}=C_{cel}\cdot q_{cel}$
4. **Difusividad efectiva:** $\alpha_{ef}=\alpha_{mol}+\alpha_{t}(\text{RPM})$ (la agitación homogeniza)
5. **Constante de tiempo de pared:** $\tau\approx\rho c_p R/(2U)$""",
        diags,
    )


def render_thermal_tab() -> None:
    p = _form_defaults("therm_params", THERM_DEFAULTS)
    col_c, col_v = st.columns([1, 2.6], gap="large")
    with col_c:
        st.subheader("⚙️ Parámetros térmicos")
        st.caption("Ej.: medio frío (25 °C) calentado por la chaqueta (40 °C). Prueba RPM bajas vs altas.")
        with st.form("therm_form"):
            ti = st.slider("T inicial del medio (°C)", 4.0, 45.0, float(p["t_init_c"]), 0.5, key="t_ti")
            tj = st.slider("T de la chaqueta (°C)", 4.0, 50.0, float(p["t_jacket_c"]), 0.5, key="t_tj")
            u = st.slider("Coef. global U (W/m²·K)", 20.0, 600.0, float(p["u_wm2k"]), 10.0, key="t_u",
                          help="Mayor U = intercambio más rápido con la chaqueta.")
            rpm = st.slider("Agitación (RPM)", 20, 600, int(p["rpm"]), 10, key="t_rpm",
                            help="Más RPM = temperatura más uniforme.")
            dens = st.slider("Densidad celular (×10⁶ cel/mL)", 1.0, 100.0, float(p["cell_density_ml"] / 1e6), 1.0, key="t_dens")
            q = st.slider("Calor metabólico (pW/célula)", 10.0, 300.0, float(p["q_cell_pw"]), 10.0, key="t_q")
            mins = st.slider("Tiempo simulado (min)", 5, 120, int(p["sim_minutes"]), 5, key="t_min",
                             help="Se reproduce en 10 s de animación.")
            if st.form_submit_button("🚀 Correr Simulación", width="stretch", type="primary"):
                st.session_state["therm_params"] = dict(t_init_c=ti, t_jacket_c=tj, u_wm2k=u, rpm=rpm,
                                                        cell_density_ml=dens * 1e6, q_cell_pw=q, sim_minutes=mins)
        p = st.session_state["therm_params"]
        sim = run_thermal_solver(**p)
        st.divider()
        st.download_button("📥 Descargar Memoria de Cálculo (.md)", thermal_report(p, sim).encode("utf-8"),
                           file_name="memoria_termico_3d.md", mime="text/markdown", width="stretch", key="dl_therm")

    k = sim["kpis"]
    with col_v:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("T media final", f"{k['t_mean']:.2f} °C")
        c2.metric("Gradiente ΔT", f"{k['delta_t']:.2f} °C", "Homogéneo" if k["delta_t"] <= 1 else "Zonas calientes/frías",
                  delta_color="normal" if k["delta_t"] <= 1 else "inverse")
        c3.metric("Calor metabólico", f"{k['q_met_w']:.1f} W")
        c4.metric("Llega a 37 °C", f"{k['reached_target_min']:.1f} min" if k["reached_target_min"] is not None else "No llega")
        st.plotly_chart(build_thermal_figure(sim, p), width="stretch", key="therm_3d", config={"displaylogo": False})
        st.caption("Plano de corte por el eje del tanque y media pared interior coloreados por temperatura "
                   "(azul = frío, rojo = caliente). Los anillos exteriores tienen el color de la chaqueta.")
        a, b = build_thermal_charts(sim, p)
        ca, cb = st.columns(2)
        ca.plotly_chart(a, width="stretch", key="therm_ts", config={"displaylogo": False})
        cb.plotly_chart(b, width="stretch", key="therm_rad", config={"displaylogo": False})


# =====================================================================
# 4. MÓDULO 3 — DINÁMICA DE FLUIDOS (TANQUE AGITADO)
# =====================================================================
IMPELLERS: Dict[str, Dict[str, Any]] = {
    "Turbina Rushton (radial)": dict(np_t=5.0, kp=70.0, nq=0.72, vcf=0.8, double=True, blades=6, kind="rushton"),
    "Palas inclinadas 45° (axial)": dict(np_t=1.3, kp=50.0, nq=0.75, vcf=0.55, double=False, blades=4, kind="pbt"),
    "Hélice marina (axial)": dict(np_t=0.35, kp=40.0, nq=0.55, vcf=0.7, double=False, blades=3, kind="marine"),
}
FLUID_DEFAULTS = dict(impeller="Turbina Rushton (radial)", tank_d_cm=20.0, d_over_t=0.33, rpm=150, mu_mpas=1.0, n_index=1.0)


def _fluid_velocity(r: np.ndarray, z: np.ndarray, pr: Dict[str, float]):
    """Campo de velocidad (vr, vz, vθ) analítico por función de corriente + remolino."""
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


def _impeller_lines(kind: str, n_blades: int, d: float, zi: float, angle: float):
    """Polilíneas del impulsor (cm) rotadas 'angle'."""
    xs, ys, zs = [], [], []

    def add(pts):
        for rr, aa, zz in pts:
            xs.append(rr * np.cos(aa))
            ys.append(rr * np.sin(aa))
            zs.append(zz)
        xs.append(None)
        ys.append(None)
        zs.append(None)

    if kind == "rushton":
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


@st.cache_data(show_spinner=False)
def run_fluid_solver(
    impeller: str, tank_d_cm: float, d_over_t: float, rpm: float, mu_mpas: float, n_index: float,
    n_frames: int = N_FRAMES, t_phys: float = 2.0, n_tracers: int = 450, seed: int = 3,
) -> Dict[str, Any]:
    """
    Tanque agitado: números adimensionales + trazadores advectados por un campo de velocidad.

      Re = ρ N D² / μ_ap ;  μ_ap = K (k_s N)^(n-1) (Metzner-Otto) ;  Np = Np∞ + Kp/Re
      P = Np ρ N³ D⁵ ;  ε = P/(ρV) ;  η_K = (ν³/ε)^¼ ;  θ_mezcla (Grenville)

    NOTA PARA EL DESARROLLADOR: sustituye el bloque marcado con tu campo de velocidad
    (p. ej. resultados de Navier-Stokes) manteniendo las llaves del diccionario de retorno.
    """
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

    # -----------------------------------------------------------------
    # >>> INYECTAR ECUACIONES PROPIAS AQUÍ (campo de velocidad / CFD) <<<
    # -----------------------------------------------------------------
    f_c = re / (re + 300.0)  # eficiencia de bombeo (bajo Re casi solo hay remolino)
    pr = dict(R=T / 2, H=H, zi=H / 3, d_imp=D, double=cfg["double"], vc=cfg["vcf"] * u_tip * f_c,
              vth=0.45 * u_tip)
    # -----------------------------------------------------------------

    rng = np.random.default_rng(seed)
    r = pr["R"] * np.sqrt(rng.uniform(0.0, 0.9, n_tracers))
    z = rng.uniform(0.04 * H, 0.96 * H, n_tracers)
    th = rng.uniform(0, 2 * np.pi, n_tracers)
    sub = 4
    dt = t_phys / (n_frames * sub)
    px, py, pz, spd = [], [], [], []
    for k in range(n_frames + 1):
        vr, vz, vt = _fluid_velocity(r, z, pr)
        px.append(r * np.cos(th) * 100)
        py.append(r * np.sin(th) * 100)
        pz.append(z * 100)
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
    return {
        "x": np.array(px), "y": np.array(py), "z": np.array(pz), "speed": np.array(spd),
        "t_phys": np.linspace(0, t_phys, n_frames + 1), "cfg": cfg, "geom": dict(T=T, D=D, H=H, zi=H / 3),
        "re_grid": re_grid, "np_curves": curves,
        "kpis": dict(re=float(re), regime=regime, np=float(npw), power_w=float(power), p_over_v=float(power / vol),
                     u_tip=float(u_tip), shear=float(gamma), eps=float(eps), eta_um=float(eta * 1e6),
                     t_mix=float(t_mix), q_pump_ls=float(q_pump * 1000), mu_app=float(mu), circ=float(f_c)),
    }


def build_fluid_figure(sim: Dict[str, Any], params: Dict[str, Any]) -> go.Figure:
    g = sim["geom"]
    Rc, Hc, zi, d = g["T"] * 50, g["H"] * 100, g["zi"] * 100, g["D"] * 100
    cfg = sim["cfg"]
    th = np.linspace(0, 2 * np.pi, 50)
    zz = np.linspace(0, Hc, 20)
    thm, zm = np.meshgrid(th, zz)
    rpm_vis = 0.05 + 0.30 * min(params["rpm"], 400) / 400.0  # rev por segundo de animación
    times = np.arange(len(sim["t_phys"])) * FRAME_MS / 1000.0
    vmax = float(sim["speed"].max()) or 1.0

    fig = go.Figure()
    fig.add_trace(go.Surface(x=Rc * np.cos(thm), y=Rc * np.sin(thm), z=zm, showscale=False, opacity=0.07,
                             colorscale=[[0, "#adb5bd"], [1, "#dee2e6"]], hoverinfo="skip", name="Tanque"))  # 0
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[Hc * 1.02, zi], mode="lines",
                               line=dict(color="#adb5bd", width=6), name="Eje", hoverinfo="skip"))  # 1

    def imp(k: int) -> go.Scatter3d:
        xs, ys, zs = _impeller_lines(cfg["kind"], cfg["blades"], d, zi, 2 * np.pi * rpm_vis * times[k])
        return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color="#ffd43b", width=7),
                            name="Impulsor", hoverinfo="skip")

    def tracers(k: int) -> go.Scatter3d:
        return go.Scatter3d(
            x=sim["x"][k], y=sim["y"][k], z=sim["z"][k], mode="markers", name="Trazadores",
            marker=dict(size=3.5, color=sim["speed"][k], colorscale="Turbo", cmin=0, cmax=vmax, opacity=0.9,
                        colorbar=dict(title="|v| (m/s)", thickness=16, len=0.7)),
            hovertemplate="|v| = %{marker.color:.2f} m/s<extra></extra>")

    fig.add_trace(imp(0))      # 2
    fig.add_trace(tracers(0))  # 3
    frames = [go.Frame(data=[imp(k), tracers(k)], traces=[2, 3], name=str(k)) for k in range(len(times))]
    fig.update_layout(
        template="plotly_dark", height=660, margin=dict(l=0, r=0, t=30, b=0), showlegend=False,
        scene=dict(xaxis=dict(title="x (cm)", range=[-Rc * 1.05, Rc * 1.05]), yaxis=dict(title="y (cm)", range=[-Rc * 1.05, Rc * 1.05]),
                   zaxis=dict(title="z (cm)", range=[0, Hc]), aspectmode="data", camera=dict(eye=dict(x=1.7, y=1.3, z=0.7))),
    )
    return add_animation_controls(fig, frames, [f"{t:.2f} s" for t in sim["t_phys"]], "Tiempo físico: ")


def build_np_re_figure(sim: Dict[str, Any], impeller: str) -> go.Figure:
    k = sim["kpis"]
    fig = go.Figure()
    for name, curve in sim["np_curves"].items():
        fig.add_trace(go.Scatter(x=sim["re_grid"], y=curve, name=name,
                                 line=dict(width=4 if name == impeller else 1.5, dash="solid" if name == impeller else "dot")))
    fig.add_trace(go.Scatter(x=[k["re"]], y=[k["np"]], mode="markers", name="Punto de operación",
                             marker=dict(size=16, color="#ff4d6d", symbol="star", line=dict(width=1, color="white"))))
    fig.update_layout(template="plotly_dark", height=340, margin=dict(l=10, r=10, t=35, b=10),
                      title="Número de potencia vs Reynolds", xaxis_title="Re", yaxis_title="Np",
                      xaxis_type="log", yaxis_type="log", legend=dict(orientation="h", y=-0.3))
    return fig


def fluid_report(p: Dict[str, Any], sim: Dict[str, Any]) -> str:
    k = sim["kpis"]
    diags = [
        (f"⚠️ **Posible daño por turbulencia:** escala de Kolmogorov {k['eta_um']:.0f} µm < 30 µm (≈ 2× el diámetro celular)."
         if k["eta_um"] < 30 else f"✅ **Escala de Kolmogorov {k['eta_um']:.0f} µm**, segura para células en suspensión."),
        (f"⚠️ **Mezcla lenta:** θ ≈ {k['t_mix']:.1f} s; riesgo de gradientes de nutrientes/pH."
         if k["t_mix"] > 30 else f"✅ **Mezcla rápida:** θ ≈ {k['t_mix']:.1f} s."),
        f"ℹ️ **Régimen {k['regime']}** (Re = {k['re']:.0f}); potencia específica {k['p_over_v']:.1f} W/m³.",
    ]
    return make_report(
        "Dinámica de Fluidos en Tanque Agitado", "Dinámica de Fluidos (CFD simplificado)",
        [("Impulsor", p["impeller"], "-"), ("Diámetro del tanque T", f"{p['tank_d_cm']:.1f}", "cm"),
         ("Relación D/T", f"{p['d_over_t']:.2f}", "-"), ("Agitación", f"{p['rpm']}", "RPM"),
         ("Viscosidad (consistencia)", f"{p['mu_mpas']:.1f}", "mPa·s"), ("Índice de flujo n", f"{p['n_index']:.2f}", "-")],
        [("Reynolds de agitación", f"{k['re']:.0f} ({k['regime']})", "-"), ("Número de potencia Np", f"{k['np']:.2f}", "-"),
         ("Potencia P", f"{k['power_w']:.3f}", "W"), ("Potencia específica P/V", f"{k['p_over_v']:.1f}", "W/m³"),
         ("Velocidad de punta", f"{k['u_tip']:.2f}", "m/s"), ("Velocidad de corte promedio", f"{k['shear']:.0f}", "1/s"),
         ("Disipación ε", f"{k['eps']:.3f}", "W/kg"), ("Escala de Kolmogorov η", f"{k['eta_um']:.1f}", "µm"),
         ("Tiempo de mezcla θ", f"{k['t_mix']:.1f}", "s"), ("Caudal de bombeo Q", f"{k['q_pump_ls']:.2f}", "L/s")],
        r"""1. **Reynolds:** $Re=\rho N D^2/\mu_{ap}$, con $\mu_{ap}=K(k_s N)^{n-1}$ (Metzner–Otto, $k_s\approx11$)
2. **Potencia:** $P=N_p\rho N^3D^5$, con $N_p=N_{p,\infty}+K_p/Re$
3. **Disipación y Kolmogorov:** $\varepsilon=P/(\rho V)$, $\;\eta=(\nu^3/\varepsilon)^{1/4}$
4. **Bombeo:** $Q=N_q N D^3$
5. **Mezcla (Grenville):** $\theta_{95}=5.9\,T^{2/3}\varepsilon^{-1/3}(D/T)^{-1/3}$
6. **Trazadores:** advección en un campo $\mathbf{v}(r,z)$ derivado de una función de corriente toroidal + remolino azimutal.""",
        diags,
    )


def render_fluid_tab() -> None:
    p = _form_defaults("fluid_params", FLUID_DEFAULTS)
    col_c, col_v = st.columns([1, 2.6], gap="large")
    with col_c:
        st.subheader("⚙️ Parámetros del tanque")
        st.caption("Prueba subir la viscosidad: el fluido deja de circular y solo gira (régimen laminar).")
        with st.form("fluid_form"):
            names = list(IMPELLERS)
            imp = st.selectbox("Tipo de impulsor", names, index=names.index(p["impeller"]), key="fl_imp")
            tank = st.slider("Diámetro del tanque (cm)", 10.0, 50.0, float(p["tank_d_cm"]), 1.0, key="fl_t")
            dt_ = st.slider("Relación D impulsor / D tanque", 0.20, 0.60, float(p["d_over_t"]), 0.01, key="fl_dt")
            rpm = st.slider("Velocidad del impulsor (RPM)", 30, 400, int(p["rpm"]), 10, key="fl_rpm")
            mu = st.select_slider("Viscosidad del caldo (mPa·s)", options=[1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0],
                                  value=float(p["mu_mpas"]), key="fl_mu", help="Agua ≈ 1 mPa·s.")
            nn = st.slider("Índice de flujo n (1 = newtoniano, <1 pseudoplástico)", 0.3, 1.0, float(p["n_index"]), 0.05, key="fl_n")
            if st.form_submit_button("🚀 Correr Simulación", width="stretch", type="primary"):
                st.session_state["fluid_params"] = dict(impeller=imp, tank_d_cm=tank, d_over_t=dt_, rpm=rpm,
                                                        mu_mpas=mu, n_index=nn)
        p = st.session_state["fluid_params"]
        sim = run_fluid_solver(**p)
        st.divider()
        st.download_button("📥 Descargar Memoria de Cálculo (.md)", fluid_report(p, sim).encode("utf-8"),
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
        st.plotly_chart(build_fluid_figure(sim, p), width="stretch", key="fluid_3d", config={"displaylogo": False})
        st.caption("Trazadores coloreados por velocidad |v|. Rushton: dos lazos de recirculación (flujo radial). "
                   "Axiales: un lazo (bajan por el centro y suben por la pared). El impulsor se dibuja en cámara lenta; "
                   "la animación muestra 2 s físicos en 10 s.")
        a, b = st.columns([1.3, 1])
        a.plotly_chart(build_np_re_figure(sim, p["impeller"]), width="stretch", key="fluid_np", config={"displaylogo": False})
        with b:
            st.markdown("##### Otros resultados")
            st.markdown(
                f"- **Velocidad de punta:** {k['u_tip']:.2f} m/s\n"
                f"- **Corte promedio:** {k['shear']:.0f} s⁻¹\n"
                f"- **Disipación ε:** {k['eps']:.3f} W/kg\n"
                f"- **Caudal de bombeo:** {k['q_pump_ls']:.2f} L/s\n"
                f"- **Np:** {k['np']:.2f}\n"
                f"- **Eficiencia de circulación:** {k['circ'] * 100:.0f} %"
            )


# =====================================================================
# 5. INTERFAZ PRINCIPAL
# =====================================================================
def main() -> None:
    st.title("🧬 Simulador Computacional de Biorreactores 3D")
    st.caption(
        "Plataforma interactiva: filtración rotatoria, transferencia de calor 3D y dinámica de fluidos. "
        "Cada simulación dura ~10 s: pulsa **▶ Reproducir** en la gráfica 3D."
    )
    tab_f, tab_t, tab_d = st.tabs(["🌀 Filtración (Mallas)", "🌡️ Térmico 3D", "🌊 Dinámica de Fluidos"])
    with tab_f:
        render_filtration_tab()
    with tab_t:
        render_thermal_tab()
    with tab_d:
        render_fluid_tab()

    with st.expander("🔌 Guía rápida para inyectar tus ecuaciones numéricas"):
        st.markdown(
            """
            Cada módulo tiene un solver independiente en `app.py` con el bloque `>>> INYECTAR ECUACIONES PROPIAS AQUÍ <<<`:
            - `run_filtration_solver()` → devuelve `x, y, z, states (0 suspensión · 1 retenida · 2 permeada)` por cuadro.
            - `run_thermal_solver()` → devuelve `frames[t, r, z]` con la temperatura.
            - `run_fluid_solver()` → devuelve `x, y, z, speed` de los trazadores por cuadro y los KPIs.

            Mientras conserves las llaves del diccionario de retorno, la interfaz no necesita cambios.
            """
        )


if __name__ == "__main__":
    main()
