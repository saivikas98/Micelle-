import streamlit as st
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==============================================================================
# MEMBRANE MATERIAL DATABASE (0.22 um STERILIZING GRADE)
# ==============================================================================
MEMBRANE_DB = {
    "Hydrophilic PVDF (Ultra-Low Binding)": {
        "A_sp_m2_g": 18.5,
        "eps_m": 0.76,
        "tau_m": 1.45,
        "rho_p_g_cm3": 1.78,
        "qmax_CsA_I": 0.42,     # mg/g dry membrane (Hydrophobic monomer site)
        "KL_CsA_I": 18.0,       # mM^-1
        "qmax_surf_I": 1.10,    # mg/g
        "KL_surf_I": 24.0,      # mM^-1
        "qmax_surf_II": 0.65,   # mg/g (PEO-40 H-bonding site)
        "KL_surf_II": 1.8,      # mM^-1
        "omega_co": 0.12,       # Co-adsorption factor
        "k_surf_s": 0.18        # Intrinsic attachment rate (s^-1)
    },
    "Hydrophilic PES (Polyethersulfone)": {
        "A_sp_m2_g": 26.0,
        "eps_m": 0.78,
        "tau_m": 1.55,
        "rho_p_g_cm3": 1.37,
        "qmax_CsA_I": 1.65,
        "KL_CsA_I": 52.0,
        "qmax_surf_I": 2.80,
        "KL_surf_I": 45.0,
        "qmax_surf_II": 3.90,   # Sulfone-PEO hydrogen bonding
        "KL_surf_II": 4.2,
        "omega_co": 0.38,
        "k_surf_s": 0.25
    },
    "Nylon 6,6 (High Amide H-Bonding - Worst Case)": {
        "A_sp_m2_g": 31.0,
        "eps_m": 0.72,
        "tau_m": 1.70,
        "rho_p_g_cm3": 1.14,
        "qmax_CsA_I": 3.40,
        "KL_CsA_I": 95.0,
        "qmax_surf_I": 4.90,
        "KL_surf_I": 70.0,
        "qmax_surf_II": 7.80,   # Strong amide-PEO hydrogen bonding
        "KL_surf_II": 7.5,
        "omega_co": 0.62,
        "k_surf_s": 0.32
    }
}

# ==============================================================================
# 1D METHOD-OF-LINES PDE SOLVER FOR 0.22 um MEMBRANE ADSORPTION & BREAKTHROUGH
# ==============================================================================
def simulate_sterile_filtration_adsorption(
    mem_name, A_EFA_cm2, L_m_um, Q_filt_mL_min, V_tot_mL,
    c_hco_wt=1.00, c_oc40_wt=0.05, c_csa_wt=0.09, T_C=25.0
):
    mem = MEMBRANE_DB[mem_name]
    eps_m = mem["eps_m"]
    tau_m = mem["tau_m"]
    rho_p = mem["rho_p_g_cm3"]  # g/cm^3 of solid polymer
    A_sp = mem["A_sp_m2_g"]     # m^2/g

    # Geometry & Flow conversions (working in cm, g, mL, s)
    L_cm = L_m_um * 1e-4
    A_cm2 = A_EFA_cm2
    Q_cm3_s = Q_filt_mL_min / 60.0
    J_v_cm_s = Q_cm3_s / A_cm2
    u_z_cm_s = J_v_cm_s / eps_m

    # Total dry membrane mass (g) and pore volume (mL)
    V_bulk_cm3 = A_cm2 * L_cm
    V_pore_mL = V_bulk_cm3 * eps_m
    M_mem_g = V_bulk_cm3 * (1.0 - eps_m) * rho_p
    solid_to_void = ((1.0 - eps_m) / eps_m) * rho_p  # g_solid / mL_void

    # Feed Concentrations (mg/mL and mM)
    MW_surf_avg = 2460.0
    MW_CsA = 1202.61
    C_surf_in_mg = (c_hco_wt + c_oc40_wt) * 10.0     # mg/mL (e.g., 10.5 mg/mL)
    C_CsA_in_mg = c_csa_wt * 10.0                    # mg/mL (e.g., 0.90 mg/mL)
    C_surf_in_mM = (C_surf_in_mg / MW_surf_avg) * 1e3
    C_CsA_in_mM = (C_CsA_in_mg / MW_CsA) * 1e3

    # Thermodynamic Micelle-Monomer Parameters (coupled from Rubingh / Flory-Huggins)
    CMC_mix_mM = 0.0082
    CMC_mix_mg = CMC_mix_mM * MW_surf_avg / 1e3      # ~0.020 mg/mL
    K_m_part = 480.0                                 # Micelle-water partition factor (mM^-1)
    kappa_mg_mg = 0.098                              # mg CsA solubilized per mg micellar surfactant
    S_w_mg = 0.023                                   # Intrinsic aqueous CsA solubility (mg/mL)

    # Pore-Scale Graetz-Leveque Mass Transfer & LDF Rate Coefficient
    d_pore_cm = 0.22e-4                              # 0.22 um = 2.2e-5 cm
    r_p0_nm = 110.0
    D_eff_cm2_s = 2.4e-7                             # Micelle/monomer diffusivity in PVP vehicle
    Gz = (u_z_cm_s * (d_pore_cm ** 2)) / (D_eff_cm2_s * L_cm * tau_m)
    Sh = 3.66 + (0.0668 * Gz) / (1.0 + 0.04 * (Gz ** (2.0 / 3.0)))
    k_f_cm_s = (Sh * D_eff_cm2_s) / d_pore_cm
    a_v_cm_inv = (4.0 * eps_m) / d_pore_cm

    # Effective LDF mass transfer coefficients (s^-1)
    k_LDF_surf = 1.0 / (1.0 / mem["k_surf_s"] + 12.0 / (k_f_cm_s * a_v_cm_inv))
    k_LDF_CsA = 1.0 / (1.0 / (mem["k_surf_s"] * 0.85) + 15.0 / (k_f_cm_s * a_v_cm_inv))

    # Axial Dispersion Coefficient (cm^2/s)
    D_ax = (D_eff_cm2_s / tau_m) + 0.5 * u_z_cm_s * d_pore_cm * 15.0

    # Spatial Discretization (N_z = 25 axial nodes across membrane thickness)
    N_z = 25
    dz = L_cm / N_z

    # Isotherm helper function returning q_surf* (mg/g) and q_CsA* (mg/g)
    def compute_equilibrium_q(C_surf_mg_arr, C_CsA_mg_arr):
        C_surf_mM = np.maximum(0.0, C_surf_mg_arr / MW_surf_avg * 1e3)
        C_CsA_mM = np.maximum(0.0, C_CsA_mg_arr / MW_CsA * 1e3)

        # Free monomer vs micellar surfactant (mM)
        C_surf_free_mM = np.minimum(C_surf_mM, CMC_mix_mM)
        C_mic_mM = np.maximum(0.0, C_surf_mM - CMC_mix_mM)

        # Free unencapsulated CsA via Flory-Huggins partition K_m
        f_free_CsA = 1.0 / (1.0 + K_m_part * (C_mic_mM * 1e-3))
        C_CsA_free_mM = f_free_CsA * C_CsA_mM

        # Competitive Site I (Hydrophobic Monomer Binding)
        denom_I = 1.0 + mem["KL_surf_I"] * C_surf_free_mM + mem["KL_CsA_I"] * C_CsA_free_mM
        q_surf_I = (mem["qmax_surf_I"] * mem["KL_surf_I"] * C_surf_free_mM) / denom_I
        q_CsA_I = (mem["qmax_CsA_I"] * mem["KL_CsA_I"] * C_CsA_free_mM) / denom_I

        # Cooperative Site II (PEO-40 Corona H-Bonding)
        q_surf_II = (mem["qmax_surf_II"] * mem["KL_surf_II"] * C_mic_mM) / (1.0 + mem["KL_surf_II"] * C_mic_mM)

        # Admicellar co-adsorbed CsA carried by Site-II bound micelles
        payload_ratio = np.where(C_surf_mg_arr > 1e-5, C_CsA_mg_arr / np.maximum(1e-4, C_surf_mg_arr), 0.0)
        q_CsA_II = mem["omega_co"] * payload_ratio * q_surf_II

        return (q_surf_I + q_surf_II), (q_CsA_I + q_CsA_II)

    # State vector y of length 4 * N_z: [C_surf(N_z), C_CsA(N_z), q_surf(N_z), q_CsA(N_z)]
    y0 = np.zeros(4 * N_z)
    t_end_s = V_tot_mL / Q_cm3_s
    t_eval = np.linspace(0.0, t_end_s, 160)

    def adr_ode(t, y):
        C_s = np.maximum(0.0, y[0:N_z])
        C_d = np.maximum(0.0, y[N_z:2*N_z])
        q_s = np.maximum(0.0, y[2*N_z:3*N_z])
        q_d = np.maximum(0.0, y[3*N_z:4*N_z])

        # Solid-phase LDF adsorption rates (mg / g / s)
        q_s_star, q_d_star = compute_equilibrium_q(C_s, C_d)
        dq_s_dt = k_LDF_surf * (q_s_star - q_s)
        dq_d_dt = k_LDF_CsA * (q_d_star - q_d)

        # Upwind convection + central axial dispersion
        dC_s_dt = np.zeros(N_z)
        dC_d_dt = np.zeros(N_z)

        # Inlet node (z = 0) with Danckwerts flux boundary
        dC_s_dt[0] = (u_z_cm_s * (C_surf_in_mg - C_s[0]) / dz) + D_ax * (C_s[1] - C_s[0]) / (dz ** 2) - solid_to_void * dq_s_dt[0]
        dC_d_dt[0] = (u_z_cm_s * (C_CsA_in_mg - C_d[0]) / dz) + D_ax * (C_d[1] - C_d[0]) / (dz ** 2) - solid_to_void * dq_d_dt[0]

        # Interior nodes
        for i in range(1, N_z - 1):
            conv_s = -u_z_cm_s * (C_s[i] - C_s[i - 1]) / dz
            diff_s = D_ax * (C_s[i + 1] - 2.0 * C_s[i] + C_s[i - 1]) / (dz ** 2)
            dC_s_dt[i] = conv_s + diff_s - solid_to_void * dq_s_dt[i]

            conv_d = -u_z_cm_s * (C_d[i] - C_d[i - 1]) / dz
            diff_d = D_ax * (C_d[i + 1] - 2.0 * C_d[i] + C_d[i - 1]) / (dz ** 2)
            dC_d_dt[i] = conv_d + diff_d - solid_to_void * dq_d_dt[i]

        # Outlet node (z = L_m, zero diffusive gradient dC/dz = 0)
        dC_s_dt[-1] = -u_z_cm_s * (C_s[-1] - C_s[-2]) / dz + D_ax * (C_s[-2] - C_s[-1]) / (dz ** 2) - solid_to_void * dq_s_dt[-1]
        dC_d_dt[-1] = -u_z_cm_s * (C_d[-1] - C_d[-2]) / dz + D_ax * (C_d[-2] - C_d[-1]) / (dz ** 2) - solid_to_void * dq_d_dt[-1]

        return np.concatenate([dC_s_dt, dC_d_dt, dq_s_dt, dq_d_dt])

    sol = solve_ivp(adr_ode, [0.0, t_end_s], y0, t_eval=t_eval, method="BDF")

    # Post-Process Filtrate Breakthrough, TMP Rise, and Pore Supersaturation
    V_filt_arr = sol.t * Q_cm3_s
    records = []
    for idx, V_mL in enumerate(V_filt_arr):
        C_s_out = max(0.0, sol.y[N_z - 1, idx])
        C_d_out = max(0.0, sol.y[2 * N_z - 1, idx])
        q_s_prof = sol.y[2 * N_z:3 * N_z, idx]
        q_d_prof = sol.y[3 * N_z:4 * N_z, idx]

        # Adsorbed hydrated brush layer thickness (nm) along pore walls
        # Density of hydrated PEO brush ~ 1.08 g/cm^3 -> 1080 mg/cm^3
        delta_ads_nm = ((q_s_prof + q_d_prof) / (1080.0 * A_sp)) * 1e5
        r_eff_nm = np.maximum(25.0, r_p0_nm - delta_ads_nm)
        tmp_ratio = float(np.mean((r_p0_nm / r_eff_nm) ** 4))

        # Local pore supersaturation ratio at outlet
        C_mic_out_mg = max(0.0, C_s_out - CMC_mix_mg)
        S_eq_out_mg = S_w_mg + kappa_mg_mg * C_mic_out_mg
        S_R_pore = C_d_out / max(1e-4, S_eq_out_mg) if C_d_out > 0.01 else 0.0

        records.append({
            "V_filt_mL": V_mL,
            "V_per_Area_L_m2": (V_mL * 1e-3) / (A_cm2 * 1e-4),
            "CsA_Assay_pct": (C_d_out / C_CsA_in_mg) * 100.0,
            "Surf_Assay_pct": (C_s_out / C_surf_in_mg) * 100.0,
            "CsA_Out_mg_mL": C_d_out,
            "Mean_q_CsA_mg_g": float(np.mean(q_d_prof)),
            "Mean_q_Surf_mg_g": float(np.mean(q_s_prof)),
            "Adsorbed_Layer_nm": float(np.mean(delta_ads_nm)),
            "TMP_Ratio": tmp_ratio,
            "Pore_S_R": S_R_pore
        })

    df_bt = pd.DataFrame(records)

    # Calculate Minimum Pre-Flush Discard Volume to hit >= 98.0% and >= 99.5% instantaneous assay
    above_98 = df_bt[df_bt["CsA_Assay_pct"] >= 98.0]
    V_flush_98_mL = float(above_98["V_filt_mL"].iloc[0]) if not above_98.empty else V_tot_mL
    above_995 = df_bt[df_bt["CsA_Assay_pct"] >= 99.5]
    V_flush_995_mL = float(above_995["V_filt_mL"].iloc[0]) if not above_995.empty else V_tot_mL

    # Total mass of CsA adsorbed onto filter at end of filtration (mg)
    total_CsA_lost_mg = df_bt["Mean_q_CsA_mg_g"].iloc[-1] * M_mem_g
    total_Surf_lost_mg = df_bt["Mean_q_Surf_mg_g"].iloc[-1] * M_mem_g

    # Discrete BFS Single-Use Vial Potency (0.25 mL fill per vial) for first 200 vials (without pre-flush)
    V_vial = 0.25
    vial_rows = []
    for n in range(1, min(400, int(V_tot_mL / V_vial)) + 1):
        v_mid = (n - 0.5) * V_vial
        assay_vial = float(np.interp(v_mid, df_bt["V_filt_mL"], df_bt["CsA_Assay_pct"]))
        surf_vial = float(np.interp(v_mid, df_bt["V_filt_mL"], df_bt["Surf_Assay_pct"]))
        vial_rows.append({
            "BFS_Vial_Number": n,
            "Cumulative_Vol_mL": n * V_vial,
            "Vial_CsA_Assay_pct": assay_vial,
            "Vial_Surf_Assay_pct": surf_vial
        })
    df_vials = pd.DataFrame(vial_rows)

    return {
        "df_bt": df_bt,
        "df_vials": df_vials,
        "summary": {
            "M_mem_g": M_mem_g,
            "V_pore_mL": V_pore_mL,
            "V_flush_98_mL": V_flush_98_mL,
            "V_flush_995_mL": V_flush_995_mL,
            "BFS_Vials_Lost_98": int(np.ceil(V_flush_98_mL / 0.25)),
            "Total_CsA_Adsorbed_mg": total_CsA_lost_mg,
            "Total_Surf_Adsorbed_mg": total_Surf_lost_mg,
            "Max_Brush_Thickness_nm": float(df_bt["Adsorbed_Layer_nm"].max()),
            "Max_TMP_Increase_pct": float((df_bt["TMP_Ratio"].max() - 1.0) * 100.0),
            "Max_Pore_SR": float(df_bt["Pore_S_R"].max())
        }
    }


# ==============================================================================
# STREAMLIT UI RENDERER FOR STERILE FILTRATION TAB
# ==============================================================================
def render_sterile_filtration_tab():
    st.subheader("0.22 μm Sterile Filtration Membrane Adsorption, Breakthrough & BFS Pre-Flush Sizing")

    col_cfg, col_plots = st.columns([1, 3])
    with col_cfg:
        mem_choice = st.selectbox(
            "Select 0.22 μm Membrane Polymer:",
            list(MEMBRANE_DB.keys()),
            index=1
        )
        A_efa = st.slider("Effective Filtration Area A_EFA (cm²)", 100.0, 5000.0, 1000.0, 100.0,
                          help="1000 cm² (0.1 m²) is a standard 4-inch pilot capsule; 5000 cm² (0.5 m²) is a 10-inch cartridge.")
        L_bed = st.slider("Effective Membrane Thickness L_m (μm)", 120.0, 450.0, 240.0, 20.0,
                          help="Dual-layer (0.45/0.22 μm) sterilizing membranes typically range 220–300 μm.")
        Q_flow = st.slider("Filtration Flow Rate Q (mL/min)", 20.0, 1000.0, 150.0, 10.0)
        V_sim = st.slider("Simulated Initial Filtrate Volume (mL)", 100.0, 1500.0, 500.0, 50.0)

    res = simulate_sterile_filtration_adsorption(mem_choice, A_efa, L_bed, Q_flow, V_sim)
    s = res["summary"]
    df_bt = res["df_bt"]
    df_vials = res["df_vials"]

    # Top KPI Metrics
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Pre-Flush to 98% Assay", f"{s['V_flush_98_mL']:.1f} mL", f"{s['BFS_Vials_Lost_98']} BFS Vials")
    k2.metric("Pre-Flush to 99.5% Assay", f"{s['V_flush_995_mL']:.1f} mL", f"{s['V_flush_995_mL']/A_efa*10:.2f} L/m²")
    k3.metric("Total CsA Adsorbed", f"{s['Total_CsA_Adsorbed_mg']:.2f} mg", f"Membrane: {s['M_mem_g']:.1f} g")
    k4.metric("Total Surfactant Adsorbed", f"{s['Total_Surf_Adsorbed_mg']:.1f} mg", "Site I + Site II")
    k5.metric("Adsorbed Brush δ_ads", f"{s['Max_Brush_Thickness_nm']:.2f} nm", f"ΔP_TMP +{s['Max_TMP_Increase_pct']:.1f}%")
    k6.metric("Peak Pore S_R Wave", f"{s['Max_Pore_SR']:.3f}", "Safe (< 1.0)" if s["Max_Pore_SR"] <= 1.0 else "Pore Nucleation Risk!")

    with col_plots:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                "Filtrate Breakthrough Curve: CsA vs. Mixed Surfactant (% Label Claim)",
                "Cross-Polymer Comparison of CsA Breakthrough (PVDF vs. PES vs. Nylon)"
            )
        )

        # Plot 1: Selected Membrane CsA vs Surfactant Breakthrough
        fig.add_trace(
            go.Scatter(
                x=df_bt["V_filt_mL"], y=df_bt["CsA_Assay_pct"],
                mode="lines", name="Cyclosporine A Filtrate Assay (%)",
                line=dict(color="#d62728", width=3.0)
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df_bt["V_filt_mL"], y=df_bt["Surf_Assay_pct"],
                mode="lines", name="Mixed Surfactant (HCO-40 + OC-40) (%)",
                line=dict(color="#1f77b4", width=2.5, dash="dash")
            ),
            row=1, col=1
        )
        fig.add_hline(y=98.0, line_dash="dot", line_color="green",
                      annotation_text="98.0% Release Spec Limit", row=1, col=1)
        fig.add_vline(x=s["V_flush_98_mL"], line_dash="dash", line_color="black",
                      annotation_text=f"Discard Flush = {s['V_flush_98_mL']:.0f} mL", row=1, col=1)

        # Plot 2: Multi-Polymer Comparison at Current Geometry & Flow
        colors = {"Hydrophilic PVDF (Ultra-Low Binding)": "#2ca02c",
                  "Hydrophilic PES (Polyethersulfone)": "#ff7f0e",
                  "Nylon 6,6 (High Amide H-Bonding - Worst Case)": "#9467bd"}
        for m_name, c_hex in colors.items():
            r_cmp = simulate_sterile_filtration_adsorption(m_name, A_efa, L_bed, Q_flow, V_sim)
            fig.add_trace(
                go.Scatter(
                    x=r_cmp["df_bt"]["V_filt_mL"],
                    y=r_cmp["df_bt"]["CsA_Assay_pct"],
                    mode="lines",
                    name=f"{m_name.split(' (')[0]} (Flush: {r_cmp['summary']['V_flush_98_mL']:.0f} mL)",
                    line=dict(color=c_hex, width=2.8)
                ),
                row=1, col=2
            )
        fig.add_hline(y=98.0, line_dash="dot", line_color="green", row=1, col=2)

        fig.update_xaxes(title_text="Cumulative Filtrate Volume V_filt (mL)", row=1, col=1)
        fig.update_yaxes(title_text="Outlet Filtrate Potency (% of Feed)", range=[0, 105], row=1, col=1)
        fig.update_xaxes(title_text="Cumulative Filtrate Volume V_filt (mL)", row=1, col=2)
        fig.update_yaxes(title_text="CsA Filtrate Potency (% of Feed)", range=[0, 105], row=1, col=2)
        fig.update_layout(height=450, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # Bottom Row: BFS Vial-by-Vial Potency & Adsorbed Layer / TMP Dynamics
    fig2 = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Unflushed Blow-Fill-Seal (BFS) Vial-by-Vial Assay (0.25 mL Single-Use Vials)",
            "Internal Pore Adsorbed Brush Thickness δ_ads (nm) & Local Pore S_R"
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": True}]]
    )
    fig2.add_trace(
        go.Scatter(
            x=df_vials["BFS_Vial_Number"], y=df_vials["Vial_CsA_Assay_pct"],
            mode="lines+markers", marker=dict(size=4),
            name="Individual 0.25 mL BFS Vial Assay (%)",
            line=dict(color="#d62728", width=2.0)
        ),
        row=1, col=1
    )
    fig2.add_hline(y=98.0, line_dash="dash", line_color="green",
                   annotation_text="98% Spec Lower Limit", row=1, col=1)

    fig2.add_trace(
        go.Scatter(
            x=df_bt["V_filt_mL"], y=df_bt["Adsorbed_Layer_nm"],
            mode="lines", name="Adsorbed Brush Layer δ_ads (nm)",
            line=dict(color="#1f77b4", width=2.5)
        ),
        row=1, col=2, secondary_y=False
    )
    fig2.add_trace(
        go.Scatter(
            x=df_bt["V_filt_mL"], y=df_bt["Pore_S_R"],
            mode="lines", name="Local Pore Supersaturation Ratio S_R",
            line=dict(color="#ff7f0e", width=2.5, dash="dot")
        ),
        row=1, col=2, secondary_y=True
    )

    fig2.update_xaxes(title_text="Sequential BFS Vial Number (0.25 mL Fill Each)", row=1, col=1)
    fig2.update_yaxes(title_text="Vial Cyclosporine A Assay (%)", range=[0, 105], row=1, col=1)
    fig2.update_xaxes(title_text="Cumulative Filtrate Volume (mL)", row=1, col=2)
    fig2.update_yaxes(title_text="Adsorbed Layer Thickness δ_ads (nm)", secondary_y=False, row=1, col=2)
    fig2.update_yaxes(title_text="Local Pore Supersaturation Ratio S_R", secondary_y=True, row=1, col=2)
    fig2.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig2, use_container_width=True)


if __name__ == "__main__":
    st.set_page_config(page_title="0.22 μm Sterile Filtration Adsorption Simulator", layout="wide")
    render_sterile_filtration_tab()
