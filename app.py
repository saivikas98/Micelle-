import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==============================================================================
# 1. PAGE CONFIGURATION (MUST BE FIRST STREAMLIT COMMAND)
# ==============================================================================
st.set_page_config(
    page_title="Nanomicellar QbD, PBE & Sterile Filtration Workbench",
    layout="wide",
    initial_sidebar_state="expanded"
)

# NumPy 2.0+ compatibility wrapper for trapezoidal integration
def trapz_compat(y, x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)

# Physical Constants
k_B = 1.380649e-23       # Boltzmann constant (J/K)
N_A = 6.02214076e23      # Avogadro's number (1/mol)
R_gas = 8.314462         # Universal gas constant (J/mol*K)

# ==============================================================================
# 2. FIRST-PRINCIPLES THERMODYNAMIC & STABILITY ENGINE
# ==============================================================================
class NanomicelleFormulation:
    def __init__(self, c_hco_wt=1.00, c_oc40_wt=0.05, c_csa_wt=0.09, c_pvp_wt=0.30,
                 pH_0=6.80, ionic_strength=0.125, beta_12=-2.45):
        self.C_HCO_wt = float(c_hco_wt)
        self.C_OC40_wt = float(c_oc40_wt)
        self.C_CsA_wt = float(c_csa_wt)
        self.C_PVP_wt = float(c_pvp_wt)
        self.pH_0 = float(pH_0)
        self.I_ionic = float(ionic_strength)
        self.beta_12 = float(beta_12)
        self.beta_buf = 0.035

        # Molecular Weights (g/mol)
        self.MW_1 = 2500.0     # Polyoxyl 40 Hydrogenated Castor Oil (HCO-40)
        self.MW_2 = 1966.0     # Octoxynol-40 (OC-40)
        self.MW_CsA = 1202.61  # Cyclosporine A

        # Molar Concentrations (mM)
        self.C1_0 = (self.C_HCO_wt * 10.0 / self.MW_1) * 1e3
        self.C2_0 = (self.C_OC40_wt * 10.0 / self.MW_2) * 1e3
        self.C_CsA = (self.C_CsA_wt * 10.0 / self.MW_CsA) * 1e3

        # Pure Surfactant Reference Values (298.15 K)
        self.CMC1_ref = 0.014
        self.CMC2_ref = 0.160
        self.dH_mic1 = -12500.0
        self.dH_mic2 = -9800.0
        self.ks_1 = 1.15
        self.ks_2 = 0.95
        self.kp_1 = 0.18
        self.kp_2 = 0.15

        # Molecular Geometry (nm & nm^3)
        self.vt_1 = 1.48
        self.vt_2 = 0.36
        self.v_CsA = 1.74
        self.a0_shield = 0.42
        self.N_EO = 40.0
        self.l_EO = 0.36
        self.v_EO = 0.065
        self.sigma_core = 0.030

        # Hansen Solubility Parameters [d_d, d_p, d_h] (MPa^0.5)
        self.HSP_1 = np.array([17.2, 4.5, 6.8])
        self.HSP_2 = np.array([18.4, 3.2, 4.1])
        self.HSP_CsA = np.array([17.8, 5.1, 7.4])

        # HCO-40 Ester Hydrolysis Kinetics (month^-1)
        self.Ea_hyd = 78500.0
        self.k_H = 1.8e-2
        self.k_0 = 4.5e-4
        self.k_OH = 8.2e2


def chi_EO_water(T):
    return 0.81 - 110.5 / T


def water_viscosity(T):
    mu_w = 2.414e-5 * (10.0 ** (247.8 / (T - 140.0)))
    return mu_w * 1.35


def solve_mixed_cmc(C1, C2, T, params):
    T_ref = 298.15
    CMC1 = params.CMC1_ref * np.exp(
        (params.dH_mic1 / R_gas) * (1.0 / T - 1.0 / T_ref)
        - params.ks_1 * params.I_ionic
        - params.kp_1 * params.C_PVP_wt
    )
    CMC2 = params.CMC2_ref * np.exp(
        (params.dH_mic2 / R_gas) * (1.0 / T - 1.0 / T_ref)
        - params.ks_2 * params.I_ionic
        - params.kp_2 * params.C_PVP_wt
    )

    C_tot = max(1e-9, C1 + C2)
    alpha_1 = float(np.clip(C1 / C_tot, 1e-5, 1.0 - 1e-5))
    beta = params.beta_12

    def rubingh_res(x1):
        g1 = np.exp(beta * (1.0 - x1) ** 2)
        g2 = np.exp(beta * x1 ** 2)
        cmc_m = 1.0 / (alpha_1 / (g1 * CMC1) + (1.0 - alpha_1) / (g2 * CMC2))
        a1 = (alpha_1 * cmc_m) / (x1 * CMC1)
        a2 = ((1.0 - alpha_1) * cmc_m) / ((1.0 - x1) * CMC2)
        if a1 <= 0 or a2 <= 0:
            return 0.0
        return (x1 ** 2) * np.log(a1) - ((1.0 - x1) ** 2) * np.log(a2)

    try:
        x1 = float(brentq(rubingh_res, 1e-5, 1.0 - 1e-5))
    except Exception:
        x1 = alpha_1

    g1 = float(np.exp(beta * (1.0 - x1) ** 2))
    g2 = float(np.exp(beta * x1 ** 2))
    CMC_mix = float(1.0 / (alpha_1 / (g1 * CMC1) + (1.0 - alpha_1) / (g2 * CMC2)))
    CMC_ideal = float(1.0 / (alpha_1 / CMC1 + (1.0 - alpha_1) / CMC2))

    return {
        "x1": x1, "CMC1": float(CMC1), "CMC2": float(CMC2),
        "CMC_mix": CMC_mix, "CMC_ideal": CMC_ideal,
        "gamma1": g1, "gamma2": g2
    }


def solve_equilibrium_micelle(C1, C2, T, params):
    thermo = solve_mixed_cmc(C1, C2, T, params)
    x1 = thermo["x1"]
    x2 = 1.0 - x1
    CMC_mix = thermo["CMC_mix"]

    v_tail_avg = x1 * params.vt_1 + x2 * params.vt_2
    chi_ew = chi_EO_water(T)
    sigma_kT = (params.sigma_core * 1e-18) / (k_B * T)

    def delta_mu_kT(N_agg):
        V_core = N_agg * v_tail_avg * 1.22
        R_c = ((3.0 * V_core) / (4.0 * np.pi)) ** (1.0 / 3.0)
        area = (4.0 * np.pi * R_c ** 2) / N_agg
        mu_int = sigma_kT * max(0.0, area - params.a0_shield)
        L_tail = x1 * 2.1 + x2 * 1.1
        mu_def = (3.0 * np.pi ** 2 / 80.0) * (R_c ** 2 / (L_tail ** 2)) * 2.8
        solv_qual = max(0.015, 0.5 - chi_ew)
        mu_steric = 1.85 * params.N_EO * (params.v_EO / (area * R_c)) ** 0.85 * solv_qual
        return mu_int + mu_def + mu_steric

    res = minimize_scalar(delta_mu_kT, bounds=(25, 450), method="bounded")
    N_agg = float(res.x)

    V_core = N_agg * v_tail_avg * 1.22
    R_c = float(((3.0 * V_core) / (4.0 * np.pi)) ** (1.0 / 3.0))
    a_opt = (4.0 * np.pi * R_c ** 2) / N_agg
    solv_qual = max(0.008, 0.5 - chi_ew)

    L_corona = float(R_c * (
        (1.0 + (5.0 / 3.0) * (params.N_EO * (params.l_EO ** (5.0 / 3.0))) /
         (R_c * (a_opt ** (1.0 / 3.0))) * (solv_qual ** (1.0 / 3.0))) ** (3.0 / 5.0) - 1.0
    ))
    delta_PVP = 1.15 * (params.C_PVP_wt / 0.30) ** 0.25
    D_h = float(2.0 * (R_c + L_corona + delta_PVP))

    phi_1 = (x1 * params.vt_1) / v_tail_avg
    HSP_core = phi_1 * params.HSP_1 + (1.0 - phi_1) * params.HSP_2
    dHSP_sq = float(
        (params.HSP_CsA[0] - HSP_core[0]) ** 2 +
        0.25 * (params.HSP_CsA[1] - HSP_core[1]) ** 2 +
        0.25 * (params.HSP_CsA[2] - HSP_core[2]) ** 2
    )
    V_molar_CsA = params.v_CsA * 1e-27 * N_A
    chi_dc = float((V_molar_CsA * dHSP_sq * 1e6) / (R_gas * T))
    laplace_term = (2.0 * params.sigma_core * (params.v_CsA * 1e-27)) / (R_c * 1e-9 * k_B * T)

    S_w = 0.019 * np.exp(-1800.0 * (1.0 / 298.15 - 1.0 / T))
    kappa_base = 0.285 * (v_tail_avg / params.v_CsA)
    kappa = float(kappa_base * np.exp(-0.55 * (chi_dc - 0.35) - 0.15 * (laplace_term - 1.2)))

    C_mic_surf = max(0.0, (C1 + C2) - CMC_mix)
    S_tot_eq = float(S_w + kappa * C_mic_surf)
    S_R = float(params.C_CsA / max(1e-6, S_tot_eq))

    return {
        **thermo,
        "N_agg": N_agg,
        "R_c_nm": R_c,
        "L_corona_nm": L_corona,
        "D_h_nm": D_h,
        "chi_dc": chi_dc,
        "kappa": kappa,
        "S_tot_eq_mM": S_tot_eq,
        "S_tot_eq_mg_mL": S_tot_eq * params.MW_CsA / 1e3,
        "S_R": S_R
    }


def dlvo_steric_profile(R_c_nm, L_corona_nm, T):
    R_c = R_c_nm * 1e-9
    L = L_corona_nm * 1e-9
    R_h = R_c + L
    A_eff = 0.85e-20
    chi_ew = chi_EO_water(T)
    phi_EO = 0.16
    v_w = 3.0e-29

    h_nm_arr = np.linspace(0.4, 2.5 * L_corona_nm, 140)
    h_arr = h_nm_arr * 1e-9

    t1 = (2.0 * R_c ** 2) / (h_arr * (4.0 * R_c + h_arr))
    t2 = (2.0 * R_c ** 2) / ((2.0 * R_c + h_arr) ** 2)
    t3 = np.log((h_arr * (4.0 * R_c + h_arr)) / ((2.0 * R_c + h_arr) ** 2))
    V_vdW_arr = -(A_eff / (6.0 * k_B * T)) * (t1 + t2 + t3)

    V_osm_arr = np.where(
        h_arr < 2.0 * L,
        ((4.0 * np.pi * R_c * (phi_EO ** 2) * (0.5 - chi_ew) * (L ** 2)) / v_w) * (
            (h_arr / (2.0 * L)) - 0.25 - np.log(h_arr / L)
        ),
        0.0
    )

    V_el_arr = np.where(
        h_arr < L,
        45.0 * (1.0 - h_arr / L) ** 2.2,
        0.0
    )

    V_tot_arr = V_vdW_arr + V_osm_arr + V_el_arr
    V_max_kT = float(np.max(V_tot_arr))

    integrand = np.exp(np.clip(V_tot_arr, -30.0, 65.0)) / ((2.0 * R_h + h_arr) ** 2)
    W_fuchs = 2.0 * R_h * trapz_compat(integrand, h_arr)
    log10_W = float(np.log10(max(1.0, W_fuchs)))

    return h_nm_arr, V_vdW_arr, V_osm_arr, V_el_arr, V_tot_arr, V_max_kT, log10_W


@st.cache_data(show_spinner=False)
def simulate_shelf_life_ode(T_celsius, months_max, c_hco, c_oc40, c_csa, c_pvp, pH_0, beta_12):
    params = NanomicelleFormulation(c_hco, c_oc40, c_csa, c_pvp, pH_0, 0.125, beta_12)
    T = T_celsius + 273.15
    T_ref = 298.15
    pKw = 14.0 - 0.032 * (T_celsius - 25.0)

    def rhs(t, y):
        C1, pH = y
        arrh = np.exp(-(params.Ea_hyd / R_gas) * (1.0 / T - 1.0 / T_ref))
        k_hyd = arrh * (
            params.k_H * (10.0 ** (-pH)) +
            params.k_0 +
            params.k_OH * (10.0 ** (pH - pKw))
        )
        dC1_dt = -k_hyd * C1
        dpH_dt = 1.5 * (dC1_dt * 1e-3) / params.beta_buf
        return [dC1_dt, dpH_dt]

    t_eval = np.linspace(0, months_max, 45)
    sol = solve_ivp(rhs, [0, months_max], [params.C1_0, params.pH_0], t_eval=t_eval)

    records = []
    for i, t_m in enumerate(sol.t):
        c1_t = sol.y[0, i]
        ph_t = sol.y[1, i]
        st_eq = solve_equilibrium_micelle(c1_t, params.C2_0, T, params)
        records.append({
            "Month": float(t_m),
            "Intact_HCO40_pct": float((c1_t / params.C1_0) * 100.0),
            "pH": float(ph_t),
            "S_eq_mg_mL": float(st_eq["S_tot_eq_mg_mL"]),
            "S_R": float(st_eq["S_R"]),
            "D_h_nm": float(st_eq["D_h_nm"])
        })
    return pd.DataFrame(records)


# ==============================================================================
# 3. POPULATION BALANCE EQUATION (PBE) COMPOUNDING ENGINE
# ==============================================================================
@st.cache_data(show_spinner=False)
def run_compounding_pbe(c_hco, c_oc40, c_csa, c_pvp, pH_0, beta_12,
                        V_batch_L, D_imp_m, RPM, t_feed_min, T_comp_C, N_p=1.27, N_q=0.72):
    params = NanomicelleFormulation(c_hco, c_oc40, c_csa, c_pvp, pH_0, 0.125, beta_12)
    T_K = T_comp_C + 273.15
    eq_state = solve_equilibrium_micelle(params.C1_0, params.C2_0, T_K, params)
    D_eq = eq_state["D_h_nm"]
    _, _, _, _, _, V_max_kT, log10_W = dlvo_steric_profile(eq_state["R_c_nm"], eq_state["L_corona_nm"], T_K)
    W_steric = 10.0 ** min(12.0, log10_W)

    V_m3 = V_batch_L * 1e-3
    N_rps = RPM / 60.0
    rho_liq = 1005.0
    mu_liq = water_viscosity(T_K)
    nu_kin = mu_liq / rho_liq

    P_watts = N_p * rho_liq * (N_rps ** 3) * (D_imp_m ** 5)
    epsilon = max(1e-4, P_watts / (rho_liq * V_m3))
    v_tip = np.pi * D_imp_m * N_rps
    shear_rate = np.sqrt(epsilon / nu_kin)
    eta_K_um = ((nu_kin ** 3) / epsilon) ** 0.25 * 1e6

    tau_micro_s = 17.24 * np.sqrt(nu_kin / epsilon)
    Q_pump_m3s = max(1e-6, N_q * N_rps * (D_imp_m ** 3))
    tau_macro_s = 3.0 * V_m3 / Q_pump_m3s

    V_preconc_m3 = 0.0115 * V_m3
    t_feed_s = max(10.0, t_feed_min * 60.0)
    Q_feed_m3s = V_preconc_m3 / t_feed_s

    plume_ratio = Q_feed_m3s / Q_pump_m3s
    S_R_local = eq_state["S_R"] * (1.0 + 420.0 * plume_ratio * np.sqrt(tau_macro_s / 5.0))
    tau_nuc_s = 0.12 * np.exp(2.8 / (max(0.05, np.log(max(1.02, S_R_local))) ** 2))
    Da_mix = (tau_micro_s * (1.0 + 150.0 * plume_ratio)) / tau_nuc_s

    M = 50
    D_grid = np.geomspace(6.0, 320.0, M)
    v_grid = (np.pi / 6.0) * (D_grid ** 3)

    D_feed_0 = np.clip(165.0 * (epsilon ** (-0.12)), 75.0, 240.0)
    sigma_feed = 0.28
    feed_shape = np.exp(-0.5 * ((np.log(D_grid / D_feed_0)) / sigma_feed) ** 2) / (D_grid * sigma_feed)
    feed_shape /= np.sum(feed_shape * v_grid)

    sigma_eq = 0.11
    eq_shape = np.exp(-0.5 * ((np.log(D_grid / D_eq)) / sigma_eq) ** 2) / (D_grid * sigma_eq)
    eq_shape /= np.sum(eq_shape * v_grid)

    k_fiss_base = 0.018 * (shear_rate ** 0.55) * np.exp(-1200.0 * (1.0 / T_K - 1.0 / 333.15))
    Fiss_rate = k_fiss_base * np.maximum(0.0, (D_grid - D_eq) / D_eq) ** 1.6

    # Precompute volume-redistribution matrix for fast vectorized ODE calls
    W_redist = np.zeros((M, M))
    for j in range(1, M):
        if Fiss_rate[j] > 0:
            w = np.exp(-0.5 * ((D_grid[:j] - D_eq * 1.12) / (0.32 * D_eq)) ** 2)
            w_sum = np.sum(w)
            if w_sum > 0:
                W_redist[:j, j] = w / w_sum

    k_AW = 0.025 * (shear_rate ** 0.35) * np.exp(-1500.0 * (1.0 / T_K - 1.0 / 333.15))
    k_coal = (1.5e-2 * shear_rate) / max(1.0, W_steric)

    t_total_s = t_feed_s + 900.0
    t_eval = np.linspace(0.0, t_total_s, 75)

    y0 = np.zeros(M + 1)
    y0[:M] = 1e-6 * eq_shape * v_grid

    V_feed_vec = feed_shape * v_grid
    V_eq_vec = eq_shape * v_grid

    def pbe_ode(t, y):
        V_bins = np.maximum(0.0, y[:M])
        cryst_ppm = y[M]

        feed_rate = (1.0 / t_feed_s) * V_feed_vec if t <= t_feed_s else np.zeros(M)
        loss_fiss = Fiss_rate * V_bins
        gain_fiss = W_redist @ loss_fiss

        current_mass = np.sum(V_bins)
        aw_relax = k_AW * (current_mass * V_eq_vec - V_bins)

        coal_shift = np.zeros(M)
        if k_coal > 1e-6:
            coal_shift[:-1] -= k_coal * V_bins[:-1]
            coal_shift[1:] += k_coal * V_bins[:-1]

        dV_dt = feed_rate - loss_fiss + gain_fiss + aw_relax + coal_shift

        dcryst_dt = 0.0
        if t <= t_feed_s and Da_mix > 0.12:
            dcryst_dt = (900.0 / t_feed_s) * (1.0 - np.exp(-1.8 * (Da_mix - 0.12)))
        elif t > t_feed_s and cryst_ppm > 0 and eq_state["S_R"] < 1.0:
            dcryst_dt = -0.0004 * (1.0 - eq_state["S_R"]) * cryst_ppm

        return np.concatenate([dV_dt, [dcryst_dt]])

    sol = solve_ivp(pbe_ode, [0.0, t_total_s], y0, t_eval=t_eval, method="RK45")

    traj_records = []
    psd_snapshots = {}
    snap_indices = [int(0.15 * len(t_eval)), int(0.45 * len(t_eval)), int(0.70 * len(t_eval)), len(t_eval) - 1]
    grad_logD = np.gradient(np.log10(D_grid))

    for idx, t_s in enumerate(sol.t):
        V_b = np.maximum(0.0, sol.y[:M, idx])
        v_norm = V_b / max(1e-12, np.sum(V_b))
        cdf = np.cumsum(v_norm)

        D10 = float(np.interp(0.10, cdf, D_grid))
        D50 = float(np.interp(0.50, cdf, D_grid))
        D90 = float(np.interp(0.90, cdf, D_grid))
        mean_D = float(np.sum(v_norm * D_grid))
        var_D = float(np.sum(v_norm * (D_grid - mean_D) ** 2))
        PDI = float(var_D / (mean_D ** 2))

        traj_records.append({
            "Time_min": float(t_s / 60.0),
            "D10_nm": D10,
            "D50_nm": D50,
            "D90_nm": D90,
            "PDI": PDI,
            "Unincorporated_CsA_ppm": float(max(0.0, sol.y[M, idx]))
        })
        if idx in snap_indices:
            psd_snapshots[f"{t_s/60.0:.1f} min"] = v_norm / grad_logD

    return {
        "D_grid": D_grid,
        "traj_df": pd.DataFrame(traj_records),
        "psd_snapshots": psd_snapshots,
        "metrics": {
            "epsilon_W_kg": float(epsilon),
            "v_tip_m_s": float(v_tip),
            "shear_rate_s": float(shear_rate),
            "eta_K_um": float(eta_K_um),
            "tau_micro_ms": float(tau_micro_s * 1e3),
            "tau_macro_s": float(tau_macro_s),
            "Da_mix": float(Da_mix),
            "D_eq_nm": float(D_eq),
            "Final_D50_nm": traj_records[-1]["D50_nm"],
            "Final_D90_nm": traj_records[-1]["D90_nm"],
            "Final_PDI": traj_records[-1]["PDI"],
            "Final_Cryst_ppm": traj_records[-1]["Unincorporated_CsA_ppm"]
        }
    }


# ==============================================================================
# 4. 0.22 um STERILE FILTRATION MEMBRANE ADSORPTION ENGINE
# ==============================================================================
MEMBRANE_DB = {
    "Hydrophilic PVDF (Ultra-Low Binding)": {
        "A_sp_m2_g": 18.5, "eps_m": 0.76, "tau_m": 1.45, "rho_p_g_cm3": 1.78,
        "qmax_CsA_I": 0.42, "KL_CsA_I": 18.0, "qmax_surf_I": 1.10, "KL_surf_I": 24.0,
        "qmax_surf_II": 0.65, "KL_surf_II": 1.8, "omega_co": 0.12, "k_surf_s": 0.18
    },
    "Hydrophilic PES (Polyethersulfone)": {
        "A_sp_m2_g": 26.0, "eps_m": 0.78, "tau_m": 1.55, "rho_p_g_cm3": 1.37,
        "qmax_CsA_I": 1.65, "KL_CsA_I": 52.0, "qmax_surf_I": 2.80, "KL_surf_I": 45.0,
        "qmax_surf_II": 3.90, "KL_surf_II": 4.2, "omega_co": 0.38, "k_surf_s": 0.25
    },
    "Nylon 6,6 (High Amide H-Bonding - Worst Case)": {
        "A_sp_m2_g": 31.0, "eps_m": 0.72, "tau_m": 1.70, "rho_p_g_cm3": 1.14,
        "qmax_CsA_I": 3.40, "KL_CsA_I": 95.0, "qmax_surf_I": 4.90, "KL_surf_I": 70.0,
        "qmax_surf_II": 7.80, "KL_surf_II": 7.5, "omega_co": 0.62, "k_surf_s": 0.32
    }
}


@st.cache_data(show_spinner=False)
def simulate_sterile_filtration_adsorption(
    mem_name, A_EFA_cm2, L_m_um, Q_filt_mL_min, V_tot_mL,
    c_hco_wt=1.00, c_oc40_wt=0.05, c_csa_wt=0.09
):
    mem = MEMBRANE_DB[mem_name]
    eps_m = mem["eps_m"]
    tau_m = mem["tau_m"]
    rho_p = mem["rho_p_g_cm3"]
    A_sp = mem["A_sp_m2_g"]

    L_cm = L_m_um * 1e-4
    A_cm2 = A_EFA_cm2
    Q_cm3_s = Q_filt_mL_min / 60.0
    J_v_cm_s = Q_cm3_s / A_cm2
    u_z_cm_s = J_v_cm_s / eps_m

    V_bulk_cm3 = A_cm2 * L_cm
    V_pore_mL = V_bulk_cm3 * eps_m
    M_mem_g = V_bulk_cm3 * (1.0 - eps_m) * rho_p
    solid_to_void = ((1.0 - eps_m) / eps_m) * rho_p

    MW_surf_avg = 2460.0
    MW_CsA = 1202.61
    C_surf_in_mg = (c_hco_wt + c_oc40_wt) * 10.0
    C_CsA_in_mg = c_csa_wt * 10.0

    CMC_mix_mM = 0.0082
    CMC_mix_mg = CMC_mix_mM * MW_surf_avg / 1e3
    K_m_part = 480.0
    kappa_mg_mg = 0.098
    S_w_mg = 0.023

    d_pore_cm = 0.22e-4
    r_p0_nm = 110.0
    D_eff_cm2_s = 2.4e-7
    Gz = (u_z_cm_s * (d_pore_cm ** 2)) / (D_eff_cm2_s * L_cm * tau_m)
    Sh = 3.66 + (0.0668 * Gz) / (1.0 + 0.04 * (Gz ** (2.0 / 3.0)))
    k_f_cm_s = (Sh * D_eff_cm2_s) / d_pore_cm
    a_v_cm_inv = (4.0 * eps_m) / d_pore_cm

    k_LDF_surf = 1.0 / (1.0 / mem["k_surf_s"] + 12.0 / (k_f_cm_s * a_v_cm_inv))
    k_LDF_CsA = 1.0 / (1.0 / (mem["k_surf_s"] * 0.85) + 15.0 / (k_f_cm_s * a_v_cm_inv))
    D_ax = (D_eff_cm2_s / tau_m) + 0.5 * u_z_cm_s * d_pore_cm * 15.0

    N_z = 20
    dz = L_cm / N_z

    def compute_equilibrium_q(C_surf_mg_arr, C_CsA_mg_arr):
        C_surf_mM = np.maximum(0.0, C_surf_mg_arr / MW_surf_avg * 1e3)
        C_CsA_mM = np.maximum(0.0, C_CsA_mg_arr / MW_CsA * 1e3)

        C_surf_free_mM = np.minimum(C_surf_mM, CMC_mix_mM)
        C_mic_mM = np.maximum(0.0, C_surf_mM - CMC_mix_mM)

        f_free_CsA = 1.0 / (1.0 + K_m_part * (C_mic_mM * 1e-3))
        C_CsA_free_mM = f_free_CsA * C_CsA_mM

        denom_I = 1.0 + mem["KL_surf_I"] * C_surf_free_mM + mem["KL_CsA_I"] * C_CsA_free_mM
        q_surf_I = (mem["qmax_surf_I"] * mem["KL_surf_I"] * C_surf_free_mM) / denom_I
        q_CsA_I = (mem["qmax_CsA_I"] * mem["KL_CsA_I"] * C_CsA_free_mM) / denom_I

        q_surf_II = (mem["qmax_surf_II"] * mem["KL_surf_II"] * C_mic_mM) / (1.0 + mem["KL_surf_II"] * C_mic_mM)
        payload_ratio = np.where(C_surf_mg_arr > 1e-5, C_CsA_mg_arr / np.maximum(1e-4, C_surf_mg_arr), 0.0)
        q_CsA_II = mem["omega_co"] * payload_ratio * q_surf_II

        return (q_surf_I + q_surf_II), (q_CsA_I + q_CsA_II)

    y0 = np.zeros(4 * N_z)
    t_end_s = V_tot_mL / Q_cm3_s
    t_eval = np.linspace(0.0, t_end_s, 120)

    def adr_ode(t, y):
        C_s = np.maximum(0.0, y[0:N_z])
        C_d = np.maximum(0.0, y[N_z:2*N_z])
        q_s = np.maximum(0.0, y[2*N_z:3*N_z])
        q_d = np.maximum(0.0, y[3*N_z:4*N_z])

        q_s_star, q_d_star = compute_equilibrium_q(C_s, C_d)
        dq_s_dt = k_LDF_surf * (q_s_star - q_s)
        dq_d_dt = k_LDF_CsA * (q_d_star - q_d)

        # Vectorized finite differences across N_z nodes
        dC_s_dt = np.zeros(N_z)
        dC_d_dt = np.zeros(N_z)

        dC_s_dt[0] = (u_z_cm_s * (C_surf_in_mg - C_s[0]) / dz) + D_ax * (C_s[1] - C_s[0]) / (dz ** 2) - solid_to_void * dq_s_dt[0]
        dC_d_dt[0] = (u_z_cm_s * (C_CsA_in_mg - C_d[0]) / dz) + D_ax * (C_d[1] - C_d[0]) / (dz ** 2) - solid_to_void * dq_d_dt[0]

        dC_s_dt[1:-1] = (
            -u_z_cm_s * (C_s[1:-1] - C_s[:-2]) / dz
            + D_ax * (C_s[2:] - 2.0 * C_s[1:-1] + C_s[:-2]) / (dz ** 2)
            - solid_to_void * dq_s_dt[1:-1]
        )
        dC_d_dt[1:-1] = (
            -u_z_cm_s * (C_d[1:-1] - C_d[:-2]) / dz
            + D_ax * (C_d[2:] - 2.0 * C_d[1:-1] + C_d[:-2]) / (dz ** 2)
            - solid_to_void * dq_d_dt[1:-1]
        )

        dC_s_dt[-1] = -u_z_cm_s * (C_s[-1] - C_s[-2]) / dz + D_ax * (C_s[-2] - C_s[-1]) / (dz ** 2) - solid_to_void * dq_s_dt[-1]
        dC_d_dt[-1] = -u_z_cm_s * (C_d[-1] - C_d[-2]) / dz + D_ax * (C_d[-2] - C_d[-1]) / (dz ** 2) - solid_to_void * dq_d_dt[-1]

        return np.concatenate([dC_s_dt, dC_d_dt, dq_s_dt, dq_d_dt])

    sol = solve_ivp(adr_ode, [0.0, t_end_s], y0, t_eval=t_eval, method="RK45", rtol=1e-4, atol=1e-6)

    V_filt_arr = sol.t * Q_cm3_s
    records = []
    for idx, V_mL in enumerate(V_filt_arr):
        C_s_out = max(0.0, sol.y[N_z - 1, idx])
        C_d_out = max(0.0, sol.y[2 * N_z - 1, idx])
        q_s_prof = sol.y[2 * N_z:3 * N_z, idx]
        q_d_prof = sol.y[3 * N_z:4 * N_z, idx]

        delta_ads_nm = ((q_s_prof + q_d_prof) / (1080.0 * A_sp)) * 1e5
        r_eff_nm = np.maximum(25.0, r_p0_nm - delta_ads_nm)
        tmp_ratio = float(np.mean((r_p0_nm / r_eff_nm) ** 4))

        C_mic_out_mg = max(0.0, C_s_out - CMC_mix_mg)
        S_eq_out_mg = S_w_mg + kappa_mg_mg * C_mic_out_mg
        S_R_pore = C_d_out / max(1e-4, S_eq_out_mg) if C_d_out > 0.01 else 0.0

        records.append({
            "V_filt_mL": float(V_mL),
            "CsA_Assay_pct": float((C_d_out / C_CsA_in_mg) * 100.0),
            "Surf_Assay_pct": float((C_s_out / C_surf_in_mg) * 100.0),
            "Mean_q_CsA_mg_g": float(np.mean(q_d_prof)),
            "Mean_q_Surf_mg_g": float(np.mean(q_s_prof)),
            "Adsorbed_Layer_nm": float(np.mean(delta_ads_nm)),
            "TMP_Ratio": tmp_ratio,
            "Pore_S_R": float(S_R_pore)
        })

    df_bt = pd.DataFrame(records)
    above_98 = df_bt[df_bt["CsA_Assay_pct"] >= 98.0]
    V_flush_98_mL = float(above_98["V_filt_mL"].iloc[0]) if not above_98.empty else float(V_tot_mL)
    above_995 = df_bt[df_bt["CsA_Assay_pct"] >= 99.5]
    V_flush_995_mL = float(above_995["V_filt_mL"].iloc[0]) if not above_995.empty else float(V_tot_mL)

    total_CsA_lost_mg = float(df_bt["Mean_q_CsA_mg_g"].iloc[-1] * M_mem_g)
    total_Surf_lost_mg = float(df_bt["Mean_q_Surf_mg_g"].iloc[-1] * M_mem_g)

    V_vial = 0.25
    n_max = min(400, int(V_tot_mL / V_vial))
    vial_nums = np.arange(1, n_max + 1)
    v_mids = (vial_nums - 0.5) * V_vial
    df_vials = pd.DataFrame({
        "BFS_Vial_Number": vial_nums,
        "Cumulative_Vol_mL": vial_nums * V_vial,
        "Vial_CsA_Assay_pct": np.interp(v_mids, df_bt["V_filt_mL"], df_bt["CsA_Assay_pct"]),
        "Vial_Surf_Assay_pct": np.interp(v_mids, df_bt["V_filt_mL"], df_bt["Surf_Assay_pct"])
    })

    return {
        "df_bt": df_bt,
        "df_vials": df_vials,
        "summary": {
            "M_mem_g": float(M_mem_g),
            "V_pore_mL": float(V_pore_mL),
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
# 5. SIDEBAR & GLOBAL STATE
# ==============================================================================
st.sidebar.title("Formulation & Process Parameters")

st.sidebar.subheader("1. Formulation Composition (Q1/Q2)")
c_hco = st.sidebar.slider("HCO-40 Concentration (% w/v)", 0.40, 2.00, 1.00, 0.05)
c_oc40 = st.sidebar.slider("Octoxynol-40 Concentration (% w/v)", 0.01, 0.25, 0.05, 0.01)
c_csa = st.sidebar.slider("Cyclosporine A Active (% w/v)", 0.05, 0.15, 0.09, 0.01)
c_pvp = st.sidebar.slider("Povidone K90 (% w/v)", 0.10, 0.60, 0.30, 0.05)
pH_init = st.sidebar.slider("Initial Compounding pH", 6.00, 7.60, 6.80, 0.05)
beta_12_val = st.sidebar.slider("Rubingh Interaction Parameter (β₁₂)", -4.0, -0.5, -2.45, 0.05)

st.sidebar.subheader("2. Compounding & Reactor Scale-Up")
V_batch = st.sidebar.number_input("Batch Volume (L)", min_value=1.0, max_value=2000.0, value=100.0, step=10.0)
D_imp = st.sidebar.slider("Impeller Diameter Dᵢ (m)", 0.05, 0.60, 0.22, 0.01)
rpm_val = st.sidebar.slider("Agitation Speed (RPM)", 50, 1200, 350, 25)
t_feed = st.sidebar.slider("Pre-Concentrate Addition Time (min)", 2.0, 60.0, 15.0, 1.0)
T_comp = st.sidebar.slider("Hydration Temperature (°C)", 25.0, 75.0, 55.0, 1.0)

formulation = NanomicelleFormulation(c_hco, c_oc40, c_csa, c_pvp, pH_init, 0.125, beta_12_val)
state_25C = solve_equilibrium_micelle(formulation.C1_0, formulation.C2_0, 298.15, formulation)
_, _, _, _, _, V_max_25, logW_25 = dlvo_steric_profile(state_25C["R_c_nm"], state_25C["L_corona_nm"], 298.15)
shelf_25C = simulate_shelf_life_ode(25.0, 24.0, c_hco, c_oc40, c_csa, c_pvp, pH_init, beta_12_val)
end_24m = shelf_25C.iloc[-1]

# ==============================================================================
# 6. HEADER & 5-TAB WORKBENCH UI
# ==============================================================================
st.title("First-Principles Ophthalmic Nanomicellar QbD, PBE & Filtration Workbench")
st.markdown(
    "**System:** Binary Non-Ionic Mixed Micelles (HCO-40 + Octoxynol-40) encapsulating "
    "Cyclosporine A in a Povidone K90 buffered vehicle | **Modules:** Rubingh RST + "
    "Nagarajan–Ruckenstein + Discrete-Sectional PBE + 0.22 μm Membrane Adsorption"
)

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Mixed CMC (25 °C)", f"{state_25C['CMC_mix']:.4f} mM", f"Ideal: {state_25C['CMC_ideal']:.4f} mM")
col2.metric("Equilibrium Dₕ", f"{state_25C['D_h_nm']:.2f} nm", f"N_agg = {state_25C['N_agg']:.0f}")
col3.metric("Initial S_R (t = 0)", f"{state_25C['S_R']:.3f}", "Stable (< 1.0)" if state_25C["S_R"] <= 1.0 else "Supersaturated!")
col4.metric("24-Mo S_R (25 °C)", f"{end_24m['S_R']:.3f}", f"S_eq = {end_24m['S_eq_mg_mL']:.2f} mg/mL")
col5.metric("Steric Barrier V_max", f"{V_max_25:.1f} k_B T", f"log₁₀(W) = {logW_25:.1f}")
col6.metric("24-Mo Intact HCO-40", f"{end_24m['Intact_HCO40_pct']:.1f}%", f"pH = {end_24m['pH']:.2f}")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. QbD Design Space & MODR",
    "2. Compounding PBE & Scale-Up",
    "3. Colloidal DLVO & 24-Mo Shelf-Life",
    "4. Multi-Scale Reactor Comparison",
    "5. 0.22 μm Sterile Filtration Adsorption"
])

# ------------------------------------------------------------------------------
# TAB 1: 2D QbD DESIGN SPACE
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("2D Quality by Design (QbD) Response Surfaces & Method Operable Design Region (MODR)")
    c_left, c_right = st.columns([1, 3])

    with c_left:
        design_axes = st.radio(
            "Select Design Space Axes:",
            ["Surfactant Space (HCO-40 % vs. Octoxynol-40 %)",
             "Chemical Stability Space (Initial pH vs. Storage Temp °C)"]
        )
        grid_res = st.select_slider("Contour Grid Resolution", options=[12, 16, 20], value=16)
        st.markdown("""
        **CQA Acceptance Criteria (MODR):**
        * **24-Mo Supersaturation Ratio:** $S_R(24\text{ mo}) \le 1.00$
        * **Hydrodynamic Size:** $18.0 \le D_h \le 24.5\text{ nm}$
        * **24-Mo Formulation pH:** $6.50 \le \text{pH} \le 7.20$
        """)

    with c_right:
        if "Surfactant Space" in design_axes:
            hco_grid = np.linspace(0.50, 1.60, grid_res)
            oc_grid = np.linspace(0.01, 0.20, grid_res)
            SR_24_mat = np.zeros((grid_res, grid_res))
            Dh_mat = np.zeros((grid_res, grid_res))
            retention_24m = end_24m["Intact_HCO40_pct"] / 100.0

            for i, oc_v in enumerate(oc_grid):
                for j, hco_v in enumerate(hco_grid):
                    tmp_f = NanomicelleFormulation(hco_v, oc_v, c_csa, c_pvp, pH_init, 0.125, beta_12_val)
                    st_0 = solve_equilibrium_micelle(tmp_f.C1_0, tmp_f.C2_0, 298.15, tmp_f)
                    st_24 = solve_equilibrium_micelle(tmp_f.C1_0 * retention_24m, tmp_f.C2_0, 298.15, tmp_f)
                    Dh_mat[i, j] = st_0["D_h_nm"]
                    SR_24_mat[i, j] = st_24["S_R"]

            fig_qbd = make_subplots(
                rows=1, cols=2,
                subplot_titles=(
                    "24-Month Supersaturation Ratio S_R (Failure Boundary S_R = 1.0)",
                    "Equilibrium Micelle Diameter Dₕ (nm)"
                )
            )
            fig_qbd.add_trace(
                go.Contour(
                    x=hco_grid, y=oc_grid, z=SR_24_mat,
                    colorscale="RdYlGn_r",
                    contours=dict(start=0.5, end=1.5, size=0.05, showlabels=True),
                    colorbar=dict(title="S_R (24m)", x=0.45)
                ),
                row=1, col=1
            )
            fig_qbd.add_trace(
                go.Contour(
                    x=hco_grid, y=oc_grid, z=SR_24_mat,
                    contours=dict(coloring="none", start=1.0, end=1.0, size=0.1, showlabels=True),
                    line=dict(color="black", width=4, dash="dash"),
                    showscale=False, name="S_R = 1.0 Limit"
                ),
                row=1, col=1
            )
            fig_qbd.add_trace(
                go.Scatter(
                    x=[c_hco], y=[c_oc40], mode="markers+text",
                    marker=dict(size=12, color="blue", symbol="star"),
                    text=["Setpoint"], textposition="top center", name="Setpoint"
                ),
                row=1, col=1
            )
            fig_qbd.add_trace(
                go.Contour(
                    x=hco_grid, y=oc_grid, z=Dh_mat,
                    colorscale="Viridis", contours=dict(showlabels=True),
                    colorbar=dict(title="Dₕ (nm)", x=1.0)
                ),
                row=1, col=2
            )
            fig_qbd.add_trace(
                go.Scatter(
                    x=[c_hco], y=[c_oc40], mode="markers",
                    marker=dict(size=12, color="red", symbol="star"), showlegend=False
                ),
                row=1, col=2
            )
            fig_qbd.update_xaxes(title_text="HCO-40 Concentration (% w/v)", row=1, col=1)
            fig_qbd.update_yaxes(title_text="Octoxynol-40 Concentration (% w/v)", row=1, col=1)
            fig_qbd.update_xaxes(title_text="HCO-40 Concentration (% w/v)", row=1, col=2)
            fig_qbd.update_yaxes(title_text="Octoxynol-40 Concentration (% w/v)", row=1, col=2)
            fig_qbd.update_layout(height=460, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_qbd, use_container_width=True)

        else:
            pH_grid = np.linspace(6.0, 7.6, grid_res)
            T_grid = np.linspace(5.0, 42.0, grid_res)
            SR_mat = np.zeros((grid_res, grid_res))
            HCO_ret_mat = np.zeros((grid_res, grid_res))

            for i, T_val in enumerate(T_grid):
                T_K = T_val + 273.15
                pKw = 14.0 - 0.032 * (T_val - 25.0)
                arrh = np.exp(-(formulation.Ea_hyd / R_gas) * (1.0 / T_K - 1.0 / 298.15))
                for j, ph_val in enumerate(pH_grid):
                    k_hyd = arrh * (
                        formulation.k_H * (10.0 ** (-ph_val)) +
                        formulation.k_0 +
                        formulation.k_OH * (10.0 ** (ph_val - pKw))
                    )
                    c1_24 = formulation.C1_0 * np.exp(-k_hyd * 24.0)
                    HCO_ret_mat[i, j] = (c1_24 / formulation.C1_0) * 100.0
                    st_24 = solve_equilibrium_micelle(c1_24, formulation.C2_0, T_K, formulation)
                    SR_mat[i, j] = st_24["S_R"]

            fig_ph = make_subplots(
                rows=1, cols=2,
                subplot_titles=(
                    "24-Month Supersaturation Ratio S_R(pH, T)",
                    "24-Month Intact HCO-40 Ester Retention (%)"
                )
            )
            fig_ph.add_trace(
                go.Contour(
                    x=pH_grid, y=T_grid, z=SR_mat,
                    colorscale="RdYlGn_r",
                    contours=dict(start=0.6, end=1.6, size=0.05, showlabels=True),
                    colorbar=dict(title="S_R (24m)", x=0.45)
                ),
                row=1, col=1
            )
            fig_ph.add_trace(
                go.Contour(
                    x=pH_grid, y=T_grid, z=SR_mat,
                    contours=dict(coloring="none", start=1.0, end=1.0, size=0.1, showlabels=True),
                    line=dict(color="black", width=4, dash="dash"), showscale=False
                ),
                row=1, col=1
            )
            fig_ph.add_trace(
                go.Contour(
                    x=pH_grid, y=T_grid, z=HCO_ret_mat,
                    colorscale="Blues", contours=dict(showlabels=True),
                    colorbar=dict(title="Intact HCO-40 (%)", x=1.0)
                ),
                row=1, col=2
            )
            fig_ph.update_xaxes(title_text="Initial Formulation pH", row=1, col=1)
            fig_ph.update_yaxes(title_text="Storage Temperature (°C)", row=1, col=1)
            fig_ph.update_xaxes(title_text="Initial Formulation pH", row=1, col=2)
            fig_ph.update_yaxes(title_text="Storage Temperature (°C)", row=1, col=2)
            fig_ph.update_layout(height=460, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_ph, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 2: COMPOUNDING PBE & SCALE-UP SIMULATOR
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("Population Balance Equation (PBE) — High-Shear Aqueous Hydration & Self-Assembly")
    pbe_res = run_compounding_pbe(
        c_hco, c_oc40, c_csa, c_pvp, pH_init, beta_12_val,
        V_batch, D_imp, rpm_val, t_feed, T_comp
    )
    m = pbe_res["metrics"]

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Specific Power (ε)", f"{m['epsilon_W_kg']:.3f} W/kg", f"v_tip = {m['v_tip_m_s']:.2f} m/s")
    m2.metric("Kolmogorov Scale (η_K)", f"{m['eta_K_um']:.1f} μm", f"γ̇ = {m['shear_rate_s']:.0f} s⁻¹")
    m3.metric("Micromixing Time (τ_micro)", f"{m['tau_micro_ms']:.1f} ms", f"τ_macro = {m['tau_macro_s']:.1f} s")
    m4.metric("Micromixing Da_mix", f"{m['Da_mix']:.3f}", "Safe (< 0.12)" if m["Da_mix"] <= 0.12 else "Nucleation Risk!")
    m5.metric("Final Micelle D₅₀ / D₉₀", f"{m['Final_D50_nm']:.1f} / {m['Final_D90_nm']:.1f} nm", f"PDI = {m['Final_PDI']:.3f}")
    m6.metric("Unincorporated CsA", f"{m['Final_Cryst_ppm']:.2f} ppm", "0 ppm Target")

    fig_pbe = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "PBE Evolution of Volume-Weighted PSD During Hydration",
            "Transient D₁₀, D₅₀, D₉₀ & Unincorporated Crystal Trajectory"
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": True}]]
    )

    for label, psd_curve in pbe_res["psd_snapshots"].items():
        fig_pbe.add_trace(
            go.Scatter(x=pbe_res["D_grid"], y=psd_curve, mode="lines", name=f"PSD @ {label}", line=dict(width=2.5)),
            row=1, col=1
        )
    fig_pbe.add_vline(
        x=m["D_eq_nm"], line_dash="dash", line_color="green",
        annotation_text=f"Dₕ* ({m['D_eq_nm']:.1f} nm)", row=1, col=1
    )

    df_t = pbe_res["traj_df"]
    fig_pbe.add_trace(
        go.Scatter(x=df_t["Time_min"], y=df_t["D90_nm"], name="D₉₀ (nm)", line=dict(color="#d62728", width=2.5)),
        row=1, col=2, secondary_y=False
    )
    fig_pbe.add_trace(
        go.Scatter(x=df_t["Time_min"], y=df_t["D50_nm"], name="D₅₀ (nm)", line=dict(color="#1f77b4", width=3.0)),
        row=1, col=2, secondary_y=False
    )
    fig_pbe.add_trace(
        go.Scatter(x=df_t["Time_min"], y=df_t["D10_nm"], name="D₁₀ (nm)", line=dict(color="#2ca02c", width=2.0, dash="dot")),
        row=1, col=2, secondary_y=False
    )
    fig_pbe.add_trace(
        go.Scatter(
            x=df_t["Time_min"], y=df_t["Unincorporated_CsA_ppm"],
            name="Unincorporated CsA Crystals (ppm)", line=dict(color="#9467bd", width=2.5, dash="dashdot")
        ),
        row=1, col=2, secondary_y=True
    )

    fig_pbe.update_xaxes(title_text="Aggregate Hydrodynamic Diameter Dₕ (nm) [Log Scale]", type="log", row=1, col=1)
    fig_pbe.update_yaxes(title_text="Volume Density dV / d(log₁₀ D)", row=1, col=1)
    fig_pbe.update_xaxes(title_text="Compounding Batch Time (min)", row=1, col=2)
    fig_pbe.update_yaxes(title_text="Percentile Diameter (nm)", secondary_y=False, row=1, col=2)
    fig_pbe.update_yaxes(title_text="Unincorporated Crystalline CsA (ppm)", secondary_y=True, row=1, col=2)
    fig_pbe.update_layout(height=460, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_pbe, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 3: COLLOIDAL DLVO STERIC POTENTIAL & 24-MONTH STABILITY
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("Extended DLVO + PEO-40 Brush Steric Potential & 24-Month ICH Stability Kinetics")
    fig_stab = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Pair Interaction Potential V_tot(h) Across Temperatures",
            "24-Month Supersaturation Ratio S_R(t) Trajectory (ICH Conditions)"
        )
    )

    for T_test, col_c in [(25.0, "#1f77b4"), (40.0, "#ff7f0e"), (60.0, "#d62728"), (78.0, "#7f7f7f")]:
        T_K = T_test + 273.15
        st_temp = solve_equilibrium_micelle(formulation.C1_0, formulation.C2_0, T_K, formulation)
        h_arr, _, _, _, V_tot, V_max, _ = dlvo_steric_profile(st_temp["R_c_nm"], st_temp["L_corona_nm"], T_K)
        fig_stab.add_trace(
            go.Scatter(
                x=h_arr, y=V_tot, mode="lines",
                name=f"{T_test:.0f} °C (V_max = {V_max:.1f} k_B T)",
                line=dict(color=col_c, width=2.5)
            ),
            row=1, col=1
        )

    for T_ich, col_c in [(5.0, "#2ca02c"), (25.0, "#1f77b4"), (30.0, "#ff7f0e"), (40.0, "#d62728")]:
        df_ich = simulate_shelf_life_ode(T_ich, 24.0, c_hco, c_oc40, c_csa, c_pvp, pH_init, beta_12_val)
        fig_stab.add_trace(
            go.Scatter(
                x=df_ich["Month"], y=df_ich["S_R"],
                mode="lines", name=f"ICH {T_ich:.0f} °C", line=dict(color=col_c, width=2.8)
            ),
            row=1, col=2
        )

    fig_stab.add_hline(y=1.0, line_dash="dash", line_color="black",
                       annotation_text="Precipitation Threshold (S_R = 1.0)", row=1, col=2)
    fig_stab.update_xaxes(title_text="Surface-to-Surface Separation h (nm)", row=1, col=1)
    fig_stab.update_yaxes(title_text="Interaction Potential V(h) / k_B T", range=[-10, 80], row=1, col=1)
    fig_stab.update_xaxes(title_text="Storage Time (Months)", row=1, col=2)
    fig_stab.update_yaxes(title_text="Supersaturation Ratio S_R = C_nominal / S_eq(t)", row=1, col=2)
    fig_stab.update_layout(height=450, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_stab, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 4: MULTI-SCALE REACTOR COMPARISON
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("Multi-Scale Reactor Scale-Up Comparison (5 L Lab vs. 50 L Pilot vs. 500 L Commercial)")
    scale_criterion = st.selectbox(
        "Select Scale-Up Criterion from Current Setpoint:",
        ["Constant Specific Power Input (ε = P/V)",
         "Constant Impeller Tip Speed (v_tip = π N Dᵢ)",
         "Constant Micromixing Damköhler Number (Da_mix)"]
    )

    scales = [
        {"Scale": "Lab R&D Batch", "V_L": 5.0, "D_i_m": 0.08},
        {"Scale": "Pilot GMP Scale", "V_L": 50.0, "D_i_m": 0.18},
        {"Scale": "Commercial Scale", "V_L": 500.0, "D_i_m": 0.38}
    ]
    base_eps = pbe_res["metrics"]["epsilon_W_kg"]
    base_tip = pbe_res["metrics"]["v_tip_m_s"]

    comp_rows = []
    for s in scales:
        V_s = s["V_L"]
        D_s = s["D_i_m"]
        if "Constant Specific Power" in scale_criterion:
            N_rps = ((base_eps * (V_s * 1e-3)) / (1.27 * (D_s ** 5))) ** (1.0 / 3.0)
        elif "Constant Impeller Tip" in scale_criterion:
            N_rps = base_tip / (np.pi * D_s)
        else:
            N_rps = ((base_eps * (V_s * 1e-3)) / (1.27 * (D_s ** 5))) ** (1.0 / 3.0) * 1.12

        rpm_s = float(N_rps * 60.0)
        res_s = run_compounding_pbe(
            c_hco, c_oc40, c_csa, c_pvp, pH_init, beta_12_val,
            V_s, D_s, rpm_s, t_feed, T_comp
        )
        ms = res_s["metrics"]
        comp_rows.append({
            "Batch Scale": s["Scale"],
            "Volume (L)": V_s,
            "Impeller Dᵢ (m)": D_s,
            "Required Speed (RPM)": round(rpm_s, 1),
            "Power Input ε (W/kg)": round(ms["epsilon_W_kg"], 3),
            "Tip Speed (m/s)": round(ms["v_tip_m_s"], 2),
            "τ_micro (ms)": round(ms["tau_micro_ms"], 2),
            "τ_macro (s)": round(ms["tau_macro_s"], 1),
            "Micromixing Da_mix": round(ms["Da_mix"], 3),
            "Final D₅₀ (nm)": round(ms["Final_D50_nm"], 2),
            "Final D₉₀ (nm)": round(ms["Final_D90_nm"], 2),
            "Final PDI": round(ms["Final_PDI"], 3),
            "Unincorporated CsA (ppm)": round(ms["Final_Cryst_ppm"], 2)
        })

    df_comp = pd.DataFrame(comp_rows)
    st.dataframe(df_comp, use_container_width=True)

    csv_data = df_comp.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Scale-Up & PBE Summary Report (CSV)",
        data=csv_data,
        file_name="nanomicellar_qbd_pbe_scaleup.csv",
        mime="text/csv"
    )

# ------------------------------------------------------------------------------
# TAB 5: 0.22 um STERILE FILTRATION ADSORPTION & BFS PRE-FLUSH
# ------------------------------------------------------------------------------
with tab5:
    st.subheader("0.22 μm Sterile Filtration Membrane Adsorption, Breakthrough & BFS Pre-Flush Sizing")

    col_cfg, col_plots = st.columns([1, 3])
    with col_cfg:
        mem_choice = st.selectbox("Select 0.22 μm Membrane Polymer:", list(MEMBRANE_DB.keys()), index=1)
        A_efa = st.slider("Effective Filtration Area A_EFA (cm²)", 100.0, 5000.0, 1000.0, 100.0)
        L_bed = st.slider("Effective Membrane Thickness L_m (μm)", 120.0, 450.0, 240.0, 20.0)
        Q_flow = st.slider("Filtration Flow Rate Q (mL/min)", 20.0, 1000.0, 150.0, 10.0)
        V_sim = st.slider("Simulated Initial Filtrate Volume (mL)", 100.0, 1500.0, 500.0, 50.0)

    res_f = simulate_sterile_filtration_adsorption(mem_choice, A_efa, L_bed, Q_flow, V_sim, c_hco, c_oc40, c_csa)
    sf = res_f["summary"]
    df_bt = res_f["df_bt"]
    df_vials = res_f["df_vials"]

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Pre-Flush to 98% Assay", f"{sf['V_flush_98_mL']:.1f} mL", f"{sf['BFS_Vials_Lost_98']} BFS Vials")
    k2.metric("Pre-Flush to 99.5% Assay", f"{sf['V_flush_995_mL']:.1f} mL", f"{sf['V_flush_995_mL']/A_efa*10:.2f} L/m²")
    k3.metric("Total CsA Adsorbed", f"{sf['Total_CsA_Adsorbed_mg']:.2f} mg", f"Membrane: {sf['M_mem_g']:.1f} g")
    k4.metric("Total Surfactant Adsorbed", f"{sf['Total_Surf_Adsorbed_mg']:.1f} mg", "Site I + Site II")
    k5.metric("Adsorbed Brush δ_ads", f"{sf['Max_Brush_Thickness_nm']:.2f} nm", f"ΔP_TMP +{sf['Max_TMP_Increase_pct']:.1f}%")
    k6.metric("Peak Pore S_R Wave", f"{sf['Max_Pore_SR']:.3f}", "Safe (< 1.0)" if sf["Max_Pore_SR"] <= 1.0 else "Nucleation Risk!")

    with col_plots:
        fig_f1 = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                "Filtrate Breakthrough: CsA vs. Mixed Surfactant (% Label Claim)",
                "Cross-Polymer Comparison of CsA Breakthrough (PVDF vs. PES vs. Nylon)"
            )
        )
        fig_f1.add_trace(
            go.Scatter(x=df_bt["V_filt_mL"], y=df_bt["CsA_Assay_pct"], mode="lines",
                       name="Cyclosporine A Filtrate Assay (%)", line=dict(color="#d62728", width=3.0)),
            row=1, col=1
        )
        fig_f1.add_trace(
            go.Scatter(x=df_bt["V_filt_mL"], y=df_bt["Surf_Assay_pct"], mode="lines",
                       name="Mixed Surfactant (%)", line=dict(color="#1f77b4", width=2.5, dash="dash")),
            row=1, col=1
        )
        fig_f1.add_hline(y=98.0, line_dash="dot", line_color="green", row=1, col=1)
        fig_f1.add_vline(x=sf["V_flush_98_mL"], line_dash="dash", line_color="black", row=1, col=1)

        colors_m = {
            "Hydrophilic PVDF (Ultra-Low Binding)": "#2ca02c",
            "Hydrophilic PES (Polyethersulfone)": "#ff7f0e",
            "Nylon 6,6 (High Amide H-Bonding - Worst Case)": "#9467bd"
        }
        for m_name, c_hex in colors_m.items():
            r_cmp = simulate_sterile_filtration_adsorption(m_name, A_efa, L_bed, Q_flow, V_sim, c_hco, c_oc40, c_csa)
            fig_f1.add_trace(
                go.Scatter(
                    x=r_cmp["df_bt"]["V_filt_mL"], y=r_cmp["df_bt"]["CsA_Assay_pct"], mode="lines",
                    name=f"{m_name.split(' (')[0]} ({r_cmp['summary']['V_flush_98_mL']:.0f} mL)",
                    line=dict(color=c_hex, width=2.8)
                ),
                row=1, col=2
            )
        fig_f1.add_hline(y=98.0, line_dash="dot", line_color="green", row=1, col=2)
        fig_f1.update_xaxes(title_text="Cumulative Filtrate Volume V_filt (mL)", row=1, col=1)
        fig_f1.update_yaxes(title_text="Outlet Filtrate Potency (% of Feed)", range=[0, 105], row=1, col=1)
        fig_f1.update_xaxes(title_text="Cumulative Filtrate Volume V_filt (mL)", row=1, col=2)
        fig_f1.update_yaxes(title_text="CsA Filtrate Potency (% of Feed)", range=[0, 105], row=1, col=2)
        fig_f1.update_layout(height=430, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_f1, use_container_width=True)

    fig_f2 = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Unflushed Blow-Fill-Seal (BFS) Vial-by-Vial Assay (0.25 mL Vials)",
            "Internal Pore Adsorbed Brush Thickness δ_ads (nm) & Local Pore S_R"
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": True}]]
    )
    fig_f2.add_trace(
        go.Scatter(x=df_vials["BFS_Vial_Number"], y=df_vials["Vial_CsA_Assay_pct"],
                   mode="lines+markers", marker=dict(size=4), name="0.25 mL BFS Vial Assay (%)",
                   line=dict(color="#d62728", width=2.0)),
        row=1, col=1
    )
    fig_f2.add_hline(y=98.0, line_dash="dash", line_color="green", row=1, col=1)

    fig_f2.add_trace(
        go.Scatter(x=df_bt["V_filt_mL"], y=df_bt["Adsorbed_Layer_nm"], mode="lines",
                   name="Adsorbed Brush δ_ads (nm)", line=dict(color="#1f77b4", width=2.5)),
        row=1, col=2, secondary_y=False
    )
    fig_f2.add_trace(
        go.Scatter(x=df_bt["V_filt_mL"], y=df_bt["Pore_S_R"], mode="lines",
                   name="Local Pore S_R", line=dict(color="#ff7f0e", width=2.5, dash="dot")),
        row=1, col=2, secondary_y=True
    )
    fig_f2.update_xaxes(title_text="Sequential BFS Vial Number (0.25 mL Fill Each)", row=1, col=1)
    fig_f2.update_yaxes(title_text="Vial Cyclosporine A Assay (%)", range=[0, 105], row=1, col=1)
    fig_f2.update_xaxes(title_text="Cumulative Filtrate Volume (mL)", row=1, col=2)
    fig_f2.update_yaxes(title_text="Adsorbed Layer Thickness δ_ads (nm)", secondary_y=False, row=1, col=2)
    fig_f2.update_yaxes(title_text="Local Pore Supersaturation Ratio S_R", secondary_y=True, row=1, col=2)
    fig_f2.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_f2, use_container_width=True)
