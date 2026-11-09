import copy
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

# ==============================================================================
# 2. DEFAULT LITERATURE CONSTANTS & SESSION STATE INITIALIZATION
# ==============================================================================
DEFAULT_CONSTANTS = {
    # Universal Physical Constants
    "k_B": 1.380649e-23,        # Boltzmann constant (J/K)
    "N_A": 6.02214076e23,       # Avogadro's number (1/mol)
    "R_gas": 8.314462,          # Universal gas constant (J/mol*K)

    # Molecular Weights & Volumes
    "MW_1": 2500.0,             # HCO-40 MW (g/mol)
    "MW_2": 1966.0,             # Octoxynol-40 MW (g/mol)
    "MW_CsA": 1202.61,          # Cyclosporine A MW (g/mol)
    "vt_1": 1.48,               # HCO-40 hydrophobic tail volume (nm^3)
    "vt_2": 0.36,               # OC-40 hydrophobic tail volume (nm^3)
    "v_CsA": 1.74,              # CsA molecular volume (nm^3)
    "a0_shield": 0.42,          # Shielded core interfacial area per monomer (nm^2)
    "N_EO": 40.0,               # Ethylene oxide units per chain
    "l_EO": 0.36,               # Monomer Kuhn length of EO (nm)
    "v_EO": 0.065,              # Volume of a single EO unit (nm^3)
    "sigma_core": 0.030,        # Core-water interfacial tension (J/m^2)

    # Mixed Micellization Thermodynamics (Rubingh & Setschenow)
    "CMC1_ref": 0.014,          # Pure HCO-40 CMC at 298.15 K (mM)
    "CMC2_ref": 0.160,          # Pure OC-40 CMC at 298.15 K (mM)
    "dH_mic1": -12500.0,        # HCO-40 enthalpy of micellization (J/mol)
    "dH_mic2": -9800.0,         # OC-40 enthalpy of micellization (J/mol)
    "ks_1": 1.15,               # Setschenow constant HCO-40 (M^-1)
    "ks_2": 0.95,               # Setschenow constant OC-40 (M^-1)
    "kp_1": 0.18,               # PVP interaction constant HCO-40 ((% w/v)^-1)
    "kp_2": 0.15,               # PVP interaction constant OC-40 ((% w/v)^-1)
    "I_ionic": 0.125,           # Ionic strength of phosphate buffer + NaCl (M)
    "beta_buf": 0.035,          # Phosphate buffer capacity (mol/L/pH unit)

    # Hansen Solubility Parameters [d_d, d_p, d_h] (MPa^0.5) & Intrinsic Solubility
    "HSP_1_d": 17.2, "HSP_1_p": 4.5, "HSP_1_h": 6.8,
    "HSP_2_d": 18.4, "HSP_2_p": 3.2, "HSP_2_h": 4.1,
    "HSP_CsA_d": 17.8, "HSP_CsA_p": 5.1, "HSP_CsA_h": 7.4,
    "Sw_ref_mM": 0.019,         # Intrinsic aqueous CsA solubility at 298.15 K (mM)

    # DLVO Steric, Ester Hydrolysis & Reactor Hydrodynamic Constants
    "A_eff_1e20": 0.85,         # Effective Hamaker constant (x 10^-20 J)
    "phi_EO_brush": 0.16,       # Mean EO segment volume fraction in corona
    "Ea_hyd": 78500.0,          # HCO-40 ester hydrolysis activation energy (J/mol)
    "k_H": 1.8e-2,              # Acid-catalyzed hydrolysis rate (month^-1 M^-1)
    "k_0": 4.5e-4,              # Neutral hydrolysis rate (month^-1)
    "k_OH": 8.2e2,              # Base-catalyzed hydrolysis rate (month^-1 M^-1)
    "N_p": 1.27,                # Impeller power number (dimensionless)
    "N_q": 0.72                 # Impeller pumping number (dimensionless)
}

DEFAULT_MEMBRANE_DF = pd.DataFrame([
    {
        "Polymer": "Hydrophilic PVDF (Ultra-Low Binding)",
        "A_sp_m2_g": 18.5, "eps_m": 0.76, "tau_m": 1.45, "rho_p_g_cm3": 1.78,
        "qmax_CsA_I": 0.42, "KL_CsA_I": 18.0, "qmax_surf_I": 1.10, "KL_surf_I": 24.0,
        "qmax_surf_II": 0.65, "KL_surf_II": 1.8, "omega_co": 0.12, "k_surf_s": 0.18
    },
    {
        "Polymer": "Hydrophilic PES (Polyethersulfone)",
        "A_sp_m2_g": 26.0, "eps_m": 0.78, "tau_m": 1.55, "rho_p_g_cm3": 1.37,
        "qmax_CsA_I": 1.65, "KL_CsA_I": 52.0, "qmax_surf_I": 2.80, "KL_surf_I": 45.0,
        "qmax_surf_II": 3.90, "KL_surf_II": 4.2, "omega_co": 0.38, "k_surf_s": 0.25
    },
    {
        "Polymer": "Nylon 6,6 (High Amide H-Bonding - Worst Case)",
        "A_sp_m2_g": 31.0, "eps_m": 0.72, "tau_m": 1.70, "rho_p_g_cm3": 1.14,
        "qmax_CsA_I": 3.40, "KL_CsA_I": 95.0, "qmax_surf_I": 4.90, "KL_surf_I": 70.0,
        "qmax_surf_II": 7.80, "KL_surf_II": 7.5, "omega_co": 0.62, "k_surf_s": 0.32
    }
])

if "const" not in st.session_state:
    st.session_state.const = copy.deepcopy(DEFAULT_CONSTANTS)
if "mem_df" not in st.session_state:
    st.session_state.mem_df = DEFAULT_MEMBRANE_DF.copy()

# ==============================================================================
# 3. FIRST-PRINCIPLES THERMODYNAMIC & STABILITY ENGINE
# ==============================================================================
def chi_EO_water(T):
    return 0.81 - 110.5 / T

def water_viscosity(T, c_pvp_wt=0.30):
    mu_w = 2.414e-5 * (10.0 ** (247.8 / (T - 140.0)))
    return mu_w * (1.0 + 1.16 * c_pvp_wt)

def get_molar_concentrations(c_hco_wt, c_oc40_wt, c_csa_wt, cfg):
    C1_0 = (c_hco_wt * 10.0 / cfg["MW_1"]) * 1e3
    C2_0 = (c_oc40_wt * 10.0 / cfg["MW_2"]) * 1e3
    C_CsA = (c_csa_wt * 10.0 / cfg["MW_CsA"]) * 1e3
    return C1_0, C2_0, C_CsA

def solve_mixed_cmc(C1, C2, T, c_pvp_wt, beta_12, cfg):
    T_ref = 298.15
    R_gas = cfg["R_gas"]
    CMC1 = cfg["CMC1_ref"] * np.exp(
        (cfg["dH_mic1"] / R_gas) * (1.0 / T - 1.0 / T_ref)
        - cfg["ks_1"] * cfg["I_ionic"]
        - cfg["kp_1"] * c_pvp_wt
    )
    CMC2 = cfg["CMC2_ref"] * np.exp(
        (cfg["dH_mic2"] / R_gas) * (1.0 / T - 1.0 / T_ref)
        - cfg["ks_2"] * cfg["I_ionic"]
        - cfg["kp_2"] * c_pvp_wt
    )

    C_tot = max(1e-9, C1 + C2)
    alpha_1 = float(np.clip(C1 / C_tot, 1e-5, 1.0 - 1e-5))

    def rubingh_res(x1):
        g1 = np.exp(beta_12 * (1.0 - x1) ** 2)
        g2 = np.exp(beta_12 * x1 ** 2)
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

    g1 = float(np.exp(beta_12 * (1.0 - x1) ** 2))
    g2 = float(np.exp(beta_12 * x1 ** 2))
    CMC_mix = float(1.0 / (alpha_1 / (g1 * CMC1) + (1.0 - alpha_1) / (g2 * CMC2)))
    CMC_ideal = float(1.0 / (alpha_1 / CMC1 + (1.0 - alpha_1) / CMC2))

    return {
        "alpha_1": alpha_1, "x1": x1, "CMC1": float(CMC1), "CMC2": float(CMC2),
        "CMC_mix": CMC_mix, "CMC_ideal": CMC_ideal,
        "gamma1": g1, "gamma2": g2
    }

def solve_equilibrium_micelle(C1, C2, C_CsA, T, c_pvp_wt, beta_12, cfg):
    thermo = solve_mixed_cmc(C1, C2, T, c_pvp_wt, beta_12, cfg)
    x1 = thermo["x1"]
    x2 = 1.0 - x1
    CMC_mix = thermo["CMC_mix"]
    k_B = cfg["k_B"]
    N_A = cfg["N_A"]
    R_gas = cfg["R_gas"]

    v_tail_avg = x1 * cfg["vt_1"] + x2 * cfg["vt_2"]
    chi_ew = chi_EO_water(T)
    sigma_kT = (cfg["sigma_core"] * 1e-18) / (k_B * T)

    def delta_mu_kT(N_agg):
        V_core = N_agg * v_tail_avg * 1.22
        R_c = ((3.0 * V_core) / (4.0 * np.pi)) ** (1.0 / 3.0)
        area = (4.0 * np.pi * R_c ** 2) / N_agg
        mu_int = sigma_kT * max(0.0, area - cfg["a0_shield"])
        L_tail = x1 * 2.1 + x2 * 1.1
        mu_def = (3.0 * np.pi ** 2 / 80.0) * (R_c ** 2 / (L_tail ** 2)) * 2.8
        solv_qual = max(0.015, 0.5 - chi_ew)
        mu_steric = 1.85 * cfg["N_EO"] * (cfg["v_EO"] / (area * R_c)) ** 0.85 * solv_qual
        return mu_int + mu_def + mu_steric

    res = minimize_scalar(delta_mu_kT, bounds=(25, 450), method="bounded")
    N_agg = float(res.x)

    V_core = N_agg * v_tail_avg * 1.22
    R_c = float(((3.0 * V_core) / (4.0 * np.pi)) ** (1.0 / 3.0))
    a_opt = float((4.0 * np.pi * R_c ** 2) / N_agg)
    solv_qual = max(0.008, 0.5 - chi_ew)

    L_corona = float(R_c * (
        (1.0 + (5.0 / 3.0) * (cfg["N_EO"] * (cfg["l_EO"] ** (5.0 / 3.0))) /
         (R_c * (a_opt ** (1.0 / 3.0))) * (solv_qual ** (1.0 / 3.0))) ** (3.0 / 5.0) - 1.0
    ))
    delta_PVP = float(1.15 * (c_pvp_wt / 0.30) ** 0.25)
    D_h = float(2.0 * (R_c + L_corona + delta_PVP))

    phi_1 = (x1 * cfg["vt_1"]) / v_tail_avg
    HSP_1 = np.array([cfg["HSP_1_d"], cfg["HSP_1_p"], cfg["HSP_1_h"]])
    HSP_2 = np.array([cfg["HSP_2_d"], cfg["HSP_2_p"], cfg["HSP_2_h"]])
    HSP_CsA = np.array([cfg["HSP_CsA_d"], cfg["HSP_CsA_p"], cfg["HSP_CsA_h"]])

    HSP_core = phi_1 * HSP_1 + (1.0 - phi_1) * HSP_2
    dHSP_sq = float(
        (HSP_CsA[0] - HSP_core[0]) ** 2 +
        0.25 * (HSP_CsA[1] - HSP_core[1]) ** 2 +
        0.25 * (HSP_CsA[2] - HSP_core[2]) ** 2
    )
    V_molar_CsA = cfg["v_CsA"] * 1e-27 * N_A
    chi_dc = float((V_molar_CsA * dHSP_sq * 1e6) / (R_gas * T))
    laplace_term = float((2.0 * cfg["sigma_core"] * (cfg["v_CsA"] * 1e-27)) / (R_c * 1e-9 * k_B * T))

    S_w = float(cfg["Sw_ref_mM"] * np.exp(-1800.0 * (1.0 / 298.15 - 1.0 / T)))
    kappa_base = 0.285 * (v_tail_avg / cfg["v_CsA"])
    kappa = float(kappa_base * np.exp(-0.55 * (chi_dc - 0.35) - 0.15 * (laplace_term - 1.2)))

    C_mic_surf = max(0.0, (C1 + C2) - CMC_mix)
    S_tot_eq = float(S_w + kappa * C_mic_surf)
    S_R = float(C_CsA / max(1e-6, S_tot_eq))

    return {
        **thermo,
        "v_tail_avg": float(v_tail_avg),
        "N_agg": N_agg,
        "R_c_nm": R_c,
        "a_opt_nm2": a_opt,
        "L_corona_nm": L_corona,
        "delta_PVP_nm": delta_PVP,
        "D_h_nm": D_h,
        "chi_ew": float(chi_ew),
        "chi_dc": chi_dc,
        "laplace_term": laplace_term,
        "S_w_mM": S_w,
        "kappa": kappa,
        "S_tot_eq_mM": S_tot_eq,
        "S_tot_eq_mg_mL": S_tot_eq * cfg["MW_CsA"] / 1e3,
        "S_R": S_R
    }

def dlvo_steric_profile(R_c_nm, L_corona_nm, T, cfg):
    R_c = R_c_nm * 1e-9
    L = L_corona_nm * 1e-9
    R_h = R_c + L
    A_eff = cfg["A_eff_1e20"] * 1e-20
    k_B = cfg["k_B"]
    chi_ew = chi_EO_water(T)
    phi_EO = cfg["phi_EO_brush"]
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
def simulate_shelf_life_ode(T_celsius, months_max, c_hco, c_oc40, c_csa, c_pvp, pH_0, beta_12, cfg):
    C1_0, C2_0, C_CsA = get_molar_concentrations(c_hco, c_oc40, c_csa, cfg)
    T = T_celsius + 273.15
    T_ref = 298.15
    pKw = 14.0 - 0.032 * (T_celsius - 25.0)
    R_gas = cfg["R_gas"]

    def rhs(t, y):
        C1, pH = y
        arrh = np.exp(-(cfg["Ea_hyd"] / R_gas) * (1.0 / T - 1.0 / T_ref))
        k_hyd = arrh * (
            cfg["k_H"] * (10.0 ** (-pH)) +
            cfg["k_0"] +
            cfg["k_OH"] * (10.0 ** (pH - pKw))
        )
        dC1_dt = -k_hyd * C1
        dpH_dt = 1.5 * (dC1_dt * 1e-3) / cfg["beta_buf"]
        return [dC1_dt, dpH_dt]

    t_eval = np.linspace(0, months_max, 45)
    sol = solve_ivp(rhs, [0, months_max], [C1_0, pH_0], t_eval=t_eval)

    records = []
    for i, t_m in enumerate(sol.t):
        c1_t = sol.y[0, i]
        ph_t = sol.y[1, i]
        st_eq = solve_equilibrium_micelle(c1_t, C2_0, C_CsA, T, c_pvp, beta_12, cfg)
        records.append({
            "Month": float(t_m),
            "Intact_HCO40_pct": float((c1_t / C1_0) * 100.0),
            "pH": float(ph_t),
            "S_eq_mg_mL": float(st_eq["S_tot_eq_mg_mL"]),
            "S_R": float(st_eq["S_R"]),
            "D_h_nm": float(st_eq["D_h_nm"])
        })
    return pd.DataFrame(records)

# ==============================================================================
# 4. POPULATION BALANCE EQUATION (PBE) COMPOUNDING ENGINE
# ==============================================================================
@st.cache_data(show_spinner=False)
def run_compounding_pbe(c_hco, c_oc40, c_csa, c_pvp, pH_0, beta_12,
                        V_batch_L, D_imp_m, RPM, t_feed_min, T_comp_C, cfg):
    C1_0, C2_0, C_CsA = get_molar_concentrations(c_hco, c_oc40, c_csa, cfg)
    T_K = T_comp_C + 273.15
    eq_state = solve_equilibrium_micelle(C1_0, C2_0, C_CsA, T_K, c_pvp, beta_12, cfg)
    D_eq = eq_state["D_h_nm"]
    _, _, _, _, _, V_max_kT, log10_W = dlvo_steric_profile(eq_state["R_c_nm"], eq_state["L_corona_nm"], T_K, cfg)
    W_steric = 10.0 ** min(12.0, log10_W)

    N_p = cfg["N_p"]
    N_q = cfg["N_q"]
    V_m3 = V_batch_L * 1e-3
    N_rps = RPM / 60.0
    rho_liq = 1005.0
    mu_liq = water_viscosity(T_K, c_pvp)
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
            "Q_pump_L_s": float(Q_pump_m3s * 1e3),
            "Da_mix": float(Da_mix),
            "D_eq_nm": float(D_eq),
            "Final_D50_nm": traj_records[-1]["D50_nm"],
            "Final_D90_nm": traj_records[-1]["D90_nm"],
            "Final_PDI": traj_records[-1]["PDI"],
            "Final_Cryst_ppm": traj_records[-1]["Unincorporated_CsA_ppm"]
        }
    }

# ==============================================================================
# 5. 0.22 um STERILE FILTRATION MEMBRANE ADSORPTION ENGINE
# ==============================================================================
@st.cache_data(show_spinner=False)
def simulate_sterile_filtration_adsorption(
    mem_row_dict, A_EFA_cm2, L_m_um, Q_filt_mL_min, V_tot_mL,
    c_hco_wt, c_oc40_wt, c_csa_wt, CMC_mix_mM, kappa_mol, Sw_mM, cfg
):
    mem = mem_row_dict
    eps_m = float(mem["eps_m"])
    tau_m = float(mem["tau_m"])
    rho_p = float(mem["rho_p_g_cm3"])
    A_sp = float(mem["A_sp_m2_g"])

    L_cm = L_m_um * 1e-4
    A_cm2 = A_EFA_cm2
    Q_cm3_s = Q_filt_mL_min / 60.0
    J_v_cm_s = Q_cm3_s / A_cm2
    u_z_cm_s = J_v_cm_s / eps_m

    V_bulk_cm3 = A_cm2 * L_cm
    V_pore_mL = V_bulk_cm3 * eps_m
    M_mem_g = V_bulk_cm3 * (1.0 - eps_m) * rho_p
    solid_to_void = ((1.0 - eps_m) / eps_m) * rho_p

    MW_surf_avg = 0.94 * cfg["MW_1"] + 0.06 * cfg["MW_2"]
    MW_CsA = cfg["MW_CsA"]
    C_surf_in_mg = (c_hco_wt + c_oc40_wt) * 10.0
    C_CsA_in_mg = c_csa_wt * 10.0

    CMC_mix_mg = CMC_mix_mM * MW_surf_avg / 1e3
    K_m_part = 480.0
    kappa_mg_mg = kappa_mol * (MW_CsA / MW_surf_avg)
    S_w_mg = Sw_mM * MW_CsA / 1e3

    d_pore_cm = 0.22e-4
    r_p0_nm = 110.0
    D_eff_cm2_s = 2.4e-7
    Gz = (u_z_cm_s * (d_pore_cm ** 2)) / (D_eff_cm2_s * L_cm * tau_m)
    Sh = 3.66 + (0.0668 * Gz) / (1.0 + 0.04 * (Gz ** (2.0 / 3.0)))
    k_f_cm_s = (Sh * D_eff_cm2_s) / d_pore_cm
    a_v_cm_inv = (4.0 * eps_m) / d_pore_cm

    k_LDF_surf = 1.0 / (1.0 / float(mem["k_surf_s"]) + 12.0 / (k_f_cm_s * a_v_cm_inv))
    k_LDF_CsA = 1.0 / (1.0 / (float(mem["k_surf_s"]) * 0.85) + 15.0 / (k_f_cm_s * a_v_cm_inv))
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

        denom_I = 1.0 + float(mem["KL_surf_I"]) * C_surf_free_mM + float(mem["KL_CsA_I"]) * C_CsA_free_mM
        q_surf_I = (float(mem["qmax_surf_I"]) * float(mem["KL_surf_I"]) * C_surf_free_mM) / denom_I
        q_CsA_I = (float(mem["qmax_CsA_I"]) * float(mem["KL_CsA_I"]) * C_CsA_free_mM) / denom_I

        q_surf_II = (float(mem["qmax_surf_II"]) * float(mem["KL_surf_II"]) * C_mic_mM) / (1.0 + float(mem["KL_surf_II"]) * C_mic_mM)
        payload_ratio = np.where(C_surf_mg_arr > 1e-5, C_CsA_mg_arr / np.maximum(1e-4, C_surf_mg_arr), 0.0)
        q_CsA_II = float(mem["omega_co"]) * payload_ratio * q_surf_II

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
            "Gz": float(Gz),
            "Sh": float(Sh),
            "k_LDF_CsA_s": float(k_LDF_CsA),
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
# 6. SIDEBAR CONTROLS
# ==============================================================================
st.sidebar.title("Formulation & Process Setpoints")

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

cfg = st.session_state.const
C1_0, C2_0, C_CsA = get_molar_concentrations(c_hco, c_oc40, c_csa, cfg)
state_25C = solve_equilibrium_micelle(C1_0, C2_0, C_CsA, 298.15, c_pvp, beta_12_val, cfg)
_, _, _, _, _, V_max_25, logW_25 = dlvo_steric_profile(state_25C["R_c_nm"], state_25C["L_corona_nm"], 298.15, cfg)
shelf_25C = simulate_shelf_life_ode(25.0, 24.0, c_hco, c_oc40, c_csa, c_pvp, pH_init, beta_12_val, cfg)
end_24m = shelf_25C.iloc[-1]

# ==============================================================================
# 7. MAIN HEADER & REAL-TIME CQA SCORECARD
# ==============================================================================
st.title("First-Principles Ophthalmic Nanomicellar QbD, PBE & Filtration Workbench")
st.markdown(
    "**System:** Binary Non-Ionic Mixed Micelles (**HCO-40 + Octoxynol-40**) encapsulating "
    "**Cyclosporine A (0.09% w/v)** in a **Povidone K90** buffered aqueous vehicle | "
    "**Architecture:** Complete Theory, Manual Constants Manager & 5 Coupled Physics Simulators"
)

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Mixed CMC (25 °C)", f"{state_25C['CMC_mix']:.4f} mM", f"Ideal: {state_25C['CMC_ideal']:.4f} mM")
col2.metric("Equilibrium Dₕ", f"{state_25C['D_h_nm']:.2f} nm", f"N_agg = {state_25C['N_agg']:.0f}")
col3.metric("Initial S_R (t = 0)", f"{state_25C['S_R']:.3f}", "Stable (< 1.0)" if state_25C["S_R"] <= 1.0 else "Supersaturated!")
col4.metric("24-Mo S_R (25 °C)", f"{end_24m['S_R']:.3f}", f"S_eq = {end_24m['S_eq_mg_mL']:.2f} mg/mL")
col5.metric("Steric Barrier V_max", f"{V_max_25:.1f} k_B T", f"log₁₀(W) = {logW_25:.1f}")
col6.metric("24-Mo Intact HCO-40", f"{end_24m['Intact_HCO40_pct']:.1f}%", f"pH = {end_24m['pH']:.2f}")

tab_theory, tab_qbd, tab_pbe, tab_dlvo, tab_scale, tab_filt = st.tabs([
    "📘 1. Theory, Equations, Constants & Walkthrough",
    "🎯 2. QbD Design Space & MODR",
    "⚙️ 3. Compounding PBE & Scale-Up",
    "🧪 4. Colloidal DLVO & 24-Mo Stability",
    "🏭 5. Multi-Scale Reactor Comparison",
    "🔬 6. 0.22 μm Sterile Filtration Adsorption"
])

# ==============================================================================
# TAB 1: COMPLETE THEORY, ASSUMPTIONS, EQUATIONS, MANUAL CONSTANTS & WALKTHROUGH
# ==============================================================================
with tab_theory:
    st.header("1. Conceptual Overview for First-Time Users")
    st.markdown("""
    ### Why Nanomicellar Encapsulation is Required for Ophthalmic Cyclosporine A
    **Cyclosporine A (CsA)** is a cyclic undecapeptide immunomodulator ($MW = 1202.61\text{ g/mol}$, $\log P \approx 2.92$) used to treat chronic Dry Eye Disease (Keratoconjunctivitis Sicca) by inhibiting calcineurin in T-lymphocytes. Because CsA is macrocyclic and highly lipophilic, its intrinsic solubility in water at $25^\circ\text{C}$ is only $\sim 23\text{ }\mu\text{g/mL}$ ($0.0023\%\text{ w/v}$)—nearly **40 times lower** than the clinical $0.09\%\text{ w/v}$ ($0.90\text{ mg/mL}$) dose required for optimal corneal penetration.

    Conventional $0.05\%\text{ w/v}$ oil-in-water emulsions (e.g., castor oil droplets stabilized by Polysorbate 80) carry CsA inside large $150\text{--}500\text{ nm}$ oil droplets, causing light scattering (ocular blurring), poor thermodynamic stability (creaming/coalescence), and slow drug partitioning across the tear film.

    **The Binary Polymeric Nanomicellar Solution (NCELL® Architecture):**
    Instead of a two-phase oil emulsion, this system uses two synergistic non-ionic amphiphilic surfactants that self-assemble in water into ultra-small ($\sim 21.4\text{ nm}$) core–shell nanomicelles:
    1. **Primary Surfactant — Polyoxyl 40 Hydrogenated Castor Oil (HCO-40, $1.00\%\text{ w/v}$):** Contains a bulky, branched tri-hydroxystearate hydrophobic tail and a 40-unit polyethylene oxide (PEO-40) hydrophilic headgroup.
    2. **Co-Surfactant — Octoxynol-40 (OC-40, $0.05\%\text{ w/v}$):** Contains a compact, planar octylphenyl hydrophobic tail and a PEO-40 hydrophilic corona. Intercalating Octoxynol-40 between the bulky triglyceride branches of HCO-40 eliminates core free-volume voids (yielding a strongly negative Rubingh interaction parameter $\beta_{12} \approx -2.45$), depressing the mixed Critical Micelle Concentration ($\text{CMC}_{\text{mix}}$) so micelles resist disassembly upon tear-film dilution.
    3. **Mucoadhesive Polymer — Povidone K90 (PVP, $0.30\%\text{ w/v}$):** Forms a hydrogen-bonded hydrodynamic corona shell ($\delta_{\text{PVP}} \approx 1.15\text{ nm}$) around the PEO-40 brush and increases precorneal residence time.
    """)

    st.divider()
    st.header("2. Interactive Manual Entry for Model Constants")
    st.info(
        "Every physical, thermodynamic, kinetic, and membrane constant used across all six modules is editable below. "
        "Any change you make here immediately updates the live numerical calculations below and re-runs all simulation tabs."
    )

    if st.button("🔄 Reset All Constants to Literature Defaults"):
        st.session_state.const = copy.deepcopy(DEFAULT_CONSTANTS)
        st.session_state.mem_df = DEFAULT_MEMBRANE_DF.copy()
        st.rerun()

    with st.expander("🛠️ A. Molecular Weights, Volumes & Core–Corona Geometry Constants (Click to Edit)", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        cfg["MW_1"] = c1.number_input("HCO-40 MW (g/mol)", value=float(cfg["MW_1"]), step=10.0)
        cfg["MW_2"] = c2.number_input("Octoxynol-40 MW (g/mol)", value=float(cfg["MW_2"]), step=10.0)
        cfg["MW_CsA"] = c3.number_input("Cyclosporine A MW (g/mol)", value=float(cfg["MW_CsA"]), step=1.0)
        cfg["sigma_core"] = c4.number_input("Core-Water Interfacial Tension σ (J/m²)", value=float(cfg["sigma_core"]), step=0.002, format="%.4f")

        c5, c6, c7, c8 = st.columns(4)
        cfg["vt_1"] = c5.number_input("HCO-40 Tail Volume vt,1 (nm³)", value=float(cfg["vt_1"]), step=0.05)
        cfg["vt_2"] = c6.number_input("OC-40 Tail Volume vt,2 (nm³)", value=float(cfg["vt_2"]), step=0.02)
        cfg["v_CsA"] = c7.number_input("CsA Molecular Volume v_CsA (nm³)", value=float(cfg["v_CsA"]), step=0.05)
        cfg["a0_shield"] = c8.number_input("Shielded Core Area a₀ (nm²)", value=float(cfg["a0_shield"]), step=0.02)

        c9, c10, c11 = st.columns(3)
        cfg["N_EO"] = c9.number_input("PEO Chain Length N_EO (units)", value=float(cfg["N_EO"]), step=1.0)
        cfg["l_EO"] = c10.number_input("EO Kuhn Length l_EO (nm)", value=float(cfg["l_EO"]), step=0.01)
        cfg["v_EO"] = c11.number_input("Single EO Unit Volume v_EO (nm³)", value=float(cfg["v_EO"]), step=0.005, format="%.4f")

    with st.expander("🛠️ B. Mixed Micellization & Electrolyte/Polymer Constants (Rubingh & Setschenow)", expanded=False):
        m1, m2, m3, m4 = st.columns(4)
        cfg["CMC1_ref"] = m1.number_input("Pure HCO-40 CMC_ref at 25 °C (mM)", value=float(cfg["CMC1_ref"]), step=0.001, format="%.4f")
        cfg["CMC2_ref"] = m2.number_input("Pure OC-40 CMC_ref at 25 °C (mM)", value=float(cfg["CMC2_ref"]), step=0.01, format="%.4f")
        cfg["dH_mic1"] = m3.number_input("HCO-40 ΔH°_mic (J/mol)", value=float(cfg["dH_mic1"]), step=500.0)
        cfg["dH_mic2"] = m4.number_input("OC-40 ΔH°_mic (J/mol)", value=float(cfg["dH_mic2"]), step=500.0)

        m5, m6, m7, m8 = st.columns(4)
        cfg["ks_1"] = m5.number_input("Setschenow Constant ks,1 (M⁻¹)", value=float(cfg["ks_1"]), step=0.05)
        cfg["ks_2"] = m6.number_input("Setschenow Constant ks,2 (M⁻¹)", value=float(cfg["ks_2"]), step=0.05)
        cfg["kp_1"] = m7.number_input("PVP Interaction Factor kp,1 ((% w/v)⁻¹)", value=float(cfg["kp_1"]), step=0.02)
        cfg["kp_2"] = m8.number_input("PVP Interaction Factor kp,2 ((% w/v)⁻¹)", value=float(cfg["kp_2"]), step=0.02)

        m9, m10 = st.columns(2)
        cfg["I_ionic"] = m9.number_input("Vehicle Ionic Strength I (M)", value=float(cfg["I_ionic"]), step=0.01, format="%.3f")
        cfg["beta_buf"] = m10.number_input("Phosphate Buffer Capacity β_buf (mol/L/pH)", value=float(cfg["beta_buf"]), step=0.005, format="%.3f")

    with st.expander("🛠️ C. Hansen Solubility Parameters (HSP) & Intrinsic Solubility Constants", expanded=False):
        h1, h2, h3, h4 = st.columns(4)
        cfg["HSP_1_d"] = h1.number_input("HCO-40 δ_d (MPa⁰.⁵)", value=float(cfg["HSP_1_d"]), step=0.1)
        cfg["HSP_1_p"] = h2.number_input("HCO-40 δ_p (MPa⁰.⁵)", value=float(cfg["HSP_1_p"]), step=0.1)
        cfg["HSP_1_h"] = h3.number_input("HCO-40 δ_h (MPa⁰.⁵)", value=float(cfg["HSP_1_h"]), step=0.1)
        cfg["Sw_ref_mM"] = h4.number_input("CsA Intrinsic Aqueous S_w at 25 °C (mM)", value=float(cfg["Sw_ref_mM"]), step=0.001, format="%.4f")

        h5, h6, h7, h8, h9, h10 = st.columns(6)
        cfg["HSP_2_d"] = h5.number_input("OC-40 δ_d", value=float(cfg["HSP_2_d"]), step=0.1)
        cfg["HSP_2_p"] = h6.number_input("OC-40 δ_p", value=float(cfg["HSP_2_p"]), step=0.1)
        cfg["HSP_2_h"] = h7.number_input("OC-40 δ_h", value=float(cfg["HSP_2_h"]), step=0.1)
        cfg["HSP_CsA_d"] = h8.number_input("CsA δ_d", value=float(cfg["HSP_CsA_d"]), step=0.1)
        cfg["HSP_CsA_p"] = h9.number_input("CsA δ_p", value=float(cfg["HSP_CsA_p"]), step=0.1)
        cfg["HSP_CsA_h"] = h10.number_input("CsA δ_h", value=float(cfg["HSP_CsA_h"]), step=0.1)

    with st.expander("🛠️ D. DLVO Steric, Ester Hydrolysis Kinetics & Impeller Constants", expanded=False):
        k1, k2, k3, k4 = st.columns(4)
        cfg["A_eff_1e20"] = k1.number_input("Effective Hamaker Constant A_eff (×10⁻²⁰ J)", value=float(cfg["A_eff_1e20"]), step=0.05)
        cfg["phi_EO_brush"] = k2.number_input("Corona EO Volume Fraction φ_EO", value=float(cfg["phi_EO_brush"]), step=0.01)
        cfg["Ea_hyd"] = k3.number_input("Hydrolysis Activation Energy E_a (J/mol)", value=float(cfg["Ea_hyd"]), step=1000.0)
        cfg["k_0"] = k4.number_input("Neutral Hydrolysis Rate k₀ (month⁻¹)", value=float(cfg["k_0"]), step=1e-4, format="%.5f")

        k5, k6, k7, k8 = st.columns(4)
        cfg["k_H"] = k5.number_input("Acid Rate Constant k_H (month⁻¹ M⁻¹)", value=float(cfg["k_H"]), step=1e-3, format="%.4f")
        cfg["k_OH"] = k6.number_input("Base Rate Constant k_OH (month⁻¹ M⁻¹)", value=float(cfg["k_OH"]), step=10.0, format="%.1f")
        cfg["N_p"] = k7.number_input("Impeller Power Number N_p", value=float(cfg["N_p"]), step=0.05)
        cfg["N_q"] = k8.number_input("Impeller Pumping Number N_q", value=float(cfg["N_q"]), step=0.05)

    with st.expander("🛠️ E. 0.22 μm Sterilizing Membrane Polymer Database (Editable Table)", expanded=False):
        st.caption("Edit any cell below to customize membrane porosity, tortuosity, BET area, and Site I / Site II Langmuir capacities:")
        st.session_state.mem_df = st.data_editor(st.session_state.mem_df, use_container_width=True, num_rows="fixed")

    # Re-evaluate baseline with updated constants
    C1_0, C2_0, C_CsA = get_molar_concentrations(c_hco, c_oc40, c_csa, cfg)
    state_25C = solve_equilibrium_micelle(C1_0, C2_0, C_CsA, 298.15, c_pvp, beta_12_val, cfg)
    _, _, _, _, _, V_max_25, logW_25 = dlvo_steric_profile(state_25C["R_c_nm"], state_25C["L_corona_nm"], 298.15, cfg)
    pbe_live = run_compounding_pbe(c_hco, c_oc40, c_csa, c_pvp, pH_init, beta_12_val, V_batch, D_imp, rpm_val, t_feed, T_comp, cfg)
    mem_dict_0 = st.session_state.mem_df.iloc[1].to_dict()
    filt_live = simulate_sterile_filtration_adsorption(
        mem_dict_0, 1000.0, 240.0, 150.0, 500.0,
        c_hco, c_oc40, c_csa, state_25C["CMC_mix"], state_25C["kappa"], state_25C["S_w_mM"], cfg
    )

    st.divider()
    st.header("3. Fundamental Model Assumptions")
    a_col1, a_col2 = st.columns(2)
    with a_col1:
        st.markdown("""
        **Thermodynamic & Micellar Core–Corona Assumptions:**
        1. **Pseudophase Separation & Regular Solution Approximation:** Micelles are treated as a distinct thermodynamic pseudophase in equilibrium with monomeric surfactant in the aqueous bulk. Excess enthalpy of tail mixing follows Rubingh's single-parameter ($\beta_{12}$) regular solution model whereas excess entropy of mixing is ideal ($\Delta S_{\text{mix}}^E = 0$).
        2. **Spherical Core–Brush Geometry:** Because the Israelachvili packing parameter $P = v_{\text{tail}} / (a_0 l_{\text{tail}}) \approx 0.28 < 1/3$ for PEO-40 surfactants, equilibrium aggregates adopt a spherical hydrophobic core ($R_c$) surrounded by a semi-dilute curved Daoud–Cotton polymer brush ($L_{\text{corona}}$).
        3. **Laplace Capillary Equilibrium:** The curved hydrocarbon-water interface exerts an internal Laplace pressure $\Delta P = 2\sigma_{\text{core/w}} / R_c$ ($\approx 140\text{--}160\text{ bar}$) that shifts the Flory–Huggins chemical potential of solubilized Cyclosporine A.
        """)
    with a_col2:
        st.markdown("""
        **Kinetics, Hydrodynamics & Membrane Transport Assumptions:**
        4. **Pseudo-First-Order Ester Cleavage:** Polyoxyl 40 Hydrogenated Castor Oil contains three ester linkages per molecule susceptible to $\text{H}^+$, neutral water, and $\text{OH}^-$ catalysis, while ether-linked Octoxynol-40 is hydrolytically inert over $24\text{ months}$.
        5. **Isotropic Turbulent Micromixing in the Impeller Zone:** During pre-concentrate hydration, energy dissipation $\epsilon$ governs Kolmogorov microscale engulfment ($\tau_{\text{micro}} = 17.24\sqrt{\nu/\epsilon}$) and capillary fission of swollen lamellar globules.
        6. **Fast Aniansson–Wall Monomer Relaxation in $0.22\text{ }\mu\text{m}$ Pores:** Because micellar monomer exchange ($\tau_{\text{AW}} \sim 1\text{--}10\text{ ms}$) is much faster than pore hydraulic residence time ($\tau_{\text{res}} \sim 0.2\text{--}2.0\text{ s}$), local micelle-monomer equilibrium holds instantaneously at every axial coordinate $z$ inside the sterilizing filter.
        """)

    st.divider()
    st.header("4. Governing Equations & Live Step-by-Step Numerical Walkthrough")
    st.caption("All sample calculations below dynamically plug in your current Sidebar Setpoints and Manual Constants.")

    # MODULE 1 WALKTHROUGH
    with st.expander("📐 Module 1: Synergistic Mixed Micellization (Rubingh–Setschenow Theory)", expanded=True):
        st.markdown("**Step 1.1 — Electrolyte, Polymer & Temperature Corrected Pure CMCs:**")
        st.latex(r"\ln \text{CMC}_i(T, I, C_{\text{PVP}}) = \ln \text{CMC}_{i,\text{ref}} + \frac{\Delta H_{\text{mic},i}^\circ}{R}\left(\frac{1}{T} - \frac{1}{T_{\text{ref}}}\right) - k_{s,i} I - k_{p,i} C_{\text{PVP}}")
        st.markdown("**Step 1.2 — Rubingh Regular Solution Transcendental Equation & Activity Coefficients:**")
        st.latex(r"\frac{x_1^2 \ln\left(\frac{\alpha_1 \text{CMC}_{\text{mix}}}{x_1 \text{CMC}_1}\right)}{(1 - x_1)^2 \ln\left(\frac{(1 - \alpha_1) \text{CMC}_{\text{mix}}}{(1 - x_1) \text{CMC}_2}\right)} = 1, \quad \gamma_1 = \exp\left[\beta_{12}(1-x_1)^2\right], \quad \gamma_2 = \exp\left[\beta_{12}x_1^2\right]")
        st.latex(r"\frac{1}{\text{CMC}_{\text{mix}}} = \frac{\alpha_1}{\gamma_1 \text{CMC}_1} + \frac{1 - \alpha_1}{\gamma_2 \text{CMC}_2}")
        st.success(
            f"**Live Numerical Calculation at T = 298.15 K (25 °C):**\n"
            f"- Bulk Molar Concentrations: $C_1 = {C1_0:.4f}\\text{{ mM}}$ (HCO-40), $C_2 = {C2_0:.4f}\\text{{ mM}}$ (OC-40) $\\implies$ Bulk mole fraction $\\alpha_1 = {state_25C['alpha_1']:.4f}$\n"
            f"- Corrected Pure CMCs: $\\text{{CMC}}_1 = {state_25C['CMC1']:.5f}\\text{{ mM}}$, $\\text{{CMC}}_2 = {state_25C['CMC2']:.5f}\\text{{ mM}}$\n"
            f"- Solving Rubingh's equation with $\\beta_{{12}} = {beta_12_val:.2f}$ yields micellar mole fraction $x_1 = {state_25C['x1']:.4f}$ and activity coefficients $\\gamma_1 = {state_25C['gamma1']:.4f}$, $\\gamma_2 = {state_25C['gamma2']:.4f}$\n"
            f"- **Resulting Synergistic Mixed CMC:** $\\text{{CMC}}_{{\\text{{mix}}}} = \\mathbf{{{state_25C['CMC_mix']:.5f}\\text{{ mM}}}}$ (compared to Ideal Clint $\\text{{CMC}}_{{\\text{{ideal}}}} = {state_25C['CMC_ideal']:.5f}\\text{{ mM}}$)"
        )

    # MODULE 2 WALKTHROUGH
    with st.expander("📐 Module 2: Core–Corona Free Energy Minimization (Nagarajan–Ruckenstein & Daoud–Cotton)", expanded=True):
        st.markdown("**Step 2.1 — Total Micellar Free Energy per Monomer $\Delta \mu_{\text{mic}}^\circ(N_{\text{agg}})$:**")
        st.latex(r"\frac{\Delta \mu_{\text{mic}}^\circ(N_{\text{agg}})}{k_B T} = \underbrace{\frac{\sigma_{\text{core/w}}}{k_B T}\left(a(N_{\text{agg}}) - a_0\right)}_{\text{Core Interfacial Penalty}} + \underbrace{\frac{3\pi^2}{80}\frac{R_c^2}{\bar{N}_{\text{seg}}l_{\text{seg}}^2}}_{\text{Semenov Tail Stretching}} + \underbrace{\nu_F N_{\text{EO}}\left(\frac{v_{\text{EO}}}{a(N_{\text{agg}}) R_c}\right)^{0.85}\left(\frac{1}{2} - \chi_{\text{EO-w}}(T)\right)}_{\text{PEO-40 Corona Brush Repulsion}}")
        st.markdown("**Step 2.2 — Daoud–Cotton Corona Thickness & Total Hydrodynamic Diameter ($D_h$):**")
        st.latex(r"L_{\text{corona}} = R_c \left[ \left( 1 + \frac{5}{3} \frac{N_{\text{EO}} l_{\text{EO}}^{5/3}}{R_c a(N_{\text{agg}}^*)^{1/3}} \left(\frac{1}{2} - \chi_{\text{EO-w}}(T)\right)^{1/3} \right)^{3/5} - 1 \right], \quad D_h = 2\left(R_c + L_{\text{corona}} + \delta_{\text{PVP}}\right)")
        st.success(
            f"**Live Numerical Calculation at T = 298.15 K:**\n"
            f"- Mean Hydrophobic Tail Volume: $\\bar{{v}}_{{\\text{{tail}}}} = x_1 v_{{t,1}} + (1-x_1) v_{{t,2}} = {state_25C['v_tail_avg']:.3f}\\text{{ nm}}^3$ | PEO Flory–Huggins $\\chi_{{\\text{{EO-w}}}} = {state_25C['chi_ew']:.4f}$\n"
            f"- Free-Energy Minimization $\\partial \\Delta \\mu_{{\\text{{mic}}}}^\\circ / \\partial N_{{\\text{{agg}}}} = 0 \\implies$ Equilibrium Aggregation Number $N_{{\\text{{agg}}}}^* = \\mathbf{{{state_25C['N_agg']:.1f}\\text{{ monomers/micelle}}}}$\n"
            f"- Core Radius $R_c = {state_25C['R_c_nm']:.2f}\\text{{ nm}}$, Interfacial Area $a^* = {state_25C['a_opt_nm2']:.2f}\\text{{ nm}}^2$, Corona Brush $L_{{\\text{{corona}}}} = {state_25C['L_corona_nm']:.2f}\\text{{ nm}}$, PVP Shell $\\delta_{{\\text{{PVP}}}} = {state_25C['delta_PVP_nm']:.2f}\\text{{ nm}}$\n"
            f"- **Predicted Hydrodynamic Diameter:** $D_h = 2({state_25C['R_c_nm']:.2f} + {state_25C['L_corona_nm']:.2f} + {state_25C['delta_PVP_nm']:.2f}) = \\mathbf{{{state_25C['D_h_nm']:.2f}\\text{{ nm}}}}$"
        )

    # MODULE 3 WALKTHROUGH
    with st.expander("📐 Module 3: Laplace-Corrected Flory–Huggins Solubilization & Supersaturation Ratio (S_R)", expanded=True):
        st.markdown("**Step 3.1 — Hansen Solubility Parameter Matching & Laplace Core Penalty:**")
        st.latex(r"\chi_{\text{dc}}(x_1, T) = \frac{V_{\text{molar,CsA}}}{RT}\left[(\delta_{d,\text{CsA}} - \bar{\delta}_{d,\text{core}})^2 + 0.25(\delta_{p,\text{CsA}} - \bar{\delta}_{p,\text{core}})^2 + 0.25(\delta_{h,\text{CsA}} - \bar{\delta}_{h,\text{core}})^2\right], \quad \Pi_{\text{Laplace}} = \frac{2\sigma_{\text{core/w}} v_{\text{CsA}}}{R_c k_B T}")
        st.markdown("**Step 3.2 — Total Equilibrium Solubility $S_{\text{tot}}^{\text{eq}}$ & Supersaturation Ratio $S_R$:**")
        st.latex(r"S_{\text{tot}}^{\text{eq}}(t, T) = S_w(T) + \kappa(T, x_1)\left[C_{\text{surf,tot}}(t) - \text{CMC}_{\text{mix}}(t, T)\right], \quad S_R(t, T) = \frac{C_{\text{CsA,nominal}}}{S_{\text{tot}}^{\text{eq}}(t, T)}")
        st.success(
            f"**Live Numerical Calculation at T = 298.15 K:**\n"
            f"- Core–Drug Flory–Huggins Parameter: $\\chi_{{\\text{{dc}}}} = {state_25C['chi_dc']:.4f}$ | Dimensionless Laplace Penalty: $\\Pi_{{\\text{{Laplace}}}} = {state_25C['laplace_term']:.3f}$\n"
            f"- Molar Solubilization Capacity: $\\kappa = {state_25C['kappa']:.4f}\\text{{ mol CsA / mol micellar surfactant}}$\n"
            f"- Total Equilibrium CsA Solubility: $S_{{\\text{{tot}}}}^{{\\text{{eq}}}} = {state_25C['S_tot_eq_mM']:.4f}\\text{{ mM}} = \\mathbf{{{state_25C['S_tot_eq_mg_mL']:.3f}\\text{{ mg/mL}}}}$ (vs. Nominal Dose $= {c_csa*10:.3f}\\text{{ mg/mL}}$)\n"
            f"- **Initial Supersaturation Ratio:** $S_R(t=0) = \\mathbf{{{state_25C['S_R']:.3f}}}$ ($S_R \\le 1.0$ guarantees thermodynamic stability against crystallization)"
        )

    # MODULE 4 WALKTHROUGH
    with st.expander("📐 Module 4: Extended DLVO Steric Barrier, Fuchs Ratio (W) & HCO-40 Ester Hydrolysis", expanded=False):
        st.markdown("**Step 4.1 — Pair Interaction Potential $V_{\text{tot}}(h)$ & Fuchs Stability Ratio $W$:**")
        st.latex(r"V_{\text{tot}}(h) = V_{\text{vdW}}(h) + \frac{4\pi R_c k_B T}{v_w}\phi_{\text{EO}}^2\left(\frac{1}{2} - \chi_{\text{EO-w}}(T)\right)L_{\text{corona}}^2\left(\frac{h}{2L_{\text{corona}}} - \frac{1}{4} - \ln\frac{h}{L_{\text{corona}}}\right) + V_{\text{el}}(h)")
        st.markdown("**Step 4.2 — Arrhenius + pH-Catalyzed Ester Cleavage of HCO-40 Over 24 Months:**")
        st.latex(r"-\frac{dC_1}{dt} = \exp\left[-\frac{E_a}{R}\left(\frac{1}{T} - \frac{1}{T_{\text{ref}}}\right)\right]\left(k_H 10^{-\text{pH}} + k_0 + k_{OH} 10^{\text{pH} - \text{p}K_w(T)}\right) C_1(t)")
        st.success(
            f"**Live Numerical Calculation (25 °C & 24 Months):**\n"
            f"- Peak Steric Energy Barrier: $V_{{\\text{{max}}}} = \\mathbf{{{V_max_25:.1f}\\,k_B T}}$ | Log Fuchs Stability Ratio: $\\log_{{10}}(W) = \\mathbf{{{logW_25:.2f}}}$\n"
            f"- After 24 months at $25^\\circ\\text{{C}}$ and initial $\\text{{pH }} {pH_init:.2f}$: Intact HCO-40 $= \\mathbf{{{end_24m['Intact_HCO40_pct']:.1f}\\%}}$, Formulation $\\text{{pH}} = {end_24m['pH']:.2f}$, 24-Month $S_R = \\mathbf{{{end_24m['S_R']:.3f}}}$"
        )

    # MODULE 5 & 6 WALKTHROUGH
    with st.expander("📐 Module 5 & 6: Compounding PBE Hydrodynamics & 0.22 μm Membrane Adsorption PDE", expanded=False):
        st.markdown("**Step 5.1 — Turbulent Energy Dissipation $\epsilon$, Baldyga–Bourne Micromixing $\tau_{\text{micro}}$ & $Da_{\text{mix}}$:**")
        st.latex(r"\epsilon = \frac{N_p N^3 D_i^5}{V_b}, \quad \tau_{\text{micro}} = 17.24\sqrt{\frac{\nu}{\epsilon}}, \quad Da_{\text{mix}} = \frac{\tau_{\text{micro}}\left(1 + 150\,Q_{\text{feed}}/Q_{\text{pump}}\right)}{\tau_{\text{nuc}}(S_{R,\text{loc}})}")
        st.markdown("**Step 6.1 — 1D Advection–Dispersion–Reaction (ADR) & Dual-Site Competitive Langmuir Isotherm:**")
        st.latex(r"\frac{\partial C_k}{\partial t} + u_z\frac{\partial C_k}{\partial z} = D_{\text{ax},k}\frac{\partial^2 C_k}{\partial z^2} - \frac{1-\varepsilon_m}{\varepsilon_m}\rho_p k_{\text{LDF},k}\left(q_k^*(z,t) - q_k(z,t)\right)")
        pm = pbe_live["metrics"]
        fs = filt_live["summary"]
        st.success(
            f"**Live Numerical Calculation (Reactor & 0.22 μm PES Filter):**\n"
            f"- **Reactor ($V_b = {V_batch:.0f}\\text{{ L}}, N = {rpm_val}\\text{{ RPM}}, D_i = {D_imp:.2f}\\text{{ m}}$):** $\\epsilon = {pm['epsilon_W_kg']:.3f}\\text{{ W/kg}}$, Kolmogorov $\\eta_K = {pm['eta_K_um']:.1f}\\text{{ }}\\mu\\text{{m}}$, $\\tau_{{\\text{{micro}}}} = {pm['tau_micro_ms']:.2f}\\text{{ ms}}$, $Da_{{\\text{{mix}}}} = \\mathbf{{{pm['Da_mix']:.3f}}}$, Final $D_{{50}} = \\mathbf{{{pm['Final_D50_nm']:.2f}\\text{{ nm}}}}$\n"
            f"- **0.22 μm Filtration ($A_{{\\text{{EFA}}}} = 1000\\text{{ cm}}^2, L_m = 240\\text{{ }}\\mu\\text{{m}}, Q = 150\\text{{ mL/min}}$):** Graetz $Gz = {fs['Gz']:.4f}$, Sherwood $Sh = {fs['Sh']:.2f}$, $k_{{\\text{{LDF,CsA}}}} = {fs['k_LDF_CsA_s']:.4f}\\text{{ s}}^{{-1}}$, Pre-Flush to $\\ge 98\\%$ Assay $= \\mathbf{{{fs['V_flush_98_mL']:.1f}\\text{{ mL}}}}$ ({fs['BFS_Vials_Lost_98']} BFS vials)"
        )

    st.divider()
    st.header("5. Literature & Equation References")
    ref_df = pd.DataFrame([
        {"Module": "Mixed Micellization (RST)", "Authors & Year": "Rubingh, D. N. (1979)", "Citation / Source": "Solution Chemistry of Surfactants, Vol. 1, pp. 337–354, Plenum Press", "Equation / Concept": "Non-ideal binary mixed CMC and micellar activity coefficients (β₁₂)"},
        {"Module": "Electrolyte Salting-Out", "Authors & Year
