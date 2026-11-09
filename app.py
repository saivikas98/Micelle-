import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Nanomicellar Ophthalmic QbD, PBE & Filtration Workbench",
    layout="wide",
    initial_sidebar_state="expanded"
)

def trapz_compat(y, x):
    return np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x)

k_B = 1.380649e-23
N_A = 6.02214076e23
R_gas = 8.314462

# ==============================================================================
# DEFAULT PHYSICAL, THERMODYNAMIC, KINETIC & MEMBRANE CONSTANTS
# ==============================================================================
DEFAULT_CONSTANTS = {
    # Molecular Weights & Geometry
    "MW_1": 2500.0, "MW_2": 1966.0, "MW_CsA": 1202.61,
    "vt_1": 1.48, "vt_2": 0.36, "v_CsA": 1.74,
    "a0_shield": 0.42, "N_EO": 40.0, "l_EO": 0.36, "v_EO": 0.065,
    "sigma_core": 0.030,
    # Micellization Thermodynamics
    "CMC1_ref": 0.014, "CMC2_ref": 0.160,
    "dH_mic1": -12500.0, "dH_mic2": -9800.0,
    "ks_1": 1.15, "ks_2": 0.95, "kp_1": 0.18, "kp_2": 0.15,
    "I_ionic": 0.125, "beta_buf": 0.035,
    # Hansen Solubility Parameters [d_d, d_p, d_h] (MPa^0.5)
    "d1_d": 17.2, "d1_p": 4.5, "d1_h": 6.8,
    "d2_d": 18.4, "d2_p": 3.2, "d2_h": 4.1,
    "dC_d": 17.8, "dC_p": 5.1, "dC_h": 7.4,
    # Colloidal DLVO & Ester Hydrolysis Kinetics
    "A_eff": 0.85e-20, "Ea_hyd": 78500.0,
    "k_H": 1.8e-2, "k_0": 4.5e-4, "k_OH": 8.2e2,
    # Reactor Hydrodynamics (PBE)
    "N_p": 1.27, "N_q": 0.72,
    # 0.22 um Membrane Adsorption Parameters (Active Selected Polymer Override)
    "A_sp": 26.0, "eps_m": 0.78, "tau_m": 1.55, "rho_p": 1.37,
    "qmax_CsA_I": 1.65, "KL_CsA_I": 52.0,
    "qmax_surf_I": 2.80, "KL_surf_I": 45.0,
    "qmax_surf_II": 3.90, "KL_surf_II": 4.2,
    "omega_co": 0.38, "k_surf_s": 0.25
}

MEMBRANE_PRESETS = {
    "Hydrophilic PVDF (Ultra-Low Binding)": {
        "A_sp": 18.5, "eps_m": 0.76, "tau_m": 1.45, "rho_p": 1.78,
        "qmax_CsA_I": 0.42, "KL_CsA_I": 18.0, "qmax_surf_I": 1.10, "KL_surf_I": 24.0,
        "qmax_surf_II": 0.65, "KL_surf_II": 1.8, "omega_co": 0.12, "k_surf_s": 0.18
    },
    "Hydrophilic PES (Polyethersulfone)": {
        "A_sp": 26.0, "eps_m": 0.78, "tau_m": 1.55, "rho_p": 1.37,
        "qmax_CsA_I": 1.65, "KL_CsA_I": 52.0, "qmax_surf_I": 2.80, "KL_surf_I": 45.0,
        "qmax_surf_II": 3.90, "KL_surf_II": 4.2, "omega_co": 0.38, "k_surf_s": 0.25
    },
    "Nylon 6,6 (High Amide H-Bonding)": {
        "A_sp": 31.0, "eps_m": 0.72, "tau_m": 1.70, "rho_p": 1.14,
        "qmax_CsA_I": 3.40, "KL_CsA_I": 95.0, "qmax_surf_I": 4.90, "KL_surf_I": 70.0,
        "qmax_surf_II": 7.80, "KL_surf_II": 7.5, "omega_co": 0.62, "k_surf_s": 0.32
    }
}

if "consts" not in st.session_state:
    st.session_state.consts = DEFAULT_CONSTANTS.copy()

# ==============================================================================
# CORE FIRST-PRINCIPLES PHYSICS FUNCTIONS
# ==============================================================================
def chi_EO_water(T):
    return 0.81 - 110.5 / T

def water_viscosity(T, c_pvp_wt=0.30):
    mu_w = 2.414e-5 * (10.0 ** (247.8 / (T - 140.0)))
    return mu_w * (1.0 + 1.15 * c_pvp_wt)

def solve_mixed_cmc(C1, C2, T, c_pvp_wt, beta_12, c):
    T_ref = 298.15
    CMC1 = c["CMC1_ref"] * np.exp(
        (c["dH_mic1"] / R_gas) * (1.0 / T - 1.0 / T_ref)
        - c["ks_1"] * c["I_ionic"] - c["kp_1"] * c_pvp_wt
    )
    CMC2 = c["CMC2_ref"] * np.exp(
        (c["dH_mic2"] / R_gas) * (1.0 / T - 1.0 / T_ref)
        - c["ks_2"] * c["I_ionic"] - c["kp_2"] * c_pvp_wt
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
        "CMC_mix": CMC_mix, "CMC_ideal": CMC_ideal, "gamma1": g1, "gamma2": g2
    }

def solve_equilibrium_micelle(C1, C2, C_CsA_mM, T, c_pvp_wt, beta_12, c):
    th = solve_mixed_cmc(C1, C2, T, c_pvp_wt, beta_12, c)
    x1, x2 = th["x1"], 1.0 - th["x1"]
    v_tail_avg = x1 * c["vt_1"] + x2 * c["vt_2"]
    chi_ew = chi_EO_water(T)
    sigma_kT = (c["sigma_core"] * 1e-18) / (k_B * T)

    def delta_mu_kT(N_agg):
        V_core = N_agg * v_tail_avg * 1.22
        R_c = ((3.0 * V_core) / (4.0 * np.pi)) ** (1.0 / 3.0)
        area = (4.0 * np.pi * R_c ** 2) / N_agg
        mu_int = sigma_kT * max(0.0, area - c["a0_shield"])
        L_tail = x1 * 2.1 + x2 * 1.1
        mu_def = (3.0 * np.pi ** 2 / 80.0) * (R_c ** 2 / (L_tail ** 2)) * 2.8
        solv_q = max(0.015, 0.5 - chi_ew)
        mu_steric = 1.85 * c["N_EO"] * (c["v_EO"] / (area * R_c)) ** 0.85 * solv_q
        return mu_int + mu_def + mu_steric

    res = minimize_scalar(delta_mu_kT, bounds=(25, 450), method="bounded")
    N_agg = float(res.x)
    V_core = N_agg * v_tail_avg * 1.22
    R_c = float(((3.0 * V_core) / (4.0 * np.pi)) ** (1.0 / 3.0))
    a_opt = float((4.0 * np.pi * R_c ** 2) / N_agg)
    solv_q = max(0.008, 0.5 - chi_ew)

    L_corona = float(R_c * (
        (1.0 + (5.0 / 3.0) * (c["N_EO"] * (c["l_EO"] ** (5.0 / 3.0))) /
         (R_c * (a_opt ** (1.0 / 3.0))) * (solv_q ** (1.0 / 3.0))) ** (3.0 / 5.0) - 1.0
    ))
    delta_PVP = float(1.15 * (c_pvp_wt / 0.30) ** 0.25)
    D_h = float(2.0 * (R_c + L_corona + delta_PVP))

    phi_1 = (x1 * c["vt_1"]) / v_tail_avg
    HSP_1 = np.array([c["d1_d"], c["d1_p"], c["d1_h"]])
    HSP_2 = np.array([c["d2_d"], c["d2_p"], c["d2_h"]])
    HSP_C = np.array([c["dC_d"], c["dC_p"], c["dC_h"]])
    HSP_core = phi_1 * HSP_1 + (1.0 - phi_1) * HSP_2
    dHSP_sq = float(
        (HSP_C[0] - HSP_core[0]) ** 2 +
        0.25 * (HSP_C[1] - HSP_core[1]) ** 2 +
        0.25 * (HSP_C[2] - HSP_core[2]) ** 2
    )
    V_molar_CsA = c["v_CsA"] * 1e-27 * N_A
    chi_dc = float((V_molar_CsA * dHSP_sq * 1e6) / (R_gas * T))
    laplace_term = float((2.0 * c["sigma_core"] * (c["v_CsA"] * 1e-27)) / (R_c * 1e-9 * k_B * T))

    S_w = float(0.019 * np.exp(-1800.0 * (1.0 / 298.15 - 1.0 / T)))
    kappa_base = 0.285 * (v_tail_avg / c["v_CsA"])
    kappa = float(kappa_base * np.exp(-0.55 * (chi_dc - 0.35) - 0.15 * (laplace_term - 1.2)))

    C_mic_surf = max(0.0, (C1 + C2) - th["CMC_mix"])
    S_tot_eq = float(S_w + kappa * C_mic_surf)
    S_R = float(C_CsA_mM / max(1e-6, S_tot_eq))

    return {
        **th, "N_agg": N_agg, "R_c_nm": R_c, "a_opt_nm2": a_opt,
        "L_corona_nm": L_corona, "delta_PVP_nm": delta_PVP, "D_h_nm": D_h,
        "chi_ew": float(chi_ew), "chi_dc": chi_dc, "laplace_kT": laplace_term,
        "S_w_mM": S_w, "kappa": kappa, "S_tot_eq_mM": S_tot_eq,
        "S_tot_eq_mg_mL": S_tot_eq * c["MW_CsA"] / 1e3, "S_R": S_R
    }

def dlvo_steric_profile(R_c_nm, L_corona_nm, T, c):
    R_c = R_c_nm * 1e-9
    L = L_corona_nm * 1e-9
    R_h = R_c + L
    A_eff = c["A_eff"]
    chi_ew = chi_EO_water(T)
    phi_EO = 0.16
    v_w = 3.0e-29

    h_nm_arr = np.linspace(0.4, 2.5 * L_corona_nm, 120)
    h_arr = h_nm_arr * 1e-9

    t1 = (2.0 * R_c ** 2) / (h_arr * (4.0 * R_c + h_arr))
    t2 = (2.0 * R_c ** 2) / ((2.0 * R_c + h_arr) ** 2)
    t3 = np.log((h_arr * (4.0 * R_c + h_arr)) / ((2.0 * R_c + h_arr) ** 2))
    V_vdW = -(A_eff / (6.0 * k_B * T)) * (t1 + t2 + t3)

    V_osm = np.where(
        h_arr < 2.0 * L,
        ((4.0 * np.pi * R_c * (phi_EO ** 2) * (0.5 - chi_ew) * (L ** 2)) / v_w) * (
            (h_arr / (2.0 * L)) - 0.25 - np.log(h_arr / L)
        ), 0.0
    )
    V_el = np.where(h_arr < L, 45.0 * (1.0 - h_arr / L) ** 2.2, 0.0)
    V_tot = V_vdW + V_osm + V_el
    V_max_kT = float(np.max(V_tot))

    integrand = np.exp(np.clip(V_tot, -30.0, 65.0)) / ((2.0 * R_h + h_arr) ** 2)
    W_fuchs = 2.0 * R_h * trapz_compat(integrand, h_arr)
    log10_W = float(np.log10(max(1.0, W_fuchs)))
    return h_nm_arr, V_vdW, V_osm, V_el, V_tot, V_max_kT, log10_W

def simulate_shelf_life_ode(T_celsius, months_max, C1_0, C2_0, C_CsA_mM, c_pvp_wt, pH_0, beta_12, c):
    T = T_celsius + 273.15
    T_ref = 298.15
    pKw = 14.0 - 0.032 * (T_celsius - 25.0)

    def rhs(t, y):
        C1, pH = y
        arrh = np.exp(-(c["Ea_hyd"] / R_gas) * (1.0 / T - 1.0 / T_ref))
        k_hyd = arrh * (c["k_H"] * (10.0 ** (-pH)) + c["k_0"] + c["k_OH"] * (10.0 ** (pH - pKw)))
        dC1_dt = -k_hyd * C1
        dpH_dt = 1.5 * (dC1_dt * 1e-3) / c["beta_buf"]
        return [dC1_dt, dpH_dt]

    t_eval = np.linspace(0, months_max, 40)
    sol = solve_ivp(rhs, [0, months_max], [C1_0, pH_0], t_eval=t_eval)
    records = []
    for i, t_m in enumerate(sol.t):
        c1_t, ph_t = sol.y[0, i], sol.y[1, i]
        st_eq = solve_equilibrium_micelle(c1_t, C2_0, C_CsA_mM, T, c_pvp_wt, beta_12, c)
        records.append({
            "Month": float(t_m), "Intact_HCO40_pct": float((c1_t / C1_0) * 100.0),
            "pH": float(ph_t), "S_eq_mg_mL": float(st_eq["S_tot_eq_mg_mL"]),
            "S_R": float(st_eq["S_R"]), "D_h_nm": float(st_eq["D_h_nm"])
        })
    return pd.DataFrame(records)

def run_compounding_pbe(C1_0, C2_0, C_CsA_mM, c_pvp_wt, beta_12, V_batch_L, D_imp_m, RPM, t_feed_min, T_comp_C, c):
    T_K = T_comp_C + 273.15
    eq_state = solve_equilibrium_micelle(C1_0, C2_0, C_CsA_mM, T_K, c_pvp_wt, beta_12, c)
    D_eq = eq_state["D_h_nm"]
    _, _, _, _, _, _, log10_W = dlvo_steric_profile(eq_state["R_c_nm"], eq_state["L_corona_nm"], T_K, c)
    W_steric = 10.0 ** min(12.0, log10_W)

    V_m3 = V_batch_L * 1e-3
    N_rps = RPM / 60.0
    rho_liq = 1005.0
    mu_liq = water_viscosity(T_K, c_pvp_wt)
    nu_kin = mu_liq / rho_liq

    P_watts = c["N_p"] * rho_liq * (N_rps ** 3) * (D_imp_m ** 5)
    epsilon = max(1e-4, P_watts / (rho_liq * V_m3))
    v_tip = np.pi * D_imp_m * N_rps
    shear_rate = np.sqrt(epsilon / nu_kin)
    eta_K_um = ((nu_kin ** 3) / epsilon) ** 0.25 * 1e6

    tau_micro_s = 17.24 * np.sqrt(nu_kin / epsilon)
    Q_pump_m3s = max(1e-6, c["N_q"] * N_rps * (D_imp_m ** 3))
    tau_macro_s = 3.0 * V_m3 / Q_pump_m3s

    V_preconc_m3 = 0.0115 * V_m3
    t_feed_s = max(10.0, t_feed_min * 60.0)
    Q_feed_m3s = V_preconc_m3 / t_feed_s

    plume_ratio = Q_feed_m3s / Q_pump_m3s
    S_R_local = eq_state["S_R"] * (1.0 + 420.0 * plume_ratio * np.sqrt(tau_macro_s / 5.0))
    tau_nuc_s = 0.12 * np.exp(2.8 / (max(0.05, np.log(max(1.02, S_R_local))) ** 2))
    Da_mix = (tau_micro_s * (1.0 + 150.0 * plume_ratio)) / tau_nuc_s

    M = 45
    D_grid = np.geomspace(6.0, 320.0, M)
    v_grid = (np.pi / 6.0) * (D_grid ** 3)

    D_feed_0 = np.clip(165.0 * (epsilon ** (-0.12)), 75.0, 240.0)
    feed_shape = np.exp(-0.5 * ((np.log(D_grid / D_feed_0)) / 0.28) ** 2) / (D_grid * 0.28)
    feed_shape /= np.sum(feed_shape * v_grid)

    eq_shape = np.exp(-0.5 * ((np.log(D_grid / D_eq)) / 0.11) ** 2) / (D_grid * 0.11)
    eq_shape /= np.sum(eq_shape * v_grid)

    k_fiss_base = 0.018 * (shear_rate ** 0.55) * np.exp(-1200.0 * (1.0 / T_K - 1.0 / 333.15))
    Fiss_rate = k_fiss_base * np.maximum(0.0, (D_grid - D_eq) / D_eq) ** 1.6

    W_redist = np.zeros((M, M))
    for j in range(1, M):
        if Fiss_rate[j] > 0:
            w = np.exp(-0.5 * ((D_grid[:j] - D_eq * 1.12) / (0.32 * D_eq)) ** 2)
            if np.sum(w) > 0:
                W_redist[:j, j] = w / np.sum(w)

    k_AW = 0.025 * (shear_rate ** 0.35) * np.exp(-1500.0 * (1.0 / T_K - 1.0 / 333.15))
    k_coal = (1.5e-2 * shear_rate) / max(1.0, W_steric)

    t_total_s = t_feed_s + 900.0
    t_eval = np.linspace(0.0, t_total_s, 65)
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
        aw_relax = k_AW * (np.sum(V_bins) * V_eq_vec - V_bins)
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
    traj_records, psd_snapshots = [], {}
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
        PDI = float(np.sum(v_norm * (D_grid - mean_D) ** 2) / (mean_D ** 2))
        traj_records.append({
            "Time_min": float(t_s / 60.0), "D10_nm": D10, "D50_nm": D50,
            "D90_nm": D90, "PDI": PDI, "Unincorporated_CsA_ppm": float(max(0.0, sol.y[M, idx]))
        })
        if idx in snap_indices:
            psd_snapshots[f"{t_s/60.0:.1f} min"] = v_norm / grad_logD

    return {
        "D_grid": D_grid, "traj_df": pd.DataFrame(traj_records), "psd_snapshots": psd_snapshots,
        "metrics": {
            "epsilon_W_kg": float(epsilon), "v_tip_m_s": float(v_tip), "shear_rate_s": float(shear_rate),
            "eta_K_um": float(eta_K_um), "tau_micro_ms": float(tau_micro_s * 1e3), "tau_macro_s": float(tau_macro_s),
            "Da_mix": float(Da_mix), "D_eq_nm": float(D_eq), "Final_D50_nm": traj_records[-1]["D50_nm"],
            "Final_D90_nm": traj_records[-1]["D90_nm"], "Final_PDI": traj_records[-1]["PDI"],
            "Final_Cryst_ppm": traj_records[-1]["Unincorporated_CsA_ppm"]
        }
    }

def simulate_sterile_filtration(mem_params, A_EFA_cm2, L_m_um, Q_filt_mL_min, V_tot_mL, c_hco_wt, c_oc40_wt, c_csa_wt, c):
    eps_m, tau_m, rho_p, A_sp = mem_params["eps_m"], mem_params["tau_m"], mem_params["rho_p"], mem_params["A_sp"]
    L_cm = L_m_um * 1e-4
    Q_cm3_s = Q_filt_mL_min / 60.0
    u_z_cm_s = (Q_cm3_s / A_EFA_cm2) / eps_m
    V_bulk_cm3 = A_EFA_cm2 * L_cm
    M_mem_g = V_bulk_cm3 * (1.0 - eps_m) * rho_p
    solid_to_void = ((1.0 - eps_m) / eps_m) * rho_p

    MW_surf_avg = 2460.0
    C_surf_in_mg = (c_hco_wt + c_oc40_wt) * 10.0
    C_CsA_in_mg = c_csa_wt * 10.0
    CMC_mix_mM = 0.0082
    CMC_mix_mg = CMC_mix_mM * MW_surf_avg / 1e3
    K_m_part, kappa_mg_mg, S_w_mg = 480.0, 0.098, 0.023

    d_pore_cm, r_p0_nm, D_eff = 0.22e-4, 110.0, 2.4e-7
    Gz = (u_z_cm_s * (d_pore_cm ** 2)) / (D_eff * L_cm * tau_m)
    Sh = 3.66 + (0.0668 * Gz) / (1.0 + 0.04 * (Gz ** (2.0 / 3.0)))
    k_f = (Sh * D_eff) / d_pore_cm
    a_v = (4.0 * eps_m) / d_pore_cm

    k_LDF_surf = 1.0 / (1.0 / mem_params["k_surf_s"] + 12.0 / (k_f * a_v))
    k_LDF_CsA = 1.0 / (1.0 / (mem_params["k_surf_s"] * 0.85) + 15.0 / (k_f * a_v))
    D_ax = (D_eff / tau_m) + 0.5 * u_z_cm_s * d_pore_cm * 15.0

    N_z = 18
    dz = L_cm / N_z

    def eq_q(C_s_mg, C_d_mg):
        C_s_mM = np.maximum(0.0, C_s_mg / MW_surf_avg * 1e3)
        C_d_mM = np.maximum(0.0, C_d_mg / c["MW_CsA"] * 1e3)
        C_s_free = np.minimum(C_s_mM, CMC_mix_mM)
        C_mic = np.maximum(0.0, C_s_mM - CMC_mix_mM)
        C_d_free = (1.0 / (1.0 + K_m_part * (C_mic * 1e-3))) * C_d_mM

        den_I = 1.0 + mem_params["KL_surf_I"] * C_s_free + mem_params["KL_CsA_I"] * C_d_free
        q_s_I = (mem_params["qmax_surf_I"] * mem_params["KL_surf_I"] * C_s_free) / den_I
        q_d_I = (mem_params["qmax_CsA_I"] * mem_params["KL_CsA_I"] * C_d_free) / den_I
        q_s_II = (mem_params["qmax_surf_II"] * mem_params["KL_surf_II"] * C_mic) / (1.0 + mem_params["KL_surf_II"] * C_mic)
        ratio = np.where(C_s_mg > 1e-5, C_d_mg / np.maximum(1e-4, C_s_mg), 0.0)
        q_d_II = mem_params["omega_co"] * ratio * q_s_II
        return (q_s_I + q_s_II), (q_d_I + q_d_II)

    y0 = np.zeros(4 * N_z)
    t_end_s = V_tot_mL / Q_cm3_s
    t_eval = np.linspace(0.0, t_end_s, 100)

    def adr_ode(t, y):
        C_s, C_d = np.maximum(0.0, y[:N_z]), np.maximum(0.0, y[N_z:2*N_z])
        q_s, q_d = np.maximum(0.0, y[2*N_z:3*N_z]), np.maximum(0.0, y[3*N_z:4*N_z])
        qs_st, qd_st = eq_q(C_s, C_d)
        dqs = k_LDF_surf * (qs_st - q_s)
        dqd = k_LDF_CsA * (qd_st - q_d)

        dCs, dCd = np.zeros(N_z), np.zeros(N_z)
        dCs[0] = u_z_cm_s * (C_surf_in_mg - C_s[0]) / dz + D_ax * (C_s[1] - C_s[0]) / (dz**2) - solid_to_void * dqs[0]
        dCd[0] = u_z_cm_s * (C_CsA_in_mg - C_d[0]) / dz + D_ax * (C_d[1] - C_d[0]) / (dz**2) - solid_to_void * dqd[0]
        dCs[1:-1] = -u_z_cm_s * (C_s[1:-1] - C_s[:-2]) / dz + D_ax * (C_s[2:] - 2*C_s[1:-1] + C_s[:-2]) / (dz**2) - solid_to_void * dqs[1:-1]
        dCd[1:-1] = -u_z_cm_s * (C_d[1:-1] - C_d[:-2]) / dz + D_ax * (C_d[2:] - 2*C_d[1:-1] + C_d[:-2]) / (dz**2) - solid_to_void * dqd[1:-1]
        dCs[-1] = -u_z_cm_s * (C_s[-1] - C_s[-2]) / dz + D_ax * (C_s[-2] - C_s[-1]) / (dz**2) - solid_to_void * dqs[-1]
        dCd[-1] = -u_z_cm_s * (C_d[-1] - C_d[-2]) / dz + D_ax * (C_d[-2] - C_d[-1]) / (dz**2) - solid_to_void * dqd[-1]
        return np.concatenate([dCs, dCd, dqs, dqd])

    sol = solve_ivp(adr_ode, [0.0, t_end_s], y0, t_eval=t_eval, method="RK45", rtol=1e-4, atol=1e-6)
    V_filt_arr = sol.t * Q_cm3_s
    records = []
    for idx, V_mL in enumerate(V_filt_arr):
        C_s_out, C_d_out = max(0.0, sol.y[N_z - 1, idx]), max(0.0, sol.y[2*N_z - 1, idx])
        q_s_p, q_d_p = sol.y[2*N_z:3*N_z, idx], sol.y[3*N_z:4*N_z, idx]
        delta_nm = ((q_s_p + q_d_p) / (1080.0 * A_sp)) * 1e5
        r_eff = np.maximum(25.0, r_p0_nm - delta_nm)
        tmp_ratio = float(np.mean((r_p0_nm / r_eff) ** 4))
        S_eq_out = S_w_mg + kappa_mg_mg * max(0.0, C_s_out - CMC_mix_mg)
        S_R_pore = C_d_out / max(1e-4, S_eq_out) if C_d_out > 0.01 else 0.0
        records.append({
            "V_filt_mL": float(V_mL), "CsA_Assay_pct": float((C_d_out / C_CsA_in_mg) * 100.0),
            "Surf_Assay_pct": float((C_s_out / C_surf_in_mg) * 100.0),
            "Mean_q_CsA_mg_g": float(np.mean(q_d_p)), "Mean_q_Surf_mg_g": float(np.mean(q_s_p)),
            "Adsorbed_Layer_nm": float(np.mean(delta_nm)), "TMP_Ratio": tmp_ratio, "Pore_S_R": float(S_R_pore)
        })
    df_bt = pd.DataFrame(records)
    ab98 = df_bt[df_bt["CsA_Assay_pct"] >= 98.0]
    V_fl_98 = float(ab98["V_filt_mL"].iloc[0]) if not ab98.empty else float(V_tot_mL)
    ab995 = df_bt[df_bt["CsA_Assay_pct"] >= 99.5]
    V_fl_995 = float(ab995["V_filt_mL"].iloc[0]) if not ab995.empty else float(V_tot_mL)

    vial_nums = np.arange(1, min(400, int(V_tot_mL / 0.25)) + 1)
    v_mids = (vial_nums - 0.5) * 0.25
    df_vials = pd.DataFrame({
        "BFS_Vial_Number": vial_nums,
        "Vial_CsA_Assay_pct": np.interp(v_mids, df_bt["V_filt_mL"], df_bt["CsA_Assay_pct"])
    })
    return {
        "df_bt": df_bt, "df_vials": df_vials, "Sh": float(Sh), "k_LDF_CsA": float(k_LDF_CsA),
        "summary": {
            "M_mem_g": float(M_mem_g), "V_flush_98_mL": V_fl_98, "V_flush_995_mL": V_fl_995,
            "BFS_Vials_Lost_98": int(np.ceil(V_fl_98 / 0.25)),
            "Total_CsA_Adsorbed_mg": float(df_bt["Mean_q_CsA_mg_g"].iloc[-1] * M_mem_g),
            "Total_Surf_Adsorbed_mg": float(df_bt["Mean_q_Surf_mg_g"].iloc[-1] * M_mem_g),
            "Max_Brush_Thickness_nm": float(df_bt["Adsorbed_Layer_nm"].max()),
            "Max_TMP_Increase_pct": float((df_bt["TMP_Ratio"].max() - 1.0) * 100.0),
            "Max_Pore_SR": float(df_bt["Pore_S_R"].max())
        }
    }

# ==============================================================================
# SIDEBAR OPERATING CONTROLS
# ==============================================================================
st.sidebar.title("Operating Setpoints")
st.sidebar.subheader("1. Formulation Q1/Q2 (% w/v)")
c_hco = st.sidebar.slider("HCO-40 (% w/v)", 0.40, 2.00, 1.00, 0.05)
c_oc40 = st.sidebar.slider("Octoxynol-40 (% w/v)", 0.01, 0.25, 0.05, 0.01)
c_csa = st.sidebar.slider("Cyclosporine A (% w/v)", 0.05, 0.15, 0.09, 0.01)
c_pvp = st.sidebar.slider("Povidone K90 (% w/v)", 0.10, 0.60, 0.30, 0.05)
pH_init = st.sidebar.slider("Initial Compounding pH", 6.00, 7.60, 6.80, 0.05)
beta_12_val = st.sidebar.slider("Rubingh Synergy β₁₂", -4.0, -0.5, -2.45, 0.05)

st.sidebar.subheader("2. Compounding Reactor")
V_batch = st.sidebar.number_input("Batch Volume V_b (L)", 1.0, 2000.0, 100.0, 10.0)
D_imp = st.sidebar.slider("Impeller Diameter Dᵢ (m)", 0.05, 0.60, 0.22, 0.01)
rpm_val = st.sidebar.slider("Impeller Speed (RPM)", 50, 1200, 350, 25)
t_feed = st.sidebar.slider("Pre-Concentrate Feed Time (min)", 2.0, 60.0, 15.0, 1.0)
T_comp = st.sidebar.slider("Hydration Temp (°C)", 25.0, 75.0, 55.0, 1.0)

c_dict = st.session_state.consts
C1_0 = (c_hco * 10.0 / c_dict["MW_1"]) * 1e3
C2_0 = (c_oc40 * 10.0 / c_dict["MW_2"]) * 1e3
C_CsA_mM = (c_csa * 10.0 / c_dict["MW_CsA"]) * 1e3

state_25C = solve_equilibrium_micelle(C1_0, C2_0, C_CsA_mM, 298.15, c_pvp, beta_12_val, c_dict)
_, _, _, _, _, V_max_25, logW_25 = dlvo_steric_profile(state_25C["R_c_nm"], state_25C["L_corona_nm"], 298.15, c_dict)
shelf_25C = simulate_shelf_life_ode(25.0, 24.0, C1_0, C2_0, C_CsA_mM, c_pvp, pH_init, beta_12_val, c_dict)
end_24m = shelf_25C.iloc[-1]

# ==============================================================================
# MAIN BANNER & KPI SCORECARD
# ==============================================================================
st.title("First-Principles Ophthalmic Nanomicellar Process Engineering Workbench")
st.markdown(
    "**Formulation Architecture:** Binary Non-Ionic Mixed Nanomicelles (**HCO-40 + Octoxynol-40**) "
    "encapsulating **Cyclosporine A (0.09% w/v)** in a **Povidone K90** buffered aqueous solution."
)

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Mixed CMC (25 °C)", f"{state_25C['CMC_mix']:.4f} mM", f"Ideal: {state_25C['CMC_ideal']:.4f} mM")
k2.metric("Equilibrium Dₕ", f"{state_25C['D_h_nm']:.2f} nm", f"N_agg = {state_25C['N_agg']:.0f}")
k3.metric("Initial S_R (t = 0)", f"{state_25C['S_R']:.3f}", "Stable (< 1.0)" if state_25C["S_R"] <= 1.0 else "Supersaturated!")
k4.metric("24-Mo S_R (25 °C)", f"{end_24m['S_R']:.3f}", f"S_eq = {end_24m['S_eq_mg_mL']:.2f} mg/mL")
k5.metric("Steric Barrier V_max", f"{V_max_25:.1f} k_B T", f"log₁₀(W) = {logW_25:.1f}")
k6.metric("24-Mo Intact HCO-40", f"{end_24m['Intact_HCO40_pct']:.1f}%", f"pH = {end_24m['pH']:.2f}")

tab_theory, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📖 1. Theory, Equations, Assumptions & Constants",
    "🎯 2. QbD Design Space & MODR",
    "⚙️ 3. Compounding PBE & Scale-Up",
    "🧪 4. Colloidal DLVO & 24-Mo Stability",
    "🏭 5. Multi-Scale Reactor Comparison",
    "🔬 6. 0.22 μm Sterile Filtration"
])

# ==============================================================================
# TAB 1: COMPLETE THEORY, ASSUMPTIONS, MANUAL CONSTANTS, EQUATIONS & REFERENCES
# ==============================================================================
with tab_theory:
    st.subheader("1. Conceptual Theory: First-Time User Guide to Ophthalmic Nanomicellar Engineering")
    st.markdown("""
    ### Why Nanomicellar Encapsulation is Required for Cyclosporine A
    **Cyclosporine A (CsA)** is a cyclic undecapeptide ($MW = 1202.61\\text{ g/mol}$, $\\log P \\approx 2.92$) with extreme lipophilicity and poor intrinsic water solubility ($S_w \\approx 19\\text{ }\\mu\\text{M}$ or $0.023\\text{ mg/mL}$ at $25^\\circ\\text{C}$). Conventional oil-in-water emulsions ($0.05\\%\\text{ w/v}$) suffer from ocular burning, blurred vision, phase separation, and low corneal bioavailability because free drug partitioning into the aqueous tear film is kinetically limited.

    In a **Binary Polymeric Nanomicellar System (NCELL® / Cequa® $0.09\\%\\text{ w/v}$ architecture)**, two non-ionic amphiphilic surfactants self-assemble into optically clear ($D_h \\approx 18\\text{--}24\\text{ nm}$) core–shell nanocarriers:
    1. **Primary Surfactant — Polyoxyl 40 Hydrogenated Castor Oil (HCO-40, $1.00\\%\\text{ w/v}$):** Possesses a bulky, three-branched 12-hydroxystearate hydrophobic tail and a 40-unit polyethylene oxide (PEO-40) hydrophilic headgroup. Alone, its branched tail leaves steric free-volume voids in the micellar core.
    2. **Co-Surfactant — Octoxynol-40 ($0.05\\%\\text{ w/v}$):** Possesses a compact, planar $p$-tert-octylphenyl hydrophobic tail and a PEO-40 corona. Intercalating Octoxynol-40 between the bulky HCO-40 chains reliefs core packing frustration, yielding a strongly negative **Rubingh interaction parameter ($\\beta_{12} \\approx -2.45$)**. This depresses the mixed critical micelle concentration ($\\text{CMC}_{\\text{mix}}$) nearly $2\\times$ below ideal mixing, preventing premature micellar disassembly upon 1:5 tear-film dilution.
    3. **Mucoadhesive Polymer — Povidone K90 (PVP, $0.30\\%\\text{ w/v}$):** Associates with the outer PEO-40 brush via hydrogen-bonding/hydrophobic stabilization, lowering the critical aggregation concentration (CAC) and increasing precorneal residence time.
    """)

    st.markdown("---")
    st.subheader("2. Fundamental Engineering & Physical Assumptions")
    st.markdown("""
    * **A1. Pseudophase Separation & Regular Solution Approximation:** Micelles ($N_{\\text{agg}} \\approx 120\\text{--}220$) are treated as a thermodynamic pseudophase in chemical potential equilibrium with monomeric surfactant in the bulk aqueous phase. Excess entropy of tail mixing in the core is assumed negligible compared to enthalpic tail-packing synergy ($\\Delta G_{\\text{mix}}^{\\text{E}} \\approx RT \\beta_{12} x_1 x_2$).
    * **A2. Spherical Core–Corona Geometry with Incompressible Volume Additivity:** The hydrophobic core radius $R_c$ is determined strictly by the sum of surfactant tail molecular volumes ($v_{t,1}, v_{t,2}$) and solubilized drug volume ($v_{\\text{CsA}}$), bounded by a sharp hydrocarbon–water interface ($\\sigma_{\\text{core/w}} \\approx 30\\text{ mN/m}$).
    * **A3. High-Ionic-Strength Electrostatic Screening:** In $0.125\\text{ M}$ sodium phosphate + NaCl buffer, the Debye screening length is $\\kappa^{-1} \\approx 0.86\\text{ nm}$. Because both surfactants are non-ionic, electrostatic double-layer repulsion is zero; colloidal stability is governed strictly by **Hamaker van der Waals attraction ($V_{\\text{vdW}}$)** versus **Vincent–Napper PEO-40 osmotic + elastic brush repulsion ($V_{\\text{osm}} + V_{\\text{el}}$)**.
    * **A4. Selective Ester Cleavage on Shelf-Life:** HCO-40 contains 3 hydrolysable glycerol–hydroxystearate ester linkages per molecule that undergo specific acid ($k_H$), neutral ($k_0$), and base ($k_{OH}$) catalyzed hydrolysis following Arrhenius kinetics ($E_a = 78.5\\text{ kJ/mol}$). Octoxynol-40 is an ether-linked octylphenol ethoxylate and is chemically non-hydrolysable over 24 months.
    * **A5. Fast Aniansson–Wall Monomer Exchange in $0.22\\text{ }\\mu\\text{m}$ Membrane Pores:** The micelle–monomer relaxation time ($\\tau_{\\text{AW}} \\sim 2\\text{ ms}$) is two orders of magnitude shorter than the hydraulic residence time inside the sterilizing membrane ($\\tau_{\\text{res}} = \\varepsilon_m L_m / J_v \\sim 0.4\\text{ s}$), maintaining local monomer–micelle equilibrium across every axial node $z$.
    """)

    st.markdown("---")
    st.subheader("3. Interactive Manual Constants Editor (Live Override Across All Tabs)")
    st.info("Modify any physical, thermodynamic, kinetic, or membrane constant below. All equations, step-by-step live calculations, and simulation tabs update automatically.")

    if st.button("Reset All Constants to Literature Defaults"):
        st.session_state.consts = DEFAULT_CONSTANTS.copy()
        st.rerun()

    exp1, exp2 = st.columns(2)
    with exp1:
        with st.expander("A. Molecular Weights & Geometric Volumes", expanded=False):
            c_dict["MW_1"] = st.number_input("HCO-40 Molecular Weight MW₁ (g/mol)", 500.0, 5000.0, float(c_dict["MW_1"]), 10.0)
            c_dict["MW_2"] = st.number_input("Octoxynol-40 Molecular Weight MW₂ (g/mol)", 500.0, 5000.0, float(c_dict["MW_2"]), 10.0)
            c_dict["MW_CsA"] = st.number_input("Cyclosporine A MW_CsA (g/mol)", 500.0, 2000.0, float(c_dict["MW_CsA"]), 1.0)
            c_dict["vt_1"] = st.number_input("HCO-40 Tail Volume v_t1 (nm³)", 0.20, 5.00, float(c_dict["vt_1"]), 0.05)
            c_dict["vt_2"] = st.number_input("Octoxynol-40 Tail Volume v_t2 (nm³)", 0.10, 2.00, float(c_dict["vt_2"]), 0.02)
            c_dict["v_CsA"] = st.number_input("Cyclosporine A Molecular Volume v_CsA (nm³)", 0.50, 4.00, float(c_dict["v_CsA"]), 0.05)
            c_dict["a0_shield"] = st.number_input("Shielded Interfacial Area a₀ (nm²)", 0.10, 1.50, float(c_dict["a0_shield"]), 0.02)
            c_dict["N_EO"] = st.number_input("PEO Chain Length N_EO (monomers)", 10.0, 100.0, float(c_dict["N_EO"]), 1.0)
            c_dict["l_EO"] = st.number_input("PEO Kuhn Length l_EO (nm)", 0.15, 0.80, float(c_dict["l_EO"]), 0.01)
            c_dict["v_EO"] = st.number_input("PEO Monomer Volume v_EO (nm³)", 0.02, 0.20, float(c_dict["v_EO"]), 0.005)
            c_dict["sigma_core"] = st.number_input("Core-Water Interfacial Tension σ (J/m²)", 0.005, 0.060, float(c_dict["sigma_core"]), 0.001, format="%.3f")

        with st.expander("B. Mixed Micellization & Electrolyte/PVP Constants", expanded=False):
            c_dict["CMC1_ref"] = st.number_input("Pure HCO-40 CMC₁,ref at 25°C (mM)", 0.001, 1.000, float(c_dict["CMC1_ref"]), 0.001, format="%.4f")
            c_dict["CMC2_ref"] = st.number_input("Pure Octoxynol-40 CMC₂,ref at 25°C (mM)", 0.010, 2.000, float(c_dict["CMC2_ref"]), 0.010, format="%.4f")
            c_dict["dH_mic1"] = st.number_input("HCO-40 Micellization Enthalpy ΔH°_mic,1 (J/mol)", -40000.0, 10000.0, float(c_dict["dH_mic1"]), 500.0)
            c_dict["dH_mic2"] = st.number_input("Octoxynol-40 Enthalpy ΔH°_mic,2 (J/mol)", -40000.0, 10000.0, float(c_dict["dH_mic2"]), 500.0)
            c_dict["ks_1"] = st.number_input("HCO-40 Setschenow Constant k_s1 (M⁻¹)", 0.0, 5.0, float(c_dict["ks_1"]), 0.05)
            c_dict["ks_2"] = st.number_input("Octoxynol-40 Setschenow Constant k_s2 (M⁻¹)", 0.0, 5.0, float(c_dict["ks_2"]), 0.05)
            c_dict["kp_1"] = st.number_input("HCO-40 PVP Complexation Factor k_p1 ((%w/v)⁻¹)", 0.0, 2.0, float(c_dict["kp_1"]), 0.02)
            c_dict["kp_2"] = st.number_input("Octoxynol-40 PVP Factor k_p2 ((%w/v)⁻¹)", 0.0, 2.0, float(c_dict["kp_2"]), 0.02)
            c_dict["I_ionic"] = st.number_input("Buffer Ionic Strength I (M)", 0.01, 0.50, float(c_dict["I_ionic"]), 0.005, format="%.3f")
            c_dict["beta_buf"] = st.number_input("Phosphate Buffer Capacity β_buf (mol/L/pH)", 0.005, 0.200, float(c_dict["beta_buf"]), 0.005, format="%.3f")

    with exp2:
        with st.expander("C. Hansen Solubility (HSP), DLVO & Ester Hydrolysis Kinetics", expanded=False):
            c_dict["d1_d"] = st.number_input("HCO-40 Core δ_d1 (MPa^0.5)", 12.0, 24.0, float(c_dict["d1_d"]), 0.1)
            c_dict["d1_p"] = st.number_input("HCO-40 Core δ_p1 (MPa^0.5)", 0.5, 12.0, float(c_dict["d1_p"]), 0.1)
            c_dict["d1_h"] = st.number_input("HCO-40 Core δ_h1 (MPa^0.5)", 0.5, 15.0, float(c_dict["d1_h"]), 0.1)
            c_dict["dC_d"] = st.number_input("Cyclosporine A δ_d,CsA (MPa^0.5)", 12.0, 24.0, float(c_dict["dC_d"]), 0.1)
            c_dict["dC_p"] = st.number_input("Cyclosporine A δ_p,CsA (MPa^0.5)", 0.5, 12.0, float(c_dict["dC_p"]), 0.1)
            c_dict["dC_h"] = st.number_input("Cyclosporine A δ_h,CsA (MPa^0.5)", 0.5, 15.0, float(c_dict["dC_h"]), 0.1)
            c_dict["Ea_hyd"] = st.number_input("HCO-40 Hydrolysis Activation Energy E_a (J/mol)", 30000.0, 130000.0, float(c_dict["Ea_hyd"]), 500.0)
            c_dict["k_H"] = st.number_input("Acid Rate Constant k_H (month⁻¹ M⁻¹)", 1e-4, 1.0, float(c_dict["k_H"]), 1e-3, format="%.4f")
            c_dict["k_0"] = st.number_input("Neutral Rate Constant k_0 (month⁻¹)", 1e-5, 1e-2, float(c_dict["k_0"]), 5e-5, format="%.5f")
            c_dict["k_OH"] = st.number_input("Base Rate Constant k_OH (month⁻¹ M⁻¹)", 10.0, 10000.0, float(c_dict["k_OH"]), 10.0)

        with st.expander("D. Reactor Impeller (PBE) & Custom 0.22 μm Membrane Constants", expanded=False):
            c_dict["N_p"] = st.number_input("Impeller Power Number N_p", 0.20, 6.00, float(c_dict["N_p"]), 0.05)
            c_dict["N_q"] = st.number_input("Impeller Pumping Number N_q", 0.20, 2.00, float(c_dict["N_q"]), 0.05)
            c_dict["A_sp"] = st.number_input("Custom Membrane BET Area A_sp (m²/g)", 5.0, 60.0, float(c_dict["A_sp"]), 0.5)
            c_dict["eps_m"] = st.number_input("Membrane Porosity ε_m", 0.40, 0.90, float(c_dict["eps_m"]), 0.01)
            c_dict["tau_m"] = st.number_input("Membrane Tortuosity τ_m", 1.00, 3.00, float(c_dict["tau_m"]), 0.05)
            c_dict["qmax_CsA_I"] = st.number_input("Site I Max CsA Capacity q_max,CsA^I (mg/g)", 0.05, 10.0, float(c_dict["qmax_CsA_I"]), 0.05)
            c_dict["KL_CsA_I"] = st.number_input("Site I CsA Affinity K_L,CsA^I (mM⁻¹)", 1.0, 250.0, float(c_dict["KL_CsA_I"]), 1.0)
            c_dict["qmax_surf_II"] = st.number_input("Site II PEO Corona Capacity q_max,surf^II (mg/g)", 0.10, 15.0, float(c_dict["qmax_surf_II"]), 0.10)
            c_dict["omega_co"] = st.number_input("Admicellar CsA Co-Adsorption Factor ω_co", 0.0, 1.0, float(c_dict["omega_co"]), 0.02)

    # Re-evaluate live walkthrough metrics with updated constants
    st_live = solve_equilibrium_micelle(C1_0, C2_0, C_CsA_mM, 298.15, c_pvp, beta_12_val, c_dict)
    pbe_live = run_compounding_pbe(C1_0, C2_0, C_CsA_mM, c_pvp, beta_12_val, V_batch, D_imp, rpm_val, t_feed, T_comp, c_dict)
    filt_live = simulate_sterile_filtration(c_dict, 1000.0, 240.0, 150.0, 500.0, c_hco, c_oc40, c_csa, c_dict)

    st.markdown("---")
    st.subheader("4. Governing Equations & Live Step-by-Step Numerical Calculation Walkthrough")

    st.markdown("#### Module 1: Rubingh–Setschenow Non-Ideal Mixed Micellization")
    st.latex(r"\ln \text{CMC}_i(T, I, C_{\text{PVP}}) = \ln \text{CMC}_{i,\text{ref}} + \frac{\Delta H_{\text{mic},i}^\circ}{R}\left(\frac{1}{T} - \frac{1}{T_{\text{ref}}}\right) - k_{s,i} I - k_{p,i} C_{\text{PVP}}")
    st.latex(r"\frac{x_1^2 \ln\left(\frac{\alpha_1 \text{CMC}_{\text{mix}}}{x_1 \text{CMC}_1}\right)}{(1 - x_1)^2 \ln\left(\frac{(1 - \alpha_1) \text{CMC}_{\text{mix}}}{(1 - x_1) \text{CMC}_2}\right)} = 1, \quad \gamma_1 = \exp\left[\beta_{12}(1-x_1)^2\right], \quad \frac{1}{\text{CMC}_{\text{mix}}} = \frac{\alpha_1}{\gamma_1 \text{CMC}_1} + \frac{1-\alpha_1}{\gamma_2 \text{CMC}_2}")
    st.code(
        f"[Live Calculation @ T = 298.15 K]\n"
        f"  Bulk Molar Concentrations : C1 (HCO-40) = {C1_0:.4f} mM | C2 (OC-40) = {C2_0:.4f} mM | Bulk Mole Fraction alpha_1 = {st_live['alpha_1']:.4f}\n"
        f"  Salted-Out Pure CMCs      : CMC_1 = {st_live['CMC1']:.5f} mM | CMC_2 = {st_live['CMC2']:.5f} mM\n"
        f"  Rubingh Root & Activities : Micellar x_1 = {st_live['x1']:.4f} | gamma_1 = {st_live['gamma1']:.4f} | gamma_2 = {st_live['gamma2']:.4f}\n"
        f"  Synergistic Mixed CMC     : CMC_mix = {st_live['CMC_mix']:.5f} mM  (vs. Ideal Clint CMC_ideal = {st_live['CMC_ideal']:.5f} mM)",
        language="text"
    )

    st.markdown("#### Module 2: Nagarajan–Ruckenstein Core–Corona Free Energy & Daoud–Cotton Brush Scaling")
    st.latex(r"\frac{\Delta \mu_{\text{mic}}^\circ(N_{\text{agg}})}{k_B T} = \frac{\sigma_{\text{core/w}}}{k_B T}(a - a_0) + \frac{3\pi^2 R_c^2}{80 L_{\text{tail}}^2} + \nu_F N_{\text{EO}}\left(\frac{v_{\text{EO}}}{a R_c}\right)^{0.85}\left(\frac{1}{2} - \chi_{\text{EO-w}}(T)\right)")
    st.latex(r"V_{\text{core}} = \frac{4}{3}\pi R_c^3, \quad L_{\text{corona}} = R_c\left[\left(1 + \frac{5}{3}\frac{N_{\text{EO}} l_{\text{EO}}^{5/3}}{R_c a^{1/3}}\left(\frac{1}{2}-\chi_{\text{EO-w}}\right)^{1/3}\right)^{3/5} - 1\right], \quad D_h = 2(R_c + L_{\text{corona}} + \delta_{\text{PVP}})")
    st.code(
        f"[Live Calculation @ Equilibrium Minimum d(Delta_mu)/d(N_agg) = 0]\n"
        f"  Optimal Aggregation Number : N_agg* = {st_live['N_agg']:.1f} monomers/micelle | Interfacial Area a* = {st_live['a_opt_nm2']:.3f} nm^2/monomer\n"
        f"  Flory-Huggins PEO-Water    : chi_EO-w(298.15 K) = {st_live['chi_ew']:.4f} (< 0.50 good solvent regime)\n"
        f"  Radial Dimensions          : Core R_c = {st_live['R_c_nm']:.2f} nm | Corona L = {st_live['L_corona_nm']:.2f} nm | PVP Layer = {st_live['delta_PVP_nm']:.2f} nm\n"
        f"  Hydrodynamic Diameter      : D_h = {st_live['D_h_nm']:.2f} nm",
        language="text"
    )

    st.markdown("#### Module 3: Laplace-Corrected Flory–Huggins Core Solubilization & Supersaturation Ratio ($S_R$)")
    st.latex(r"\chi_{\text{dc}} = \frac{V_{m,\text{CsA}}}{RT}\left[(\delta_{d,\text{CsA}} - \bar{\delta}_{d,\text{core}})^2 + 0.25(\delta_{p,\text{CsA}} - \bar{\delta}_{p,\text{core}})^2 + 0.25(\delta_{h,\text{CsA}} - \bar{\delta}_{h,\text{core}})^2\right], \quad \Delta P_{\text{Laplace}} = \frac{2\sigma_{\text{core/w}}}{R_c}")
    st.latex(r"S_{\text{tot}}^{\text{eq}}(t, T) = S_w(T) + \kappa(T, \chi_{\text{dc}}, R_c)\left[C_{\text{surf,tot}}(t) - \text{CMC}_{\text{mix}}(t, T)\right], \quad S_R(t, T) = \frac{C_{\text{CsA,nominal}}}{S_{\text{tot}}^{\text{eq}}(t, T)}")
    st.code(
        f"[Live Calculation @ C_CsA,nominal = {c_csa:.2f}% w/v ({C_CsA_mM:.3f} mM)]\n"
        f"  Core-Drug Compatibility : Flory-Huggins chi_dc = {st_live['chi_dc']:.4f} | Laplace Penalty (2*sigma*v_CsA / R_c*k_B*T) = {st_live['laplace_kT']:.3f} k_B*T\n"
        f"  Solubilization Capacity : Molar kappa = {st_live['kappa']:.4f} mol CsA / mol micellar surfactant | Intrinsic S_w = {st_live['S_w_mM']:.4f} mM\n"
        f"  Total Equilibrium Solub.: S_tot^eq(t=0) = {st_live['S_tot_eq_mg_mL']:.3f} mg/mL ({st_live['S_tot_eq_mM']:.3f} mM)\n"
        f"  Supersaturation Ratio   : S_R(t=0) = {st_live['S_R']:.3f}  |  S_R(t=24 mo @ 25 C) = {end_24m['S_R']:.3f}",
        language="text"
    )

    st.markdown("#### Module 4 & 5: Turbulent Compounding PBE & 0.22 μm Membrane Dual-Site ADR Transport")
    st.latex(r"\epsilon = \frac{N_p N^3 D_i^5}{V_b}, \quad \tau_{\text{micro}} = 17.24\left(\frac{\nu}{\epsilon}\right)^{1/2}, \quad Da_{\text{mix}} = \frac{\tau_{\text{micro}}(1 + 150 Q_{\text{feed}}/Q_{\text{pump}})}{\tau_{\text{nuc}}(S_{R,\text{loc}})}")
    st.latex(r"\frac{\partial C_k}{\partial t} + u_z \frac{\partial C_k}{\partial z} = D_{\text{ax}} \frac{\partial^2 C_k}{\partial z^2} - \frac{1-\varepsilon_m}{\varepsilon_m}\rho_p k_{\text{LDF},k}\left(q_k^{\text{I}*}(C_{\text{free}}) + q_k^{\text{II}*}(C_{\text{mic}}) - q_k\right)")
    pm = pbe_live["metrics"]
    fs = filt_live["summary"]
    st.code(
        f"[Live Compounding PBE & 0.22 um Filtration Audit]\n"
        f"  Reactor Hydrodynamics : Power epsilon = {pm['epsilon_W_kg']:.3f} W/kg | Kolmogorov eta_K = {pm['eta_K_um']:.1f} um | tau_micro = {pm['tau_micro_ms']:.2f} ms\n"
        f"  PBE Self-Assembly     : Micromixing Da_mix = {pm['Da_mix']:.3f} | Final D50 = {pm['Final_D50_nm']:.2f} nm | Unincorporated CsA = {pm['Final_Cryst_ppm']:.2f} ppm\n"
        f"  0.22 um Pore Kinetics : Graetz Sherwood Sh = {filt_live['Sh']:.2f} | CsA LDF Rate k_LDF = {filt_live['k_LDF_CsA']:.3f} s^-1 | Dry Membrane Mass = {fs['M_mem_g']:.2f} g\n"
        f"  Pre-Flush Requirement : V_flush(>=98% Assay) = {fs['V_flush_98_mL']:.1f} mL ({fs['BFS_Vials_Lost_98']} BFS Vials) | Total CsA Bound = {fs['Total_CsA_Adsorbed_mg']:.2f} mg",
        language="text"
    )

    st.markdown("---")
    st.subheader("5. Literature References & Model Provenance")
    ref_df = pd.DataFrame([
        {"Scale / Module": "1. Mixed Micellization (RST)", "Reference": "Rubingh, D. N. (1979). Solution Chemistry of Surfactants, Vol. 1, 337–354; Clint, J. H. (1975). J. Chem. Soc., Faraday Trans. 1, 71, 1327.", "Role in Workbench": "Binary non-ideal interaction parameter β₁₂, micellar mole fraction x₁, and CMC_mix depression."},
        {"Scale / Module": "2. Electrolyte & Polymer Salting-Out", "Reference": "Setschenow, J. (1889). Z. Phys. Chem., 4, 117; Goddard, E. D. (1986). Colloids Surf., 19, 255–300.", "Role in Workbench": "Ionic strength (k_s) and Povidone K90 corona complexation (k_p) shift on intrinsic CMCs."},
        {"Scale / Module": "3. Micellar Core–Corona Packing", "Reference": "Nagarajan, R., & Ruckenstein, E. (1991). Langmuir, 7(12), 2938–2969; Daoud, M., & Cotton, J. P. (1982). J. Phys. (Paris), 43, 531.", "Role in Workbench": "Free energy minimization for aggregation number N_agg*, core radius R_c, and curved PEO-40 brush thickness L."},
        {"Scale / Module": "4. Core Solubilization & HSP", "Reference": "Flory, P. J. (1953). Principles of Polymer Chemistry; Hansen, C. M. (2007). Hansen Solubility Parameters: A User's Handbook.", "Role in Workbench": "Drug-core compatibility parameter χ_dc coupled with Laplace curvature penalty 2σ/R_c."},
        {"Scale / Module": "5. Steric Stability & Ester Kinetics", "Reference": "Vincent, B., et al. (1986). Colloids Surf., 18, 261; Fuchs, N. (1934). Z. Physik, 89, 736; Florence, A. T. (2011). Physicochemical Principles of Pharmacy.", "Role in Workbench": "PEO-40 osmotic/elastic brush barrier V_tot(h), Fuchs stability ratio W, and pH-rate profile of HCO-40 ester cleavage."},
        {"Scale / Module": "6. Compounding PBE & Micromixing", "Reference": "Aniansson, E. A. G., & Wall, S. N. (1974). J. Phys. Chem., 78(10), 1024; Baldyga, J., & Bourne, J. R. (1999). Turbulent Mixing and Chemical Reactions.", "Role in Workbench": "Engulfment micromixing time τ_micro, Damköhler number Da_mix, droplet fission, and monomer relaxation to D_h*."},
        {"Scale / Module": "7. 0.22 μm Membrane Adsorption", "Reference": "Graetz, L. (1885). Ann. Phys., 261, 337; Ruthven, D. M. (1984). Principles of Adsorption and Adsorption Processes.", "Role in Workbench": "1D advection-dispersion-reaction PDE with Graetz-Lévêque pore mass transfer and dual-site monomer/PEO binding."}
    ])
    st.dataframe(ref_df, use_container_width=True)

# ==============================================================================
# TAB 2: 2D QbD DESIGN SPACE & MODR
# ==============================================================================
with tab1:
    st.subheader("2D Quality by Design (QbD) Response Surfaces & Method Operable Design Region (MODR)")
    c_left, c_right = st.columns([1, 3])
    with c_left:
        design_axes = st.radio(
            "Select Design Space Axes:",
            ["Surfactant Space (HCO-40 % vs. Octoxynol-40 %)",
             "Chemical Stability Space (Initial pH vs. Storage Temp °C)"]
        )
        
        st.markdown("**MODR Criteria:** $S_R(24\\text{ mo}) \\le 1.00$, $18 \\le D_h \\le 24.5\\text{ nm}$, $6.5 \\le \\text{pH} \\le 7.2$.")

    with c_right:
        if "Surfactant Space" in design_axes:
            hco_grid = np.linspace(0.50, 1.60, grid_res)
            oc_grid = np.linspace(0.01, 0.20, grid_res)
            SR_24_mat, Dh_mat = np.zeros((grid_res, grid_res)), np.zeros((grid_res, grid_res))
            ret_24m = end_24m["Intact_HCO40_pct"] / 100.0

            for i, oc_v in enumerate(oc_grid):
                c2_v = (oc_v * 10.0 / c_dict["MW_2"]) * 1e3
                for j, hco_v in enumerate(hco_grid):
                    c1_v = (hco_v * 10.0 / c_dict["MW_1"]) * 1e3
                    st_0 = solve_equilibrium_micelle(c1_v, c2_v, C_CsA_mM, 298.15, c_pvp, beta_12_val, c_dict)
                    st_24 = solve_equilibrium_micelle(c1_v * ret_24m, c2_v, C_CsA_mM, 298.15, c_pvp, beta_12_val, c_dict)
                    Dh_mat[i, j] = st_0["D_h_nm"]
                    SR_24_mat[i, j] = st_24["S_R"]

            fig_qbd = make_subplots(rows=1, cols=2, subplot_titles=("24-Mo Supersaturation Ratio S_R (Limit = 1.0)", "Equilibrium Micelle Diameter Dₕ (nm)"))
            fig_qbd.add_trace(go.Contour(x=hco_grid, y=oc_grid, z=SR_24_mat, colorscale="RdYlGn_r", contours=dict(start=0.5, end=1.5, size=0.05, showlabels=True), colorbar=dict(title="S_R", x=0.45)), row=1, col=1)
            fig_qbd.add_trace(go.Contour(x=hco_grid, y=oc_grid, z=SR_24_mat, contours=dict(coloring="none", start=1.0, end=1.0, size=0.1, showlabels=True), line=dict(color="black", width=4, dash="dash"), showscale=False), row=1, col=1)
            fig_qbd.add_trace(go.Scatter(x=[c_hco], y=[c_oc40], mode="markers+text", marker=dict(size=12, color="blue", symbol="star"), text=["Setpoint"], textposition="top center"), row=1, col=1)
            fig_qbd.add_trace(go.Contour(x=hco_grid, y=oc_grid, z=Dh_mat, colorscale="Viridis", contours=dict(showlabels=True), colorbar=dict(title="Dₕ (nm)", x=1.0)), row=1, col=2)
            fig_qbd.add_trace(go.Scatter(x=[c_hco], y=[c_oc40], mode="markers", marker=dict(size=12, color="red", symbol="star"), showlegend=False), row=1, col=2)
            fig_qbd.update_xaxes(title_text="HCO-40 (% w/v)")
            fig_qbd.update_yaxes(title_text="Octoxynol-40 (% w/v)")
            fig_qbd.update_layout(height=440, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_qbd, use_container_width=True)
        else:
            pH_grid = np.linspace(6.0, 7.6, grid_res)
            T_grid = np.linspace(5.0, 42.0, grid_res)
            SR_mat, HCO_ret_mat = np.zeros((grid_res, grid_res)), np.zeros((grid_res, grid_res))
            for i, T_val in enumerate(T_grid):
                T_K = T_val + 273.15
                pKw = 14.0 - 0.032 * (T_val - 25.0)
                arrh = np.exp(-(c_dict["Ea_hyd"] / R_gas) * (1.0 / T_K - 1.0 / 298.15))
                for j, ph_val in enumerate(pH_grid):
                    k_hyd = arrh * (c_dict["k_H"] * (10.0 ** (-ph_val)) + c_dict["k_0"] + c_dict["k_OH"] * (10.0 ** (ph_val - pKw)))
                    c1_24 = C1_0 * np.exp(-k_hyd * 24.0)
                    HCO_ret_mat[i, j] = (c1_24 / C1_0) * 100.0
                    SR_mat[i, j] = solve_equilibrium_micelle(c1_24, C2_0, C_CsA_mM, T_K, c_pvp, beta_12_val, c_dict)["S_R"]

            fig_ph = make_subplots(rows=1, cols=2, subplot_titles=("24-Month Supersaturation Ratio S_R(pH, T)", "24-Month Intact HCO-40 Retention (%)"))
            fig_ph.add_trace(go.Contour(x=pH_grid, y=T_grid, z=SR_mat, colorscale="RdYlGn_r", contours=dict(start=0.6, end=1.6, size=0.05, showlabels=True), colorbar=dict(title="S_R", x=0.45)), row=1, col=1)
            fig_ph.add_trace(go.Contour(x=pH_grid, y=T_grid, z=SR_mat, contours=dict(coloring="none", start=1.0, end=1.0, size=0.1, showlabels=True), line=dict(color="black", width=4, dash="dash"), showscale=False), row=1, col=1)
            fig_ph.add_trace(go.Contour(x=pH_grid, y=T_grid, z=HCO_ret_mat, colorscale="Blues", contours=dict(showlabels=True), colorbar=dict(title="HCO-40 (%)", x=1.0)), row=1, col=2)
            fig_ph.update_xaxes(title_text="Initial Formulation pH")
            fig_ph.update_yaxes(title_text="Storage Temperature (°C)")
            fig_ph.update_layout(height=440, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_ph, use_container_width=True)

# ==============================================================================
# TAB 3: COMPOUNDING PBE & SCALE-UP SIMULATOR
# ==============================================================================
with tab2:
    st.subheader("Population Balance Equation (PBE) — High-Shear Hydration & Self-Assembly")
    pbe_res = run_compounding_pbe(C1_0, C2_0, C_CsA_mM, c_pvp, beta_12_val, V_batch, D_imp, rpm_val, t_feed, T_comp, c_dict)
    m = pbe_res["metrics"]

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Specific Power (ε)", f"{m['epsilon_W_kg']:.3f} W/kg", f"v_tip = {m['v_tip_m_s']:.2f} m/s")
    m2.metric("Kolmogorov Scale (η_K)", f"{m['eta_K_um']:.1f} μm", f"γ̇ = {m['shear_rate_s']:.0f} s⁻¹")
    m3.metric("Micromixing Time (τ_micro)", f"{m['tau_micro_ms']:.1f} ms", f"τ_macro = {m['tau_macro_s']:.1f} s")
    m4.metric("Micromixing Da_mix", f"{m['Da_mix']:.3f}", "Safe (< 0.12)" if m["Da_mix"] <= 0.12 else "Nucleation Risk!")
    m5.metric("Final Micelle D₅₀ / D₉₀", f"{m['Final_D50_nm']:.1f} / {m['Final_D90_nm']:.1f} nm", f"PDI = {m['Final_PDI']:.3f}")
    m6.metric("Unincorporated CsA", f"{m['Final_Cryst_ppm']:.2f} ppm", "0 ppm Target")

    fig_pbe = make_subplots(rows=1, cols=2, subplot_titles=("PBE Volume-Weighted PSD Evolution", "Transient D₁₀, D₅₀, D₉₀ & Unincorporated Crystals"), specs=[[{"secondary_y": False}, {"secondary_y": True}]])
    for label, psd_curve in pbe_res["psd_snapshots"].items():
        fig_pbe.add_trace(go.Scatter(x=pbe_res["D_grid"], y=psd_curve, mode="lines", name=f"PSD @ {label}"), row=1, col=1)
    fig_pbe.add_vline(x=m["D_eq_nm"], line_dash="dash", line_color="green", annotation_text=f"Dₕ* ({m['D_eq_nm']:.1f} nm)", row=1, col=1)

    df_t = pbe_res["traj_df"]
    fig_pbe.add_trace(go.Scatter(x=df_t["Time_min"], y=df_t["D90_nm"], name="D₉₀ (nm)", line=dict(color="#d62728", width=2.5)), row=1, col=2, secondary_y=False)
    fig_pbe.add_trace(go.Scatter(x=df_t["Time_min"], y=df_t["D50_nm"], name="D₅₀ (nm)", line=dict(color="#1f77b4", width=3.0)), row=1, col=2, secondary_y=False)
    fig_pbe.add_trace(go.Scatter(x=df_t["Time_min"], y=df_t["D10_nm"], name="D₁₀ (nm)", line=dict(color="#2ca02c", width=2.0, dash="dot")), row=1, col=2, secondary_y=False)
    fig_pbe.add_trace(go.Scatter(x=df_t["Time_min"], y=df_t["Unincorporated_CsA_ppm"], name="Unincorporated CsA (ppm)", line=dict(color="#9467bd", width=2.5, dash="dashdot")), row=1, col=2, secondary_y=True)
    fig_pbe.update_xaxes(title_text="Hydrodynamic Diameter Dₕ (nm) [Log]", type="log", row=1, col=1)
    fig_pbe.update_yaxes(title_text="Volume Density dV/d(log₁₀ D)", row=1, col=1)
    fig_pbe.update_xaxes(title_text="Batch Time (min)", row=1, col=2)
    fig_pbe.update_yaxes(title_text="Diameter (nm)", secondary_y=False, row=1, col=2)
    fig_pbe.update_yaxes(title_text="Unincorporated CsA (ppm)", secondary_y=True, row=1, col=2)
    fig_pbe.update_layout(height=440, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_pbe, use_container_width=True)

# ==============================================================================
# TAB 4: COLLOIDAL DLVO STERIC POTENTIAL & 24-MONTH STABILITY
# ==============================================================================
with tab3:
    st.subheader("Extended DLVO + PEO-40 Brush Steric Potential & 24-Month ICH Stability Kinetics")
    fig_stab = make_subplots(rows=1, cols=2, subplot_titles=("Pair Interaction Potential V_tot(h) Across Temperatures", "24-Month Supersaturation Ratio S_R(t) Trajectory"))
    for T_test, col_c in [(25.0, "#1f77b4"), (40.0, "#ff7f0e"), (60.0, "#d62728"), (78.0, "#7f7f7f")]:
        T_K = T_test + 273.15
        st_temp = solve_equilibrium_micelle(C1_0, C2_0, C_CsA_mM, T_K, c_pvp, beta_12_val, c_dict)
        h_arr, _, _, _, V_tot, V_max, _ = dlvo_steric_profile(st_temp["R_c_nm"], st_temp["L_corona_nm"], T_K, c_dict)
        fig_stab.add_trace(go.Scatter(x=h_arr, y=V_tot, mode="lines", name=f"{T_test:.0f}°C (V_max={V_max:.1f} k_B T)", line=dict(color=col_c, width=2.5)), row=1, col=1)

    for T_ich, col_c in [(5.0, "#2ca02c"), (25.0, "#1f77b4"), (30.0, "#ff7f0e"), (40.0, "#d62728")]:
        df_ich = simulate_shelf_life_ode(T_ich, 24.0, C1_0, C2_0, C_CsA_mM, c_pvp, pH_init, beta_12_val, c_dict)
        fig_stab.add_trace(go.Scatter(x=df_ich["Month"], y=df_ich["S_R"], mode="lines", name=f"ICH {T_ich:.0f}°C", line=dict(color=col_c, width=2.8)), row=1, col=2)

    fig_stab.add_hline(y=1.0, line_dash="dash", line_color="black", annotation_text="Precipitation Threshold (S_R = 1.0)", row=1, col=2)
    fig_stab.update_xaxes(title_text="Surface Separation h (nm)", row=1, col=1)
    fig_stab.update_yaxes(title_text="Potential V(h) / k_B T", range=[-10, 80], row=1, col=1)
    fig_stab.update_xaxes(title_text="Storage Time (Months)", row=1, col=2)
    fig_stab.update_yaxes(title_text="Supersaturation Ratio S_R", row=1, col=2)
    fig_stab.update_layout(height=440, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_stab, use_container_width=True)

# ==============================================================================
# TAB 5: MULTI-SCALE REACTOR COMPARISON
# ==============================================================================
with tab4:
    st.subheader("Multi-Scale Reactor Scale-Up Comparison (5 L Lab vs. 50 L Pilot vs. 500 L Commercial)")
    scale_criterion = st.selectbox(
        "Select Scale-Up Criterion:",
        ["Constant Specific Power Input (ε = P/V)", "Constant Impeller Tip Speed (v_tip = π N Dᵢ)", "Constant Micromixing Damköhler Number (Da_mix)"]
    )
    scales = [{"Scale": "Lab R&D Batch", "V_L": 5.0, "D_i_m": 0.08}, {"Scale": "Pilot GMP Scale", "V_L": 50.0, "D_i_m": 0.18}, {"Scale": "Commercial Scale", "V_L": 500.0, "D_i_m": 0.38}]
    base_eps, base_tip = pbe_res["metrics"]["epsilon_W_kg"], pbe_res["metrics"]["v_tip_m_s"]

    comp_rows = []
    for s in scales:
        V_s, D_s = s["V_L"], s["D_i_m"]
        if "Constant Specific Power" in scale_criterion:
            N_rps = ((base_eps * (V_s * 1e-3)) / (c_dict["N_p"] * (D_s ** 5))) ** (1.0 / 3.0)
        elif "Constant Impeller Tip" in scale_criterion:
            N_rps = base_tip / (np.pi * D_s)
        else:
            N_rps = ((base_eps * (V_s * 1e-3)) / (c_dict["N_p"] * (D_s ** 5))) ** (1.0 / 3.0) * 1.12
        rpm_s = float(N_rps * 60.0)
        ms = run_compounding_pbe(C1_0, C2_0, C_CsA_mM, c_pvp, beta_12_val, V_s, D_s, rpm_s, t_feed, T_comp, c_dict)["metrics"]
        comp_rows.append({
            "Batch Scale": s["Scale"], "Volume (L)": V_s, "Impeller Dᵢ (m)": D_s, "Speed (RPM)": round(rpm_s, 1),
            "Power ε (W/kg)": round(ms["epsilon_W_kg"], 3), "Tip Speed (m/s)": round(ms["v_tip_m_s"], 2),
            "τ_micro (ms)": round(ms["tau_micro_ms"], 2), "Da_mix": round(ms["Da_mix"], 3),
            "Final D₅₀ (nm)": round(ms["Final_D50_nm"], 2), "Final D₉₀ (nm)": round(ms["Final_D90_nm"], 2),
            "Final PDI": round(ms["Final_PDI"], 3), "Unincorporated CsA (ppm)": round(ms["Final_Cryst_ppm"], 2)
        })
    df_comp = pd.DataFrame(comp_rows)
    st.dataframe(df_comp, use_container_width=True)
    st.download_button("Download Scale-Up Report (CSV)", df_comp.to_csv(index=False).encode("utf-8"), "nanomicellar_scaleup.csv", "text/csv")

# ==============================================================================
# TAB 6: 0.22 um STERILE FILTRATION ADSORPTION & BFS PRE-FLUSH
# ==============================================================================
with tab5:
    st.subheader("0.22 μm Sterile Filtration Membrane Adsorption, Breakthrough & BFS Pre-Flush Sizing")
    col_cfg, col_plots = st.columns([1, 3])
    with col_cfg:
        mem_mode = st.selectbox("Select 0.22 μm Membrane Polymer:", ["Use Manual Constants from Tab 1"] + list(MEMBRANE_PRESETS.keys()), index=2)
        A_efa = st.slider("Effective Filtration Area A_EFA (cm²)", 100.0, 5000.0, 1000.0, 100.0)
        L_bed = st.slider("Effective Membrane Thickness L_m (μm)", 120.0, 450.0, 240.0, 20.0)
        Q_flow = st.slider("Filtration Flow Rate Q (mL/min)", 20.0, 1000.0, 150.0, 10.0)
        V_sim = st.slider("Simulated Filtrate Volume (mL)", 100.0, 1500.0, 500.0, 50.0)

    active_mem = c_dict if mem_mode == "Use Manual Constants from Tab 1" else MEMBRANE_PRESETS[mem_mode]
    res_f = simulate_sterile_filtration(active_mem, A_efa, L_bed, Q_flow, V_sim, c_hco, c_oc40, c_csa, c_dict)
    sf, df_bt, df_vials = res_f["summary"], res_f["df_bt"], res_f["df_vials"]

    f1, f2, f3, f4, f5, f6 = st.columns(6)
    f1.metric("Pre-Flush to 98% Assay", f"{sf['V_flush_98_mL']:.1f} mL", f"{sf['BFS_Vials_Lost_98']} BFS Vials")
    f2.metric("Pre-Flush to 99.5% Assay", f"{sf['V_flush_995_mL']:.1f} mL", f"{sf['V_flush_995_mL']/A_efa*10:.2f} L/m²")
    f3.metric("Total CsA Adsorbed", f"{sf['Total_CsA_Adsorbed_mg']:.2f} mg", f"Filter Mass: {sf['M_mem_g']:.1f} g")
    f4.metric("Total Surfactant Bound", f"{sf['Total_Surf_Adsorbed_mg']:.1f} mg", "Site I + Site II")
    f5.metric("Adsorbed Brush δ_ads", f"{sf['Max_Brush_Thickness_nm']:.2f} nm", f"ΔP_TMP +{sf['Max_TMP_Increase_pct']:.1f}%")
    f6.metric("Peak Pore S_R Wave", f"{sf['Max_Pore_SR']:.3f}", "Safe (< 1.0)" if sf["Max_Pore_SR"] <= 1.0 else "Nucleation Risk!")

    with col_plots:
        fig_f1 = make_subplots(rows=1, cols=2, subplot_titles=("Selected Membrane Breakthrough (% Label Claim)", "Cross-Polymer CsA Breakthrough Comparison"))
        fig_f1.add_trace(go.Scatter(x=df_bt["V_filt_mL"], y=df_bt["CsA_Assay_pct"], mode="lines", name="CsA Filtrate Assay (%)", line=dict(color="#d62728", width=3.0)), row=1, col=1)
        fig_f1.add_trace(go.Scatter(x=df_bt["V_filt_mL"], y=df_bt["Surf_Assay_pct"], mode="lines", name="Mixed Surfactant (%)", line=dict(color="#1f77b4", width=2.5, dash="dash")), row=1, col=1)
        fig_f1.add_hline(y=98.0, line_dash="dot", line_color="green", row=1, col=1)
        fig_f1.add_vline(x=sf["V_flush_98_mL"], line_dash="dash", line_color="black", row=1, col=1)

        for m_name, c_hex in [("Hydrophilic PVDF (Ultra-Low Binding)", "#2ca02c"), ("Hydrophilic PES (Polyethersulfone)", "#ff7f0e"), ("Nylon 6,6 (High Amide H-Bonding)", "#9467bd")]:
            r_cmp = simulate_sterile_filtration(MEMBRANE_PRESETS[m_name], A_efa, L_bed, Q_flow, V_sim, c_hco, c_oc40, c_csa, c_dict)
            fig_f1.add_trace(go.Scatter(x=r_cmp["df_bt"]["V_filt_mL"], y=r_cmp["df_bt"]["CsA_Assay_pct"], mode="lines", name=f"{m_name.split(' (')[0]} ({r_cmp['summary']['V_flush_98_mL']:.0f} mL)", line=dict(color=c_hex, width=2.5)), row=1, col=2)
        fig_f1.add_hline(y=98.0, line_dash="dot", line_color="green", row=1, col=2)
        fig_f1.update_xaxes(title_text="Cumulative Filtrate Volume (mL)")
        fig_f1.update_yaxes(title_text="Filtrate Potency (%)", range=[0, 105])
        fig_f1.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_f1, use_container_width=True)
