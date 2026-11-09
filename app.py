import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar
from scipy.integrate import quad, solve_ivp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Nanomicellar QbD & PBE Scale-Up Workbench",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Physical Constants
k_B = 1.380649e-23       # Boltzmann constant (J/K)
N_A = 6.02214076e23      # Avogadro's number (1/mol)
R_gas = 8.314462         # Universal gas constant (J/mol*K)

# ==============================================================================
# FIRST-PRINCIPLES THERMODYNAMIC & STABILITY ENGINE
# ==============================================================================
class NanomicelleFormulation:
    def __init__(self, c_hco_wt=1.00, c_oc40_wt=0.05, c_csa_wt=0.09, c_pvp_wt=0.30,
                 pH_0=6.80, ionic_strength=0.125, beta_12=-2.45):
        self.C_HCO_wt = c_hco_wt
        self.C_OC40_wt = c_oc40_wt
        self.C_CsA_wt = c_csa_wt
        self.C_PVP_wt = c_pvp_wt
        self.pH_0 = pH_0
        self.I_ionic = ionic_strength
        self.beta_12 = beta_12
        self.beta_buf = 0.035  # Phosphate buffer capacity (mol/L/pH)

        # Molecular Weights (g/mol)
        self.MW_1 = 2500.0     # Polyoxyl 40 Hydrogenated Castor Oil (HCO-40)
        self.MW_2 = 1966.0     # Octoxynol-40 (OC-40)
        self.MW_CsA = 1202.61  # Cyclosporine A

        # Molar Concentrations (mM)
        self.C1_0 = (self.C_HCO_wt * 10.0 / self.MW_1) * 1e3
        self.C2_0 = (self.C_OC40_wt * 10.0 / self.MW_2) * 1e3
        self.C_CsA = (self.C_CsA_wt * 10.0 / self.MW_CsA) * 1e3

        # Pure Surfactant Thermodynamic Reference Values (298.15 K)
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

        # HCO-40 Ester Hydrolysis Arrhenius & pH Rate Constants (month^-1)
        self.Ea_hyd = 78500.0
        self.k_H = 1.8e-2
        self.k_0 = 4.5e-4
        self.k_OH = 8.2e2


def chi_EO_water(T):
    """Temperature-dependent Flory-Huggins parameter for PEO-40 in water."""
    return 0.81 - 110.5 / T


def water_viscosity(T):
    """Dynamic viscosity of aqueous PVP vehicle (Pa.s) as function of T (K)."""
    mu_w = 2.414e-5 * (10.0 ** (247.8 / (T - 140.0)))
    return mu_w * 1.35  # 0.3% PVP K90 viscosity multiplier


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
    alpha_1 = np.clip(C1 / C_tot, 1e-5, 1.0 - 1e-5)
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
        x1 = brentq(rubingh_res, 1e-5, 1.0 - 1e-5)
    except ValueError:
        x1 = alpha_1

    g1 = np.exp(beta * (1.0 - x1) ** 2)
    g2 = np.exp(beta * x1 ** 2)
    CMC_mix = 1.0 / (alpha_1 / (g1 * CMC1) + (1.0 - alpha_1) / (g2 * CMC2))
    CMC_ideal = 1.0 / (alpha_1 / CMC1 + (1.0 - alpha_1) / CMC2)

    return {
        "x1": x1, "CMC1": CMC1, "CMC2": CMC2,
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
    N_agg = res.x

    V_core = N_agg * v_tail_avg * 1.22
    R_c = ((3.0 * V_core) / (4.0 * np.pi)) ** (1.0 / 3.0)
    a_opt = (4.0 * np.pi * R_c ** 2) / N_agg
    solv_qual = max(0.008, 0.5 - chi_ew)

    L_corona = R_c * (
        (1.0 + (5.0 / 3.0) * (params.N_EO * (params.l_EO ** (5.0 / 3.0))) /
         (R_c * (a_opt ** (1.0 / 3.0))) * (solv_qual ** (1.0 / 3.0))) ** (3.0 / 5.0) - 1.0
    )
    delta_PVP = 1.15 * (params.C_PVP_wt / 0.30) ** 0.25
    D_h = 2.0 * (R_c + L_corona + delta_PVP)

    # Flory-Huggins + Laplace Solubilization
    phi_1 = (x1 * params.vt_1) / v_tail_avg
    HSP_core = phi_1 * params.HSP_1 + (1.0 - phi_1) * params.HSP_2
    dHSP_sq = (
        (params.HSP_CsA[0] - HSP_core[0]) ** 2 +
        0.25 * (params.HSP_CsA[1] - HSP_core[1]) ** 2 +
        0.25 * (params.HSP_CsA[2] - HSP_core[2]) ** 2
    )
    V_molar_CsA = params.v_CsA * 1e-27 * N_A
    chi_dc = (V_molar_CsA * dHSP_sq * 1e6) / (R_gas * T)
    laplace_term = (2.0 * params.sigma_core * (params.v_CsA * 1e-27)) / (R_c * 1e-9 * k_B * T)

    S_w = 0.019 * np.exp(-1800.0 * (1.0 / 298.15 - 1.0 / T))
    kappa_base = 0.285 * (v_tail_avg / params.v_CsA)
    kappa = kappa_base * np.exp(-0.55 * (chi_dc - 0.35) - 0.15 * (laplace_term - 1.2))

    C_mic_surf = max(0.0, (C1 + C2) - CMC_mix)
    S_tot_eq = S_w + kappa * C_mic_surf
    S_R = params.C_CsA / max(1e-6, S_tot_eq)

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

    h_nm_arr = np.linspace(0.4, 2.5 * L_corona_nm, 160)
    V_vdW_arr, V_osm_arr, V_el_arr, V_tot_arr = [], [], [], []

    for h_nm in h_nm_arr:
        h = h_nm * 1e-9
        t1 = (2.0 * R_c ** 2) / (h * (4.0 * R_c + h))
        t2 = (2.0 * R_c ** 2) / ((2.0 * R_c + h) ** 2)
        t3 = np.log((h * (4.0 * R_c + h)) / ((2.0 * R_c + h) ** 2))
        v_vdw = -(A_eff / (6.0 * k_B * T)) * (t1 + t2 + t3)

        v_osm = 0.0
        if h < 2.0 * L:
            v_osm = ((4.0 * np.pi * R_c * (phi_EO ** 2) * (0.5 - chi_ew) * (L ** 2)) / v_w) * (
                (h / (2.0 * L)) - 0.25 - np.log(h / L)
            )

        v_el = 0.0
        if h < L:
            v_el = 45.0 * (1.0 - h / L) ** 2.2

        V_vdW_arr.append(v_vdw)
        V_osm_arr.append(v_osm)
        V_el_arr.append(v_el)
        V_tot_arr.append(v_vdw + v_osm + v_el)

    V_tot_arr = np.array(V_tot_arr)
    V_max_kT = float(np.max(V_tot_arr))

    # Fuchs stability ratio W
    integrand = np.exp(np.clip(V_tot_arr, -30.0, 65.0)) / ((2.0 * R_h + h_nm_arr * 1e-9) ** 2)
    W_fuchs = 2.0 * R_h * np.trapz(integrand, h_nm_arr * 1e-9)
    log10_W = float(np.log10(max(1.0, W_fuchs)))

    return h_nm_arr, np.array(V_vdW_arr), np.array(V_osm_arr), np.array(V_el_arr), V_tot_arr, V_max_kT, log10_W


def simulate_shelf_life_ode(T_celsius, months_max, params):
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

    t_eval = np.linspace(0, months_max, 50)
    sol = solve_ivp(rhs, [0, months_max], [params.C1_0, params.pH_0], t_eval=t_eval)

    records = []
    for i, t_m in enumerate(sol.t):
        c1_t = sol.y[0, i]
        ph_t = sol.y[1, i]
        st_eq = solve_equilibrium_micelle(c1_t, params.C2_0, T, params)
        records.append({
            "Month": t_m,
            "Intact_HCO40_pct": (c1_t / params.C1_0) * 100.0,
            "pH": ph_t,
            "S_eq_mg_mL": st_eq["S_tot_eq_mg_mL"],
            "S_R": st_eq["S_R"],
            "D_h_nm": st_eq["D_h_nm"]
        })
    return pd.DataFrame(records)


# ==============================================================================
# POPULATION BALANCE EQUATION (PBE) COMPOUNDING & SCALE-UP ENGINE
# ==============================================================================
def run_compounding_pbe(params, V_batch_L, D_imp_m, RPM, t_feed_min, T_comp_C, N_p=1.27, N_q=0.72):
    """
    Solves the Discrete-Sectional Population Balance Equation (PBE) during
    aqueous hydration of the molten HCO-40 / Octoxynol-40 / CsA pre-concentrate.
    """
    T_K = T_comp_C + 273.15
    eq_state = solve_equilibrium_micelle(params.C1_0, params.C2_0, T_K, params)
    D_eq = eq_state["D_h_nm"]
    _, _, _, _, _, V_max_kT, log10_W = dlvo_steric_profile(eq_state["R_c_nm"], eq_state["L_corona_nm"], T_K)
    W_steric = 10.0 ** min(12.0, log10_W)

    # 1. Hydrodynamics & Mixing Time Scales
    V_m3 = V_batch_L * 1e-3
    N_rps = RPM / 60.0
    rho_liq = 1005.0  # kg/m^3
    mu_liq = water_viscosity(T_K)
    nu_kin = mu_liq / rho_liq

    P_watts = N_p * rho_liq * (N_rps ** 3) * (D_imp_m ** 5)
    epsilon = max(1e-4, P_watts / (rho_liq * V_m3))  # W/kg (m^2/s^3)
    v_tip = np.pi * D_imp_m * N_rps
    shear_rate = np.sqrt(epsilon / nu_kin)            # s^-1
    eta_K_um = ((nu_kin ** 3) / epsilon) ** 0.25 * 1e6

    tau_micro_s = 17.24 * np.sqrt(nu_kin / epsilon)
    Q_pump_m3s = max(1e-6, N_q * N_rps * (D_imp_m ** 3))
    tau_macro_s = 3.0 * V_m3 / Q_pump_m3s

    # Pre-concentrate feed rate (~1.15% of batch volume)
    V_preconc_m3 = 0.0115 * V_m3
    t_feed_s = max(10.0, t_feed_min * 60.0)
    Q_feed_m3s = V_preconc_m3 / t_feed_s

    # Local plume supersaturation & Micromixing Damkohler Number (Da_mix)
    plume_ratio = Q_feed_m3s / Q_pump_m3s
    S_R_local = eq_state["S_R"] * (1.0 + 420.0 * plume_ratio * np.sqrt(tau_macro_s / 5.0))
    tau_nuc_s = 0.12 * np.exp(2.8 / (max(0.05, np.log(max(1.02, S_R_local))) ** 2))
    Da_mix = (tau_micro_s * (1.0 + 150.0 * plume_ratio)) / tau_nuc_s

    # 2. Discretized Diameter Grid (M = 55 bins from 6 nm to 320 nm)
    M = 55
    D_grid = np.geomspace(6.0, 320.0, M)
    v_grid = (np.pi / 6.0) * (D_grid ** 3)

    # Feed globule distribution (coarse pre-concentrate droplets centered at D_feed_0)
    D_feed_0 = np.clip(165.0 * (epsilon ** (-0.12)), 75.0, 240.0)
    sigma_feed = 0.28
    feed_shape = np.exp(-0.5 * ((np.log(D_grid / D_feed_0)) / sigma_feed) ** 2) / (D_grid * sigma_feed)
    feed_shape /= np.sum(feed_shape * v_grid)  # Normalized per unit volume injected

    # Target Aniansson-Wall thermodynamic equilibrium distribution centered at D_eq
    sigma_eq = 0.11
    eq_shape = np.exp(-0.5 * ((np.log(D_grid / D_eq)) / sigma_eq) ** 2) / (D_grid * sigma_eq)
    eq_shape /= np.sum(eq_shape * v_grid)

    # Characteristic kinetic rate constants (s^-1)
    k_fiss_base = 0.018 * (shear_rate ** 0.55) * np.exp(-1200.0 * (1.0 / T_K - 1.0 / 333.15))
    Fiss_rate = k_fiss_base * np.maximum(0.0, (D_grid - D_eq) / D_eq) ** 1.6

    # Daughter redistribution matrix: upper triangular matrix conserving volume
    Daughter_Mat = np.zeros((M, M))
    for j in range(1, M):
        if D_grid[j] > D_eq * 1.08:
            # Break toward bins closer to D_eq while conserving volume v_grid[j]
            weights = np.exp(-0.5 * ((D_grid[:j] - D_eq * 1.15) / (0.35 * D_eq)) ** 2)
            if np.sum(weights) > 0:
                weights /= np.sum(weights)
                # Volume conservation factor: v_j / v_i
                Daughter_Mat[:j, j] = weights * (v_grid[j] / v_grid[:j])

    # Aniansson-Wall monomer relaxation rate toward thermodynamic equilibrium
    k_AW = 0.025 * (shear_rate ** 0.35) * np.exp(-1500.0 * (1.0 / T_K - 1.0 / 333.15))

    # Coalescence rate (only active if steric barrier W_steric is compromised at high T)
    k_coal = (1.5e-2 * shear_rate) / max(1.0, W_steric)

    # Total simulation time: feed addition + post-addition homogenization hold (15 min)
    t_total_s = t_feed_s + 900.0
    t_eval = np.linspace(0.0, t_total_s, 80)

    # State vector y = [V_vol_bins (M), M_cryst_ppm (1)]
    # Working in volume-concentration per bin V_i = v_i * N_i guarantees strict mass conservation
    y0 = np.zeros(M + 1)
    y0[:M] = 1e-6 * eq_shape * v_grid

    V_feed_vec = feed_shape * v_grid
    V_eq_vec = eq_shape * v_grid

    def pbe_ode(t, y):
        V_bins = np.maximum(0.0, y[:M])
        cryst_ppm = y[M]

        # 1. Pre-concentrate feed injection
        feed_rate = (1.0 / t_feed_s) * V_feed_vec if t <= t_feed_s else np.zeros(M)

        # 2. Turbulent/capillary fission of coarse lamellar globules
        loss_fiss = Fiss_rate * V_bins
        gain_fiss = np.zeros(M)
        for j in range(1, M):
            if Fiss_rate[j] > 0 and V_bins[j] > 0:
                w = np.exp(-0.5 * ((D_grid[:j] - D_eq * 1.12) / (0.32 * D_eq)) ** 2)
                w_sum = np.sum(w)
                if w_sum > 0:
                    gain_fiss[:j] += loss_fiss[j] * (w / w_sum)

        # 3. Aniansson-Wall monomer-exchange relaxation toward equilibrium micelle mode
        current_mass = np.sum(V_bins)
        aw_relax = k_AW * (current_mass * V_eq_vec - V_bins)

        # 4. Thermal/shear coalescence toward larger aggregates if near cloud point
        coal_shift = np.zeros(M)
        if k_coal > 1e-6:
            coal_shift[:-1] -= k_coal * V_bins[:-1]
            coal_shift[1:] += k_coal * V_bins[:-1]

        dV_dt = feed_rate - loss_fiss + gain_fiss + aw_relax + coal_shift

        # 5. Parasitic unincorporated CsA crystal growth if Da_mix > 0.15 during feed
        dcryst_dt = 0.0
        if t <= t_feed_s and Da_mix > 0.12:
            dcryst_dt = (900.0 / t_feed_s) * (1.0 - np.exp(-1.8 * (Da_mix - 0.12)))
        elif t > t_feed_s and cryst_ppm > 0 and eq_state["S_R"] < 1.0:
            # Slow micellar re-dissolution of kinetically trapped nuclei during hold
            dcryst_dt = -0.0004 * (1.0 - eq_state["S_R"]) * cryst_ppm

        return np.concatenate([dV_dt, [dcryst_dt]])

    sol = solve_ivp(pbe_ode, [0.0, t_total_s], y0, t_eval=t_eval, method="RK45")

    # Post-process moments and percentiles
    traj_records = []
    psd_snapshots = {}
    snap_indices = [int(0.15 * len(t_eval)), int(0.45 * len(t_eval)), int(0.70 * len(t_eval)), len(t_eval) - 1]

    for idx, t_s in enumerate(sol.t):
        V_b = np.maximum(0.0, sol.y[:M, idx])
        v_norm = V_b / max(1e-12, np.sum(V_b))
        cdf = np.cumsum(v_norm)

        D10 = float(np.interp(0.10, cdf, D_grid))
        D50 = float(np.interp(0.50, cdf, D_grid))
        D90 = float(np.interp(0.90, cdf, D_grid))
        mean_D = float(np.sum(v_norm * D_grid))
        var_D = float(np.sum(v_norm * (D_grid - mean_D) ** 2))
        PDI = var_D / (mean_D ** 2)

        traj_records.append({
            "Time_min": t_s / 60.0,
            "D10_nm": D10,
            "D50_nm": D50,
            "D90_nm": D90,
            "PDI": PDI,
            "Unincorporated_CsA_ppm": float(max(0.0, sol.y[M, idx]))
        })
        if idx in snap_indices:
            psd_snapshots[f"{t_s/60.0:.1f} min"] = v_norm / np.gradient( np.log10(D_grid) )

    return {
        "D_grid": D_grid,
        "traj_df": pd.DataFrame(traj_records),
        "psd_snapshots": psd_snapshots,
        "metrics": {
            "epsilon_W_kg": epsilon,
            "v_tip_m_s": v_tip,
            "shear_rate_s": shear_rate,
            "eta_K_um": eta_K_um,
            "tau_micro_ms": tau_micro_s * 1e3,
            "tau_macro_s": tau_macro_s,
            "Da_mix": Da_mix,
            "D_eq_nm": D_eq,
            "Final_D50_nm": traj_records[-1]["D50_nm"],
            "Final_D90_nm": traj_records[-1]["D90_nm"],
            "Final_PDI": traj_records[-1]["PDI"],
            "Final_Cryst_ppm": traj_records[-1]["Unincorporated_CsA_ppm"]
        }
    }


# ==============================================================================
# SIDEBAR CONTROLS
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

# Instantiate baseline system
formulation = NanomicelleFormulation(
    c_hco_wt=c_hco, c_oc40_wt=c_oc40, c_csa_wt=c_csa,
    c_pvp_wt=c_pvp, pH_0=pH_init, beta_12=beta_12_val
)
state_25C = solve_equilibrium_micelle(formulation.C1_0, formulation.C2_0, 298.15, formulation)
_, _, _, _, _, V_max_25, logW_25 = dlvo_steric_profile(state_25C["R_c_nm"], state_25C["L_corona_nm"], 298.15)
shelf_25C = simulate_shelf_life_ode(25.0, 24.0, formulation)
end_24m = shelf_25C.iloc[-1]

# ==============================================================================
# MAIN HEADER & REAL-TIME CQA SCORECARD
# ==============================================================================
st.title("First-Principles Ophthalmic Nanomicellar QbD & PBE Simulator")
st.markdown(
    "**System:** Binary Non-Ionic Mixed Micelles (HCO-40 + Octoxynol-40) encapsulating "
    "Cyclosporine A in a Povidone K90 buffered vehicle | **Physics:** Rubingh RST + "
    "Nagarajan–Ruckenstein + Laplace–Flory–Huggins + Discrete-Sectional Compounding PBE"
)

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Mixed CMC (25 °C)", f"{state_25C['CMC_mix']:.4f} mM", f"Ideal: {state_25C['CMC_ideal']:.4f} mM")
col2.metric("Equilibrium Dₕ", f"{state_25C['D_h_nm']:.2f} nm", f"N_agg = {state_25C['N_agg']:.0f}")
col3.metric("Initial S_R (t = 0)", f"{state_25C['S_R']:.3f}", "Stable (< 1.0)" if state_25C["S_R"] <= 1.0 else "Supersaturated!")
col4.metric("24-Mo S_R (25 °C)", f"{end_24m['S_R']:.3f}", f"S_eq = {end_24m['S_eq_mg_mL']:.2f} mg/mL")
col5.metric("Steric Barrier V_max", f"{V_max_25:.1f} k_B T", f"log₁₀(W) = {logW_25:.1f}")
col6.metric("24-Mo Intact HCO-40", f"{end_24m['Intact_HCO40_pct']:.1f}%", f"pH = {end_24m['pH']:.2f}")

# ==============================================================================
# NAVIGATION TABS
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "1. QbD Design Space & MODR",
    "2. Compounding PBE & Scale-Up",
    "3. Colloidal DLVO & 24-Mo Shelf-Life",
    "4. Multi-Scale Regime & Batch Comparison"
])

# ------------------------------------------------------------------------------
# TAB 1: 2D QbD DESIGN SPACE & MODR
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
        grid_res = st.select_slider("Contour Grid Resolution", options=[15, 20, 25], value=20)
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

            # Pre-compute 24-month fractional ester retention at 25 C and current pH_init
            retention_24m = end_24m["Intact_HCO40_pct"] / 100.0

            for i, oc_v in enumerate(oc_grid):
                for j, hco_v in enumerate(hco_grid):
                    tmp_f = NanomicelleFormulation(
                        c_hco_wt=hco_v, c_oc40_wt=oc_v, c_csa_wt=c_csa,
                        c_pvp_wt=c_pvp, pH_0=pH_init, beta_12=beta_12_val
                    )
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
            # Highlight S_R = 1.0 edge of failure
            fig_qbd.add_trace(
                go.Contour(
                    x=hco_grid, y=oc_grid, z=SR_24_mat,
                    contours=dict(coloring="none", start=1.0, end=1.0, size=0.1, showlabels=True),
                    line=dict(color="black", width=4, dash="dash"),
                    showscale=False, name="S_R = 1.0 Limit"
                ),
                row=1, col=1
            )
            # Operating point marker
            fig_qbd.add_trace(
                go.Scatter(
                    x=[c_hco], y=[c_oc40], mode="markers+text",
                    marker=dict(size=12, color="blue", symbol="star"),
                    text=["Current Setpoint"], textposition="top center",
                    name="Setpoint"
                ),
                row=1, col=1
            )

            fig_qbd.add_trace(
                go.Contour(
                    x=hco_grid, y=oc_grid, z=Dh_mat,
                    colorscale="Viridis",
                    contours=dict(showlabels=True),
                    colorbar=dict(title="Dₕ (nm)", x=1.0)
                ),
                row=1, col=2
            )
            fig_qbd.add_trace(
                go.Scatter(
                    x=[c_hco], y=[c_oc40], mode="markers",
                    marker=dict(size=12, color="red", symbol="star"),
                    showlegend=False
                ),
                row=1, col=2
            )
            fig_qbd.update_xaxes(title_text="HCO-40 Concentration (% w/v)", row=1, col=1)
            fig_qbd.update_yaxes(title_text="Octoxynol-40 Concentration (% w/v)", row=1, col=1)
            fig_qbd.update_xaxes(title_text="HCO-40 Concentration (% w/v)", row=1, col=2)
            fig_qbd.update_yaxes(title_text="Octoxynol-40 Concentration (% w/v)", row=1, col=2)
            fig_qbd.update_layout(height=480, margin=dict(l=20, r=20, t=40, b=20))
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
                    line=dict(color="black", width=4, dash="dash"),
                    showscale=False
                ),
                row=1, col=1
            )
            fig_ph.add_trace(
                go.Contour(
                    x=pH_grid, y=T_grid, z=HCO_ret_mat,
                    colorscale="Blues",
                    contours=dict(showlabels=True),
                    colorbar=dict(title="Intact HCO-40 (%)", x=1.0)
                ),
                row=1, col=2
            )
            fig_ph.update_xaxes(title_text="Initial Formulation pH", row=1, col=1)
            fig_ph.update_yaxes(title_text="Storage Temperature (°C)", row=1, col=1)
            fig_ph.update_xaxes(title_text="Initial Formulation pH", row=1, col=2)
            fig_ph.update_yaxes(title_text="Storage Temperature (°C)", row=1, col=2)
            fig_ph.update_layout(height=480, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_ph, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 2: COMPOUNDING PBE & SCALE-UP SIMULATOR
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("Population Balance Equation (PBE) — High-Shear Aqueous Hydration & Micellar Self-Assembly")
    pbe_res = run_compounding_pbe(formulation, V_batch, D_imp, rpm_val, t_feed, T_comp)
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
            go.Scatter(
                x=pbe_res["D_grid"], y=psd_curve,
                mode="lines", name=f"PSD @ {label}",
                line=dict(width=2.5)
            ),
            row=1, col=1
        )
    fig_pbe.add_vline(
        x=m["D_eq_nm"], line_dash="dash", line_color="green",
        annotation_text=f"Thermodynamic Dₕ* ({m['D_eq_nm']:.1f} nm)",
        row=1, col=1
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
            name="Unincorporated CsA Crystals (ppm)",
            line=dict(color="#9467bd", width=2.5, dash="dashdot")
        ),
        row=1, col=2, secondary_y=True
    )

    fig_pbe.update_xaxes(title_text="Aggregate Hydrodynamic Diameter Dₕ (nm) [Log Scale]", type="log", row=1, col=1)
    fig_pbe.update_yaxes(title_text="Volume Density dV / d(log₁₀ D)", row=1, col=1)
    fig_pbe.update_xaxes(title_text="Compounding Batch Time (min)", row=1, col=2)
    fig_pbe.update_yaxes(title_text="Percentile Diameter (nm)", secondary_y=False, row=1, col=2)
    fig_pbe.update_yaxes(title_text="Unincorporated Crystalline CsA (ppm)", secondary_y=True, row=1, col=2)
    fig_pbe.update_layout(height=480, margin=dict(l=20, r=20, t=40, b=20))
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
        df_ich = simulate_shelf_life_ode(T_ich, 24.0, formulation)
        fig_stab.add_trace(
            go.Scatter(
                x=df_ich["Month"], y=df_ich["S_R"],
                mode="lines", name=f"ICH {T_ich:.0f} °C",
                line=dict(color=col_c, width=2.8)
            ),
            row=1, col=2
        )

    fig_stab.add_hline(
        y=1.0, line_dash="dash", line_color="black",
        annotation_text="Precipitation Threshold (S_R = 1.0)",
        row=1, col=2
    )
    fig_stab.update_xaxes(title_text="Surface-to-Surface Separation h (nm)", row=1, col=1)
    fig_stab.update_yaxes(title_text="Interaction Potential V(h) / k_B T", range=[-10, 80], row=1, col=1)
    fig_stab.update_xaxes(title_text="Storage Time (Months)", row=1, col=2)
    fig_stab.update_yaxes(title_text="Supersaturation Ratio S_R = C_nominal / S_eq(t)", row=1, col=2)
    fig_stab.update_layout(height=460, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_stab, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 4: MULTI-SCALE REGIME & BATCH COMPARISON TABLE
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
            # N ~ (epsilon * V / D^5)^(1/3)
            N_rps = ((base_eps * (V_s * 1e-3)) / (1.27 * (D_s ** 5))) ** (1.0 / 3.0)
        elif "Constant Impeller Tip" in scale_criterion:
            N_rps = base_tip / (np.pi * D_s)
        else:
            # Scale RPM and feed time to keep Da_mix constant
            N_rps = ((base_eps * (V_s * 1e-3)) / (1.27 * (D_s ** 5))) ** (1.0 / 3.0) * 1.12

        rpm_s = N_rps * 60.0
        res_s = run_compounding_pbe(formulation, V_s, D_s, rpm_s, t_feed, T_comp)
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
