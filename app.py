import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.integrate import quad, solve_ivp

# ==============================================================================
# PHYSICAL CONSTANTS & FORMULATION INPUTS (CEQUA 0.09% w/v SYSTEM)
# ==============================================================================
k_B = 1.380649e-23       # Boltzmann constant (J/K)
N_A = 6.02214076e23      # Avogadro's number (1/mol)
R_gas = 8.314462         # Universal gas constant (J/mol*K)

class NanomicelleSystemParams:
    def __init__(self):
        # Nominal Composition
        self.C_HCO_wt = 1.00       # % w/v (10 mg/mL)
        self.C_OC40_wt = 0.05      # % w/v (0.5 mg/mL)
        self.C_CsA_wt = 0.09       # % w/v (0.9 mg/mL)
        self.C_PVP_wt = 0.30       # % w/v (3.0 mg/mL)
        self.I_ionic = 0.125       # Molar ionic strength (Phosphate buffer + NaCl)
        self.pH_0 = 6.80           # Initial formulation pH
        self.beta_buf = 0.035      # Buffer capacity (mol/L/pH unit)

        # Molecular Weights (g/mol)
        self.MW_1 = 2500.0         # HCO-40
        self.MW_2 = 1966.0         # Octoxynol-40
        self.MW_CsA = 1202.61      # Cyclosporine A

        # Molar Concentrations (mM)
        self.C1_0 = (self.C_HCO_wt * 10.0 / self.MW_1) * 1e3   # ~4.000 mM
        self.C2_0 = (self.C_OC40_wt * 10.0 / self.MW_2) * 1e3  # ~0.254 mM
        self.C_CsA = (self.C_CsA_wt * 10.0 / self.MW_CsA) * 1e3 # ~0.748 mM

        # Surfactant Thermodynamic Parameters
        self.CMC1_ref = 0.014      # mM at 298.15 K in pure water
        self.CMC2_ref = 0.160      # mM at 298.15 K in pure water
        self.dH_mic1 = -12500.0    # J/mol
        self.dH_mic2 = -9800.0     # J/mol
        self.ks_1 = 1.15           # Setschenow constant (M^-1)
        self.ks_2 = 0.95           # Setschenow constant (M^-1)
        self.kp_1 = 0.18           # PVP interaction factor ((% w/v)^-1)
        self.kp_2 = 0.15           # PVP interaction factor ((% w/v)^-1)
        self.beta_12 = -2.45       # Rubingh binary interaction parameter

        # Molecular Geometry (nm and nm^3)
        self.vt_1 = 1.48           # HCO-40 hydrophobic tail volume (nm^3)
        self.vt_2 = 0.36           # OC-40 hydrophobic tail volume (nm^3)
        self.v_CsA = 1.74          # CsA molecular volume (nm^3)
        self.a0_shield = 0.42      # Shielded core area per monomer (nm^2)
        self.N_EO = 40.0           # Ethylene oxide units per chain
        self.l_EO = 0.36           # Monomer Kuhn length of EO (nm)
        self.v_EO = 0.065          # Volume of single EO unit (nm^3)
        self.sigma_core = 0.030    # Core-water interfacial tension (J/m^2 = 30 mN/m)

        # Hansen Solubility Parameters (MPa^0.5) [d_d, d_p, d_h]
        self.HSP_1 = np.array([17.2, 4.5, 6.8])    # HCO-40 core (12-hydroxystearate)
        self.HSP_2 = np.array([18.4, 3.2, 4.1])    # OC-40 core (octylphenyl)
        self.HSP_CsA = np.array([17.8, 5.1, 7.4])  # Cyclosporine A

        # Ester Hydrolysis Kinetics of HCO-40 (Reference T = 298.15 K)
        self.Ea_hyd = 78500.0      # Activation energy (J/mol)
        self.k_H = 1.8e-2          # Acid catalyzed rate (month^-1 M^-1)
        self.k_0 = 4.5e-4          # Neutral water rate (month^-1)
        self.k_OH = 8.2e2          # Base catalyzed rate (month^-1 M^-1)

# ==============================================================================
# MODULE 1: RUBINGH–SETSCHENOW MIXED MICELLIZATION THERMODYNAMICS
# ==============================================================================
def solve_mixed_cmc(C1, C2, T, params):
    T_ref = 298.15
    # 1. Pure surfactant CMCs corrected for T, Ionic Strength, and PVP
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

    C_tot = C1 + C2
    alpha_1 = C1 / C_tot
    beta = params.beta_12

    # 2. Solve Rubingh's equation for micellar mole fraction x1
    def rubingh_residual(x1):
        gamma1 = np.exp(beta * (1.0 - x1)**2)
        gamma2 = np.exp(beta * x1**2)
        CMC_mix = 1.0 / (alpha_1 / (gamma1 * CMC1) + (1.0 - alpha_1) / (gamma2 * CMC2))
        arg1 = (alpha_1 * CMC_mix) / (x1 * CMC1)
        arg2 = ((1.0 - alpha_1) * CMC_mix) / ((1.0 - x1) * CMC2)
        if arg1 <= 0 or arg2 <= 0:
            return 0.0
        return (x1**2) * np.log(arg1) - ((1.0 - x1)**2) * np.log(arg2)

    try:
        x1 = brentq(rubingh_residual, 1e-5, 1.0 - 1e-5)
    except ValueError:
        x1 = alpha_1

    gamma1 = np.exp(beta * (1.0 - x1)**2)
    gamma2 = np.exp(beta * x1**2)
    CMC_mix = 1.0 / (alpha_1 / (gamma1 * CMC1) + (1.0 - alpha_1) / (gamma2 * CMC2))
    CMC_ideal = 1.0 / (alpha_1 / CMC1 + (1.0 - alpha_1) / CMC2)

    return {
        "x1": x1, "CMC1": CMC1, "CMC2": CMC2,
        "CMC_mix": CMC_mix, "CMC_ideal": CMC_ideal,
        "gamma1": gamma1, "gamma2": gamma2
    }

# ==============================================================================
# MODULE 2 & 3: NAGARAJAN-RUCKENSTEIN GEOMETRY & DRUG SOLUBILIZATION
# ==============================================================================
def chi_EO_water(T):
    # Temperature-dependent Flory-Huggins parameter for PEO in water
    # Approaches 0.5 at ~358 K (85 C cloud point regime)
    return 0.81 - 110.5 / T

def solve_micelle_structure_and_solubility(C1, C2, T, params):
    mic_thermo = solve_mixed_cmc(C1, C2, T, params)
    x1 = mic_thermo["x1"]
    x2 = 1.0 - x1
    CMC_mix = mic_thermo["CMC_mix"]

    v_tail_avg = x1 * params.vt_1 + x2 * params.vt_2
    chi_ew = chi_EO_water(T)
    sigma_kT = (params.sigma_core * 1e-18) / (k_B * T) # in nm^-2

    # Minimize free energy per monomer with respect to N_agg
    def delta_mu_per_kT(N_agg):
        if N_agg <= 10:
            return 1e6
        # Core volume with approximate drug loading fraction (~0.18 of core)
        V_core = N_agg * v_tail_avg * 1.22
        R_c = ((3.0 * V_core) / (4.0 * np.pi))**(1.0 / 3.0)
        area_per_monomer = (4.0 * np.pi * R_c**2) / N_agg

        # 1. Core-water interfacial energy
        mu_int = sigma_kT * max(0.0, area_per_monomer - params.a0_shield)

        # 2. Tail stretching elastic penalty (Semenov)
        L_tail_max = x1 * 2.1 + x2 * 1.1 # nm
        mu_def = (3.0 * np.pi**2 / 80.0) * (R_c**2 / (L_tail_max**2)) * 2.8

        # 3. PEO-40 Corona Brush Crowding (Daoud-Cotton / de Gennes)
        solvent_quality = max(0.02, 0.5 - chi_ew)
        mu_steric = 1.85 * params.N_EO * (params.v_EO / (area_per_monomer * R_c))**(0.85) * solvent_quality

        return mu_int + mu_def + mu_steric

    res = minimize_scalar(delta_mu_per_kT, bounds=(30, 400), method='bounded')
    N_agg_opt = res.x

    # Compute Core & Corona Radii at N_agg_opt
    V_core = N_agg_opt * v_tail_avg * 1.22
    R_c = ((3.0 * V_core) / (4.0 * np.pi))**(1.0 / 3.0)
    a_opt = (4.0 * np.pi * R_c**2) / N_agg_opt
    solvent_quality = max(0.01, 0.5 - chi_ew)

    # Daoud-Cotton Corona Thickness + adsorbed PVP layer
    L_corona = R_c * ((1.0 + (5.0 / 3.0) * (params.N_EO * (params.l_EO**(5.0/3.0))) /
                      (R_c * (a_opt**(1.0/3.0))) * (solvent_quality**(1.0/3.0)))**(3.0/5.0) - 1.0)
    delta_PVP = 1.15 # nm hydrodynamic shell from Povidone K90
    R_h = R_c + L_corona + delta_PVP
    D_h = 2.0 * R_h

    # Module 3: Flory-Huggins + Laplace Pressure Drug Solubilization
    phi_1 = (x1 * params.vt_1) / v_tail_avg
    phi_2 = 1.0 - phi_1
    HSP_core = phi_1 * params.HSP_1 + phi_2 * params.HSP_2
    dHSP_sq = ((params.HSP_CsA[0] - HSP_core[0])**2 +
               0.25 * (params.HSP_CsA[1] - HSP_core[1])**2 +
               0.25 * (params.HSP_CsA[2] - HSP_core[2])**2)

    # Convert v_CsA from nm^3/molecule to m^3/mol for RT
    V_molar_CsA = params.v_CsA * 1e-27 * N_A # m^3/mol
    chi_dc = (V_molar_CsA * dHSP_sq * 1e6) / (R_gas * T)

    # Laplace pressure penalty inside curved core
    laplace_term = (2.0 * params.sigma_core * (params.v_CsA * 1e-27)) / (R_c * 1e-9 * k_B * T)

    # Intrinsic aqueous solubility of CsA (retrograde solubility with T) in mM
    S_w = 0.019 * np.exp(-1800.0 * (1.0 / 298.15 - 1.0 / T))

    # Molar solubilization capacity kappa (mol CsA / mol micellar surfactant)
    # Derived from core volume capacity and Flory-Huggins + Laplace chemical potential
    kappa_base = 0.285 * (v_tail_avg / params.v_CsA)
    kappa = kappa_base * np.exp(-0.55 * (chi_dc - 0.35) - 0.15 * (laplace_term - 1.2))

    C_surf_tot = C1 + C2
    C_mic_surf = max(0.0, C_surf_tot - CMC_mix)
    S_tot_eq = S_w + kappa * C_mic_surf
    S_R = params.C_CsA / S_tot_eq

    return {
        **mic_thermo,
        "N_agg": N_agg_opt,
        "R_c_nm": R_c,
        "L_corona_nm": L_corona,
        "D_h_nm": D_h,
        "chi_dc": chi_dc,
        "kappa": kappa,
        "S_tot_eq_mM": S_tot_eq,
        "S_tot_eq_mg_mL": S_tot_eq * params.MW_CsA / 1e3,
        "Supersaturation_Ratio": S_R
    }

# ==============================================================================
# MODULE 4A: EXTENDED DLVO + BRUSH STERIC POTENTIAL & FUCHS STABILITY RATIO (W)
# ==============================================================================
def compute_colloidal_stability(R_c_nm, L_corona_nm, T, params):
    R_c = R_c_nm * 1e-9
    L = L_corona_nm * 1e-9
    R_h = R_c + L
    A_eff = 0.85e-20 # Effective Hamaker constant for hydrocarbon/PEO in water (J)
    chi_ew = chi_EO_water(T)
    phi_EO = 0.16    # Average volume fraction of EO segments in brush
    v_w = 3.0e-29    # Molecular volume of water (m^3)

    def V_tot_over_kT(h_nm):
        h = h_nm * 1e-9
        # 1. van der Waals attraction
        term1 = (2.0 * R_c**2) / (h * (4.0 * R_c + h))
        term2 = (2.0 * R_c**2) / ((2.0 * R_c + h)**2)
        term3 = np.log((h * (4.0 * R_c + h)) / ((2.0 * R_c + h)**2))
        V_vdW = -(A_eff / (6.0 * k_B * T)) * (term1 + term2 + term3)

        # 2. Vincent-Napper Osmotic Steric Interaction (h < 2L)
        V_osm = 0.0
        if h < 2.0 * L:
            V_osm = ((4.0 * np.pi * R_c * (phi_EO**2) * (0.5 - chi_ew) * (L**2)) / v_w) * (
                (h / (2.0 * L)) - 0.25 - np.log(h / L)
            )

        # 3. Elastic compression (h < L)
        V_el = 0.0
        if h < L:
            u = h / L
            V_el = 45.0 * (1.0 - u)**2.2

        return V_vdW + V_osm + V_el

    # Peak energy barrier in kT
    h_grid = np.linspace(0.5, 2.0 * L_corona_nm, 200)
    V_grid = [V_tot_over_kT(h_val) for h_val in h_grid]
    V_max_kT = max(V_grid)

    # Fuchs Stability Ratio W integration
    def integrand(h_nm):
        h = h_nm * 1e-9
        V_kT = min(60.0, V_tot_over_kT(h_nm)) # Cap to avoid float overflow
        return np.exp(V_kT) / ((2.0 * R_h + h)**2)

    integral_val, _ = quad(integrand, 0.4, 3.0 * L_corona_nm, limit=100)
    W_fuchs = 2.0 * R_h * (integral_val * 1e-9)
    return V_max_kT, np.log10(max(1.0, W_fuchs))

# ==============================================================================
# MODULE 4B: 24-MONTH SHELF-LIFE ODE (ESTER HYDROLYSIS & SUPERSATURATION)
# ==============================================================================
def simulate_shelf_life(T_celsius, months, params):
    T = T_celsius + 273.15
    T_ref = 298.15
    pKw = 14.0 - 0.032 * (T_celsius - 25.0)

    def ode_rhs(t, y):
        C1, pH = y
        arrhenius = np.exp(-(params.Ea_hyd / R_gas) * (1.0 / T - 1.0 / T_ref))
        k_hyd = arrhenius * (
            params.k_H * (10.0**(-pH)) +
            params.k_0 +
            params.k_OH * (10.0**(pH - pKw))
        )
        dC1_dt = -k_hyd * C1
        # 1.5 moles of ester cleaved per mole of HCO-40 destabilized (mM -> M conversion = 1e-3)
        dpH_dt = 1.5 * (dC1_dt * 1e-3) / params.beta_buf
        return [dC1_dt, dpH_dt]

    sol = solve_ivp(ode_rhs, [0, months], [params.C1_0, params.pH_0], t_eval=[0, 6, 12, 18, 24])
    results = []
    for idx, t_m in enumerate(sol.t):
        C1_t = sol.y[0, idx]
        pH_t = sol.y[1, idx]
        state = solve_micelle_structure_and_solubility(C1_t, params.C2_0, T, params)
        results.append({
            "time_mo": t_m, "C1_mM": C1_t, "pH": pH_t,
            "S_eq_mg_mL": state["S_tot_eq_mg_mL"],
            "S_R": state["Supersaturation_Ratio"],
            "D_h_nm": state["D_h_nm"]
        })
    return results

if __name__ == "__main__":
    p = NanomicelleSystemParams()
    state_25C = solve_micelle_structure_and_solubility(p.C1_0, p.C2_0, 298.15, p)
    V_max_25, logW_25 = compute_colloidal_stability(state_25C["R_c_nm"], state_25C["L_corona_nm"], 298.15, p)

    print("=== T = 25 C INITIAL EQUILIBRIUM PREDICTIONS ===")
    print(f"Mixed CMC (Rubingh)   : {state_25C['CMC_mix']:.5f} mM (vs Ideal: {state_25C['CMC_ideal']:.5f} mM)")
    print(f"Micellar HCO-40 (x1)  : {state_25C['x1']:.4f} (Bulk alpha_1 = {p.C1_0/(p.C1_0+p.C2_0):.4f})")
    print(f"Aggregation Number    : {state_25C['N_agg']:.1f} monomers/micelle")
    print(f"Core Radius (R_c)     : {state_25C['R_c_nm']:.2f} nm | Corona (L) : {state_25C['L_corona_nm']:.2f} nm")
    print(f"Hydrodynamic Dia (Dh) : {state_25C['D_h_nm']:.2f} nm")
    print(f"Equilibrium CsA Solub.: {state_25C['S_tot_eq_mg_mL']:.3f} mg/mL (Nominal = 0.900 mg/mL)")
    print(f"Supersaturation Ratio : {state_25C['Supersaturation_Ratio']:.3f} (< 1.0 = Thermodynamically Stable)")
    print(f"Steric Barrier V_max  : {V_max_25:.1f} k_B*T | log10(Fuchs W) : {logW_25:.1f}\n")

    print("=== 24-MONTH SHELF-LIFE STABILITY COMPARISON (25 C vs 40 C Accelerated) ===")
    for temp in [25.0, 40.0]:
        traj = simulate_shelf_life(temp, 24, p)
        end = traj[-1]
        print(f"[{temp:.0f} C @ 24 Mo] Intact HCO-40: {end['C1_mM']/p.C1_0*100:.1f}% | "
              f"pH: {end['pH']:.2f} | S_eq: {end['S_eq_mg_mL']:.3f} mg/mL | S_R: {end['S_R']:.3f}")
