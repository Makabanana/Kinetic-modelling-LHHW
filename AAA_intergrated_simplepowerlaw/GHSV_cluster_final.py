from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution


# ============================================================
# 0. Global settings
# ============================================================

R = 8.314
EPS = 1e-30

# If the second row in Excel is a unit row, keep True.
# If not, change to False.
SKIP_SECOND_ROW = True

# Rough mode is much faster. For final refined fitting, set FAST_MODE = False.
FAST_MODE = True

# Fit only one GHSV group.
# Keep 12000.0 if you only want GHSV = 12000.
# Change to None if you want to fit all GHSV groups again.
TARGET_GHSV = 12000.0

# Data file name
EXCEL_FILE_NAME = "full data.xlsx"

# Current script folder
BASE_DIR = Path(__file__).resolve().parent

# Output folder
if TARGET_GHSV is None:
    OUT_DIR = BASE_DIR / "R_GHSV_cluster_integral_improved"
else:
    OUT_DIR = BASE_DIR / f"R_GHSV_{int(TARGET_GHSV)}_integral_improved"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Experiment log
EXPERIMENT_LOG_PATH = OUT_DIR / "experiment_log_integral_powerlaw.csv"

# Catalyst loading
# 0.2 g = 0.0002 kg
W_CAT_KG = 0.0002

# Standard molar volume, L/mol
STANDARD_MOLAR_VOLUME_L_PER_MOL = 22.414

# If your Excel has no flow column, the code will use this map.
# Please check these values against your real experimental flow rates.
# Unit: mL/min at standard condition.
FLOW_ML_MIN_BY_GHSV = {
    4000: 31.5,
    8000: 42.0,
    12000: 42.0,
}

if FAST_MODE:
    RK4_STEPS = 20
    DE_POPSIZE = 8
    DE_MAXITER = 120
    DE_TOL = 1e-4
    DE_POLISH = False
else:
    RK4_STEPS = 40
    DE_POPSIZE = 15
    DE_MAXITER = 1000
    DE_TOL = 1e-6
    DE_POLISH = True

DE_SEED = 42
DE_MUTATION = (0.5, 1.0)
DE_RECOMBINATION = 0.7
DE_WORKERS = 1
DE_UPDATING = "immediate"

PARAMETER_NAMES = [
    "ln_k1_ref",
    "E1_J_per_mol",
    "n1_CO2",
    "n1_H2",
    "ln_k2_ref",
    "E2_J_per_mol",
    "n2_CO2",
    "n2_H2",
]

# Parameter order:
# ln_k1_ref, E1, n1_CO2, n1_H2,
# ln_k2_ref, E2, n2_CO2, n2_H2
BOUNDS = [
    (-30.0, 30.0),       # ln_k1_ref
    (0.0, 150000.0),     # E1, J/mol
    (-2.0, 5.0),         # n1_CO2
    (-2.0, 5.0),         # n1_H2
    (-30.0, 30.0),       # ln_k2_ref
    (0.0, 150000.0),     # E2, J/mol
    (-2.0, 5.0),         # n2_CO2
    (-2.0, 5.0),         # n2_H2
]

BOUND_WARNING_FRACTION = 0.02


# ============================================================
# 1. Find and load data
# ============================================================

def find_data_file():
    candidate_paths = [
        BASE_DIR / EXCEL_FILE_NAME,
        BASE_DIR / "data" / EXCEL_FILE_NAME,
        BASE_DIR.parent / EXCEL_FILE_NAME,
        BASE_DIR.parent / "data" / EXCEL_FILE_NAME,
        Path.cwd() / EXCEL_FILE_NAME,
        Path.cwd() / "data" / EXCEL_FILE_NAME,
    ]

    for path in candidate_paths:
        if path.exists():
            return path

    message = f"Cannot find {EXCEL_FILE_NAME}. The script searched:\n"
    for path in candidate_paths:
        message += f"  {path}\n"
    raise FileNotFoundError(message)


EXCEL_FILE = find_data_file()


def normalise_columns(df):
    df.columns = [str(col).strip().replace("\n", " ") for col in df.columns]

    rename_map = {
        "H/C RATIO": "HC",
        "H/C": "HC",
        "p": "p_MPa",
        "P": "p_MPa",
        "Pressure": "p_MPa",
        "pressure": "p_MPa",
        "r CH3OH": "rMeOH",
        "rCH3OH": "rMeOH",
        "r MeOH": "rMeOH",
        "r CO": "rCO",
        "rCO": "rCO",
        "r CO2": "rCO2",
        "flow_mL_min": "flow_mL_min",
        "flow mL/min": "flow_mL_min",
        "Flow mL/min": "flow_mL_min",
        "Flow_mL_min": "flow_mL_min",
        "standard_flow_mL_min": "flow_mL_min",
        "standard flow mL/min": "flow_mL_min",
    }

    df = df.rename(columns=rename_map)
    return df


def load_data():
    if SKIP_SECOND_ROW:
        df = pd.read_excel(EXCEL_FILE, skiprows=[1])
    else:
        df = pd.read_excel(EXCEL_FILE)

    df = normalise_columns(df)

    print("实际列名如下：")
    print(df.columns.tolist())

    required_cols = [
        "HC",
        "p_MPa",
        "GHSV",
        "T",
        "rMeOH",
        "rCO",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"缺少列: {missing_cols}")

    optional_fugacity_cols = ["fCO2", "fH2", "fCH3OH", "fH2O", "fCO"]
    numeric_cols = required_cols + [col for col in optional_fugacity_cols if col in df.columns]

    if "flow_mL_min" in df.columns:
        numeric_cols.append("flow_mL_min")

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required_cols).copy()

    if "flow_mL_min" not in df.columns:
        df["flow_mL_min"] = df["GHSV"].round().astype(int).map(FLOW_ML_MIN_BY_GHSV)

    if df["flow_mL_min"].isna().any():
        bad_ghsv = sorted(df.loc[df["flow_mL_min"].isna(), "GHSV"].dropna().unique())
        raise ValueError(
            "Some rows do not have flow_mL_min and cannot be mapped from GHSV. "
            f"Please update FLOW_ML_MIN_BY_GHSV. Missing GHSV values: {bad_ghsv}"
        )

    df = df.dropna(subset=["flow_mL_min"]).copy()

    print(f"总数据点数: {len(df)}")
    print("使用的流量列: flow_mL_min")
    print(df[["GHSV", "flow_mL_min"]].drop_duplicates().sort_values("GHSV"))

    return df


# ============================================================
# 2. Equilibrium constants
# ============================================================

def calculate_equilibrium_constants(T):
    """
    K_f1:
        CO2 + 3H2 <-> CH3OH + H2O

    K_f2:
        CO2 + H2 <-> CO + H2O
    """
    T = np.asarray(T, dtype=float)

    K_f1 = np.exp(
        1.6654 + 4553.34 / T - 2.72613 * np.log(T)
        - 1.422914e-2 * T + 0.172060e-4 * T**2
        - 1.106294e-8 * T**3 + 0.319698e-11 * T**4
    ) * (0.101325) ** (-2)

    K_f2 = np.exp(
        -11.4998 - 4649.92 / T + 3.2066 * np.log(T)
        - 0.0107251 * T + 0.697955e-5 * T**2
        - 0.336848e-8 * T**3 + 0.811184e-12 * T**4
    )

    return K_f1, K_f2


# ============================================================
# 3. Kinetic parameter conversion
# ============================================================

def calc_k_from_lnk_ref(ln_k_ref, E_J_per_mol, T, Tave):
    """
    Tave reparameterisation:
        ln k = ln_k_ref - E / R * (1 / T - 1 / Tave)
    """
    ln_k = ln_k_ref - E_J_per_mol / R * (1.0 / T - 1.0 / Tave)
    return np.exp(np.clip(ln_k, -700, 700))


def unpack_params(par):
    (
        ln_k1_ref,
        E1_J_per_mol,
        n1_CO2,
        n1_H2,
        ln_k2_ref,
        E2_J_per_mol,
        n2_CO2,
        n2_H2,
    ) = par

    return {
        "ln_k1_ref": ln_k1_ref,
        "k1_ref_at_Tave": np.exp(np.clip(ln_k1_ref, -700, 700)),
        "E1_J_per_mol": E1_J_per_mol,
        "E1_kJ_per_mol": E1_J_per_mol / 1000.0,
        "n1_CO2": n1_CO2,
        "n1_H2": n1_H2,
        "ln_k2_ref": ln_k2_ref,
        "k2_ref_at_Tave": np.exp(np.clip(ln_k2_ref, -700, 700)),
        "E2_J_per_mol": E2_J_per_mol,
        "E2_kJ_per_mol": E2_J_per_mol / 1000.0,
        "n2_CO2": n2_CO2,
        "n2_H2": n2_H2,
    }


# ============================================================
# 4. Local rate expressions
# ============================================================

def calculate_local_powerlaw_rates(model_kind, par, T, Tave, fCO2, fH2, fCH3OH, fH2O, fCO):
    """
    Local rates inside the catalyst bed.

    model_kind:
        simple_powerlaw_integral
        powerlaw_with_1_minus_beta_integral
    """
    (
        ln_k1_ref,
        E1_J_per_mol,
        n1_CO2,
        n1_H2,
        ln_k2_ref,
        E2_J_per_mol,
        n2_CO2,
        n2_H2,
    ) = par

    T = float(T)
    fCO2 = max(float(fCO2), EPS)
    fH2 = max(float(fH2), EPS)
    fCH3OH = max(float(fCH3OH), 0.0)
    fH2O = max(float(fH2O), 0.0)
    fCO = max(float(fCO), 0.0)

    k1 = calc_k_from_lnk_ref(ln_k1_ref, E1_J_per_mol, T, Tave)
    k2 = calc_k_from_lnk_ref(ln_k2_ref, E2_J_per_mol, T, Tave)

    r1 = k1 * (fCO2 ** n1_CO2) * (fH2 ** n1_H2)
    r2 = k2 * (fCO2 ** n2_CO2) * (fH2 ** n2_H2)

    if model_kind == "powerlaw_with_1_minus_beta_integral":
        K_f1, K_f2 = calculate_equilibrium_constants(T)
        K_f1 = max(float(np.asarray(K_f1)), EPS)
        K_f2 = max(float(np.asarray(K_f2)), EPS)

        beta1 = (fCH3OH * fH2O) / max(K_f1 * fCO2 * (fH2 ** 3.0), EPS)
        beta2 = (fCO * fH2O) / max(K_f2 * fCO2 * fH2, EPS)

        r1 = r1 * (1.0 - beta1)
        r2 = r2 * (1.0 - beta2)

    elif model_kind == "simple_powerlaw_integral":
        pass
    else:
        raise ValueError(f"Unknown model_kind: {model_kind}")

    if not np.isfinite(r1) or not np.isfinite(r2):
        raise FloatingPointError("Non-finite local rate.")

    return float(r1), float(r2)


# ============================================================
# 5. Inlet flow and PFR derivatives
# ============================================================

def calculate_total_inlet_flow_mol_s(flow_mL_min):
    """
    Convert standard volumetric flow rate to mol/s.
    Unit of flow_mL_min: mL/min.
    """
    flow_L_min = float(flow_mL_min) * 1e-3
    return flow_L_min / STANDARD_MOLAR_VOLUME_L_PER_MOL / 60.0


def calculate_inlet_flows_from_hc_and_flow(HC, flow_mL_min):
    """
    Build inlet molar flows from H2/CO2 ratio and total inlet flow.

    flow order:
        [CO2, H2, CH3OH, H2O, CO]
    """
    HC = float(HC)
    F_total_in = calculate_total_inlet_flow_mol_s(flow_mL_min)

    y_co2_in = 1.0 / (1.0 + HC)
    y_h2_in = HC / (1.0 + HC)

    flows_in = np.array([
        y_co2_in * F_total_in,
        y_h2_in * F_total_in,
        0.0,
        0.0,
        0.0,
    ], dtype=float)

    return flows_in


def calculate_pfr_derivatives(W, flows, par, T, p_MPa, Tave, model_kind):
    """
    Calculate dF/dW.

    flow order:
        [CO2, H2, CH3OH, H2O, CO]
    """
    flows = np.maximum(flows, 0.0)

    F_total = max(float(np.sum(flows)), EPS)
    y = flows / F_total

    y_CO2, y_H2, y_CH3OH, y_H2O, y_CO = y

    # Ideal fugacity approximation.
    # If your p is not in MPa, convert it before using this function.
    fCO2 = y_CO2 * p_MPa
    fH2 = y_H2 * p_MPa
    fCH3OH = y_CH3OH * p_MPa
    fH2O = y_H2O * p_MPa
    fCO = y_CO * p_MPa

    r1, r2 = calculate_local_powerlaw_rates(
        model_kind=model_kind,
        par=par,
        T=T,
        Tave=Tave,
        fCO2=fCO2,
        fH2=fH2,
        fCH3OH=fCH3OH,
        fH2O=fH2O,
        fCO=fCO,
    )

    dF_dW = np.array([
        -(r1 + r2),
        -(3.0 * r1 + r2),
        r1,
        r1 + r2,
        r2,
    ], dtype=float)

    return dF_dW


# ============================================================
# 6. Integrate one experiment and all experiments
# ============================================================

def integrate_one_experiment(row, par, Tave, model_kind):
    """
    PFR integration for one experimental condition.

    Integration range:
        W = 0 to W = W_CAT_KG
    """
    inlet = calculate_inlet_flows_from_hc_and_flow(
        HC=row["HC"],
        flow_mL_min=row["flow_mL_min"],
    )

    T = float(row["T"])
    p_MPa = float(row["p_MPa"])

    flows = inlet.copy()

    h = W_CAT_KG / RK4_STEPS
    W = 0.0

    for _ in range(RK4_STEPS):
        k1 = calculate_pfr_derivatives(W, flows, par, T, p_MPa, Tave, model_kind)

        k2 = calculate_pfr_derivatives(
            W + 0.5 * h,
            flows + 0.5 * h * k1,
            par,
            T,
            p_MPa,
            Tave,
            model_kind,
        )

        k3 = calculate_pfr_derivatives(
            W + 0.5 * h,
            flows + 0.5 * h * k2,
            par,
            T,
            p_MPa,
            Tave,
            model_kind,
        )

        k4 = calculate_pfr_derivatives(
            W + h,
            flows + h * k3,
            par,
            T,
            p_MPa,
            Tave,
            model_kind,
        )

        flows = flows + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        flows = np.maximum(flows, 0.0)

        if not np.all(np.isfinite(flows)):
            raise FloatingPointError("Non-finite molar flow during PFR integration.")

        W += h

    outlet = flows

    rMeOH_pred = (outlet[2] - inlet[2]) / W_CAT_KG
    rCO_pred = (outlet[4] - inlet[4]) / W_CAT_KG

    return rMeOH_pred, rCO_pred, outlet


def calculate_integral_predictions(par, df_group, Tave, model_kind):
    pred_rates = []
    outlet_flows = []

    for _, row in df_group.iterrows():
        rMeOH_pred, rCO_pred, outlet = integrate_one_experiment(
            row=row,
            par=par,
            Tave=Tave,
            model_kind=model_kind,
        )
        pred_rates.append([rMeOH_pred, rCO_pred])
        outlet_flows.append(outlet)

    pred_rates = np.asarray(pred_rates, dtype=float)
    outlet_flows = np.asarray(outlet_flows, dtype=float)

    rMeOH_pred = pred_rates[:, 0]
    rCO_pred = pred_rates[:, 1]

    return rMeOH_pred, rCO_pred, outlet_flows


# ============================================================
# 7. Objective function and metrics
# ============================================================

def objective_integral(par, df_group, Tave, model_kind, rMeOH_exp, rCO_exp):
    try:
        rMeOH_pred, rCO_pred, _ = calculate_integral_predictions(
            par=par,
            df_group=df_group,
            Tave=Tave,
            model_kind=model_kind,
        )

        denom_meoh = np.maximum(np.abs(rMeOH_exp), 1e-6)
        denom_co = np.maximum(np.abs(rCO_exp), 1e-6)

        sse_meoh = np.sum(((rMeOH_pred - rMeOH_exp) / denom_meoh) ** 2)
        sse_co = np.sum(((rCO_pred - rCO_exp) / denom_co) ** 2)

        total_sse = sse_meoh + sse_co

        if not np.isfinite(total_sse):
            return 1e30

        return float(total_sse)

    except Exception as error:
        print("objective_integral 出错:", error)
        return 1e30


def calc_r2(y_exp, y_pred):
    ss_res = np.sum((y_exp - y_pred) ** 2)
    ss_tot = np.sum((y_exp - np.mean(y_exp)) ** 2)

    if ss_tot < 1e-12:
        return np.nan

    return 1.0 - ss_res / ss_tot


def calc_rmse(y_exp, y_pred):
    return np.sqrt(np.mean((y_exp - y_pred) ** 2))


def calc_mre(y_exp, y_pred):
    denom = np.maximum(np.abs(y_exp), 1e-12)
    return np.mean(np.abs((y_pred - y_exp) / denom)) * 100.0


# ============================================================
# 8. Diagnostics and plots
# ============================================================

def diagnose_boundary_status(par):
    status = {}
    warnings = []

    for name, value, bound in zip(PARAMETER_NAMES, par, BOUNDS):
        lower, upper = bound
        width = upper - lower

        lower_threshold = lower + BOUND_WARNING_FRACTION * width
        upper_threshold = upper - BOUND_WARNING_FRACTION * width

        if value <= lower_threshold:
            param_status = "near_lower_bound"
            warnings.append(f"{name} near lower bound")
        elif value >= upper_threshold:
            param_status = "near_upper_bound"
            warnings.append(f"{name} near upper bound")
        else:
            param_status = "inside_bounds"

        status[f"{name}_boundary_status"] = param_status

    warning_text = "none" if len(warnings) == 0 else "; ".join(warnings)
    return status, warning_text


def diagnose_beta_from_excel_fugacity(df_group, ghsv_label, save_dir=None):
    fugacity_cols = ["fCO2", "fH2", "fCH3OH", "fH2O", "fCO"]
    if not all(col in df_group.columns for col in fugacity_cols):
        return pd.DataFrame([{
            "GHSV": ghsv_label,
            "note": "Skipped beta diagnosis because fugacity columns are missing."
        }])

    T = df_group["T"].to_numpy(dtype=float)
    K_f1, K_f2 = calculate_equilibrium_constants(T)

    fCO2 = np.maximum(df_group["fCO2"].to_numpy(dtype=float), EPS)
    fH2 = np.maximum(df_group["fH2"].to_numpy(dtype=float), EPS)
    fCH3OH = np.maximum(df_group["fCH3OH"].to_numpy(dtype=float), EPS)
    fH2O = np.maximum(df_group["fH2O"].to_numpy(dtype=float), EPS)
    fCO = np.maximum(df_group["fCO"].to_numpy(dtype=float), EPS)

    beta1 = (fCH3OH * fH2O) / np.maximum(K_f1 * fCO2 * (fH2 ** 3.0), EPS)
    beta2 = (fCO * fH2O) / np.maximum(K_f2 * fCO2 * fH2, EPS)

    beta_df = df_group.copy()
    beta_df["beta1_from_excel_fugacity"] = beta1
    beta_df["beta2_from_excel_fugacity"] = beta2

    beta_summary = pd.DataFrame([{
        "GHSV": ghsv_label,
        "n_points": len(df_group),
        "beta1_min": beta1.min(),
        "beta1_max": beta1.max(),
        "beta1_mean": beta1.mean(),
        "beta1_median": np.median(beta1),
        "beta2_min": beta2.min(),
        "beta2_max": beta2.max(),
        "beta2_mean": beta2.mean(),
        "beta2_median": np.median(beta2),
    }])

    if save_dir is not None:
        beta_df.to_excel(save_dir / f"GHSV_{ghsv_label}_beta_from_excel_fugacity.xlsx", index=False)
        beta_summary.to_excel(save_dir / f"GHSV_{ghsv_label}_beta_summary.xlsx", index=False)

    return beta_summary


def make_single_parity_plot(exp, pred, xlabel, ylabel, title, save_path):
    plt.figure(figsize=(6, 6))
    plt.scatter(exp, pred, alpha=0.75)

    min_val = min(np.min(exp), np.min(pred))
    max_val = max(np.max(exp), np.max(pred))

    plt.plot([min_val, max_val], [min_val, max_val], "k--")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def make_residual_vs_temperature_plot(T, residual, ylabel, title, save_path):
    plt.figure(figsize=(6, 4))
    plt.scatter(T, residual, alpha=0.75)
    plt.axhline(0, color="k", linestyle="--", linewidth=1)
    plt.xlabel("Temperature / K")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ============================================================
# 9. Fit one integral model
# ============================================================

def fit_integral_model(model_kind, model_name, df_group):
    rMeOH_exp = df_group["rMeOH"].to_numpy(dtype=float)
    rCO_exp = df_group["rCO"].to_numpy(dtype=float)
    temperature = df_group["T"].to_numpy(dtype=float)

    Tave = float(np.mean(temperature))

    print("\n" + "=" * 60)
    print(f"开始拟合 {model_name}")
    print(f"数据点数 = {len(df_group)}")
    print(f"Tave = {Tave:.2f} K")
    print("=" * 60)

    result = differential_evolution(
        objective_integral,
        bounds=BOUNDS,
        args=(df_group, Tave, model_kind, rMeOH_exp, rCO_exp),
        seed=DE_SEED,
        popsize=DE_POPSIZE,
        maxiter=DE_MAXITER,
        tol=DE_TOL,
        mutation=DE_MUTATION,
        recombination=DE_RECOMBINATION,
        polish=DE_POLISH,
        workers=DE_WORKERS,
        updating=DE_UPDATING,
        disp=True,
    )

    par_opt = result.x

    rMeOH_pred, rCO_pred, outlet_flows = calculate_integral_predictions(
        par=par_opt,
        df_group=df_group,
        Tave=Tave,
        model_kind=model_kind,
    )

    r2_meoh = calc_r2(rMeOH_exp, rMeOH_pred)
    r2_co = calc_r2(rCO_exp, rCO_pred)
    avg_r2 = np.nanmean([r2_meoh, r2_co])

    rmse_meoh = calc_rmse(rMeOH_exp, rMeOH_pred)
    rmse_co = calc_rmse(rCO_exp, rCO_pred)

    mre_meoh = calc_mre(rMeOH_exp, rMeOH_pred)
    mre_co = calc_mre(rCO_exp, rCO_pred)

    boundary_status, boundary_warning = diagnose_boundary_status(par_opt)
    params = unpack_params(par_opt)

    pred_df = pd.DataFrame(index=df_group.index)
    pred_df[f"{model_name}_F_CO2_out_mol_s"] = outlet_flows[:, 0]
    pred_df[f"{model_name}_F_H2_out_mol_s"] = outlet_flows[:, 1]
    pred_df[f"{model_name}_F_CH3OH_out_mol_s"] = outlet_flows[:, 2]
    pred_df[f"{model_name}_F_H2O_out_mol_s"] = outlet_flows[:, 3]
    pred_df[f"{model_name}_F_CO_out_mol_s"] = outlet_flows[:, 4]

    pred_df[f"{model_name}_rMeOH_pred"] = rMeOH_pred
    pred_df[f"{model_name}_rCO_pred"] = rCO_pred
    pred_df[f"{model_name}_res_MeOH"] = rMeOH_pred - rMeOH_exp
    pred_df[f"{model_name}_res_CO"] = rCO_pred - rCO_exp
    pred_df[f"{model_name}_rel_error_rMeOH_%"] = (
        np.abs((rMeOH_pred - rMeOH_exp) / np.maximum(np.abs(rMeOH_exp), 1e-12)) * 100.0
    )
    pred_df[f"{model_name}_rel_error_rCO_%"] = (
        np.abs((rCO_pred - rCO_exp) / np.maximum(np.abs(rCO_exp), 1e-12)) * 100.0
    )

    summary = {
        "model": model_name,
        "model_kind": model_kind,
        "FAST_MODE": FAST_MODE,
        "Tave": Tave,
        "optimizer_success": result.success,
        "optimizer_message": str(result.message),
        "objective": result.fun,
        "r2_meoh": r2_meoh,
        "r2_co": r2_co,
        "avg_r2": avg_r2,
        "rmse_meoh": rmse_meoh,
        "rmse_co": rmse_co,
        "mre_meoh_%": mre_meoh,
        "mre_co_%": mre_co,
        "boundary_warning": boundary_warning,
    }

    params_output = {
        "model": model_name,
        "model_kind": model_kind,
        "Tave": Tave,
        **params,
        "boundary_warning": boundary_warning,
    }
    params_output.update(boundary_status)

    print(f"\n模型 {model_name} 拟合完成")
    print("success =", result.success)
    print("message =", result.message)
    print("objective =", result.fun)
    print(f"Tave = {Tave:.2f} K")
    print(f"r2_meoh = {r2_meoh:.6f}")
    print(f"r2_co   = {r2_co:.6f}")
    print(f"avg_r2  = {avg_r2:.6f}")
    print(f"mre_meoh% = {mre_meoh:.4f}")
    print(f"mre_co%   = {mre_co:.4f}")
    print("boundary_warning =", boundary_warning)

    return summary, params_output, pred_df


# ============================================================
# 10. Fit one GHSV group
# ============================================================

def fit_one_ghsv_group(df_group, ghsv_value):
    ghsv_label = int(round(float(ghsv_value)))

    group_dir = OUT_DIR / f"GHSV_{ghsv_label}"
    group_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"开始拟合 GHSV = {ghsv_label}")
    print(f"该组数据点数 = {len(df_group)}")
    print(f"Tave = {df_group['T'].mean():.2f} K")
    print(f"flow_mL_min unique = {sorted(df_group['flow_mL_min'].unique())}")
    print("=" * 60)

    beta_summary = diagnose_beta_from_excel_fugacity(
        df_group=df_group,
        ghsv_label=ghsv_label,
        save_dir=group_dir,
    )

    summary_simple, params_simple, pred_simple = fit_integral_model(
        model_kind="simple_powerlaw_integral",
        model_name="simple_powerlaw_integral",
        df_group=df_group,
    )

    summary_beta, params_beta, pred_beta = fit_integral_model(
        model_kind="powerlaw_with_1_minus_beta_integral",
        model_name="powerlaw_with_1_minus_beta_integral",
        df_group=df_group,
    )

    summary_df = pd.DataFrame([summary_simple, summary_beta])
    params_df = pd.DataFrame([params_simple, params_beta])

    result_df = df_group.copy()

    for col in pred_simple.columns:
        result_df[col] = pred_simple[col].values

    for col in pred_beta.columns:
        result_df[col] = pred_beta[col].values

    summary_df.insert(0, "GHSV", ghsv_label)
    params_df.insert(0, "GHSV", ghsv_label)

    summary_df.to_excel(group_dir / f"GHSV_{ghsv_label}_integral_summary.xlsx", index=False)
    params_df.to_excel(group_dir / f"GHSV_{ghsv_label}_integral_parameters.xlsx", index=False)
    result_df.to_excel(group_dir / f"GHSV_{ghsv_label}_integral_predictions.xlsx", index=False)

    for model_name in ["simple_powerlaw_integral", "powerlaw_with_1_minus_beta_integral"]:
        make_single_parity_plot(
            exp=result_df["rMeOH"].values,
            pred=result_df[f"{model_name}_rMeOH_pred"].values,
            xlabel="Experimental MeOH",
            ylabel="Predicted MeOH",
            title=f"{model_name} MeOH, GHSV = {ghsv_label}",
            save_path=group_dir / f"GHSV_{ghsv_label}_{model_name}_parity_meoh.png",
        )

        make_single_parity_plot(
            exp=result_df["rCO"].values,
            pred=result_df[f"{model_name}_rCO_pred"].values,
            xlabel="Experimental CO",
            ylabel="Predicted CO",
            title=f"{model_name} CO, GHSV = {ghsv_label}",
            save_path=group_dir / f"GHSV_{ghsv_label}_{model_name}_parity_co.png",
        )

        make_residual_vs_temperature_plot(
            T=result_df["T"].values,
            residual=result_df[f"{model_name}_res_MeOH"].values,
            ylabel="Residual MeOH",
            title=f"{model_name} MeOH residual vs T, GHSV = {ghsv_label}",
            save_path=group_dir / f"GHSV_{ghsv_label}_{model_name}_residual_vs_T_meoh.png",
        )

        make_residual_vs_temperature_plot(
            T=result_df["T"].values,
            residual=result_df[f"{model_name}_res_CO"].values,
            ylabel="Residual CO",
            title=f"{model_name} CO residual vs T, GHSV = {ghsv_label}",
            save_path=group_dir / f"GHSV_{ghsv_label}_{model_name}_residual_vs_T_co.png",
        )

    return summary_df, params_df, result_df, beta_summary


# ============================================================
# 11. Main program
# ============================================================

def main():
    print("\n==============================================")
    print("Integral power law and beta model fitting")
    print("==============================================")
    print(f"Excel file: {EXCEL_FILE}")
    print(f"Output folder: {OUT_DIR}")
    print(f"FAST_MODE = {FAST_MODE}")
    print(f"W_CAT_KG = {W_CAT_KG}")
    print(f"RK4_STEPS = {RK4_STEPS}")
    print(f"DE seed = {DE_SEED}")
    print(f"DE popsize = {DE_POPSIZE}")
    print(f"DE maxiter = {DE_MAXITER}")
    print(f"DE tol = {DE_TOL}")
    print(f"DE polish = {DE_POLISH}")

    df = load_data()

    if TARGET_GHSV is not None:
        df = df[np.isclose(df["GHSV"], TARGET_GHSV)].copy()
        if df.empty:
            raise ValueError(f"No valid rows were found for TARGET_GHSV = {TARGET_GHSV}.")
        print(f"\n只拟合 GHSV = {TARGET_GHSV:g}")

    ghsv_values = sorted(df["GHSV"].dropna().unique())
    print("\n检测到的 GHSV 分组:", ghsv_values)

    all_summary = []
    all_params = []
    all_predictions = []
    all_beta_summary = []

    for ghsv in ghsv_values:
        df_group = df[np.isclose(df["GHSV"], ghsv)].copy()

        if len(df_group) < 8:
            print(f"GHSV = {ghsv} 的数据点太少，跳过")
            continue

        summary_df, params_df, result_df, beta_summary = fit_one_ghsv_group(
            df_group=df_group,
            ghsv_value=ghsv,
        )

        all_summary.append(summary_df)
        all_params.append(params_df)
        all_predictions.append(result_df)
        all_beta_summary.append(beta_summary)

    if len(all_summary) == 0:
        print("没有成功拟合任何 GHSV 组")
        return

    all_summary_df = pd.concat(all_summary, ignore_index=True)
    all_params_df = pd.concat(all_params, ignore_index=True)
    all_predictions_df = pd.concat(all_predictions, ignore_index=True)
    all_beta_summary_df = pd.concat(all_beta_summary, ignore_index=True)

    all_summary_df.to_excel(OUT_DIR / "all_ghsv_integral_summary.xlsx", index=False)
    all_params_df.to_excel(OUT_DIR / "all_ghsv_integral_parameters.xlsx", index=False)
    all_predictions_df.to_excel(OUT_DIR / "all_ghsv_integral_predictions.xlsx", index=False)
    all_beta_summary_df.to_excel(OUT_DIR / "all_ghsv_beta_summary.xlsx", index=False)

    log_row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "excel_file": str(EXCEL_FILE),
        "TARGET_GHSV": TARGET_GHSV,
        "FAST_MODE": FAST_MODE,
        "W_CAT_KG": W_CAT_KG,
        "RK4_STEPS": RK4_STEPS,
        "DE_SEED": DE_SEED,
        "DE_POPSIZE": DE_POPSIZE,
        "DE_MAXITER": DE_MAXITER,
        "DE_TOL": DE_TOL,
        "DE_POLISH": DE_POLISH,
        "n_total_points": len(all_predictions_df),
    }

    log_df = pd.DataFrame([log_row])
    if EXPERIMENT_LOG_PATH.exists():
        old_log = pd.read_csv(EXPERIMENT_LOG_PATH)
        log_df = pd.concat([old_log, log_df], ignore_index=True)
    log_df.to_csv(EXPERIMENT_LOG_PATH, index=False)

    print("\n全部 GHSV 分组积分拟合完成")
    print("\n总汇总表：")
    print(all_summary_df)

    print("\n总参数表：")
    print(all_params_df)

    print("\n总 beta 汇总表：")
    print(all_beta_summary_df)

    print("\n结果已保存到文件夹：")
    print(OUT_DIR)


if __name__ == "__main__":
    main()