from pathlib import Path
from datetime import datetime
import importlib.util

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares


# ============================================================
# 1. Model E configuration
# ============================================================

# Model E is kept separate from the direct-rate baseline. It imports the same
# data loader, Kf equations, parameter transform, and metric helpers from the
# baseline script without changing that baseline.
BASE_DIR = Path(__file__).resolve().parent
BASELINE_SCRIPT = BASE_DIR / "12000GHSV.py"
DATA_PATH = BASE_DIR / "data" / "12000GHSV.xlsx"
OUTPUT_DIR = BASE_DIR / "output"
MODEL_OUTPUT_DIR = OUTPUT_DIR / "model_E_integral"
EXPERIMENT_LOG_PATH = OUTPUT_DIR / "experiment_log.csv"
PREDICTION_PATH = MODEL_OUTPUT_DIR / "predictions_model_E_integral.xlsx"

MODEL_NAME = "Model_E_integral_two_reaction_LHHW"
EPS = 1e-30
RANDOM_SEED = 1

# Reactor information specified for the integral plug-flow calculation.
W_CAT_KG = 0.0002
STANDARD_FLOW_L_PER_MIN = 42e-3
STANDARD_MOLAR_VOLUME_L_PER_MOL = 22.414
F_TOTAL_IN = STANDARD_FLOW_L_PER_MIN / STANDARD_MOLAR_VOLUME_L_PER_MOL / 60.0
RK4_STEPS = 40


def load_baseline_module():
    """Load reusable baseline functions from 12000GHSV.py."""

    spec = importlib.util.spec_from_file_location("direct_rate_baseline", BASELINE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


baseline = load_baseline_module()


# ============================================================
# 2. Same local two-reaction AAA_LHHW expression
# ============================================================

def lhhw_rates_from_fugacity(x, T, T_ref, fCO2, fH2, fCH3OH, fH2O, fCO):
    """
    Evaluate the current two-reaction AAA_LHHW expression at local PFR conditions.

    This function keeps Kf1, Kf2, the adsorption denominator, and the two
    driving-force expressions unchanged. The difference from direct-rate
    fitting is not the kinetic equation; it is that fugacities are recalculated
    from changing molar flows inside the reactor instead of taken directly from
    the experimental row.
    """

    p = baseline.unpack_params(x)

    Kf1, Kf2 = baseline.calc_Kf(T)

    fH2_safe = max(float(fH2), EPS)
    Kf1_safe = max(float(np.asarray(Kf1)), EPS)
    Kf2_safe = max(float(np.asarray(Kf2)), EPS)

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        # Same reference-temperature Arrhenius form as the direct-rate baseline.
        k1_eff = p["k1_eff_ref"] * np.exp(
            -p["E1_over_R"] * (1.0 / T - 1.0 / T_ref)
        )

        k2_eff = p["k2_eff_ref"] * np.exp(
            -p["E2_over_R"] * (1.0 / T - 1.0 / T_ref)
        )

    # Same carbon-site adsorption term: CO2 and CO compete for sites.
    ads_carbon = 1.0 + p["KCO2"] * fCO2 + p["KCO"] * fCO

    # Same hydrogen/water adsorption term used in the direct-rate baseline.
    ads_hydrogen_water = np.sqrt(fH2_safe) + p["KH2O_H2"] * fH2O

    adsorption_term = ads_carbon * ads_hydrogen_water
    adsorption_term = max(float(adsorption_term), EPS)

    # R1: CO2 + 3H2 <-> CH3OH + H2O
    driving_1 = (
        fCO2 * fH2_safe ** 1.5
        - (fCH3OH * fH2O) / (Kf1_safe * fH2_safe ** 1.5)
    )

    # R2: CO2 + H2 <-> CO + H2O
    driving_2 = (
        fCO2 * fH2_safe
        - (fCO * fH2O) / Kf2_safe
    )

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        r1 = k1_eff * driving_1 / adsorption_term
        r2 = k2_eff * driving_2 / adsorption_term

    return float(r1), float(r2)


# ============================================================
# 3. Integral plug-flow reactor balances
# ============================================================

def inlet_flows_from_hc(HC):
    """
    Construct inlet molar flows from the H2/CO2 ratio.

    Only CO2 and H2 enter the reactor. CH3OH, H2O, and CO start at zero as
    specified for Model E.
    """

    y_co2_in = 1.0 / (1.0 + HC)
    y_h2_in = HC / (1.0 + HC)

    return np.array([
        y_co2_in * F_TOTAL_IN,
        y_h2_in * F_TOTAL_IN,
        0.0,
        0.0,
        0.0,
    ], dtype=float)


def pfr_rhs(_W, flows, x, T, p_MPa, T_ref):
    """
    PFR material balances in catalyst-mass coordinates.

    The direct-rate baseline evaluates one rate at the measured row fugacities.
    This integral method instead updates composition along W, so inhibition,
    reverse driving forces, and product formation are accumulated over the bed.
    """

    flows = np.maximum(flows, 0.0)
    F_total = max(float(np.sum(flows)), EPS)
    y = flows / F_total

    # Ideal-gas fugacity approximation requested for the local reactor state.
    fCO2, fH2, fCH3OH, fH2O, fCO = y * p_MPa

    r1, r2 = lhhw_rates_from_fugacity(
        x, T, T_ref, fCO2, fH2, fCH3OH, fH2O, fCO
    )

    return np.array([
        -(r1 + r2),
        -(3.0 * r1 + r2),
        r1,
        r1 + r2,
        r2,
    ], dtype=float)


def integrate_row(row, x, T_ref):
    """
    Integrate one experimental condition from W = 0 to W_cat.

    A fixed RK4 grid is used instead of an adaptive solver because Model E calls
    the reactor calculation many times during fitting. The bed is short and
    isothermal for each row, so this gives a deterministic, fast integral
    approximation while still updating composition through the catalyst mass.
    """

    inlet = inlet_flows_from_hc(float(row["HC"]))
    T = float(row["T"])
    p_MPa = float(row["p_MPa"])

    flows = inlet.copy()
    step = W_CAT_KG / RK4_STEPS
    W = 0.0

    for _ in range(RK4_STEPS):
        k1 = pfr_rhs(W, flows, x, T, p_MPa, T_ref)
        k2 = pfr_rhs(W + 0.5 * step, flows + 0.5 * step * k1, x, T, p_MPa, T_ref)
        k3 = pfr_rhs(W + 0.5 * step, flows + 0.5 * step * k2, x, T, p_MPa, T_ref)
        k4 = pfr_rhs(W + step, flows + step * k3, x, T, p_MPa, T_ref)
        flows = flows + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        flows = np.maximum(flows, 0.0)
        if not np.all(np.isfinite(flows)):
            raise FloatingPointError("Non-finite molar flow during PFR integration.")
        W += step

    outlet = flows

    r_ch3oh_pred = (outlet[2] - inlet[2]) / W_CAT_KG
    r_co_pred = (outlet[4] - inlet[4]) / W_CAT_KG

    return r_ch3oh_pred, r_co_pred, outlet


def predict_integral_rates(df, x):
    """Predict outlet-averaged production rates for every experimental row."""

    T_ref = float(np.mean(df["T"].values.astype(float)))
    predictions = []
    outlets = []

    for _, row in df.iterrows():
        r_ch3oh_pred, r_co_pred, outlet = integrate_row(row, x, T_ref)
        predictions.append((r_ch3oh_pred, r_co_pred))
        outlets.append(outlet)

    predictions = np.asarray(predictions, dtype=float)
    outlets = np.asarray(outlets, dtype=float)
    return predictions[:, 0], predictions[:, 1], outlets


# ============================================================
# 4. Objective function using only CH3OH and CO
# ============================================================

def residuals_integral(x, df, weight_ch3oh=1.0, weight_co=1.0):
    """
    Residual vector for Model E.

    rCO2 is not included. The objective compares only integral PFR outlet
    production rates for CH3OH and CO against their experimental rates.
    """

    try:
        r_ch3oh_pred, r_co_pred, _ = predict_integral_rates(df, x)
    except (RuntimeError, FloatingPointError, ValueError):
        penalty = np.full(2 * len(df), 1e6, dtype=float)
        return penalty

    r_ch3oh_exp = df["rCH3OH"].values.astype(float)
    r_co_exp = df["rCO"].values.astype(float)

    scale_ch3oh = np.maximum(
        np.abs(r_ch3oh_exp),
        0.05 * np.max(np.abs(r_ch3oh_exp)),
    )
    scale_co = np.maximum(
        np.abs(r_co_exp),
        0.05 * np.max(np.abs(r_co_exp)),
    )

    res_ch3oh = weight_ch3oh * (r_ch3oh_pred - r_ch3oh_exp) / scale_ch3oh
    res_co = weight_co * (r_co_pred - r_co_exp) / scale_co

    return np.concatenate([res_ch3oh, res_co])


def objective_integral_for_de(x, df):
    """Scalar objective for the global-search stage."""

    res = residuals_integral(x, df)
    return float(np.sum(res ** 2))


# ============================================================
# 5. Model E fitting
# ============================================================

def fit_model_E_integral(df):
    """
    Fit Model E with the same seven kinetic parameters as the baseline.

    The initial point is the already reproducible direct-rate baseline. This is
    faster than another full global search because the integral PFR objective is
    much more expensive: each residual evaluation integrates all 75 reactors.
    """

    bounds = [
        (-30, 10),      # ln_k1_eff_ref
        (0, 30000),     # E1_over_R
        (-30, 10),      # ln_k2_eff_ref
        (0, 30000),     # E2_over_R
        (-20, 10),      # ln_KCO2
        (-20, 10),      # ln_KCO
        (-20, 10),      # ln_KH2O_H2
    ]

    lower_bounds = np.array([b[0] for b in bounds], dtype=float)
    upper_bounds = np.array([b[1] for b in bounds], dtype=float)

    direct_result, _, _ = baseline.fit_lhhw_two_reactions(df)
    initial_objective = objective_integral_for_de(direct_result.x, df)

    ls_result = least_squares(
        fun=lambda x: residuals_integral(x, df),
        x0=direct_result.x,
        bounds=(lower_bounds, upper_bounds),
        max_nfev=1000,
        loss="soft_l1",
        f_scale=1.0,
        x_scale="jac",
    )

    return ls_result, direct_result, initial_objective


# ============================================================
# 6. Metrics, plots, and saved outputs
# ============================================================

def evaluate_model_E(df, result):
    """Build predictions, residuals, metrics, and fitted parameter report."""

    r_ch3oh_pred, r_co_pred, outlets = predict_integral_rates(df, result.x)

    pred_df = df.copy()
    pred_df["F_CO2_out_mol_s"] = outlets[:, 0]
    pred_df["F_H2_out_mol_s"] = outlets[:, 1]
    pred_df["F_CH3OH_out_mol_s"] = outlets[:, 2]
    pred_df["F_H2O_out_mol_s"] = outlets[:, 3]
    pred_df["F_CO_out_mol_s"] = outlets[:, 4]
    pred_df["rCH3OH_pred"] = r_ch3oh_pred
    pred_df["rCO_pred"] = r_co_pred
    pred_df["res_CH3OH"] = pred_df["rCH3OH_pred"] - pred_df["rCH3OH"]
    pred_df["res_CO"] = pred_df["rCO_pred"] - pred_df["rCO"]

    metrics = {
        "R2_CH3OH": baseline.calc_r2(pred_df["rCH3OH"], pred_df["rCH3OH_pred"]),
        "R2_CO": baseline.calc_r2(pred_df["rCO"], pred_df["rCO_pred"]),
        "RMSE_CH3OH": baseline.calc_rmse(pred_df["rCH3OH"], pred_df["rCH3OH_pred"]),
        "RMSE_CO": baseline.calc_rmse(pred_df["rCO"], pred_df["rCO_pred"]),
        "MRE_CH3OH_percent": baseline.calc_mre_percent(
            pred_df["rCH3OH"], pred_df["rCH3OH_pred"]
        ),
        "MRE_CO_percent": baseline.calc_mre_percent(
            pred_df["rCO"], pred_df["rCO_pred"]
        ),
    }

    params = baseline.unpack_params(result.x)
    params_report = params.copy()
    params_report["E1_kJ_per_mol"] = params["E1_over_R"] * 8.314 / 1000.0
    params_report["E2_kJ_per_mol"] = params["E2_over_R"] * 8.314 / 1000.0

    return pred_df, metrics, params_report


def parity_plot(y_exp, y_pred, title, output_path):
    """Save a parity plot for one Model E target."""

    y_exp = np.asarray(y_exp, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    lower = min(np.min(y_exp), np.min(y_pred))
    upper = max(np.max(y_exp), np.max(y_pred))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_exp, y_pred)
    ax.plot([lower, upper], [lower, upper], color="black", linewidth=1)
    ax.set_xlabel("Experimental rate")
    ax.set_ylabel("Predicted integral rate")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def residual_vs_T_plot(df, residual_col, title, output_path):
    """Save a residual versus temperature plot for one Model E target."""

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["T"], df[residual_col])
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Temperature / K")
    ax.set_ylabel("Residual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def append_experiment_log(pred_df, metrics, params_report, result, initial_objective):
    """Append Model E results to the shared experiment log."""

    log_row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_name": MODEL_NAME,
        "data_file": str(DATA_PATH.relative_to(BASE_DIR)),
        "n_points": len(pred_df),
        "random_seed": RANDOM_SEED,
        "W_cat_kg": W_CAT_KG,
        "F_total_in_mol_s": F_TOTAL_IN,
        "rk4_steps": RK4_STEPS,
        "least_squares_success": result.success,
        "least_squares_message": result.message,
        "initial_integral_objective": initial_objective,
        "ls_objective": np.sum(result.fun ** 2),
        **metrics,
        **params_report,
    }

    log_df = pd.DataFrame([log_row])
    if EXPERIMENT_LOG_PATH.exists():
        previous_log = pd.read_csv(EXPERIMENT_LOG_PATH)
        log_df = pd.concat([previous_log, log_df], ignore_index=True)
    log_df.to_csv(EXPERIMENT_LOG_PATH, index=False)


def save_model_E_outputs(pred_df, metrics, params_report, result, initial_objective):
    """Save Model E predictions, figures, and log entry."""

    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pred_df.to_excel(PREDICTION_PATH, index=False)

    parity_plot(
        pred_df["rCH3OH"],
        pred_df["rCH3OH_pred"],
        "Model E parity plot for CH3OH",
        MODEL_OUTPUT_DIR / "parity_CH3OH_model_E_integral.png",
    )
    parity_plot(
        pred_df["rCO"],
        pred_df["rCO_pred"],
        "Model E parity plot for CO",
        MODEL_OUTPUT_DIR / "parity_CO_model_E_integral.png",
    )
    residual_vs_T_plot(
        pred_df,
        "res_CH3OH",
        "Model E CH3OH residual vs temperature",
        MODEL_OUTPUT_DIR / "residual_vs_T_CH3OH_model_E_integral.png",
    )
    residual_vs_T_plot(
        pred_df,
        "res_CO",
        "Model E CO residual vs temperature",
        MODEL_OUTPUT_DIR / "residual_vs_T_CO_model_E_integral.png",
    )

    append_experiment_log(pred_df, metrics, params_report, result, initial_objective)


# ============================================================
# 7. Main Model E run
# ============================================================

def main():
    """Fit and save the integral PFR version of the current AAA_LHHW model."""

    df = baseline.load_data(DATA_PATH, sheet_name=0)

    print(f"Model name: {MODEL_NAME}")
    print(f"Loaded data points: {len(df)}")
    print("Fitting targets: rCH3OH, rCO")
    print("Excluded from objective: rCO2")
    print(f"W_cat_kg: {W_CAT_KG:.12g}")
    print(f"F_total_in_mol_s: {F_TOTAL_IN:.12g}")

    result, direct_result, initial_objective = fit_model_E_integral(df)
    pred_df, metrics, params_report = evaluate_model_E(df, result)
    save_model_E_outputs(pred_df, metrics, params_report, result, initial_objective)

    print("\n==============================")
    print("Fit status")
    print("==============================")
    print("least_squares success:", result.success)
    print("least_squares message:", result.message)
    print(f"Initial integral objective from direct-rate baseline: {initial_objective:.12g}")
    print(f"LS objective: {np.sum(result.fun ** 2):.12g}")

    print("\n==============================")
    print("Metrics")
    print("==============================")
    for key, value in metrics.items():
        print(f"{key}: {value:.12g}")

    print("\n==============================")
    print("Fitted parameters")
    print("==============================")
    for key, value in params_report.items():
        print(f"{key}: {value:.12g}")

    print("\n==============================")
    print("Saved outputs")
    print("==============================")
    print(PREDICTION_PATH.relative_to(BASE_DIR))
    print((MODEL_OUTPUT_DIR / "parity_CH3OH_model_E_integral.png").relative_to(BASE_DIR))
    print((MODEL_OUTPUT_DIR / "parity_CO_model_E_integral.png").relative_to(BASE_DIR))
    print((MODEL_OUTPUT_DIR / "residual_vs_T_CH3OH_model_E_integral.png").relative_to(BASE_DIR))
    print((MODEL_OUTPUT_DIR / "residual_vs_T_CO_model_E_integral.png").relative_to(BASE_DIR))
    print(EXPERIMENT_LOG_PATH.relative_to(BASE_DIR))


if __name__ == "__main__":
    main()
