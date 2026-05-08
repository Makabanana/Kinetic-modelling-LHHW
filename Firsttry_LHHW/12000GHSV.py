from pathlib import Path
from datetime import datetime

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, least_squares


# ============================================================
# 1. Reproducible baseline configuration
# ============================================================

# All paths are anchored to this script location so the fit can be rerun from
# any working directory while still using the Firsttry_LHHW project folders.
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "12000GHSV.xlsx"
OUTPUT_DIR = BASE_DIR / "output"
EXPERIMENT_LOG_PATH = OUTPUT_DIR / "experiment_log.csv"
PREDICTION_PATH = OUTPUT_DIR / "predictions_baseline.xlsx"

# Small positive floor used only to avoid division by zero in rates and metrics.
EPS = 1e-30

# Fixed random seed for the global optimizer. This makes the baseline fit
# reproducible as long as package versions are unchanged.
RANDOM_SEED = 1


# ============================================================
# 2. Thermodynamic equilibrium constants
# ============================================================

def calc_Kf(T):
    """
    Calculate equilibrium constants for the two reversible reactions.

    Kf1 is for CO2 + 3H2 <-> CH3OH + H2O.
    Kf2 is for CO2 + H2 <-> CO + H2O.

    The equations below are intentionally unchanged from the current baseline.
    """

    T = np.asarray(T, dtype=float)

    Kf1 = np.exp(
        1.6654 + 4553.34 / T - 2.72613 * np.log(T)
        - 1.422914e-2 * T + 0.172060e-4 * T**2
        - 1.106294e-8 * T**3 + 0.319698e-11 * T**4
    ) * (0.101325) ** (-2)

    Kf2 = np.exp(
        -11.4998 - 4649.92 / T + 3.2066 * np.log(T)
        - 0.0107251 * T + 0.697955e-5 * T**2
        - 0.336848e-8 * T**3 + 0.811184e-12 * T**4
    )

    return Kf1, Kf2


# ============================================================
# 3. Experimental data loading
# ============================================================

def load_data(file_path, sheet_name=0):
    """
    Read the experimental 12000 GHSV data.

    Only rCH3OH and rCO are loaded as fitted kinetic targets. rCO2 is not used
    in the objective function for this reproducible baseline.
    """

    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Normalize column names because Excel headers may contain extra spaces or
    # line breaks. This keeps the kinetic code independent of header formatting.
    df.columns = [str(c).strip().replace("\n", " ") for c in df.columns]

    rename_dict = {
        "H/C RATIO": "HC",
        "H/C": "HC",
        "p": "p_MPa",
        "GHSV": "GHSV",
        "T": "T",
        "fCO2": "fCO2",
        "fH2": "fH2",
        "fCH3OH": "fCH3OH",
        "fH2O": "fH2O",
        "fCO": "fCO",
        "r CH3OH": "rCH3OH",
        "rCH3OH": "rCH3OH",
        "r CO": "rCO",
        "rCO": "rCO",
    }
    df = df.rename(columns=rename_dict)

    required_cols = [
        "HC", "p_MPa", "GHSV", "T",
        "fCO2", "fH2", "fCH3OH", "fH2O", "fCO",
        "rCH3OH", "rCO",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Convert unit rows or nonnumeric cells to NaN, then keep only complete
    # rows needed by the two-reaction kinetic model.
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required_cols).copy()
    df = df[df["GHSV"] == 12000].copy()

    # Add thermodynamic driving-force constants to each experimental row.
    df["Kf1"], df["Kf2"] = calc_Kf(df["T"].values)

    if df.empty:
        raise ValueError("No valid GHSV = 12000 rows were found.")

    return df


# ============================================================
# 4. Kinetic parameter transformation
# ============================================================

def unpack_params(x):
    """
    Convert optimizer variables into positive kinetic parameters.

    The optimizer uses log variables for rate and adsorption constants so that
    k and K remain positive during fitting.
    """

    (
        ln_k1_eff_ref,
        E1_over_R,
        ln_k2_eff_ref,
        E2_over_R,
        ln_KCO2,
        ln_KCO,
        ln_KH2O_H2,
    ) = x

    params = {
        "k1_eff_ref": np.exp(ln_k1_eff_ref),
        "E1_over_R": E1_over_R,
        "k2_eff_ref": np.exp(ln_k2_eff_ref),
        "E2_over_R": E2_over_R,
        "KCO2": np.exp(ln_KCO2),
        "KCO": np.exp(ln_KCO),
        "KH2O_H2": np.exp(ln_KH2O_H2),
    }

    return params


# ============================================================
# 5. Two-reaction AAA_LHHW kinetic model
# ============================================================

def lhhw_rate_model(x, df):
    """
    Predict methanol and CO formation rates with the current AAA_LHHW model.

    r1 represents CO2 hydrogenation to CH3OH.
    r2 represents the reverse water-gas shift route to CO.

    The adsorption structure, driving forces, and rate equations are unchanged
    from the current two-reaction baseline.
    """

    p = unpack_params(x)

    T = df["T"].values.astype(float)
    T_ref = np.mean(T)

    fCO2 = df["fCO2"].values.astype(float)
    fH2 = df["fH2"].values.astype(float)
    fCH3OH = df["fCH3OH"].values.astype(float)
    fH2O = df["fH2O"].values.astype(float)
    fCO = df["fCO"].values.astype(float)

    Kf1 = df["Kf1"].values.astype(float)
    Kf2 = df["Kf2"].values.astype(float)

    fH2_safe = np.maximum(fH2, EPS)
    Kf1_safe = np.maximum(Kf1, EPS)
    Kf2_safe = np.maximum(Kf2, EPS)

    # Reference-temperature Arrhenius form for the effective rate constants.
    # k1_eff controls methanol formation, and k2_eff controls CO formation.
    k1_eff = p["k1_eff_ref"] * np.exp(
        -p["E1_over_R"] * (1.0 / T - 1.0 / T_ref)
    )

    k2_eff = p["k2_eff_ref"] * np.exp(
        -p["E2_over_R"] * (1.0 / T - 1.0 / T_ref)
    )

    # Carbon-site adsorption term: CO2 and CO compete for the same sites.
    ads_carbon = 1.0 + p["KCO2"] * fCO2 + p["KCO"] * fCO

    # Hydrogen/water-site term: H2 supplies surface hydrogen while H2O inhibits.
    ads_hydrogen_water = np.sqrt(fH2_safe) + p["KH2O_H2"] * fH2O

    adsorption_term = ads_carbon * ads_hydrogen_water
    adsorption_term = np.maximum(adsorption_term, EPS)

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

    r1 = k1_eff * driving_1 / adsorption_term
    r2 = k2_eff * driving_2 / adsorption_term

    return r1, r2


# ============================================================
# 6. Objective function using only CH3OH and CO
# ============================================================

def residuals(x, df, weight_ch3oh=1.0, weight_co=1.0):
    """
    Build the residual vector for fitting.

    Only rCH3OH and rCO are included. rCO2 is deliberately excluded so this
    baseline fits exactly the two measured production-rate targets requested.
    """

    r1_pred, r2_pred = lhhw_rate_model(x, df)

    r_ch3oh_exp = df["rCH3OH"].values.astype(float)
    r_co_exp = df["rCO"].values.astype(float)

    # Relative residuals with a floor prevent very small observed rates from
    # dominating the objective numerically.
    scale_ch3oh = np.maximum(
        np.abs(r_ch3oh_exp),
        0.05 * np.max(np.abs(r_ch3oh_exp)),
    )

    scale_co = np.maximum(
        np.abs(r_co_exp),
        0.05 * np.max(np.abs(r_co_exp)),
    )

    res_ch3oh = weight_ch3oh * (r1_pred - r_ch3oh_exp) / scale_ch3oh
    res_co = weight_co * (r2_pred - r_co_exp) / scale_co

    return np.concatenate([res_ch3oh, res_co])


def objective_for_de(x, df):
    """Scalar sum of squared residuals for differential evolution."""

    res = residuals(x, df)
    return np.sum(res ** 2)


# ============================================================
# 7. Baseline optimizer
# ============================================================

def fit_lhhw_two_reactions(df):
    """
    Fit the current AAA_LHHW model in two stages.

    Differential evolution gives a reproducible global search starting point,
    and least_squares performs the final local refinement.
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

    de_result = differential_evolution(
        func=lambda x: objective_for_de(x, df),
        bounds=bounds,
        seed=RANDOM_SEED,
        maxiter=1500,
        popsize=20,
        polish=False,
        workers=1,
    )

    ls_result = least_squares(
        fun=lambda x: residuals(x, df),
        x0=de_result.x,
        bounds=(lower_bounds, upper_bounds),
        max_nfev=10000,
        loss="soft_l1",
        f_scale=1.0,
        x_scale="jac",
    )

    return ls_result, de_result, bounds


# ============================================================
# 8. Fit metrics
# ============================================================

def calc_r2(y_exp, y_pred):
    """Coefficient of determination for one fitted rate target."""

    y_exp = np.asarray(y_exp, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    ss_res = np.sum((y_exp - y_pred) ** 2)
    ss_tot = np.sum((y_exp - np.mean(y_exp)) ** 2)

    return 1.0 - ss_res / ss_tot


def calc_rmse(y_exp, y_pred):
    """Root mean squared error for one fitted rate target."""

    y_exp = np.asarray(y_exp, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return np.sqrt(np.mean((y_exp - y_pred) ** 2))


def calc_mre_percent(y_exp, y_pred):
    """Mean relative error in percent for one fitted rate target."""

    y_exp = np.asarray(y_exp, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denominator = np.maximum(np.abs(y_exp), EPS)

    return np.mean(np.abs((y_pred - y_exp) / denominator)) * 100.0


def evaluate_fit(df, result):
    """Create prediction table and metrics for CH3OH and CO only."""

    r1_pred, r2_pred = lhhw_rate_model(result.x, df)

    pred_df = df.copy()

    pred_df["rCH3OH_pred"] = r1_pred
    pred_df["rCO_pred"] = r2_pred

    pred_df["res_CH3OH"] = pred_df["rCH3OH_pred"] - pred_df["rCH3OH"]
    pred_df["res_CO"] = pred_df["rCO_pred"] - pred_df["rCO"]

    metrics = {
        "R2_CH3OH": calc_r2(pred_df["rCH3OH"], pred_df["rCH3OH_pred"]),
        "R2_CO": calc_r2(pred_df["rCO"], pred_df["rCO_pred"]),
        "RMSE_CH3OH": calc_rmse(pred_df["rCH3OH"], pred_df["rCH3OH_pred"]),
        "RMSE_CO": calc_rmse(pred_df["rCO"], pred_df["rCO_pred"]),
        "MRE_CH3OH_percent": calc_mre_percent(pred_df["rCH3OH"], pred_df["rCH3OH_pred"]),
        "MRE_CO_percent": calc_mre_percent(pred_df["rCO"], pred_df["rCO_pred"]),
    }

    params = unpack_params(result.x)
    params_report = params.copy()
    params_report["E1_kJ_per_mol"] = params["E1_over_R"] * 8.314 / 1000.0
    params_report["E2_kJ_per_mol"] = params["E2_over_R"] * 8.314 / 1000.0

    return pred_df, metrics, params_report


# ============================================================
# 9. Diagnostic plots
# ============================================================

def parity_plot(y_exp, y_pred, title, output_path):
    """
    Save a parity plot for one fitted target.

    Points close to y = x indicate that the fitted rate matches the experiment.
    """

    y_exp = np.asarray(y_exp, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    lower = min(np.min(y_exp), np.min(y_pred))
    upper = max(np.max(y_exp), np.max(y_pred))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_exp, y_pred)
    ax.plot([lower, upper], [lower, upper], color="black", linewidth=1)
    ax.set_xlabel("Experimental rate")
    ax.set_ylabel("Predicted rate")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def residual_vs_T_plot(df, residual_col, title, output_path):
    """
    Save residual versus temperature plot for one fitted target.

    A clear residual trend with temperature would indicate that the current
    temperature dependence is not capturing the data well.
    """

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["T"], df[residual_col])
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Temperature / K")
    ax.set_ylabel("Residual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# ============================================================
# 10. Reproducible output files
# ============================================================

def save_outputs(pred_df, metrics, params_report, result, de_result):
    """Write tables and plots that define this baseline run."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pred_df.to_excel(PREDICTION_PATH, index=False)

    parity_plot(
        pred_df["rCH3OH"],
        pred_df["rCH3OH_pred"],
        "Parity plot for CH3OH",
        OUTPUT_DIR / "parity_CH3OH_baseline.png",
    )
    parity_plot(
        pred_df["rCO"],
        pred_df["rCO_pred"],
        "Parity plot for CO",
        OUTPUT_DIR / "parity_CO_baseline.png",
    )
    residual_vs_T_plot(
        pred_df,
        "res_CH3OH",
        "CH3OH residual vs temperature",
        OUTPUT_DIR / "residual_vs_T_CH3OH_baseline.png",
    )
    residual_vs_T_plot(
        pred_df,
        "res_CO",
        "CO residual vs temperature",
        OUTPUT_DIR / "residual_vs_T_CO_baseline.png",
    )

    log_row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "data_file": str(DATA_PATH.relative_to(BASE_DIR)),
        "n_points": len(pred_df),
        "random_seed": RANDOM_SEED,
        "least_squares_success": result.success,
        "least_squares_message": result.message,
        "de_objective": de_result.fun,
        "ls_objective": np.sum(result.fun ** 2),
        **metrics,
        **params_report,
    }

    log_df = pd.DataFrame([log_row])
    if EXPERIMENT_LOG_PATH.exists():
        previous_log = pd.read_csv(EXPERIMENT_LOG_PATH)
        log_df = pd.concat([previous_log, log_df], ignore_index=True)
    log_df.to_csv(EXPERIMENT_LOG_PATH, index=False)


# ============================================================
# 11. Main baseline run
# ============================================================

def main():
    """Run the current two-reaction AAA_LHHW baseline from data to saved outputs."""

    df = load_data(DATA_PATH, sheet_name=0)
    print(f"Loaded data points: {len(df)}")
    print(f"Fitting targets: rCH3OH, rCO")
    print("Excluded from objective: rCO2")

    result, de_result, _ = fit_lhhw_two_reactions(df)
    pred_df, metrics, params_report = evaluate_fit(df, result)
    save_outputs(pred_df, metrics, params_report, result, de_result)

    print("\n==============================")
    print("Fit status")
    print("==============================")
    print("least_squares success:", result.success)
    print("least_squares message:", result.message)
    print(f"DE objective: {de_result.fun:.12g}")
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
    print(EXPERIMENT_LOG_PATH.relative_to(BASE_DIR))
    print(PREDICTION_PATH.relative_to(BASE_DIR))
    print((OUTPUT_DIR / "parity_CH3OH_baseline.png").relative_to(BASE_DIR))
    print((OUTPUT_DIR / "parity_CO_baseline.png").relative_to(BASE_DIR))
    print((OUTPUT_DIR / "residual_vs_T_CH3OH_baseline.png").relative_to(BASE_DIR))
    print((OUTPUT_DIR / "residual_vs_T_CO_baseline.png").relative_to(BASE_DIR))


if __name__ == "__main__":
    main()
