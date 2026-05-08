import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.optimize import differential_evolution, least_squares


# ============================================================
# 1. Settings
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Xlessthan20.xlsx"

OUTPUT_DIR = BASE_DIR / "GHSV12000_clean_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

SHEET_NAME = 0
TARGET_GHSV = 12000

R = 8.314462618
EPS = 1e-30
DRIVE_EPS = 1e-8

MAX_XCO2 = 0.20
# MAX_XCO2 = None

MAXITER_DE = 1000
POPSIZE_DE = 15

MODEL_LIST = [
    "simple_powerlaw",
    "one_minus_beta",
    "rf_minus_rr"
]


# ============================================================
# 2. Load data
# ============================================================

def load_data(path, sheet_name=0):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Cannot find file: {path.resolve()}")

    df = pd.read_excel(path, sheet_name=sheet_name, header=0, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    first_col = df.columns[0]
    if first_col.startswith("Unnamed") or first_col == "" or first_col.lower() == "nan":
        df = df.rename(columns={first_col: "HC"})

    df.columns = [str(c).strip() for c in df.columns]

    required_cols = [
        "HC", "p", "GHSV", "T", "XCO2",
        "fCO2", "fH2", "fCH3OH", "fH2O", "fCO",
        "KfA", "KfB",
        "rCH3OH", "rCO"
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in Excel: {missing}")

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required_cols).copy()

    if MAX_XCO2 is not None:
        df = df[df["XCO2"] <= MAX_XCO2].copy()

    df = df[np.isclose(df["GHSV"], TARGET_GHSV)].copy()
    df = df.reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No data left after filtering GHSV = 12000.")

    print("Data points for GHSV 12000:", len(df))
    print("Tave =", df["T"].mean())

    return df


# ============================================================
# 3. Basic functions
# ============================================================

def safe_power(x, n):
    return np.maximum(np.asarray(x, dtype=float), EPS) ** n


def calc_k(ln_k_ref, E, T, Tave):
    """
    ln k = ln_k_ref - E / R * (1 / T - 1 / Tave)
    """
    return np.exp(ln_k_ref - E / R * (1.0 / T - 1.0 / Tave))


# ============================================================
# 4. Models
# ============================================================

def get_model_info(model):
    if model in ["simple_powerlaw", "one_minus_beta"]:
        names = [
            "ln_k1_ref", "E1_J_per_mol", "n1_CO2", "n1_H2",
            "ln_k2_ref", "E2_J_per_mol", "n2_CO2", "n2_H2"
        ]

        bounds = [
            (-30, 10), (-100000, 200000), (-3, 5), (-3, 5),
            (-30, 10), (-100000, 200000), (-3, 5), (-3, 5)
        ]

    elif model == "rf_minus_rr":
        names = [
            "ln_k1f_ref", "E1f_J_per_mol", "n1f_CO2", "n1f_H2",
            "ln_k1r_ref", "E1r_J_per_mol", "n1r_CH3OH", "n1r_H2O",
            "ln_k2f_ref", "E2f_J_per_mol", "n2f_CO2", "n2f_H2",
            "ln_k2r_ref", "E2r_J_per_mol", "n2r_CO", "n2r_H2O"
        ]

        bounds = [
            (-30, 10), (-100000, 200000), (-3, 5), (-3, 5),
            (-30, 10), (-100000, 200000), (-3, 5), (-3, 5),
            (-30, 10), (-100000, 200000), (-3, 5), (-3, 5),
            (-30, 10), (-100000, 200000), (-3, 5), (-3, 5)
        ]

    else:
        raise ValueError(f"Unknown model: {model}")

    return names, bounds


def predict_rates(model, params, df, Tave):
    T = df["T"].values

    fCO2 = df["fCO2"].values
    fH2 = df["fH2"].values
    fCH3OH = df["fCH3OH"].values
    fH2O = df["fH2O"].values
    fCO = df["fCO"].values

    if model == "simple_powerlaw":
        ln_k1, E1, n1_CO2, n1_H2, ln_k2, E2, n2_CO2, n2_H2 = params

        k1 = calc_k(ln_k1, E1, T, Tave)
        k2 = calc_k(ln_k2, E2, T, Tave)

        r_meoh = k1 * safe_power(fCO2, n1_CO2) * safe_power(fH2, n1_H2)
        r_co = k2 * safe_power(fCO2, n2_CO2) * safe_power(fH2, n2_H2)

    elif model == "one_minus_beta":
        ln_k1, E1, n1_CO2, n1_H2, ln_k2, E2, n2_CO2, n2_H2 = params

        k1 = calc_k(ln_k1, E1, T, Tave)
        k2 = calc_k(ln_k2, E2, T, Tave)

        KfA = np.maximum(df["KfA"].values, EPS)
        KfB = np.maximum(df["KfB"].values, EPS)

        Q1 = np.maximum(fCH3OH, EPS) * np.maximum(fH2O, EPS) / (
            np.maximum(fCO2, EPS) * np.maximum(fH2, EPS) ** 3
        )
        beta1 = Q1 / KfA
        drive1 = np.maximum(1.0 - beta1, DRIVE_EPS)

        Q2 = np.maximum(fCO, EPS) * np.maximum(fH2O, EPS) / (
            np.maximum(fCO2, EPS) * np.maximum(fH2, EPS)
        )
        beta2 = Q2 / KfB
        drive2 = np.maximum(1.0 - beta2, DRIVE_EPS)

        r_meoh = k1 * safe_power(fCO2, n1_CO2) * safe_power(fH2, n1_H2) * drive1
        r_co = k2 * safe_power(fCO2, n2_CO2) * safe_power(fH2, n2_H2) * drive2

    elif model == "rf_minus_rr":
        (
            ln_k1f, E1f, n1f_CO2, n1f_H2,
            ln_k1r, E1r, n1r_CH3OH, n1r_H2O,
            ln_k2f, E2f, n2f_CO2, n2f_H2,
            ln_k2r, E2r, n2r_CO, n2r_H2O
        ) = params

        k1f = calc_k(ln_k1f, E1f, T, Tave)
        k1r = calc_k(ln_k1r, E1r, T, Tave)
        k2f = calc_k(ln_k2f, E2f, T, Tave)
        k2r = calc_k(ln_k2r, E2r, T, Tave)

        r1f = k1f * safe_power(fCO2, n1f_CO2) * safe_power(fH2, n1f_H2)
        r1r = k1r * safe_power(fCH3OH, n1r_CH3OH) * safe_power(fH2O, n1r_H2O)

        r2f = k2f * safe_power(fCO2, n2f_CO2) * safe_power(fH2, n2f_H2)
        r2r = k2r * safe_power(fCO, n2r_CO) * safe_power(fH2O, n2r_H2O)

        r_meoh = r1f - r1r
        r_co = r2f - r2r

    else:
        raise ValueError(f"Unknown model: {model}")

    return r_meoh, r_co


# ============================================================
# 5. Objective
# ============================================================

def residuals(params, model, df, Tave):
    r_meoh_pred, r_co_pred = predict_rates(model, params, df, Tave)

    r_meoh_exp = np.maximum(df["rCH3OH"].values, EPS)
    r_co_exp = np.maximum(df["rCO"].values, EPS)

    res_meoh = (r_meoh_pred - r_meoh_exp) / r_meoh_exp
    res_co = (r_co_pred - r_co_exp) / r_co_exp

    return np.concatenate([res_meoh, res_co])


def objective(params, model, df, Tave):
    res = residuals(params, model, df, Tave)

    if not np.all(np.isfinite(res)):
        return 1e100

    return np.sum(res ** 2)


# ============================================================
# 6. Metrics
# ============================================================

def calc_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return np.nan

    return 1.0 - ss_res / ss_tot


def calc_metrics(df):
    rows = []

    for name, exp_col, pred_col in [
        ("CH3OH", "rCH3OH", "rCH3OH_pred"),
        ("CO", "rCO", "rCO_pred")
    ]:
        y_true = df[exp_col].values
        y_pred = df[pred_col].values

        rows.append({
            "species": name,
            "R2": calc_r2(y_true, y_pred),
            "RMSE": np.sqrt(np.mean((y_pred - y_true) ** 2)),
            "MRE_%": np.mean(np.abs((y_pred - y_true) / y_true)) * 100
        })

    return pd.DataFrame(rows)


# ============================================================
# 7. Fitting
# ============================================================

def fit_model(df, model):
    Tave = df["T"].mean()
    names, bounds = get_model_info(model)

    print("\n" + "=" * 80)
    print("Model:", model)
    print("Tave:", Tave)
    print("Data points:", len(df))
    print("Parameters:", len(names))

    de_result = differential_evolution(
        objective,
        bounds=bounds,
        args=(model, df, Tave),
        maxiter=MAXITER_DE,
        popsize=POPSIZE_DE,
        tol=1e-8,
        polish=False,
        seed=42,
        workers=1
    )

    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])

    ls_result = least_squares(
        residuals,
        x0=de_result.x,
        bounds=(lower, upper),
        args=(model, df, Tave),
        max_nfev=20000
    )

    params = ls_result.x

    r_meoh_pred, r_co_pred = predict_rates(model, params, df, Tave)

    fit_df = df.copy()
    fit_df["model"] = model
    fit_df["Tave"] = Tave
    fit_df["rCH3OH_pred"] = r_meoh_pred
    fit_df["rCO_pred"] = r_co_pred
    fit_df["err_CH3OH_%"] = (fit_df["rCH3OH_pred"] - fit_df["rCH3OH"]) / fit_df["rCH3OH"] * 100
    fit_df["err_CO_%"] = (fit_df["rCO_pred"] - fit_df["rCO"]) / fit_df["rCO"] * 100

    metric_df = calc_metrics(fit_df)
    metric_df.insert(0, "model", model)

    param_record = {
        "model": model,
        "GHSV": TARGET_GHSV,
        "Tave": Tave,
        "n_data": len(df),
        "DE_objective": de_result.fun,
        "LS_cost": ls_result.cost
    }

    for name, value in zip(names, params):
        param_record[name] = value

        if name.startswith("ln_k"):
            param_record[name.replace("ln_", "k_")] = np.exp(value)

        if name.endswith("_J_per_mol"):
            param_record[name.replace("_J_per_mol", "_kJ_per_mol")] = value / 1000

    param_df = pd.DataFrame([param_record])

    print(metric_df.to_string(index=False))

    return param_df, metric_df, fit_df


# ============================================================
# 8. Plot
# ============================================================

def plot_parity(all_fit_df):
    for species, exp_col, pred_col in [
        ("CH3OH", "rCH3OH", "rCH3OH_pred"),
        ("CO", "rCO", "rCO_pred")
    ]:
        plt.figure(figsize=(5.5, 5.5))

        y_exp_all = all_fit_df[exp_col].values
        y_pred_all = all_fit_df[pred_col].values

        vmin = min(np.min(y_exp_all), np.min(y_pred_all))
        vmax = max(np.max(y_exp_all), np.max(y_pred_all))

        for model in MODEL_LIST:
            temp = all_fit_df[all_fit_df["model"] == model]
            plt.scatter(temp[exp_col], temp[pred_col], label=model, s=45)

        plt.plot([vmin, vmax], [vmin, vmax], "k--")
        plt.xlabel(f"Experimental r{species}")
        plt.ylabel(f"Predicted r{species}")
        plt.title(f"GHSV {TARGET_GHSV} {species}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"parity_{species}.png", dpi=300)
        plt.show()


# ============================================================
# 9. Main
# ============================================================

df = load_data(DATA_PATH, SHEET_NAME)

all_params = []
all_metrics = []
all_fits = []

for model in MODEL_LIST:
    param_df, metric_df, fit_df = fit_model(df, model)

    all_params.append(param_df)
    all_metrics.append(metric_df)
    all_fits.append(fit_df)

final_param_df = pd.concat(all_params, ignore_index=True)
final_metric_df = pd.concat(all_metrics, ignore_index=True)
final_fit_df = pd.concat(all_fits, ignore_index=True)

output_excel = OUTPUT_DIR / "GHSV12000_clean_results.xlsx"

with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
    final_param_df.to_excel(writer, sheet_name="parameters", index=False)
    final_metric_df.to_excel(writer, sheet_name="metrics", index=False)
    final_fit_df.to_excel(writer, sheet_name="fit_results", index=False)

plot_parity(final_fit_df)

print("\nDone.")
print("Results saved to:")
print(output_excel)
print("Figures saved to:")
print(OUTPUT_DIR)