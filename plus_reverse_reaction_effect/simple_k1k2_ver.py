import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
EXCEL_FILE = BASE_DIR / "full data.xlsx"
OUT_DIR = BASE_DIR / "ghsv_fit_results_independent_reverse"
OUT_DIR.mkdir(exist_ok=True)

R = 8.314
eps = 1e-12


def load_data():
    df = pd.read_excel(EXCEL_FILE, skiprows=[1])
    df.columns = df.columns.str.strip()

    df = df.rename(columns={
        "r CH3OH": "rMeOH",
        "r CO": "rCO",
        "r CO2": "rCO2"
    })

    required_cols = [
        "H/C", "p", "GHSV", "T",
        "fCO2", "fH2", "fCH3OH", "fH2O", "fCO",
        "rMeOH", "rCO"
    ]

    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"缺少列: {col}")

    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required_cols).copy()

    print("Excel 路径:", EXCEL_FILE)
    print("总数据点数:", len(df))
    print("GHSV 分组:", sorted(df["GHSV"].unique()))
    return df


def predict(par, fuga, T, Tave):
    (
        ln_k1f_ref, E1f, a1, b1,
        ln_k1r_ref, E1r, c1, d1,
        ln_k2f_ref, E2f, a2, b2,
        ln_k2r_ref, E2r, d2, e2
    ) = par

    fA = np.maximum(fuga[:, 0], eps)   # CO2
    fB = np.maximum(fuga[:, 1], eps)   # H2
    fC = np.maximum(fuga[:, 2], eps)   # CH3OH
    fD = np.maximum(fuga[:, 3], eps)   # H2O
    fE = np.maximum(fuga[:, 4], eps)   # CO

    k1f = np.exp(ln_k1f_ref - (E1f / R) * (1.0 / T - 1.0 / Tave))
    k1r = np.exp(ln_k1r_ref - (E1r / R) * (1.0 / T - 1.0 / Tave))
    k2f = np.exp(ln_k2f_ref - (E2f / R) * (1.0 / T - 1.0 / Tave))
    k2r = np.exp(ln_k2r_ref - (E2r / R) * (1.0 / T - 1.0 / Tave))

    r1 = k1f * fA**a1 * fB**b1 - k1r * fC**c1 * fD**d1
    r2 = k2f * fA**a2 * fB**b2 - k2r * fD**d2 * fE**e2

    return r1, r2


def objective(par, fuga, T, rMeOH, rCO, Tave):
    try:
        y1, y2 = predict(par, fuga, T, Tave)

        err1 = np.sum(((y1 - rMeOH) / np.maximum(np.abs(rMeOH), 1e-6)) ** 2)
        err2 = np.sum(((y2 - rCO) / np.maximum(np.abs(rCO), 1e-6)) ** 2)

        sse = err1 + err2
        return sse if np.isfinite(sse) else 1e30
    except Exception:
        return 1e30


def calc_r2(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot < 1e-12:
        return np.nan
    return 1 - ss_res / ss_tot


def calc_rmse(y, yhat):
    return np.sqrt(np.mean((y - yhat) ** 2))


def calc_mre(y, yhat):
    return np.mean(np.abs((yhat - y) / np.maximum(np.abs(y), 1e-12))) * 100


def near_bound(value, low, high, frac=0.01):
    width = high - low
    if width <= 0:
        return False, False
    return (value <= low + frac * width), (value >= high - frac * width)


def fit_one_cluster(dfc, ghsv):
    fuga = dfc[["fCO2", "fH2", "fCH3OH", "fH2O", "fCO"]].to_numpy(dtype=float)
    T = dfc["T"].to_numpy(dtype=float)
    rMeOH = dfc["rMeOH"].to_numpy(dtype=float)
    rCO = dfc["rCO"].to_numpy(dtype=float)

    Tave = np.mean(T)

    bounds = [
        (-20, 10),      # ln_k1f_ref
        (0, 150000),    # E1f
        (0, 5),         # a1
        (0, 5),         # b1

        (-20, 10),      # ln_k1r_ref
        (0, 150000),    # E1r
        (0, 5),         # c1
        (0, 5),         # d1

        (-20, 10),      # ln_k2f_ref
        (0, 150000),    # E2f
        (0, 5),         # a2
        (0, 5),         # b2

        (-20, 10),      # ln_k2r_ref
        (0, 150000),    # E2r
        (0, 5),         # d2
        (0, 5)          # e2
    ]

    res = differential_evolution(
        objective,
        bounds=bounds,
        args=(fuga, T, rMeOH, rCO, Tave),
        seed=42,
        popsize=15,
        maxiter=1000,
        tol=1e-6,
        mutation=(0.5, 1.0),
        recombination=0.7,
        polish=True,
        workers=1
    )

    p = res.x
    y1, y2 = predict(p, fuga, T, Tave)

    lnA1f = p[0] + p[1] / (R * Tave)
    lnA1r = p[4] + p[5] / (R * Tave)
    lnA2f = p[8] + p[9] / (R * Tave)
    lnA2r = p[12] + p[13] / (R * Tave)

    A1f = np.exp(np.clip(lnA1f, -700, 700))
    A1r = np.exp(np.clip(lnA1r, -700, 700))
    A2f = np.exp(np.clip(lnA2f, -700, 700))
    A2r = np.exp(np.clip(lnA2r, -700, 700))

    r2_meoh = calc_r2(rMeOH, y1)
    r2_co = calc_r2(rCO, y2)
    rmse_meoh = calc_rmse(rMeOH, y1)
    rmse_co = calc_rmse(rCO, y2)
    mre_meoh = calc_mre(rMeOH, y1)
    mre_co = calc_mre(rCO, y2)

    df_out = dfc.copy()
    df_out["Tave"] = Tave
    df_out["rMeOH_pred"] = y1
    df_out["rCO_pred"] = y2
    df_out["rel_error_rMeOH_%"] = np.abs((y1 - rMeOH) / np.maximum(np.abs(rMeOH), 1e-12)) * 100
    df_out["rel_error_rCO_%"] = np.abs((y2 - rCO) / np.maximum(np.abs(rCO), 1e-12)) * 100
    df_out["GHSV_cluster"] = ghsv
    df_out.to_excel(OUT_DIR / f"fit_results_GHSV_{int(ghsv)}.xlsx", index=False)

    param_names = [
        "ln_k1f_ref", "E1f", "n1_A", "n1_B",
        "ln_k1r_ref", "E1r", "n1_C", "n1_D",
        "ln_k2f_ref", "E2f", "n2_A", "n2_B",
        "ln_k2r_ref", "E2r", "n2_D", "n2_E"
    ]

    param_df = pd.DataFrame({
        "param_name": param_names,
        "value": p
    })
    param_df.loc[len(param_df)] = ["Tave", Tave]
    param_df.loc[len(param_df)] = ["lnA1f_backcalc", lnA1f]
    param_df.loc[len(param_df)] = ["lnA1r_backcalc", lnA1r]
    param_df.loc[len(param_df)] = ["lnA2f_backcalc", lnA2f]
    param_df.loc[len(param_df)] = ["lnA2r_backcalc", lnA2r]
    param_df.loc[len(param_df)] = ["A1f_backcalc", A1f]
    param_df.loc[len(param_df)] = ["A1r_backcalc", A1r]
    param_df.loc[len(param_df)] = ["A2f_backcalc", A2f]
    param_df.loc[len(param_df)] = ["A2r_backcalc", A2r]
    param_df.to_excel(OUT_DIR / f"fitted_parameters_GHSV_{int(ghsv)}.xlsx", index=False)

    lower_flags = []
    upper_flags = []
    for val, (low, high) in zip(p, bounds):
        near_low, near_high = near_bound(val, low, high)
        lower_flags.append(near_low)
        upper_flags.append(near_high)

    print(f"\nGHSV = {ghsv}")
    print("success =", res.success)
    print("message =", res.message)
    print("objective =", res.fun)
    print("Tave =", Tave)
    print("r2_meoh =", r2_meoh)
    print("r2_co =", r2_co)

    return {
        "GHSV": ghsv,
        "n_points": len(dfc),
        "Tave": Tave,
        "optimizer_success": res.success,
        "optimizer_message": str(res.message),
        "objective": res.fun,
        "r2_meoh": r2_meoh,
        "r2_co": r2_co,
        "avg_r2": np.nanmean([r2_meoh, r2_co]),
        "rmse_meoh": rmse_meoh,
        "rmse_co": rmse_co,
        "mre_meoh_%": mre_meoh,
        "mre_co_%": mre_co,

        "ln_k1f_ref": p[0],
        "E1f": p[1],
        "n1_A": p[2],
        "n1_B": p[3],

        "ln_k1r_ref": p[4],
        "E1r": p[5],
        "n1_C": p[6],
        "n1_D": p[7],

        "ln_k2f_ref": p[8],
        "E2f": p[9],
        "n2_A": p[10],
        "n2_B": p[11],

        "ln_k2r_ref": p[12],
        "E2r": p[13],
        "n2_D": p[14],
        "n2_E": p[15],

        "lnA1f_backcalc": lnA1f,
        "lnA1r_backcalc": lnA1r,
        "lnA2f_backcalc": lnA2f,
        "lnA2r_backcalc": lnA2r,
        "A1f_backcalc": A1f,
        "A1r_backcalc": A1r,
        "A2f_backcalc": A2f,
        "A2r_backcalc": A2r,

        "df": df_out
    }


def plot_parity(df_all, exp_col, pred_col, title, save_name):
    plt.figure(figsize=(6, 6))

    for ghsv in sorted(df_all["GHSV_cluster"].unique()):
        part = df_all[df_all["GHSV_cluster"] == ghsv]
        plt.scatter(part[exp_col], part[pred_col], label=f"GHSV={int(ghsv)}", alpha=0.7)

    min_val = min(df_all[exp_col].min(), df_all[pred_col].min())
    max_val = max(df_all[exp_col].max(), df_all[pred_col].max())

    plt.plot([min_val, max_val], [min_val, max_val], "k--")
    plt.xlabel(f"Experimental {exp_col}")
    plt.ylabel(f"Predicted {pred_col}")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / save_name, dpi=300)
    plt.show()


def main():
    df = load_data()

    results = []
    all_df = []

    for ghsv in sorted(df["GHSV"].unique()):
        dfc = df[df["GHSV"] == ghsv].copy()
        res = fit_one_cluster(dfc, ghsv)
        results.append({k: v for k, v in res.items() if k != "df"})
        all_df.append(res["df"])

    summary = pd.DataFrame(results)
    summary.to_excel(OUT_DIR / "comparison_summary_by_GHSV.xlsx", index=False)

    df_all = pd.concat(all_df, ignore_index=True)
    df_all.to_excel(OUT_DIR / "all_cluster_predictions.xlsx", index=False)

    print("\n全部完成")
    print(summary)

    plot_parity(
        df_all=df_all,
        exp_col="rMeOH",
        pred_col="rMeOH_pred",
        title="MeOH Parity Plot by GHSV with Independent Reverse Rates",
        save_name="parity_plot_MeOH_by_GHSV_independent_reverse.png"
    )

    plot_parity(
        df_all=df_all,
        exp_col="rCO",
        pred_col="rCO_pred",
        title="CO Parity Plot by GHSV with Independent Reverse Rates",
        save_name="parity_plot_CO_by_GHSV_independent_reverse.png"
    )


if __name__ == "__main__":
    main()