import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
EXCEL_FILE = BASE_DIR / "full data.xlsx"
OUT_DIR = BASE_DIR / "temp_window_fit_12000_Tave_scaled"
OUT_DIR.mkdir(exist_ok=True)

R = 8.314
eps = 1e-12

PARAM_NAMES = [
    "ln_k1_ref", "E1", "n1_A", "n1_B", "n1_C", "n1_D",
    "ln_k2_ref", "E2", "n2_A", "n2_B", "n2_D", "n2_E"
]

# 真实参数边界
REAL_BOUNDS = [
    (-20.0, 10.0),        # ln_k1_ref
    (1e-6, 150000.0),     # E1
    (0.0, 5.0),           # n1_A
    (0.0, 5.0),           # n1_B
    (0.0, 5.0),           # n1_C
    (0.0, 5.0),           # n1_D
    (-20.0, 10.0),        # ln_k2_ref
    (1e-6, 150000.0),     # E2
    (0.0, 5.0),           # n2_A
    (-3, 5.0),           # n2_B
    (0.0, 5.0),           # n2_D
    (0.0, 5.0)            # n2_E
]

# 优化器看到的边界：全部统一成 0 到 1
OPT_BOUNDS = [(0.0, 1.0)] * len(PARAM_NAMES)


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
    df["T_round"] = df["T"].round(2)

    print("Excel 路径:", EXCEL_FILE)
    print("总数据点数:", len(df))
    print("GHSV 分组:", sorted(df["GHSV"].unique()))
    return df


def keq(T):
    K1 = np.exp(
        1.6654 + 4553.34 / T - 2.72613 * np.log(T)
        - 1.422914e-2 * T + 0.172060e-4 * T**2
        - 1.106294e-8 * T**3 + 0.319698e-11 * T**4
    ) * (0.101325) ** (-2)

    K2 = np.exp(
        -11.4998 - 4649.92 / T + 3.2066 * np.log(T)
        - 0.0107251 * T + 0.697955e-5 * T**2
        - 0.336848e-8 * T**3 + 0.811184e-12 * T**4
    )
    return K1, K2


def scale_to_real(u, real_bounds):
    params = []
    for ui, (low, high) in zip(u, real_bounds):
        value = low + (high - low) * ui
        params.append(value)
    return np.array(params, dtype=float)


def real_to_scale(params, real_bounds):
    u = []
    for value, (low, high) in zip(params, real_bounds):
        ui = (value - low) / (high - low)
        u.append(ui)
    return np.array(u, dtype=float)


def decode_params(u):
    return scale_to_real(u, REAL_BOUNDS)


def predict(u, fuga, T, Tave):
    params = decode_params(u)
    ln_k1_ref, E1, a1, b1, c1, d1, ln_k2_ref, E2, a2, b2, d2, e2 = params

    fA = np.maximum(fuga[:, 0], eps)   # CO2
    fB = np.maximum(fuga[:, 1], eps)   # H2
    fC = np.maximum(fuga[:, 2], eps)   # CH3OH
    fD = np.maximum(fuga[:, 3], eps)   # H2O
    fE = np.maximum(fuga[:, 4], eps)   # CO

    k1 = np.exp(ln_k1_ref - (E1 / R) * (1.0 / T - 1.0 / Tave))
    k2 = np.exp(ln_k2_ref - (E2 / R) * (1.0 / T - 1.0 / Tave))

    K1, K2 = keq(T)
    K1 = np.maximum(K1, eps)
    K2 = np.maximum(K2, eps)

    r1 = k1 * fA**a1 * fB**b1 - (k1 / K1) * fC**c1 * fD**d1
    r2 = k2 * fA**a2 * fB**b2 - (k2 / K2) * fD**d2 * fE**e2

    return r1, r2


def objective(u, fuga, T, rMeOH, rCO, Tave):
    try:
        y1, y2 = predict(u, fuga, T, Tave)
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


def get_upper_bound_flags(params, bounds, param_names):
    flags = {}
    for name, value, (low, high) in zip(param_names, params, bounds):
        _, near_high = near_bound(value, low, high)
        flags[f"{name}_near_upper_bound"] = near_high
    return flags


def pack_result(window_info, dfc, res, params, Tave, metrics, extra_params, df_out):
    meta = {
        "window_label": window_info["label"],
        "window_size": window_info["size"],
        "temps": ", ".join(f"{t:.2f}" for t in window_info["temps"]),
        "T_min": window_info["T_min"],
        "T_max": window_info["T_max"],
        "n_points": len(dfc),
        "Tave": Tave,
        "optimizer_success": res.success,
        "optimizer_message": str(res.message),
        "objective": res.fun,
    }

    param_dict = dict(zip(PARAM_NAMES, params))
    bound_flags = get_upper_bound_flags(params, REAL_BOUNDS, PARAM_NAMES)

    return {
        **meta,
        **metrics,
        **param_dict,
        **extra_params,
        **bound_flags,
        "df": df_out
    }


def build_temp_windows(temps):
    windows = []
    for size in [2, 3, 4, 5]:
        for i in range(len(temps) - size + 1):
            group = temps[i:i + size]
            label = f"{group[0]:.2f}_{group[-1]:.2f}_n{size}"
            windows.append({
                "size": size,
                "temps": group,
                "label": label,
                "T_min": group[0],
                "T_max": group[-1]
            })
    return windows


def fit_one_window(dfc, window_info):
    fuga = dfc[["fCO2", "fH2", "fCH3OH", "fH2O", "fCO"]].to_numpy(dtype=float)
    T = dfc["T"].to_numpy(dtype=float)
    rMeOH = dfc["rMeOH"].to_numpy(dtype=float)
    rCO = dfc["rCO"].to_numpy(dtype=float)

    Tave = np.mean(T)

    res = differential_evolution(
        objective,
        bounds=OPT_BOUNDS,
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

    u_opt = res.x
    params = decode_params(u_opt)
    y1, y2 = predict(u_opt, fuga, T, Tave)

    ln_k1_ref, E1 = params[0], params[1]
    ln_k2_ref, E2 = params[6], params[7]

    lnA1 = ln_k1_ref + E1 / (R * Tave)
    lnA2 = ln_k2_ref + E2 / (R * Tave)

    A1_backcalc = np.exp(np.clip(lnA1, -700, 700))
    A2_backcalc = np.exp(np.clip(lnA2, -700, 700))

    r2_meoh = calc_r2(rMeOH, y1)
    r2_co = calc_r2(rCO, y2)

    metrics = {
        "r2_meoh": r2_meoh,
        "r2_co": r2_co,
        "avg_r2": np.nanmean([r2_meoh, r2_co]),
        "rmse_meoh": calc_rmse(rMeOH, y1),
        "rmse_co": calc_rmse(rCO, y2),
        "mre_meoh_%": calc_mre(rMeOH, y1),
        "mre_co_%": calc_mre(rCO, y2),
    }

    extra_params = {
        "lnA1_backcalc": lnA1,
        "lnA2_backcalc": lnA2,
        "A1_backcalc": A1_backcalc,
        "A2_backcalc": A2_backcalc,
    }

    df_out = dfc.copy()
    df_out["window_label"] = window_info["label"]
    df_out["window_size"] = window_info["size"]
    df_out["T_window_min"] = window_info["T_min"]
    df_out["T_window_max"] = window_info["T_max"]
    df_out["Tave"] = Tave
    df_out["rMeOH_pred"] = y1
    df_out["rCO_pred"] = y2
    df_out["rel_error_rMeOH_%"] = np.abs((y1 - rMeOH) / np.maximum(np.abs(rMeOH), 1e-12)) * 100
    df_out["rel_error_rCO_%"] = np.abs((y2 - rCO) / np.maximum(np.abs(rCO), 1e-12)) * 100

    label = window_info["label"]
    df_out.to_excel(OUT_DIR / f"fit_results_{label}.xlsx", index=False)

    # 同时保存归一化变量和真实参数
    param_df = pd.DataFrame({
        "param_name": PARAM_NAMES,
        "u_opt": u_opt,
        "value": params
    })
    param_df.loc[len(param_df)] = ["Tave", np.nan, Tave]
    param_df.loc[len(param_df)] = ["lnA1_backcalc", np.nan, lnA1]
    param_df.loc[len(param_df)] = ["lnA2_backcalc", np.nan, lnA2]
    param_df.loc[len(param_df)] = ["A1_backcalc", np.nan, A1_backcalc]
    param_df.loc[len(param_df)] = ["A2_backcalc", np.nan, A2_backcalc]
    param_df.to_excel(OUT_DIR / f"fitted_parameters_{label}.xlsx", index=False)

    print(f"\n窗口 = {label}")
    print("温度 =", window_info["temps"])
    print("数据点数 =", len(dfc))
    print("success =", res.success)
    print("message =", res.message)
    print("objective =", res.fun)
    print("Tave =", Tave)
    print("r2_meoh =", metrics["r2_meoh"])
    print("r2_co =", metrics["r2_co"])
    print("E1 =", params[1])
    print("E2 =", params[7])

    return pack_result(
        window_info=window_info,
        dfc=dfc,
        res=res,
        params=params,
        Tave=Tave,
        metrics=metrics,
        extra_params=extra_params,
        df_out=df_out
    )


def plot_parity(df_all, exp_col, pred_col, title, save_name):
    plt.figure(figsize=(8, 8))

    for label in df_all["window_label"].unique():
        part = df_all[df_all["window_label"] == label]
        plt.scatter(part[exp_col], part[pred_col], label=label, alpha=0.7)

    min_val = min(df_all[exp_col].min(), df_all[pred_col].min())
    max_val = max(df_all[exp_col].max(), df_all[pred_col].max())

    plt.plot([min_val, max_val], [min_val, max_val], "k--")
    plt.xlabel(f"Experimental {exp_col}")
    plt.ylabel(f"Predicted {pred_col}")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / save_name, dpi=300)
    plt.show()


def main():
    df = load_data()

    df_12000 = df[df["GHSV"] == 12000].copy()
    if df_12000.empty:
        raise ValueError("没有找到 GHSV = 12000 的数据")

    unique_temps = sorted(df_12000["T_round"].unique())
    print("\nGHSV = 12000 的温度点:", unique_temps)

    windows = build_temp_windows(unique_temps)
    print("\n将进行以下温度窗口拟合:")
    for w in windows:
        print(w["label"], "->", w["temps"])

    results = []
    all_df = []

    for w in windows:
        df_window = df_12000[df_12000["T_round"].isin(w["temps"])].copy()
        res = fit_one_window(df_window, w)

        result_row = res.copy()
        all_df.append(result_row.pop("df"))
        results.append(result_row)

    summary = pd.DataFrame(results)
    summary.to_excel(OUT_DIR / "comparison_summary_temp_windows_12000.xlsx", index=False)

    df_all = pd.concat(all_df, ignore_index=True)
    df_all.to_excel(OUT_DIR / "all_window_predictions_12000.xlsx", index=False)

    print("\n全部完成")
    print(summary[[
        "window_label", "window_size", "temps", "n_points",
        "objective", "r2_meoh", "r2_co", "avg_r2"
    ]])

    plot_parity(
        df_all=df_all,
        exp_col="rMeOH",
        pred_col="rMeOH_pred",
        title="MeOH Parity Plot, GHSV=12000, Different Temperature Windows",
        save_name="parity_plot_MeOH_12000_temp_windows.png"
    )

    plot_parity(
        df_all=df_all,
        exp_col="rCO",
        pred_col="rCO_pred",
        title="CO Parity Plot, GHSV=12000, Different Temperature Windows",
        save_name="parity_plot_CO_12000_temp_windows.png"
    )


if __name__ == "__main__":
    main()