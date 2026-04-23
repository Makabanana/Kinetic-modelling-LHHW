#全部数据参与拟合，对比原始的powerlaw和加入(1-beta)的两种形式
import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import differential_evolution, least_squares

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =========================
# 0. 常数与全局设置
# =========================
R = 8.314
eps = 1e-12

EXCEL_PATH = "full data.xlsx"
SKIPROWS = [1]   # 空出单位行再阅读

OUTPUT_DIR = "fit_fixed_GHSV_P_outputs"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# 至少多少个点才允许拟合
MIN_POINTS_PER_GROUP = 8

# 是否保存每个组的 parity plot
SAVE_PLOTS = True

# =========================
# 1. 优化器参数
# 这里就是你要“调大次数”的地方
# =========================
DE_CONFIG = {
    "strategy": "best1bin",
    "maxiter": 3000,          # 现在调大
    "popsize": 25,            # 现在调大
    "tol": 1e-7,              # 更严格
    "mutation": (0.5, 1.2),
    "recombination": 0.7,
    "seed": 42,
    "polish": False,          # 先不让 DE 自己 polish，后面单独 least_squares 精修
    "workers": 1,             # Windows 下更稳
    "updating": "immediate",
    "init": "latinhypercube",
    "disp": False
}

LS_CONFIG = {
    "method": "trf",
    "ftol": 1e-10,
    "xtol": 1e-10,
    "gtol": 1e-10,
    "max_nfev": 20000,        # 局部精修次数也调大
    "verbose": 0
}

# 参数边界
# A, E, n
BOUNDS = [
    (1e-8, 1e5),   (-30000, 50000),   (-2, 5),   (-2, 5),
    (1e-8, 1e5),   (-30000, 50000),   (-2, 5),   (-2, 5)
]
LOWER_BOUNDS = np.array([b[0] for b in BOUNDS], dtype=float)
UPPER_BOUNDS = np.array([b[1] for b in BOUNDS], dtype=float)

PARAM_NAMES = ["A1", "E1", "n1_A", "n1_B", "A2", "E2", "n2_A", "n2_B"]


# =========================
# 2. 工具函数
# =========================
def safe_filename(s):
    s = str(s)
    s = re.sub(r"[\\/*?:\"<>|]", "_", s)
    s = s.replace(" ", "_")
    return s

def calc_r2(y_exp, y_pred):
    y_exp = np.asarray(y_exp, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_exp - y_pred) ** 2)
    ss_tot = np.sum((y_exp - np.mean(y_exp)) ** 2)
    if ss_tot < 1e-12:
        return np.nan
    return 1 - ss_res / ss_tot

def calc_rmse(y_exp, y_pred):
    y_exp = np.asarray(y_exp, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.sqrt(np.mean((y_exp - y_pred) ** 2))

def calc_mre(y_exp, y_pred):
    y_exp = np.asarray(y_exp, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_exp), 1e-12)
    return np.mean(np.abs((y_pred - y_exp) / denom)) * 100

def calculate_equilibrium_constants(T):
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


# =========================
# 3. 读取数据
# =========================
def load_data():
    if SKIPROWS is None:
        df = pd.read_excel(EXCEL_PATH)
    else:
        df = pd.read_excel(EXCEL_PATH, skiprows=SKIPROWS)

    df.columns = df.columns.str.strip()

    rename_map = {
        "r CH3OH": "rMeOH",
        "r CO": "rCO",
        "r CO2": "rCO2"
    }
    df = df.rename(columns=rename_map)

    print("实际列名如下：")
    print(df.columns.tolist())

    required_cols = [
        "p", "GHSV", "T",
        "fCO2", "fH2", "fCH3OH", "fH2O", "fCO",
        "rMeOH", "rCO"
    ]

    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"缺少列: {col}")

    # 转成数值
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # H/C 如果有，也转
    if "H/C" in df.columns:
        df["H/C"] = pd.to_numeric(df["H/C"], errors="coerce")

    before_drop = len(df)
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    after_drop = len(df)

    print(f"原始总数据点数: {before_drop}")
    print(f"清洗后数据点数: {after_drop}")

    print("GHSV 分组:", sorted(df["GHSV"].unique().tolist()))
    print("总压 p 分组:", sorted(df["p"].unique().tolist()))
    print("温度点:", sorted(df["T"].unique().tolist()))

    return df


# =========================
# 4. 模型表达式
# =========================
def calc_predictions_simple(par, fuga, temperature):
    A1, E1, n1_A, n1_B, A2, E2, n2_A, n2_B = par

    fCO2 = np.maximum(fuga[:, 0], eps)
    fH2 = np.maximum(fuga[:, 1], eps)

    k1 = A1 * np.exp(-E1 / (R * temperature))
    k2 = A2 * np.exp(-E2 / (R * temperature))

    r1 = k1 * (fCO2 ** n1_A) * (fH2 ** n1_B)
    r2 = k2 * (fCO2 ** n2_A) * (fH2 ** n2_B)

    rate_meoh_pred = r1
    rate_co_pred = r2

    return rate_meoh_pred, rate_co_pred, r1, r2, k1, k2


def calc_predictions_beta(par, fuga, temperature):
    A1, E1, n1_A, n1_B, A2, E2, n2_A, n2_B = par

    fCO2 = np.maximum(fuga[:, 0], eps)
    fH2 = np.maximum(fuga[:, 1], eps)
    fCH3OH = np.maximum(fuga[:, 2], eps)
    fH2O = np.maximum(fuga[:, 3], eps)
    fCO = np.maximum(fuga[:, 4], eps)

    k1 = A1 * np.exp(-E1 / (R * temperature))
    k2 = A2 * np.exp(-E2 / (R * temperature))

    K_f1, K_f2 = calculate_equilibrium_constants(temperature)
    K_f1 = np.maximum(K_f1, eps)
    K_f2 = np.maximum(K_f2, eps)

    beta1 = (fCH3OH * fH2O) / np.maximum(K_f1 * fCO2 * (fH2 ** 3), eps)
    beta2 = (fCO * fH2O) / np.maximum(K_f2 * fCO2 * fH2, eps)

    r1 = k1 * (fCO2 ** n1_A) * (fH2 ** n1_B) * (1 - beta1)
    r2 = k2 * (fCO2 ** n2_A) * (fH2 ** n2_B) * (1 - beta2)

    rate_meoh_pred = r1
    rate_co_pred = r2

    return rate_meoh_pred, rate_co_pred, r1, r2, k1, k2


# =========================
# 5. 残差与目标函数
# =========================
def residuals_simple(par, fuga, temperature, rMeOH_exp, rCO_exp):
    rate_meoh_pred, rate_co_pred, _, _, _, _ = calc_predictions_simple(par, fuga, temperature)

    denom_meoh = np.maximum(np.abs(rMeOH_exp), 1e-6)
    denom_co = np.maximum(np.abs(rCO_exp), 1e-6)

    res_meoh = (rate_meoh_pred - rMeOH_exp) / denom_meoh
    res_co = (rate_co_pred - rCO_exp) / denom_co

    res = np.concatenate([res_meoh, res_co])

    if np.any(~np.isfinite(res)):
        return np.full_like(res, 1e15)

    return res


def residuals_beta(par, fuga, temperature, rMeOH_exp, rCO_exp):
    rate_meoh_pred, rate_co_pred, _, _, _, _ = calc_predictions_beta(par, fuga, temperature)

    denom_meoh = np.maximum(np.abs(rMeOH_exp), 1e-6)
    denom_co = np.maximum(np.abs(rCO_exp), 1e-6)

    res_meoh = (rate_meoh_pred - rMeOH_exp) / denom_meoh
    res_co = (rate_co_pred - rCO_exp) / denom_co

    res = np.concatenate([res_meoh, res_co])

    if np.any(~np.isfinite(res)):
        return np.full_like(res, 1e15)

    return res


def objective_from_residuals(res):
    if np.any(~np.isfinite(res)):
        return 1e30
    return float(np.sum(res ** 2))


def objective_simple(par, fuga, temperature, rMeOH_exp, rCO_exp):
    try:
        res = residuals_simple(par, fuga, temperature, rMeOH_exp, rCO_exp)
        return objective_from_residuals(res)
    except Exception as e:
        print("objective_simple 出错:", e)
        return 1e30


def objective_beta(par, fuga, temperature, rMeOH_exp, rCO_exp):
    try:
        res = residuals_beta(par, fuga, temperature, rMeOH_exp, rCO_exp)
        return objective_from_residuals(res)
    except Exception as e:
        print("objective_beta 出错:", e)
        return 1e30


# =========================
# 6. 画图
# =========================
def make_group_parity_plot(
    y_exp_1, y_pred_1, y_exp_2, y_pred_2,
    label_1, label_2,
    xlabel, ylabel, title, save_path
):
    plt.figure(figsize=(6, 6))

    plt.scatter(y_exp_1, y_pred_1, alpha=0.75, label=label_1)
    plt.scatter(y_exp_2, y_pred_2, alpha=0.75, label=label_2)

    all_vals = np.concatenate([
        np.asarray(y_exp_1), np.asarray(y_pred_1),
        np.asarray(y_exp_2), np.asarray(y_pred_2)
    ])
    all_vals = all_vals[np.isfinite(all_vals)]

    if len(all_vals) == 0:
        min_val, max_val = 0, 1
    else:
        min_val = np.min(all_vals)
        max_val = np.max(all_vals)
        if np.isclose(min_val, max_val):
            max_val = min_val + 1e-6

    plt.plot([min_val, max_val], [min_val, max_val], "k--")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# =========================
# 7. 单个模型拟合
# 先 DE，再 least_squares 精修
# =========================
def fit_model(
    model_name,
    objective_func,
    residual_func,
    prediction_func,
    df_group,
    fuga,
    rMeOH,
    rCO,
    temperature,
    ghsv_val,
    p_val
):
    # 第一步：全局优化 DE
    de_result = differential_evolution(
        objective_func,
        bounds=BOUNDS,
        args=(fuga, temperature, rMeOH, rCO),
        strategy=DE_CONFIG["strategy"],
        maxiter=DE_CONFIG["maxiter"],
        popsize=DE_CONFIG["popsize"],
        tol=DE_CONFIG["tol"],
        mutation=DE_CONFIG["mutation"],
        recombination=DE_CONFIG["recombination"],
        seed=DE_CONFIG["seed"],
        polish=DE_CONFIG["polish"],
        workers=DE_CONFIG["workers"],
        updating=DE_CONFIG["updating"],
        init=DE_CONFIG["init"],
        disp=DE_CONFIG["disp"]
    )

    de_x = de_result.x.copy()
    de_obj = float(de_result.fun)

    # 第二步：局部精修 least_squares
    try:
        ls_result = least_squares(
            residual_func,
            x0=de_x,
            bounds=(LOWER_BOUNDS, UPPER_BOUNDS),
            args=(fuga, temperature, rMeOH, rCO),
            method=LS_CONFIG["method"],
            ftol=LS_CONFIG["ftol"],
            xtol=LS_CONFIG["xtol"],
            gtol=LS_CONFIG["gtol"],
            max_nfev=LS_CONFIG["max_nfev"],
            verbose=LS_CONFIG["verbose"]
        )

        ls_x = ls_result.x.copy()
        ls_res = residual_func(ls_x, fuga, temperature, rMeOH, rCO)
        ls_obj = objective_from_residuals(ls_res)

        if np.isfinite(ls_obj) and ls_obj <= de_obj:
            best_x = ls_x
            final_obj = ls_obj
            final_stage = "DE + least_squares"
        else:
            best_x = de_x
            final_obj = de_obj
            final_stage = "DE_only_kept"

    except Exception as e:
        print(f"[{model_name}] least_squares 出错，退回 DE 结果: {e}")
        ls_result = None
        best_x = de_x
        final_obj = de_obj
        final_stage = "DE_only_due_to_LS_error"

    rate_meoh_pred, rate_co_pred, r1_pred, r2_pred, k1, k2 = prediction_func(best_x, fuga, temperature)

    r2_meoh = calc_r2(rMeOH, rate_meoh_pred)
    r2_co = calc_r2(rCO, rate_co_pred)
    avg_r2 = np.nanmean([r2_meoh, r2_co])

    rmse_meoh = calc_rmse(rMeOH, rate_meoh_pred)
    rmse_co = calc_rmse(rCO, rate_co_pred)

    mre_meoh = calc_mre(rMeOH, rate_meoh_pred)
    mre_co = calc_mre(rCO, rate_co_pred)

    result_df = df_group.copy()
    result_df["GHSV_fixed"] = ghsv_val
    result_df["p_fixed"] = p_val
    result_df["model"] = model_name
    result_df["rMeOH_pred"] = rate_meoh_pred
    result_df["rCO_pred"] = rate_co_pred
    result_df["r1_pred"] = r1_pred
    result_df["r2_pred"] = r2_pred
    result_df["k1"] = k1
    result_df["k2"] = k2
    result_df["rel_error_rMeOH_%"] = np.abs((rate_meoh_pred - rMeOH) / np.maximum(np.abs(rMeOH), 1e-12)) * 100
    result_df["rel_error_rCO_%"] = np.abs((rate_co_pred - rCO) / np.maximum(np.abs(rCO), 1e-12)) * 100

    if model_name == "powerlaw_with_1_minus_beta":
        K_f1, K_f2 = calculate_equilibrium_constants(temperature)
        fCO2 = np.maximum(fuga[:, 0], eps)
        fH2 = np.maximum(fuga[:, 1], eps)
        fCH3OH = np.maximum(fuga[:, 2], eps)
        fH2O = np.maximum(fuga[:, 3], eps)
        fCO = np.maximum(fuga[:, 4], eps)

        beta1 = (fCH3OH * fH2O) / np.maximum(K_f1 * fCO2 * (fH2 ** 3), eps)
        beta2 = (fCO * fH2O) / np.maximum(K_f2 * fCO2 * fH2, eps)

        result_df["beta1"] = beta1
        result_df["beta2"] = beta2

    param_row = {
        "GHSV": ghsv_val,
        "p": p_val,
        "n_points": len(df_group),
        "model": model_name
    }
    for name, val in zip(PARAM_NAMES, best_x):
        param_row[name] = val

    summary_row = {
        "GHSV": ghsv_val,
        "p": p_val,
        "n_points": len(df_group),
        "model": model_name,
        "de_success": bool(de_result.success),
        "de_message": str(de_result.message),
        "de_nit": int(getattr(de_result, "nit", -1)),
        "de_nfev": int(getattr(de_result, "nfev", -1)),
        "de_objective": de_obj,
        "final_stage": final_stage,
        "final_objective": final_obj,
        "r2_meoh": r2_meoh,
        "r2_co": r2_co,
        "avg_r2": avg_r2,
        "rmse_meoh": rmse_meoh,
        "rmse_co": rmse_co,
        "mre_meoh_%": mre_meoh,
        "mre_co_%": mre_co
    }

    if ls_result is not None:
        summary_row["ls_success"] = bool(ls_result.success)
        summary_row["ls_status"] = int(ls_result.status)
        summary_row["ls_message"] = str(ls_result.message)
        summary_row["ls_nfev"] = int(getattr(ls_result, "nfev", -1))
    else:
        summary_row["ls_success"] = False
        summary_row["ls_status"] = -999
        summary_row["ls_message"] = "least_squares_not_available"
        summary_row["ls_nfev"] = -1

    print(f"\n组别 GHSV={ghsv_val}, p={p_val}, 模型={model_name}")
    print(f"DE success   = {de_result.success}")
    print(f"DE message   = {de_result.message}")
    print(f"DE nit       = {getattr(de_result, 'nit', None)}")
    print(f"DE nfev      = {getattr(de_result, 'nfev', None)}")
    print(f"final stage  = {final_stage}")
    print(f"final obj    = {final_obj:.6f}")
    print(f"r2_meoh      = {r2_meoh:.6f}")
    print(f"r2_co        = {r2_co:.6f}")
    print(f"avg_r2       = {avg_r2:.6f}")
    print(f"rmse_meoh    = {rmse_meoh:.6e}")
    print(f"rmse_co      = {rmse_co:.6e}")
    print(f"mre_meoh %   = {mre_meoh:.4f}")
    print(f"mre_co %     = {mre_co:.4f}")

    return summary_row, param_row, result_df


# =========================
# 8. 主程序
# 固定 GHSV 和固定 p 双重分组拟合
# =========================
def main():
    df = load_data()

    all_summary_rows = []
    all_param_rows = []
    all_simple_results = []
    all_beta_results = []
    skipped_groups = []

    ghsv_values = sorted(df["GHSV"].dropna().unique().tolist())
    p_values = sorted(df["p"].dropna().unique().tolist())

    print("\n" + "=" * 70)
    print("开始固定 GHSV 和固定 p 的双重分组拟合")
    print("=" * 70)

    total_groups = 0

    for ghsv_val in ghsv_values:
        for p_val in p_values:
            df_group = df[(df["GHSV"] == ghsv_val) & (df["p"] == p_val)].copy()

            if len(df_group) == 0:
                continue

            total_groups += 1

            print("\n" + "=" * 70)
            print(f"开始拟合组别: GHSV = {ghsv_val}, p = {p_val}")
            print(f"该组数据点数 = {len(df_group)}")
            print("=" * 70)

            if len(df_group) < MIN_POINTS_PER_GROUP:
                print(f"该组点数少于 {MIN_POINTS_PER_GROUP}，跳过")
                skipped_groups.append({
                    "GHSV": ghsv_val,
                    "p": p_val,
                    "n_points": len(df_group),
                    "reason": f"less_than_{MIN_POINTS_PER_GROUP}_points"
                })
                continue

            fuga = df_group[["fCO2", "fH2", "fCH3OH", "fH2O", "fCO"]].to_numpy(dtype=float)
            rMeOH = df_group["rMeOH"].to_numpy(dtype=float)
            rCO = df_group["rCO"].to_numpy(dtype=float)
            temperature = df_group["T"].to_numpy(dtype=float)

            # 模型 1：simple power law
            summary_simple, param_simple, result_simple = fit_model(
                model_name="simple_powerlaw",
                objective_func=objective_simple,
                residual_func=residuals_simple,
                prediction_func=calc_predictions_simple,
                df_group=df_group,
                fuga=fuga,
                rMeOH=rMeOH,
                rCO=rCO,
                temperature=temperature,
                ghsv_val=ghsv_val,
                p_val=p_val
            )

            # 模型 2：power law with (1 - beta)
            summary_beta, param_beta, result_beta = fit_model(
                model_name="powerlaw_with_1_minus_beta",
                objective_func=objective_beta,
                residual_func=residuals_beta,
                prediction_func=calc_predictions_beta,
                df_group=df_group,
                fuga=fuga,
                rMeOH=rMeOH,
                rCO=rCO,
                temperature=temperature,
                ghsv_val=ghsv_val,
                p_val=p_val
            )

            all_summary_rows.extend([summary_simple, summary_beta])
            all_param_rows.extend([param_simple, param_beta])
            all_simple_results.append(result_simple)
            all_beta_results.append(result_beta)

            # 保存该组 parity plot
            if SAVE_PLOTS:
                group_tag = safe_filename(f"GHSV_{ghsv_val}_p_{p_val}")

                meoh_plot_path = os.path.join(PLOT_DIR, f"{group_tag}_parity_MeOH.png")
                co_plot_path = os.path.join(PLOT_DIR, f"{group_tag}_parity_CO.png")

                make_group_parity_plot(
                    y_exp_1=result_simple["rMeOH"].values,
                    y_pred_1=result_simple["rMeOH_pred"].values,
                    y_exp_2=result_beta["rMeOH"].values,
                    y_pred_2=result_beta["rMeOH_pred"].values,
                    label_1="simple_powerlaw",
                    label_2="powerlaw_with_1_minus_beta",
                    xlabel="Experimental MeOH",
                    ylabel="Predicted MeOH",
                    title=f"Parity Plot for MeOH | GHSV={ghsv_val}, p={p_val}",
                    save_path=meoh_plot_path
                )

                make_group_parity_plot(
                    y_exp_1=result_simple["rCO"].values,
                    y_pred_1=result_simple["rCO_pred"].values,
                    y_exp_2=result_beta["rCO"].values,
                    y_pred_2=result_beta["rCO_pred"].values,
                    label_1="simple_powerlaw",
                    label_2="powerlaw_with_1_minus_beta",
                    xlabel="Experimental CO",
                    ylabel="Predicted CO",
                    title=f"Parity Plot for CO | GHSV={ghsv_val}, p={p_val}",
                    save_path=co_plot_path
                )

    # 汇总输出
    summary_df = pd.DataFrame(all_summary_rows)
    params_df = pd.DataFrame(all_param_rows)
    skipped_df = pd.DataFrame(skipped_groups)

    if len(all_simple_results) > 0:
        results_simple_df = pd.concat(all_simple_results, axis=0, ignore_index=True)
    else:
        results_simple_df = pd.DataFrame()

    if len(all_beta_results) > 0:
        results_beta_df = pd.concat(all_beta_results, axis=0, ignore_index=True)
    else:
        results_beta_df = pd.DataFrame()

    # 排序更清楚
    if not summary_df.empty:
        summary_df = summary_df.sort_values(by=["GHSV", "p", "model"]).reset_index(drop=True)

    if not params_df.empty:
        params_df = params_df.sort_values(by=["GHSV", "p", "model"]).reset_index(drop=True)

    if not results_simple_df.empty:
        results_simple_df = results_simple_df.sort_values(by=["GHSV", "p", "T"]).reset_index(drop=True)

    if not results_beta_df.empty:
        results_beta_df = results_beta_df.sort_values(by=["GHSV", "p", "T"]).reset_index(drop=True)

    # 保存 Excel
    summary_path = os.path.join(OUTPUT_DIR, "grouped_fit_summary_fixed_GHSV_P.xlsx")
    params_path = os.path.join(OUTPUT_DIR, "grouped_fit_parameters_fixed_GHSV_P.xlsx")
    simple_path = os.path.join(OUTPUT_DIR, "grouped_fit_results_simple_powerlaw_fixed_GHSV_P.xlsx")
    beta_path = os.path.join(OUTPUT_DIR, "grouped_fit_results_powerlaw_beta_fixed_GHSV_P.xlsx")
    skipped_path = os.path.join(OUTPUT_DIR, "grouped_fit_skipped_groups_fixed_GHSV_P.xlsx")

    summary_df.to_excel(summary_path, index=False)
    params_df.to_excel(params_path, index=False)
    results_simple_df.to_excel(simple_path, index=False)
    results_beta_df.to_excel(beta_path, index=False)

    if not skipped_df.empty:
        skipped_df.to_excel(skipped_path, index=False)

    print("\n" + "=" * 70)
    print("全部拟合完成")
    print("=" * 70)
    print(f"实际参与拟合的组数: {summary_df[['GHSV', 'p']].drop_duplicates().shape[0] if not summary_df.empty else 0}")
    print(f"总遍历组数: {total_groups}")
    print(f"summary 已保存到: {summary_path}")
    print(f"params  已保存到: {params_path}")
    print(f"simple  已保存到: {simple_path}")
    print(f"beta    已保存到: {beta_path}")
    if not skipped_df.empty:
        print(f"skipped 已保存到: {skipped_path}")
    print(f"图片文件夹: {PLOT_DIR}")

    # 控制台显示一个简要总表
    if not summary_df.empty:
        show_cols = [
            "GHSV", "p", "n_points", "model",
            "de_nit", "de_nfev", "ls_nfev",
            "final_objective", "r2_meoh", "r2_co", "avg_r2"
        ]
        show_cols = [c for c in show_cols if c in summary_df.columns]
        print("\n简要结果预览：")
        print(summary_df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()