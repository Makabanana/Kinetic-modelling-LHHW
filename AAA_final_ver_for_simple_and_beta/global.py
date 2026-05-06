# 不按 GHSV 分类
# 使用全部数据进行 global fitting
# 对比 simple power law 和加入 (1 - beta) 的两种形式
# 在原有 lnA 基础上进一步引入 Tave 重参数化
#     ln k = ln_k_ref - E / R * (1 / T - 1 / Tave)

import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from pathlib import Path


# =========================
# 0. 全局设置
# =========================
R = 8.314
eps = 1e-12

# 如果 Excel 第二行是单位行，设为 True
# 如果不是，改成 False
SKIP_SECOND_ROW = True

# Excel 文件名
EXCEL_FILE = "full data.xlsx"

# 输出文件夹
OUT_DIR = Path("R_Global_fit_improved")
OUT_DIR.mkdir(exist_ok=True)


# =========================
# 1. 读取数据
# =========================
def load_data():

    if SKIP_SECOND_ROW:
        df = pd.read_excel(EXCEL_FILE, skiprows=[1])
    else:
        df = pd.read_excel(EXCEL_FILE)

    df.columns = df.columns.str.strip()

    # 统一列名
    df = df.rename(columns={
        "r CH3OH": "rMeOH",
        "r CO": "rCO",
        "r CO2": "rCO2"
    })

    print("实际列名如下：")
    print(df.columns.tolist())

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

    print(f"总数据点数: {len(df)}")

    return df


# =========================
# 2. 平衡常数
# 反应1: CO2 + 3H2 -> CH3OH + H2O
# 反应2: CO2 + H2  -> CO + H2O
# =========================
def calculate_equilibrium_constants(T):

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
# 3. beta 诊断
# =========================
def diagnose_beta(df, save_dir=None):

    T = df["T"].to_numpy(dtype=float)

    K_f1, K_f2 = calculate_equilibrium_constants(T)

    fCO2 = np.maximum(df["fCO2"].to_numpy(dtype=float), eps)
    fH2 = np.maximum(df["fH2"].to_numpy(dtype=float), eps)
    fCH3OH = np.maximum(df["fCH3OH"].to_numpy(dtype=float), eps)
    fH2O = np.maximum(df["fH2O"].to_numpy(dtype=float), eps)
    fCO = np.maximum(df["fCO"].to_numpy(dtype=float), eps)

    beta1 = (fCH3OH * fH2O) / np.maximum(
        K_f1 * fCO2 * (fH2 ** 3),
        eps
    )

    beta2 = (fCO * fH2O) / np.maximum(
        K_f2 * fCO2 * fH2,
        eps
    )

    beta_df = df.copy()
    beta_df["beta1"] = beta1
    beta_df["beta2"] = beta2

    beta_summary = pd.DataFrame([{
        "fit_label": "global",
        "n_points": len(df),

        "beta1_min": beta1.min(),
        "beta1_max": beta1.max(),
        "beta1_mean": beta1.mean(),
        "beta1_median": np.median(beta1),
        "beta1_std": beta1.std(ddof=1) if len(beta1) > 1 else 0.0,

        "beta2_min": beta2.min(),
        "beta2_max": beta2.max(),
        "beta2_mean": beta2.mean(),
        "beta2_median": np.median(beta2),
        "beta2_std": beta2.std(ddof=1) if len(beta2) > 1 else 0.0
    }])

    print("\nGlobal beta 诊断")
    print(
        f"beta1: min = {beta1.min():.6e}, "
        f"max = {beta1.max():.6e}, "
        f"mean = {beta1.mean():.6e}, "
        f"median = {np.median(beta1):.6e}"
    )
    print(
        f"beta2: min = {beta2.min():.6e}, "
        f"max = {beta2.max():.6e}, "
        f"mean = {beta2.mean():.6e}, "
        f"median = {np.median(beta2):.6e}"
    )

    if save_dir is not None:
        beta_df.to_excel(
            save_dir / "global_beta_values.xlsx",
            index=False
        )

        beta_summary.to_excel(
            save_dir / "global_beta_summary.xlsx",
            index=False
        )

    return beta_df, beta_summary


# =========================
# 4. 用 ln_k_ref 和 Tave 计算 k
# =========================
def calc_k_from_lnk_ref(ln_k_ref, E, T, Tave):

    ln_k = ln_k_ref - E / R * (1.0 / T - 1.0 / Tave)

    k = np.exp(np.clip(ln_k, -700, 700))

    return k


# =========================
# 5A. simple power law
#
# r1 = k1 * fCO2^n1_A * fH2^n1_B
# r2 = k2 * fCO2^n2_A * fH2^n2_B
# =========================
def calc_predictions_simple(par, fuga, temperature, Tave):

    ln_k1_ref, E1, n1_A, n1_B, ln_k2_ref, E2, n2_A, n2_B = par

    fCO2 = np.maximum(fuga[:, 0], eps)
    fH2 = np.maximum(fuga[:, 1], eps)

    k1 = calc_k_from_lnk_ref(
        ln_k_ref=ln_k1_ref,
        E=E1,
        T=temperature,
        Tave=Tave
    )

    k2 = calc_k_from_lnk_ref(
        ln_k_ref=ln_k2_ref,
        E=E2,
        T=temperature,
        Tave=Tave
    )

    r1 = k1 * (fCO2 ** n1_A) * (fH2 ** n1_B)
    r2 = k2 * (fCO2 ** n2_A) * (fH2 ** n2_B)

    return r1, r2


# =========================
# 5B. power law × (1 - beta)
# =========================
def calc_predictions_beta(par, fuga, temperature, Tave):

    ln_k1_ref, E1, n1_A, n1_B, ln_k2_ref, E2, n2_A, n2_B = par

    fCO2 = np.maximum(fuga[:, 0], eps)
    fH2 = np.maximum(fuga[:, 1], eps)
    fCH3OH = np.maximum(fuga[:, 2], eps)
    fH2O = np.maximum(fuga[:, 3], eps)
    fCO = np.maximum(fuga[:, 4], eps)

    k1 = calc_k_from_lnk_ref(
        ln_k_ref=ln_k1_ref,
        E=E1,
        T=temperature,
        Tave=Tave
    )

    k2 = calc_k_from_lnk_ref(
        ln_k_ref=ln_k2_ref,
        E=E2,
        T=temperature,
        Tave=Tave
    )

    K_f1, K_f2 = calculate_equilibrium_constants(temperature)

    K_f1 = np.maximum(K_f1, eps)
    K_f2 = np.maximum(K_f2, eps)

    beta1 = (fCH3OH * fH2O) / np.maximum(
        K_f1 * fCO2 * (fH2 ** 3),
        eps
    )

    beta2 = (fCO * fH2O) / np.maximum(
        K_f2 * fCO2 * fH2,
        eps
    )

    r1 = k1 * (fCO2 ** n1_A) * (fH2 ** n1_B) * (1.0 - beta1)
    r2 = k2 * (fCO2 ** n2_A) * (fH2 ** n2_B) * (1.0 - beta2)

    return r1, r2


# =========================
# 6. 目标函数
# =========================
def objective_simple(par, fuga, temperature, Tave, rMeOH_exp, rCO_exp):

    try:
        rMeOH_pred, rCO_pred = calc_predictions_simple(
            par,
            fuga,
            temperature,
            Tave
        )

        denom_meoh = np.maximum(np.abs(rMeOH_exp), 1e-6)
        denom_co = np.maximum(np.abs(rCO_exp), 1e-6)

        sse_meoh = np.sum(((rMeOH_pred - rMeOH_exp) / denom_meoh) ** 2)
        sse_co = np.sum(((rCO_pred - rCO_exp) / denom_co) ** 2)

        total_sse = sse_meoh + sse_co

        if not np.isfinite(total_sse):
            return 1e30

        return float(total_sse)

    except Exception as e:
        print("objective_simple 出错:", e)
        return 1e30


def objective_beta(par, fuga, temperature, Tave, rMeOH_exp, rCO_exp):

    try:
        rMeOH_pred, rCO_pred = calc_predictions_beta(
            par,
            fuga,
            temperature,
            Tave
        )

        denom_meoh = np.maximum(np.abs(rMeOH_exp), 1e-6)
        denom_co = np.maximum(np.abs(rCO_exp), 1e-6)

        sse_meoh = np.sum(((rMeOH_pred - rMeOH_exp) / denom_meoh) ** 2)
        sse_co = np.sum(((rCO_pred - rCO_exp) / denom_co) ** 2)

        total_sse = sse_meoh + sse_co

        if not np.isfinite(total_sse):
            return 1e30

        return float(total_sse)

    except Exception as e:
        print("objective_beta 出错:", e)
        return 1e30


# =========================
# 7. 评价指标
# =========================
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


# =========================
# 8. 单独画 parity plot
# =========================
def make_single_parity_plot(
    exp,
    pred,
    xlabel,
    ylabel,
    title,
    save_path
):

    plt.figure(figsize=(6, 6))

    plt.scatter(exp, pred, alpha=0.75)

    min_val = min(
        np.min(exp),
        np.min(pred)
    )

    max_val = max(
        np.max(exp),
        np.max(pred)
    )

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        "k--"
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# =========================
# 9. 单个模型拟合
# =========================
def fit_model(model_name, objective_func, prediction_func, df):

    fuga = df[
        ["fCO2", "fH2", "fCH3OH", "fH2O", "fCO"]
    ].to_numpy(dtype=float)

    rMeOH = df["rMeOH"].to_numpy(dtype=float)
    rCO = df["rCO"].to_numpy(dtype=float)
    temperature = df["T"].to_numpy(dtype=float)

    # global fitting 只计算一个整体 Tave
    Tave = float(np.mean(temperature))

    bounds = [
        (-30, 30),      # ln_k1_ref
        (0, 150000),    # E1
        (-2, 5),        # n1_A
        (-2, 5),        # n1_B

        (-30, 30),      # ln_k2_ref
        (0, 150000),    # E2
        (-2, 5),        # n2_A
        (-2, 5)         # n2_B
    ]

    result = differential_evolution(
        objective_func,
        bounds=bounds,
        args=(fuga, temperature, Tave, rMeOH, rCO),
        seed=42,
        popsize=15,
        maxiter=1000,
        tol=1e-6,
        mutation=(0.5, 1.0),
        recombination=0.7,
        polish=True,
        workers=1,
        updating="immediate"
    )

    par_opt = result.x

    rMeOH_pred, rCO_pred = prediction_func(
        par_opt,
        fuga,
        temperature,
        Tave
    )

    (
        ln_k1_ref,
        E1,
        n1_A,
        n1_B,
        ln_k2_ref,
        E2,
        n2_A,
        n2_B
    ) = par_opt

    k1_ref_at_Tave = np.exp(np.clip(ln_k1_ref, -700, 700))
    k2_ref_at_Tave = np.exp(np.clip(ln_k2_ref, -700, 700))

    r2_meoh = calc_r2(rMeOH, rMeOH_pred)
    r2_co = calc_r2(rCO, rCO_pred)
    avg_r2 = np.nanmean([r2_meoh, r2_co])

    rmse_meoh = calc_rmse(rMeOH, rMeOH_pred)
    rmse_co = calc_rmse(rCO, rCO_pred)

    mre_meoh = calc_mre(rMeOH, rMeOH_pred)
    mre_co = calc_mre(rCO, rCO_pred)

    pred_df = pd.DataFrame(index=df.index)

    pred_df[f"{model_name}_rMeOH_pred"] = rMeOH_pred
    pred_df[f"{model_name}_rCO_pred"] = rCO_pred

    pred_df[f"{model_name}_rel_error_rMeOH_%"] = (
        np.abs((rMeOH_pred - rMeOH) / np.maximum(np.abs(rMeOH), 1e-12))
        * 100.0
    )

    pred_df[f"{model_name}_rel_error_rCO_%"] = (
        np.abs((rCO_pred - rCO) / np.maximum(np.abs(rCO), 1e-12))
        * 100.0
    )

    summary = {
        "fit_label": "global",
        "model": model_name,
        "n_points": len(df),
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
        "mre_co_%": mre_co
    }

    params = {
        "fit_label": "global",
        "model": model_name,
        "n_points": len(df),
        "Tave": Tave,

        "ln_k1_ref": ln_k1_ref,
        "k1_ref_at_Tave": k1_ref_at_Tave,
        "E1_J_per_mol": E1,
        "E1_kJ_per_mol": E1 / 1000.0,
        "n1_A": n1_A,
        "n1_B": n1_B,

        "ln_k2_ref": ln_k2_ref,
        "k2_ref_at_Tave": k2_ref_at_Tave,
        "E2_J_per_mol": E2,
        "E2_kJ_per_mol": E2 / 1000.0,
        "n2_A": n2_A,
        "n2_B": n2_B
    }

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

    return summary, params, pred_df


# =========================
# 10. global fitting
# =========================
def fit_global(df):

    print("\n" + "=" * 60)
    print("开始 global fitting")
    print(f"总数据点数 = {len(df)}")
    print(f"Tave = {df['T'].mean():.2f} K")
    print("=" * 60)

    beta_df, beta_summary = diagnose_beta(
        df,
        save_dir=OUT_DIR
    )

    summary_simple, params_simple, pred_simple = fit_model(
        model_name="simple_powerlaw",
        objective_func=objective_simple,
        prediction_func=calc_predictions_simple,
        df=df
    )

    summary_beta, params_beta, pred_beta = fit_model(
        model_name="powerlaw_with_1_minus_beta",
        objective_func=objective_beta,
        prediction_func=calc_predictions_beta,
        df=df
    )

    summary_df = pd.DataFrame([summary_simple, summary_beta])
    params_df = pd.DataFrame([params_simple, params_beta])

    result_df = df.copy()

    for col in pred_simple.columns:
        result_df[col] = pred_simple[col].values

    for col in pred_beta.columns:
        result_df[col] = pred_beta[col].values

    summary_df.to_excel(
        OUT_DIR / "global_summary.xlsx",
        index=False
    )

    params_df.to_excel(
        OUT_DIR / "global_parameters.xlsx",
        index=False
    )

    result_df.to_excel(
        OUT_DIR / "global_predictions.xlsx",
        index=False
    )

    # =========================
    # 四张 parity plot
    # =========================

    make_single_parity_plot(
        exp=result_df["rMeOH"].values,
        pred=result_df["simple_powerlaw_rMeOH_pred"].values,
        xlabel="Experimental MeOH",
        ylabel="Predicted MeOH",
        title="Simple Power Law - MeOH, Global Fit",
        save_path=OUT_DIR / "global_simple_powerlaw_parity_meoh.png"
    )

    make_single_parity_plot(
        exp=result_df["rCO"].values,
        pred=result_df["simple_powerlaw_rCO_pred"].values,
        xlabel="Experimental CO",
        ylabel="Predicted CO",
        title="Simple Power Law - CO, Global Fit",
        save_path=OUT_DIR / "global_simple_powerlaw_parity_co.png"
    )

    make_single_parity_plot(
        exp=result_df["rMeOH"].values,
        pred=result_df["powerlaw_with_1_minus_beta_rMeOH_pred"].values,
        xlabel="Experimental MeOH",
        ylabel="Predicted MeOH",
        title="Power Law with 1 - beta - MeOH, Global Fit",
        save_path=OUT_DIR / "global_with_1_minus_beta_parity_meoh.png"
    )

    make_single_parity_plot(
        exp=result_df["rCO"].values,
        pred=result_df["powerlaw_with_1_minus_beta_rCO_pred"].values,
        xlabel="Experimental CO",
        ylabel="Predicted CO",
        title="Power Law with 1 - beta - CO, Global Fit",
        save_path=OUT_DIR / "global_with_1_minus_beta_parity_co.png"
    )

    return summary_df, params_df, result_df, beta_summary


# =========================
# 11. 主程序
# =========================
def main():

    df = load_data()

    summary_df, params_df, result_df, beta_summary = fit_global(df)

    print("\nGlobal fitting 完成")

    print("\n总汇总表：")
    print(summary_df)

    print("\n总参数表：")
    print(params_df)

    print("\nbeta 汇总表：")
    print(beta_summary)

    print("\n结果已保存到文件夹：")
    print(OUT_DIR)


if __name__ == "__main__":
    main()