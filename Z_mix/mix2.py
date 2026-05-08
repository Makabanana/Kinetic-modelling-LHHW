# 按 GHSV 分类，只拟合 Hybrid II model
#
# Hybrid II:
#   反应 1 MeOH formation:
#       rMeOH = k1 * fCO2^n1_A * fH2^n1_B
#
#   反应 2 RWGS:
#       rCO = rf - rr
#       rf = k2f * fCO2^n2_A * fH2^n2_B
#       rr = k2r * fCO^n2_C * fH2O^n2_D
#       k2r = k2f / Keq2
#
# Arrhenius Tave 重参数化:
#       ln k = ln_k_ref - E / R * (1 / T - 1 / Tave)

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

SKIP_SECOND_ROW = True
EXCEL_FILE = "full data.xlsx"

OUT_DIR = Path("R_GHSV_cluster_hybridII_meoh_simple_rwgs_reversible")
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
#
# 反应 1:
# CO2 + 3H2 -> CH3OH + H2O
#
# 反应 2:
# CO2 + H2 -> CO + H2O
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
# 只用于诊断，不进入 MeOH 速率式
# =========================
def diagnose_beta(df_group, ghsv_label, save_dir=None):

    T = df_group["T"].to_numpy(dtype=float)

    K_f1, K_f2 = calculate_equilibrium_constants(T)

    fCO2 = np.maximum(df_group["fCO2"].to_numpy(dtype=float), eps)
    fH2 = np.maximum(df_group["fH2"].to_numpy(dtype=float), eps)
    fCH3OH = np.maximum(df_group["fCH3OH"].to_numpy(dtype=float), eps)
    fH2O = np.maximum(df_group["fH2O"].to_numpy(dtype=float), eps)
    fCO = np.maximum(df_group["fCO"].to_numpy(dtype=float), eps)

    beta1 = (fCH3OH * fH2O) / np.maximum(
        K_f1 * fCO2 * (fH2 ** 3),
        eps
    )

    beta2 = (fCO * fH2O) / np.maximum(
        K_f2 * fCO2 * fH2,
        eps
    )

    beta_df = df_group.copy()
    beta_df["beta1"] = beta1
    beta_df["beta2"] = beta2

    beta_summary = pd.DataFrame([{
        "GHSV": ghsv_label,
        "n_points": len(df_group),

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

    print(f"\nGHSV = {ghsv_label} 的 beta 诊断")
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
            save_dir / f"GHSV_{ghsv_label}_beta_values.xlsx",
            index=False
        )

        beta_summary.to_excel(
            save_dir / f"GHSV_{ghsv_label}_beta_summary.xlsx",
            index=False
        )

    return beta_df, beta_summary


# =========================
# 4. Tave 重参数化计算 k
# =========================
def calc_k_from_lnk_ref(ln_k_ref, E, T, Tave):

    ln_k = ln_k_ref - E / R * (1.0 / T - 1.0 / Tave)

    k = np.exp(np.clip(ln_k, -700, 700))

    return k


# =========================
# 5. Hybrid II 预测
#
# 参数顺序:
# ln_k1_ref, E1, n1_A, n1_B,
# ln_k2f_ref, E2f, n2_A, n2_B, n2_C, n2_D
#
# MeOH:
# r1 = k1 * fCO2^n1_A * fH2^n1_B
#
# RWGS:
# r2 = k2f * fCO2^n2_A * fH2^n2_B
#      - k2f / K2 * fCO^n2_C * fH2O^n2_D
# =========================
def calc_predictions_hybridII(par, fuga, temperature, Tave):

    (
        ln_k1_ref,
        E1,
        n1_A,
        n1_B,
        ln_k2f_ref,
        E2f,
        n2_A,
        n2_B,
        n2_C,
        n2_D
    ) = par

    fCO2 = np.maximum(fuga[:, 0], eps)
    fH2 = np.maximum(fuga[:, 1], eps)
    fH2O = np.maximum(fuga[:, 3], eps)
    fCO = np.maximum(fuga[:, 4], eps)

    # MeOH simple power law
    k1 = calc_k_from_lnk_ref(
        ln_k_ref=ln_k1_ref,
        E=E1,
        T=temperature,
        Tave=Tave
    )

    r1 = k1 * (fCO2 ** n1_A) * (fH2 ** n1_B)

    # RWGS reversible power law
    k2f = calc_k_from_lnk_ref(
        ln_k_ref=ln_k2f_ref,
        E=E2f,
        T=temperature,
        Tave=Tave
    )

    _, K2 = calculate_equilibrium_constants(temperature)
    K2 = np.maximum(K2, eps)

    k2r = k2f / K2

    r2_forward = k2f * (fCO2 ** n2_A) * (fH2 ** n2_B)
    r2_reverse = k2r * (fCO ** n2_C) * (fH2O ** n2_D)

    r2 = r2_forward - r2_reverse

    return r1, r2


# =========================
# 6. 目标函数
# 用相对误差平方和
# =========================
def objective_hybridII(par, fuga, temperature, Tave, rMeOH_exp, rCO_exp):

    try:
        rMeOH_pred, rCO_pred = calc_predictions_hybridII(
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
        print("objective_hybridII 出错:", e)
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
# 8. parity plot
# =========================
def make_parity_plot(exp, pred, xlabel, ylabel, title, save_path):

    plt.figure(figsize=(6, 6))

    plt.scatter(exp, pred, alpha=0.75, label="hybridII_model")

    min_val = min(np.min(exp), np.min(pred))
    max_val = max(np.max(exp), np.max(pred))

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        "k--",
        label="y = x"
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# =========================
# 9. 拟合 Hybrid II model
# =========================
def fit_model_hybridII(df_group):

    model_name = "hybridII_meoh_simple_rwgs_reversible"

    fuga = df_group[
        ["fCO2", "fH2", "fCH3OH", "fH2O", "fCO"]
    ].to_numpy(dtype=float)

    rMeOH = df_group["rMeOH"].to_numpy(dtype=float)
    rCO = df_group["rCO"].to_numpy(dtype=float)
    temperature = df_group["T"].to_numpy(dtype=float)

    Tave = float(np.mean(temperature))

    # 参数顺序:
    # ln_k1_ref, E1, n1_A, n1_B,
    # ln_k2f_ref, E2f, n2_A, n2_B, n2_C, n2_D
    bounds = [
        (-30, 30),       # ln_k1_ref
        (0, 150000),     # E1
        (0, 5),         # n1_A
        (0, 5),         # n1_B

        (-30, 30),       # ln_k2f_ref
        (0, 150000),     # E2f
        (0, 5),         # n2_A
        (-2, 5),         # n2_B
        (0, 5),          # n2_C
        (0, 5)           # n2_D
    ]

    result = differential_evolution(
        objective_hybridII,
        bounds=bounds,
        args=(fuga, temperature, Tave, rMeOH, rCO),
        seed=42,
        popsize=30,
        maxiter=2000,
        tol=1e-6,
        mutation=(0.5, 1.0),
        recombination=0.7,
        polish=True,
        workers=1,
        updating="immediate"
    )

    par_opt = result.x

    rMeOH_pred, rCO_pred = calc_predictions_hybridII(
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
        ln_k2f_ref,
        E2f,
        n2_A,
        n2_B,
        n2_C,
        n2_D
    ) = par_opt

    k1_ref_at_Tave = np.exp(np.clip(ln_k1_ref, -700, 700))
    k2f_ref_at_Tave = np.exp(np.clip(ln_k2f_ref, -700, 700))

    r2_meoh = calc_r2(rMeOH, rMeOH_pred)
    r2_co = calc_r2(rCO, rCO_pred)
    avg_r2 = np.nanmean([r2_meoh, r2_co])

    rmse_meoh = calc_rmse(rMeOH, rMeOH_pred)
    rmse_co = calc_rmse(rCO, rCO_pred)

    mre_meoh = calc_mre(rMeOH, rMeOH_pred)
    mre_co = calc_mre(rCO, rCO_pred)

    pred_df = pd.DataFrame(index=df_group.index)

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
        "model": model_name,
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
        "model": model_name,
        "Tave": Tave,

        "ln_k1_ref": ln_k1_ref,
        "k1_ref_at_Tave": k1_ref_at_Tave,
        "E1_J_per_mol": E1,
        "E1_kJ_per_mol": E1 / 1000.0,
        "n1_A": n1_A,
        "n1_B": n1_B,

        "ln_k2f_ref": ln_k2f_ref,
        "k2f_ref_at_Tave": k2f_ref_at_Tave,
        "E2f_J_per_mol": E2f,
        "E2f_kJ_per_mol": E2f / 1000.0,
        "n2_A": n2_A,
        "n2_B": n2_B,
        "n2_C": n2_C,
        "n2_D": n2_D
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
    print(f"n1_A = {n1_A:.6f}")
    print(f"n1_B = {n1_B:.6f}")
    print(f"n2_C = {n2_C:.6f}")
    print(f"n2_D = {n2_D:.6f}")

    return summary, params, pred_df


# =========================
# 10. 拟合单个 GHSV 分组
# =========================
def fit_one_ghsv_group(df_group, ghsv_value):

    ghsv_label = int(round(float(ghsv_value)))

    group_dir = OUT_DIR / f"GHSV_{ghsv_label}"
    group_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 60)
    print(f"开始拟合 GHSV = {ghsv_label}")
    print(f"该组数据点数 = {len(df_group)}")
    print(f"Tave = {df_group['T'].mean():.2f} K")
    print("=" * 60)

    beta_df, beta_summary = diagnose_beta(
        df_group,
        ghsv_label,
        save_dir=group_dir
    )

    summary_hybridII, params_hybridII, pred_hybridII = fit_model_hybridII(
        df_group=df_group
    )

    summary_df = pd.DataFrame([summary_hybridII])
    params_df = pd.DataFrame([params_hybridII])

    result_df = df_group.copy()

    for col in pred_hybridII.columns:
        result_df[col] = pred_hybridII[col].values

    summary_df.to_excel(
        group_dir / f"GHSV_{ghsv_label}_hybridII_summary.xlsx",
        index=False
    )

    params_df.to_excel(
        group_dir / f"GHSV_{ghsv_label}_hybridII_parameters.xlsx",
        index=False
    )

    result_df.to_excel(
        group_dir / f"GHSV_{ghsv_label}_hybridII_predictions.xlsx",
        index=False
    )

    make_parity_plot(
        exp=result_df["rMeOH"].values,
        pred=result_df["hybridII_meoh_simple_rwgs_reversible_rMeOH_pred"].values,
        xlabel="Experimental MeOH",
        ylabel="Predicted MeOH",
        title=f"Hybrid II Model Parity Plot for MeOH, GHSV = {ghsv_label}",
        save_path=group_dir / f"GHSV_{ghsv_label}_hybridII_parity_meoh.png"
    )

    make_parity_plot(
        exp=result_df["rCO"].values,
        pred=result_df["hybridII_meoh_simple_rwgs_reversible_rCO_pred"].values,
        xlabel="Experimental CO",
        ylabel="Predicted CO",
        title=f"Hybrid II Model Parity Plot for CO, GHSV = {ghsv_label}",
        save_path=group_dir / f"GHSV_{ghsv_label}_hybridII_parity_co.png"
    )

    summary_df.insert(0, "GHSV", ghsv_label)
    params_df.insert(0, "GHSV", ghsv_label)

    return summary_df, params_df, result_df, beta_summary


# =========================
# 11. 主程序
# =========================
def main():

    df = load_data()

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
            df_group,
            ghsv
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

    all_summary_df.to_excel(
        OUT_DIR / "all_ghsv_hybridII_summary.xlsx",
        index=False
    )

    all_params_df.to_excel(
        OUT_DIR / "all_ghsv_hybridII_parameters.xlsx",
        index=False
    )

    all_predictions_df.to_excel(
        OUT_DIR / "all_ghsv_hybridII_predictions.xlsx",
        index=False
    )

    all_beta_summary_df.to_excel(
        OUT_DIR / "all_ghsv_beta_summary.xlsx",
        index=False
    )

    print("\n全部 GHSV 分组 Hybrid II model 拟合完成")

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