# 按 GHSV 分组拟合
# 比较 simple power law 和 power law × (1 - beta)
# 只画 GHSV = 8000 的 simple_powerlaw parity plot

import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt


# =========================
# 0. 常数
# =========================
R = 8.314
eps = 1e-12


# =========================
# 1. 读取 Excel 数据
# =========================
def load_data(file_name="12000GHSV.xlsx", sheet_name=0):

    df = pd.read_excel(file_name, sheet_name=sheet_name, header=0)

    df.columns = df.columns.astype(str).str.strip()

    # 尝试去掉单位行
    if len(df) > 1:
        first_row = df.iloc[0].astype(str).tolist()

        numeric_like_count = 0

        for x in first_row:
            try:
                float(x)
                numeric_like_count += 1
            except:
                pass

        if numeric_like_count < max(2, len(df.columns) // 3):
            df = df.iloc[1:].copy()

    df.columns = df.columns.str.strip()

    # 列名兼容
    rename_map = {
        "r CH3OH": "rMeOH",
        "rCH3OH": "rMeOH",
        "r MeOH": "rMeOH",
        "r_CH3OH": "rMeOH",
        "r CH3OH ": "rMeOH",
        "rMeOH ": "rMeOH",

        "r CO": "rCO",
        "r_CO": "rCO",
        "rCO ": "rCO",

        "T ": "T",
        "GHSV ": "GHSV",

        "fCO2 ": "fCO2",
        "fH2 ": "fH2",
        "fCH3OH ": "fCH3OH",
        "fH2O ": "fH2O",
        "fCO ": "fCO",
    }

    df = df.rename(columns=rename_map)

    required_cols = [
        "GHSV",
        "T",
        "fCO2",
        "fH2",
        "fCH3OH",
        "fH2O",
        "fCO",
        "rMeOH",
        "rCO"
    ]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(
            f"缺少必要列: {missing}\n当前列名: {list(df.columns)}"
        )

    for c in required_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=required_cols).reset_index(drop=True)

    fuga = df[["fCO2", "fH2", "fCH3OH", "fH2O", "fCO"]].to_numpy(dtype=float)
    rMeOH = df["rMeOH"].to_numpy(dtype=float)
    rCO = df["rCO"].to_numpy(dtype=float)
    temperature = df["T"].to_numpy(dtype=float)

    print(f"总数据点数: {len(df)}")

    return df, fuga, rMeOH, rCO, temperature


# =========================
# 2. 平衡常数函数
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
# 3A. simple power law
# =========================
def calc_predictions_simple(par, fuga, temperature):
    """
    par 顺序:
        A1, E1, n1_A, n1_B,
        A2, E2, n2_A, n2_B

    R1:
        CO2 + 3H2 -> CH3OH + H2O
        r1 = k1 * fCO2^n1_A * fH2^n1_B

    R2:
        CO2 + H2 -> CO + H2O
        r2 = k2 * fCO2^n2_A * fH2^n2_B
    """

    A1, E1, n1_A, n1_B, A2, E2, n2_A, n2_B = par

    fCO2 = np.maximum(fuga[:, 0], eps)
    fH2 = np.maximum(fuga[:, 1], eps)

    k1 = A1 * np.exp(-E1 / (R * temperature))
    k2 = A2 * np.exp(-E2 / (R * temperature))

    r1 = k1 * fCO2**n1_A * fH2**n1_B
    r2 = k2 * fCO2**n2_A * fH2**n2_B

    rMeOH_pred = r1
    rCO_pred = r2

    return rMeOH_pred, rCO_pred, r1, r2, k1, k2


# =========================
# 3B. power law × (1 - beta)
# =========================
def calc_predictions_beta(par, fuga, temperature):
    """
    par 顺序:
        A1, E1, n1_A, n1_B,
        A2, E2, n2_A, n2_B
    """

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

    beta1 = (fCH3OH * fH2O) / np.maximum(
        K_f1 * fCO2 * fH2**3,
        eps
    )

    beta2 = (fCO * fH2O) / np.maximum(
        K_f2 * fCO2 * fH2,
        eps
    )

    r1 = k1 * fCO2**n1_A * fH2**n1_B * (1.0 - beta1)
    r2 = k2 * fCO2**n2_A * fH2**n2_B * (1.0 - beta2)

    rMeOH_pred = r1
    rCO_pred = r2

    return rMeOH_pred, rCO_pred, r1, r2, k1, k2


# =========================
# 4. 目标函数
# =========================
def objective_simple(par, fuga, temperature, rMeOH_exp, rCO_exp):

    try:
        rMeOH_pred, rCO_pred, _, _, _, _ = calc_predictions_simple(
            par,
            fuga,
            temperature
        )

        if not np.all(np.isfinite(rMeOH_pred)):
            return 1e30

        if not np.all(np.isfinite(rCO_pred)):
            return 1e30

        denom_meoh = np.maximum(np.abs(rMeOH_exp), 1e-6)
        denom_co = np.maximum(np.abs(rCO_exp), 1e-6)

        sse_meoh = np.sum(((rMeOH_pred - rMeOH_exp) / denom_meoh) ** 2)
        sse_co = np.sum(((rCO_pred - rCO_exp) / denom_co) ** 2)

        total_sse = sse_meoh + sse_co

        if not np.isfinite(total_sse):
            return 1e30

        return float(total_sse)

    except Exception:
        return 1e30


def objective_beta(par, fuga, temperature, rMeOH_exp, rCO_exp):

    try:
        rMeOH_pred, rCO_pred, _, _, _, _ = calc_predictions_beta(
            par,
            fuga,
            temperature
        )

        if not np.all(np.isfinite(rMeOH_pred)):
            return 1e30

        if not np.all(np.isfinite(rCO_pred)):
            return 1e30

        denom_meoh = np.maximum(np.abs(rMeOH_exp), 1e-6)
        denom_co = np.maximum(np.abs(rCO_exp), 1e-6)

        sse_meoh = np.sum(((rMeOH_pred - rMeOH_exp) / denom_meoh) ** 2)
        sse_co = np.sum(((rCO_pred - rCO_exp) / denom_co) ** 2)

        total_sse = sse_meoh + sse_co

        if not np.isfinite(total_sse):
            return 1e30

        return float(total_sse)

    except Exception:
        return 1e30


# =========================
# 5. 评价指标
# =========================
def calc_r2(y_exp, y_pred):

    ss_res = np.sum((y_exp - y_pred) ** 2)
    ss_tot = np.sum((y_exp - np.mean(y_exp)) ** 2)

    if ss_tot < 1e-12:
        return np.nan

    return 1.0 - ss_res / ss_tot


def calc_mre(y_exp, y_pred):

    denom = np.maximum(np.abs(y_exp), 1e-12)

    return np.mean(np.abs((y_pred - y_exp) / denom)) * 100.0


# =========================
# 6. 拟合单个模型
# =========================
def fit_model(
    model_name,
    objective_func,
    prediction_func,
    bounds,
    df,
    fuga,
    rMeOH,
    rCO,
    temperature
):

    result = differential_evolution(
        objective_func,
        bounds=bounds,
        args=(fuga, temperature, rMeOH, rCO),
        seed=42,
        popsize=15,
        maxiter=500,
        tol=1e-6,
        mutation=(0.5, 1.0),
        recombination=0.7,
        polish=True,
        workers=1,
        updating="immediate"
    )

    print(f"\n模型 {model_name} 优化器返回信息:")
    print("  success =", result.success)
    print("  message =", result.message)
    print("  objective =", result.fun)

    par_opt = result.x

    rMeOH_pred, rCO_pred, r1_pred, r2_pred, k1, k2 = prediction_func(
        par_opt,
        fuga,
        temperature
    )

    r2_meoh = calc_r2(rMeOH, rMeOH_pred)
    r2_co = calc_r2(rCO, rCO_pred)
    avg_r2 = np.nanmean([r2_meoh, r2_co])

    rmse_meoh = np.sqrt(np.mean((rMeOH - rMeOH_pred) ** 2))
    rmse_co = np.sqrt(np.mean((rCO - rCO_pred) ** 2))

    mre_meoh = calc_mre(rMeOH, rMeOH_pred)
    mre_co = calc_mre(rCO, rCO_pred)

    fit_success = (
        np.isfinite(result.fun)
        and mre_meoh < 10
        and mre_co < 20
    )

    result_df = df.copy()

    result_df["rMeOH_pred"] = rMeOH_pred
    result_df["rCO_pred"] = rCO_pred
    result_df["r1_pred"] = r1_pred
    result_df["r2_pred"] = r2_pred
    result_df["k1"] = k1
    result_df["k2"] = k2

    result_df["error_rMeOH"] = rMeOH_pred - rMeOH
    result_df["error_rCO"] = rCO_pred - rCO

    result_df["rel_error_rMeOH_%"] = (
        np.abs((rMeOH_pred - rMeOH) / np.maximum(np.abs(rMeOH), 1e-12))
        * 100.0
    )

    result_df["rel_error_rCO_%"] = (
        np.abs((rCO_pred - rCO) / np.maximum(np.abs(rCO), 1e-12))
        * 100.0
    )

    result_df["model"] = model_name

    param_names = [
        "A1",
        "E1",
        "n1_A",
        "n1_B",
        "A2",
        "E2",
        "n2_A",
        "n2_B"
    ]

    param_df = pd.DataFrame({
        "param_name": param_names,
        "value": par_opt
    })

    print(f"模型 {model_name} 拟合结果:")
    print(f"  r2_meoh   = {r2_meoh:.6f}")
    print(f"  r2_co     = {r2_co:.6f}")
    print(f"  avg_r2    = {avg_r2:.6f}")
    print(f"  rmse_meoh = {rmse_meoh:.6e}")
    print(f"  rmse_co   = {rmse_co:.6e}")
    print(f"  mre_meoh% = {mre_meoh:.4f}")
    print(f"  mre_co%   = {mre_co:.4f}")
    print(f"  fit_success = {fit_success}")

    return {
        "model": model_name,
        "optimizer_success": result.success,
        "optimizer_message": str(result.message),
        "fit_success": fit_success,
        "objective": result.fun,
        "r2_meoh": r2_meoh,
        "r2_co": r2_co,
        "avg_r2": avg_r2,
        "rmse_meoh": rmse_meoh,
        "rmse_co": rmse_co,
        "mre_meoh_%": mre_meoh,
        "mre_co_%": mre_co,
        "params": par_opt,
        "df": result_df,
        "param_df": param_df
    }


# =========================
# 7. 只画 GHSV = 8000 的 simple_powerlaw parity plot
# =========================
def plot_simple_parity_ghsv8000(ghsv, res_simple, target_ghsv=8000):

    if int(ghsv) != int(target_ghsv):
        return

    df_m = res_simple["df"].copy()
    model_name = res_simple["model"]

    r2_meoh = res_simple["r2_meoh"]
    r2_co = res_simple["r2_co"]

    print("\n" + "=" * 70)
    print(f"开始绘制 GHSV = {target_ghsv} 的 simple_powerlaw parity plot")
    print("=" * 70)

    # =========================
    # 7.1 MeOH parity plot
    # =========================
    plt.figure(figsize=(6, 5))

    x = df_m["rMeOH"]
    y = df_m["rMeOH_pred"]

    plt.scatter(x, y, alpha=0.8)

    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--"
    )

    plt.xlabel("Experimental rMeOH")
    plt.ylabel("Predicted rMeOH")
    plt.title(
        f"GHSV = {target_ghsv}, MeOH parity\n"
        f"{model_name}, R² = {r2_meoh:.3f}"
    )

    plt.tight_layout()

    fig_name = f"GHSV_{target_ghsv}_{model_name}_MeOH_parity.png"
    plt.savefig(fig_name, dpi=300)
    plt.show()
    plt.close()

    # =========================
    # 7.2 CO parity plot
    # =========================
    plt.figure(figsize=(6, 5))

    x = df_m["rCO"]
    y = df_m["rCO_pred"]

    plt.scatter(x, y, alpha=0.8)

    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--"
    )

    plt.xlabel("Experimental rCO")
    plt.ylabel("Predicted rCO")
    plt.title(
        f"GHSV = {target_ghsv}, CO parity\n"
        f"{model_name}, R² = {r2_co:.3f}"
    )

    plt.tight_layout()

    fig_name = f"GHSV_{target_ghsv}_{model_name}_CO_parity.png"
    plt.savefig(fig_name, dpi=300)
    plt.show()
    plt.close()

    print(f"GHSV = {target_ghsv} 的 simple_powerlaw parity plot 已保存")


# =========================
# 8. 主程序：按 GHSV 分组拟合
# =========================
def main():

    df, _, _, _, _ = load_data(file_name="full data.xlsx", sheet_name=0)

    # 参数边界
    # par 顺序:
    # A1, E1, n1_A, n1_B,
    # A2, E2, n2_A, n2_B
    bounds = [
        (1e-5, 1e3),       # A1
        (-15000, 15000),   # E1
        (0, 5),            # n1_A
        (0, 5),            # n1_B

        (1e-5, 1e3),       # A2
        (-15000, 15000),   # E2
        (0, 5),            # n2_A
        (-2, 2)            # n2_B
    ]

    all_summary = []
    all_params = []
    all_results = []

    ghsv_list = sorted(df["GHSV"].dropna().unique())

    for ghsv in ghsv_list:

        print("\n" + "=" * 70)
        print(f"开始拟合 GHSV = {ghsv}")
        print("=" * 70)

        df_g = df[df["GHSV"] == ghsv].copy().reset_index(drop=True)

        fuga_g = df_g[
            ["fCO2", "fH2", "fCH3OH", "fH2O", "fCO"]
        ].to_numpy(dtype=float)

        rMeOH_g = df_g["rMeOH"].to_numpy(dtype=float)
        rCO_g = df_g["rCO"].to_numpy(dtype=float)
        temperature_g = df_g["T"].to_numpy(dtype=float)

        print(f"GHSV = {ghsv} 的数据点数: {len(df_g)}")

        # =========================
        # 8.1 simple power law
        # =========================
        res_simple = fit_model(
            model_name="simple_powerlaw",
            objective_func=objective_simple,
            prediction_func=calc_predictions_simple,
            bounds=bounds,
            df=df_g,
            fuga=fuga_g,
            rMeOH=rMeOH_g,
            rCO=rCO_g,
            temperature=temperature_g
        )

        # =========================
        # 8.2 power law with beta
        # =========================
        res_beta = fit_model(
            model_name="powerlaw_with_1_minus_beta",
            objective_func=objective_beta,
            prediction_func=calc_predictions_beta,
            bounds=bounds,
            df=df_g,
            fuga=fuga_g,
            rMeOH=rMeOH_g,
            rCO=rCO_g,
            temperature=temperature_g
        )

        # =========================
        # 8.3 只画 GHSV = 8000 的 simple_powerlaw 图
        # =========================
        plot_simple_parity_ghsv8000(
            ghsv=ghsv,
            res_simple=res_simple,
            target_ghsv=8000
        )

        # =========================
        # 8.4 保存当前 GHSV 的 summary、parameters、results
        # =========================
        for res in [res_simple, res_beta]:

            all_summary.append({
                "GHSV": ghsv,
                "model": res["model"],
                "n_points": len(df_g),
                "optimizer_success": res["optimizer_success"],
                "optimizer_message": res["optimizer_message"],
                "fit_success": res["fit_success"],
                "objective": res["objective"],
                "r2_meoh": res["r2_meoh"],
                "r2_co": res["r2_co"],
                "avg_r2": res["avg_r2"],
                "rmse_meoh": res["rmse_meoh"],
                "rmse_co": res["rmse_co"],
                "mre_meoh_%": res["mre_meoh_%"],
                "mre_co_%": res["mre_co_%"]
            })

            param_dict = dict(
                zip(
                    res["param_df"]["param_name"],
                    res["param_df"]["value"]
                )
            )

            all_params.append({
                "GHSV": ghsv,
                "model": res["model"],
                "n_points": len(df_g),
                **param_dict
            })

            result_df = res["df"].copy()
            result_df["GHSV_fit_group"] = ghsv
            all_results.append(result_df)

        # =========================
        # 8.5 每个 GHSV 分别保存详细结果
        # =========================
        res_simple["df"].to_excel(
            f"fit_results_simple_powerlaw_GHSV_{int(ghsv)}.xlsx",
            index=False
        )

        res_beta["df"].to_excel(
            f"fit_results_powerlaw_with_1_minus_beta_GHSV_{int(ghsv)}.xlsx",
            index=False
        )

    # =========================
    # 8.6 保存所有 GHSV 的汇总结果
    # =========================
    summary_df = pd.DataFrame(all_summary)
    params_df = pd.DataFrame(all_params)
    results_df = pd.concat(all_results, ignore_index=True)

    summary_df.to_excel("GHSV_comparison_summary.xlsx", index=False)
    params_df.to_excel("GHSV_comparison_parameters.xlsx", index=False)
    results_df.to_excel("GHSV_fit_results_all.xlsx", index=False)

    print("\n全部 GHSV 分组拟合完成")
    print("\n汇总结果:")
    print(summary_df)

    print("\n参数结果:")
    print(params_df)

    print("\nGHSV = 8000 的 simple_powerlaw 图已输出:")
    print("GHSV_8000_simple_powerlaw_MeOH_parity.png")
    print("GHSV_8000_simple_powerlaw_CO_parity.png")


if __name__ == "__main__":
    main()