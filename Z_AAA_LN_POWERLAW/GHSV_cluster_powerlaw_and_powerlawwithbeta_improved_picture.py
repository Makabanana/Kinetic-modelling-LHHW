# 按 GHSV 分组拟合
# 比较 simple power law 和 power law × (1 - beta)
# 引入 ln_k_ref + Tave
# 输出关键 Excel 文件
# 额外绘制 GHSV = 8000 的 parity plot 和 residual plot

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from pathlib import Path


# =========================
# 0. 基本设置
# =========================
BASE_DIR = Path(__file__).resolve().parent
EXCEL_FILE = BASE_DIR / "12000GHSV.xlsx"
OUT_FILE = BASE_DIR / "GHSV_lnTave_key_outputs.xlsx"

# 图片输出文件夹
FIG_DIR = BASE_DIR / "GHSV_8000_figures"
FIG_DIR.mkdir(exist_ok=True)

R = 8.314
eps = 1e-12

# 是否运行 beta 模型
# True 表示同时比较 simple_powerlaw 和 powerlaw_with_1_minus_beta
# False 表示只跑 simple_powerlaw
RUN_BETA_MODEL = True


# 差分进化优化器设置
DE_CONFIG = {
    "strategy": "best1bin",
    "maxiter": 3000,
    "popsize": 30,
    "tol": 1e-7,
    "mutation": (0.5, 1.2),
    "recombination": 0.8,
    "polish": True,
    "seed": 42,
    "workers": 1,
    "updating": "immediate"
}


# =========================
# 1. 读取 Excel 数据
# =========================
def load_data(file_name=EXCEL_FILE, sheet_name=0):

    df = pd.read_excel(file_name, sheet_name=sheet_name, header=0)

    df.columns = df.columns.astype(str).str.strip()

    # 尝试去掉单位行
    # 如果第一行大部分内容不是数字，就认为它可能是单位行
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

    print("Excel 路径:", file_name)
    print("总数据点数:", len(df))
    print("GHSV 分组:", sorted(df["GHSV"].dropna().unique()))
    print("温度点:", sorted(df["T"].dropna().unique()))

    return df


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
# 3. 用 ln_k_ref 和 Tave 计算 k
# =========================
def calc_k_from_lnk_ref(ln_k_ref, E, T, Tave):
    """
    原始形式:
        k = A * exp(-E / RT)

    改写后:
        ln k = ln_k_ref - E / R * (1 / T - 1 / Tave)

    其中:
        ln_k_ref 是 Tave 附近的 ln k
        E 是表观活化能
    """

    ln_k = ln_k_ref - E / R * (1.0 / T - 1.0 / Tave)

    # 防止 exp 溢出
    k = np.exp(np.clip(ln_k, -700, 700))

    return k


# =========================
# 4A. simple power law 预测
# =========================
def calc_predictions_simple(par, fuga, temperature, Tave):
    """
    参数顺序:
        ln_k1_ref, E1, n1_A, n1_B,
        ln_k2_ref, E2, n2_A, n2_B

    R1:
        CO2 + 3H2 -> CH3OH + H2O

        r1 = k1 * fCO2^n1_A * fH2^n1_B

    R2:
        CO2 + H2 -> CO + H2O

        r2 = k2 * fCO2^n2_A * fH2^n2_B
    """

    ln_k1_ref, E1, n1_A, n1_B, ln_k2_ref, E2, n2_A, n2_B = par

    fCO2 = np.maximum(fuga[:, 0], eps)
    fH2 = np.maximum(fuga[:, 1], eps)

    k1 = calc_k_from_lnk_ref(ln_k1_ref, E1, temperature, Tave)
    k2 = calc_k_from_lnk_ref(ln_k2_ref, E2, temperature, Tave)

    r1 = k1 * fCO2**n1_A * fH2**n1_B
    r2 = k2 * fCO2**n2_A * fH2**n2_B

    rMeOH_pred = r1
    rCO_pred = r2

    return rMeOH_pred, rCO_pred, r1, r2, k1, k2


# =========================
# 4B. power law × (1 - beta) 预测
# =========================
def calc_predictions_beta(par, fuga, temperature, Tave):
    """
    参数顺序:
        ln_k1_ref, E1, n1_A, n1_B,
        ln_k2_ref, E2, n2_A, n2_B

    R1:
        r1 = k1 * fCO2^n1_A * fH2^n1_B * (1 - beta1)

    R2:
        r2 = k2 * fCO2^n2_A * fH2^n2_B * (1 - beta2)
    """

    ln_k1_ref, E1, n1_A, n1_B, ln_k2_ref, E2, n2_A, n2_B = par

    fCO2 = np.maximum(fuga[:, 0], eps)
    fH2 = np.maximum(fuga[:, 1], eps)
    fCH3OH = np.maximum(fuga[:, 2], eps)
    fH2O = np.maximum(fuga[:, 3], eps)
    fCO = np.maximum(fuga[:, 4], eps)

    k1 = calc_k_from_lnk_ref(ln_k1_ref, E1, temperature, Tave)
    k2 = calc_k_from_lnk_ref(ln_k2_ref, E2, temperature, Tave)

    K1, K2 = calculate_equilibrium_constants(temperature)

    K1 = np.maximum(K1, eps)
    K2 = np.maximum(K2, eps)

    beta1 = (fCH3OH * fH2O) / np.maximum(
        K1 * fCO2 * fH2**3,
        eps
    )

    beta2 = (fCO * fH2O) / np.maximum(
        K2 * fCO2 * fH2,
        eps
    )

    r1 = k1 * fCO2**n1_A * fH2**n1_B * (1.0 - beta1)
    r2 = k2 * fCO2**n2_A * fH2**n2_B * (1.0 - beta2)

    rMeOH_pred = r1
    rCO_pred = r2

    return rMeOH_pred, rCO_pred, r1, r2, k1, k2


# =========================
# 5. 目标函数
# =========================
def objective(par, prediction_func, fuga, temperature, Tave, rMeOH_exp, rCO_exp):

    try:
        rMeOH_pred, rCO_pred, _, _, _, _ = prediction_func(
            par,
            fuga,
            temperature,
            Tave
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
# 6. 评价指标
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
# 7. 拟合单个模型
# =========================
def fit_one_model(model_name, prediction_func, df_group):

    fuga = df_group[
        ["fCO2", "fH2", "fCH3OH", "fH2O", "fCO"]
    ].to_numpy(dtype=float)

    temperature = df_group["T"].to_numpy(dtype=float)
    rMeOH = df_group["rMeOH"].to_numpy(dtype=float)
    rCO = df_group["rCO"].to_numpy(dtype=float)

    # 每个 GHSV 分组单独计算 Tave
    Tave = float(np.mean(temperature))

    # 参数顺序:
    # ln_k1_ref, E1, n1_A, n1_B,
    # ln_k2_ref, E2, n2_A, n2_B
    bounds = [
        (-20, 5),        # ln_k1_ref
        (0, 120000),     # E1
        (-2, 3),         # n1_A
        (-2, 5),         # n1_B

        (-20, 5),        # ln_k2_ref
        (0, 120000),     # E2
        (-2, 3),         # n2_A
        (-2, 5),         # n2_B
    ]

    result = differential_evolution(
        objective,
        bounds=bounds,
        args=(
            prediction_func,
            fuga,
            temperature,
            Tave,
            rMeOH,
            rCO
        ),
        **DE_CONFIG
    )

    par_opt = result.x

    rMeOH_pred, rCO_pred, r1_pred, r2_pred, k1, k2 = prediction_func(
        par_opt,
        fuga,
        temperature,
        Tave
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

    prediction_df = df_group.copy()

    prediction_df["model"] = model_name
    prediction_df["Tave"] = Tave

    prediction_df["rMeOH_pred"] = rMeOH_pred
    prediction_df["rCO_pred"] = rCO_pred
    prediction_df["r1_pred"] = r1_pred
    prediction_df["r2_pred"] = r2_pred
    prediction_df["k1"] = k1
    prediction_df["k2"] = k2

    prediction_df["error_rMeOH"] = rMeOH_pred - rMeOH
    prediction_df["error_rCO"] = rCO_pred - rCO

    prediction_df["rel_error_rMeOH_%"] = (
        np.abs((rMeOH_pred - rMeOH) / np.maximum(np.abs(rMeOH), 1e-12))
        * 100.0
    )

    prediction_df["rel_error_rCO_%"] = (
        np.abs((rCO_pred - rCO) / np.maximum(np.abs(rCO), 1e-12))
        * 100.0
    )

    param_names = [
        "ln_k1_ref",
        "E1",
        "n1_A",
        "n1_B",
        "ln_k2_ref",
        "E2",
        "n2_A",
        "n2_B"
    ]

    param_dict = dict(zip(param_names, par_opt))

    # 反算 Tave 处的 k_ref
    # 注意这不是 A，而是 Tave 温度附近的 k
    param_dict["k1_ref_at_Tave"] = np.exp(np.clip(param_dict["ln_k1_ref"], -700, 700))
    param_dict["k2_ref_at_Tave"] = np.exp(np.clip(param_dict["ln_k2_ref"], -700, 700))

    summary = {
        "model": model_name,
        "n_points": len(df_group),
        "Tave": Tave,
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
    }

    return summary, param_dict, prediction_df


# =========================
# 8. 画图模块：考察 GHSV = 8000
# =========================
def plot_ghsv_8000(predictions_df, summary_df, target_ghsv=8000):

    df_plot = predictions_df[
        predictions_df["GHSV_fit_group"] == target_ghsv
    ].copy()

    summary_plot = summary_df[
        summary_df["GHSV"] == target_ghsv
    ].copy()

    if df_plot.empty:
        print(f"没有找到 GHSV = {target_ghsv} 的预测结果，跳过画图。")
        return

    models = df_plot["model"].dropna().unique()

    print("\n" + "=" * 70)
    print(f"开始绘制 GHSV = {target_ghsv} 的图")
    print("=" * 70)

    for model_name in models:

        df_m = df_plot[df_plot["model"] == model_name].copy()

        summary_m = summary_plot[summary_plot["model"] == model_name]

        if not summary_m.empty:
            r2_meoh = summary_m["r2_meoh"].iloc[0]
            r2_co = summary_m["r2_co"].iloc[0]
            avg_r2 = summary_m["avg_r2"].iloc[0]
        else:
            r2_meoh = np.nan
            r2_co = np.nan
            avg_r2 = np.nan

        # =========================
        # 1. MeOH parity plot
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

        )

        plt.tight_layout()

        fig_name = FIG_DIR / f"GHSV_{target_ghsv}_{model_name}_MeOH_parity.png"
        plt.savefig(fig_name, dpi=300)
        plt.close()

        # =========================
        # 2. CO parity plot
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
            f"GHSV = {target_ghsv}, CO parity
        )

        plt.tight_layout()

        fig_name = FIG_DIR / f"GHSV_{target_ghsv}_{model_name}_CO_parity.png"
        plt.savefig(fig_name, dpi=300)
        plt.close()

        # =========================
        # 3. MeOH residual vs T
        # =========================
        plt.figure(figsize=(6, 5))

        plt.scatter(
            df_m["T"],
            df_m["error_rMeOH"],
            alpha=0.8
        )

        plt.axhline(0, linestyle="--")

        plt.xlabel("Temperature / K")
        plt.ylabel("Prediction error of rMeOH")
        plt.title(
            f"GHSV = {target_ghsv}, MeOH residual vs T\n"
            f"{model_name}"
        )

        plt.tight_layout()

        fig_name = FIG_DIR / f"GHSV_{target_ghsv}_{model_name}_MeOH_residual_vs_T.png"
        plt.savefig(fig_name, dpi=300)
        plt.close()

        # =========================
        # 4. CO residual vs T
        # =========================
        plt.figure(figsize=(6, 5))

        plt.scatter(
            df_m["T"],
            df_m["error_rCO"],
            alpha=0.8
        )

        plt.axhline(0, linestyle="--")

        plt.xlabel("Temperature / K")
        plt.ylabel("Prediction error of rCO")
        plt.title(
            f"GHSV = {target_ghsv}, CO residual vs T\n"
            f"{model_name}"
        )

        plt.tight_layout()

        fig_name = FIG_DIR / f"GHSV_{target_ghsv}_{model_name}_CO_residual_vs_T.png"
        plt.savefig(fig_name, dpi=300)
        plt.close()

        print(f"已完成模型 {model_name} 的 GHSV = {target_ghsv} 图像输出")
        print(f"R2 MeOH = {r2_meoh:.4f}, R2 CO = {r2_co:.4f}, avg R2 = {avg_r2:.4f}")

    print("\n图像已保存到:")
    print(FIG_DIR)


# =========================
# 9. 主程序：按 GHSV 分组拟合
# =========================
def main():

    df = load_data(EXCEL_FILE, sheet_name=0)

    all_summary = []
    all_params = []
    all_predictions = []

    ghsv_list = sorted(df["GHSV"].dropna().unique())

    for ghsv in ghsv_list:

        print("\n" + "=" * 70)
        print(f"开始拟合 GHSV = {ghsv}")
        print("=" * 70)

        df_g = df[df["GHSV"] == ghsv].copy().reset_index(drop=True)

        print(f"GHSV = {ghsv} 的数据点数: {len(df_g)}")
        print(f"GHSV = {ghsv} 的 Tave = {df_g['T'].mean():.2f} K")

        model_list = [
            ("simple_powerlaw", calc_predictions_simple)
        ]

        if RUN_BETA_MODEL:
            model_list.append(
                ("powerlaw_with_1_minus_beta", calc_predictions_beta)
            )

        for model_name, prediction_func in model_list:

            summary, param_dict, prediction_df = fit_one_model(
                model_name=model_name,
                prediction_func=prediction_func,
                df_group=df_g
            )

            summary["GHSV"] = ghsv

            param_dict["GHSV"] = ghsv
            param_dict["model"] = model_name
            param_dict["n_points"] = len(df_g)
            param_dict["Tave"] = summary["Tave"]

            prediction_df["GHSV_fit_group"] = ghsv

            all_summary.append(summary)
            all_params.append(param_dict)
            all_predictions.append(prediction_df)

            print(f"\n模型: {model_name}")
            print(f"optimizer_success = {summary['optimizer_success']}")
            print(f"fit_success       = {summary['fit_success']}")
            print(f"objective         = {summary['objective']:.6f}")
            print(f"r2_meoh           = {summary['r2_meoh']:.6f}")
            print(f"r2_co             = {summary['r2_co']:.6f}")
            print(f"avg_r2            = {summary['avg_r2']:.6f}")
            print(f"mre_meoh_%        = {summary['mre_meoh_%']:.4f}")
            print(f"mre_co_%          = {summary['mre_co_%']:.4f}")

    summary_df = pd.DataFrame(all_summary)
    params_df = pd.DataFrame(all_params)
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    # 调整 summary 列顺序
    summary_front_cols = ["GHSV", "model", "n_points", "Tave"]
    summary_other_cols = [
        c for c in summary_df.columns
        if c not in summary_front_cols
    ]
    summary_df = summary_df[summary_front_cols + summary_other_cols]

    # 调整 parameters 列顺序
    params_front_cols = ["GHSV", "model", "n_points", "Tave"]
    params_other_cols = [
        c for c in params_df.columns
        if c not in params_front_cols
    ]
    params_df = params_df[params_front_cols + params_other_cols]

    # 输出一个关键 Excel 文件
    with pd.ExcelWriter(OUT_FILE, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        params_df.to_excel(writer, sheet_name="parameters", index=False)
        predictions_df.to_excel(writer, sheet_name="predictions", index=False)

    # 绘制 GHSV = 8000 的图
    plot_ghsv_8000(
        predictions_df=predictions_df,
        summary_df=summary_df,
        target_ghsv=8000
    )

    print("\n" + "=" * 70)
    print("全部 GHSV 分组拟合完成")
    print("关键结果已保存到:")
    print(OUT_FILE)
    print("GHSV = 8000 图片已保存到:")
    print(FIG_DIR)
    print("=" * 70)

    print("\nsummary 预览:")
    print(summary_df)

    print("\nparameters 预览:")
    print(params_df)


if __name__ == "__main__":
    main()