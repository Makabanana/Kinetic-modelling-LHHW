from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution


# ============================================================
# 0. 全局设置
# ============================================================

EPS = 1e-30

# 差分进化法随机种子，固定起点。
DE_SEED = 42

# 模型名称
MODEL_NAME = "Model_E_integral_two_reaction_LHHW_DE_original_style"

# 当前脚本所在文件夹
BASE_DIR = Path(__file__).resolve().parent

# 输出文件夹
OUT_DIR = BASE_DIR / "output" / "model_E_integral_DE_original_style"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 总 experiment log
EXPERIMENT_LOG_PATH = BASE_DIR / "output" / "experiment_log.csv"


# ============================================================
# 1. 自动寻找数据文件
# ============================================================

def find_data_file():

    candidate_paths = [
        BASE_DIR / "data" / "12000GHSV.xlsx",
        BASE_DIR / "12000GHSV.xlsx",
        BASE_DIR.parent / "data" / "12000GHSV.xlsx",
        Path.cwd() / "data" / "12000GHSV.xlsx",
        Path.cwd() / "12000GHSV.xlsx",
    ]

    for path in candidate_paths:
        if path.exists():
            return path

    message = "Cannot find 12000GHSV.xlsx. The script searched:\n"
    for path in candidate_paths:
        message += f"  {path}\n"

    raise FileNotFoundError(message)


EXCEL_FILE = find_data_file()


# ============================================================
# 2. 固定床反应器条件
# ============================================================

# 催化剂装填量
# 0.2 g = 0.0002 kg
W_CAT_KG = 0.0002

# 总气体流速
# 42 mL/min = 42e-3 L/min
STANDARD_FLOW_L_PER_MIN = 42e-3

# 标准摩尔体积，L/mol
STANDARD_MOLAR_VOLUME_L_PER_MOL = 22.414

# 入口总摩尔流量，mol/s
F_TOTAL_IN = STANDARD_FLOW_L_PER_MIN / STANDARD_MOLAR_VOLUME_L_PER_MOL / 60.0

# fourth order Runge Kutta 积分步数
# 这里把催化剂床层分成 20 小段积分
RK4_STEPS = 20


# ============================================================
# 3. 差分进化法设置

# ============================================================

DE_POPSIZE = 8
DE_MAXITER = 100
DE_TOL = 1e-6
DE_MUTATION = (0.5, 1.0)
DE_RECOMBINATION = 0.7
DE_POLISH = False
DE_WORKERS = 1
DE_UPDATING = "immediate"


# ============================================================
# 4. 参数设置
# ============================================================

PARAMETER_NAMES = [
    "ln_k1_eff_ref",
    "E1_over_R",
    "ln_k2_eff_ref",
    "E2_over_R",
    "ln_KCO2",
    "ln_KCO",
    "ln_KH2O_H2",
]

# 参数顺序：
# ln_k1_eff_ref, E1_over_R,
# ln_k2_eff_ref, E2_over_R,
# ln_KCO2, ln_KCO, ln_KH2O_H2
BOUNDS = [
    (-30.0, 10.0),          # ln_k1_eff_ref
    (0.0, 30000.0),         # E1_over_R
    (-30.0, 10.0),          # ln_k2_eff_ref
    (0.0, 30000.0),         # E2_over_R
    (-20.0, 10.0),          # ln_KCO2
    (-20.0, 10),  # ln_KCO
    (-20.0, 10.0),          # ln_KH2O_H2
]

#检验是否击打边界
LOWER_BOUNDS = np.array([item[0] for item in BOUNDS], dtype=float)
UPPER_BOUNDS = np.array([item[1] for item in BOUNDS], dtype=float)

BOUND_WARNING_FRACTION = 0.02


# ============================================================
# 5. 读取数据
# ============================================================

def load_data():

    df = pd.read_excel(EXCEL_FILE)

    df.columns = [str(col).strip().replace("\n", " ") for col in df.columns]

    df = df.rename(columns={
        "H/C RATIO": "HC",
        "H/C": "HC",
        "p": "p_MPa",
        "r CH3OH": "rCH3OH",
        "rCH3OH": "rCH3OH",
        "r CO": "rCO",
        "rCO": "rCO",
    })

    required_cols = [
        "HC",
        "p_MPa",
        "GHSV",
        "T",
        "fCO2",
        "fH2",
        "fCH3OH",
        "fH2O",
        "fCO",
        "rCH3OH",
        "rCO",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"缺少列: {missing_cols}")

    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required_cols).copy()

    df = df[df["GHSV"] == 12000].copy()

    if df.empty:
        raise ValueError("No valid GHSV = 12000 rows were found.")

    df["Kf1"], df["Kf2"] = calculate_equilibrium_constants(df["T"].values)

    print("实际列名如下：")
    print(df.columns.tolist())
    print(f"总数据点数: {len(df)}")

    return df


# ============================================================
# 6. 平衡常数
# ============================================================

def calculate_equilibrium_constants(T):
    """
    Kf1:
        CO2 + 3H2 <-> CH3OH + H2O

    Kf2:
        CO2 + H2 <-> CO + H2O
    """

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


# ============================================================
# 7. 参数转换 把参数还原成我们需要的形式
# ============================================================

def unpack_params(par):
    """
    把优化器参数转换成实际参数。

    par 顺序：
        ln_k1_eff_ref
        E1_over_R
        ln_k2_eff_ref
        E2_over_R
        ln_KCO2
        ln_KCO
        ln_KH2O_H2
    """

    (
        ln_k1_eff_ref,
        E1_over_R,
        ln_k2_eff_ref,
        E2_over_R,
        ln_KCO2,
        ln_KCO,
        ln_KH2O_H2,
    ) = par

    params = {
        "k1_eff_ref": np.exp(np.clip(ln_k1_eff_ref, -700, 700)),
        "E1_over_R": E1_over_R,
        "k2_eff_ref": np.exp(np.clip(ln_k2_eff_ref, -700, 700)),
        "E2_over_R": E2_over_R,
        "KCO2": np.exp(np.clip(ln_KCO2, -700, 700)),
        "KCO": np.exp(np.clip(ln_KCO, -700, 700)),
        "KH2O_H2": np.exp(np.clip(ln_KH2O_H2, -700, 700)),
    }

    return params


def add_energy_units(params):
    """
    E = E_over_R × R

    R = 8.314 J/mol/K
    """

    report = params.copy()
    report["E1_kJ_per_mol"] = params["E1_over_R"] * 8.314 / 1000.0
    report["E2_kJ_per_mol"] = params["E2_over_R"] * 8.314 / 1000.0

    return report


# ============================================================
# 8. 局部 LHHW 速率表达式
# ============================================================

def calculate_local_lhhw_rates(par, T, Tave, fCO2, fH2, fCH3OH, fH2O, fCO):
    """
    在固定床某个位置，根据局部 fugacity 计算 r1 和 r2。
    """

    params = unpack_params(par)

    K_f1, K_f2 = calculate_equilibrium_constants(T)

    K_f1 = max(float(np.asarray(K_f1)), EPS)
    K_f2 = max(float(np.asarray(K_f2)), EPS)

    fCO2 = max(float(fCO2), EPS)
    fH2 = max(float(fH2), EPS)
    fCH3OH = max(float(fCH3OH), 0.0)
    fH2O = max(float(fH2O), 0.0)
    fCO = max(float(fCO), 0.0)

    # ln k = ln_k_ref - E/R * (1/T - 1/Tave)
    k1_eff = params["k1_eff_ref"] * np.exp(
        -params["E1_over_R"] * (1.0 / T - 1.0 / Tave)
    )

    k2_eff = params["k2_eff_ref"] * np.exp(
        -params["E2_over_R"] * (1.0 / T - 1.0 / Tave)
    )

    # 吸附项
    ads_carbon = 1.0 + params["KCO2"] * fCO2 + params["KCO"] * fCO
    ads_hydrogen_water = np.sqrt(fH2) + params["KH2O_H2"] * fH2O

    denominator = max(float(ads_carbon * ads_hydrogen_water), EPS)

    # R1 driving force
    driving_1 = (
        fCO2 * fH2 ** 1.5
        - (fCH3OH * fH2O) / (K_f1 * fH2 ** 1.5)
    )

    # R2 driving force
    driving_2 = (
        fCO2 * fH2
        - (fCO * fH2O) / K_f2
    )

    r1 = k1_eff * driving_1 / denominator
    r2 = k2_eff * driving_2 / denominator

    return float(r1), float(r2)


# ============================================================
# 9. 入口流量
# ============================================================

def calculate_inlet_flows_from_hc(HC):
    """
    根据 H2/CO2 ratio 构造入口摩尔流量。

    flow 顺序：
        [CO2, H2, CH3OH, H2O, CO]
    """

    HC = float(HC)

    y_co2_in = 1.0 / (1.0 + HC)
    y_h2_in = HC / (1.0 + HC)

    flows_in = np.array([
        y_co2_in * F_TOTAL_IN,
        y_h2_in * F_TOTAL_IN,
        0.0,
        0.0,
        0.0,
    ], dtype=float)

    return flows_in


# ============================================================
# 10. PFR 微分方程
# ============================================================

def calculate_pfr_derivatives(W, flows, par, T, p_MPa, Tave):
    """
    计算 dF/dW。

    flows 顺序：
        [CO2, H2, CH3OH, H2O, CO]
    """

    flows = np.maximum(flows, 0.0)

    F_total = max(float(np.sum(flows)), EPS)        #保护

    y = flows / F_total

    y_CO2, y_H2, y_CH3OH, y_H2O, y_CO = y

    # 理想 fugacity 近似
    fCO2 = y_CO2 * p_MPa
    fH2 = y_H2 * p_MPa
    fCH3OH = y_CH3OH * p_MPa
    fH2O = y_H2O * p_MPa
    fCO = y_CO * p_MPa

    r1, r2 = calculate_local_lhhw_rates(
        par=par,
        T=T,
        Tave=Tave,
        fCO2=fCO2,
        fH2=fH2,
        fCH3OH=fCH3OH,
        fH2O=fH2O,
        fCO=fCO,
    )

    dF_dW = np.array([
        -(r1 + r2),
        -(3.0 * r1 + r2),
        r1,
        r1 + r2,
        r2,
    ], dtype=float)

    return dF_dW


# ============================================================
# 11. 对单个实验点积分
# ============================================================

def integrate_one_experiment(row, par, Tave):
    """
    对excel每一行实验条件进行 PFR 积分。

    积分范围：
        W = 0 到 W = W_CAT_KG
    """
#入口条件
    inlet = calculate_inlet_flows_from_hc(row["HC"])

    T = float(row["T"])
    p_MPa = float(row["p_MPa"])

    flows = inlet.copy()

    h = W_CAT_KG / RK4_STEPS        #催化剂的量等于total/份数
    W = 0.0                         #从0开始积分

    for _ in range(RK4_STEPS):
        k1 = calculate_pfr_derivatives(W, flows, par, T, p_MPa, Tave)

        k2 = calculate_pfr_derivatives(
            W + 0.5 * h,
            flows + 0.5 * h * k1,
            par,
            T,
            p_MPa,
            Tave,
        )

        k3 = calculate_pfr_derivatives(
            W + 0.5 * h,
            flows + 0.5 * h * k2,
            par,
            T,
            p_MPa,
            Tave,
        )

        k4 = calculate_pfr_derivatives(
            W + h,
            flows + h * k3,
            par,
            T,
            p_MPa,
            Tave,
        )

        flows = flows + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        flows = np.maximum(flows, 0.0)

        if not np.all(np.isfinite(flows)):
            raise FloatingPointError("Non-finite molar flow during PFR integration.")

        W += h

    outlet = flows

    rCH3OH_pred = (outlet[2] - inlet[2]) / W_CAT_KG
    rCO_pred = (outlet[4] - inlet[4]) / W_CAT_KG

    return rCH3OH_pred, rCO_pred, outlet


# ============================================================
# 12. 对所有实验点预测
# ============================================================

def calculate_integral_predictions(par, df_group, Tave):
    """
    对所有实验点进行积分预测。
    """

    pred_rates = []
    outlet_flows = []

    for _, row in df_group.iterrows():
        rCH3OH_pred, rCO_pred, outlet = integrate_one_experiment(
            row=row,
            par=par,
            Tave=Tave,
        )

        pred_rates.append([rCH3OH_pred, rCO_pred])
        outlet_flows.append(outlet)

    pred_rates = np.asarray(pred_rates, dtype=float)
    outlet_flows = np.asarray(outlet_flows, dtype=float)

    rCH3OH_pred = pred_rates[:, 0]
    rCO_pred = pred_rates[:, 1]

    return rCH3OH_pred, rCO_pred, outlet_flows


# ============================================================
# 13. 目标函数
# objective(par, df_group, Tave, rCH3OH_exp, rCO_exp)
# 返回相对误差平方和
# ============================================================

def objective_integral(par, df_group, Tave, rCH3OH_exp, rCO_exp):
    """
    差分进化法目标函数。

    只拟合：
        rCH3OH
        rCO
    """

    try:
        rCH3OH_pred, rCO_pred, _ = calculate_integral_predictions(
            par=par,
            df_group=df_group,
            Tave=Tave,
        )

        denom_meoh = np.maximum(np.abs(rCH3OH_exp), 1e-6)
        denom_co = np.maximum(np.abs(rCO_exp), 1e-6)

        sse_meoh = np.sum(((rCH3OH_pred - rCH3OH_exp) / denom_meoh) ** 2)
        sse_co = np.sum(((rCO_pred - rCO_exp) / denom_co) ** 2)

        total_sse = sse_meoh + sse_co

        if not np.isfinite(total_sse):
            return 1e30

        return float(total_sse)

    except Exception as error:
        print("objective_integral 出错:", error)
        return 1e30


# ============================================================
# 14. 评价指标
# ============================================================

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


# ============================================================
# 15. 边界诊断
# ============================================================

def diagnose_boundary_status(par):
    """
    检查参数是否接近上下边界。
    """

    status = {}
    warnings = []

    for name, value, bound in zip(PARAMETER_NAMES, par, BOUNDS):
        lower, upper = bound
        width = upper - lower

        lower_threshold = lower + 0.02 * width
        upper_threshold = upper - 0.02 * width

        if value <= lower_threshold:
            param_status = "near_lower_bound"
            warnings.append(f"{name} near lower bound")
        elif value >= upper_threshold:
            param_status = "near_upper_bound"
            warnings.append(f"{name} near upper bound")
        else:
            param_status = "inside_bounds"

        status[f"{name}_boundary_status"] = param_status

    if len(warnings) == 0:
        warning_text = "none"
    else:
        warning_text = "; ".join(warnings)

    return status, warning_text


# ============================================================
# 16. 画图
# ============================================================

def make_single_parity_plot(exp, pred, xlabel, ylabel, title, save_path):
    plt.figure(figsize=(6, 6))

    plt.scatter(exp, pred, alpha=0.75)

    min_val = min(np.min(exp), np.min(pred))
    max_val = max(np.max(exp), np.max(pred))

    plt.plot([min_val, max_val], [min_val, max_val], "k--")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def make_residual_vs_temperature_plot(T, residual, ylabel, title, save_path):
    plt.figure(figsize=(6, 4))

    plt.scatter(T, residual, alpha=0.75)
    plt.axhline(0, color="k", linestyle="--", linewidth=1)

    plt.xlabel("Temperature / K")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ============================================================
# 17. 单个积分 LHHW 模型拟合
# 完全按照你原来的 fit_model 逻辑写
# ============================================================

def fit_integral_model(df_group):
    """
    拟合 Model E integral LHHW。

    按你原来的差分进化逻辑：
        result = differential_evolution(
            objective_func,
            bounds=bounds,
            args=(...),
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
    """

    rCH3OH_exp = df_group["rCH3OH"].to_numpy(dtype=float)
    rCO_exp = df_group["rCO"].to_numpy(dtype=float)

    temperature = df_group["T"].to_numpy(dtype=float)

    # Tave 重参数化
    Tave = float(np.mean(temperature))

    print("\n" + "=" * 60)
    print("开始拟合 Model E integral LHHW")
    print(f"数据点数 = {len(df_group)}")
    print(f"Tave = {Tave:.2f} K")
    print("=" * 60)

    result = differential_evolution(
        objective_integral,
        bounds=BOUNDS,
        args=(df_group, Tave, rCH3OH_exp, rCO_exp),
        seed=DE_SEED,
        popsize=DE_POPSIZE,
        maxiter=DE_MAXITER,
        tol=DE_TOL,
        mutation=DE_MUTATION,
        recombination=DE_RECOMBINATION,
        polish=DE_POLISH,
        workers=DE_WORKERS,
        updating=DE_UPDATING,
        disp=True,
    )

    par_opt = result.x

    rCH3OH_pred, rCO_pred, outlet_flows = calculate_integral_predictions(
        par=par_opt,
        df_group=df_group,
        Tave=Tave,
    )

    params = unpack_params(par_opt)
    params_report = add_energy_units(params)

    r2_meoh = calc_r2(rCH3OH_exp, rCH3OH_pred)
    r2_co = calc_r2(rCO_exp, rCO_pred)
    avg_r2 = np.nanmean([r2_meoh, r2_co])

    rmse_meoh = calc_rmse(rCH3OH_exp, rCH3OH_pred)
    rmse_co = calc_rmse(rCO_exp, rCO_pred)

    mre_meoh = calc_mre(rCH3OH_exp, rCH3OH_pred)
    mre_co = calc_mre(rCO_exp, rCO_pred)

    boundary_status, boundary_warning = diagnose_boundary_status(par_opt)

    result_df = df_group.copy()

    result_df["F_CO2_out_mol_s"] = outlet_flows[:, 0]
    result_df["F_H2_out_mol_s"] = outlet_flows[:, 1]
    result_df["F_CH3OH_out_mol_s"] = outlet_flows[:, 2]
    result_df["F_H2O_out_mol_s"] = outlet_flows[:, 3]
    result_df["F_CO_out_mol_s"] = outlet_flows[:, 4]

    result_df["rCH3OH_pred"] = rCH3OH_pred
    result_df["rCO_pred"] = rCO_pred

    result_df["res_CH3OH"] = result_df["rCH3OH_pred"] - result_df["rCH3OH"]
    result_df["res_CO"] = result_df["rCO_pred"] - result_df["rCO"]

    result_df["rel_error_rCH3OH_%"] = (
        np.abs((rCH3OH_pred - rCH3OH_exp) / np.maximum(np.abs(rCH3OH_exp), 1e-12))
        * 100.0
    )

    result_df["rel_error_rCO_%"] = (
        np.abs((rCO_pred - rCO_exp) / np.maximum(np.abs(rCO_exp), 1e-12))
        * 100.0
    )

    summary = {
        "model": MODEL_NAME,
        "Tave": Tave,
        "optimizer_success": result.success,
        "optimizer_message": str(result.message),
        "objective": result.fun,
        "r2_ch3oh": r2_meoh,
        "r2_co": r2_co,
        "avg_r2": avg_r2,
        "rmse_ch3oh": rmse_meoh,
        "rmse_co": rmse_co,
        "mre_ch3oh_%": mre_meoh,
        "mre_co_%": mre_co,
        "boundary_warning": boundary_warning,
    }

    params_output = {
        "model": MODEL_NAME,
        "Tave": Tave,

        "ln_k1_eff_ref": par_opt[0],
        "k1_eff_ref": params_report["k1_eff_ref"],
        "E1_over_R": params_report["E1_over_R"],
        "E1_kJ_per_mol": params_report["E1_kJ_per_mol"],

        "ln_k2_eff_ref": par_opt[2],
        "k2_eff_ref": params_report["k2_eff_ref"],
        "E2_over_R": params_report["E2_over_R"],
        "E2_kJ_per_mol": params_report["E2_kJ_per_mol"],

        "ln_KCO2": par_opt[4],
        "KCO2": params_report["KCO2"],

        "ln_KCO": par_opt[5],
        "KCO": params_report["KCO"],

        "ln_KH2O_H2": par_opt[6],
        "KH2O_H2": params_report["KH2O_H2"],

        "boundary_warning": boundary_warning,
    }

    params_output.update(boundary_status)

    print("\n模型 Model E integral LHHW 拟合完成")
    print("success =", result.success)
    print("message =", result.message)
    print("objective =", result.fun)
    print(f"Tave = {Tave:.2f} K")
    print(f"r2_ch3oh = {r2_meoh:.6f}")
    print(f"r2_co    = {r2_co:.6f}")
    print(f"avg_r2   = {avg_r2:.6f}")
    print(f"mre_ch3oh% = {mre_meoh:.4f}")
    print(f"mre_co%    = {mre_co:.4f}")
    print("boundary_warning =", boundary_warning)

    return summary, params_output, result_df


# ============================================================
# 18. 保存结果
# ============================================================

def save_results(summary, params, result_df):
    """
    保存 summary、parameters、predictions 和图。
    """

    summary_df = pd.DataFrame([summary])
    params_df = pd.DataFrame([params])

    summary_df.to_excel(OUT_DIR / "model_E_integral_summary.xlsx", index=False)
    params_df.to_excel(OUT_DIR / "model_E_integral_parameters.xlsx", index=False)
    result_df.to_excel(OUT_DIR / "model_E_integral_predictions.xlsx", index=False)

    make_single_parity_plot(
        exp=result_df["rCH3OH"].values,
        pred=result_df["rCH3OH_pred"].values,
        xlabel="Experimental CH3OH",
        ylabel="Predicted CH3OH",
        title="Model E Integral LHHW - CH3OH",
        save_path=OUT_DIR / "parity_CH3OH_model_E_integral.png",
    )

    make_single_parity_plot(
        exp=result_df["rCO"].values,
        pred=result_df["rCO_pred"].values,
        xlabel="Experimental CO",
        ylabel="Predicted CO",
        title="Model E Integral LHHW - CO",
        save_path=OUT_DIR / "parity_CO_model_E_integral.png",
    )

    make_residual_vs_temperature_plot(
        T=result_df["T"].values,
        residual=result_df["res_CH3OH"].values,
        ylabel="Residual CH3OH",
        title="Model E Integral LHHW - CH3OH residual vs T",
        save_path=OUT_DIR / "residual_vs_T_CH3OH_model_E_integral.png",
    )

    make_residual_vs_temperature_plot(
        T=result_df["T"].values,
        residual=result_df["res_CO"].values,
        ylabel="Residual CO",
        title="Model E Integral LHHW - CO residual vs T",
        save_path=OUT_DIR / "residual_vs_T_CO_model_E_integral.png",
    )

    log_row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_name": MODEL_NAME,
        "data_file": str(EXCEL_FILE),
        "n_points": len(result_df),
        "W_cat_kg": W_CAT_KG,
        "F_total_in_mol_s": F_TOTAL_IN,
        "RK4_STEPS": RK4_STEPS,
        "DE_SEED": DE_SEED,
        "DE_POPSIZE": DE_POPSIZE,
        "DE_MAXITER": DE_MAXITER,
        "DE_TOL": DE_TOL,
        "DE_MUTATION": str(DE_MUTATION),
        "DE_RECOMBINATION": DE_RECOMBINATION,
        "DE_POLISH": DE_POLISH,
        **summary,
        **params,
    }

    log_df = pd.DataFrame([log_row])

    if EXPERIMENT_LOG_PATH.exists():
        old_log = pd.read_csv(EXPERIMENT_LOG_PATH)
        log_df = pd.concat([old_log, log_df], ignore_index=True)

    log_df.to_csv(EXPERIMENT_LOG_PATH, index=False)


# ============================================================
# 19. 主程序
# ============================================================

def main():
    print("\n==============================")
    print("Model E integral LHHW fitting")
    print("==============================")
    print(f"Excel file: {EXCEL_FILE}")
    print(f"Output folder: {OUT_DIR}")
    print(f"W_cat_kg = {W_CAT_KG}")
    print(f"F_TOTAL_IN = {F_TOTAL_IN:.12g} mol/s")
    print(f"RK4_STEPS = {RK4_STEPS}")
    print(f"DE seed = {DE_SEED}")
    print(f"DE popsize = {DE_POPSIZE}")
    print(f"DE maxiter = {DE_MAXITER}")
    print(f"DE mutation = {DE_MUTATION}")
    print(f"DE recombination = {DE_RECOMBINATION}")
    print(f"DE polish = {DE_POLISH}")
    print(f"ln_KCO upper bound = {LN_KCO_UPPER}")

    df = load_data()

    summary, params, result_df = fit_integral_model(df)

    save_results(summary, params, result_df)

    print("\n全部拟合完成")
    print("\nSummary:")
    print(pd.DataFrame([summary]))

    print("\nParameters:")
    print(pd.DataFrame([params]))

    print("\n结果已保存到文件夹：")
    print(OUT_DIR)


if __name__ == "__main__":
    main()