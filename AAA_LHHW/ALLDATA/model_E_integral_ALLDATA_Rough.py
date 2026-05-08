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

DE_SEED = 42

MODEL_NAME = "Global_rough_integral_two_reaction_LHHW_ALLDATA_real_W_flow"

BASE_DIR = Path(__file__).resolve().parent

OUT_DIR = BASE_DIR / "output" / "global_rough_integral_LHHW_ALLDATA_real_W_flow"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT_LOG_PATH = BASE_DIR / "output" / "experiment_log.csv"


# ============================================================
# 1. 自动寻找数据文件
# ============================================================

def find_data_file():

    candidate_paths = [
        BASE_DIR / "ALLDATA" / "full data.xlsx",
        BASE_DIR / "full data.xlsx",
        BASE_DIR.parent / "ALLDATA" / "full data.xlsx",
        Path.cwd() / "ALLDATA" / "full data.xlsx",
        Path.cwd() / "full data.xlsx",
    ]

    for path in candidate_paths:
        if path.exists():
            return path

    message = "Cannot find full data.xlsx. The script searched:\n"
    for path in candidate_paths:
        message += f"  {path}\n"

    raise FileNotFoundError(message)


EXCEL_FILE = find_data_file()


# ============================================================
# 2. 实验条件
# ============================================================

# 标准摩尔体积，L/mol
STANDARD_MOLAR_VOLUME_L_PER_MOL = 22.414

# rough 版积分步数
RK4_STEPS = 12


# 不同 GHSV 对应的真实实验条件
# W_cat 单位 kg
# flow 单位 mL/min
EXPERIMENT_CONDITIONS = {
    4000: {
        "W_cat_kg": 0.00045,
        "flow_mL_min": 31.5,
    },
    8000: {
        "W_cat_kg": 0.00030,
        "flow_mL_min": 42.0,
    },
    12000: {
        "W_cat_kg": 0.00020,
        "flow_mL_min": 42.0,
    },
}


# ============================================================
# 3. 差分进化法设置，rough 版
# ============================================================

DE_POPSIZE = 6
DE_MAXITER = 60
DE_TOL = 1e-5
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

BOUNDS = [
    (-30.0, 10.0),       # ln_k1_eff_ref
    (0.0, 30000.0),      # E1_over_R
    (-30.0, 10.0),       # ln_k2_eff_ref
    (0.0, 30000.0),      # E2_over_R
    (-20.0, 10.0),       # ln_KCO2
    (-20.0, 10.0),       # ln_KCO
    (-20.0, 10.0),       # ln_KH2O_H2
]


# ============================================================
# 5. 平衡常数
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
# 6. 读取数据
# ============================================================

def load_data():

    df = pd.read_excel(EXCEL_FILE, header=0)

    # 你的 Excel 第二行是单位行，所以这里删除
    df = df.iloc[1:].copy()

    df.columns = [str(col).strip().replace("\n", " ") for col in df.columns]

    df = df.rename(columns={
        "H/C": "HC",
        "H/C RATIO": "HC",
        "p": "p_MPa",
        "GHSV": "GHSV",
        "T": "T",
        "fCO2": "fCO2",
        "fH2": "fH2",
        "fCH3OH": "fCH3OH",
        "fH2O": "fH2O",
        "fCO": "fCO",
        "r CH3OH": "rCH3OH",
        "r CO": "rCO",
        "r CO2": "rCO2",
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
        print("当前读到的列名是：")
        print(df.columns.tolist())
        raise KeyError(f"缺少列: {missing_cols}")

    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required_cols).copy()

    df = df[df["GHSV"].isin([4000, 8000, 12000])].copy()

    if df.empty:
        raise ValueError("No valid rows were found for GHSV = 4000, 8000, 12000.")

    # 给每一行加上真实催化剂装填量和真实气体流速
    df["W_cat_kg"] = df["GHSV"].map(
        lambda x: EXPERIMENT_CONDITIONS[int(x)]["W_cat_kg"]
    )

    df["flow_mL_min"] = df["GHSV"].map(
        lambda x: EXPERIMENT_CONDITIONS[int(x)]["flow_mL_min"]
    )

    df["Kf1"], df["Kf2"] = calculate_equilibrium_constants(df["T"].values)

    print("实际列名如下：")
    print(df.columns.tolist())

    print(f"\n总数据点数: {len(df)}")
    print("\nGHSV 分组统计:")
    print(df["GHSV"].value_counts().sort_index())

    print("\n实验条件检查:")
    print(
        df.groupby("GHSV")[["W_cat_kg", "flow_mL_min"]]
        .first()
        .reset_index()
    )

    print("\n前五行数据:")
    print(df.head())

    return df


# ============================================================
# 7. 参数转换
# ============================================================

def unpack_params(par):

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

    report = params.copy()
    report["E1_kJ_per_mol"] = params["E1_over_R"] * 8.314 / 1000.0
    report["E2_kJ_per_mol"] = params["E2_over_R"] * 8.314 / 1000.0

    return report


# ============================================================
# 8. 局部 LHHW 速率表达式
# ============================================================

def calculate_local_lhhw_rates(par, T, Tave, fCO2, fH2, fCH3OH, fH2O, fCO):

    params = unpack_params(par)

    K_f1, K_f2 = calculate_equilibrium_constants(T)

    K_f1 = max(float(np.asarray(K_f1)), EPS)
    K_f2 = max(float(np.asarray(K_f2)), EPS)

    fCO2 = max(float(fCO2), EPS)
    fH2 = max(float(fH2), EPS)
    fCH3OH = max(float(fCH3OH), 0.0)
    fH2O = max(float(fH2O), 0.0)
    fCO = max(float(fCO), 0.0)

    k1_eff = params["k1_eff_ref"] * np.exp(
        -params["E1_over_R"] * (1.0 / T - 1.0 / Tave)
    )

    k2_eff = params["k2_eff_ref"] * np.exp(
        -params["E2_over_R"] * (1.0 / T - 1.0 / Tave)
    )

    ads_carbon = 1.0 + params["KCO2"] * fCO2 + params["KCO"] * fCO
    ads_hydrogen_water = np.sqrt(fH2) + params["KH2O_H2"] * fH2O

    denominator = max(float(ads_carbon * ads_hydrogen_water), EPS)

    driving_1 = (
        fCO2 * fH2 ** 1.5
        - (fCH3OH * fH2O) / (K_f1 * fH2 ** 1.5)
    )

    driving_2 = (
        fCO2 * fH2
        - (fCO * fH2O) / K_f2
    )

    r1 = k1_eff * driving_1 / denominator
    r2 = k2_eff * driving_2 / denominator

    return float(r1), float(r2)


# ============================================================
# 9. 根据真实 flow 和 H/C 计算入口流量
# ============================================================

def calculate_total_inlet_flow_from_flow_mL_min(flow_mL_min):
    """
    flow_mL_min:
        标况体积流量，mL/min

    转换为：
        mol/s
    """

    flow_L_s = float(flow_mL_min) / 1000.0 / 60.0

    F_total_in = flow_L_s / STANDARD_MOLAR_VOLUME_L_PER_MOL

    return F_total_in


def calculate_inlet_flows_from_row(row):
    """
    根据每一行的 H/C 和真实 flow_mL_min 构造入口摩尔流量。

    flow 顺序：
        [CO2, H2, CH3OH, H2O, CO]
    """

    HC = float(row["HC"])
    flow_mL_min = float(row["flow_mL_min"])

    F_total_in = calculate_total_inlet_flow_from_flow_mL_min(flow_mL_min)

    y_co2_in = 1.0 / (1.0 + HC)
    y_h2_in = HC / (1.0 + HC)

    flows_in = np.array([
        y_co2_in * F_total_in,
        y_h2_in * F_total_in,
        0.0,
        0.0,
        0.0,
    ], dtype=float)

    return flows_in


# ============================================================
# 10. PFR 微分方程
# ============================================================

def calculate_pfr_derivatives(W, flows, par, T, p_MPa, Tave):

    flows = np.maximum(flows, 0.0)

    F_total = max(float(np.sum(flows)), EPS)

    y = flows / F_total

    y_CO2, y_H2, y_CH3OH, y_H2O, y_CO = y

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

    inlet = calculate_inlet_flows_from_row(row)

    T = float(row["T"])
    p_MPa = float(row["p_MPa"])
    W_cat_kg = float(row["W_cat_kg"])

    flows = inlet.copy()

    h = W_cat_kg / RK4_STEPS
    W = 0.0

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

    rCH3OH_pred = (outlet[2] - inlet[2]) / W_cat_kg
    rCO_pred = (outlet[4] - inlet[4]) / W_cat_kg

    return rCH3OH_pred, rCO_pred, outlet, inlet


# ============================================================
# 12. 对所有实验点预测
# ============================================================

def calculate_integral_predictions(par, df_group, Tave):

    pred_rates = []
    outlet_flows = []
    inlet_flows = []

    for _, row in df_group.iterrows():

        rCH3OH_pred, rCO_pred, outlet, inlet = integrate_one_experiment(
            row=row,
            par=par,
            Tave=Tave,
        )

        pred_rates.append([rCH3OH_pred, rCO_pred])
        outlet_flows.append(outlet)
        inlet_flows.append(inlet)

    pred_rates = np.asarray(pred_rates, dtype=float)
    outlet_flows = np.asarray(outlet_flows, dtype=float)
    inlet_flows = np.asarray(inlet_flows, dtype=float)

    rCH3OH_pred = pred_rates[:, 0]
    rCO_pred = pred_rates[:, 1]

    return rCH3OH_pred, rCO_pred, outlet_flows, inlet_flows


# ============================================================
# 13. 目标函数
# ============================================================

def objective_integral(par, df_group, Tave, rCH3OH_exp, rCO_exp):

    try:
        rCH3OH_pred, rCO_pred, _, _ = calculate_integral_predictions(
            par=par,
            df_group=df_group,
            Tave=Tave,
        )

        denom_meoh = np.maximum(np.abs(rCH3OH_exp), 1e-6)
        denom_co = np.maximum(np.abs(rCO_exp), 1e-6)

        err_meoh = (rCH3OH_pred - rCH3OH_exp) / denom_meoh
        err_co = (rCO_pred - rCO_exp) / denom_co

        total_sse = np.sum(err_meoh ** 2) + np.sum(err_co ** 2)

        if not np.isfinite(total_sse):
            return 1e30

        return float(total_sse)

    except Exception:
        return 1e30


# ============================================================
# 14. 评价指标
# ============================================================

def calc_r2(y_exp, y_pred):

    y_exp = np.asarray(y_exp, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    ss_res = np.sum((y_exp - y_pred) ** 2)
    ss_tot = np.sum((y_exp - np.mean(y_exp)) ** 2)

    if ss_tot < 1e-12:
        return np.nan

    return 1.0 - ss_res / ss_tot


def calc_rmse(y_exp, y_pred):

    y_exp = np.asarray(y_exp, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return np.sqrt(np.mean((y_exp - y_pred) ** 2))


def calc_mre(y_exp, y_pred):

    y_exp = np.asarray(y_exp, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denom = np.maximum(np.abs(y_exp), 1e-12)

    return np.mean(np.abs((y_pred - y_exp) / denom)) * 100.0


# ============================================================
# 15. 边界诊断
# ============================================================

def diagnose_boundary_status(par):

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

def make_global_parity_plot(result_df, exp_col, pred_col, title, save_path):

    plt.figure(figsize=(6, 6))

    for ghsv in sorted(result_df["GHSV"].unique()):
        sub = result_df[result_df["GHSV"] == ghsv]
        plt.scatter(
            sub[exp_col],
            sub[pred_col],
            alpha=0.75,
            label=f"GHSV = {int(ghsv)}",
        )

    min_val = min(result_df[exp_col].min(), result_df[pred_col].min())
    max_val = max(result_df[exp_col].max(), result_df[pred_col].max())

    plt.plot([min_val, max_val], [min_val, max_val], "k--")

    plt.xlabel(f"Experimental {exp_col}")
    plt.ylabel(f"Predicted {pred_col}")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def make_residual_vs_temperature_plot(result_df, residual_col, title, save_path):

    plt.figure(figsize=(6, 4))

    for ghsv in sorted(result_df["GHSV"].unique()):
        sub = result_df[result_df["GHSV"] == ghsv]
        plt.scatter(
            sub["T"],
            sub[residual_col],
            alpha=0.75,
            label=f"GHSV = {int(ghsv)}",
        )

    plt.axhline(0, color="k", linestyle="--", linewidth=1)

    plt.xlabel("Temperature / K")
    plt.ylabel(residual_col)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def make_residual_vs_ghsv_plot(result_df, residual_col, title, save_path):

    plt.figure(figsize=(6, 4))

    plt.scatter(
        result_df["GHSV"],
        result_df[residual_col],
        alpha=0.75,
    )

    plt.axhline(0, color="k", linestyle="--", linewidth=1)

    plt.xlabel("GHSV / mL g^-1 h^-1")
    plt.ylabel(residual_col)
    plt.title(title)
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ============================================================
# 17. 分 GHSV 评价
# ============================================================

def calculate_group_metrics(result_df):

    rows = []

    for ghsv in sorted(result_df["GHSV"].unique()):

        sub = result_df[result_df["GHSV"] == ghsv]

        row = {
            "GHSV": ghsv,
            "n_points": len(sub),
            "W_cat_kg": sub["W_cat_kg"].iloc[0],
            "flow_mL_min": sub["flow_mL_min"].iloc[0],

            "r2_ch3oh": calc_r2(sub["rCH3OH"], sub["rCH3OH_pred"]),
            "r2_co": calc_r2(sub["rCO"], sub["rCO_pred"]),

            "rmse_ch3oh": calc_rmse(sub["rCH3OH"], sub["rCH3OH_pred"]),
            "rmse_co": calc_rmse(sub["rCO"], sub["rCO_pred"]),

            "mre_ch3oh_%": calc_mre(sub["rCH3OH"], sub["rCH3OH_pred"]),
            "mre_co_%": calc_mre(sub["rCO"], sub["rCO_pred"]),
        }

        row["avg_r2"] = np.nanmean([row["r2_ch3oh"], row["r2_co"]])

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 18. 全局拟合
# ============================================================

def fit_global_integral_model(df_group):

    rCH3OH_exp = df_group["rCH3OH"].to_numpy(dtype=float)
    rCO_exp = df_group["rCO"].to_numpy(dtype=float)

    temperature = df_group["T"].to_numpy(dtype=float)

    Tave = float(np.mean(temperature))

    print("\n" + "=" * 70)
    print("开始全局拟合 Global Rough Integral LHHW")
    print(f"总数据点数 = {len(df_group)}")
    print(f"Tave = {Tave:.2f} K")
    print("GHSV included:", sorted(df_group["GHSV"].unique()))
    print("=" * 70)

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

    rCH3OH_pred, rCO_pred, outlet_flows, inlet_flows = calculate_integral_predictions(
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

    result_df["F_CO2_in_mol_s"] = inlet_flows[:, 0]
    result_df["F_H2_in_mol_s"] = inlet_flows[:, 1]
    result_df["F_CH3OH_in_mol_s"] = inlet_flows[:, 2]
    result_df["F_H2O_in_mol_s"] = inlet_flows[:, 3]
    result_df["F_CO_in_mol_s"] = inlet_flows[:, 4]

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
        "fit_type": "global_all_GHSV_rough_real_W_flow",
        "Tave": Tave,
        "n_points": len(df_group),
        "GHSV_included": str(sorted(df_group["GHSV"].unique())),
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

    print("\n全局 rough 拟合完成")
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

    group_metrics = calculate_group_metrics(result_df)

    print("\n分 GHSV 评价:")
    print(group_metrics)

    return summary, params_output, result_df, group_metrics


# ============================================================
# 19. 保存结果
# ============================================================

def save_results(summary, params, result_df, group_metrics):

    summary_df = pd.DataFrame([summary])
    params_df = pd.DataFrame([params])

    summary_df.to_excel(OUT_DIR / "global_rough_summary.xlsx", index=False)
    params_df.to_excel(OUT_DIR / "global_rough_parameters.xlsx", index=False)
    result_df.to_excel(OUT_DIR / "global_rough_predictions.xlsx", index=False)
    group_metrics.to_excel(OUT_DIR / "global_rough_group_metrics_by_GHSV.xlsx", index=False)

    make_global_parity_plot(
        result_df=result_df,
        exp_col="rCH3OH",
        pred_col="rCH3OH_pred",
        title="Global Rough Integral LHHW CH3OH",
        save_path=OUT_DIR / "parity_CH3OH_global_rough.png",
    )

    make_global_parity_plot(
        result_df=result_df,
        exp_col="rCO",
        pred_col="rCO_pred",
        title="Global Rough Integral LHHW CO",
        save_path=OUT_DIR / "parity_CO_global_rough.png",
    )

    make_residual_vs_temperature_plot(
        result_df=result_df,
        residual_col="res_CH3OH",
        title="Global Rough Integral LHHW CH3OH residual vs T",
        save_path=OUT_DIR / "residual_vs_T_CH3OH_global_rough.png",
    )

    make_residual_vs_temperature_plot(
        result_df=result_df,
        residual_col="res_CO",
        title="Global Rough Integral LHHW CO residual vs T",
        save_path=OUT_DIR / "residual_vs_T_CO_global_rough.png",
    )

    make_residual_vs_ghsv_plot(
        result_df=result_df,
        residual_col="res_CH3OH",
        title="Global Rough Integral LHHW CH3OH residual vs GHSV",
        save_path=OUT_DIR / "residual_vs_GHSV_CH3OH_global_rough.png",
    )

    make_residual_vs_ghsv_plot(
        result_df=result_df,
        residual_col="res_CO",
        title="Global Rough Integral LHHW CO residual vs GHSV",
        save_path=OUT_DIR / "residual_vs_GHSV_CO_global_rough.png",
    )

    log_row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_name": MODEL_NAME,
        "data_file": str(EXCEL_FILE),
        "n_points": len(result_df),
        "RK4_STEPS": RK4_STEPS,
        "DE_SEED": DE_SEED,
        "DE_POPSIZE": DE_POPSIZE,
        "DE_MAXITER": DE_MAXITER,
        "DE_TOL": DE_TOL,
        "DE_MUTATION": str(DE_MUTATION),
        "DE_RECOMBINATION": DE_RECOMBINATION,
        "DE_POLISH": DE_POLISH,
        "conditions": str(EXPERIMENT_CONDITIONS),
        **summary,
        **params,
    }

    log_df = pd.DataFrame([log_row])

    if EXPERIMENT_LOG_PATH.exists():
        old_log = pd.read_csv(EXPERIMENT_LOG_PATH)
        log_df = pd.concat([old_log, log_df], ignore_index=True)

    log_df.to_csv(EXPERIMENT_LOG_PATH, index=False)


# ============================================================
# 20. 主程序
# ============================================================

def main():

    print("\n====================================================")
    print("Global rough integral LHHW fitting with real W/flow")
    print("====================================================")
    print(f"Excel file: {EXCEL_FILE}")
    print(f"Output folder: {OUT_DIR}")
    print(f"RK4_STEPS = {RK4_STEPS}")
    print(f"DE seed = {DE_SEED}")
    print(f"DE popsize = {DE_POPSIZE}")
    print(f"DE maxiter = {DE_MAXITER}")
    print(f"DE mutation = {DE_MUTATION}")
    print(f"DE recombination = {DE_RECOMBINATION}")
    print(f"DE polish = {DE_POLISH}")

    print("\n实验条件:")
    for ghsv, condition in EXPERIMENT_CONDITIONS.items():
        print(
            f"GHSV = {ghsv}, "
            f"W_cat = {condition['W_cat_kg']} kg, "
            f"flow = {condition['flow_mL_min']} mL/min"
        )

    df = load_data()

    summary, params, result_df, group_metrics = fit_global_integral_model(df)

    save_results(summary, params, result_df, group_metrics)

    print("\n全部拟合完成")

    print("\nSummary:")
    print(pd.DataFrame([summary]))

    print("\nParameters:")
    print(pd.DataFrame([params]))

    print("\nGroup metrics by GHSV:")
    print(group_metrics)

    print("\n结果已保存到文件夹：")
    print(OUT_DIR)


if __name__ == "__main__":
    main()