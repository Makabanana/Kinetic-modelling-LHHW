#按照P,GHSV进行分类，参与拟合，对比原始的powerlaw和加入(1-beta)的两种形式
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 0. 常数与设置
# =========================
R = 8.314
eps = 1e-12

EXCEL_PATH = "full data.xlsx"
SKIPROWS = [1]   # 如果第二行不是单位行，就改成 None
OUTPUT_DIR = "loglinear_fit_fixed_GHSV_P"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

MIN_POINTS_PER_GROUP = 6   # 至少要大于参数数目 4，留一点余量


# =========================
# 1. 工具函数
# =========================
def safe_filename(s):
    s = str(s)
    s = re.sub(r'[\\/*?:"<>|]', "_", s)
    return s.replace(" ", "_")


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


def make_parity_plot(y_exp, y_pred, xlabel, ylabel, title, save_path):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_exp, y_pred, alpha=0.75)

    all_vals = np.concatenate([np.asarray(y_exp), np.asarray(y_pred)])
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
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# =========================
# 2. 读取数据
# =========================
def load_data():
    if SKIPROWS is None:
        df = pd.read_excel(EXCEL_PATH)
    else:
        df = pd.read_excel(EXCEL_PATH, skiprows=SKIPROWS)

    df.columns = df.columns.str.strip()

    df = df.rename(columns={
        "r CH3OH": "rMeOH",
        "r CO": "rCO",
        "r CO2": "rCO2"
    })

    print("实际列名如下：")
    print(df.columns.tolist())

    required_cols = [
        "GHSV", "p", "T",
        "fCO2", "fH2", "fCH3OH", "fH2O", "fCO",
        "rMeOH", "rCO"
    ]

    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"缺少列: {col}")

    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "H/C" in df.columns:
        df["H/C"] = pd.to_numeric(df["H/C"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    after = len(df)

    print(f"原始总数据点数: {before}")
    print(f"清洗后数据点数: {after}")
    print("GHSV 分组:", sorted(df["GHSV"].dropna().unique().tolist()))
    print("p 分组:", sorted(df["p"].dropna().unique().tolist()))
    print("T 点:", sorted(df["T"].dropna().unique().tolist()))

    return df


# =========================
# 3. 单个反应的 log linear 拟合
# 模型:
# ln r = ln A + E * ( -1 / (R T) ) + a ln fCO2 + b ln fH2
# =========================
def fit_loglinear_reaction(df_sub, rate_col, reaction_name, ghsv_val, p_val):
    df_fit = df_sub.copy()

    # 只保留可取对数的点
    valid_mask = (
        (df_fit[rate_col] > 0) &
        (df_fit["fCO2"] > 0) &
        (df_fit["fH2"] > 0) &
        (df_fit["T"] > 0)
    )

    df_fit = df_fit.loc[valid_mask].copy().reset_index(drop=True)

    if len(df_fit) < MIN_POINTS_PER_GROUP:
        return None, None, None, {
            "GHSV": ghsv_val,
            "p": p_val,
            "reaction": reaction_name,
            "n_points_raw": len(df_sub),
            "n_points_used": len(df_fit),
            "status": "skipped_too_few_positive_points"
        }

    y_log = np.log(df_fit[rate_col].to_numpy(dtype=float))
    x1 = np.ones(len(df_fit))
    x2 = -1.0 / (R * df_fit["T"].to_numpy(dtype=float))
    x3 = np.log(df_fit["fCO2"].to_numpy(dtype=float))
    x4 = np.log(df_fit["fH2"].to_numpy(dtype=float))

    X = np.column_stack([x1, x2, x3, x4])

    # 线性最小二乘
    coef, residuals, rank, s = np.linalg.lstsq(X, y_log, rcond=None)

    lnA, E, a, b = coef
    A = np.exp(lnA)

    y_log_pred = X @ coef
    r_pred = np.exp(y_log_pred)
    r_exp = df_fit[rate_col].to_numpy(dtype=float)

    # 指标：log 空间
    r2_log = calc_r2(y_log, y_log_pred)
    rmse_log = calc_rmse(y_log, y_log_pred)

    # 指标：线性空间
    r2_linear = calc_r2(r_exp, r_pred)
    rmse_linear = calc_rmse(r_exp, r_pred)
    mre_linear = calc_mre(r_exp, r_pred)

    # 条件数，帮助判断共线性
    cond_X = np.linalg.cond(X)

    result_df = df_fit.copy()
    result_df["reaction"] = reaction_name
    result_df["GHSV_fixed"] = ghsv_val
    result_df["p_fixed"] = p_val
    result_df["ln_rate_exp"] = y_log
    result_df["ln_rate_pred"] = y_log_pred
    result_df["rate_pred"] = r_pred
    result_df["rel_error_rate_%"] = np.abs((r_pred - r_exp) / np.maximum(np.abs(r_exp), 1e-12)) * 100

    params_row = {
        "GHSV": ghsv_val,
        "p": p_val,
        "reaction": reaction_name,
        "n_points_used": len(df_fit),
        "lnA": lnA,
        "A": A,
        "E": E,
        "a_fCO2": a,
        "b_fH2": b,
        "matrix_rank": rank,
        "condition_number": cond_X
    }

    summary_row = {
        "GHSV": ghsv_val,
        "p": p_val,
        "reaction": reaction_name,
        "n_points_raw": len(df_sub),
        "n_points_used": len(df_fit),
        "status": "ok",
        "r2_log": r2_log,
        "rmse_log": rmse_log,
        "r2_linear": r2_linear,
        "rmse_linear": rmse_linear,
        "mre_linear_%": mre_linear,
        "matrix_rank": rank,
        "condition_number": cond_X
    }

    print(f"\n组别 GHSV={ghsv_val}, p={p_val}, reaction={reaction_name}")
    print(f"n_points_used  = {len(df_fit)}")
    print(f"lnA            = {lnA:.6f}")
    print(f"A              = {A:.6e}")
    print(f"E              = {E:.6f}")
    print(f"a_fCO2         = {a:.6f}")
    print(f"b_fH2          = {b:.6f}")
    print(f"r2_log         = {r2_log:.6f}")
    print(f"r2_linear      = {r2_linear:.6f}")
    print(f"rmse_log       = {rmse_log:.6e}")
    print(f"rmse_linear    = {rmse_linear:.6e}")
    print(f"mre_linear %   = {mre_linear:.4f}")
    print(f"condition_no   = {cond_X:.6e}")

    return summary_row, params_row, result_df, None


# =========================
# 4. 主程序
# =========================
def main():
    df = load_data()

    ghsv_values = sorted(df["GHSV"].dropna().unique().tolist())
    p_values = sorted(df["p"].dropna().unique().tolist())

    all_summary_rows = []
    all_params_rows = []
    all_result_rows = []
    all_skipped_rows = []

    print("\n" + "=" * 72)
    print("开始按固定 GHSV 和固定 p 做 log linear 拟合")
    print("=" * 72)

    for ghsv_val in ghsv_values:
        for p_val in p_values:
            df_group = df[(df["GHSV"] == ghsv_val) & (df["p"] == p_val)].copy().reset_index(drop=True)

            if len(df_group) == 0:
                continue

            print("\n" + "=" * 72)
            print(f"开始处理组别: GHSV = {ghsv_val}, p = {p_val}")
            print(f"该组原始数据点数 = {len(df_group)}")
            print("=" * 72)

            group_tag = safe_filename(f"GHSV_{ghsv_val}_p_{p_val}")

            # 反应1：MeOH
            summary_meoh, params_meoh, result_meoh, skipped_meoh = fit_loglinear_reaction(
                df_sub=df_group,
                rate_col="rMeOH",
                reaction_name="MeOH",
                ghsv_val=ghsv_val,
                p_val=p_val
            )

            # 反应2：CO
            summary_co, params_co, result_co, skipped_co = fit_loglinear_reaction(
                df_sub=df_group,
                rate_col="rCO",
                reaction_name="CO",
                ghsv_val=ghsv_val,
                p_val=p_val
            )

            for x in [summary_meoh, summary_co]:
                if x is not None:
                    all_summary_rows.append(x)

            for x in [params_meoh, params_co]:
                if x is not None:
                    all_params_rows.append(x)

            for x in [result_meoh, result_co]:
                if x is not None:
                    all_result_rows.append(x)

            for x in [skipped_meoh, skipped_co]:
                if x is not None:
                    all_skipped_rows.append(x)

            # 画图
            if result_meoh is not None:
                make_parity_plot(
                    y_exp=result_meoh["rMeOH"].values,
                    y_pred=result_meoh["rate_pred"].values,
                    xlabel="Experimental MeOH rate",
                    ylabel="Predicted MeOH rate",
                    title=f"Parity Plot | MeOH | GHSV={ghsv_val}, p={p_val}",
                    save_path=os.path.join(PLOT_DIR, f"{group_tag}_MeOH_parity.png")
                )

                make_parity_plot(
                    y_exp=result_meoh["ln_rate_exp"].values,
                    y_pred=result_meoh["ln_rate_pred"].values,
                    xlabel="Experimental ln(MeOH rate)",
                    ylabel="Predicted ln(MeOH rate)",
                    title=f"Log Parity Plot | MeOH | GHSV={ghsv_val}, p={p_val}",
                    save_path=os.path.join(PLOT_DIR, f"{group_tag}_MeOH_log_parity.png")
                )

            if result_co is not None:
                make_parity_plot(
                    y_exp=result_co["rCO"].values,
                    y_pred=result_co["rate_pred"].values,
                    xlabel="Experimental CO rate",
                    ylabel="Predicted CO rate",
                    title=f"Parity Plot | CO | GHSV={ghsv_val}, p={p_val}",
                    save_path=os.path.join(PLOT_DIR, f"{group_tag}_CO_parity.png")
                )

                make_parity_plot(
                    y_exp=result_co["ln_rate_exp"].values,
                    y_pred=result_co["ln_rate_pred"].values,
                    xlabel="Experimental ln(CO rate)",
                    ylabel="Predicted ln(CO rate)",
                    title=f"Log Parity Plot | CO | GHSV={ghsv_val}, p={p_val}",
                    save_path=os.path.join(PLOT_DIR, f"{group_tag}_CO_log_parity.png")
                )

    # 汇总
    summary_df = pd.DataFrame(all_summary_rows)
    params_df = pd.DataFrame(all_params_rows)

    if len(all_result_rows) > 0:
        results_df = pd.concat(all_result_rows, axis=0, ignore_index=True)
    else:
        results_df = pd.DataFrame()

    skipped_df = pd.DataFrame(all_skipped_rows)

    if not summary_df.empty:
        summary_df = summary_df.sort_values(by=["GHSV", "p", "reaction"]).reset_index(drop=True)

    if not params_df.empty:
        params_df = params_df.sort_values(by=["GHSV", "p", "reaction"]).reset_index(drop=True)

    if not results_df.empty:
        results_df = results_df.sort_values(by=["GHSV_fixed", "p_fixed", "reaction", "T"]).reset_index(drop=True)

    summary_path = os.path.join(OUTPUT_DIR, "summary_loglinear_fixed_GHSV_P.xlsx")
    params_path = os.path.join(OUTPUT_DIR, "parameters_loglinear_fixed_GHSV_P.xlsx")
    results_path = os.path.join(OUTPUT_DIR, "pointwise_results_loglinear_fixed_GHSV_P.xlsx")
    skipped_path = os.path.join(OUTPUT_DIR, "skipped_groups_loglinear_fixed_GHSV_P.xlsx")

    summary_df.to_excel(summary_path, index=False)
    params_df.to_excel(params_path, index=False)
    results_df.to_excel(results_path, index=False)

    if not skipped_df.empty:
        skipped_df.to_excel(skipped_path, index=False)

    print("\n" + "=" * 72)
    print("全部 log linear 拟合完成")
    print("=" * 72)
    print(f"summary 已保存到: {summary_path}")
    print(f"params  已保存到: {params_path}")
    print(f"results 已保存到: {results_path}")
    if not skipped_df.empty:
        print(f"skipped 已保存到: {skipped_path}")
    print(f"图片文件夹: {PLOT_DIR}")

    if not summary_df.empty:
        print("\n结果预览:")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()