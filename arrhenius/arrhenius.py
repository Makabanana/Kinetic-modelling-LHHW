import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# Basic settings
# =========================
R = 8.314462618  # J/mol/K

FILE = "full data.xlsx"
OUT_DIR = Path("arrhenius_output")
OUT_DIR.mkdir(exist_ok=True)

RATE_COLS = ["r_CO", "r_CH3OH"]


# =========================
# 1. Load and clean data
# =========================
def load_data(file_path):
    raw = pd.read_excel(file_path)

    # 第一行是单位，真正数据从第二行开始
    df = raw.iloc[1:].copy()

    # 按实际列重命名
    df.columns = [
        "H/C", "p", "GHSV", "T_C", "T_K",
        "fCO2", "fH2", "fCH3OH", "fH2O", "fCO",
        "r_CH3OH", "r_CO", "r_CO2"
    ]

    # 转成数值
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 去掉关键列缺失
    df = df.dropna(subset=["H/C", "p", "GHSV", "T_K"])

    return df


# =========================
# 2. Apparent Arrhenius fitting for one group
# ln(r) = ln(A) - Ea / (R*T)
# 这里只用于评估线性拟合效果
# =========================
def fit_arrhenius(sub_df, rate_col):
    sub = sub_df.sort_values("T_K").copy()

    # ln(r) 需要 r > 0
    sub = sub[sub[rate_col] > 0].copy()

    if len(sub) < 3:
        raise ValueError("positive points < 3, cannot fit")

    x = 1.0 / sub["T_K"].to_numpy()
    y = np.log(sub[rate_col].to_numpy())

    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))

    Ea_J_mol = -slope * R
    Ea_kJ_mol = Ea_J_mol / 1000.0
    lnA = intercept
    A = np.exp(lnA)

    result = {
        "n_points": len(sub),
        "T_min_C": sub["T_C"].min(),
        "T_max_C": sub["T_C"].max(),
        "lnA": lnA,
        "A": A,
        "Ea_J_mol": Ea_J_mol,
        "Ea_kJ_mol": Ea_kJ_mol,
        "R2_ln": r2,
        "RMSE_ln": rmse
    }

    return result


# =========================
# 3. Batch fitting for all groups
# 按 p, H/C, GHSV 分组
# =========================
def batch_fit_all(df, rate_cols):
    records = []

    grouped = df.groupby(["p", "H/C", "GHSV"])

    for (p_value, hc_value, ghsv_value), sub in grouped:
        for rate_col in rate_cols:
            try:
                result = fit_arrhenius(sub, rate_col)
                row = {
                    "p": p_value,
                    "H/C": hc_value,
                    "GHSV": ghsv_value,
                    "rate": rate_col,
                    **result,
                    "status": "ok"
                }
            except Exception as e:
                row = {
                    "p": p_value,
                    "H/C": hc_value,
                    "GHSV": ghsv_value,
                    "rate": rate_col,
                    "n_points": np.nan,
                    "T_min_C": np.nan,
                    "T_max_C": np.nan,
                    "lnA": np.nan,
                    "A": np.nan,
                    "Ea_J_mol": np.nan,
                    "Ea_kJ_mol": np.nan,
                    "R2_ln": np.nan,
                    "RMSE_ln": np.nan,
                    "status": str(e)
                }

            records.append(row)

    summary = pd.DataFrame(records)
    return summary


# =========================
# 4. Plot factor effect
# 每个 rate 画两张图
# 1. R2_ln 对 H/C, GHSV, p 的影响
# 2. RMSE_ln 对 H/C, GHSV, p 的影响
# =========================
def plot_factor_effect(summary_ok, rate_col, metric, out_dir):
    sub = summary_ok[summary_ok["rate"] == rate_col].copy()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    for ax, factor in zip(axes, ["H/C", "GHSV", "p"]):
        stat = (
            sub.groupby(factor)[metric]
            .agg(["mean", "count"])
            .reset_index()
            .sort_values(factor)
        )

        x = stat[factor].to_numpy()
        y = stat["mean"].to_numpy()

        ax.plot(x, y, marker="o", linewidth=1.8)

        ax.set_xlabel(factor)
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.linspace(0, 1.0, 6))

        if metric == "R2_ln":
            ax.set_ylabel("R²")
            ax.set_title(f"{rate_col}: R² vs {factor}")
        else:
            ax.set_ylabel(metric)
            ax.set_title(f"{rate_col}: {metric} vs {factor}")

    plt.tight_layout()
    save_name = out_dir / f"factor_effect_{metric}_{rate_col}.png"
    plt.savefig(save_name, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_name}")

# =========================
# 5. Save factor summary tables
# 方便你直接看平均值
# =========================
def save_factor_tables(summary_ok, rate_col, out_dir):
    sub = summary_ok[summary_ok["rate"] == rate_col].copy()

    for factor in ["H/C", "GHSV", "p"]:
        stat = (
            sub.groupby(factor)[["R2_ln", "RMSE_ln"]]
            .agg(["mean", "min", "max", "count"])
        )

        safe_factor = str(factor).replace("/", "_")
        safe_rate = str(rate_col).replace("/", "_")

        save_name = out_dir / f"factor_summary_{safe_factor}_{safe_rate}.csv"
        stat.to_csv(save_name)

        print(f"Saved: {save_name}")


# =========================
# 6. Main
# =========================
if __name__ == "__main__":
    df = load_data(FILE)

    combos = (
        df[["p", "H/C", "GHSV"]]
        .drop_duplicates()
        .sort_values(["p", "H/C", "GHSV"])
        .reset_index(drop=True)
    )

    print("Available combinations:")
    print(combos.to_string(index=False))

    summary = batch_fit_all(df, rate_cols=RATE_COLS)

    summary_all_file = OUT_DIR / "arrhenius_summary_all.csv"
    summary.to_csv(summary_all_file, index=False)

    summary_ok = summary[summary["status"] == "ok"].copy()
    summary_ok_file = OUT_DIR / "arrhenius_summary_ok.csv"
    summary_ok.to_csv(summary_ok_file, index=False)

    print("\nSaved summary files:")
    print(summary_all_file)
    print(summary_ok_file)

    for rate_col in RATE_COLS:
        plot_factor_effect(summary_ok, rate_col, metric="R2_ln", out_dir=OUT_DIR)
        save_factor_tables(summary_ok, rate_col, OUT_DIR)

    print("\nAll done.")