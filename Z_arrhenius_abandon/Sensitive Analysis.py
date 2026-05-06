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

# 字体大小设置
ANNOTATION_FONTSIZE = 5   # 图中每个点旁边的工况文字
AXIS_LABEL_FONTSIZE = 11  # 坐标轴标题
TITLE_FONTSIZE = 12       # 图标题
TICK_FONTSIZE = 9         # 坐标刻度字体
LEGEND_FONTSIZE = 8       # 图例字体


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
# 4. 分别画甲醇和CO两张图
# =========================
def plot_separate_ea_lna(summary_ok, out_dir):
    if summary_ok.empty:
        print("No valid data to plot")
        return

    plot_map = {
        "r_CH3OH": {
            "title": "CH3OH: Ea vs lnA",
            "filename": "Ea_lnA_CH3OH.png"
        },
        "r_CO": {
            "title": "CO: Ea vs lnA",
            "filename": "Ea_lnA_CO.png"
        }
    }

    for rate_col, cfg in plot_map.items():
        part = summary_ok[summary_ok["rate"] == rate_col].copy()

        if part.empty:
            print(f"No valid data for {rate_col}")
            continue

        plt.figure(figsize=(8, 6))
        plt.scatter(part["Ea_kJ_mol"], part["lnA"], s=50)

        for _, row in part.iterrows():
            label = f"p={int(row['p'])}, H/C={int(row['H/C'])}, GHSV={int(row['GHSV'])}"
            plt.annotate(
                label,
                (row["Ea_kJ_mol"], row["lnA"]),
                fontsize=ANNOTATION_FONTSIZE,
                xytext=(3, 3),
                textcoords="offset points"
            )

        plt.xlabel("Ea (kJ/mol)", fontsize=AXIS_LABEL_FONTSIZE)
        plt.ylabel("lnA", fontsize=AXIS_LABEL_FONTSIZE)
        plt.title(cfg["title"], fontsize=TITLE_FONTSIZE)
        plt.xticks(fontsize=TICK_FONTSIZE)
        plt.yticks(fontsize=TICK_FONTSIZE)
        plt.tight_layout()

        save_name = out_dir / cfg["filename"]
        plt.savefig(save_name, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved: {save_name}")


# =========================
# 5. Main
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

    # 分别画两张图
    plot_separate_ea_lna(summary_ok, OUT_DIR)

    print("\nAll done.")
    summary = batch_fit_all(df, rate_cols=RATE_COLS)

    summary_ok = summary[summary["status"] == "ok"].copy()

    ea_a_table = summary_ok[
        ["p", "H/C", "GHSV", "rate", "Ea_kJ_mol", "A", "lnA", "R2_ln", "RMSE_ln", "n_points"]
    ].sort_values(["rate", "H/C", "p", "GHSV"])

    ea_a_table.to_csv(OUT_DIR / "Ea_A_table.csv", index=False)

    print(ea_a_table.to_string(index=False))
    print(f"\nSaved: {OUT_DIR / 'Ea_A_table.csv'}")