import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========= 1. 读入数据 =========
file_path = "arrhenius_summary_ok.csv"
df = pd.read_csv(file_path, sep=None, engine="python", encoding="utf-8-sig")

print("原始列名:", df.columns.tolist())

# ========= 2. 清理列名 =========
df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]

rename_map = {}
for c in df.columns:
    key = str(c).strip().lower().replace(" ", "").replace("_", "").replace("-", "")

    if key in ["h/c", "hc"]:
        rename_map[c] = "HC"
    elif key in ["rate", "ratetype", "rate_type"]:
        rename_map[c] = "rate_type"
    elif key in ["eakjmol", "eakj_mol", "ea(kj/mol)", "ea_kj_mol"]:
        rename_map[c] = "Ea_kJmol"
    elif key in ["r2ln", "r2", "r^2", "r²", "rsquared"]:
        rename_map[c] = "R2"
    elif key in ["npoints", "n_point", "npoints"]:
        rename_map[c] = "n_points"

df = df.rename(columns=rename_map)

print("清理后列名:", df.columns.tolist())

# ========= 3. 检查必要列 =========
required_cols = ["p", "HC", "GHSV", "rate_type", "n_points", "lnA", "Ea_kJmol", "R2"]
missing = [c for c in required_cols if c not in df.columns]

if missing:
    raise ValueError(f"缺少这些必要列: {missing}\n当前列名为: {df.columns.tolist()}")

# ========= 4. 数值化，防止字符串列出问题 =========
for col in ["p", "HC", "GHSV", "n_points", "lnA", "Ea_kJmol", "R2"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)

# ========= 5. 筛选 =========
# 你要求按 R2 > 0.75
df = df[
    (df["R2"] > 0.75) &
    (df["n_points"] >= 4) &
    (df["Ea_kJmol"].notna()) &
    (df["lnA"].notna())
].copy()

print(f"筛选后剩余 {len(df)} 组数据")
print(df[["p", "HC", "GHSV", "rate_type", "Ea_kJmol", "lnA", "R2"]].head())

# ========= 6. 画图函数 =========
def plot_param_vs_condition(
    data: pd.DataFrame,
    rate_type: str,
    y_col: str,
    x_col: str,
    y_label: str,
    x_label: str,
    outname: str = None
):
    sub = data[data["rate_type"] == rate_type].copy()
    sub = sub.dropna(subset=[x_col, y_col])

    if sub.empty:
        print(f"No data for {rate_type}, {y_col} vs {x_col}")
        return

    sub = sub.sort_values(by=x_col)

    x = sub[x_col].astype(float).to_numpy()
    y = sub[y_col].astype(float).to_numpy()

    rng = np.random.default_rng(42)
    jitter = rng.normal(0, 0.03 * (x.max() - x.min() + 1e-9), size=len(x))
    x_jitter = x + jitter

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(x_jitter, y, s=60, alpha=0.75, label="groups")

    stat = sub.groupby(x_col)[y_col].agg(["mean", "std", "count"]).reset_index()
    stat = stat.sort_values(by=x_col)

    ax.plot(stat[x_col], stat["mean"], linewidth=2, label="mean")
    ax.errorbar(
        stat[x_col],
        stat["mean"],
        yerr=stat["std"],
        fmt="none",
        capsize=4
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{rate_type}: {y_label} vs {x_label}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if outname:
        plt.savefig(outname, dpi=300, bbox_inches="tight")
    plt.show()

# ========= 7. 画图 =========
for rate in ["r_CH3OH", "r_CO"]:
    plot_param_vs_condition(df, rate, "Ea_kJmol", "HC",   "Ea (kJ/mol)", "H/C",     f"{rate}_Ea_vs_HC.png")
    plot_param_vs_condition(df, rate, "Ea_kJmol", "p",    "Ea (kJ/mol)", "p (MPa)", f"{rate}_Ea_vs_p.png")
    plot_param_vs_condition(df, rate, "Ea_kJmol", "GHSV", "Ea (kJ/mol)", "GHSV",    f"{rate}_Ea_vs_GHSV.png")

    plot_param_vs_condition(df, rate, "lnA", "HC",   "lnA", "H/C",     f"{rate}_lnA_vs_HC.png")
    plot_param_vs_condition(df, rate, "lnA", "p",    "lnA", "p (MPa)", f"{rate}_lnA_vs_p.png")
    plot_param_vs_condition(df, rate, "lnA", "GHSV", "lnA", "GHSV",    f"{rate}_lnA_vs_GHSV.png")