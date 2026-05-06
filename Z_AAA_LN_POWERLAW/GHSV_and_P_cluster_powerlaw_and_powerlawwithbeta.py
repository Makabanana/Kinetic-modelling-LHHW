# 按 GHSV 和 p 双重分类拟合
# 比较 simple power law 和 power law × (1 - beta)
# 没有引入 Tave 和 lnA

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
def load_data(file_name='12000GHSV.xlsx', sheet_name=0):

    df = pd.read_excel(file_name, sheet_name=sheet_name, header=0)

    df.columns = df.columns.astype(str).str.strip()

    # 尝试去掉第二行单位行
    # 如果第一行大多数内容不是数字，就认为它可能是单位行
    if len(df) > 1:
        second_row = df.iloc[0].astype(str).tolist()
        numeric_like_count = 0

        for x in second_row:
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
        'r CH3OH': 'rMeOH',
        'rCH3OH': 'rMeOH',
        'r MeOH': 'rMeOH',
        'r_CH3OH': 'rMeOH',
        'r CH3OH ': 'rMeOH',
        'rMeOH ': 'rMeOH',

        'r CO': 'rCO',
        'r_CO': 'rCO',
        'rCO ': 'rCO',

        'T ': 'T',
        'p ': 'p',
        'GHSV ': 'GHSV',

        'fCO2 ': 'fCO2',
        'fH2 ': 'fH2',
        'fCH3OH ': 'fCH3OH',
        'fH2O ': 'fH2O',
        'fCO ': 'fCO',
    }

    df = df.rename(columns=rename_map)

    # 必要列
    required_cols = [
        'GHSV',
        'p',
        'fCO2',
        'fH2',
        'fCH3OH',
        'fH2O',
        'fCO',
        'rMeOH',
        'rCO',
        'T'
    ]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(
            f'缺少必要列: {missing}\n当前列名: {list(df.columns)}'
        )

    # 转成数值
    for c in required_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # 删除关键列存在空值的行
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    fuga = df[['fCO2', 'fH2', 'fCH3OH', 'fH2O', 'fCO']].to_numpy(dtype=float)
    rMeOH = df['rMeOH'].to_numpy(dtype=float)
    rCO = df['rCO'].to_numpy(dtype=float)
    temperature = df['T'].to_numpy(dtype=float)

    print(f'总数据点数: {len(df)}')
    print(f'检测到的 GHSV: {sorted(df["GHSV"].dropna().unique())}')
    print(f'检测到的 p: {sorted(df["p"].dropna().unique())}')

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
# 3A. 最原始 simple power law
# =========================
def calc_predictions_simple(par, fuga, temperature):
    """
    par 顺序：
    A1, E1, n1_A, n1_B,
    A2, E2, n2_A, n2_B

    反应 1:
    CO2 + 3H2 -> CH3OH + H2O

    r1 = k1 * fCO2^n1_A * fH2^n1_B

    反应 2:
    CO2 + H2 -> CO + H2O

    r2 = k2 * fCO2^n2_A * fH2^n2_B
    """

    A1, E1, n1_A, n1_B, A2, E2, n2_A, n2_B = par

    fA = fuga[:, 0]   # CO2
    fB = fuga[:, 1]   # H2

    fA_safe = np.maximum(fA, eps)
    fB_safe = np.maximum(fB, eps)

    k1 = A1 * np.exp(-E1 / (R * temperature))
    k2 = A2 * np.exp(-E2 / (R * temperature))

    r1 = k1 * (fA_safe ** n1_A) * (fB_safe ** n1_B)
    r2 = k2 * (fA_safe ** n2_A) * (fB_safe ** n2_B)

    rate_meoh_pred = r1
    rate_co_pred = r2

    return rate_meoh_pred, rate_co_pred, r1, r2, k1, k2


# =========================
# 3B. power law × (1 - beta)
# =========================
def calc_predictions_beta(par, fuga, temperature):
    """
    par 顺序：
    A1, E1, n1_A, n1_B,
    A2, E2, n2_A, n2_B

    反应 1:
    CO2 + 3H2 -> CH3OH + H2O

    r1 = k1 * fCO2^n1_A * fH2^n1_B * (1 - beta1)

    beta1 = (fCH3OH * fH2O) / (K_f1 * fCO2 * fH2^3)

    反应 2:
    CO2 + H2 -> CO + H2O

    r2 = k2 * fCO2^n2_A * fH2^n2_B * (1 - beta2)

    beta2 = (fCO * fH2O) / (K_f2 * fCO2 * fH2)
    """

    A1, E1, n1_A, n1_B, A2, E2, n2_A, n2_B = par

    fA = fuga[:, 0]   # CO2
    fB = fuga[:, 1]   # H2
    fC = fuga[:, 2]   # CH3OH
    fD = fuga[:, 3]   # H2O
    fE = fuga[:, 4]   # CO

    fA_safe = np.maximum(fA, eps)
    fB_safe = np.maximum(fB, eps)
    fC_safe = np.maximum(fC, eps)
    fD_safe = np.maximum(fD, eps)
    fE_safe = np.maximum(fE, eps)

    k1 = A1 * np.exp(-E1 / (R * temperature))
    k2 = A2 * np.exp(-E2 / (R * temperature))

    K_f1, K_f2 = calculate_equilibrium_constants(temperature)

    K_f1_safe = np.maximum(K_f1, eps)
    K_f2_safe = np.maximum(K_f2, eps)

    beta1 = (fC_safe * fD_safe) / np.maximum(
        K_f1_safe * fA_safe * (fB_safe ** 3),
        eps
    )

    beta2 = (fE_safe * fD_safe) / np.maximum(
        K_f2_safe * fA_safe * fB_safe,
        eps
    )

    r1 = (
        k1
        * (fA_safe ** n1_A)
        * (fB_safe ** n1_B)
        * (1 - beta1)
    )

    r2 = (
        k2
        * (fA_safe ** n2_A)
        * (fB_safe ** n2_B)
        * (1 - beta2)
    )

    rate_meoh_pred = r1
    rate_co_pred = r2

    return rate_meoh_pred, rate_co_pred, r1, r2, k1, k2


# =========================
# 4. 目标函数
# =========================
def objective_simple(par, fuga, temperature, rMeOH_exp, rCO_exp):

    try:
        rate_meoh_pred, rate_co_pred, _, _, _, _ = calc_predictions_simple(
            par,
            fuga,
            temperature
        )

        denom_meoh = np.maximum(np.abs(rMeOH_exp), 1e-6)
        denom_co = np.maximum(np.abs(rCO_exp), 1e-6)

        sse1 = np.sum(((rate_meoh_pred - rMeOH_exp) / denom_meoh) ** 2)
        sse2 = np.sum(((rate_co_pred - rCO_exp) / denom_co) ** 2)

        total_sse = sse1 + sse2

        if not np.isfinite(total_sse):
            return 1e30

        return float(total_sse)

    except Exception as e:
        print('objective_simple 出错:', e)
        return 1e30


def objective_beta(par, fuga, temperature, rMeOH_exp, rCO_exp):

    try:
        rate_meoh_pred, rate_co_pred, _, _, _, _ = calc_predictions_beta(
            par,
            fuga,
            temperature
        )

        denom_meoh = np.maximum(np.abs(rMeOH_exp), 1e-6)
        denom_co = np.maximum(np.abs(rCO_exp), 1e-6)

        sse1 = np.sum(((rate_meoh_pred - rMeOH_exp) / denom_meoh) ** 2)
        sse2 = np.sum(((rate_co_pred - rCO_exp) / denom_co) ** 2)

        total_sse = sse1 + sse2

        if not np.isfinite(total_sse):
            return 1e30

        return float(total_sse)

    except Exception as e:
        print('objective_beta 出错:', e)
        return 1e30


# =========================
# 5. 评价指标
# =========================
def calc_r2(y_exp, y_pred):

    ss_res = np.sum((y_exp - y_pred) ** 2)
    ss_tot = np.sum((y_exp - np.mean(y_exp)) ** 2)

    if ss_tot < 1e-12:
        return np.nan

    return 1 - ss_res / ss_tot


def calc_mre(y_exp, y_pred):

    denom = np.maximum(np.abs(y_exp), 1e-12)

    return np.mean(np.abs((y_pred - y_exp) / denom)) * 100


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
        popsize=25,
        maxiter=1000,
        tol=1e-6,
        mutation=(0.5, 1.0),
        recombination=0.7,
        polish=True,
        workers=1,
        updating='immediate'
    )

    print(f'\n模型 {model_name} 优化器返回信息:')
    print('  success =', result.success)
    print('  message =', result.message)
    print('  objective =', result.fun)

    par_opt = result.x

    rate_meoh_pred, rate_co_pred, r1_pred, r2_pred, k1, k2 = prediction_func(
        par_opt,
        fuga,
        temperature
    )

    r2_meoh = calc_r2(rMeOH, rate_meoh_pred)
    r2_co = calc_r2(rCO, rate_co_pred)
    avg_r2 = np.nanmean([r2_meoh, r2_co])

    rmse_meoh = np.sqrt(np.mean((rMeOH - rate_meoh_pred) ** 2))
    rmse_co = np.sqrt(np.mean((rCO - rate_co_pred) ** 2))

    mre_meoh = calc_mre(rMeOH, rate_meoh_pred)
    mre_co = calc_mre(rCO, rate_co_pred)

    fit_success = (
        np.isfinite(result.fun)
        and (mre_meoh < 10)
        and (mre_co < 20)
    )

    result_df = df.copy()

    result_df['rMeOH_pred'] = rate_meoh_pred
    result_df['rCO_pred'] = rate_co_pred
    result_df['r1_pred'] = r1_pred
    result_df['r2_pred'] = r2_pred
    result_df['k1'] = k1
    result_df['k2'] = k2

    result_df['rel_error_rMeOH_%'] = (
        np.abs((rate_meoh_pred - rMeOH) / np.maximum(np.abs(rMeOH), 1e-12))
        * 100
    )

    result_df['rel_error_rCO_%'] = (
        np.abs((rate_co_pred - rCO) / np.maximum(np.abs(rCO), 1e-12))
        * 100
    )

    result_df['model'] = model_name

    param_names = [
        'A1',
        'E1',
        'n1_A',
        'n1_B',
        'A2',
        'E2',
        'n2_A',
        'n2_B'
    ]

    param_df = pd.DataFrame({
        'param_name': param_names,
        'value': par_opt
    })

    print(f'模型 {model_name} 拟合结果:')
    print(f'  r2_meoh   = {r2_meoh:.6f}')
    print(f'  r2_co     = {r2_co:.6f}')
    print(f'  avg_r2    = {avg_r2:.6f}')
    print(f'  rmse_meoh = {rmse_meoh:.6e}')
    print(f'  rmse_co   = {rmse_co:.6e}')
    print(f'  mre_meoh% = {mre_meoh:.4f}')
    print(f'  mre_co%   = {mre_co:.4f}')
    print(f'  fit_success = {fit_success}')

    return {
        'model': model_name,
        'optimizer_success': result.success,
        'optimizer_message': str(result.message),
        'fit_success': fit_success,
        'objective': result.fun,
        'r2_meoh': r2_meoh,
        'r2_co': r2_co,
        'avg_r2': avg_r2,
        'rmse_meoh': rmse_meoh,
        'rmse_co': rmse_co,
        'mre_meoh_%': mre_meoh,
        'mre_co_%': mre_co,
        'params': par_opt,
        'df': result_df,
        'param_df': param_df
    }


# =========================
# 7. 按 GHSV 和 p 分组画 parity plot
# =========================
def plot_parity_for_group(ghsv, p_value, res_simple, res_beta):

    # =========================
    # 7.1 MeOH parity plot
    # =========================
    plt.figure(figsize=(6, 6))

    plt.scatter(
        res_simple['df']['rMeOH'],
        res_simple['df']['rMeOH_pred'],
        label='simple_powerlaw',
        alpha=0.7
    )

    plt.scatter(
        res_beta['df']['rMeOH'],
        res_beta['df']['rMeOH_pred'],
        label='powerlaw_with_1_minus_beta',
        alpha=0.7
    )

    min_val = min(
        res_simple['df']['rMeOH'].min(),
        res_simple['df']['rMeOH_pred'].min(),
        res_beta['df']['rMeOH'].min(),
        res_beta['df']['rMeOH_pred'].min()
    )

    max_val = max(
        res_simple['df']['rMeOH'].max(),
        res_simple['df']['rMeOH_pred'].max(),
        res_beta['df']['rMeOH'].max(),
        res_beta['df']['rMeOH_pred'].max()
    )

    plt.plot([min_val, max_val], [min_val, max_val], 'k--')
    plt.xlabel('Experimental MeOH')
    plt.ylabel('Predicted MeOH')
    plt.title(f'MeOH Parity, GHSV = {int(ghsv)}, p = {p_value}')
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        f'parity_plot_meoh_GHSV_{int(ghsv)}_p_{p_value}.png',
        dpi=300
    )

    plt.show()

    # =========================
    # 7.2 CO parity plot
    # =========================
    plt.figure(figsize=(6, 6))

    plt.scatter(
        res_simple['df']['rCO'],
        res_simple['df']['rCO_pred'],
        label='simple_powerlaw',
        alpha=0.7
    )

    plt.scatter(
        res_beta['df']['rCO'],
        res_beta['df']['rCO_pred'],
        label='powerlaw_with_1_minus_beta',
        alpha=0.7
    )

    min_val = min(
        res_simple['df']['rCO'].min(),
        res_simple['df']['rCO_pred'].min(),
        res_beta['df']['rCO'].min(),
        res_beta['df']['rCO_pred'].min()
    )

    max_val = max(
        res_simple['df']['rCO'].max(),
        res_simple['df']['rCO_pred'].max(),
        res_beta['df']['rCO'].max(),
        res_beta['df']['rCO_pred'].max()
    )

    plt.plot([min_val, max_val], [min_val, max_val], 'k--')
    plt.xlabel('Experimental CO')
    plt.ylabel('Predicted CO')
    plt.title(f'CO Parity, GHSV = {int(ghsv)}, p = {p_value}')
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        f'parity_plot_co_GHSV_{int(ghsv)}_p_{p_value}.png',
        dpi=300
    )

    plt.show()


# =========================
# 8. 主程序：按 GHSV 和 p 双重分组拟合
# =========================
def main():

    df, _, _, _, _ = load_data(file_name='full data.xlsx', sheet_name=0)

    # 参数边界
    # par 顺序：
    # A1, E1, n1_A, n1_B, A2, E2, n2_A, n2_B
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

    ghsv_list = sorted(df['GHSV'].dropna().unique())

    print('\n检测到的 GHSV 分组:')
    print(ghsv_list)

    # =========================
    # 8.1 第一层循环：GHSV
    # =========================
    for ghsv in ghsv_list:

        df_g = df[df['GHSV'] == ghsv].copy().reset_index(drop=True)

        p_list = sorted(df_g['p'].dropna().unique())

        print('\n' + '=' * 80)
        print(f'当前 GHSV = {ghsv}')
        print(f'该 GHSV 下检测到的 p 分组: {p_list}')
        print('=' * 80)

        # =========================
        # 8.2 第二层循环：p
        # =========================
        for p_value in p_list:

            print('\n' + '-' * 80)
            print(f'开始拟合 GHSV = {ghsv}, p = {p_value}')
            print('-' * 80)

            df_gp = df_g[df_g['p'] == p_value].copy().reset_index(drop=True)

            print(f'当前分组数据点数 = {len(df_gp)}')

            fuga_gp = df_gp[
                ['fCO2', 'fH2', 'fCH3OH', 'fH2O', 'fCO']
            ].to_numpy(dtype=float)

            rMeOH_gp = df_gp['rMeOH'].to_numpy(dtype=float)
            rCO_gp = df_gp['rCO'].to_numpy(dtype=float)
            temperature_gp = df_gp['T'].to_numpy(dtype=float)

            # =========================
            # 8.3 simple power law
            # =========================
            res_simple = fit_model(
                model_name='simple_powerlaw',
                objective_func=objective_simple,
                prediction_func=calc_predictions_simple,
                bounds=bounds,
                df=df_gp,
                fuga=fuga_gp,
                rMeOH=rMeOH_gp,
                rCO=rCO_gp,
                temperature=temperature_gp
            )

            # =========================
            # 8.4 power law with beta
            # =========================
            res_beta = fit_model(
                model_name='powerlaw_with_1_minus_beta',
                objective_func=objective_beta,
                prediction_func=calc_predictions_beta,
                bounds=bounds,
                df=df_gp,
                fuga=fuga_gp,
                rMeOH=rMeOH_gp,
                rCO=rCO_gp,
                temperature=temperature_gp
            )

            # =========================
            # 8.5 保存当前 GHSV 和 p 的 summary
            # =========================
            for res in [res_simple, res_beta]:

                all_summary.append({
                    'GHSV': ghsv,
                    'p': p_value,
                    'model': res['model'],
                    'n_points': len(df_gp),
                    'optimizer_success': res['optimizer_success'],
                    'optimizer_message': res['optimizer_message'],
                    'fit_success': res['fit_success'],
                    'objective': res['objective'],
                    'r2_meoh': res['r2_meoh'],
                    'r2_co': res['r2_co'],
                    'avg_r2': res['avg_r2'],
                    'rmse_meoh': res['rmse_meoh'],
                    'rmse_co': res['rmse_co'],
                    'mre_meoh_%': res['mre_meoh_%'],
                    'mre_co_%': res['mre_co_%']
                })

                param_dict = dict(
                    zip(
                        res['param_df']['param_name'],
                        res['param_df']['value']
                    )
                )

                all_params.append({
                    'GHSV': ghsv,
                    'p': p_value,
                    'model': res['model'],
                    'n_points': len(df_gp),
                    **param_dict
                })

                result_df = res['df'].copy()
                result_df['GHSV_fit_group'] = ghsv
                result_df['p_fit_group'] = p_value

                all_results.append(result_df)

            # =========================
            # 8.6 每个 GHSV 和 p 分别保存详细结果
            # =========================
            res_simple['df'].to_excel(
                f'fit_results_simple_powerlaw_GHSV_{int(ghsv)}_p_{p_value}.xlsx',
                index=False
            )

            res_beta['df'].to_excel(
                f'fit_results_powerlaw_with_1_minus_beta_GHSV_{int(ghsv)}_p_{p_value}.xlsx',
                index=False
            )

            # =========================
            # 8.7 每个 GHSV 和 p 分别画 parity plot
            # =========================
            plot_parity_for_group(
                ghsv=ghsv,
                p_value=p_value,
                res_simple=res_simple,
                res_beta=res_beta
            )

    # =========================
    # 8.8 保存所有 GHSV 和 p 的汇总结果
    # =========================
    summary_df = pd.DataFrame(all_summary)
    params_df = pd.DataFrame(all_params)
    results_df = pd.concat(all_results, ignore_index=True)

    summary_df.to_excel('GHSV_p_comparison_summary.xlsx', index=False)
    params_df.to_excel('GHSV_p_comparison_parameters.xlsx', index=False)
    results_df.to_excel('GHSV_p_fit_results_all.xlsx', index=False)

    print('\n全部 GHSV 和 p 双重分组拟合完成')

    print('\n汇总结果:')
    print(summary_df)

    print('\n参数结果:')
    print(params_df)


if __name__ == '__main__':
    main()