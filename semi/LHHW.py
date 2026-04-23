import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
from multiprocessing import freeze_support
import matplotlib.pyplot as plt

R = 8.314
eps = 1e-12

def load_data(sheet_name):
    df = pd.read_excel('data.xlsx', sheet_name=sheet_name, header=0, skiprows=[1])
    df.columns = df.columns.str.strip()

    fuga = df[['fCO2', 'fH2', 'fCH3OH', 'fH2O', 'fCO']].to_numpy(dtype=float)
    rMeOH = df['rMeOH'].to_numpy(dtype=float)
    rCO = df['rCO'].to_numpy(dtype=float)
    temperature = df['T'].to_numpy(dtype=float)

    print(f"Sheet = {sheet_name}，总数据点数: {len(df)}")
    return df, fuga, rMeOH, rCO, temperature


def maybe_convert_pa_to_mpa(fuga):
    # 如果 fugacity 数值明显是 Pa 量级，则自动转成 MPa
    if np.nanmax(fuga) > 1e3:
        print("检测到 fugacity 量级较大，按 Pa -> MPa 自动转换")
        return fuga / 1e6
    return fuga


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


def calc_predictions(par, fuga, temperature):
    """
    par 顺序：
    A1_star, E1, A2_star, E2, KCO2
    先固定 KCO = 1.0, KH2O_H2 = 1.0
    """
    A1_star, E1, A2_star, E2, KCO2 = par

    KCO = 1.0
    KH2O_H2 = 1.0

    fuga = maybe_convert_pa_to_mpa(fuga)

    T_av = np.mean(temperature)

    fCO2 = np.maximum(fuga[:, 0], eps)
    fH2 = np.maximum(fuga[:, 1], eps)
    fCH3OH = np.maximum(fuga[:, 2], eps)
    fH2O = np.maximum(fuga[:, 3], eps)
    fCO = np.maximum(fuga[:, 4], eps)

    k1 = A1_star * np.exp(-(E1 / R) * (1 / temperature - 1 / T_av))
    k2 = A2_star * np.exp(-(E2 / R) * (1 / temperature - 1 / T_av))

    K_f1, K_f2 = calculate_equilibrium_constants(temperature)
    K_f1_safe = np.maximum(K_f1, eps)
    K_f2_safe = np.maximum(K_f2, eps)

    den_s1 = 1.0 + KCO * fCO + KCO2 * fCO2
    den_s2 = np.sqrt(fH2) + KH2O_H2 * fH2O
    den = np.maximum(den_s1 * den_s2, eps)

    drive1 = (
        fCO2 * (fH2 ** 1.5)
        - (fCH3OH * fH2O) / np.maximum((fH2 ** 1.5) * K_f1_safe, eps)
    )

    drive2 = (
        fCO2 * fH2
        - (fCO * fH2O) / K_f2_safe
    )

    r1 = k1 * KCO2 * drive1 / den
    r2 = k2 * KCO2 * drive2 / den

    rate_meoh_pred = r1
    rate_co_pred = r2

    return rate_meoh_pred, rate_co_pred, r1, r2, k1, k2, den


def objective(par, fuga, temperature, rMeOH_exp, rCO_exp):
    try:
        rate_meoh_pred, rate_co_pred, _, _, _, _, _ = calc_predictions(par, fuga, temperature)

        denom_meoh = np.maximum(np.abs(rMeOH_exp), 1e-6)
        denom_co = np.maximum(np.abs(rCO_exp), 1e-6)

        sse1 = np.sum(((rate_meoh_pred - rMeOH_exp) / denom_meoh) ** 2)
        sse2 = np.sum(((rate_co_pred - rCO_exp) / denom_co) ** 2)
        total_sse = sse1 + sse2

        if not np.isfinite(total_sse):
            return 1e30

        return float(total_sse)

    except Exception as e:
        print("objective 出错:", e)
        return 1e30


def calc_r2(y_exp, y_pred):
    ss_res = np.sum((y_exp - y_pred) ** 2)
    ss_tot = np.sum((y_exp - np.mean(y_exp)) ** 2)
    if ss_tot < 1e-12:
        return np.nan
    return 1 - ss_res / ss_tot


def calc_mre(y_exp, y_pred):
    denom = np.maximum(np.abs(y_exp), 1e-12)
    return np.mean(np.abs((y_pred - y_exp) / denom)) * 100


def fit_one_sheet(sheet_name):
    df, fuga, rMeOH, rCO, temperature = load_data(sheet_name)

    bounds = [
        (1e-8, 1e3),      # A1_star
        (-15000, 15000),  # E1
        (1e-8, 1e3),      # A2_star
        (-15000, 15000),  # E2
        (1e-8, 1e3)       # KCO2
    ]

    result = differential_evolution(
        objective,
        bounds=bounds,
        args=(fuga, temperature, rMeOH, rCO),
        seed=200,
        popsize=15,
        maxiter=500,
        tol=1e-6,
        mutation=(0.5, 1.0),
        recombination=0.7,
        polish=True,
        workers=1,
        updating='immediate'
    )

    print(f"\nSheet {sheet_name} 优化器返回信息:")
    print("  success =", result.success)
    print("  message =", result.message)
    print("  objective =", result.fun)

    par_opt = result.x

    rate_meoh_pred, rate_co_pred, r1_pred, r2_pred, k1, k2, den = calc_predictions(
        par_opt, fuga, temperature
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
    result_df['k1_T'] = k1
    result_df['k2_T'] = k2
    result_df['den_common'] = den
    result_df['rel_error_rMeOH_%'] = np.abs((rate_meoh_pred - rMeOH) / np.maximum(np.abs(rMeOH), 1e-12)) * 100
    result_df['rel_error_rCO_%'] = np.abs((rate_co_pred - rCO) / np.maximum(np.abs(rCO), 1e-12)) * 100
    result_df['sheet'] = sheet_name
    result_df.to_excel(f'fit_results_{sheet_name}.xlsx', index=False)

    param_names = ['A1_star', 'E1', 'A2_star', 'E2', 'KCO2']
    param_df = pd.DataFrame({
        'param_name': param_names,
        'value': par_opt
    })
    param_df.to_excel(f'fitted_parameters_{sheet_name}.xlsx', index=False)

    print(f"Sheet {sheet_name} 拟合结果:")
    print(f"  r2_meoh   = {r2_meoh:.6f}")
    print(f"  r2_co     = {r2_co:.6f}")
    print(f"  avg_r2    = {avg_r2:.6f}")
    print(f"  rmse_meoh = {rmse_meoh:.6e}")
    print(f"  rmse_co   = {rmse_co:.6e}")
    print(f"  mre_meoh% = {mre_meoh:.4f}")
    print(f"  mre_co%   = {mre_co:.4f}")
    print(f"  fit_success = {fit_success}")

    return {
        'sheet': sheet_name,
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
        'df': result_df
    }


def main():
    sheet_list = ['1']

    all_results = []
    all_params = []
    all_df = []

    param_names = ['A1_star', 'E1', 'A2_star', 'E2', 'KCO2']

    for sheet_name in sheet_list:
        print("\n" + "=" * 60)
        print(f"开始拟合 Sheet: {sheet_name}")
        print("=" * 60)

        res = fit_one_sheet(sheet_name)

        all_results.append({
            'sheet': res['sheet'],
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

        row = {'sheet': sheet_name}
        for name, val in zip(param_names, res['params']):
            row[name] = val
        all_params.append(row)

        all_df.append(res['df'])

    df_all = pd.concat(all_df, ignore_index=True)

    compare_df = pd.DataFrame(all_results)
    compare_df.to_excel('comparison_summary.xlsx', index=False)

    params_compare_df = pd.DataFrame(all_params)
    params_compare_df.to_excel('comparison_parameters.xlsx', index=False)

    print("\n全部 sheet 拟合完成")
    print(compare_df)
    print("\n汇总文件已保存:")
    print("  comparison_summary.xlsx")
    print("  comparison_parameters.xlsx")

    plt.figure(figsize=(6, 6))
    plt.scatter(df_all['rMeOH'], df_all['rMeOH_pred'], alpha=0.7)
    min_val = min(df_all['rMeOH'].min(), df_all['rMeOH_pred'].min())
    max_val = max(df_all['rMeOH'].max(), df_all['rMeOH_pred'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')
    plt.xlabel('Experimental MeOH')
    plt.ylabel('Predicted MeOH')
    plt.title('Parity Plot - MeOH')
    plt.grid()
    plt.savefig('parity_plot_MeOH.png', dpi=300)
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.scatter(df_all['rCO'], df_all['rCO_pred'], alpha=0.7)
    min_val = min(df_all['rCO'].min(), df_all['rCO_pred'].min())
    max_val = max(df_all['rCO'].max(), df_all['rCO_pred'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')
    plt.xlabel('Experimental CO')
    plt.ylabel('Predicted CO')
    plt.title('Parity Plot - CO')
    plt.grid()
    plt.savefig('parity_plot_CO.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    freeze_support()
    main()