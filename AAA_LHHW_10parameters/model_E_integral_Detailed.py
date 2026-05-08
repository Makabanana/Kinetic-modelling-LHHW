from pathlib import Path
from datetime import datetime
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution


# ============================================================
# 0. Global settings
# ============================================================

FAST_TEST_MODE = False

EPS = 1e-30

MODEL_NAME = "Model_E_integral_two_reaction_LHHW_10param_optimizer_comparison"

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "output" / "optimizer_comparison_10param_detailed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT_LOG_PATH = BASE_DIR / "output" / "experiment_log.csv"


# ============================================================
# 1. Data file discovery
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
# 2. Fixed bed reactor settings
# ============================================================

W_CAT_KG = 0.0002

STANDARD_FLOW_L_PER_MIN = 42e-3
STANDARD_MOLAR_VOLUME_L_PER_MOL = 22.414
F_TOTAL_IN = STANDARD_FLOW_L_PER_MIN / STANDARD_MOLAR_VOLUME_L_PER_MOL / 60.0

RK4_STEPS = 40


# ============================================================
# 3. Optimizer settings
# ============================================================

FULL_SEEDS = [42, 123, 2026]
FAST_SEEDS = [42]
SEEDS = FAST_SEEDS if FAST_TEST_MODE else FULL_SEEDS

DE_POPSIZE = 12
DE_MAXITER = 50 if FAST_TEST_MODE else 200
DE_TOL = 1e-6
DE_MUTATION = (0.5, 1.0)
DE_RECOMBINATION = 0.7
DE_POLISH = True
DE_WORKERS = 1
DE_UPDATING = "immediate"

GA_POPULATION_SIZE = 120
GA_GENERATIONS = 50 if FAST_TEST_MODE else 200
GA_ELITE_FRACTION = 0.10
GA_CROSSOVER_RATE = 0.80
GA_MUTATION_RATE = 0.20
GA_MUTATION_SCALE = 0.10

PSO_SWARM_SIZE = 120
PSO_ITERATIONS = 50 if FAST_TEST_MODE else 200
PSO_INERTIA = 0.70
PSO_COGNITIVE = 1.50
PSO_SOCIAL = 1.50


# ============================================================
# 4. Parameter settings
# ============================================================

PARAMETER_NAMES = [
    "ln_k1_eff_ref",
    "E1_over_R",
    "ln_k2_eff_ref",
    "E2_over_R",
    "ln_KCO2_ref",
    "DeltaHad_CO2_over_R",
    "ln_KCO_ref",
    "DeltaHad_CO_over_R",
    "ln_KH2O_H2_ref",
    "DeltaHad_H2O_H2_over_R",
]

# Parameter order:
# ln_k1_eff_ref, E1_over_R,
# ln_k2_eff_ref, E2_over_R,
# ln_KCO2_ref, DeltaHad_CO2_over_R,
# ln_KCO_ref, DeltaHad_CO_over_R,
# ln_KH2O_H2_ref, DeltaHad_H2O_H2_over_R
#
# KCO2, KCO, and KH2O_H2 are recovered through exp(ln_K), so they are positive.
# DeltaHad_*_over_R bounds are negative, so adsorption enthalpies stay negative.
BOUNDS = [
    (-30.0, 10.0),
    (0.0, 30000.0),
    (-30.0, 10.0),
    (0.0, 30000.0),
    (-20.0, 10.0),
    (-30000.0, -1e-9),
    (-20.0, 10.0),
    (-30000.0, -1e-9),
    (-20.0, 10.0),
    (-30000.0, -1e-9),
]

LOWER_BOUNDS = np.array([item[0] for item in BOUNDS], dtype=float)
UPPER_BOUNDS = np.array([item[1] for item in BOUNDS], dtype=float)
BOUND_WIDTHS = UPPER_BOUNDS - LOWER_BOUNDS

BOUND_WARNING_FRACTION = 0.02


# ============================================================
# 5. Data loading
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
        raise KeyError(f"Missing columns: {missing_cols}")

    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required_cols).copy()

    # Do not filter GHSV or temperature. Fit all valid rows.
    if df.empty:
        raise ValueError("No valid rows were found.")

    df["Kf1"], df["Kf2"] = calculate_equilibrium_constants(df["T"].values)

    print("Actual columns:")
    print(df.columns.tolist())
    print(f"Number of valid data points: {len(df)}")
    print(f"GHSV values: {sorted(df['GHSV'].unique().tolist())}")
    print(f"T values: {sorted(df['T'].unique().tolist())}")

    prepare_reactor_cache(df)

    return df


def prepare_reactor_cache(df_group):
    """
    Cache numeric arrays used by the objective. This keeps the data loading logic
    the same, but avoids rebuilding Series objects during every optimizer call.
    """

    df_group.attrs["reactor_cache"] = {
        "HC": df_group["HC"].to_numpy(dtype=float),
        "T": df_group["T"].to_numpy(dtype=float),
        "p_MPa": df_group["p_MPa"].to_numpy(dtype=float),
    }

    return df_group.attrs["reactor_cache"]


# ============================================================
# 6. Equilibrium constants
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
# 7. Parameter unpacking
# ============================================================

def safe_exp(x):
    return np.exp(np.clip(x, -700.0, 700.0))


def unpack_params(par):
    """
    Convert optimizer parameters to actual parameters at reference temperature Tave.
    """

    (
        ln_k1_eff_ref,
        E1_over_R,
        ln_k2_eff_ref,
        E2_over_R,
        ln_KCO2_ref,
        DeltaHad_CO2_over_R,
        ln_KCO_ref,
        DeltaHad_CO_over_R,
        ln_KH2O_H2_ref,
        DeltaHad_H2O_H2_over_R,
    ) = par

    params = {
        "k1_eff_ref": safe_exp(ln_k1_eff_ref),
        "E1_over_R": E1_over_R,
        "k2_eff_ref": safe_exp(ln_k2_eff_ref),
        "E2_over_R": E2_over_R,
        "KCO2_ref": safe_exp(ln_KCO2_ref),
        "DeltaHad_CO2_over_R": DeltaHad_CO2_over_R,
        "KCO_ref": safe_exp(ln_KCO_ref),
        "DeltaHad_CO_over_R": DeltaHad_CO_over_R,
        "KH2O_H2_ref": safe_exp(ln_KH2O_H2_ref),
        "DeltaHad_H2O_H2_over_R": DeltaHad_H2O_H2_over_R,
    }

    return params


# ============================================================
# 8. Temperature dependent parameters
# ============================================================

def calculate_temperature_dependent_params(par, T, Tave):
    """
    Rate constants:
        k = k_ref * exp[-E/R * (1/T - 1/Tave)]

    Adsorption constants:
        K = K_ref * exp[-DeltaHad/R * (1/T - 1/Tave)]
    """

    params = unpack_params(par)

    T = float(T)
    Tave = float(Tave)

    k1_eff = params["k1_eff_ref"] * safe_exp(
        -params["E1_over_R"] * (1.0 / T - 1.0 / Tave)
    )

    k2_eff = params["k2_eff_ref"] * safe_exp(
        -params["E2_over_R"] * (1.0 / T - 1.0 / Tave)
    )

    KCO2 = params["KCO2_ref"] * safe_exp(
        -params["DeltaHad_CO2_over_R"] * (1.0 / T - 1.0 / Tave)
    )

    KCO = params["KCO_ref"] * safe_exp(
        -params["DeltaHad_CO_over_R"] * (1.0 / T - 1.0 / Tave)
    )

    KH2O_H2 = params["KH2O_H2_ref"] * safe_exp(
        -params["DeltaHad_H2O_H2_over_R"] * (1.0 / T - 1.0 / Tave)
    )

    params_T = {
        "k1_eff": max(float(k1_eff), EPS),
        "k2_eff": max(float(k2_eff), EPS),
        "KCO2": max(float(KCO2), EPS),
        "KCO": max(float(KCO), EPS),
        "KH2O_H2": max(float(KH2O_H2), EPS),
    }

    return params_T


def add_energy_units(params):
    report = params.copy()

    report["E1_kJ_per_mol"] = params["E1_over_R"] * 8.314 / 1000.0
    report["E2_kJ_per_mol"] = params["E2_over_R"] * 8.314 / 1000.0
    report["DeltaHad_CO2_kJ_per_mol"] = (
        params["DeltaHad_CO2_over_R"] * 8.314 / 1000.0
    )
    report["DeltaHad_CO_kJ_per_mol"] = (
        params["DeltaHad_CO_over_R"] * 8.314 / 1000.0
    )
    report["DeltaHad_H2O_H2_kJ_per_mol"] = (
        params["DeltaHad_H2O_H2_over_R"] * 8.314 / 1000.0
    )

    return report


# ============================================================
# 9. LHHW rates
# ============================================================

def calculate_local_lhhw_rates(par, T, Tave, fCO2, fH2, fCH3OH, fH2O, fCO):
    """
    Local LHHW rates at one bed position from local fugacities.
    """

    params_T = calculate_temperature_dependent_params(
        par=par,
        T=T,
        Tave=Tave,
    )

    K_f1, K_f2 = calculate_equilibrium_constants(T)

    K_f1 = max(float(np.asarray(K_f1)), EPS)
    K_f2 = max(float(np.asarray(K_f2)), EPS)

    fCO2 = max(float(fCO2), EPS)
    fH2 = max(float(fH2), EPS)
    fCH3OH = max(float(fCH3OH), 0.0)
    fH2O = max(float(fH2O), 0.0)
    fCO = max(float(fCO), 0.0)

    k1_eff = params_T["k1_eff"]
    k2_eff = params_T["k2_eff"]

    KCO2 = params_T["KCO2"]
    KCO = params_T["KCO"]
    KH2O_H2 = params_T["KH2O_H2"]

    ads_carbon = 1.0 + KCO2 * fCO2 + KCO * fCO
    ads_hydrogen_water = np.sqrt(fH2) + KH2O_H2 * fH2O

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
# 10. Inlet flows
# ============================================================

def calculate_inlet_flows_from_hc(HC):
    """
    Construct inlet molar flows from H2/CO2 ratio.
    flow order: [CO2, H2, CH3OH, H2O, CO]
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
# 11. RK4 reactor integration
# ============================================================

def calculate_pfr_derivatives(W, flows, par, T, p_MPa, Tave):
    """
    Calculate dF/dW.
    flows order: [CO2, H2, CH3OH, H2O, CO]
    """

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


def integrate_one_experiment(row, par, Tave):
    """
    Integrate one Excel row from W = 0 to W = W_CAT_KG.
    """

    inlet = calculate_inlet_flows_from_hc(row["HC"])

    T = float(row["T"])
    p_MPa = float(row["p_MPa"])

    flows = inlet.copy()

    h = W_CAT_KG / RK4_STEPS
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
            raise FloatingPointError("Non finite molar flow during PFR integration.")

        W += h

    outlet = flows

    rCH3OH_pred = (outlet[2] - inlet[2]) / W_CAT_KG
    rCO_pred = (outlet[4] - inlet[4]) / W_CAT_KG

    return rCH3OH_pred, rCO_pred, outlet


def integrate_one_experiment_values(HC, T, p_MPa, par, Tave):
    """
    Same RK4 integration as integrate_one_experiment, using cached numeric values.
    """

    inlet = calculate_inlet_flows_from_hc(HC)

    T = float(T)
    p_MPa = float(p_MPa)

    flows = inlet.copy()

    h = W_CAT_KG / RK4_STEPS
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
            raise FloatingPointError("Non finite molar flow during PFR integration.")

        W += h

    outlet = flows

    rCH3OH_pred = (outlet[2] - inlet[2]) / W_CAT_KG
    rCO_pred = (outlet[4] - inlet[4]) / W_CAT_KG

    return rCH3OH_pred, rCO_pred, outlet


def calculate_integral_predictions(par, df_group, Tave):
    """
    Integral reactor predictions for every experimental point.
    """

    pred_rates = []
    outlet_flows = []

    cache = df_group.attrs.get("reactor_cache")
    if cache is None:
        cache = prepare_reactor_cache(df_group)

    for HC, T, p_MPa in zip(cache["HC"], cache["T"], cache["p_MPa"]):

        rCH3OH_pred, rCO_pred, outlet = integrate_one_experiment_values(
            HC=HC,
            T=T,
            p_MPa=p_MPa,
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


def calculate_parameter_profile(par, df_group, Tave):
    rows = []

    cache = df_group.attrs.get("reactor_cache")
    if cache is None:
        cache = prepare_reactor_cache(df_group)

    for T in cache["T"]:
        params_T = calculate_temperature_dependent_params(
            par=par,
            T=T,
            Tave=Tave,
        )

        rows.append(params_T)

    return pd.DataFrame(rows)


# ============================================================
# 12. Objective function
# ============================================================

def objective_integral(par, df_group, Tave, rCH3OH_exp, rCO_exp):
    """
    Same objective for every optimizer: relative-error SSE for rCH3OH and rCO.
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
        print("objective_integral error:", error)
        return 1e30


# ============================================================
# 13. Optimizers
# ============================================================

def run_de(seed, objective_args):

    print(f"Running DE seed {seed}")
    start = time.perf_counter()

    result = differential_evolution(
        objective_integral,
        bounds=BOUNDS,
        args=objective_args,
        seed=seed,
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

    runtime_seconds = time.perf_counter() - start

    return {
        "optimizer_name": "DE",
        "seed": seed,
        "status": str(result.message),
        "success": bool(result.success),
        "objective": float(result.fun),
        "runtime_seconds": runtime_seconds,
        "parameters": np.asarray(result.x, dtype=float),
    }


def run_ga(seed, objective_args):

    print(f"Running GA seed {seed}")
    start = time.perf_counter()
    rng = np.random.default_rng(seed)

    n_params = len(BOUNDS)
    elite_count = max(1, int(round(GA_POPULATION_SIZE * GA_ELITE_FRACTION)))
    tournament_size = 3

    population = rng.uniform(
        LOWER_BOUNDS,
        UPPER_BOUNDS,
        size=(GA_POPULATION_SIZE, n_params),
    )
    scores = np.array([
        objective_integral(individual, *objective_args)
        for individual in population
    ], dtype=float)

    best_idx = int(np.argmin(scores))
    best_x = population[best_idx].copy()
    best_score = float(scores[best_idx])

    for generation in range(GA_GENERATIONS):

        elite_indices = np.argsort(scores)[:elite_count]
        new_population = [population[index].copy() for index in elite_indices]

        while len(new_population) < GA_POPULATION_SIZE:

            parent_1 = tournament_select(population, scores, rng, tournament_size)
            parent_2 = tournament_select(population, scores, rng, tournament_size)

            child_1 = parent_1.copy()
            child_2 = parent_2.copy()

            if rng.random() < GA_CROSSOVER_RATE:
                blend = rng.random(n_params)
                child_1 = blend * parent_1 + (1.0 - blend) * parent_2
                child_2 = blend * parent_2 + (1.0 - blend) * parent_1

            child_1 = mutate_ga_child(child_1, rng)
            child_2 = mutate_ga_child(child_2, rng)

            new_population.append(child_1)
            if len(new_population) < GA_POPULATION_SIZE:
                new_population.append(child_2)

        population = np.asarray(new_population, dtype=float)
        scores = np.array([
            objective_integral(individual, *objective_args)
            for individual in population
        ], dtype=float)

        generation_best_idx = int(np.argmin(scores))
        generation_best_score = float(scores[generation_best_idx])

        if generation_best_score < best_score:
            best_score = generation_best_score
            best_x = population[generation_best_idx].copy()

        if (generation + 1) % 25 == 0 or generation == 0:
            print(
                f"  GA seed {seed} generation {generation + 1}/{GA_GENERATIONS}, "
                f"best objective = {best_score:.6g}"
            )

    runtime_seconds = time.perf_counter() - start

    return {
        "optimizer_name": "GA",
        "seed": seed,
        "status": f"completed_{GA_GENERATIONS}_generations",
        "success": True,
        "objective": float(best_score),
        "runtime_seconds": runtime_seconds,
        "parameters": np.asarray(best_x, dtype=float),
    }


def tournament_select(population, scores, rng, tournament_size):

    indices = rng.integers(0, len(population), size=tournament_size)
    best_local_idx = indices[int(np.argmin(scores[indices]))]
    return population[best_local_idx].copy()


def mutate_ga_child(child, rng):

    mutation_mask = rng.random(child.shape[0]) < GA_MUTATION_RATE
    mutation_sigma = GA_MUTATION_SCALE * BOUND_WIDTHS
    child = child.copy()
    child[mutation_mask] += rng.normal(
        loc=0.0,
        scale=mutation_sigma[mutation_mask],
        size=int(np.sum(mutation_mask)),
    )
    return np.clip(child, LOWER_BOUNDS, UPPER_BOUNDS)


def run_pso(seed, objective_args):

    print(f"Running PSO seed {seed}")
    start = time.perf_counter()
    rng = np.random.default_rng(seed)

    n_params = len(BOUNDS)

    positions = rng.uniform(
        LOWER_BOUNDS,
        UPPER_BOUNDS,
        size=(PSO_SWARM_SIZE, n_params),
    )
    velocities = rng.uniform(
        -0.10 * BOUND_WIDTHS,
        0.10 * BOUND_WIDTHS,
        size=(PSO_SWARM_SIZE, n_params),
    )

    scores = np.array([
        objective_integral(position, *objective_args)
        for position in positions
    ], dtype=float)

    personal_best_positions = positions.copy()
    personal_best_scores = scores.copy()

    global_best_idx = int(np.argmin(scores))
    global_best_position = positions[global_best_idx].copy()
    global_best_score = float(scores[global_best_idx])

    vmax = 0.20 * BOUND_WIDTHS

    for iteration in range(PSO_ITERATIONS):

        r1 = rng.random(size=(PSO_SWARM_SIZE, n_params))
        r2 = rng.random(size=(PSO_SWARM_SIZE, n_params))

        velocities = (
            PSO_INERTIA * velocities
            + PSO_COGNITIVE * r1 * (personal_best_positions - positions)
            + PSO_SOCIAL * r2 * (global_best_position - positions)
        )
        velocities = np.clip(velocities, -vmax, vmax)

        positions = positions + velocities
        positions = np.clip(positions, LOWER_BOUNDS, UPPER_BOUNDS)

        scores = np.array([
            objective_integral(position, *objective_args)
            for position in positions
        ], dtype=float)

        improved = scores < personal_best_scores
        personal_best_positions[improved] = positions[improved]
        personal_best_scores[improved] = scores[improved]

        iteration_best_idx = int(np.argmin(personal_best_scores))
        iteration_best_score = float(personal_best_scores[iteration_best_idx])

        if iteration_best_score < global_best_score:
            global_best_score = iteration_best_score
            global_best_position = personal_best_positions[iteration_best_idx].copy()

        if (iteration + 1) % 25 == 0 or iteration == 0:
            print(
                f"  PSO seed {seed} iteration {iteration + 1}/{PSO_ITERATIONS}, "
                f"best objective = {global_best_score:.6g}"
            )

    runtime_seconds = time.perf_counter() - start

    return {
        "optimizer_name": "PSO",
        "seed": seed,
        "status": f"completed_{PSO_ITERATIONS}_iterations",
        "success": True,
        "objective": float(global_best_score),
        "runtime_seconds": runtime_seconds,
        "parameters": np.asarray(global_best_position, dtype=float),
    }


# ============================================================
# 14. Metrics
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


def diagnose_boundary_status(par):
    status = {}
    warnings = []

    for name, value, bound in zip(PARAMETER_NAMES, par, BOUNDS):

        lower, upper = bound
        width = upper - lower

        lower_threshold = lower + BOUND_WARNING_FRACTION * width
        upper_threshold = upper - BOUND_WARNING_FRACTION * width

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


def check_physical_constraints(par, df_group, Tave):
    params = unpack_params(par)

    param_profile = calculate_parameter_profile(
        par=par,
        df_group=df_group,
        Tave=Tave,
    )

    constraints = {
        "KCO2_ref_positive": bool(params["KCO2_ref"] > 0.0),
        "KCO_ref_positive": bool(params["KCO_ref"] > 0.0),
        "KH2O_H2_ref_positive": bool(params["KH2O_H2_ref"] > 0.0),
        "KCO2_T_min": float(param_profile["KCO2"].min()),
        "KCO_T_min": float(param_profile["KCO"].min()),
        "KH2O_H2_T_min": float(param_profile["KH2O_H2"].min()),
        "KCO2_T_all_positive": bool((param_profile["KCO2"] > 0.0).all()),
        "KCO_T_all_positive": bool((param_profile["KCO"] > 0.0).all()),
        "KH2O_H2_T_all_positive": bool((param_profile["KH2O_H2"] > 0.0).all()),
        "DeltaHad_CO2_over_R_negative": bool(params["DeltaHad_CO2_over_R"] < 0.0),
        "DeltaHad_CO_over_R_negative": bool(params["DeltaHad_CO_over_R"] < 0.0),
        "DeltaHad_H2O_H2_over_R_negative": bool(params["DeltaHad_H2O_H2_over_R"] < 0.0),
    }

    all_ok = all(
        bool(constraints[key])
        for key in [
            "KCO2_ref_positive",
            "KCO_ref_positive",
            "KH2O_H2_ref_positive",
            "KCO2_T_all_positive",
            "KCO_T_all_positive",
            "KH2O_H2_T_all_positive",
            "DeltaHad_CO2_over_R_negative",
            "DeltaHad_CO_over_R_negative",
            "DeltaHad_H2O_H2_over_R_negative",
        ]
    )
    constraints["physical_constraint_check"] = "pass" if all_ok else "fail"

    return constraints


def build_prediction_table(df_group, par, Tave):

    rCH3OH_pred, rCO_pred, outlet_flows = calculate_integral_predictions(
        par=par,
        df_group=df_group,
        Tave=Tave,
    )

    rCH3OH_exp = df_group["rCH3OH"].to_numpy(dtype=float)
    rCO_exp = df_group["rCO"].to_numpy(dtype=float)

    param_profile = calculate_parameter_profile(
        par=par,
        df_group=df_group,
        Tave=Tave,
    )

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

    result_df["k1_eff_T"] = param_profile["k1_eff"].values
    result_df["k2_eff_T"] = param_profile["k2_eff"].values
    result_df["KCO2_T"] = param_profile["KCO2"].values
    result_df["KCO_T"] = param_profile["KCO"].values
    result_df["KH2O_H2_T"] = param_profile["KH2O_H2"].values

    return result_df


def evaluate_optimizer_result(raw_result, df_group, Tave, rCH3OH_exp, rCO_exp):

    par_opt = raw_result["parameters"]

    result_df = build_prediction_table(df_group, par_opt, Tave)

    rCH3OH_pred = result_df["rCH3OH_pred"].to_numpy(dtype=float)
    rCO_pred = result_df["rCO_pred"].to_numpy(dtype=float)

    r2_meoh = calc_r2(rCH3OH_exp, rCH3OH_pred)
    r2_co = calc_r2(rCO_exp, rCO_pred)
    avg_r2 = np.nanmean([r2_meoh, r2_co])

    rmse_meoh = calc_rmse(rCH3OH_exp, rCH3OH_pred)
    rmse_co = calc_rmse(rCO_exp, rCO_pred)

    mre_meoh = calc_mre(rCH3OH_exp, rCH3OH_pred)
    mre_co = calc_mre(rCO_exp, rCO_pred)

    boundary_status, boundary_warning = diagnose_boundary_status(par_opt)

    physical_constraints = check_physical_constraints(
        par=par_opt,
        df_group=df_group,
        Tave=Tave,
    )

    summary = {
        "model": MODEL_NAME,
        "optimizer_name": raw_result["optimizer_name"],
        "seed": raw_result["seed"],
        "success": raw_result["success"],
        "status": raw_result["status"],
        "objective": raw_result["objective"],
        "runtime_seconds": raw_result["runtime_seconds"],
        "r2_ch3oh": r2_meoh,
        "r2_co": r2_co,
        "avg_r2": avg_r2,
        "rmse_ch3oh": rmse_meoh,
        "rmse_co": rmse_co,
        "mre_ch3oh_percent": mre_meoh,
        "mre_co_percent": mre_co,
        "boundary_warning": boundary_warning,
        "Tave": Tave,
    }

    for name, value in zip(PARAMETER_NAMES, par_opt):
        summary[name] = float(value)

    summary.update(physical_constraints)
    summary.update(boundary_status)

    return summary, result_df


def make_parameter_output(summary):

    par = np.array([summary[name] for name in PARAMETER_NAMES], dtype=float)
    params = unpack_params(par)
    params_report = add_energy_units(params)

    params_output = {
        "model": MODEL_NAME,
        "optimizer_name": summary["optimizer_name"],
        "seed": summary["seed"],
        "objective": summary["objective"],
        "avg_r2": summary["avg_r2"],
        "Tave": summary["Tave"],
        "ln_k1_eff_ref": par[0],
        "k1_eff_ref": params_report["k1_eff_ref"],
        "E1_over_R": params_report["E1_over_R"],
        "E1_kJ_per_mol": params_report["E1_kJ_per_mol"],
        "ln_k2_eff_ref": par[2],
        "k2_eff_ref": params_report["k2_eff_ref"],
        "E2_over_R": params_report["E2_over_R"],
        "E2_kJ_per_mol": params_report["E2_kJ_per_mol"],
        "ln_KCO2_ref": par[4],
        "KCO2_ref": params_report["KCO2_ref"],
        "DeltaHad_CO2_over_R": params_report["DeltaHad_CO2_over_R"],
        "DeltaHad_CO2_kJ_per_mol": params_report["DeltaHad_CO2_kJ_per_mol"],
        "ln_KCO_ref": par[6],
        "KCO_ref": params_report["KCO_ref"],
        "DeltaHad_CO_over_R": params_report["DeltaHad_CO_over_R"],
        "DeltaHad_CO_kJ_per_mol": params_report["DeltaHad_CO_kJ_per_mol"],
        "ln_KH2O_H2_ref": par[8],
        "KH2O_H2_ref": params_report["KH2O_H2_ref"],
        "DeltaHad_H2O_H2_over_R": params_report["DeltaHad_H2O_H2_over_R"],
        "DeltaHad_H2O_H2_kJ_per_mol": params_report["DeltaHad_H2O_H2_kJ_per_mol"],
        "physical_constraint_check": summary["physical_constraint_check"],
        "boundary_warning": summary["boundary_warning"],
    }

    return params_output


def calculate_parameter_stability(summary_df):

    rows = []

    for optimizer_name, group in summary_df.groupby("optimizer_name"):
        row = {
            "optimizer_name": optimizer_name,
            "n_runs": len(group),
            "objective_mean": group["objective"].mean(),
            "objective_std": group["objective"].std(ddof=0),
            "avg_r2_mean": group["avg_r2"].mean(),
            "avg_r2_std": group["avg_r2"].std(ddof=0),
        }

        parameter_stds = []
        for name in PARAMETER_NAMES:
            std_value = group[name].std(ddof=0)
            row[f"{name}_std"] = std_value
            parameter_stds.append(std_value / max(BOUND_WIDTHS[PARAMETER_NAMES.index(name)], EPS))

        row["mean_normalized_parameter_std"] = float(np.mean(parameter_stds))
        rows.append(row)

    return pd.DataFrame(rows).sort_values("mean_normalized_parameter_std")


# ============================================================
# 15. Plotting
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


def save_best_prediction_outputs(best_runs):

    for optimizer_name, payload in best_runs.items():
        result_df = payload["result_df"]

        result_df.to_excel(
            OUT_DIR / f"predictions_{optimizer_name}.xlsx",
            index=False,
        )

        make_single_parity_plot(
            exp=result_df["rCH3OH"].values,
            pred=result_df["rCH3OH_pred"].values,
            xlabel="Experimental CH3OH",
            ylabel="Predicted CH3OH",
            title=f"Model E Integral LHHW 10 Param CH3OH {optimizer_name}",
            save_path=OUT_DIR / f"parity_CH3OH_{optimizer_name}.png",
        )

        make_single_parity_plot(
            exp=result_df["rCO"].values,
            pred=result_df["rCO_pred"].values,
            xlabel="Experimental CO",
            ylabel="Predicted CO",
            title=f"Model E Integral LHHW 10 Param CO {optimizer_name}",
            save_path=OUT_DIR / f"parity_CO_{optimizer_name}.png",
        )


# ============================================================
# 16. Saving outputs
# ============================================================

def save_comparison_outputs(summary_rows, best_runs):

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(["objective", "optimizer_name", "seed"]).reset_index(drop=True)

    stability_df = calculate_parameter_stability(summary_df)

    best_summary_rows = []
    best_parameter_rows = []

    for optimizer_name, group in summary_df.groupby("optimizer_name"):
        best_index = group["objective"].idxmin()
        best_summary = summary_df.loc[best_index].to_dict()
        best_summary_rows.append(best_summary)
        best_parameter_rows.append(make_parameter_output(best_summary))

    best_summary_df = pd.DataFrame(best_summary_rows).sort_values("objective")
    best_parameters_df = pd.DataFrame(best_parameter_rows).sort_values("objective")

    with pd.ExcelWriter(OUT_DIR / "optimizer_comparison_summary.xlsx") as writer:
        summary_df.to_excel(writer, sheet_name="all_runs", index=False)
        best_summary_df.to_excel(writer, sheet_name="best_by_optimizer", index=False)
        stability_df.to_excel(writer, sheet_name="parameter_stability", index=False)

    best_parameters_df.to_excel(
        OUT_DIR / "optimizer_best_parameters.xlsx",
        index=False,
    )

    save_best_prediction_outputs(best_runs)

    log_row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_name": MODEL_NAME,
        "data_file": str(EXCEL_FILE),
        "n_runs": len(summary_df),
        "n_points": int(best_runs[next(iter(best_runs))]["result_df"].shape[0]),
        "W_cat_kg": W_CAT_KG,
        "F_total_in_mol_s": F_TOTAL_IN,
        "RK4_STEPS": RK4_STEPS,
        "FAST_TEST_MODE": FAST_TEST_MODE,
        "SEEDS": str(SEEDS),
        "DE_POPSIZE": DE_POPSIZE,
        "DE_MAXITER": DE_MAXITER,
        "DE_TOL": DE_TOL,
        "DE_MUTATION": str(DE_MUTATION),
        "DE_RECOMBINATION": DE_RECOMBINATION,
        "DE_POLISH": DE_POLISH,
        "GA_POPULATION_SIZE": GA_POPULATION_SIZE,
        "GA_GENERATIONS": GA_GENERATIONS,
        "GA_ELITE_FRACTION": GA_ELITE_FRACTION,
        "GA_CROSSOVER_RATE": GA_CROSSOVER_RATE,
        "GA_MUTATION_RATE": GA_MUTATION_RATE,
        "GA_MUTATION_SCALE": GA_MUTATION_SCALE,
        "PSO_SWARM_SIZE": PSO_SWARM_SIZE,
        "PSO_ITERATIONS": PSO_ITERATIONS,
        "PSO_INERTIA": PSO_INERTIA,
        "PSO_COGNITIVE": PSO_COGNITIVE,
        "PSO_SOCIAL": PSO_SOCIAL,
    }

    for _, row in best_summary_df.iterrows():
        prefix = row["optimizer_name"]
        log_row[f"{prefix}_best_objective"] = row["objective"]
        log_row[f"{prefix}_best_avg_r2"] = row["avg_r2"]
        log_row[f"{prefix}_best_seed"] = row["seed"]

    log_df = pd.DataFrame([log_row])

    if EXPERIMENT_LOG_PATH.exists():
        old_log = pd.read_csv(EXPERIMENT_LOG_PATH)
        log_df = pd.concat([old_log, log_df], ignore_index=True)

    log_df.to_csv(EXPERIMENT_LOG_PATH, index=False)

    return summary_df, best_summary_df, stability_df


# ============================================================
# 17. Main function
# ============================================================

def main():

    print("\n==============================")
    print("Model E integral LHHW 10 parameter optimizer comparison")
    print("==============================")
    print(f"Excel file: {EXCEL_FILE}")
    print(f"Output folder: {OUT_DIR}")
    print(f"FAST_TEST_MODE = {FAST_TEST_MODE}")
    print(f"Seeds = {SEEDS}")
    print(f"W_cat_kg = {W_CAT_KG}")
    print(f"F_TOTAL_IN = {F_TOTAL_IN:.12g} mol/s")
    print(f"RK4_STEPS = {RK4_STEPS}")
    print(f"DE popsize = {DE_POPSIZE}")
    print(f"DE maxiter = {DE_MAXITER}")
    print(f"DE polish = {DE_POLISH}")
    print(f"GA population_size = {GA_POPULATION_SIZE}")
    print(f"GA generations = {GA_GENERATIONS}")
    print(f"PSO swarm_size = {PSO_SWARM_SIZE}")
    print(f"PSO iterations = {PSO_ITERATIONS}")

    df = load_data()

    rCH3OH_exp = df["rCH3OH"].to_numpy(dtype=float)
    rCO_exp = df["rCO"].to_numpy(dtype=float)
    temperature = df["T"].to_numpy(dtype=float)
    Tave = float(np.mean(temperature))

    print("\n" + "=" * 60)
    print("Starting optimizer comparison for Model E integral LHHW 10 parameter model")
    print(f"Data points = {len(df)}")
    print(f"Tave = {Tave:.2f} K")
    print("=" * 60)

    objective_args = (df, Tave, rCH3OH_exp, rCO_exp)

    optimizer_functions = [
        ("DE", run_de),
        ("GA", run_ga),
        ("PSO", run_pso),
    ]

    summary_rows = []
    best_runs = {}

    for optimizer_name, optimizer_function in optimizer_functions:
        for seed in SEEDS:
            raw_result = optimizer_function(seed, objective_args)
            summary, result_df = evaluate_optimizer_result(
                raw_result=raw_result,
                df_group=df,
                Tave=Tave,
                rCH3OH_exp=rCH3OH_exp,
                rCO_exp=rCO_exp,
            )
            summary_rows.append(summary)

            current_best = best_runs.get(optimizer_name)
            if current_best is None or summary["objective"] < current_best["summary"]["objective"]:
                best_runs[optimizer_name] = {
                    "summary": summary,
                    "result_df": result_df,
                }

            print(
                f"Completed {optimizer_name} seed {seed}: "
                f"objective = {summary['objective']:.6g}, "
                f"avg_r2 = {summary['avg_r2']:.6f}, "
                f"constraint = {summary['physical_constraint_check']}, "
                f"boundary = {summary['boundary_warning']}"
            )

    summary_df, best_summary_df, stability_df = save_comparison_outputs(
        summary_rows=summary_rows,
        best_runs=best_runs,
    )

    best_objective_row = summary_df.loc[summary_df["objective"].idxmin()]
    best_avg_r2_row = summary_df.loc[summary_df["avg_r2"].idxmax()]
    most_stable_row = stability_df.iloc[0]

    print("\nAll optimizer runs completed.")
    print("\nBest objective:")
    print(
        f"{best_objective_row['optimizer_name']} seed {int(best_objective_row['seed'])}, "
        f"objective = {best_objective_row['objective']:.6g}"
    )

    print("\nBest avg_r2:")
    print(
        f"{best_avg_r2_row['optimizer_name']} seed {int(best_avg_r2_row['seed'])}, "
        f"avg_r2 = {best_avg_r2_row['avg_r2']:.6f}"
    )

    print("\nMost stable parameters across seeds:")
    print(
        f"{most_stable_row['optimizer_name']}, "
        f"mean normalized parameter std = "
        f"{most_stable_row['mean_normalized_parameter_std']:.6g}"
    )

    print("\nBest run by optimizer:")
    print(best_summary_df[["optimizer_name", "seed", "objective", "avg_r2"]])

    print("\nOutputs saved to:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
