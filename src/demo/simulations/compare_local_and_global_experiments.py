from demo.method import (
    ExperimentParams,
    SupplierParameters,
    baseline_global_experimentation,
    create_logistic_choice,
    create_lognormal_costs,
    create_queue_allocation,
    create_linear_revenue,
    run_learning_algorithm,
    run_baseline_global_learning,
    run_kiefer_wolfowitz_global_learning,
    # TODO will need to calculate true p^* first.
    #compute_cumulative_regret,
    DemandParameters,
    GlobalState,
)
from demo.method.experiment import Experiment
from demo.utils import (
    get_data_path
)
import numpy as np
import polars as pl


supplier = SupplierParameters(
    create_logistic_choice(alpha = 1),
    create_lognormal_costs(
        log_mean = 0.0,
        log_std = 1.0,
        scale = 20.0
    )
)

demand = DemandParameters(
    states={
        "low": GlobalState(name="low", d_a=0.4, probability=0.25),
        "medium": GlobalState(name="medium", d_a=0.5, probability=0.5),
        "high": GlobalState(name="high", d_a=0.6, probability=0.25),
    },
    concentration_param=50.0
)

allocation = create_queue_allocation(L = 8)

rng = np.random.default_rng(seed=20251215)

n_days = 250
n_suppliers = 100
gamma = 100

# Create revenue function (needs allocation)
revenue_fn = create_linear_revenue(gamma=gamma, allocation=allocation)

# Local experimentation parameters
local_params = ExperimentParams(
    T=n_days,
    n=n_suppliers,
    p_init=25,
    revenue_fn=revenue_fn,
    p_bounds=(0, 100),
    allocation=allocation,
    supplier_params=supplier,
    demand=demand,
    eta=20,
    experiment_type="local",
    zeta=0.15,
    alpha=0.3,
    delta=None,
    store_detailed_data=False
)

global_baseline_params = ExperimentParams(
    T=n_days,
    n=n_suppliers,
    p_init=25,
    revenue_fn=revenue_fn,
    p_bounds=(0, 100),
    allocation=allocation,
    supplier_params=supplier,
    demand=demand,
    experiment_type="global",
    zeta=None,
    alpha=None,
    T_explore = int(n_days / 5),
    store_detailed_data=False    
)

# Global experimentation parameters
global_kiefer_wolfowitz_params = ExperimentParams(
    T=n_days,
    n=n_suppliers,
    p_init=25,
    revenue_fn=revenue_fn,
    p_bounds=(0, 100),
    allocation=allocation,
    supplier_params=supplier,
    demand=demand,
    eta=1,
    experiment_type="global",
    zeta=None,
    alpha=None,
    delta=1.0,
    store_detailed_data=False
)


if __name__ == "__main__":
    print("Running local experimentation...")
    local_exp = run_learning_algorithm(params=local_params, rng=rng)

    print("Running baseline global experimentation...")
    global_baseline_exp = run_baseline_global_learning(params=global_baseline_params, rng=rng)

    print("Running Kiefer-Wolfowitz global experimentation...")
    global_kiefer_wolfowitz_exp = run_kiefer_wolfowitz_global_learning(params=global_kiefer_wolfowitz_params, rng=rng)

    local_df = local_exp.to_polars().with_columns(pl.lit('Local').alias('experimentation'))
    global_baseline_df = global_baseline_exp.to_polars().with_columns(pl.lit('Global (Baseline Bandit)').alias('experimentation'))
    global_kiefer_wolfowitz_df = global_kiefer_wolfowitz_exp.to_polars().with_columns(pl.lit('Global (Kiefer Wolfowitz)').alias('experimentation'))
    df = pl.concat([local_df, global_baseline_df, global_kiefer_wolfowitz_df], how = 'diagonal')
    df_path = get_data_path('local_vs_global_experiment.csv')
    df.write_csv(df_path)
    print(f"Saved results to {df_path}")
