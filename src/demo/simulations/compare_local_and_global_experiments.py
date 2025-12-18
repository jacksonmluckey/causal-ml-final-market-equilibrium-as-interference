from demo.method import (
    ExperimentParams,
    SupplierParameters,
    create_logistic_choice,
    create_lognormal_costs,
    create_queue_allocation,
    create_linear_revenue,
    run_learning_algorithm,
    run_global_learning,
    # TODO will need to calculate true p^* first.
    #compute_cumulative_regret,
    DemandParameters,
    GlobalState,
)
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

n_days = 200
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
    rng_seed=20251215,
    store_detailed_data=False
)

# Global experimentation parameters
global_params = ExperimentParams(
    T=n_days,
    n=n_suppliers,
    p_init=25,
    revenue_fn=revenue_fn,
    p_bounds=(0, 100),
    allocation=allocation,
    supplier_params=supplier,
    demand=demand,
    eta=20,
    experiment_type="global",
    zeta=None,
    alpha=None,
    delta=1.0,
    rng_seed=20251215,
    store_detailed_data=False
)


if __name__ == "__main__":
    print("Running local experimentation...")
    local_exp = run_learning_algorithm(params=local_params, rng=rng)

    print("Running global experimentation...")
    global_exp = run_global_learning(params=global_params, rng=rng)

    local_df = local_exp.to_polars().with_columns(pl.lit('local').alias('experimentation'))
    global_df = global_exp.to_polars().with_columns(pl.lit('global').alias('experimentation'))
    df = pl.concat([local_df, global_df], how = 'diagonal')
    df_path = get_data_path('local_vs_global_experiment.csv')
    df.write_csv(df_path)
    print(f"Saved results to {df_path}")
