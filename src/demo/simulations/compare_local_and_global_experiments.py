from demo.method import (
    ExperimentParams,
    SupplierParameters,
    create_logistic_choice,
    create_lognormal_costs,
    create_queue_allocation,
    create_linear_revenue,
    run_learning_algorithm,
    run_global_experimentation,
    compute_cumulative_regret,
    DemandParameters,
    GlobalState,
)
from demo.utils import get_data_path
import numpy as np
import polars as pl


supplier = SupplierParameters(
    create_logistic_choice(alpha=1),
    create_lognormal_costs(log_mean=0.0, log_std=1.0, scale=20.0),
)

demand = DemandParameters(
    states={
        "low": GlobalState(name=r"Low ($d_a = 0.3$)", d_a=0.3, probability=0.25),
        "medium": GlobalState(name=r"Medium ($d_a = 0.5$)", d_a=0.5, probability=0.5),
        "high": GlobalState(name=r"High ($d_a = 0.7$)", d_a=0.7, probability=0.25),
    },
    concentration_param=50.0,
)

allocation = create_queue_allocation(L=5)

rng = np.random.default_rng(seed=20251219)

n_days = 200
n_suppliers = 10000
gamma = 80

# Create revenue function (needs allocation)
revenue_fn = create_linear_revenue(gamma=gamma, allocation=allocation)

# Local experimentation parameters
local_params = ExperimentParams(
    T=n_days,
    n=n_suppliers,
    p_init=50,
    revenue_fn=revenue_fn,
    p_bounds=(0, 100),
    allocation=allocation,
    supplier_params=supplier,
    demand=demand,
    eta=10,
    experiment_type="local",
    zeta=0.5,
    alpha=0.3,
    delta=None,
    store_detailed_data=False,
)

global_baseline_params = ExperimentParams(
    T=n_days,
    n=n_suppliers,
    p_init=50,
    revenue_fn=revenue_fn,
    p_bounds=(0, 100),
    allocation=allocation,
    supplier_params=supplier,
    demand=demand,
    experiment_type="global",
    zeta=None,
    alpha=None,
    store_detailed_data=False,
)


if __name__ == "__main__":
    print("Running local experimentation...")
    local_exp = run_learning_algorithm(params=local_params, rng=rng)

    print("Running epsilon-greedy (epsilon = 0.2) global experimentation...")
    global_exp = run_global_experimentation(
        strategy="epsilon_greedy", params=global_baseline_params, epsilon=0.2, rng=rng
    )

    # Imperfect, but true p* is very close to final payment from the local experiment
    optimal_p = local_exp.results.final_payment
    local_exp.compute_regret(optimal_p)
    global_exp.compute_regret(optimal_p)

    local_df = local_exp.to_polars().with_columns(
        pl.lit("Local").alias("experimentation")
    )
    global_df = global_exp.to_polars().with_columns(
        pl.lit("Global (Epsilon-Greedy)").alias("experimentation")
    )
    df = pl.concat([local_df, global_df], how="diagonal")
    df_path = get_data_path("local_vs_global_experiment.csv")
    df.write_csv(df_path)
    print(f"Saved results to {df_path}")
