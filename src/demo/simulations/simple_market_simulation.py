from demo import (
    run_learning_algorithm,
    create_queue_allocation,
    SupplierParameters,
    create_logistic_choice,
    create_lognormal_costs,
    DemandParameters,
    GlobalState,
    experiment_to_dataframe,
)
import numpy as np
import polars as pl
from plotnine import *

supplier = SupplierParameters(
    create_logistic_choice(alpha=1),
    create_lognormal_costs(log_mean=0.0, log_std=1.0, scale=20.0),
)

demand = DemandParameters(
    states={
        "very low": GlobalState(name="very low", d_a=0.1),
        "low": GlobalState(name="low", d_a=0.3, probability=0.3),
        "medium": GlobalState(name="medium", d_a=0.5, probability=0.4),
        "high": GlobalState(name="high", d_a=0.8, probability=0.15),
        "very high": GlobalState(name="very high", d_a=1, probability=0.05),
    },
    concentration_param=50.0,
)

rng = np.random.default_rng(seed=20251214)

experiment_results = run_learning_algorithm(
    T=250,
    n=10000,
    p_init=10,
    eta=1,
    zeta=1,
    gamma=50,
    allocation=create_queue_allocation(L=2),
    supplier_params=supplier,
    demand_params=demand,
    p_bounds=(0, 100),
    verbose=True,
    rng=rng,
)

df = experiment_to_dataframe(experiment_results)


def experiment_results_to_df(experiment_results):
    timepoints = len(experiment_results.utility_history)
    return pl.DataFrame(
        {
            "d_a": [state.d_a for state in experiment_results.state_history],
            # exclude the initial payment
            "p": experiment_results.payment_history[1:],
            "u": experiment_results.utility_history,
            "t": [x + 1 for x in range(0, timepoints)],
        },
        strict=False,
    )


df = experiment_results_to_df(experiment_results)

(ggplot(df, aes(x="p", y="u", color="d_a")) + geom_point() + theme_minimal())

(ggplot(df, aes(x="t", y="p")) + geom_point() + theme_minimal())

(ggplot(df, aes(x="t", y="u", color="d_a")) + geom_point() + theme_minimal())
