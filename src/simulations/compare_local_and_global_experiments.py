from demo import (
    ExperimentParams,
    SupplierParameters,
    create_logistic_choice,
    create_lognormal_costs,
    create_queue_allocation,
    DemandParameters,
    GlobalState
)
import numpy as np

from demo.experiment_results import Experiment


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

ExperimentParams(
    T = n_days
    n = n_suppliers
    p_init = 25
    gamma = gamma
    p_bounds = (0, 100)
    allocation = allocation
    supplier_params = supplier
    demand = demand
    eta = 20
    experiment_type = 
    zeta =
    alpha =
    delta = 
    rng = rng
)