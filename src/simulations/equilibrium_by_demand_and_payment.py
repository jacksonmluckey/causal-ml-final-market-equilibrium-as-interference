from demo.find_equilibrium import compute_mean_field_equilibrium
from demo.allocation import create_queue_allocation
from demo.supplier import create_lognormal_costs, create_logistic_choice
import polars as pl
from itertools import product
from concurrent.futures import ProcessPoolExecutor


def main(d_a_values, p_values, gamma, choice, private_features, allocation):
    # Find equilibrium with every combination of p and d_a
    params = list(product(p_values, d_a_values))

    print(f'Running {len(params)} simulations')
    
    # Find equilibriums in parallel
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                compute_mean_field_equilibrium,
                p = p,
                d_a = d_a,
                gamma = gamma,
                choice = choice,
                private_features = private_features,
                allocation = allocation
            )
            for p, d_a in params
        ]
        equilibriums = [f.result() for f in futures]

    print("did I run??")

    # Want: demand served, fraction of suppliers active, demand per active supplier
    df = (pl.DataFrame(equilibriums)
        .with_columns(demand_served = pl.col("mu") * pl.col("q")))
    
    # TODO use something better than `../../`
    df.write_csv('data/equilibrium_by_demand_and_payment.csv')
    print('Simulations saved to data/equilibrium_by_demand_and_payment.csv')


if __name__ == "__main__":
    d_a_values = [x / 10 for x in range(1, 10)]
    p_values = [x for x in range(1, 100)]
    gamma = 100
    choice = create_logistic_choice(alpha = 1),
    private_features = create_lognormal_costs(log_mean = 0, log_std = 1, scale = 20),
    allocation = create_queue_allocation(L = 8)
    main(d_a_values, p_values, gamma, choice, private_features, allocation)