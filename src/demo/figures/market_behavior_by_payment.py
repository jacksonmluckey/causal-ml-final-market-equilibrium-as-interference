from demo.method import (
    compute_mean_field_equilibrium,
    create_queue_allocation,
    create_lognormal_costs,
    create_logistic_choice
)
import polars as pl
import plotnine as gg
import matplotlib.pyplot as plt
from demo.utils import get_figures_path


d_a = 0.4
gamma = 100

choice = create_logistic_choice(alpha = 1)
private_features = create_lognormal_costs(log_mean = 0, log_std = 1, scale = 20)
allocation = create_queue_allocation(L = 8)

d_a_values = [x / 10 for x in range(1, 10)]
p_values = [x for x in range(1, 100)]
equilibriums = []
for d_a in d_a_values:
    for p in p_values:
        equilibrium = compute_mean_field_equilibrium(
            p,
            d_a,
            gamma,
            choice,
            private_features,
            allocation
        )
        print(f'Found equilibrium for p={p} d_a={d_a}')
        equilibriums.append(equilibrium)

# Want: demand served, fraction of suppliers active, demand per active supplier
df = (pl.DataFrame(equilibriums)
    .with_columns(demand_served = pl.col("mu") * pl.col("q")))
df_long = df.unpivot(
    index = ["p", "d_a"],
    on = ["mu", "q", "demand_served"],
    variable_name = "var",
    value_name = "value"
)

plt.rcParams['text.usetex'] = True

(
    gg.ggplot(
        df_long,
        aes(x = "p", y = "value", color = "var", group = "var")
    ) +
    gg.geom_line() +
    gg.facet_wrap(
        "~d_a",
        labeller = lambda d_a: rf'$d_a$ = {d_a}'
    ) +
    gg.scale_color_manual(
        values = {
            'mu': 'red',
            'q': 'blue',
            'demand_served': 'black'
        },
        labels = {
            'mu': r'Proportion of Suppliers Active $\mu_{d_a}(p)$',
            'q': r'Demand Per Active Supplier $q_{d_a}(p)$',
            'demand_served': r'Demand Served'
        }
    ) +
    gg.labs(
        x = r'Payment $p$',
        y = 'Fraction',
        title = r'\noindent Proportion of Suppliers Active $\mu_{d_a}(p)$ and Demand Per Active Supplier $q_{d_a}(p)$\\ as Demand $d_a$ and Payment $p$ Vary',
        subtitle = r'Allocation with $L=8$ in mean-field limit as $n \rightarrow \infty$'
    ) +
    gg.theme_minimal() +
    gg.theme(
        legend_position = "bottom",
        legend_title = gg.element_blank()
    )
).save(
    width = 7.5,
    height = 5,
    filename = get_figures_path("market_behavior_by_payment_and_demand.png")
)
