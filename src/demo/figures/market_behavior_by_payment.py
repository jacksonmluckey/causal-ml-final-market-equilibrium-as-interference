import polars as pl
import plotnine as gg
import matplotlib.pyplot as plt
from demo.utils import get_data_path, get_figures_path


def main():
    df = (
        pl.read_csv(get_data_path("equilibrium_by_demand_and_payment.csv"))
        # Only want 0.1, 0.2, ... , 0.8, 0.9
        .filter(pl.col("d_a").is_in([x / 10 for x in range(1, 10)]))
        # demand served = active * demand per active
        .with_columns(demand_served=pl.col("mu") * pl.col("q"))
        # want long (1 row per d_a-p-outcome_var)
        .unpivot(
            index=["p", "d_a"],
            on=["mu", "q", "demand_served"],
            variable_name="var",
            value_name="value",
        )
    )

    # Use LaTeX for rendering
    plt.rcParams["text.usetex"] = True

    # Generate the faceted plot (1 subplot per value of d_a)
    (
        gg.ggplot(df, gg.aes(x="p", y="value", color="var", group="var"))
        + gg.geom_line()
        + gg.facet_wrap("~d_a", labeller=lambda d_a: rf"$d_a$ = {d_a}")
        + gg.scale_color_manual(
            values={"mu": "red", "q": "blue", "demand_served": "black"},
            labels={
                "mu": r"Proportion of Suppliers Active $\mu_{d_a}(p)$",
                "q": r"Demand Per Active Supplier $q_{d_a}(p)$",
                "demand_served": r"Demand Served",
            },
        )
        + gg.labs(
            x=r"Payment $p$",
            y="Fraction",
            title=r"\noindent Proportion of Suppliers Active $\mu_{d_a}(p)$ and Demand Per Active Supplier $q_{d_a}(p)$\\ as Demand $d_a$ and Payment $p$ Vary",
            subtitle=r"Allocation with $L=8$ in mean-field limit as $n \rightarrow \infty$",
        )
        + gg.theme_minimal()
        + gg.theme(legend_position="bottom", legend_title=gg.element_blank())
    ).save(
        width=7.5,
        height=5,
        filename=get_figures_path("market_behavior_by_payment_and_demand.png"),
    )


if __name__ == "__main__":
    main()
