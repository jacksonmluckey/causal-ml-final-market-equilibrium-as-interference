from demo.method import (
    create_queue_allocation,
    create_linear_allocation
)
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from plotnine import (
    ggplot,
    aes,
    geom_line,
    theme_minimal,
    labs
)
from demo.utils import get_figures_path


def allocations_by_function_and_demand():
    allocations = []
    # Calculate allocations using queue allocation mechanism
    demand_per_active_supplier_values = np.linspace(0.01, 5, 1000)
    supplier_queue_capacity_values = [2, 3, 4, 8, 15, 30, 50]
    for queue_capacity in supplier_queue_capacity_values:
        queue_allocation = create_queue_allocation(L = queue_capacity)
        for demand_per_active_supplier in demand_per_active_supplier_values:
            allocation_per_supplier = queue_allocation(demand_per_active_supplier)
            omega_derivative = queue_allocation.derivative(demand_per_active_supplier)
            allocation = {
                "allocation_function": "queue",
                "demand_per_active_supplier": demand_per_active_supplier,
                "allocation_per_supplier": allocation_per_supplier,
                "omega_derivative": omega_derivative,
                "queue_capacity": queue_capacity
            }
            allocations.append(allocation)

    # Calculate allocations using linear allocation mechanism
    linear_allocation = create_linear_allocation()
    for demand_per_active_supplier in demand_per_active_supplier_values:
        allocation_per_supplier = linear_allocation(demand_per_active_supplier)
        omega_derivative = linear_allocation.derivative(demand_per_active_supplier)
        allocation = {
            "allocation_function": "linear",
            "demand_per_active_supplier": demand_per_active_supplier,
            "allocation_per_supplier": allocation_per_supplier,
            "omega_derivative": omega_derivative,
            "queue_capacity": 999999999 # L -> \infty for the Linear Allocation function
        }
        allocations.append(allocation)

    allocation_df = pl.DataFrame(
        allocations,
        schema = ["allocation_function", "demand_per_active_supplier", "allocation_per_supplier", "omega_derivative", "queue_capacity"]
    ).with_columns(
        # Add nice labels for plotting
        pl.when(pl.col.allocation_function == "queue")
        .then(pl.concat_str([
            pl.lit(r"$L=$"),
            pl.col.queue_capacity,
            pl.lit("")
        ]))
        .when(pl.col.allocation_function == "linear")
        .then(pl.lit(r"$\lim_{L \rightarrow \infty}$"))
        .alias("allocation_mechanism")
    )

    # Want to be able to use LaTeX expressions in plot
    #plt.rcParams['backend'] = 'pgf'
    plt.rcParams['text.usetex'] = True

    (
        ggplot(allocation_df, aes(
            x = "demand_per_active_supplier",
            y = "allocation_per_supplier",
            color = "reorder(allocation_mechanism, queue_capacity)")
        ) +
        geom_line() +
        theme_minimal() +
        labs(
            title = (
                r'Relationship Between Demand Per Active Supplier $x$' +
                '\n' +
                r'and Allocation Per Active Supplier $\omega(x)$'
                + '\n'
                + r'by Queue Length $L$ of Allocation Function $\omega$'
            ),
            subtitle = r'As $L$ increases, allocation per active supplier $\omega(x)$ becomes more linear',
            x = r'{\huge $x$}' + '\n\n' + r'{\small (Demand Per Active Supplier)}',
            y = r'{\huge $\omega(x)$}' + '\n\n' + r'{\small (Allocation Per Active Supplier)}',
            color = 'Allocation' + '\n' + 'Function' + '\n' + '(Queue)'
        )
    ).save(get_figures_path("allocations_by_allocation_function.png"))


if __name__ == "__main__":
    allocations_by_function_and_demand()