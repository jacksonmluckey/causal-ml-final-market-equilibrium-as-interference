from mvp import (
    create_queue_allocation,
    create_linear_allocation,
    compute_omega,
    compute_omega_derivative,
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
from utils import get_figures_path


allocations = []
# Calculate allocations using queue allocation mechanism
demand_per_active_supplier_values = np.linspace(0.01, 5, 1000)
supplier_queue_capacity_values = [2, 3, 4, 8, 15, 30, 50]
for queue_capacity in supplier_queue_capacity_values:
    queue_allocation = create_queue_allocation(L = queue_capacity)
    for demand_per_active_supplier in demand_per_active_supplier_values:
        allocation_per_supplier = compute_omega(queue_allocation, demand_per_active_supplier)
        omega_derivative = compute_omega_derivative(queue_allocation, demand_per_active_supplier)
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
    allocation_per_supplier = compute_omega(linear_allocation, demand_per_active_supplier)
    omega_derivative = compute_omega_derivative(linear_allocation, demand_per_active_supplier)
    allocation = {
        "allocation_function": "linear",
        "demand_per_active_supplier": demand_per_active_supplier,
        "allocation_per_supplier": allocation_per_supplier,
        "omega_derivative": omega_derivative,
        "queue_capacity": None
    }
    allocations.append(allocation)

allocation_df = pl.DataFrame(
    allocations,
    schema = ["allocation_function", "demand_per_active_supplier", "allocation_per_supplier", "omega_derivative", "queue_capacity"]
).with_columns(
    # Add nice labels for plotting
    pl.when(pl.col.allocation_function == "queue")
    .then(pl.concat_str([
        pl.lit("Queue (L="),
        pl.col.queue_capacity,
        pl.lit(")")
    ]))
    .when(pl.col.allocation_function == "linear")
    .then(pl.lit(r"Linear ($\lim_{L \rightarrow \infty}$)"))
    .alias("allocation_mechanism")
)

# Want to be able to use LaTeX expressions in plot
#plt.rcParams['backend'] = 'pgf'
plt.rcParams['text.usetex'] = True

(
    ggplot(allocation_df, aes(
        x = "demand_per_active_supplier",
        y = "allocation_per_supplier",
        color = "allocation_mechanism")
    ) +
    geom_line() +
    theme_minimal() +
    labs(
        x = r'$x$ (Demand Per Active Supplier)',
        y = r'$\omega(x)$ (Allocation Per Active Supplier)'
    )
).save(get_figures_path("allocations_by_allocation_function.png"))
