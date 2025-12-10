import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# =============================================================================
# STEP 1: Define the Congestion Model
# =============================================================================

class Road:
    """
    A road where:
    - n potential drivers decide whether to use a tolled road
    - Travel time increases with the number of drivers (congestion)
    - Each driver has a trip value (willingness to pay) and time cost
    """

    def __init__(self, n_drivers=1000, free_flow_time=20, capacity=600):
        self.n = n_drivers
        self.free_flow_time = free_flow_time  # minutes, when road is empty
        self.capacity = capacity  # practical capacity of the road

    def travel_time(self, n_drivers_on_road: int, alpha = 0.15, beta = 0.4):
        r"""
        Standard congestion model used in transportation economics.
    
        Parameters
        ----------
        n_drivers_on_road : int
            Number of drivers on the road
    
        Notes
        -----
        Uses the BPR (Bureau of Public Roads) congestion function:
    
        .. math::
    
            T(n) = T_0 \times \left(1 + \alpha \times \left(\frac{n}{C}\right)^{\beta}\right)
    
        where:
    
        - :math:`T_0`: free-flow travel time (minutes)
        - :math:`C`: road capacity (vehicles/hour)
        - :math:`n`: number of drivers on road
        - :math:`\alpha`: calibration parameter (typically 0.15)
        - :math:`\beta`: calibration parameter (typically 4)
        """
        alpha = 0.15
        beta = 4

        if n_drivers_on_road <= 0:
            return self.free_flow_time

        volume_ratio = n_drivers_on_road / self.capacity
        return self.free_flow_time * (1 + alpha * (volume_ratio ** beta))

    def travel_time_derivative(self, n_drivers_on_road):
        r"""
        Derivative of travel time with respect to number of drivers.
        .. math::

        \frac{dT}{dn} = T_0 \times \alpha \times \beta \times \left(\frac{n}{C}\right)^{\beta-1} \times \frac{1}{C}
        """
        alpha = 0.15
        beta = 4

        if n_drivers_on_road <= 0:
            return 0

        volume_ratio = n_drivers_on_road / self.capacity
        return self.free_flow_time * alpha * beta * (volume_ratio ** (beta - 1)) / self.capacity

    def driver_utility(self, trip_value, toll, travel_time, value_of_time=1.0):
        """
        Driver utility from taking the tolled road:
        U = V - p - γ × T(n)

        where:
        - V: value of completing the trip via this road
        - p: toll paid
        - γ: value of time (converts minutes to dollars)
        - T(n): travel time given n drivers
        """
        return trip_value - toll - value_of_time * travel_time

    def alternative_utility(self, trip_value):
        """
        Utility from taking alternative (other road, public transit, not traveling).

        For simplicity, assume alternative has fixed travel time but no toll.
        Different drivers have different alternatives (captured in trip_value distribution).
        """
        alternative_time = 35  # Alternative is slower but free
        value_of_time = 1.0
        return trip_value - value_of_time * alternative_time

# Create road
road = Road(n_drivers=1000, free_flow_time=20, capacity=600)

# Visualize congestion function
driver_counts = np.linspace(0, 1000, 100)
travel_times = [road.travel_time(n) for n in driver_counts]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(driver_counts, travel_times, 'b-', linewidth=2)
axes[0].axhline(y=35, color='r', linestyle='--', label='Alternative route time')
axes[0].axvline(x=road.capacity, color='g', linestyle=':', label='Road capacity')
axes[0].set_xlabel('Number of Drivers on Road')
axes[0].set_ylabel('Travel Time (minutes)')
axes[0].set_title('Congestion Function (BPR Model)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Show marginal congestion cost (externality)
marginal_costs = [road.travel_time_derivative(n) * n for n in driver_counts]
axes[1].plot(driver_counts, marginal_costs, 'm-', linewidth=2)
axes[1].set_xlabel('Number of Drivers on Road')
axes[1].set_ylabel('Total Externality Cost ($)')
axes[1].set_title('Congestion Externality (Cost Imposed on Others)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()