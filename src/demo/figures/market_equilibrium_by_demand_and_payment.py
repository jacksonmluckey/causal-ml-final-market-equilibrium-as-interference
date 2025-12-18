from demo.method import (
   create_queue_allocation,
   create_linear_revenue,
   create_logistic_choice,
   create_lognormal_costs,
   find_equilibrium_supply_mu,
   compute_mean_field_utility,
   MarketParameters,
   MeanFieldMarketParameters,
   SupplierParameters
)

# Set up the suppliers
choice_fn = create_logistic_choice(alpha=1.0)
cost_dist = create_lognormal_costs(log_mean=0.0, log_std=1.0, scale=20.0)
supplier_params = SupplierParameters(choice=choice_fn, private_features=cost_dist)

# Set up the market
allocation = create_queue_allocation(L=8)
gamma = 100  # Platform revenue per unit served
revenue_fn = create_linear_revenue(gamma, allocation)
market_params = MarketParameters(allocation, supplier_params, gamma)

# Analyze for different demand levels
d_a_values = [x / 10 for x in range(1, 10)]
payments = [x for x in range(0, 101)]

#

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Store results for comparison
results = {}

for d_a in d_a_values:
    utilities = []
    mus = []

    # Create mean-field parameters for this demand level
    mean_field_params = MeanFieldMarketParameters(
        allocation=allocation,
        choice=choice_fn,
        private_features=cost_dist,
        d_a=d_a,
        gamma=gamma,
        n_monte_carlo=10000
    )

    for p in payments:
        mu = find_equilibrium_supply_mu(p, mean_field_params)
        u = compute_mean_field_utility(revenue_fn, allocation, supplier_params, d_a, p)
        utilities.append(u)
        mus.append(mu)
    
    results[d_a] = {'utilities': utilities, 'mus': mus}
    
    # Find optimal
    opt_idx = np.argmax(utilities)
    