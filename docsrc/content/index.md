# Market Equilibrium as Interference Documentation

This package implements the marketplace equilibrium model from Wager & Xu (2021)
"Experimenting in Equilibrium", demonstrating how market interference affects
platform experiments.

The implementation focuses on stochastic markets where:

- Suppliers make activation decisions based on expected revenue
- Demand is allocated among active suppliers via a regular allocation function
- Market equilibrium emerges from the interaction of supply and demand
- Platform experiments (local vs. global) have different equilibrium effects

## Key Components

**Allocation Functions** (`demo.allocation`)
:   Regular allocation functions $\omega(x)$ that map demand-to-supply ratios
    to expected demand served per active supplier.

**Supplier Choice Models** (`demo.supplier`)
:   Choice functions $f_b(x)$ modeling supplier activation decisions based on
    expected revenue and private features (e.g., outside options).

**Equilibrium Computation** (`demo.find_equilibrium`)
:   Mean-field equilibrium solver implementing the fixed-point equation:
    $\mu = E[f_{B_1}(p \cdot \omega(d_a/\mu)) | A=a]$

**Experimentation** (`demo.local_experimentation`, `demo.global_experimentation`)
:   Local and global experiment implementations for comparing platform strategies.

**Utility and Revenue** (`demo.platform_utility`, `demo.revenue`)
:   Platform utility and revenue functions for different market scenarios.

## Getting Started

Install dependencies using uv:

```bash
uv sync --group docs
```

Build documentation locally:

```bash
uv run python scripts/build_docs.py
```

Open the generated documentation:

```bash
open site/index.html
```

## API Reference

See the [API Reference](api/allocation.md) section for detailed documentation of all modules.
