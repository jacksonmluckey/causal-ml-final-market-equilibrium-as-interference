# Markets as Interference: Experimenting in Equilibrium

Implements [_Experimenting in Equilibrium_ by Stefen Wager and Kuang Xu](https://doi.org/10.48550/arXiv.1903.02124)

## Setup

First create a Jupyter kernel: `uv run python -m ipykernel install --user --name=markets-as-interference --display-name="Market Equilibrium as Global Interference"`

Then edit the Kernel's JSON file (eg on MacOS `~/Library/Jupyter/kernels/markets-as-interference/kernel.json`) to include:

```{json}
{
    # Everything else
    "env": {
        "PYTHONPATH": PROJECT_PATH_GOES_HERE
  }
}
```

I do not understand why this is necessary, but otherwise `uv run quarto render` cannot access the `demo` package despite it being installed in the uv environment.