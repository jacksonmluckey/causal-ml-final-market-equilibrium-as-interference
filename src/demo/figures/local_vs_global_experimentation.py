from demo.utils import (
    get_data_path,
    get_figures_path
)
import polars as pl
import plotnine as gg
import matplotlib.pyplot as plt


df = pl.read_csv(get_data_path('local_vs_global_experiment.csv'))

df_long = df.unpivot(
    on = ['p', 'U'],
    index = ['experimentation', 't', 'state_name']
)

def exp_labeller(x):
    return x.title()

def var_labeller(x):
    mapping = {
        'p': r'Payment $p$',
        'U': r'Utility $U$'
    }
    return mapping.get(x, x)

# Use LaTeX for rendering
plt.rcParams['text.usetex'] = True

(
    gg.ggplot(df_long, gg.aes(x = 't', y = 'value', color = 'state_name')) +
    gg.geom_point() +
    gg.facet_grid(
        cols = 'experimentation',
        rows = 'variable',
        scales = 'free_y',
        labeller = gg.labeller(
            experimentation = exp_labeller,
            variable = var_labeller
        )
    ) +
    gg.geom_smooth(gg.aes(group = None)) +
    gg.labs(
        title = r'\noindent Global Versus Local Experimentation:\\ Utility $U$ and Payment $p$ Over Time',
        x = r'Day ($t$)',
        color = r'Demand ($d_a$)'
    ) +
    gg.theme_minimal() +
    gg.theme(
        legend_position = "bottom",
        axis_title_y = gg.element_blank()
    )
)
