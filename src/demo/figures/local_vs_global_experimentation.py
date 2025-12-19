from demo.utils import (
    get_data_path,
    get_figures_path
)
import polars as pl
import plotnine as gg
import matplotlib.pyplot as plt


df = (pl.read_csv(
        get_data_path('local_vs_global_experiment.csv')
    )
    # add cumulative sum
    .with_columns(
        pl.col("U")
        .cum_sum()
        .over("experimentation")
        .alias("cumulative_utility")
    )
    # pivot longer for ggplot (1 row per timepoint-experiment-variable)
    .unpivot(
        on = ['p', 'U', 'cumulative_utility', 'cumulative_regret'],
        index = ['experimentation', 't', 'state_name']
    )
    .with_columns(
        # order the variables on the plot
        pl.col('variable')
        .cast(pl.Enum(['U', 'p', 'cumulative_regret', 'cumulative_utility'])),
        # only want to color in the utility facet and want to order high -> low
        pl.when(pl.col('variable') == 'U')
        .then(pl.col('state_name'))
        .otherwise(None)
        .cast(pl.Enum([r"High ($d_a = 0.7$)", r"Medium ($d_a = 0.5$)", r"Low ($d_a = 0.3$)"]))
        .alias('state_name_colored')
    )
)

def exp_labeller(x):
    mapping = {
        'Global (Epsilon-Greedy)': r'\large Global (Epsilon-Greedy)' + '\n' + r'\small ($\epsilon = 0.2$, Step Size = $\frac{p_t}{10})$',
        'Local': r'\large Local' + '\n' + r'\small ($\eta = 10, \zeta = 0.5$)'
    }
    return mapping.get(x, x)

def var_labeller(x):
    mapping = {
        'p': r'Payment $p$',
        'U': r'Utility $U$',
        'cumulative_utility': 'Cumulative Utility',
        'cumulative_regret': 'Cumulative Regret'
    }
    return mapping.get(x, x)

# Use LaTeX for rendering
plt.rcParams['text.usetex'] = True

(
    gg.ggplot(df, gg.aes(x = 't', y = 'value', color = 'state_name_colored')) +
    gg.geom_point() +
    gg.scale_color_discrete(
        breaks = [x for x in df['state_name_colored'].unique() if x is not None]
    ) +
    gg.facet_grid(
        cols = 'experimentation',
        rows = 'variable',
        scales = 'free_y',
        labeller = gg.labeller(
            experimentation = exp_labeller,
            variable = var_labeller
        )
    ) +
    #gg.geom_smooth(gg.aes(group = None)) +
    gg.labs(
        title = 'Global Versus Local Experimentation in Market Equilibrium:' + '\n' + r'Payment $p$, Utility $U$, and Cumulative Utility and Regret Over Time',
        subtitle = r'Local experimentation finds optimal price $p^*$ faster and has lower cumulative regret' + '\n',
        x = r'Day $t$',
        color = r'Demand Per Supplier $d_a$ on Day $t$:',
        caption = r'Platform receives $\gamma = 80$ per unit of demand supplied.' \
            + '\n' + r'Allocation made with queue length $L = 5$.' \
            + '\n' + r'Demand per supplier $d_a$ is sampled from: $P(d_a = 0.3) = 0.25, P(d_a = 0.5) = 0.5, P(d_a = 0.7) = 0.25$.' \
            + '\n' + r'Supplier costs $B_i$ are distributed lognormal ($\mu = 0, \sigma = 1$) with scale 20.'
    ) +
    gg.theme_minimal() +
    gg.theme(
        legend_position = "bottom",
        axis_title_y = gg.element_blank(),
        axis_text_y = gg.element_blank(),
        axis_ticks_y = gg.element_blank(),
        text = gg.element_text(ma = 'center'),
        plot_caption = gg.element_text(ma = 'right')
    )
).save(
    width = 6,
    height = 8,
    filename = get_figures_path('global_vs_local_experimentation.png')
)