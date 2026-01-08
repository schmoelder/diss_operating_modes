import copy
import math
import os
from pathlib import Path
from typing import Optional

from cadetrdm import Case
from CADETProcess.fractionation import Fractionator
from CADETProcess.optimization import (
    OptimizationProblem,
    OptimizerBase,
    OptimizationResults,
)
from CADETProcess.simulationResults import SimulationResults
from CADETProcess import plotting
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import tabulate

from operating_modes.run_all import setup_study, setup_cases


def load_optimization_config(
    case: Case
) -> tuple[OptimizationProblem, OptimizerBase]:
    """Set up optimization problem and optimizer."""
    module = case.project_repo.module.optimization
    options = case.options
    optimization_problem = module.setup_optimization_problem(options)
    optimizer = module.setup_optimizer(optimization_problem, options.optimizer_options)
    return optimization_problem, optimizer


def load_optimization_results(case: Case) -> OptimizationResults:
    """Load optimization results for a given case."""
    case.load()
    options = case.options
    optimization_problem, optimizer = load_optimization_config(case)
    checkpoint_path = case.results_path / options.objective / "final.h5"
    optimization_results = OptimizationResults(optimization_problem, optimizer)
    optimization_results.load_results(checkpoint_path)
    return optimization_results


# %% Simulate and fractionate results

def simulate_results(
    optimization_problem: OptimizationProblem,
    x: list[float],
) -> SimulationResults:
    """Simulate individual of optimization results."""
    optimization_problem.set_variables(x)
    process = copy.deepcopy(optimization_problem.evaluation_objects[0])

    simulator = optimization_problem.evaluators[0]
    simulation_results = simulator.evaluate(process)

    return simulation_results


def fractionate_results(
    optimization_problem: OptimizationProblem,
    simulation_results: SimulationResults,
) -> Fractionator:
    """Optimize fractionation."""
    frac_opt = optimization_problem.evaluators[1]
    fractionator = frac_opt.evaluate(simulation_results)

    return fractionator


def simulate_and_plot(
    optimization_problem: OptimizationProblem,
    x: list[float],
) -> tuple[plt.Figure, plt.Axes]:
    """Simulate and plot individual of optimization results."""
    simulation_results = simulate_results(optimization_problem, x)
    fractionator = fractionate_results(optimization_problem, simulation_results)

    fig, ax = fractionator.plot_fraction_signal()

    return fig, ax


# %% Formatting

def format_float_adaptive(
    x: float,
    precision: int =3
) -> str:
    """
    Format a float using scientific notation.

    Note, this method keeps decimal notation if value is in [0.1, 1000).

    Parameters
    ----------
    x : float
        Value to format.
    precision : int
        Number of significant digits.

    Returns
    -------
    str
        Formatted string.
    """
    ax = abs(x)

    # Decide whether to use scientific notation
    if ax >= 0.1 and ax < 1000:
        # Keep significant digits and convert to decimal places
        if ax == 0:
            # Edge case: zero has no order of magnitude
            decimals = precision - 1
        else:
            order = math.floor(np.log10(ax))
            decimals = max(precision - order - 1, 0)
        return f"{x:.{decimals}f}"

    # Format scientific notation via numpy
    formatted = np.format_float_scientific(x, precision=precision-1, exp_digits=1)

    # Convert to LaTeX-style scientific notation
    if 'e' in formatted:
        base, exponent = formatted.split('e')
        base = base.rstrip('0').rstrip('.') if '.' in base else base
        formatted = f"{base} \\times 10^{{{int(exponent)}}}"

        # Remove unnecessary 10^0
        if float(exponent) == 0:
            formatted = base

    return formatted


def format_value_to_latex(
    value: float,
    bold: bool = False,
    precision: int = 3,
) -> str:
    """Format a single value to LaTeX with optional bolding."""
    formatted_val = format_float_adaptive(value, precision)

    # Apply bold formatting if needed
    return f"\\mathbf{{{formatted_val}}}" if bold else formatted_val


def get_next_n_entries(
    lst: list,
    n: int
) -> list:
    """Iterate over a list and yield the next n entries."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# %% Embed in MyST directives

def embed_table_in_directive(
    table: str,
    caption: Optional[str] = None,
    name: Optional[str] = None,
    align: Optional[str] = "center"
) -> str:
    """Format table to embed it in MyST table directive."""
    formatted_table = "```{table}"
    if caption:
        formatted_table += f" {caption}"

    formatted_table += "\n"
    if name:
        formatted_table += f":name: {name}\n"
    formatted_table += ":widths: grid\n"
    formatted_table += f":align: {align}\n"
    formatted_table += "\n"
    formatted_table += f"{table}\n"
    formatted_table += "```"

    return formatted_table


def embed_figure_in_directive(
    case: Case,
    figure_path: os.PathLike,
    name: None,
    caption: str,
    scale: Optional[int] = 100,
    width: Optional[int] = None,
) -> str:
    """Format figure to embed it in MyST figure directive."""
    results_branch = case.results_branch

    cache_path = case.project_repo.copy_data_to_cache(results_branch)
    results_dir = cache_path / case.name.split("_")[0]
    relative_results_dir = results_dir.relative_to(Path.cwd(), walk_up=True)

    figure_path = relative_results_dir / figure_path
    if not figure_path.exists():
        raise FileNotFoundError("Figure not found.")

    embedded_figure = f"```{{figure}} {str(figure_path)}"

    embedded_figure += "\n"
    if name:
        embedded_figure += f":name: {name}\n"
    embedded_figure += f":scale: {scale}\n"
    if width:
        embedded_figure += f":width: {width}\n"
    embedded_figure += "\n"
    embedded_figure += f"{caption}\n"
    embedded_figure += "```"

    return embedded_figure


# %% Single objective results

def setup_soo_results_table(
    soo_results: OptimizationResults,
    fractionator: Fractionator,
    variable_units: dict[str, str]=None
):
    """Set up KPI table for single-objective optimization results."""
    soo_problem = soo_results.optimization_problem
    process_name = soo_problem.evaluation_objects[0].name
    f = soo_results.f[0]

    ind = soo_results.pareto_front.individuals[0]

    headers = ["Category", "Parameter", "Value(s)"]

    # Variable values
    if variable_units is None:
        variable_units = {rf"\text{{{var_name}}}": "-" for var_name in soo_problem.variable_names}

    categories = ind.n_x * [" "]
    categories[0] = "**Variables**"

    rows = []
    rows += [
        [category, rf"${name}~/~{unit}$", f"${format_float_adaptive(x)}$"]
        for category, name, unit, x
        in zip(categories, variable_units.keys(), variable_units.values(), ind.x)
    ]

    # Metric values
    rows += [
        [
            "**Metrics**",
            r"$f(x)~/~\text{mol}^3~\text{m}_{\text{el}}^{-3}~\text{m}_{\text{s}}^{-3}~\text{s}^{-1}$",
            format_value_to_latex(f[0])
        ],
        [
            " ",
            r"$PU_i~/~\%$",
            format_value_to_latex(fractionator.purity*100)
        ],
        [
            " ",
            r"$PR_i~/~\text{mol}~\text{L}_{\text{s}}{-1}~\text{d}^{-1}$",
            format_value_to_latex(fractionator.productivity*3600*24/1000)
        ],
        [
            " ",
            r"$Y_i~/~\%$",
            format_value_to_latex(fractionator.recovery*100)
        ],
        [
            " ",
            r"$EC_i~/~\text{m}_{\text{el}}^3~\text{mol}^{-1}$",
            format_value_to_latex(fractionator.eluent_consumption)
        ],
    ]

    # Generate table
    table = tabulate(rows, headers=headers, floatfmt=".2f", tablefmt="github")

    caption = f"Optimization variables and KPIs of single objective {process_name}."
    name = f"{process_name.replace(" ", "_").lower()}_soo_kpi"
    formatted_table = embed_table_in_directive(table, caption, name)

    return formatted_table


def process_soo_results(
    soo_results: OptimizationResults,
    variable_units=None,
):
    """Process single objective optimization results."""
    soo_problem = soo_results.optimization_problem

    x = soo_results.x[0]
    simulation_results = simulate_results(soo_problem, x)

    fractionator = fractionate_results(soo_problem, simulation_results)
    fig, ax = fractionator.plot_fraction_signal()

    table = setup_soo_results_table(soo_results, fractionator, variable_units)

    return fig, ax, table


# %% Multi-objective results

def setup_moo_results_table(
    moo_results: OptimizationResults,
    fractionators: Optional[list[Fractionator]],
    variable_units: dict[str, str] = None,
) -> str:
    """Set up KPI table for multi-objective optimization results."""
    n_comp = fractionators[0].n_comp

    moo_problem = moo_results.optimization_problem
    process_name = moo_problem.evaluation_objects[0].name

    # Meta score
    f_meta = np.sum(moo_results.f_minimized, axis=0)
    f_meta_index = np.argmin(f_meta)  # TODO: Add to table (optional)

    # Default variable units
    if variable_units is None:
        variable_units = {rf"{var_name}": " " for var_name in moo_problem.variable_names}

    # Metric units (Note: Order matters here!)
    metric_units = {
        r"PU_i": r"\text{\%}",
        r"PR_i": r"\text{mol}~\text{L}_{\text{s}}^{-1}~\text{d}^{-1}",
        r"Y_i": r"\text{\%}",
        r"EC_i": r"\text{m}_{\text{el}}^3~\text{mol}^{-1}",
    }

    # Get best indices and values
    f_best_indices = moo_results.f_best_indices  # Indices of Pareto edge points
    best_individuals = [moo_results.meta_front.individuals[ind] for ind in f_best_indices]

    # Headers
    headers = [
        *[rf"${var_name}~/~{unit}$" for var_name, unit in variable_units.items()],
        *[rf"${metric_name}~/~{unit}$" for metric_name, unit in metric_units.items()],
    ]

    # Initialize rows
    rows = []

    # Add rows for each Pareto edge point
    for i_case, (ind, frac) in enumerate(zip(best_individuals, fractionators)):
        row = []

        # Add variables
        for x_i in ind.x:
            row.append(f"${format_value_to_latex(x_i)}$")

        # Add metrics
        values = []
        values.extend(frac.purity*100)
        values.extend(frac.productivity*3600*24/1000)
        values.extend(frac.recovery*100)
        values.extend(frac.eluent_consumption)

        formatted_values = []
        for i_v, value in enumerate(values):
            if i_v == i_case + n_comp:
                formatted_value = format_value_to_latex(value, bold=True)
            else:
                formatted_value = format_value_to_latex(value)

            formatted_values.append(formatted_value)

        for chunk in get_next_n_entries(formatted_values, n_comp):
            row.append(f"$[{', '.join(chunk)}]$")

        # Add f_meta TODO
        rows.append(row)

    # Generate table
    table = tabulate(
        rows,
        headers=headers,
        tablefmt="github",
    )

    # Caption and name
    caption = f"Pareto edge points for multi-objective {process_name} optimization."
    name = f"{process_name.replace(' ', '_').lower()}_moo_kpi"

    return embed_table_in_directive(table, caption, name)


def process_moo_results(
    moo_results: OptimizationResults,
    variable_units=None
) -> tuple[plt.Figure, npt.NDArray[plt.Axes], str]:
    """Process multi-objective optimization results."""
    moo_problem = moo_results.optimization_problem

    x_best = moo_results.x[moo_results.f_best_indices]
    simulation_results = [simulate_results(moo_problem, x) for x in x_best]

    fractionators = [
        fractionate_results(moo_problem, sim_results)
        for sim_results in simulation_results
    ]

    n_comp = fractionators[0].n_comp

    fig, axs = plotting.setup_figure(
        n_rows=int(len(x_best)/n_comp),
        n_cols=n_comp,
        scale_with_subplots=True,
    )
    for frac, ax in zip(fractionators, axs.ravel()):
        frac.plot_fraction_signal(fig=fig, ax=ax)

    table = setup_moo_results_table(moo_results, fractionators, variable_units)

    return fig, axs, table


# %% Combined wrapper

def create_figure_and_table(
    studies_root,
    study,
    case_name,
    variable_units,
    processing_kwargs = None,
    **embed_kwargs,
):
    study = setup_study(studies_root, study)
    cases = setup_cases(study, load=True)
    case = cases[case_name]

    if "single-objective" in case_name:
        processing_method = process_soo_results
        case_name = case_name.replace("single-objective", "soo")
    else:
        processing_method = process_moo_results
        case_name = case_name.replace("multi-objective", "moo")

    problem, _ = load_optimization_config(case)
    results = load_optimization_results(case)

    if variable_units is None:
        variable_units = {
            rf"{var_name}": r"\text{{}}"
            for var_name in problem.variable_names
        }

    *figures, table = processing_method(
        results,
        variable_units,
        **processing_kwargs or {},
    )

    default_embed_kwargs = {
        "caption": "Objective function values.",
    }
    default_embed_kwargs.update(embed_kwargs)
    fig_objectives_embedded = embed_figure_in_directive(
        case,
        "figures/objectives.png",
        f"{study.name}_{case_name}_objectives",
        **default_embed_kwargs,
    )

    return *figures, table, fig_objectives_embedded
