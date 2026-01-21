import copy
import math
import os
from pathlib import Path
import string
from typing import Any, Literal, Optional

from CADETProcess.fractionation import Fractionator
from CADETProcess.optimization import (
    OptimizationProblem,
    OptimizerBase,
    OptimizationResults,
)
from CADETProcess.simulationResults import SimulationResults
from CADETProcess import plotting
from CADETProcess.performance import PerformanceProduct
from cadetrdm import Case
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import sys; sys.path.insert(0, "../")
from operating_modes.run_all import setup_cases
from operating_modes.main import setup_optimization_problem_from_options, setup_optimizer


# %% Utils

# Metric units (order matters for iteration!)
metrics = {
    "productivity": {
        "symbol": r"PR_i",
        "unit": r"\text{mol}~\text{L}_{\text{s}}^{-1}~\text{d}^{-1}",
        "factor": 3600*24/1000,
    },
    "recovery": {
        "symbol": r"Y_i",
        "unit": r"\text{\%}",
        "factor": 100,
    },
    "eluent_consumption": {
        "symbol": r"EC_i",
        "unit": r"\text{m}_{\text{el}}^3~\text{mol}^{-1}",
        "factor": 1,
    },
    "meta": {
        "symbol": r"f(x)",
        "unit": r"\text{mol}^3~\text{m}_{\text{el}}^{-3}~\text{m}_{\text{s}}^{-3}~\text{s}^{-1}",
        "factor": 1,
    },
    "purity": {
        "symbol": r"PU_i",
        "unit": r"\text{\%}",
        "factor": 100,
    },
}

# %% Setup cases

def get_cases_by_operating_mode(
    operating_mode: Literal["batch-elution", "clr", "flip-flop", "mrssr", "serial-columns"],
    **kwargs: Any,
) -> list[Case]:
    """Return cases for a given operating mode."""
    cases = setup_cases(**kwargs)

    return [
        case for case in cases
        if case.options.process_options.operating_mode == operating_mode
    ]


def index_cases_by_name(cases: list[Case]) -> dict[str, Case]:
    """Return dict with cases indexed by name."""
    return {case.name: case for case in cases}


# %% Load results

def load_optimization_config(
    case: Case
) -> tuple[OptimizationProblem, OptimizerBase]:
    """Set up optimization problem and optimizer."""
    options = case.options
    optimization_problem = setup_optimization_problem_from_options(options)
    optimizer = setup_optimizer(
        optimization_problem,
        options["optimizer_options"],
    )
    return optimization_problem, optimizer


def load_optimization_results(
    case: Case,
    load_kwargs: Any = None,
) -> OptimizationResults:
    """Load optimization results for a given case."""
    results_path = case.load(**load_kwargs or {})
    if not results_path:
        raise FileNotFoundError("Could not find matching results.")
    optimization_problem, optimizer = load_optimization_config(case)
    checkpoint_path = results_path / case.name / "final.h5"
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
    evaluator_index: int = 1,
) -> Fractionator:
    """Optimize fractionation."""
    frac_opt = optimization_problem.evaluators[evaluator_index]
    return frac_opt.evaluate(simulation_results)


def simulate_and_plot(
    optimization_problem: OptimizationProblem,
    x: list[float],
    fractionator_index: int = 1,
) -> tuple[plt.Figure, plt.Axes]:
    """Simulate and plot individual of optimization results."""
    simulation_results = simulate_results(optimization_problem, x)
    fractionator = fractionate_results(
        optimization_problem, simulation_results, fractionator_index
    )

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


def format_mm_ss(seconds):
    """Format a duration as mm:ss.

    Parameters
    ----------
    seconds : int
        Duration in seconds.

    Returns
    -------
    str
        Duration formatted as mm:ss.
    """
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


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
    formatted_table += f":align: {align}\n"
    formatted_table += "\n"
    formatted_table += f"{table}\n"
    formatted_table += "```"

    return formatted_table


def embed_table_in_list_table_directive(
    data: str,
    caption: str | None = None,
    name: str | None = None,
    align: str = "center",
    header_rows: int = 1,
) -> str:
    """Format table to embed it in MyST list-table directive."""
    formatted_table = "```{list-table}"
    if caption:
        formatted_table += f" {caption}"

    formatted_table += "\n"
    if name:
        formatted_table += f":name: {name}\n"
    formatted_table += f":header-rows: {header_rows}\n"
    formatted_table += f":align: {align}\n"
    formatted_table += "\n"

    for row in data:
        # First item in row gets the '*' bullet, others get '-'
        formatted_table += f"* - {row[0]}\n"
        for item in row[1:]:
            formatted_table += f"  - {item}\n"

    formatted_table += "```"

    return formatted_table


def embed_figure_in_directive(
    case: Case,
    figure_path: os.PathLike,
    name: None,
    caption: str,
    load_kwargs: dict | None = None,
    scale: Optional[int] = 100,
    width: Optional[int] = None,
) -> str:
    """Format figure to embed it in MyST figure directive."""
    results_path = case.load(**load_kwargs or {})
    results_dir = results_path / case.name
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


# %% Single objective

def setup_soo_results_table(
    case: Case,
    x: list[float],
    frac: Fractionator,
    variables: dict,
) -> str:
    """
    Set up KPI table for single-objective optimization results.

    Parameters
    ----------
    case : Case
        The case study.
    x : list[float]
        Best individual.
    frac : Fractionator
        Fractionator corresponding to best individual.
    variables : dict
        Dictionary mapping variable names to LaTeX-formatted units.

    Returns
    -------
    str
        Markdown table as a string, with LaTeX-formatted values and units.
    """
    operating_mode = case.options.process_options.operating_mode
    separation_problem = case.options.process_options.separation_problem
    objective = case.options.optimization_options.objective

    process_name = (
        operating_mode if separation_problem == "standard"
        else f"{operating_mode}_{separation_problem}"
    )
    objective_short = objective.replace("single-objective", "soo")

    f_meta = PerformanceProduct(ranking="equal")

    # Initialize rows
    rows = []

    # Headers
    rows.append([
        *[rf"${var_info['symbol']}~/$" for var_info in variables.values()],
        *[rf"${metric_info['symbol']}~/$" for metric_info in metrics.values()],
    ])
    rows.append([
        *[rf"${var_info['unit']}$" for var_info in variables.values()],
        *[rf"${metric_info['unit']}$" for metric_info in metrics.values()],
    ])

    # Data
    row = []

    # Add variables
    for i_x, var_info in enumerate(variables.values()):
        if var_info.get("format_mm_ss"):
            x_i = f"${format_mm_ss(x[i_x])}$"
        else:
            x_i = x[i_x]*var_info["factor"]
            x_i = f"${format_value_to_latex(x_i)}$"

        row.append(x_i)

    # Add metrics with bold diagonal
    for i_v, (metric_name, metric_info) in enumerate(metrics.items()):
        if metric_name == "meta":
            metric_values = f_meta(frac.performance)
        else:
            metric_values = getattr(frac, metric_name)
        scaled_values = metric_values * metric_info["factor"]
        formatted_values = []
        for value in scaled_values:
            formatted_values.append(format_value_to_latex(value))

        formatted_values = (
            f"${formatted_values[0]}$" if len(formatted_values) == 1
            else f"$[{', '.join(formatted_values)}]$"
        )
        row.append(formatted_values)

    rows.append(row)

    # Caption and name
    table_caption = (
        f"Optimization variables and KPIs of {objective} optimization of "
        f"{operating_mode} process with a {separation_problem} component system. "
    )
    table_name = f"{process_name}_{objective_short}_kpi"

    return embed_table_in_list_table_directive(
        rows,
        table_caption,
        table_name,
        header_rows=2
    )


def process_soo_results(
    case: Case,
    variables: dict | None = None,
    load_kwargs: dict | None = None,
):
    """Process multi-objective optimization results."""
    operating_mode = case.options.process_options.operating_mode
    separation_problem = case.options.process_options.separation_problem
    objective = case.options.optimization_options.objective

    optimization_results = load_optimization_results(
        case,
        load_kwargs,
    )
    optimization_problem = optimization_results.optimization_problem

    # --- Objectives Figure ---
    fig_objectives, axs_objectives = optimization_results.plot_objectives()

    for i_var, (variable_name, variable_info) in enumerate(variables.items()):
        ax = axs_objectives[0, i_var]

        ax.set_xlabel(f"${variable_info['symbol']}~/~{variable_info['unit']}$")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{x*variable_info['factor']:.0f}")
        )

        ax.set_ylabel(f"${metrics['meta']['symbol']}~/~{metrics['meta']['unit']}$")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda y, _: f"{y*metrics['meta']['factor']:.0f}")
        )

    fig_objectives_caption = (
        f"Objective function values for {objective} optimization of "
        f"{operating_mode} process with {separation_problem} component system."
    )

    # --- Chromatograms ---
    x = optimization_results.x[0]
    simulation_results = simulate_results(optimization_problem, x)

    frac = fractionate_results(optimization_problem, simulation_results)
    fig_chrom, ax_chrom = frac.plot_fraction_signal()
    fig_chrom_caption = (
        f"Optimal chromatogram of {objective} optimization of "
        f"{operating_mode} process with {separation_problem} component system."
    )

    # --- Table ---
    table = setup_soo_results_table(
        case,
        x,
        frac,
        variables,
    )

    return (
        (fig_objectives, axs_objectives, fig_objectives_caption),
        (fig_chrom, ax_chrom, fig_chrom_caption),
        table,
    )


# %% Multi-objective

def setup_moo_results_table(
    case: Case,
    moo_problem: OptimizationProblem,
    best_individuals: list[tuple[list, Fractionator]],
    variables: dict,
) -> str:
    """
    Set up KPI table for multi-objective-per-component optimization results.

    Parameters
    ----------
    case : Case
        The case study.
    moo_problem : OptimizationProblem
        Multi-objective problem.
    best_individuals : list[tuple[list, Fractionator]]
        List of tuples containing best individuals and corresponding fractionators.
    variables : dict
        Dictionary mapping variable names to LaTeX-formatted units.

    Returns
    -------
    str
        Markdown table as a string, with LaTeX-formatted values and units.
    """
    operating_mode = case.options.process_options.operating_mode
    separation_problem = case.options.process_options.separation_problem
    objective = case.options.optimization_options.objective

    process_name = (
        operating_mode if separation_problem == "standard"
        else f"{operating_mode}_{separation_problem}"
    )
    objective_short = objective.replace("multi-objective", "moo")
    objective_short = objective_short.replace("per-component", "pc")

    f_meta = PerformanceProduct(ranking="equal")

    # Initialize rows
    rows = []

    # Headers
    rows.append([
        " ",
        *[rf"${var_info['symbol']}~/$" for var_info in variables.values()],
        *[rf"${metric_info['symbol']}~/$" for metric_info in metrics.values()],
    ])
    rows.append([
        " ",
        *[rf"${var_info['unit']}$" for var_info in variables.values()],
        *[rf"${metric_info['unit']}$" for metric_info in metrics.values()],
    ])

    # Data
    for i_case, (x, frac) in enumerate(best_individuals):
        row = []

        # Add label
        row.append(f"({string.ascii_lowercase[i_case]})")

        # Add variables
        for i_x, var_info in enumerate(variables.values()):
            if var_info.get("format_mm_ss"):
                x_i = f"${format_mm_ss(x[i_x])}$"
            else:
                x_i = x[i_x]*var_info["factor"]
                x_i = f"${format_value_to_latex(x_i)}$"

            row.append(x_i)

        # Add metrics with bold diagonal
        counter = 0
        for i_v, (metric_name, metric_info) in enumerate(metrics.items()):
            if metric_name == "meta":
                metric_values = f_meta(frac.performance)
            else:
                metric_values = getattr(frac, metric_name)
            scaled_values = metric_values * metric_info["factor"]
            formatted_values = []
            for value in scaled_values:
                bold = counter == i_case
                formatted_values.append(format_value_to_latex(value, bold=bold))
                counter += 1
            formatted_values = (
                f"${formatted_values[0]}$" if len(formatted_values) == 1
                else f"$[{', '.join(formatted_values)}]$"
            )
            row.append(formatted_values)

        rows.append(row)

    # Caption and name
    table_caption = (
        f"Optimization variables and KPIs for Pareto edge points of {objective} "
        f"optimization of the {operating_mode} process with a {separation_problem} "
        "component system. Each row corresponds to a non-dominated solution "
        f"that is extreme with respect to one objective (highlighted in bold)."
    )
    table_name = f"{process_name}_{objective_short}_kpi"

    return embed_table_in_list_table_directive(
        rows,
        table_caption,
        table_name,
        header_rows=2
    )


def process_moo_results(
    case: Case,
    variables: dict | None = None,
    load_kwargs: dict | None = None,
    use_population_all: bool = True,
):
    """Process multi-objective optimization results."""
    operating_mode = case.options.process_options.operating_mode
    separation_problem = case.options.process_options.separation_problem
    objective = case.options.optimization_options.objective

    optimization_results = load_optimization_results(
        case,
        load_kwargs,
    )
    optimization_problem = optimization_results.optimization_problem

    n_comp = optimization_problem.evaluation_objects[0].n_comp
    n_metrics = int(optimization_problem.n_objectives / n_comp)

    # --- Objectives Figure ---
    fig_objectives, axs_objectives = optimization_results.plot_objectives()

    for i_metric, (metric_name, metric_info) in enumerate(metrics.items()):
        if metric_name == "purity":
            break
        for i_comp in range(n_comp):
            for i_var, (variable_name, variable_info) in enumerate(variables.items()):
                ax = axs_objectives[n_comp*i_metric+i_comp, i_var]

                ax.set_xlabel(f"${variable_info['symbol']}~/~{variable_info['unit']}$")
                ax.xaxis.set_major_formatter(ticker.FuncFormatter(
                    lambda x, _: f"{x*variable_info['factor']:.0f}")
                )

                ax.set_ylabel(f"${metric_info['symbol']}~/~{metric_info['unit']}$")
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(
                    lambda y, _: f"{y*metric_info['factor']:.0f}")
                )
            if metric_name == "meta":
                break

    fig_objectives_caption = (
        f"Objective function values for {objective} optimization of "
        f"{operating_mode} process with {separation_problem} component system."
    )

    # --- Chromatograms ---
    if use_population_all:
        population = optimization_results.population_all
    else:
        population = optimization_results.meta_front

    # Build x_best upfront
    x_best = population.x[population.f_best_indices]
    simulation_results = [
        simulate_results(optimization_problem, x) for x in x_best
    ]

    # Simulate and Fractionate
    simulation_results = np.array(simulation_results).reshape(n_metrics, n_comp)

    fractionators = np.zeros_like(simulation_results)
    for i_metric in range(n_metrics):
        for i_comp in range(n_comp):
            sim_results = simulation_results[i_metric, i_comp]
            index = i_comp+1 if objective == "multi-objective-per-component" else 1
            fractionators[i_metric, i_comp] = fractionate_results(
                optimization_problem, sim_results, index
            )

    # Plot
    fig_chrom, axs_chrom = plotting.setup_figure(
        nrows=n_metrics+1,
        ncols=n_comp,
        scale_with_subplots=True,
    )

    counter = 0
    for i_metric in range(n_metrics):
        for i_comp in range(n_comp):
            frac = fractionators[i_metric, i_comp]
            ax = axs_chrom[i_metric][i_comp]
            frac.plot_fraction_signal(ax=ax)

            label = f"({string.ascii_lowercase[counter]})"
            plotting.add_text(ax, label)
            counter += 1

    # Include meta score (performance product)
    f_meta_index = population.m_best_indices[0]
    x_meta = population.x[f_meta_index]
    ax = axs_chrom[-1, 0]

    sim_meta = simulate_results(optimization_problem, x_meta)
    frac_meta = fractionate_results(optimization_problem, sim_meta, -1)
    frac_meta.plot_fraction_signal(ax=ax)
    label = f"({string.ascii_lowercase[counter]})"
    plotting.add_text(ax, label)

    for ax in axs_chrom[-1, 1:]:
        ax.axis('off')

    fig_chrom_caption = (
        f"Chromatograms of Pareto edge points of {objective} optimization of "
        f"{operating_mode} process with {separation_problem} component system."
    )

    # --- Table ---
    # Update args for table
    x_best = np.vstack((x_best, x_meta))
    fractionators = fractionators.ravel().tolist()
    fractionators.append(frac_meta)

    # Build table
    table = setup_moo_results_table(
        case,
        optimization_results,
        list(zip(x_best, fractionators)),
        variables,
    )

    return (
        (fig_objectives, axs_objectives, fig_objectives_caption),
        (fig_chrom, axs_chrom, fig_chrom_caption),
        table,
    )
