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
from operating_modes.process_optimization import ProcessOptimization
from operating_modes.main import (
    setup_optimization_problem_from_options, setup_optimizer
)


# %% Utils

# Metric units (order matters for iteration!)
metrics = {
    "productivity": {
        "symbol": r"PR",
        "unit": r"\text{mol}~\text{L}_{\text{s}}^{-1}~\text{d}^{-1}",
        "factor": 3600*24/1000,
    },
    "recovery": {
        "symbol": r"Y",
        "unit": r"\text{\%}",
        "factor": 100,
    },
    "eluent_consumption": {
        "symbol": r"EC",
        "unit": r"\text{m}_{\text{el}}^3~\text{mol}^{-1}",
        "factor": 1,
    },
    "meta": {
        "symbol": r"f(x)",
        "unit": r"\text{mol}^3~\text{m}_{\text{el}}^{-3}~\text{m}_{\text{s}}^{-3}~\text{d}^{-1}",
        "factor": 3600*24,
    },
    "purity": {
        "symbol": r"PU",
        "unit": r"\text{\%}",
        "factor": 100,
    },
}


def get_variables(
    operating_mode: Literal["batch-elution", "CLR", "flip-flop", "MRSSR", "serial-columns"],
    include_cycle_time: bool,
) -> dict[str, str]:
    variables = {}

    if include_cycle_time:
        variables["cycle_time"] = {
            "symbol": r"\Delta t_{\text{cycle}}",
            "unit": r"\text{min}",
            "factor": 1/60,
            "format_mm_ss": True,
        }

    match operating_mode:
        case "batch-elution":
            variables["feed_duration.time"] = {
                "symbol": r"\Delta t_{\text{feed}}",
                "unit": r"\text{min}",
                "factor": 1/60,
                "format_mm_ss": True,
            }
        case "CLR":
            variables["feed_duration.time"] = {
                "symbol": r"\Delta t_{\text{feed}}",
                "unit": r"\text{min}",
                "factor": 1/60,
                "format_mm_ss": True,
            }
            variables["recycle_off_output_state.time"] = {
                "symbol": r"t_{\text{recycle,off}}",
                "unit": r"\text{min}",
                "factor": 1/60,
                "format_mm_ss": True,
            }
        case "flip-flop":
            variables["feed_duration.time"] = {
                "symbol": r"\Delta t_{\text{feed}}",
                "unit": r"\text{min}",
                "factor": 1/60,
                "format_mm_ss": True,
            }
            variables["delay_flip.time"] = {
                "symbol": r"\Delta t_{\text{delay,flip}}",
                "unit": r"\text{min}",
                "factor": 1/60,
                "format_mm_ss": True,
            }
            variables["delay_injection.time"] = {
                "symbol": r"\Delta t_{\text{delay,inject}}",
                "unit": r"\text{min}",
                "factor": 1/60,
                "format_mm_ss": True,
            }
        case "MRSSR":
            variables["feed_duration.time"] = {
                "symbol": r"\Delta t_{\text{feed}}",
                "unit": r"\text{min}",
                "factor": 1/60,
                "format_mm_ss": True,
            }
            variables["recycle_on.time"] = {
                "symbol": r"t_{\text{recycle,on}}",
                "unit": r"\text{min}",
                "factor": 1/60,
                "format_mm_ss": True,
            }
            variables["recycle_off.time"] = {
                "symbol": r"t_{\text{recycle,off}}",
                "unit": r"\text{min}",
                "factor": 1/60,
                "format_mm_ss": True,
            }
            variables["recycle_duration"] = {
                "symbol": r"\Delta t_{\text{recycle}}",
                "unit": r"\text{min}",
                "factor": 1/60,
                "format_mm_ss": True,
            }
        case "serial-columns":
            variables["feed_duration.time"] = {
                "symbol": r"\Delta t_{\text{feed}}",
                "unit": r"\text{min}",
                "factor": 1/60,
                "format_mm_ss": True,
            }
            variables["serial_off.time"] = {
                "symbol": r"t_{\text{serial,off}}",
                "unit": r"\text{min}",
                "factor": 1/60,
                "format_mm_ss": True,
            }
            variables["serial_on.time"] = {
                "symbol": r"t_{\text{serial,on}}",
                "unit": r"\text{min}",
                "factor": 1/60,
                "format_mm_ss": True,
            }
            variables["serial_duration"] = {
                "symbol": r"\Delta t_{\text{serial}}",
                "unit": r"\text{min}",
                "factor": 1/60,
                "format_mm_ss": True,
            }
            variables["flow_sheet.column_1.length"] = {
                "symbol": r"L_{\text{c,1}}",
                "unit": r"\text{cm}",
                "factor": 100,
            }
            variables["flow_sheet.column_2.length"] = {
                "symbol": r"L_{\text{c,1}}",
                "unit": r"\text{cm}",
                "factor": 100,
            }
    return variables


def get_variable_dependencies(
    operating_mode: Literal["batch-elution", "CLR", "flip-flop", "MRSSR", "serial-columns"],
    include_cycle_time: bool,
) -> list[str]:
    """Setup variable dependencies."""
    match operating_mode:
        case "flip-flop":
            if include_cycle_time:
                return [
                    r"$\Delta t_{\text{cycle}} = "
                    r"2 (\Delta t_{\text{feed}} "
                    r"+ \Delta t_{\text{delay,flip}} "
                    r"+ \Delta t_{\text{delay,inject}})$",
                ]
        case "MRSSR":
            return [
                r"$t_{\text{recycle,off}} = t_{\text{recycle,on}} + \Delta t_{\text{recycle}}$",
            ]
        case "serial-columns":
            return [
                r"$t_{\text{serial,on}} = t_{\text{serial,off}} + \Delta t_{\text{serial}}$",
                r"$L_{\text{c,2}} = 0.6~\text{m} - L_{\text{c,1}}$",
            ]
    return []


def get_case_id(case: Case) -> str:
    id = case.name

    id = id.replace("standard_", "")
    id = id.replace("auto-cycle-time", "auto-cycle")
    id = id.replace("single-objective", "soo")
    id = id.replace("multi-objective", "moo")
    id = id.replace("per-component", "pc")

    return id


def get_title(case: Case) -> str:
    operating_mode = case.options.process_options.operating_mode
    separation_problem = case.options.process_options.separation_problem

    convert_to_linear = case.options.process_options.convert_to_linear
    apply_et_assumptions = case.options.process_options.apply_et_assumptions

    objective = case.options.optimization_options.objective

    return (
        f"{objective} optimization of the {operating_mode} process with a "
        f"{separation_problem} "
        f"{'linear' if convert_to_linear else 'Langmuir'} separation problem "
        f"{'applying ET assumptions' if apply_et_assumptions else ""}"
    )

# %% Setup cases

def get_cases_by_operating_mode(
    operating_mode: Literal["batch-elution", "CLR", "flip-flop", "MRSSR", "serial-columns"],
    index_by_name: bool = False,
    **kwargs: Any,
) -> list[Case]:
    """Return cases for a given operating mode."""
    cases = setup_cases(**kwargs)

    cases_by_mode = [
        case for case in cases
        if case.options.process_options.operating_mode == operating_mode
    ]

    if index_by_name:
        cases_by_mode = index_cases_by_name(cases_by_mode)

    return cases_by_mode


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
    checkpoint_path = results_path / "results" / "final.h5"
    optimization_results = OptimizationResults(optimization_problem, optimizer)
    optimization_results.load_results(checkpoint_path)
    return optimization_results


# %% Simulate and fractionate results

def simulate_results(
    optimization_problem: ProcessOptimization,
    x: list[float],
    determine_cycle_time: bool = True,
) -> SimulationResults:
    """Simulate individual of optimization results."""
    optimization_problem.set_variables(x)
    process = copy.deepcopy(optimization_problem.evaluation_objects[0])

    simulation_results = optimization_problem.process_simulator.evaluate(process)

    if determine_cycle_time and optimization_problem.cycle_time_determinator:
        simulation_results = optimization_problem.cycle_time_determinator.evaluate(
            simulation_results
        )

    return simulation_results


def fractionate_results(
    optimization_problem: ProcessOptimization,
    simulation_results: SimulationResults,
    comp_index: int | None = None,
) -> Fractionator:
    """Optimize fractionation."""
    frac_opt = optimization_problem.get_fractionator(comp_index)
    return frac_opt.evaluate(simulation_results)


def simulate_and_plot(
    optimization_problem: OptimizationProblem,
    x: list[float],
    comp_index: int | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Simulate and plot individual of optimization results."""
    simulation_results = simulate_results(optimization_problem, x)
    fractionator = fractionate_results(
        optimization_problem, simulation_results, comp_index
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

    # Convert infinities
    if "inf" in formatted:
        formatted = formatted.replace(r"inf", r"\infty")

    # Convert to LaTeX-style scientific notation
    if 'e' in formatted:
        base, exponent = formatted.split('e')
        base = base.rstrip('0').rstrip('.') if '.' in base else base
        formatted = rf"{base} \times 10^{{{int(exponent)}}}"

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
    return rf"\mathbf{{{formatted_val}}}" if bold else formatted_val


def format_mm_ss(seconds: float, as_text:bool = True) -> str:
    """Format a duration as mm:ss.

    Parameters
    ----------
    seconds : float
        Duration in seconds.
    as_text : bool
        If True,

    Returns
    -------
    str
        Duration formatted as mm:ss.
    """
    m, s = divmod(int(seconds), 60)
    mm_ss = f"{m:02d}:{s:02d}"

    if as_text:
        mm_ss = rf"\text{{{mm_ss}}}"

    return mm_ss


def convert_mm_ss_to_s(mm_ss: str) -> float:
    m, s = mm_ss.split(":")

    return float(m)*60 + float(s)


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
    results_dir = results_path / "results"
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


# %% Overview listing

def format_linear_constraint(opt_vars, lhs, b, is_equality=False):
    """
    Formats linear constraints from variables, coefficients, and a right-hand side value.

    Args:
        opt_vars (list): List of variable names.
        lhs (list): List of coefficients for the variables.
        b (float): Right-hand side value.
        is_equality (bool): If True, consider it as linear equality constraint

    Returns:
        str: Formatted linear constraint string
    """
    terms = []
    for coeff, var in zip(lhs, opt_vars):
        if coeff == 1:
            terms.append(f"{var}")
        elif coeff == -1:
            terms.append(f"- {var}")
        else:
            terms.append(rf"{coeff} {var}" if coeff > 0 else f"- {abs(coeff)} * {var}")

    # Join terms with " + " or " - " as appropriate
    constraint = " ".join(terms).replace("+ -", "- ").replace("  ", " ")

    if is_equality:
        return rf"{constraint} = {b}"
    else:
        return rf"{constraint} \le {b}"


def setup_overview(
    case: Case,
    include_operating_mode: bool = False,
    include_et_assumption: bool = False,
    caption: str | None = None,
) -> str:
    title = get_title(case)

    optimization_problem, optimizer = load_optimization_config(case)

    operating_mode = case.options.process_options.operating_mode
    separation_problem = case.options.process_options.separation_problem
    purity_required = case.options.optimization_options.fractionation_options.purity_required

    convert_to_linear = case.options.process_options.convert_to_linear
    apply_et_assumptions = case.options.process_options.apply_et_assumptions

    include_cycle_time = case.options.optimization_options.include_cycle_time

    objective = case.options.optimization_options.objective

    rows = []
    if include_operating_mode:
        rows.append(["**Operating mode**", f"{operating_mode}"])
    if include_et_assumption:
        rows.append(["**ET assumption**", f"{apply_et_assumptions}"])
    rows.append(["**Separation problem**", f"{separation_problem}"])
    rows.append(["**Binding model**", f"{'Linear' if convert_to_linear else 'Langmuir'}"])
    rows.append(["**Purity required**", rf"${purity_required*100}\%$"])

    # Variables
    var_info = get_variables(
        operating_mode,
        include_cycle_time,
    )
    for i, var in enumerate(optimization_problem.variables):
        symbol = var_info[var.name]["symbol"]
        lb = var.lb
        ub = var.ub
        if var_info[var.name].get("format_mm_ss"):
            if not np.isinf(lb):
                lb = rf"{format_mm_ss(lb)}"
            if not np.isinf(ub):
                ub = rf"{format_mm_ss(ub)}"
        else:
            lb = lb*var_info[var.name]["factor"]
            lb = f"{format_value_to_latex(lb)}"
            ub = ub*var_info[var.name]["factor"]
            ub = f"{format_value_to_latex(ub)}"

        unit = var_info[var.name]["unit"]

        # Variables
        if i == 0:
            prefix = "**Variables**"
        else:
            prefix = " "
        rows.append([prefix, rf"${symbol} \in [{lb},{ub}]~/~{unit}$"])

    # Linear constraints
    for i, lincon in enumerate(optimization_problem.linear_constraints):
        if i == 0:
            prefix = "**Linear constraints**"
        else:
            prefix = " "

        opt_vars = [var_info[var]["symbol"] for var in lincon["opt_vars"]]
        rows.append([prefix, f"${format_linear_constraint(opt_vars, lincon['lhs'], lincon['b'])}$"])

    # Linear equality constraints
    for i, lineqcon in enumerate(optimization_problem.linear_equality_constraints):
        if i == 0:
            prefix = "**Linear equality constraints**"
        else:
            prefix = " "

        opt_vars = [var_info[var]["symbol"] for var in lineqcon["opt_vars"]]
        rows.append([
            prefix,
            f"${format_linear_constraint(opt_vars, lineqcon['lhs'], lineqcon['beq'], True)}$"
        ])

    # Variable dependencies (might be tricky...)
    variable_dependencies = get_variable_dependencies(operating_mode, include_cycle_time)
    for i, var_dep in enumerate(variable_dependencies):
        if i == 0:
            prefix = "**Variable dependencies**"
        else:
            prefix = " "
        rows.append([prefix, f"{var_dep}"])

    # Objective
    rows.append(["**Objective**", f"{objective}"])

    if caption is None:
        caption = f"Overview of {title}."
    name = f"{get_case_id(case)}_overview"

    return embed_table_in_list_table_directive(
        rows,
        caption,
        name=name,
        header_rows=0,
        align="left",
    )



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
    title = get_title(case)

    operating_mode = case.options.process_options.operating_mode
    include_cycle_time = case.options.optimization_options.include_cycle_time

    f_meta = PerformanceProduct(ranking="equal")

    # Initialize rows
    rows = []

    # Headers
    # First row: Symbols
    row = []
    row += [
        *[rf"${var_info['symbol']}~/$" for var_info in variables.values()],
    ]
    if not include_cycle_time:
        variables_with_cycle_time = get_variables(
            operating_mode,
            include_cycle_time=True,
        )
        row.append(rf"${variables_with_cycle_time['cycle_time']['symbol']}^*~/$")
    row += [
        *[rf"${metric_info['symbol']}_i~/$" for metric_info in metrics.values()]
    ]
    rows.append(row)
    # Second row: units
    row = []
    row += [
        *[rf"${var_info['unit']}$" for var_info in variables.values()]
    ]
    if not include_cycle_time:
        row.append(rf"${variables_with_cycle_time['cycle_time']['unit']}$")
    row += [
        *[rf"${metric_info['unit']}$" for metric_info in metrics.values()]
    ]
    rows.append(row)

    # Data
    row = []

    # Add variables
    for i_x, var_info in enumerate(variables.values()):
        if var_info.get("format_mm_ss"):
            x_i = rf"${format_mm_ss(x[i_x])}$"
        else:
            x_i = x[i_x]*var_info["factor"]
            x_i = f"${format_value_to_latex(x_i)}$"
        row.append(x_i)

    if not include_cycle_time:
        if variables_with_cycle_time["cycle_time"].get("format_mm_ss"):
            x_i = rf"${format_mm_ss(frac.cycle_time)}$"
        else:
            x_i = frac.cycle_time*variables_with_cycle_time["cycle_time"]["factor"]
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
    table_caption = (f"Optimization variables and KPIs of {title}.")
    table_name = f"{get_case_id(case)}_kpi"

    return embed_table_in_list_table_directive(
        rows,
        table_caption,
        table_name,
        header_rows=2
    )


def process_soo_results(
    case: Case,
    load_kwargs: dict | None = None,
    return_results: bool = False,
) -> (
    tuple[tuple, tuple, str]
    |
    tuple[tuple, tuple, str, OptimizationResults, SimulationResults, Fractionator]
):
    """Process single-objective optimization results."""
    title = get_title(case)

    operating_mode = case.options.process_options.operating_mode

    optimization_results = load_optimization_results(
        case,
        load_kwargs,
    )
    optimization_problem = optimization_results.optimization_problem

    variables = get_variables(
        operating_mode,
        case.options.optimization_options.include_cycle_time,
    )

    # --- Objectives Figure ---
    fig_objectives, axs_objectives = optimization_results.plot_objectives(
        autoscale=False,
    )

    for i_var, (variable_name, variable_info) in enumerate(variables.items()):
        ax = axs_objectives[0, i_var]

        ax.set_xlabel(f"${variable_info['symbol']}~/~{variable_info['unit']}$")
        if variable_info.get("format_mm_ss"):
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(
                lambda x, _: rf"${format_mm_ss(x)}$"
            ))
        else:
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(
                lambda x, _: f"{x*variable_info['factor']:.0f}"
            ))

        ax.set_ylabel(f"${metrics['meta']['symbol']}~/~{metrics['meta']['unit']}$")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda y, _: f"{y*metrics['meta']['factor']:.0f}")
        )

    fig_objectives_caption = (f"Objective function values for {title}.")

    # --- Chromatograms ---
    x = optimization_results.x[0]
    simulation_results = simulate_results(optimization_problem, x)

    frac = fractionate_results(optimization_problem, simulation_results)
    fig_chrom, ax_chrom = frac.plot_fraction_signal()
    fig_chrom_caption = (f"Optimal chromatogram of {title}.")

    # --- Table ---
    table = setup_soo_results_table(
        case,
        x,
        frac,
        variables,
    )

    if not return_results:
        return (
            (fig_objectives, axs_objectives, fig_objectives_caption),
            (fig_chrom, ax_chrom, fig_chrom_caption),
            table,
        )

    return (
        (fig_objectives, axs_objectives, fig_objectives_caption),
        (fig_chrom, ax_chrom, fig_chrom_caption),
        table,
        optimization_results, simulation_results, frac
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
    include_cycle_time = case.options.optimization_options.include_cycle_time

    f_meta = PerformanceProduct(ranking="equal")

    # Initialize rows
    rows = []

    # Headers
    # First row: Symbols
    row = [" "]
    row += [
        *[rf"${var_info['symbol']}~/$" for var_info in variables.values()],
    ]
    if not include_cycle_time:
        variables_with_cycle_time = get_variables(
            operating_mode,
            include_cycle_time=True,
        )
        row.append(rf"${variables_with_cycle_time['cycle_time']['symbol']}^*~/$")
    row += [
        *[rf"${metric_info['symbol']}_i~/$" for metric_info in metrics.values()]
    ]
    rows.append(row)
    # Second row: units
    row = [" "]
    row += [
        *[rf"${var_info['unit']}$" for var_info in variables.values()]
    ]
    if not include_cycle_time:
        row.append(rf"${variables_with_cycle_time['cycle_time']['unit']}$")
    row += [
        *[rf"${metric_info['unit']}$" for metric_info in metrics.values()]
    ]
    rows.append(row)

    # Data
    for i_case, (x, frac) in enumerate(best_individuals):
        row = []

        # Add label
        row.append(f"({string.ascii_lowercase[i_case]})")

        # Add variables
        for i_x, var_info in enumerate(variables.values()):
            if var_info.get("format_mm_ss"):
                x_i = rf"${format_mm_ss(x[i_x])}$"
            else:
                x_i = x[i_x]*var_info["factor"]
                x_i = rf"${format_value_to_latex(x_i)}$"

            row.append(x_i)

        if not include_cycle_time:
            if variables_with_cycle_time["cycle_time"].get("format_mm_ss"):
                x_i = rf"${format_mm_ss(frac.cycle_time)}$"
            else:
                x_i = frac.cycle_time*variables_with_cycle_time["cycle_time"]["factor"]
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
    table_name = f"{get_case_id(case)}_kpi"

    return embed_table_in_list_table_directive(
        rows,
        table_caption,
        table_name,
        header_rows=2
    )


def process_moo_results(
    case: Case,
    load_kwargs: dict | None = None,
    use_population_all: bool = True,
    return_results: bool = False,
    set_gobal_limits: bool = True,
) -> (
    tuple[tuple, tuple, str]
    |
    tuple[tuple, tuple, str, OptimizationResults, list[SimulationResults], list[Fractionator]]
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

    variables = get_variables(
        operating_mode,
        case.options.optimization_options.include_cycle_time,
    )

    n_comp = optimization_problem.evaluation_objects[0].n_comp
    n_metrics = int(optimization_problem.n_objectives / n_comp)

    # --- Objectives Figure ---
    fig_objectives, axs_objectives = optimization_results.plot_objectives(
        autoscale=False,
    )

    for i_metric, (metric_name, metric_info) in enumerate(metrics.items()):
        if metric_name == "purity":
            break
        for i_comp in range(n_comp):
            for i_var, (variable_name, variable_info) in enumerate(variables.items()):
                ax = axs_objectives[n_comp*i_metric+i_comp, i_var]

                ax.set_xlabel(f"${variable_info['symbol']}~/~{variable_info['unit']}$")
                if variable_info.get("format_mm_ss"):
                    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
                        lambda x, _: rf"${format_mm_ss(x)}$"
                    ))
                else:
                    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
                        lambda x, _: f"{x*variable_info['factor']:.0f}"
                    ))

                ax.set_ylabel(f"${metric_info['symbol']}_{{{i_comp}}}~/~{metric_info['unit']}$")
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
            index = i_comp if objective == "multi-objective-per-component" else None
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
    frac_meta = fractionate_results(optimization_problem, sim_meta)
    frac_meta.plot_fraction_signal(ax=ax)
    label = f"({string.ascii_lowercase[counter]})"
    plotting.add_text(ax, label)

    for ax in axs_chrom[-1, 1:]:
        ax.axis('off')

    # Get global min/max for x and y
    if set_gobal_limits:
        x_min = min(ax.get_xlim()[0] for ax in axs_chrom.flatten())
        x_max = max(ax.get_xlim()[1] for ax in axs_chrom.flatten())
        y_min = min(ax.get_ylim()[0] for ax in axs_chrom.flatten())
        y_max = max(ax.get_ylim()[1] for ax in axs_chrom.flatten())

        for ax in axs_chrom.flatten():
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

    fig_chrom_caption = (
        f"Chromatograms of Pareto edge points of {objective} optimization of "
        f"{operating_mode} process with {separation_problem} separation problem."
    )

    # --- Table ---
    # Update args for table
    x_best = np.vstack((x_best, x_meta))
    simulation_results = simulation_results.ravel().tolist()
    simulation_results.append(sim_meta)
    fractionators = fractionators.ravel().tolist()
    fractionators.append(frac_meta)

    # Build table
    table = setup_moo_results_table(
        case,
        optimization_results,
        list(zip(x_best, fractionators)),
        variables,
    )

    if not return_results:
        return (
            (fig_objectives, axs_objectives, fig_objectives_caption),
            (fig_chrom, axs_chrom, fig_chrom_caption),
            table,
        )

    return (
        (fig_objectives, axs_objectives, fig_objectives_caption),
        (fig_chrom, axs_chrom, fig_chrom_caption),
        table,
        optimization_results,
        simulation_results,
        fractionators,
    )
