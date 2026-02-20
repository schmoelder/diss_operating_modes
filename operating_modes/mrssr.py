from typing import Literal

from CADETProcess import plotting
from CADETProcess.modelBuilder import MRSSR
from CADETProcess.processModel import ChromatographicColumnBase
from CADETProcess.simulationResults import SimulationResults
import matplotlib.pyplot as plt
import numpy.typing as npt

from operating_modes.model_parameters import(
    c_feed,
    flow_rate,
)


def setup_process(
    column: ChromatographicColumnBase,
) -> MRSSR:
    """Setup MR-SSR process."""
    return MRSSR(
        column,
        c_feed,
        flow_rate,
        feed_duration=60,
        recycle_on=6.6*60,
        recycle_off=8.0*60,
        cycle_time=600,
        V_tank=0.001,
        c_tank_init=0,
    )


def setup_variables(
    transform: Literal["auto", "linear", "log"] | None = None,
)-> list[dict]:
    """Setup optimization variables."""
    variables = []
    variables.append({
        "name": "cycle_time",
        "lb": 10, "ub": 600,
        "transform": transform,
    })
    variables.append({
        "name": "feed_duration.time",
        "lb": 10, "ub": 300,
        "transform": transform,
    })
    variables.append({
        "name": "recycle_on.time",
        "lb": 0, "ub": 600,
        "transform": transform,
    })
    variables.append({
        "name": "recycle_off.time",
    })
    variables.append({
        "name": "recycle_duration",
        "evaluation_objects": None,
        "lb": 0, "ub": 600,
        "transform": transform,
    })
    return variables


def setup_linear_constraints(n_comp: int | None) -> list[dict]:
    """Setup linear constraints."""
    linear_constraints = []
    # Ensure total injection is shorter than cycle time
    linear_constraints.append({
        "opt_vars": [
            "feed_duration.time",
            "recycle_duration",
            "cycle_time",
        ],
        "lhs": [1, 1, -1],
        "b": 0.0,
    })

    return linear_constraints


def setup_variable_dependencies() -> list[dict]:
    """Setup variable dependencies."""
    variable_dependencies = []
    variable_dependencies.append({
        "dependent_variable": "recycle_off.time",
        "independent_variables": ["recycle_on.time", "recycle_duration"],
        "transform": lambda *x: sum(x),
    })
    return variable_dependencies


def plot_results(
    simulation_results: SimulationResults,
) -> tuple[plt.Figure, npt.NDArray[plt.Axes]]:
    """Plot simulation results."""
    fig, axs = plotting.setup_figure(ncols=2, scale_with_subplots=True)
    simulation_results.solution.column.outlet.plot(ax=axs[0])
    simulation_results.solution.outlet.outlet.plot(ax=axs[1])

    return fig, axs


# %%

import matplotlib.pyplot as plt

def plot_overlay(
    simulation_results: SimulationResults,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot overlay of all cycles."""
    n_cycles = simulation_results.n_cycles
    solution = simulation_results.solution_cycles.column.outlet

    alpha_start = 0.25

    fig, ax = None, None
    update_layout = True
    for i_cyc, solution_cyc in enumerate(solution):
        if i_cyc == 1:
            update_layout = False

        alpha=(alpha_start + (1-alpha_start) * i_cyc / n_cycles)
        fig, ax = solution_cyc.plot(
            ax=ax,
            alpha=alpha,
            update_layout=update_layout
        )

    return fig, ax


def plot_last_cycle(
    simulation_results: SimulationResults,
) -> tuple[plt.Figure, npt.NDArray[plt.Axes]]:
    """Plot last cycle."""
    fig, axs = plotting.setup_figure(ncols=2, scale_with_subplots=True)
    simulation_results.solution_cycles.column.outlet[-1].plot(ax=axs[0])
    simulation_results.solution_cycles.outlet.outlet[-1].plot(ax=axs[1])

    return fig, axs
