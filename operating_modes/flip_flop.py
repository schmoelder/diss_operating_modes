from typing import Literal

from CADETProcess.modelBuilder import FlipFlop
from CADETProcess.processModel import ChromatographicColumnBase
from CADETProcess.simulationResults import SimulationResults
from CADETProcess.solution import slice_solution
import matplotlib.pyplot as plt
import numpy as np

from operating_modes.model_parameters import(
    c_feed,
    flow_rate,
)


def setup_process(
    column: ChromatographicColumnBase,
) -> FlipFlop:
    """Setup batch-elution process."""
    return FlipFlop(
        column,
        c_feed,
        flow_rate,
        feed_duration=60,
        delay_flip=330,
        delay_injection=700,
    )


def setup_variables(
    transform: Literal["auto", "linear", "log"] | None = None,
)-> list[dict]:
    """Setup optimization variables."""
    variables = []
    variables.append({
        "name": "cycle_time",
        "lb": 10, "ub": 3000,
        "transform": transform,
    })
    variables.append({
        "name": "feed_duration.time",
        "lb": 10, "ub": 300,
        "transform": transform,
    })
    variables.append({
        "name": "delay_flip.time",
        "lb": 10, "ub": 1000,
        "transform": transform,
    })
    variables.append({
        "name": "delay_injection.time",
        "lb": 10, "ub": 1000,
        "transform": transform,
    })
    return variables


def setup_linear_constraints(n_comp: int | None) -> list[dict]:
    """Setup linear constraints."""
    linear_constraints = []
    return linear_constraints


def setup_variable_dependencies() -> list[dict]:
    """Setup variable dependencies."""
    variable_dependencies = []
    variable_dependencies.append({
        "dependent_variable": "cycle_time",
        "independent_variables": [
            "feed_duration.time", "delay_flip.time", "delay_injection.time"
        ],
        "transform": lambda x0, x1, x2: np.ceil(2*(x0 + x1 + x2)),
    })
    return variable_dependencies


def plot_results(
    simulation_results: SimulationResults,
    n_times: int  = 12
) -> [plt.Figure, list[plt.Axes]]:
    """Plot simulation results."""
    process = simulation_results.process

    def plot_at_time(ax, bulk_solution, time):
        z = bulk_solution.axial_coordinates
        bulk = bulk_solution.solution[round(time), ...]
        ax.plot(z, bulk)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim(-1, 11)

        if 0 <= time < process.backward_flow.time:
            ax.annotate(
                "", xytext=(0.44, 0.5), xy=(0.55, 0.5),
                xycoords="axes fraction",
                arrowprops=dict(facecolor='black', shrink=0.05)
            )
        elif process.backward_flow.time <= time < process.forward_flow.time:
            ax.annotate(
                "", xytext=(0.51, 0.5), xy=(0.41, 0.5),
                xycoords="axes fraction",
                arrowprops=dict(facecolor='black')
            )
        else:
            ax.annotate(
                "", xytext=(0.44, 0.5), xy=(0.55, 0.5),
                xycoords="axes fraction",
                arrowprops=dict(facecolor='black', shrink=0.05)
            )

    times = np.linspace(0, process.cycle_time, n_times)

    fig = plt.figure(figsize=(15, 0.75*n_times))
    gs = plt.GridSpec(n_times, 3, width_ratios=[1, 3, 1], wspace=0.1)

    # First column: Inlet profile
    ax_rotated = fig.add_subplot(gs[:, 0])
    solution = slice_solution(
        simulation_results.solution.column.inlet,
        coordinates={"time": [0, process.cycle_time]},
    )
    ax_rotated.plot(solution.solution, -solution.time)
    ax_rotated.set_xticks([])
    ax_rotated.set_yticks([])
    ax_rotated.set_xlim(-1, 11)

    for i, time in enumerate(times):
        ax = fig.add_subplot(gs[i, 1])
        plot_at_time(ax, simulation_results.solution.column.bulk, time)

    # Second column: Bulk profile over time
    for i, time in enumerate(times):
        ax = fig.add_subplot(gs[i, 1])
        plot_at_time(ax, simulation_results.solution.column.bulk, time)

    # Third column: Chromatogram
    ax_rotated = fig.add_subplot(gs[:, 2])
    solution = slice_solution(
        simulation_results.solution.column.outlet,
        coordinates={"time": [0, process.cycle_time]},
    )
    ax_rotated.plot(solution.solution, -solution.time)
    ax_rotated.set_xticks([])
    ax_rotated.set_yticks([])
    ax_rotated.set_xlim(-1, 11)

    fig.tight_layout()

    return fig, fig.axes
