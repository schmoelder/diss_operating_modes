from typing import Literal

from CADETProcess import plotting
from CADETProcess.modelBuilder import CLR
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
) -> CLR:
    """Setup CLR process."""
    return CLR(
        column,
        c_feed,
        flow_rate,
        feed_duration=60,
        recycle_off=900,
        cycle_time=1800,
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
        "name": "recycle_off_output_state.time",
        "lb": 10, "ub": 3000,
        "transform": transform,
    })
    return variables


def setup_linear_constraints() -> list[dict]:
    """Setup linear constraints."""
    linear_constraints = []
    linear_constraints.append({
        "opt_vars": ["feed_duration.time", "cycle_time"],
        "lhs": [1, -1],
        "b": 0.0,
    })
    return linear_constraints


def setup_variable_dependencies() -> list[dict]:
    """Setup variable dependencies."""
    variable_dependencies = []
    return variable_dependencies


def plot_results(
    simulation_results: SimulationResults,
) -> tuple[plt.Figure, npt.NDArray[plt.Axes]]:
    """Plot simulation results."""
    fig, axs = plotting.setup_figure(ncols=2, scale_with_subplots=True)
    simulation_results.solution.column.outlet.plot(ax=axs[0])
    simulation_results.solution.outlet.outlet.plot(ax=axs[1])

    return fig, axs
