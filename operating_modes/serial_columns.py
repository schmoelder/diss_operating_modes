from typing import Literal

from CADETProcess import plotting
from CADETProcess.modelBuilder import SerialColumns
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
    split_ratio=1/3,
    c_feed=c_feed,
    flow_rate=flow_rate,
    feed_duration=60,
    t_serial_off=6.6*60,
    t_serial_on=8.8*60,
    cycle_time=1000,
) -> SerialColumns:
    """Setup serial-columns process."""
    return SerialColumns(
        column,
        split_ratio=split_ratio,
        c_feed=c_feed,
        flow_rate=flow_rate,
        feed_duration=feed_duration,
        t_serial_off=t_serial_off,
        t_serial_on=t_serial_on,
        cycle_time=cycle_time,
    )


def setup_variables(
    include_cycle_time: bool = True,
    transform: Literal["auto", "linear", "log"] | None = None,
)-> list[dict]:
    """Setup optimization variables."""
    variables = []
    if include_cycle_time:
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
        "name": "serial_off.time",
        "lb": 0, "ub": 600,
        "transform": transform,
    })
    variables.append({
        "name": "serial_on.time",
    })
    variables.append({
        "name": "serial_duration",
        "evaluation_objects": None,
        "lb": 0, "ub": 600,
        "transform": transform,
    })
    variables.append({
        "name": "flow_sheet.column_1.length",
        "lb": 0.01, "ub": 0.6,
        "transform": transform,
    })
    variables.append({
        "name": "flow_sheet.column_2.length",
        "lb": 0.01, "ub": 0.6,
        "transform": transform,
    })
    return variables


def setup_linear_constraints(
    include_cycle_time: bool = True,
    n_comp: int | None = None,
) -> list[dict]:
    """Setup linear constraints."""
    linear_constraints = []
    if include_cycle_time:
        # Ensure injection is shorter than cycle time
        linear_constraints.append({
            "opt_vars": ["feed_duration.time", "cycle_time"],
            "lhs": [1 if not n_comp else n_comp, -1],
            "b": 0.0,
        })
        # Ensure serial off starts before end of cycle
        linear_constraints.append({
            "opt_vars": ["serial_off.time", "cycle_time"],
            "lhs": [1 if not n_comp else n_comp, -1],
            "b": 0.0,
        })
        # Ensure serial duration is shorter than cycle time
        linear_constraints.append({
            "opt_vars": ["serial_duration", "cycle_time"],
            "lhs": [1],
            "b": 0.0,
        })
    return linear_constraints


def setup_variable_dependencies(
    include_cycle_time: bool = True,
) -> list[dict]:
    """Setup variable dependencies."""
    variable_dependencies = []
    variable_dependencies.append({
        "dependent_variable": "flow_sheet.column_2.length",
        "independent_variables": ["flow_sheet.column_1.length"],
        "transform": lambda x: 0.6 - x,
    })
    variable_dependencies.append({
        "dependent_variable": "serial_on.time",
        "independent_variables": ["serial_off.time", "serial_duration"],
        "transform": lambda *x: sum(x),
    })
    return variable_dependencies


# %%

def plot_results(
    simulation_results: SimulationResults,
) -> tuple[plt.Figure, npt.NDArray[plt.Axes]]:
    """Plot simulation results."""
    fig, axs = plotting.setup_figure(
        ncols=3,
        layout="1.5_col",
        scale_with_subplots=True,
    )
    simulation_results.solution.column_1.outlet.plot(ax=axs[0])
    simulation_results.solution.outlet_1.inlet.plot(ax=axs[1])
    simulation_results.solution.outlet_2.inlet.plot(ax=axs[2])

    return fig, axs
