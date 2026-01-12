from typing import Literal

from CADETProcess.processModel import (
    ComponentSystem,
    BindingBaseClass,
    Langmuir,
)
from CADETProcess.modelBuilder import MRSSR

from operating_modes.model_parameters import(
    setup_column,
    c_feed,
    flow_rate,
)


def setup_process(
    column: BindingBaseClass,
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
        "lb": 0, "ub": 600,
        "transform": transform,
    })
    return variables


def setup_linear_constraints() -> list[dict]:
    """Setup linear constraints."""
    linear_constraints = []
    # Ensure recycling starts after injection (could be removed later)
    linear_constraints.append({
        "opt_vars":  ['feed_duration.time', 'recycle_on.time'],
        "lhs": [1, -1],
        "b": 0.0,
    })
    # Ensure recycle_off is after recycle_on
    linear_constraints.append({
        "opt_vars": ['recycle_on.time', 'recycle_off.time'],
        "lhs": [1, -1],
        "b": 0.0,
    })
    # Ensure recycling is shorter than cycle_time
    linear_constraints.append({
        "opt_vars": ['recycle_off.time', 'cycle_time'],
        "lhs": [1, -1],
        "b": 0.0,
    })
    # Ensure recycling is shorter than injection
    linear_constraints.append({
        "opt_vars": ['feed_duration.time', 'recycle_off.time', 'recycle_on.time'],
        "lhs": [-1, 1, -1],
        "b": 0.0,
    })
    # Ensure injection is shorter than cycle time
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
