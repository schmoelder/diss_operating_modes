from CADETProcess.processModel import (
    ComponentSystem,
    BindingBaseClass,
    Langmuir,
)
from CADETProcess.modelBuilder import SerialColumns

from operating_modes.model_parameters import(
    setup_column,
    c_feed,
    flow_rate,
)


def setup_process(
    column: BindingBaseClass,
) -> SerialColumns:
    """Setup serial-columns process."""
    return SerialColumns(
        column,
        split_ratio=1/3,
        c_feed=c_feed,
        flow_rate=flow_rate,
        feed_duration=60,
        t_serial_on=6.6*60,
        t_serial_off=8.0*60,
        cycle_time=600,
    )


def setup_variables() -> list[dict] | None:
    """Setup optimization variables."""
    variables = []
    variables.append({
        "name": "cycle_time",
        "lb": 10, "ub": 600,
        "transform": "auto"
    })
    variables.append({
        "name": "feed_duration.time",
        "lb": 10, "ub": 100,
        "transform": "auto"
    })
    variables.append({
        "name": "serial_off.time",
        "lb": 0, "ub": 600,
        "transform": "auto"
    })
    variables.append({
        "name": "serial_on.time",
        "lb": 0, "ub": 600,
        "transform": "auto"
    })
    variables.append({
        "name": "flow_sheet.column_1.length",
        "lb": 0.01, "ub": 0.6,
        "transform": "auto"
    })
    variables.append({
        "name": "flow_sheet.column_2.length",
        "lb": 0.01, "ub": 0.6,
        "transform": "auto"
    })
    return variables


def setup_linear_constraints() -> list[dict]:
    """Setup linear constraints."""
    linear_constraints = []
    # Ensure injection is shorter than cycle time
    linear_constraints.append({
        "opt_vars": ["feed_duration.time", "cycle_time"],
        "lhs": [1, -1],
        "b": 0.0,
    })
    # Ensure serial off is after serial on
    linear_constraints.append({
        "opt_vars": ["serial_off.time", "serial_on.time"],
        "lhs": [1, -1],
        "b": 0.0,
    })
    # Ensure serial on is before end of cycle
    linear_constraints.append({
        "opt_vars": ["serial_on.time", "cycle_time"],
        "lhs": [1, -1],
        "b": 0.0,
    })
    return linear_constraints


def setup_variable_dependencies() -> list[dict]:
    """Setup variable dependencies."""
    variable_dependencies = []
    variable_dependencies.append({
        "dependent_variable": "flow_sheet.column_2.length",
        "independent_variables": ["flow_sheet.column_1.length"],
        "transform": lambda x: 0.6 - x,
    })
    return variable_dependencies
