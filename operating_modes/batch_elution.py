from typing import Literal

from CADETProcess.processModel import (
    ChromatographicColumnBase,
    ComponentSystem,
    Langmuir,
)
from CADETProcess.modelBuilder import BatchElution

from operating_modes.model_parameters import(
    setup_column,
    c_feed,
    flow_rate,
)


def setup_process(
    column: ChromatographicColumnBase,
) -> BatchElution:
    """Setup batch-elution process."""
    return BatchElution(
        column,
        c_feed,
        flow_rate,
        feed_duration=60,
        cycle_time=600,
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
