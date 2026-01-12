from typing import Literal

from CADETProcess.processModel import (
    ChromatographicColumnBase,
    ComponentSystem,
    Langmuir,
)
from CADETProcess.modelBuilder import FlipFlop
import numpy as np

from operating_modes.model_parameters import(
    setup_column,
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
        "lb": 10, "ub": 100,
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


def setup_linear_constraints() -> list[dict]:
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
