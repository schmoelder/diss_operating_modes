import copy
import importlib
import os
from pathlib import Path
import sys; sys.path.insert(0, "../")
from typing import Literal
from types import ModuleType

from cadetrdm import Options, ProjectRepo, tracks_results
from CADETProcess import settings
from CADETProcess.processModel import Process

from operating_modes.model_parameters import (
    setup_binding_model,
    setup_column,
)
from operating_modes.process_optimization import (
    setup_optimizer,
    ProcessOptimization,
)


# %% Setup Process

def setup_process(
    case_module: ModuleType,
    separation_problem: Literal["standard", "difficult", "simple", "ternary"],
    apply_et_assumptions: bool = False,
) -> Process:
    """Set up and return a configured `Process` object."""
    binding_model = setup_binding_model(separation_problem)
    column = setup_column(binding_model, apply_et_assumptions)

    process = case_module.setup_process(column)

    return process



# %% Setup OptimizationProblem

def setup_optimization_problem(
    case_module: ModuleType,
    process: Process,
    objective: str,
    cadet_options: dict,
    fractionation_options: dict,
    name: str,
    options_hash: str = "",
    _temp_directory_base: os.PathLike | None = None,
    _cache_directory_base: os.PathLike | None = None,
) -> ProcessOptimization:
    """Set up and return a configured `ProcessOptimization` object."""
    variables = case_module.setup_variables()
    linear_constraints = case_module.setup_linear_constraints()
    variable_dependencies = case_module.setup_variable_dependencies()

    # Handle directories
    if _temp_directory_base is not None:
        settings.temp_dir = Path(_temp_directory_base) / name / options_hash

    cache_directory = (
        None if _cache_directory_base is None
        else Path(_cache_directory_base) / name / options_hash
    )

    # Setup and return the optimization problem
    return ProcessOptimization(
        name=name,
        process=process,
        variables=variables,
        linear_constraints=linear_constraints,
        variable_dependencies=variable_dependencies,
        objective=objective,
        cadet_options=cadet_options,
        fractionation_options=fractionation_options,
        cache_directory=cache_directory,
    )


# %% Run Optimization

@tracks_results
def main(
    repo: ProjectRepo,
    options: Options,
):
    options_hash = options.get_hash()
    options = copy.deepcopy(options)  # Saveguard agains modification
    name = options.name

    operating_mode = options["process_options"].pop("operating_mode").replace("-", "_")
    case_module = importlib.import_module(f"operating_modes.{operating_mode}")

    process = setup_process(
        case_module,
        **options["process_options"]
    )

    optimization_problem = setup_optimization_problem(
        case_module,
        process,
        **options["optimization_options"],
        name=name,
        options_hash=options_hash,
    )
    optimizer = setup_optimizer(
        optimization_problem,
        options["optimizer_options"],
    )

    results = optimizer.optimize(
        optimization_problem,
        save_results=True,
        use_checkpoint=False,
        results_directory=repo.output_path / name,
    )

    return results
