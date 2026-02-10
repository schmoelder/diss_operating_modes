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

from operating_modes.et_simulator import apply_et_assumptions_to_process
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
    convert_to_linear: bool = False,
) -> Process:
    """Set up and return a configured `Process` object."""
    binding_model = setup_binding_model(
        separation_problem,
        convert_to_linear=convert_to_linear,
    )
    column = setup_column(binding_model, convert_to_lrm=apply_et_assumptions)

    process = case_module.setup_process(column)

    if apply_et_assumptions:
        apply_et_assumptions_to_process(process)

    return process


# %% Setup OptimizationProblem

def setup_optimization_problem(
    case_module: ModuleType,
    process: Process,
    objective: str,
    cadet_options: dict,
    fractionation_options: dict,
    transform_variables: Literal["auto", "linear", "log"] | None,
    consider_n_comp_in_linear_constraints: bool,
    add_meta_score: bool,
    name: str,
    options_hash: str,
    _temp_directory_base: os.PathLike | None = None,
    _cache_directory_base: os.PathLike | None = None,
) -> ProcessOptimization:
    """Set up and return a configured `ProcessOptimization` object."""
    variables = case_module.setup_variables(
        transform=transform_variables
    )
    linear_constraints = case_module.setup_linear_constraints(
        process.n_comp if consider_n_comp_in_linear_constraints else None
    )
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
        add_meta_score=add_meta_score,
        cache_directory=cache_directory,
    )


def setup_optimization_problem_from_options(
    options: Options,
) -> ProcessOptimization:
    """Set up the optimization problem and optimizer from options."""
    options = copy.deepcopy(options)
    operating_mode = options["process_options"].pop("operating_mode")
    operating_mode = operating_mode.lower().replace("-", "_")
    case_module = importlib.import_module(f"operating_modes.{operating_mode}")

    process = setup_process(
        case_module,
        **options["process_options"]
    )

    optimization_problem = setup_optimization_problem(
        case_module,
        process,
        **options["optimization_options"],
        name=options["name"],
        options_hash=options.get_hash(),
    )

    return optimization_problem


# %% Run Optimization

@tracks_results
def main(
    repo: ProjectRepo,
    options: Options,
):
    options = copy.deepcopy(options)  # Safeguard against modification

    optimization_problem = setup_optimization_problem_from_options(options)

    optimizer = setup_optimizer(
        optimization_problem,
        options["optimizer_options"],
    )

    results = optimizer.optimize(
        optimization_problem,
        save_results=True,
        use_checkpoint=False,
        results_directory=repo.output_path / "results",
    )

    return results
