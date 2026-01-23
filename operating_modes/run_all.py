from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import date
from itertools import product
import os
from pathlib import Path
from typing import Any, Literal

from cadetrdm import Case, Options, ProjectRepo


# %% Setup Process

@dataclass
class ProcessOptions:
    operating_mode: Literal["batch-elution", "clr", "flip-flop", "mrssr", "serial-columns"]
    separation_problem: Literal["standard", "difficult", "simple", "ternary"]
    apply_et_assumptions: bool = False


# %% Setup OptimizationProblem

@dataclass
class FractionationOptions:
    purity_required: float = 0.95
    ranking: Literal["equal"] | list[int] = "equal"
    allow_empty_fractions: bool = True
    ignore_failed: bool = False
    optimizer: Literal["COBYLA", "COBYQA"] = "COBYLA"


@dataclass
class CadetOptions:
    install_path: os.PathLike | None = None
    use_dll: bool = True


@dataclass
class OptimizationOptions:
    objective: Literal["single-objective", "multi-objective", "multi-objective-per-component"]
    cadet_options: CadetOptions
    fractionation_options: FractionationOptions
    transform_variables: Literal["auto", "linear", "log"] | None = None
    add_meta_score: bool = True
    _cache_directory_base: os.PathLike | None = None
    _temp_directory_base: os.PathLike | None = None


# %% Setup Optimizer

@dataclass
class OptimizerOptions:
    optimizer: Literal["U_NSGA3"] = "U_NSGA3"
    n_cores: int = -4
    n_max_gen: int = 64
    pop_size: int | None = None
    progress_frequency: int | None = None


# %% Setup options

def setup_options(
    study: ProjectRepo,
    operating_mode: Literal["batch-elution", "clr", "flip-flop", "mrssr", "serial-columns"],
    objective: Literal["single-objective", "multi-objective", "multi-objective-per-component"],
    separation_problem: Literal["standard", "difficult", "simple", "ternary"] | None = None,
    ranking: Literal["equal"] | list[int] = "equal",
    load: bool = False,
    push: bool = True,
    debug: bool = False,
    **kwargs,
) -> Case:
    """
    Set up a single case study with the given parameters.

    Args:
        operating_mode: The operating mode for the process.
        objective: The optimization objective.
        separation_problem: The binding model (defaults to mode-specific if None).
        ranking: The ranking for fractionation (default: "equal").
        load: If True, try loading previously run results.
        push: If True, push results after running the case.
        debug: If True, set debug mode for CADET-RDM.
        **kwargs: Additional arguments for other options.
    """
    # Default binding models per operating mode
    default_separation_problems = {
        "batch-elution": "standard",
        "clr": "standard",
        "flip-flop": "simple",
        "mrssr": "standard",
        "serial-columns": "ternary",
        "smb": "standard",
    }

    name = f"{operating_mode}"
    if separation_problem is not None:
        name = f"{name}_{separation_problem}"

    separation_problem = separation_problem or default_separation_problems[operating_mode]

    process_options = ProcessOptions(
        operating_mode=operating_mode,
        separation_problem=separation_problem,
        apply_et_assumptions=kwargs.get("apply_et_assumptions", False),
    )

    cadet_options = CadetOptions(
        install_path=kwargs.get("install_path"),
        use_dll=kwargs.get("use_dll"),
    )

    fractionation_opts = FractionationOptions(
        purity_required=kwargs.get("purity_required", 0.95),
        ranking=ranking,
        allow_empty_fractions=kwargs.get("allow_empty_fractions", True),
        ignore_failed=kwargs.get("ignore_failed", False),
        optimizer=kwargs.get("fractionation_optimizer", "COBYLA"),
    )

    optimization_options = OptimizationOptions(
        objective=objective,
        cadet_options=cadet_options,
        fractionation_options=fractionation_opts,
        add_meta_score=kwargs.get("add_meta_score", True),
        transform_variables=kwargs.get("transform_variables", "auto"),
        _cache_directory_base=kwargs.get("cache_directory_base", None),
        _temp_directory_base=kwargs.get("temp_directory_base", None),
    )

    optimizer_options = OptimizerOptions(
        optimizer=kwargs.get("optimizer", "U_NSGA3"),
        n_cores=kwargs.get("n_cores", -4),
        n_max_gen=kwargs.get("n_max_gen", 64),
        pop_size=kwargs.get("pop_size", None),
        progress_frequency=kwargs.get("progress_frequency", None),
    )

    options = Options({
        "process_options": asdict(process_options),
        "optimization_options": asdict(optimization_options),
        "optimizer_options": asdict(optimizer_options),
    })

    name = f"{name}_{objective}"
    if ranking != "equal":
        name = f"{name}_{ranking}"

    options.name = name
    options.commit_message = f"{name}_{str(date.today())}"
    options.branch_prefix = name.replace(" ", "_")
    options.debug = debug
    options.push = push

    case = Case(study, options=options, name=name)

    if load:
        case.load()

    return case


def iterate_cases(
    operating_modes: list[str],
    objectives: list[str],
    special_cases: list[dict] | None = None,
    work_dir: os.PathLike = "./",
    **kwargs,
) -> list[Case]:
    """
    Iterate over all combinations of operating modes and objectives.

    Args:
        operating_modes: List of operating modes.
        objectives: List of objectives.
        special_cases: List of dicts, each defining a special case.
            Example: [{
                "operating_mode": "clr",
                "objective": "multi-objective",
                "ranking": [1, 1, 0]
            }]
        work_dir: Path to store the resul
        **kwargs: Additional arguments for setup_case.

    Returns:
        List of Case objects.
    """
    # Set up individual study."""
    study = ProjectRepo(
        work_dir,
        url="git@github.com/schmoelder/diss_operating_modes.git",
        branch="main",
        package_dir="operating_modes"
    )

    cases = []
    special_cases = special_cases or []

    # Standard cases: all modes × all objectives
    for mode, objective in product(operating_modes, objectives):
        cases.append(setup_options(study, mode, objective, **kwargs))

    # Special cases: user-defined configurations
    for case_config in special_cases:
        cases.append(setup_options(study, **case_config, **kwargs))

    return cases


def setup_cases(
    **kwargs: Any,
) -> list[Case]:
    """
    Set up cases based on the current environment and parameters.

    Args:
        kwargs: Additional kwargs for iterate cases.
    Returns:
        List of cases.
    """
    username = os.getlogin()
    if username == 'jo':
        temp_directory_base = Path('/dev/shm/CADET-Process/tmp')
        cache_directory_base = Path('/dev/shm/CADET-Process/cache/')
        install_path = None
    elif username == 'schmoelder':
        temp_directory_base = Path('/dev/shm/schmoelder/CADET-Process/tmp')
        cache_directory_base = Path('/dev/shm/schmoelder/CADET-Process/cache/')
        install_path = None
        if os.uname().release[0] != '6':
            raise Exception(
                "Please ensure that all environments are consistent. "
                "All studies should be performed on IBT Servers running Linux 6.8"
            )
    else:
        raise Exception("Unknown environment.")

    operating_modes = [
        "batch-elution",
        "clr",
        "flip-flop",
        "mrssr",
        "serial-columns",
    ]
    objectives = [
        "single-objective",
        "multi-objective",
        "multi-objective-per-component",
    ]

    special_cases = [
        *[
            {
                "operating_mode": "batch-elution",
                "separation_problem": "ternary",
                "objective": objective,
            }
            for objective in objectives
        ],
        *[
            {
                "operating_mode": "batch-elution",
                "separation_problem": "ternary",
                "objective": objective,
                "ranking": [1, 1, 0],
            }
            for objective in objectives
        ],
        *[
            {
                "operating_mode": "serial-columns",
                "objective": objective,
                "ranking": [1, 1, 0],
            }
            for objective in objectives
        ],
    ]

    return iterate_cases(
        operating_modes,
        objectives,
        special_cases=special_cases,
        temp_directory_base=temp_directory_base,
        cache_directory_base=cache_directory_base,
        install_path=install_path,
        **kwargs,
    )


if __name__ == "__main__":
    cases = setup_cases(
        push=True,
        load=True,
        debug=False,
        fractionation_optimizer="COBYLA",
        ignore_failed=False,
        transform_variables="auto",
        add_meta_score=True,
    )

    run = True
    force = False

    if run:
        for case in cases:
            results = case.run_study(force=force)
