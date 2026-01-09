from dataclasses import dataclass, asdict
from datetime import date
from itertools import product
import os
from pathlib import Path
import sys; sys.path.insert(0, "../")
from typing import Literal

from cadetrdm import Case, Options


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


@dataclass
class CadetOptions:
    install_path: os.PathLike | None = None
    use_dll: bool = True


@dataclass
class OptimizationOptions:
    objective: Literal["single-objective", "multi-objective", "multi-objective-per-component"]
    fractionation_options: FractionationOptions
    cadet_options: CadetOptions
    _cache_directory_base: os.PathLike | None = None
    _temp_directory_base: os.PathLike | None = None


# %% Setup Optimizer

@dataclass
class OptimizerOptions:
    optimizer: Literal["U_NSGA3"] = "U_NSGA3"
    n_cores: int = -4
    n_max_gen: int = 64
    pop_size: int | None = None


# %% Setup options

def setup_options(
    operating_mode: Literal["batch-elution", "clr", "flip-flop", "mrssr", "serial-columns"],
    objective: Literal["single-objective", "multi-objective", "multi-objective-per-component"],
    separation_problem: Literal["standard", "difficult", "simple", "ternary"] | None = None,
    ranking: Literal["equal"] | list[int] = "equal",
    debug: bool = False,
    push: bool = True,
    load: bool = False,
    **kwargs,
) -> Case:
    """
    Set up a single case study with the given parameters.

    Args:
        operating_mode: The operating mode for the process.
        objective: The optimization objective.
        separation_problem: The binding model (defaults to mode-specific if None).
        ranking: The ranking for fractionation (default: "equal").
        debug: If True, set debug mode for CADET-RDM
        push: If True, push results after running the case.
        load: If True, try loading previously run results.
        **kwargs: Additional arguments for ProcessParameters or OptimizationOptions.
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

    fractionation_opts = FractionationOptions(
        purity_required=kwargs.get("purity_required", 0.95),
        ranking=ranking,
        allow_empty_fractions=kwargs.get("allow_empty_fractions", True),
        ignore_failed=kwargs.get("ignore_failed", False),
    )

    cadet_options = CadetOptions(
        install_path=kwargs.get("install_path"),
        use_dll=kwargs.get("use_dll"),
    )

    optimization_options = OptimizationOptions(
        objective=objective,
        fractionation_options=fractionation_opts,
        cadet_options=cadet_options,
        _cache_directory_base=kwargs.get("cache_directory_base", None),
        _temp_directory_base=kwargs.get("temp_directory_base", None),
    )

    optimizer_options = OptimizerOptions(
        optimizer=kwargs.get("optimizer", "U_NSGA3"),
        n_cores=kwargs.get("n_cores", -4),
        n_max_gen=kwargs.get("n_max_gen", 64),
        pop_size=kwargs.get("n_max_gen", None),
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
    options.debug = debug
    options.push = push

    case = Case(options=options, name=name)

    if load:
        case.load()

    return case


def iterate_cases(
    operating_modes: list[str],
    objectives: list[str],
    special_cases: list[dict] = None,
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
        **kwargs: Additional arguments for setup_case.

    Returns:
        List of Case objects.
    """
    cases = []
    special_cases = special_cases or []

    # Standard cases: all modes × all objectives
    for mode, objective in product(operating_modes, objectives):
        cases.append(setup_options(mode, objective, **kwargs))

    # Special cases: user-defined configurations
    for case_config in special_cases:
        cases.append(setup_options(**case_config, **kwargs))

    return cases


if __name__ == "__main__":

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

    cases = iterate_cases(
        operating_modes,
        objectives,
        special_cases=special_cases,
        temp_directory_base=temp_directory_base,
        cache_directory_base=cache_directory_base,
        install_path=install_path,
        load=True,
        debug=False,
        push=True,
    )

    run = True
    force = False

    if run:
        for case in cases:
            results = case.run_study(force=force)
