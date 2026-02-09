from dataclasses import dataclass, asdict
from datetime import date
import os
from pathlib import Path
from typing import Any, Literal

from cadetrdm import Case, Options, ProjectRepo


# %% Setup Process

@dataclass
class ProcessOptions:
    operating_mode: Literal["batch-elution", "CLR", "flip-flop", "MRSSR", "serial-columns"]
    separation_problem: Literal["standard", "difficult", "simple", "ternary"]
    convert_to_linear: bool = False
    apply_et_assumptions: bool = False


# %% Setup OptimizationProblem

@dataclass
class FractionationOptions:
    purity_required: float = 0.95
    ranking: Literal["equal"] | list[int] = "equal"
    allow_empty_fractions: bool = True
    ignore_failed: bool = False
    scale_trust_radius: bool = True
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
    n_max_gen: int | None = None
    pop_size: int | None = None
    n_ref_dirs: int | None = None
    progress_frequency: int | None = None


# %% Setup options

def setup_options(
    study: ProjectRepo,
    operating_mode: Literal["batch-elution", "CLR", "flip-flop", "MRSSR", "serial-columns"],
    objective: Literal["single-objective", "multi-objective", "multi-objective-per-component"],
    separation_problem: Literal["standard", "difficult", "simple", "ternary"],
    convert_to_linear: bool = False,
    apply_et_assumptions: bool = False,
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
    name = f"{operating_mode}"
    if separation_problem:
        name = f"{name}_{separation_problem}"
    if convert_to_linear:
        name = f"{name}_linear"
    if apply_et_assumptions:
        name = f"{name}_et"

    process_options = ProcessOptions(
        operating_mode=operating_mode,
        separation_problem=separation_problem,
        convert_to_linear=convert_to_linear,
        apply_et_assumptions=apply_et_assumptions,
    )

    cadet_options = CadetOptions(
        install_path=kwargs.get("install_path"),
        use_dll=kwargs.get("use_dll", True),
    )

    fractionation_opts = FractionationOptions(
        purity_required=kwargs.get("purity_required", 0.95),
        ranking=ranking,
        allow_empty_fractions=kwargs.get("allow_empty_fractions", True),
        ignore_failed=kwargs.get("ignore_failed", False),
        scale_trust_radius=kwargs.get("scale_trust_radius", True),
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
        n_max_gen=kwargs.get("n_max_gen", None),
        pop_size=kwargs.get("pop_size", None),
        n_ref_dirs=kwargs.get("n_ref_dirs", None),
        progress_frequency=kwargs.get("progress_frequency", None),
    )

    options = Options({
        "process_options": asdict(process_options),
        "optimization_options": asdict(optimization_options),
        "optimizer_options": asdict(optimizer_options),
    })

    name = f"{name}_{objective}"
    if ranking != "equal":
        ranking_str = str(ranking).replace(", ", "-").replace("[", "").replace("]", "")
        name = f"{name}_{ranking_str}"

    options.name = name
    options.commit_message = f"{name}_{str(date.today())}"
    options.branch_prefix = name.replace(" ", "_")
    options.debug = debug
    options.push = push

    case = Case(study, options=options, name=name)

    if load:
        case.load()

    return case


def setup_cases(
    work_dir: os.PathLike = "./",
    **kwargs: Any,
) -> list[Case]:
    """
    Set up cases based on the current environment and parameters.

    Args:
        kwargs: Additional kwargs for iterate cases.
    Returns:
        List of cases.
    """
    # Setup environment
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

    kwargs["temp_directory_base"] = temp_directory_base
    kwargs["cache_directory_base"] = cache_directory_base
    kwargs["install_path"] = install_path

    # Setup project repository
    study = ProjectRepo(
        work_dir,
        url="git@github.com:schmoelder/diss_operating_modes.git",
        branch="main",
        package_dir="operating_modes"
    )

    # Configure case studies
    objectives = [
        "single-objective",
        "multi-objective-per-component",
    ]
    cases = [
        # Batch-Elution (standard)
        *[
            {
                "operating_mode": "batch-elution",
                "separation_problem": "standard",
                "objective": objective,
            }
            for objective in objectives
        ],
        # Batch-Elution (standard, linear, ET assumptions)
        *[
            {
                "operating_mode": "batch-elution",
                "separation_problem": "standard",
                "convert_to_linear": True,
                "apply_et_assumptions": True,
                "objective": objective,
            }
            for objective in objectives
        ],
        # Batch-Elution (ternary)
        *[
            {
                "operating_mode": "batch-elution",
                "separation_problem": "ternary",
                "objective": objective,
            }
            for objective in objectives
        ],
        # CLR (standard)
        *[
            {
                "operating_mode": "CLR",
                "separation_problem": "standard",
                "objective": objective,
            }
            for objective in objectives
        ],
        # CLR (difficult, linear)
        *[
            {
                "operating_mode": "CLR",
                "separation_problem": "difficult",
                "convert_to_linear": True,
                "objective": objective,
            }
            for objective in objectives
        ],
        # Flip-Flop (simple, linear)
        *[
            {
                "operating_mode": "flip-flop",
                "separation_problem": "simple",
                "convert_to_linear": True,
                "objective": objective,
            }
            for objective in objectives
        ],
        # MRSSR (standard)
        *[
            {
                "operating_mode": "MRSSR",
                "separation_problem": "standard",
                "objective": objective,
            }
            for objective in objectives
        ],
        # Serial Columns (ternary)
        *[
            {
                "operating_mode": "serial-columns",
                "separation_problem": "ternary",
                "objective": objective,
            }
            for objective in objectives
        ],
    ]

    # Setup options for case studies
    return [
        setup_options(
            study,
            **case_config,
            **kwargs,
        )
        for case_config in cases
    ]


if __name__ == "__main__":
    cases = setup_cases(
        push=True,
        load=True,
        debug=False,
        fractionation_optimizer="COBYLA",
        ignore_failed=True,
        scale_trust_radius=True,
        transform_variables="auto",
        add_meta_score=True,
    )

    run = True
    force = False

    if run:
        for case in cases:
            results = case.run_study(force=force)
