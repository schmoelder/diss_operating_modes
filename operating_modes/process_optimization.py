import copy
import os
from typing import Literal, Optional

from CADETProcess.fractionation import FractionationOptimizer
from CADETProcess.processModel import Process
from CADETProcess.optimization import OptimizationProblem
from CADETProcess.simulator import Cadet
from CADETProcess.performance import (
    PerformanceProduct,
    Productivity,
    EluentConsumption,
    Recovery,
)
from CADETProcess.optimization import OptimizerBase, U_NSGA3, COBYQA


class ProcessOptimization(OptimizationProblem):
    """Set-up process optimization problems."""

    def __init__(
        self,
        name: str,
        process: Process,
        variables: list[dict],
        objective: Literal[
            "single-objective",
            "multi-objective-ranked",
            "multi-objective",
            "multi-objective-per-component",
        ],
        linear_constraints: Optional[list[dict]] = None,
        variable_dependencies: Optional[list[dict]] = None,
        cadet_options: Optional[dict] = None,
        fractionation_options: Optional[dict] = None,
        cache_directory: Optional[os.PathLike] = None,
    ) -> None:
        super().__init__(
            name=name,
            cache_directory=cache_directory,
        )

        self.add_evaluation_object(process)

        for var in variables:
            self.add_variable(**var)

        if linear_constraints is not None:
            for lin_con in linear_constraints:
                self.add_linear_constraint(**lin_con)

        if variable_dependencies is not None:
            for var_dep in variable_dependencies:
                self.add_variable_dependency(**var_dep)

        simulator = self._setup_simulator(cadet_options)
        frac_opt = self._setup_fractionator(objective, fractionation_options)
        self._add_objectives(
            objective,
            simulator,
            frac_opt,
            fractionation_options.get("ranking", None),
            process.n_comp,
        )
        self._add_callback(simulator, frac_opt)

    def _setup_simulator(self, cadet_options) -> Cadet:
        process_simulator = Cadet(**cadet_options)
        process_simulator.evaluate_stationarity = True

        self.add_evaluator(process_simulator)

        return process_simulator

    def _setup_fractionator(
        self,
        objective: Literal[
            "single-objective",
            "multi-objective-ranked",
            "multi-objective",
            "multi-objective-per-component",
        ],
        fractionation_options: dict,
    ) -> FractionationOptimizer | dict[int, FractionationOptimizer]:
        fractionation_options = copy.deepcopy(fractionation_options)

        match fractionation_options.pop("optimizer", None):
            case "COBYLA":
                optimizer = None
            case "COBYQA":
                optimizer = COBYQA()
                optimizer.x_tol = 1e-4
                optimizer.cv_nonlincon_tol = 5e-3
                optimizer.initial_tr_radius = 1e-3
            case _:
                optimizer = None

        if objective != "multi-objective-per-component":
            frac_opt = FractionationOptimizer(
                optimizer=optimizer
            )

            self.add_evaluator(
                frac_opt,
                kwargs=fractionation_options,
            )
        else:
            frac_opt = {}
            for i, pur in enumerate(fractionation_options.purity_required):
                if pur > 0:
                    fractionation_options = copy.deepcopy(fractionation_options)
                    fractionation_options.ranking = i
                    frac_opt_i = FractionationOptimizer(
                        optimizer = copy.deepcopy(optimizer)
                    )
                    self.add_evaluator(
                        frac_opt_i,
                        kwargs=fractionation_options,
                    )

                    frac_opt[i] = frac_opt_i

        return frac_opt

    def _add_objectives(
        self,
        objective,
        simulator,
        frac_opt,
        ranking,
        n_comp,
    ):
        """Add objectives to optimization problem."""
        if objective == "single-objective":
            performance = PerformanceProduct(ranking=ranking)
            self.add_objective(
                performance,
                requires=[simulator, frac_opt],
                minimize=False,
            )
        elif objective == "multi-objective-ranked":
            productivity = Productivity(ranking=ranking)
            self.add_objective(
                productivity,
                requires=[simulator, frac_opt],
                minimize=False,
            )

            recovery = Recovery(ranking=ranking)
            self.add_objective(
                recovery,
                requires=[simulator, frac_opt],
                minimize=False,
            )

            eluent_consumption = EluentConsumption(ranking=ranking)
            self.add_objective(
                eluent_consumption,
                requires=[simulator, frac_opt],
                minimize=False,
            )
        elif objective == "multi-objective":
            productivity = Productivity()
            self.add_objective(
                productivity,
                n_objectives=n_comp,
                requires=[simulator, frac_opt],
                minimize=False,
            )

            recovery = Recovery()
            self.add_objective(
                recovery,
                n_objectives=2,
                requires=[simulator, frac_opt],
                minimize=False,
            )

            eluent_consumption = EluentConsumption()
            self.add_objective(
                eluent_consumption,
                n_objectives=2,
                requires=[simulator, frac_opt],
                minimize=False,
            )
        elif objective == "multi-objective-per-component":
            for i in range():
                productivity = Productivity()
                self.add_objective(
                    productivity,
                    name=f"productivity_{i}",
                    requires=[simulator, frac_opt[i]],
                    minimize=False,
                )

                recovery = Recovery()
                self.add_objective(
                    recovery,
                    name=f"recovery_{i}",
                    requires=[simulator, frac_opt[i]],
                    minimize=False,
                )

                eluent_consumption = EluentConsumption()
                self.add_objective(
                    eluent_consumption,
                    name=f"eluent_consumption_{i}",
                    requires=[simulator, frac_opt[i]],
                    minimize=False,
                )
        else:
            raise ValueError(f"Unknown objective: '{objective}'")

    def _add_callback(self, simulator, frac_opt):
        """Add callback for post-processing."""
        def callback(fractionator, individual, evaluation_object, callbacks_dir):
            return fractionator.plot_fraction_signal(
                file_name=f"{callbacks_dir}/{individual.id_short}_{evaluation_object}_fractionation.png",
                show=False
            )

        self.add_callback(
            callback, requires=[simulator, frac_opt]
        )


def setup_optimizer(
    optimization_problem: OptimizationProblem,
    optimizer_options: dict,
    progress_frequency: int | None = None
) -> OptimizerBase:
    """
    Set-up optimizer.

    Note that by default the number of individuals is scaled by the number of
    variables in the problem.

    Parameters
    ----------
    optimization_problem : OptimizationProblem
        The optimization problem.
    optimizer_options : dict
        Options for the optimizer.
    progress_frequency : int | None, default=None
        Number of generations after which the optimizer reports progress.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    OptimizerBase
        The configured optimizer.

    """
    if optimizer_options["optimizer"] == "U_NSGA3":
        optimizer = U_NSGA3()
        default_options = {
            "n_cores": -4,
            "pop_size": optimization_problem.n_variables * 64,
            "n_max_gen": 64,
            "progress_frequency": None,
        }
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_options.optimizer}")

    default_options.update({
        k: v for k, v in optimizer_options.items() if v is not None
    })

    for key, value in default_options.items():
        setattr(optimizer, key, value)

    return optimizer
