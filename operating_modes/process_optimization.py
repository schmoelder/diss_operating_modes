import copy
import os
from typing import Literal, Optional

from CADETProcess.fractionation import FractionationOptimizer
from CADETProcess.processModel import Process
from CADETProcess.optimization import OptimizationProblem
from CADETProcess.simulator import Cadet
from CADETProcess.stationarity import MassBalance
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
        add_meta_score: bool = False,
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

        self._setup_simulator(cadet_options)

        if "cycle_time" not in self.variable_names:
            self.add_evaluator(CycleTimeDeterminator())

        fractionation_options = self._sync_purity_and_ranking(
            fractionation_options, process.n_comp
        )

        self._setup_fractionation_optimzer(
            objective,
            fractionation_options,
        )
        self._add_objectives(
            objective,
            fractionation_options.get("ranking", None),
            process.n_comp,
        )
        if add_meta_score and objective != "single-objective":
            self._add_meta_score(
                fractionation_options.get("ranking", None),
            )
        self._add_callback(objective, process.n_comp)

    def _setup_simulator(self, cadet_options) -> Cadet:
        process_simulator = Cadet(**cadet_options)
        process_simulator.time_resolution = 0.5
        process_simulator.evaluate_stationarity = True
        process_simulator.raise_exception_on_max_cycles = True
        process_simulator.stationarity_evaluator.add_criterion(MassBalance())
        if (
            "cycle_time" not in self.variable_names
            and self.evaluation_objects[0].name != "MRSSR"
        ):
            process_simulator.evaluate_stationarity = False

        self.add_evaluator(process_simulator)

    @property
    def process_simulator(self):
        return self.evaluators_dict["CADET"]

    @property
    def cycle_time_determinator(self):
        try:
            return self.evaluators_dict["CycleTimeDeterminator"]
        except KeyError:
            return

    def _sync_purity_and_ranking(
        self,
        fractionation_options: dict,
        n_comp: int,
    ) -> dict:
        fractionation_options = copy.deepcopy(fractionation_options)
        purity_required = fractionation_options["purity_required"]
        if isinstance(purity_required, float):
            purity_required = n_comp * [purity_required]

        ranking = fractionation_options["ranking"]
        if ranking == "equal":
            ranking = n_comp * [1.0]
        if isinstance(ranking, int):
            index = ranking
            ranking = n_comp * [0.0]
            ranking[index] = 1.0

        # Synchronize zeros between purity_required and ranking
        for i in range(n_comp):
            if purity_required[i] == 0 or ranking[i] == 0:
                purity_required[i] = 0.0
                ranking[i] = 0.0

        fractionation_options["purity_required"] = purity_required
        fractionation_options["ranking"] = ranking

        return fractionation_options

    def _setup_fractionation_optimzer(
        self,
        objective: Literal[
            "single-objective",
            "multi-objective-ranked",
            "multi-objective",
            "multi-objective-per-component",
        ],
        fractionation_options: dict,
    ) -> FractionationOptimizer | dict[int, FractionationOptimizer]:
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

        frac_opt = FractionationOptimizer(
            optimizer=optimizer
        )

        self.add_evaluator(
            frac_opt,
            kwargs=fractionation_options,
        )

        if objective == "multi-objective-per-component":
            purity_required = fractionation_options["purity_required"]
            ranking = fractionation_options["ranking"]

            for i, (pur, rank) in enumerate(zip(purity_required, ranking)):
                if pur > 0 and rank > 0:
                    fractionation_options = copy.deepcopy(fractionation_options)
                    fractionation_options.ranking = i
                    frac_opt_i = FractionationOptimizer(
                        optimizer = copy.deepcopy(optimizer)
                    )
                    self.add_evaluator(
                        frac_opt_i,
                        kwargs=fractionation_options,
                        name=f"FractionationOptimizer_{i}"
                    )

    def get_fractionator(self, comp_index: int | None=None):
        if comp_index is None:
            return self.evaluators_dict["FractionationOptimizer"]

        try:
            return self.evaluators_dict[f"FractionationOptimizer_{comp_index}"]
        except KeyError:
            return

    def get_evaluation_pipeline(self, comp_index: int | None=None):
        pipeline = [self.process_simulator.evaluator]

        if self.cycle_time_determinator:
            pipeline.append(self.cycle_time_determinator.evaluator)

        fractionator = self.get_fractionator(comp_index)
        if not fractionator:
            raise ValueError(f"Fractionator not configured for component {comp_index}")
        pipeline.append(fractionator.evaluator)

        return pipeline

    def _add_objectives(
        self,
        objective,
        ranking,
        n_comp,
    ):
        """Add objectives to optimization problem."""
        if objective == "single-objective":
            performance = PerformanceProduct(ranking=ranking)
            self.add_objective(
                performance,
                requires=self.get_evaluation_pipeline(),
                minimize=False,
            )
        elif objective == "multi-objective-ranked":
            productivity = Productivity(ranking=ranking)
            self.add_objective(
                productivity,
                requires=self.get_evaluation_pipeline(),
                minimize=False,
            )

            recovery = Recovery(ranking=ranking)
            self.add_objective(
                recovery,
                requires=self.get_evaluation_pipeline(),
                minimize=False,
            )

            eluent_consumption = EluentConsumption(ranking=ranking)
            self.add_objective(
                eluent_consumption,
                requires=self.get_evaluation_pipeline(),
                minimize=False,
            )
        elif objective == "multi-objective":
            productivity = Productivity()
            self.add_objective(
                productivity,
                n_objectives=n_comp,
                requires=self.get_evaluation_pipeline(),
                minimize=False,
            )

            recovery = Recovery()
            self.add_objective(
                recovery,
                n_objectives=n_comp,
                requires=self.get_evaluation_pipeline(),
                minimize=False,
            )

            eluent_consumption = EluentConsumption()
            self.add_objective(
                eluent_consumption,
                n_objectives=n_comp,
                requires=self.get_evaluation_pipeline(),
                minimize=False,
            )
        elif objective == "multi-objective-per-component":
            def _add_KPI_objectives(KPI, ranking):
                for i in range(n_comp):
                    try:
                        pipeline = self.get_evaluation_pipeline(comp_index=i)
                    except ValueError:
                        continue

                    kpi = KPI(ranking=i)
                    self.add_objective(
                        kpi,
                        name=f"{kpi}_{i}",
                        requires=pipeline,
                        minimize=False,
                    )
            _add_KPI_objectives(Productivity, ranking)
            _add_KPI_objectives(Recovery, ranking)
            _add_KPI_objectives(EluentConsumption, ranking)
        else:
            raise ValueError(f"Unknown objective: '{objective}'")

    def _add_meta_score(
        self,
        ranking,
    ):
        """Add meta score."""
        performance = PerformanceProduct(ranking)
        self.add_meta_score(
            performance,
            requires=self.get_evaluation_pipeline(),
            minimize=False,
        )

    def _add_callback(self, objective, n_comp):
        """Add callback for post-processing."""
        if objective == "multi-objective-per-component":
            for i in range(n_comp):
                try:
                    pipeline = self.get_evaluation_pipeline(comp_index=i)
                except ValueError:
                    continue

            def callback(
                fractionator,
                individual,
                evaluation_object,
                callbacks_dir
            ):
                name = f"{individual.id_short}_{evaluation_object}_fractionation_{i}"
                return fractionator.plot_fraction_signal(
                    file_name=f"{callbacks_dir}/{name}.png",
                    show=False
                )

            self.add_callback(
                callback,
                requires=pipeline,
                name=f"plot_fractionation_{i}",
                frequency=10,
            )
        else:
            def callback(
                fractionator,
                individual,
                evaluation_object,
                callbacks_dir
            ):
                return fractionator.plot_fraction_signal(
                    file_name=f"{callbacks_dir}/{individual.id_short}_{evaluation_object}_fractionation.png",
                    show=False
                )

            self.add_callback(
                callback, requires=self.get_evaluation_pipeline()
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


# %%

import numpy as np
from CADETProcess.solution import slice_solution
from CADETProcess.simulationResults import SimulationResults


class CycleTimeDeterminator():

    def __init__(self, threshold_percent: float = 1):
        self.threshold_percent = threshold_percent

    def evaluate(
        self,
        simulation_results: SimulationResults,
    ) -> SimulationResults:
        """
        Update simulation results with ideal cycle time.

        The cycle time is determined by finding the first and last element of
        the solution arrays that are above a threshold concentration.

        Parameters
        ----------
        simulation_results : SimulationResults
            The simulation results to be updated.

        Returns
        -------
        SimulationResults
            The updated simulation results.
        """
        # Check if cycle time has alredy been updated
        if simulation_results.process.cycle_time != simulation_results.time_cycle[-1]:
            return simulation_results

        simulation_results = copy.deepcopy(simulation_results)

        # Use column outlets to determine cycle time since using the outlet
        # profiles is not sufficient for processes with internal recycles
        # (especially for CLR)
        process = simulation_results.process
        units_with_binding = process.flow_sheet.units_with_binding
        unit_solutions = [
            simulation_results.solution_cycles[unit.name].outlet[-1]
            for unit in units_with_binding
        ]

        solutions = np.stack(
            [sol.solution for sol in unit_solutions],
            axis=0,
        )  # shape: (n_chrom, n_time, n_comp)

        c_max = np.max(solutions, axis=(0, 1))  # per component
        threshold = self.threshold_percent/100 * c_max

        mask_time = np.any(solutions >= threshold, axis=(0, 2))
        indices = np.flatnonzero(mask_time)

        if indices.size == 0:
            first, last = 0, solutions.shape[1] - 1
        else:
            first, last = indices[0], indices[-1]

        time_first = simulation_results.time_cycle[first]
        time_last = simulation_results.time_cycle[last]
        cycle_time = time_last - time_first

        chromatograms_new = [
            slice_solution(chrom, coordinates={"time": [time_first, time_last]})
            for chrom in simulation_results.chromatograms
        ]
        # Shift time
        chromatograms_new = [
            chrom.offset(-time_first)
            for chrom in chromatograms_new
        ]

        process.cycle_time = cycle_time
        simulation_results.chromatograms = chromatograms_new

        return simulation_results

    __call__ = evaluate

    def __str__(self):
        return "CycleTimeDeterminator"
