"""
Simulate chromatographic processes analytically using equilibrium theory.

## Binding Models

Note, this currently only works for linear isotherms.
Binding kintics will be ignored.

## Processes

- BatchElution
- CLR: Simulate recycling by virtually extending column.
- SSR: Simulate recycling by running sequential batch elution cycles and determining concentration of mixed fraction.
- Flip-Flop
- Serial Columns

Note, LRMP / GRM models. will be converted to a LRM with equal total porosity and zero axial dispersion.

---

Further ideas / future plans follow below

## Simulator
Currently, we simply provide a runner method.
We could also consider subclassing the SimulatorBase for a cleaner interface.
However, with the many limitations, it"s not clear whether this is worth the effort.

"""

import warnings
from collections import defaultdict
from typing import Optional

from addict import Dict
import matplotlib.pyplot as plt
import numpy as np

from CADETProcess.plotting import chromapy_cycler
from CADETProcess.processModel import (
    ChromatographicColumnBase,
    Linear,
    Langmuir,
    LumpedRateModelWithoutPores,
    Process,
)
from CADETProcess.simulator import Cadet
from CADETProcess.solution import SolutionIO
from CADETProcess.simulationResults import SimulationResults
from CADETProcess.modelBuilder import (
    BatchElution,
    CLR,
    FlipFlop,
    MRSSR,
    SerialColumns,
)


# %% Utils for ET assumptions

def convert_column_to_lrm(
    column: ChromatographicColumnBase,
) -> LumpedRateModelWithoutPores:
    """Convert column model to LumpedRateModelWithoutPores."""
    if not isinstance(column, LumpedRateModelWithoutPores):
        binding_model = column.binding_model
        column = LumpedRateModelWithoutPores(
            column.component_system,
            column.name,
            length=column.length,
            diameter=column.diameter,
            total_porosity=column.total_porosity,
            axial_dispersion=0,
        )
        column.binding_model = binding_model
    return column


def convert_binding_to_linear(
    binding_model: Langmuir,
) -> Linear:
    """Convert binding model to Linear."""
    linear = Linear(binding_model.component_system)
    linear.adsorption_rate = binding_model.henry_coefficient
    linear.desorption_rate = binding_model.desorption_rate
    linear.is_kinetic = False

    return linear


def apply_et_assumptions(
    process: Process,
    ncol: int = 500,
    weno_order: int = 2,
) -> Process:
    """Apply assumptions of equilibrium theory to the process."""
    for column in process.flow_sheet.units_with_binding:
        if not isinstance(column, LumpedRateModelWithoutPores):
            warnings.warn("The column model contains transport limiting effects.")
            column.film_diffusion = 1
            try:
                column.pore_diffusion = 1
            except AttributeError:
                pass
            try:
                column.surface_diffusion = 1
            except AttributeError:
                pass
        if not np.all(column.axial_dispersion == 0):
            warnings.warn("Axial dispersion is not 0. Equilibrium theory assumes no dispersion.")
            column.axial_dispersion = 0
        if column.binding_model.is_kinetic:
            warnings.warn("Binding model is kinetic. Equilibrium theory assumes rapid equilibrium.")
            column.binding_model.is_kinetic = False
        column.discretization.ncol = ncol
        column.discretization.weno_parameters.weno_order = weno_order
    return process


# %% ET Simulator

class ETSimulator:
    """Analytical solutions for different processes using equilibrium theory."""

    supported_binding_models = (Linear,)
    supported_processes = (BatchElution, CLR, FlipFlop, MRSSR, SerialColumns)

    @staticmethod
    def _phase_ratio(eps: float) -> float:
        return (1 - eps) / eps

    @staticmethod
    def _velocity(u0: float, F: float, a_i: float) -> float:
        return u0 / (1 + F * a_i)

    @staticmethod
    def _empty_solution(
        process: Process,
        n: int
    ) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
        """Create empty solution arrays for all units."""
        time = np.linspace(0, process.cycle_time, n)
        solutions = {}
        for unit in process.flow_sheet.units:
            solutions[unit.name] = {
                "inlet": np.zeros((n, process.n_comp)),
                "column_0": np.zeros((n, process.n_comp)),
                "column_L": np.zeros((n, process.n_comp)),
                "outlet": np.zeros((n, process.n_comp))
            }
        return time, solutions

    @staticmethod
    def _write_pulse(
        time: np.ndarray,
        solution: np.ndarray,
        comp: int,
        t_start: float,
        t_end: float,
        value: float,
    ) -> None:
        """Write a pulse into the solution array."""
        mask = (t_start < time) & (time <= t_end)
        solution[mask, comp] = value

    @staticmethod
    def _elution_window(
        length: float,
        u0: float,
        F: float,
        a_i: float,
        width: float,
        shift: float = 0.0,
    ) -> tuple[float, float]:
        t_start = length / ETSimulator._velocity(u0, F, a_i) + shift
        return t_start, t_start + width

    def _build_results(
        self,
        process: Process,
        time: np.ndarray,
        solutions: Dict[str, Dict[str, np.ndarray]],
    ) -> SimulationResults:
        """Build simulation results for all units, including flow rates."""
        solution = Dict()
        for unit in process.flow_sheet.units:
            solution[unit.name] = defaultdict(list)

        # Add solutions for all units, with flow rates determined by the process
        for unit_name, unit_solutions in solutions.items():
            for port, sol in unit_solutions.items():
                if "column" in port:
                    continue
                solution[unit_name][port].append(
                    SolutionIO(
                        unit_name,
                        process.component_system,
                        time,
                        sol,
                        process.flow_rate_timelines[unit_name]["total_out"][None].value(time)                    )
                    )

        chromatograms = [
            solution[o.name]["outlet"][-1]
            for o in process.flow_sheet.product_outlets
        ]

        return SimulationResults(
            solver_name="ETSimulator",
            solver_parameters=dict(),
            exit_flag=0,
            exit_message="Simulation Successful",
            time_elapsed=0.0,
            process=process,
            solution_cycles=solution,
            sensitivity_cycles=None,
            system_state=None,
            chromatograms=chromatograms,
        )

    def simulate(
        self,
        process: Process,
        n: int = 1001,
        **kwargs,
    ) -> SimulationResults:
        """Simulate the process using equilibrium theory."""
        if not isinstance(process, self.supported_processes):
            raise TypeError(f"Unexpected Process. Supported: {self.supported_processes}")

        process = apply_et_assumptions(process)

        for eluent in process.flow_sheet.eluent_inlets:
            if not np.all(eluent.c == 0):
                warnings.warn("Eluent concentration is non-zero. Equilibrium theory assumes pulse injections.")
                eluent.c = 0

        for unit in process.flow_sheet.units_with_binding:
            if not isinstance(unit.binding_model, self.supported_binding_models):
                raise TypeError(f"Unsupported binding model. Supported: {self.supported_binding_models}")

        if isinstance(process, BatchElution):
            return self._simulate_batch(process, n)
        if isinstance(process, CLR):
            return self._simulate_clr(process, n)
        if isinstance(process, FlipFlop):
            return self._simulate_flip_flop(process, n)
        if isinstance(process, MRSSR):
            return self._simulate_mrssr(process, n, **kwargs)
        if isinstance(process, SerialColumns):
            return self._simulate_serial_columns(process, n)

        raise RuntimeError("Unreachable process dispatch")

    def _simulate_batch(
        self,
        process: BatchElution,
        n: int = 1001,
    ) -> SimulationResults:
        """Simulate Batch Elution process, including all inlets and outlets."""
        column = convert_column_to_lrm(process.flow_sheet.column)
        binding = column.binding_model

        c_feed = process.flow_sheet.feed.c[:, 0]
        t_inj = process.feed_duration.time

        u0 = column.calculate_interstitial_velocity(process.feed_on.state)
        F = self._phase_ratio(column.total_porosity)

        time, solutions = self._empty_solution(process, n)

        # Feed: pulse from 0 to t_inj
        for i in range(process.n_comp):
            self._write_pulse(
                time,
                solutions["feed"]["outlet"],
                i,
                0.0,
                t_inj,
                c_feed[i]
            )

        # Column: inlet and outlet
        for i, a_i in enumerate(binding.k_eq):
            self._write_pulse(
                time,
                solutions["column"]["inlet"],
                i,
                0.0,
                t_inj,
                c_feed[i]
            )
            ts, te = self._elution_window(
                length=column.length,
                u0=u0,
                F=F,
                a_i=a_i,
                width=t_inj,
            )
            self._write_pulse(
                time,
                solutions["column"]["outlet"],
                i,
                ts,
                te,
                c_feed[i]
            )
            self._write_pulse(
                time,
                solutions["outlet"]["outlet"],
                i,
                ts,
                te,
                c_feed[i]
            )

        return self._build_results(process, time, solutions)

    def _simulate_clr(
        self,
        process: CLR,
        n: int = 1001,
    ) -> SimulationResults:
        """Simulate Closed-Loop Recycling process, including all inlets and outlets."""
        column = convert_column_to_lrm(process.flow_sheet.column)
        binding = column.binding_model

        c_feed = process.flow_sheet.feed.c[:, 0]
        t_inj = process.feed_duration.time
        t_recycle_off = process.recycle_off_output_state.time

        u0 = column.calculate_interstitial_velocity(process.feed_on.state)
        F = self._phase_ratio(column.total_porosity)

        time, solutions = self._empty_solution(process, n)

        # Feed: pulse from 0 to t_inj
        for i in range(process.n_comp):
            self._write_pulse(
                time,
                solutions["feed"]["outlet"],
                i,
                0.0,
                t_inj,
                c_feed[i]
            )

        # Column: inlet and outlet
        for i, a_i in enumerate(binding.k_eq):
            j = 0
            peak_width = t_inj
            while True:
                ts, te = self._elution_window(
                    length=(j + 1) * column.length,
                    u0=u0,
                    F=F,
                    a_i=a_i,
                    width=peak_width,
                )
                if te < t_recycle_off:
                    # Full recycle
                    self._write_pulse(
                        time,
                        solutions["column"]["outlet"],
                        i,
                        ts,
                        te,
                        c_feed[i]
                    )
                elif ts < t_recycle_off < te:
                    # Partial elution
                    self._write_pulse(
                        time,
                        solutions["column"]["outlet"],
                        i,
                        ts,
                        te,
                        c_feed[i]
                    )
                    self._write_pulse(
                        time,
                        solutions["outlet"]["outlet"],
                        i,
                        t_recycle_off,
                        te,
                        c_feed[i]
                    )
                    # Reduce peak remaining peak width
                    peak_width = t_inj - (te - t_recycle_off)
                else:
                    # Elution
                    self._write_pulse(
                        time,
                        solutions["column"]["outlet"],
                        i,
                        ts,
                        te,
                        c_feed[i]
                    )
                    self._write_pulse(
                        time,
                        solutions["outlet"]["outlet"],
                        i,
                        ts,
                        te,
                        c_feed[i]
                    )
                    break
                j += 1

        return self._build_results(process, time, solutions)

    def _simulate_flip_flop(
        self,
        process: FlipFlop,
        n: int = 1001,
    ) -> SimulationResults:
        """Simulate Flip-Flop process, including all inlets and outlets."""
        column = convert_column_to_lrm(process.flow_sheet.column)
        binding = column.binding_model

        c_feed = process.flow_sheet.feed.c[:, 0]
        t_inj = process.feed_duration.time
        t_delay_flip = process.delay_flip.time
        t_delay_injection = process.delay_injection.time

        t_flip = t_inj + t_delay_flip
        t_inj_2 = t_flip + t_delay_injection

        u0 = column.calculate_interstitial_velocity(process.feed_on_1.state)
        F = self._phase_ratio(column.total_porosity)

        time, solutions = self._empty_solution(process, n)

        # Feed: pulse from 0 to t_inj
        for i in range(process.n_comp):
            self._write_pulse(
                time,
                solutions["feed"]["outlet"],
                i,
                0.0,
                t_inj,
                c_feed[i]
            )

        # Column: inlet and outlet
        for i, a_i in enumerate(binding.k_eq):
            ts, te = self._elution_window(
                length=column.length,
                u0=u0,
                F=F,
                a_i=a_i,
                width=t_inj,
            )
            if te < t_flip:
                self._write_pulse(
                    time,
                    solutions["column"]["outlet"],
                    i,
                    ts,
                    te,
                    c_feed[i]
                )
                self._write_pulse(
                    time,
                    solutions["outlet"]["outlet"],
                    i,
                    ts,
                    te,
                    c_feed[i]
                )
                self._write_pulse(
                    time,
                    solutions["column"]["outlet"],
                    i,
                    ts + t_inj_2,
                    te + t_inj_2,
                    c_feed[i]
                )
                self._write_pulse(
                    time,
                    solutions["outlet"]["outlet"],
                    i,
                    ts + t_inj_2,
                    te + t_inj_2,
                    c_feed[i]
                )
            elif ts < t_flip < te:
                self._write_pulse(
                    time,
                    solutions["column"]["outlet"],
                    i,
                    ts,
                    t_flip,
                    c_feed[i]
                )
                self._write_pulse(
                    time,
                    solutions["outlet"]["outlet"],
                    i,
                    ts,
                    t_flip,
                    c_feed[i]
                )
                self._write_pulse(
                    time,
                    solutions["column"]["outlet"],
                    i,
                    ts + t_inj_2,
                    t_flip + t_inj_2,
                    c_feed[i]
                )
                self._write_pulse(
                    time,
                    solutions["outlet"]["outlet"],
                    i,
                    ts + t_inj_2,
                    t_flip + t_inj_2,
                    c_feed[i]
                )
                peak_width = te - t_flip
                start = t_inj + 2 * t_delay_flip
                end = start + peak_width
                self._write_pulse(
                    time,
                    solutions["column"]["outlet"],
                    i,
                    start,
                    end,
                    c_feed[i]
                )
                self._write_pulse(
                    time,
                    solutions["outlet"]["outlet"],
                    i,
                    start,
                    end,
                    c_feed[i]
                )
                self._write_pulse(
                    time,
                    solutions["column"]["outlet"],
                    i,
                    start + t_inj_2,
                    end + t_inj_2,
                    c_feed[i]
                )
                self._write_pulse(
                    time,
                    solutions["outlet"]["outlet"],
                    i,
                    start + t_inj_2,
                    end + t_inj_2,
                    c_feed[i]
                )
            else:
                start = t_inj + 2 * t_delay_flip
                end = start + t_inj
                self._write_pulse(
                    time,
                    solutions["column"]["outlet"],
                    i,
                    start,
                    end,
                    c_feed[i]
                )
                self._write_pulse(
                    time,
                    solutions["outlet"]["outlet"],
                    i,
                    start,
                    end,
                    c_feed[i]
                )
                self._write_pulse(
                    time,
                    solutions["column"]["outlet"],
                    i,
                    start + t_inj_2,
                    end + t_inj_2,
                    c_feed[i]
                )
                self._write_pulse(
                    time,
                    solutions["outlet"]["outlet"],
                    i,
                    start + t_inj_2,
                    end + t_inj_2,
                    c_feed[i]
                )

        return self._build_results(process, time, solutions)

    def _simulate_mrssr(
        self,
        process: MRSSR,
        n: int = 1001,
        n_cycles: int = 1,
    ) -> SimulationResults:
        """
        Simulate MRSSR process using analytical solution for equilibrium theory.

        The recycling process is simulated by virtually increasing the column
        length for each cycle.
        """
        column = convert_column_to_lrm(process.flow_sheet.column)

        c_feed = process.flow_sheet.feed.c[:, 0]
        t_inj = process.feed_duration.time
        flow_rate = process.feed_on.state

        V_feed = t_inj * flow_rate
        V_tank = process.flow_sheet.tank.V
        c_tank_init = np.array(process.flow_sheet.tank.c)
        recycle_duration = process.recycle_off.time - process.recycle_on.time
        injection_duration = t_inj + recycle_duration
        V_inj = injection_duration * flow_rate

        n_tank = V_tank * c_tank_init
        c_tank = c_tank_init
        n_feed = V_feed * c_feed

        # Initialize a BatchElution process for the first cycle
        batch_process = BatchElution(
            column,
            c_feed=c_tank,
            flow_rate=flow_rate,
            feed_duration=injection_duration,
            cycle_time=process.cycle_time,
        )

        simulation_results = None

        for _ in range(n_cycles):
            batch_process.flow_sheet.feed.c = c_tank

            # Simulate the current cycle
            new_results = self._simulate_batch(batch_process, n)

            # Zero out the recycled fraction in the outlet
            solution_outlet = new_results.solution.outlet.outlet
            time = solution_outlet.time
            recycle_indices = np.where(
                (time >= process.recycle_on.time) & (time < process.recycle_off.time)
            )[0]
            solution_outlet.solution[recycle_indices] = 0
            solution_outlet.flow_rate = process.flow_rate_timelines["outlet"]["total_out"][None]

            solution_outlet.update_solution()

            # Update or initialize simulation_results
            if simulation_results is None:
                simulation_results = new_results
            else:
                simulation_results.update(new_results)

            # Calculate the recycled mass and update tank concentration
            recycle_fraction = new_results.solution.column.outlet.create_fraction(
                process.recycle_on.time, process.recycle_off.time
            )
            n_rec = recycle_fraction.mass
            n_inj = c_tank * V_inj
            n_tank = n_tank - n_inj + n_rec + n_feed
            c_tank = n_tank / V_tank

        # Ensure the process reference is correct
        simulation_results.process = process

        return simulation_results


    def _simulate_serial_columns(
        self,
        process: SerialColumns,
        n: int = 1001,
    ) -> SimulationResults:
        """Simulate Serial Columns process, including all inlets and outlets."""
        column_1 = convert_column_to_lrm(process.flow_sheet.column_1)
        column_2 = convert_column_to_lrm(process.flow_sheet.column_2)
        binding = column_1.binding_model

        c_feed = process.flow_sheet.feed.c[:, 0]
        t_inj = process.feed_duration.time
        t_serial_on = process.serial_on.time
        t_serial_off = process.serial_off.time

        u0 = column_1.calculate_interstitial_velocity(process.feed_on.state)
        F = self._phase_ratio(column_1.total_porosity)

        time, solutions = self._empty_solution(process, n)

        # Feed: pulse from 0 to t_inj
        for i in range(process.n_comp):
            self._write_pulse(
                time,
                solutions["feed"]["outlet"],
                i,
                0.0,
                t_inj,
                c_feed[i]
            )

        # Column 1: inlet and outlet
        for i, a_i in enumerate(binding.k_eq):
            self._write_pulse(
                time,
                solutions["column_1"]["inlet"],
                i,
                0.0,
                t_inj,
                c_feed[i]
            )
            ts, te = self._elution_window(
                length=column_1.length,
                u0=u0,
                F=F,
                a_i=a_i,
                width=t_inj,
            )
            self._write_pulse(
                time,
                solutions["column_1"]["outlet"],
                i,
                ts,
                te,
                c_feed[i]
            )

        # Column 2: inlet (from column_1 outlet during serial phase)
        mask_serial = (min(t_serial_on, t_serial_off) <= time) & (time < max(t_serial_on, t_serial_off))
        for i in range(process.n_comp):
            solutions["column_2"]["inlet"][mask_serial, i] = solutions["column_1"]["outlet"][mask_serial, i]

        # Column 2: outlet
        for i, a_i in enumerate(binding.k_eq):
            ts, te = self._elution_window(
                length=column_1.length + column_2.length,
                u0=u0,
                F=F,
                a_i=a_i,
                width=t_inj,
            )
            self._write_pulse(
                time,
                solutions["column_2"]["outlet"],
                i,
                ts,
                te,
                c_feed[i]
            )

        # Outlet 1: column_1 outlet, zero during serial phase
        solutions["outlet_1"]["outlet"] = solutions["column_1"]["outlet"].copy()
        solutions["outlet_1"]["outlet"][mask_serial, :] = 0.0

        # Outlet 2: column_2 outlet
        solutions["outlet_2"]["outlet"] = solutions["column_2"]["outlet"].copy()

        return self._build_results(process, time, solutions)


# %% Simulation utils

def simulate_and_plot(process, **kwargs):
    """Simulate process and plot results."""
    simulator = ETSimulator()
    et_results = simulator.simulate(process, **kwargs)
    et_results.solution.outlet.outlet.plot()


def compare_cadet_with_et(
    process: Process,
    plot_column_outlet: Optional[bool] = False,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes] | tuple[plt.Figure, plt.Axes, plt.Figure, plt.Axes]:
    """Compare of CADET with analytical solution using equilibrium theory."""
    process = apply_et_assumptions(process)

    cadet_simulator = Cadet()
    cadet_simulator.time_resolution = 0.1
    if "n_cycles" in kwargs:
        cadet_simulator.n_cycles = kwargs["n_cycles"]
    cadet_results = cadet_simulator.simulate(process)

    et_simulator = ETSimulator()
    et_results = et_simulator.simulate(process, **kwargs)
    et_time = et_results.time_cycle / 60

    for outlet in process.flow_sheet.product_outlets:
        fig, ax = cadet_results.solution[outlet.name].outlet.plot()

        # Reset the color cycler
        ax.set_prop_cycle(chromapy_cycler)

        et_outlet_solution = et_results.solution[outlet.name].outlet.solution
        ax.plot(et_time, et_outlet_solution, "--")

        fig.tight_layout()

    if not plot_column_outlet:
        return fig, ax

    for column in process.flow_sheet.units_with_binding:
        fig_col, ax_col = cadet_results.solution[column.name].outlet.plot()

        et_column_solution = et_results.solution[outlet.name].outlet.solution
        ax_col.plot(et_time, et_column_solution)

        fig_col.tight_layout()

    return fig, ax, fig_col, ax_col
