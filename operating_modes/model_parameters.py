from typing import Literal

from CADETProcess.processModel import (
    BindingBaseClass,
    ChromatographicColumnBase,
    ComponentSystem,
    Langmuir,
    LumpedRateModelWithPores,
)

from operating_modes.et_simulator import (
    convert_binding_to_linear,
    convert_column_to_lrm,
)


c_feed = 10
flow_rate = 60e-6/60


def setup_binding_model(
    separation_problem: Literal["standard", "difficult", "simple", "ternary"],
    is_kinetic: bool = False,
    convert_to_linear: bool = False,
) -> BindingBaseClass:
    """
    Create a binding model for the given scenario.

    Parameters
    ----------
    separation_problem : Literal["standard", "difficult", "simple", "ternary"]
        The type of binding model to create.
    convert_to_linear : bool, optional
        If True, convert the model to a linear binding model with equivalent
        Henry coefficients. The default is False.

    Returns
    -------
    BindingBaseClass
        A configured binding model instance.
    """
    # Initialize component system
    if separation_problem == "ternary":
        component_system = ComponentSystem(["A", "B", "C"])
    else:
        component_system = ComponentSystem(["A", "B"])

    # Create and configure the binding model
    binding_model = Langmuir(component_system)
    binding_model.is_kinetic = is_kinetic

    match separation_problem:
        case "standard":
            binding_model.adsorption_rate = [0.02, 0.03]
            binding_model.desorption_rate = [1, 1]
            binding_model.capacity = [100, 100]
        case "simple":
            binding_model.adsorption_rate = [0.01, 0.20]
            binding_model.desorption_rate = [1, 1]
            binding_model.capacity = [100, 100]
        case "difficult":
            binding_model.adsorption_rate = [0.01, 0.015]
            binding_model.desorption_rate = [1, 1]
            binding_model.capacity = [100, 100]
        case "ternary":
            binding_model.is_kinetic = is_kinetic
            binding_model.adsorption_rate = [0.02, 0.03, 0.05]
            binding_model.desorption_rate = [1, 1, 1]
            binding_model.capacity = [100, 100, 200]

    if convert_to_linear:
        binding_model = convert_binding_to_linear(binding_model)

    return binding_model


def setup_column(
    binding_model: BindingBaseClass,
    apply_et_assumptions: bool = False,
) -> ChromatographicColumnBase:
    """
    Setup a chromatographic column for process simulation.

    Parameters
    ----------
    binding_model : BindingBaseClass
        The binding model to use for the column.
    apply_et_assumptions : bool, optional
        If True, apply equilibrium theory assumptions for validation:
        - LRM with equivalent total porosity
        - Axial dispersion = 0
        - Linear binding model in rapid equilibrium with equivalent Henry
          coefficients.

    Returns
    -------
    ChromatographicColumnBase
        The configured column.
    """
    component_system = binding_model.component_system
    column = LumpedRateModelWithPores(component_system, name="column")
    column.binding_model = binding_model

    # Column geometry
    column.length = 0.6
    column.diameter = 0.024
    column.bed_porosity = 0.3

    # Particle properties
    column.particle_porosity = 0.6
    column.particle_radius = 5.0e-6
    column.film_diffusion = [1e-3] * binding_model.n_comp

    # Transport
    column.axial_dispersion = 1e-6

    # Equilibrium theory assumptions
    if apply_et_assumptions:
        binding_model = convert_binding_to_linear(binding_model)
        column.binding_model = binding_model
        column = convert_column_to_lrm(column)

    return column
