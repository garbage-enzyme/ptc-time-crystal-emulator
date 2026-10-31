"""ptc_emulator: reusable, audited numerical tools for photonic time crystals.

Method-first: every solver that backs a reported quantity must pass the
independent numerical gates encoded in the test suite (monodromy vs Hill
vs direct time-domain cross-checks, differential residuals, harmonic tails,
Jury-consistent stability verdicts). Until those gates pass, outputs are
diagnostic and must not be labelled accepted physical results.
"""

from .lowdim import (
    MODULATION_FORM,
    cond2_normalised,
    growth_and_stability,
    hill_candidates,
    hill_matrix,
    hill_monodromy,
    monodromy,
    time_domain_growth,
)
from .drude import (
    Cumulative,
    drude_history,
    kernel_chi_literal,
    memory_state_rhs,
    pulse_envelope,
    solve_j_log_form,
    solve_memory_state,
    stationary_ade_rhs,
    stationary_chi,
)
from .drude_maxwell import (
    augmented_rhs,
    direct_time_growth,
    drude_dispersion_roots,
    harmonic_matrix,
    match_sets,
    monodromy_augmented,
    multipliers_from_harmonic,
    neutral_direction,
    project_out_neutral,
    solve_augmented,
    static_matrix,
)

__version__ = "0.3.3"

__all__ = [
    "MODULATION_FORM",
    "cond2_normalised",
    "growth_and_stability",
    "hill_candidates",
    "hill_matrix",
    "hill_monodromy",
    "monodromy",
    "time_domain_growth",
    "Cumulative",
    "drude_history",
    "kernel_chi_literal",
    "memory_state_rhs",
    "pulse_envelope",
    "solve_j_log_form",
    "solve_memory_state",
    "stationary_ade_rhs",
    "stationary_chi",
    "augmented_rhs",
    "direct_time_growth",
    "drude_dispersion_roots",
    "harmonic_matrix",
    "match_sets",
    "monodromy_augmented",
    "multipliers_from_harmonic",
    "neutral_direction",
    "project_out_neutral",
    "solve_augmented",
    "static_matrix",
    "__version__",
]
