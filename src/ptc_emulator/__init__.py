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

__version__ = "0.1.0"

__all__ = [
    "MODULATION_FORM",
    "cond2_normalised",
    "growth_and_stability",
    "hill_candidates",
    "hill_matrix",
    "hill_monodromy",
    "monodromy",
    "time_domain_growth",
    "__version__",
]
