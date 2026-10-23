# Emulator package

Reusable, method-first implementations. Paper-specific adapters and
digitization live under separate modules so the core solvers remain useful
when published parameters are incomplete.

## Modules

- `lowdim` — low-dimensional Floquet engine for
  `q'' + 2γq' + ω0²(1 + m(1 + cos ωm t))q = 0` (dimensionless units of a
  reference frequency): monodromy by direct integration, Hill harmonic
  balance with differential-residual/tail screening, Jury-consistent
  stability verdicts, renormalised time-domain growth. No I/O, no import
  side effects.

Every module that backs reported quantities must have its numerical gates
encoded in `tests/`.
