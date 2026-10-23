# PTC Time Crystal Emulator

Reusable, method-first numerical tools for photonic time crystals. The first
target is a physically audited low-dimensional Maxwell/Floquet emulator that
can later bridge to qBIC and coupled-mode models.

This repository is the public code workspace for the emulator. Research
evidence and execution data are maintained separately from this codebase.

## Scope

- independent monodromy, Hill truncation, and direct-time checks;
- fixed-real-`k` Maxwell time-varying-medium controls;
- complex response and Floquet-pole preservation;
- convergence, residual, harmonic-tail, and energy-accounting reports;
- later, source-bounded qBIC/coupled-mode bridges.

Paper-specific reproduction of *Photonic time crystals assisted by quasi-bound
states in the continuum* is a downstream validation case. It does not define
the emulator's physical correctness.

## Provenance boundary

Research-specific source material and evidence are maintained outside this
public repository. Public code must carry source and configuration hashes for
each reproducible run.

## Development status

`ptc_emulator.lowdim` is the first landed module: the audited
low-dimensional Floquet engine (monodromy by direct integration, Hill
harmonic balance with differential-residual and harmonic-tail screening,
Jury-consistent stability verdicts, and a renormalised direct-time-domain
growth cross-check). The gate tests in `tests/` are the acceptance bar for
any quantity this engine reports.

Run the tests with:

    PYTHONPATH=src python -m unittest discover -s tests -v

Outputs of the current engine remain diagnostic until the independent gates
of the project handoff pass.
