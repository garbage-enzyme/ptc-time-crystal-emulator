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

The initial emulator implementation and tests are to be added by the execution
agent under the project handoff plan. Until
the independent gates pass, outputs are diagnostic and must not be labelled as
accepted physical results.
