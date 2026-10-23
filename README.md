# PTC Time Crystal Emulator

Reusable, method-first numerical tools for photonic time crystals. The first
target is a physically audited low-dimensional Maxwell/Floquet emulator that
can later bridge to qBIC and coupled-mode models.

This repository is the public code workspace for the local project at
`D:\ptc-time-crystal-emulator`. It is separate from the private 26-1 evidence
and execution roots.

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

Private PDFs, supplementary material, unpublished material fits, COMSOL/Lumerical
models, credentials, raw licensed-solver logs, and unaccepted 26-1 results stay
outside this public repository. Public code must carry source and configuration
hashes for each reproducible run. The local handoff and inherited 26-1 assets
are indexed in:

`C:\Users\陆星\Desktop\plans\paper_projects\Photonic_Time_Crystals_qBIC_reproduction_plan.md`

The local review package is at:

`C:\Users\陆星\Desktop\finals\Photonic_Time_Crystals_qBIC\`

## Development status

The repository was cloned on 2026-10-24. The initial emulator implementation
and tests are to be added by the execution agent under the handoff plan. Until
the independent gates pass, outputs are diagnostic and must not be labelled as
accepted physical results.
