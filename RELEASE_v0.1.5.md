# Release v0.1.5 Notes

These are historical release notes. Placeholder expected-accuracy claims have been removed.

## Main Changes

- Added FedAvg-style epoch aggregation.
- Added learning-rate scheduling.
- Added gradient clipping.
- Added worker diversity through learning-rate and dropout variation where supported by the model constructor.
- Improved logging around training and aggregation.

## Reporting Policy

Do not use this release note as a source for benchmark results. Run the benchmark scripts/notebooks and report measured metrics with hardware, seed, package versions, and raw output artifacts.
