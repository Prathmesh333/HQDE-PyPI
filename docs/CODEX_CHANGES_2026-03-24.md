# Codex Changes Log - 2026-03-24

## Scope

This document records the HQDE changes made during the current Codex session, including:

- core package changes that were committed and pushed
- test coverage added for those core changes
- notebook changes made locally but not committed/pushed

Repository: `HQDE-PyPI`  
Branch: `main`  
Pushed commit: `c301c83`  
Commit message: `Improve HQDE federated training with FedAdam and local BN`

## Pushed Changes

### 1. Core trainer redesign

File:
- `hqde/core/hqde_system.py`

Main changes:

- Added stronger federated-training configuration options:
  - `training_aggregation`
  - `server_optimizer`
  - `server_learning_rate`
  - `server_beta1`
  - `server_beta2`
  - `server_epsilon`
  - `federated_normalization`
- Expanded training-config validation for the new keys and constraints.
- Reworked the adaptive quantization path from full-weight quantization to safer delta-based quantization.
- Added blockwise symmetric quantization with:
  - warmup rounds
  - skip rules for sensitive tensors
  - residual error feedback
  - real byte accounting
- Improved worker efficiency scoring so weighted aggregation has more range.
- Ensured federated workers start from synchronized initial weights.
- Disabled per-worker learning-rate/dropout diversity in `fedavg` mode so synchronized aggregation is not harmed by intentional worker drift.

### 2. Better federated optimization

File:
- `hqde/core/hqde_system.py`

Main changes:

- Replaced plain round-wise averaging as the default federated server update with a stronger setup:
  - `training_aggregation='sample_weighted'`
  - `server_optimizer='fedadam'`
  - `federated_normalization='local_bn'`
- Added server-side first- and second-moment state for FedAdam.
- Added support for:
  - `sample_weighted` training aggregation
  - `mean` training aggregation
  - `efficiency_weighted` training aggregation
- Preserved local BatchNorm state in federated mode instead of forcing global overwrite each round.

### 3. State persistence

File:
- `hqde/core/hqde_system.py`

Main changes:

- Added checkpoint save/load support for:
  - quantizer residual state
  - round counters
  - reference weights
  - server optimizer state
  - local normalization key metadata
  - per-round sample-count state

### 4. Tests added and expanded

File:
- `tests/test_local_mode.py`

Main changes:

- Added validation coverage for the new training-config fields.
- Added tests for:
  - synchronized federated worker initialization
  - blockwise delta quantization behavior
  - quantization metrics and byte accounting
  - local BatchNorm preservation in federated broadcast
  - server optimizer state being updated in federated mode

## Verification Performed

The following checks passed before/after the pushed changes:

```bash
python -m py_compile hqde/core/hqde_system.py tests/test_local_mode.py
python -m unittest discover -s tests -q
python test_imports.py
```

Observed result:

- unit tests passed
- import smoke test passed
- no full Colab/A100 benchmark rerun was performed from this environment

## Local-Only Notebook Changes

These changes were made locally and were **not** committed or pushed.

Files currently local/untracked:

- `examples/hqde_comprehensive_benchmark.ipynb`
- `examples/hqde_single_node_a100_scheduled_benchmark.ipynb`
- `examples/single_node_a100_experiment_utils.py`

### `examples/hqde_comprehensive_benchmark.ipynb`

Main local edits:

- Updated the scheduled HQDE path to reflect the new core training direction:
  - `sample_weighted`
  - `FedAdam`
  - `local_bn`
  - quantization off by default for the main accuracy run
- Replaced the older notebook-side quantization logic with a safer delta-based version.
- Updated scheduled aggregation to keep local BatchNorm state.
- Added timing fields intended for paper reporting:
  - `training_time_sec`
  - `setup_time_sec`
  - `ray_init_time_sec`
  - `avg_iteration_time_sec`
  - `median_iteration_time_sec`
  - `total_runtime_sec`
- Changed reporting so the main training-time metric excludes Ray initialization time.
- Updated the summary table to include the new timing fields.
- Updated the time figure label to indicate that reported training time excludes Ray init.
- Updated the printed HQDE scheduled experiment label to match the new default recipe.

Validation performed on the notebook:

- notebook JSON parsing succeeded
- edited code-cell source snippets compiled successfully as Python

Not performed:

- no full notebook execution from this environment

## Practical Effect of the Changes

Expected accuracy impact:

- better non-IID performance from local BatchNorm preservation
- more stable federated optimization from FedAdam
- less training damage from communication compression due to delta/blockwise quantization
- clearer separation between:
  - pure training time
  - setup time
  - Ray initialization overhead

Expected paper/reporting impact:

- the scheduled HQDE row is now closer to a serious federated training baseline
- the timing table can distinguish runtime components instead of reporting one opaque number

## Current Repo State At Time Of Writing

Tracked and pushed:

- `hqde/core/hqde_system.py`
- `tests/test_local_mode.py`

Local but not pushed:

- `examples/hqde_comprehensive_benchmark.ipynb`
- `examples/hqde_single_node_a100_scheduled_benchmark.ipynb`
- `examples/single_node_a100_experiment_utils.py`

## Recommended Next Steps

1. Rerun the scheduled benchmark notebook on Colab/A100 using the updated default HQDE scheduled path.
2. Compare:
   - `hqde_scheduled`
   - `vanilla_fedavg`
   - `flower_fedavg`
   - `flower_fedprox`
3. If the new notebook results look reasonable, commit the notebook separately from the package code.
4. Keep quantization as an ablation until it consistently helps the accuracy/communication tradeoff.
