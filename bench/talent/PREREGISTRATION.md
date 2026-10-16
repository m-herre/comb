# TALENT held-out read: pre-registration

Copied verbatim from the research repository (nanotabiclv2-v3, `docs/INDUCT_V3.md`, commit `a8218f6`,
2026-10-11 10:54 +0200, before any stage-4 training). The manifest it names is [manifest.csv](manifest.csv) here
(same sha256). "The selected stage-4 arm" is the released weights (`comb_s4`). The stage-3 checkpoint it also names
is not released, so its part of the read is not in this repository. Runs that `run_talent.py` marks as post-hoc are
not part of this pre-registration.

## Pre-registration: TALENT held-out read (frozen 2026-10-11, before any stage-4 training)

* Suite: `docs/talent_heldout_manifest.csv` (sha256 71121b30d1ebbdad59be7f2ff2aa6078b35f4d7a7fc91ac274e4292b7bf8d9dc),
  `include == True`: 106 TALENT classification tasks (63 binary, 43 multiclass), source dataset-250115.zip;
  excluded: every TabArena-v0.1 / BeyondArena / heldout21 overlap (71), > 10 classes (11), in-TALENT duplicates
  (12). No model has been run on it. The realism prior is designed from TA / BA metadata only, never from TALENT.
* Protocol: TALENT's fixed splits; context = train + val rows, evaluation = test rows; one fit per task; every model
  through its sklearn wrapper at defaults (n_estimators 8 for nanotabiclv2 checkpoints, fp32), V3 models at the
  checkpoint's inference M. Same preprocessing for all (TALENT N_/C_ arrays concatenated, categoricals ordinal).
* Models read (at most): the selected stage-4 arm, the V3 stage-3 final, released TabICLv2; XGBoost and CatBoost
  defaults as context. The stage-4 arm is chosen on TA / BA single Elo before the read; nothing is tuned after it.
* Primary metric: mean over tasks of the paired AUC delta (selected arm - TabICLv2), one-vs-rest macro AUC for
  multiclass, 95 % bootstrap CI over tasks (10 000 resamples, seed 0). Secondary: accuracy delta, win / loss count,
  delta of the stage-4 arm vs the V3 final, size slices (< 5k, 5k-50k, > 50k rows).
* Pass rules: "TabICLv2-level" = CI lower bound > -0.25 pp; "beats TabICLv2" = CI lower bound > 0;
  "stage 4 helps" = arm - V3 final CI lower bound > 0. Read once; the outcome is reported whatever it is.
* Known confound: TabPFN-3.5's report also evaluates on TALENT, so the RK35 teacher may be tuned toward it.
