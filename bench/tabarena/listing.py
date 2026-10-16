"""Leaderboard entries that are not (yet) on the public TabArena leaderboard, by TabArena's own metadata: hosted
closed-source APIs under review (tag ``closed-source-api``) and TabArena-Lite entries with ``verified=False`` (not
signed off by the maintainers). BeyondArena's ``verified=False`` marks the maintainers' own 2026-09-22
foundation-model run (a recorded 16 h time-limit deviation), not pending entries, so those stay in (TabSwift is left
out on both, being unverified on TabArena-Lite).
Checked against the tabarena checkout of 2026-10-16 (590b351e)."""

NOT_LISTED = frozenset({
    "Chakra-Tab (full)", "Chakra-Tab (medium)",  # hosted API, under review
    "TABDPT_TURBO (default)", "TA-LIMIX (default)", "TA-TABSWIFT (default)", "SAP-RPT-OSS (default)",
    "TA-ORION-MSP (default)", "BETA_GPU (default)", "TABFLEX_GPU (default)",  # verified=False on TabArena-Lite
})
