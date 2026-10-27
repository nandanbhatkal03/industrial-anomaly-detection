# Context for AI coding assistants
Project: lightweight visual anomaly detection for industrial defect inspection (M.Tech, MCSP34/MCSP41).
Rules: train on normal images only; calibrate thresholds on normal validation images only; never tune on test labels;
one shared `Preprocessor`; keep checkpoints self-contained; model size includes memory bank/statistics;
open-source dependencies only; MVTec AD must never be committed. Keep scope small (3-5 categories first).
Read `docs/DOCUMENT_CORRECTIONS.md` before changing the evaluation protocol.
