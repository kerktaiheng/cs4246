# Resuming the version 2 run

State at 23:30 on 5 October 2026 (Singapore).

## Done

- Version 1 report: `output/pdf/final_report_v1.pdf` (source `docs/report_v1/`).
- Version 2 code (`latency_arb/v2/`) and tests: `tests/test_v2.py`, 12 passing.
- Nine PPO runs, trained and frozen: six at 150 ms in `runs/v2-leadlag-ppo/`, three at 450 ms in `runs/v2-leadlag-ppo-lat450/`.
- Validation selection for both groups, hash-frozen in `selection.json` and `selection_extra.json`.
- Headline simulator results for both groups on 29 Sep-2 Oct and 3 Oct, in `runs/*/sim/`.
- Protocol and amendments 1-4: `runs/v2-leadlag-ppo/protocol*.json`.
- Design notes and protocol history: `docs/V2_LEADLAG.md`.
- Training-day signal analysis and sub-second analysis: `docs/report_v2/analysis/`.
- ONNX export of the selected 150 ms actor: `runs/v2-leadlag-ppo/export/`.

## Still to run

PPO experiments finished at 13:26 (`runs/v2_driver.log` ends with `ALL_DONE`).

Amendment 5 adds Double DQN, fitted-Q iteration (FQI) and a contextual-bandit ablation in the 450 ms group (`runs/v2-algos-lat450/`). The FQI and bandit models are trained. The pipeline that waits for DQN, then selects, evaluates and compares, is:

~~~bash
setsid nohup runs/v2_algos_driver.sh > runs/v2_algos_driver.log 2>&1 < /dev/null &
~~~

It is resumable, but if DQN training was interrupted, retrain the missing seeds first with `latency_arb/v2/train_dqn.py`; see `runs/v2-algos-lat450/dqn/*/settings.json`. It is finished when `runs/v2_algos_driver.log` ends with `ALGOS_DONE`.

Then write the version 2 report (`docs/report_v2/main.tex`, `output/pdf/final_report_v2.pdf`) from:

- `docs/V2_LEADLAG.md`, including the formal SMDP model and the algorithm table;
- `runs/*/stats.json`;
- `docs/report_v2/generated/results.json`;
- `docs/report_v2/analysis/`.

Use `docs/report_v1/main.tex` as the template.

## Memory

The machine has 7 GB. Do not raise worker counts above 8, because each worker holds one table of about 120 MB plus a PPO model. The two WSL crashes on 5 October happened while 14 workers each cached several tables.

## Confirmatory data

The 29 Sep-2 Oct and 3 Oct test sets were opened by earlier attempts. Clean confirmation needs days from 4 October onward.

1. Export them with the existing exporter, `runs/data_inventory/export_research_data.py`, after editing its date range and adding the new days to the split mapping, as `runs/v2_prepare_data.py` does.
2. Prepare the days:

   ~~~bash
   .venv/bin/python runs/v2_prepare_data.py data/v2-fresh-2026-10-04 2026-10-04 2026-10-04
   ~~~

3. Evaluate the frozen choices. Nothing is retrained or reselected:

   ~~~bash
   .venv/bin/python -m latency_arb.v2.confirm --manifest data/v2-fresh-2026-10-04/manifest.json --label oct04
   ~~~
