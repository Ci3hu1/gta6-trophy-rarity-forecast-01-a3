# GTA VI trophy earn-rate forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Ci3hu1/gta6-trophy-rarity-forecast-01-a3/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Ci3hu1/gta6-trophy-rarity-forecast-01-a3/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)


A deterministic, standard-library Python forecaster for the make.fun GTA VI trophy-rarity task. It reads all case files from `MAKEFUN_INPUT_DIR` (default `/input`) and writes matching `<id>.out` files under `MAKEFUN_OUTPUT_DIR` (default `/output`).

The model treats unchanged Exophase snapshots as one observation, predicts trophy-earner counts rather than extrapolating percentages directly, shrinks noisy per-trophy trends toward a common lifecycle trend, and emits five ordered logit-scale quantiles.

Run locally with practice data:

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=/path/to/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

No network, third-party packages, randomness, or runtime model calls are used.

---

<sub>Entered on [make.fun](https://make.fun/?ref=cc7tcnw6&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
