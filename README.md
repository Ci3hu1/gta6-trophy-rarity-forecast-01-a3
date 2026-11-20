# GTA VI trophy earn-rate forecast

A deterministic, standard-library Python forecaster for the make.fun GTA VI trophy-rarity task. It reads all case files from `MAKEFUN_INPUT_DIR` (default `/input`) and writes matching `<id>.out` files under `MAKEFUN_OUTPUT_DIR` (default `/output`).

The model treats unchanged Exophase snapshots as one observation, predicts trophy-earner counts rather than extrapolating percentages directly, shrinks noisy per-trophy trends toward a common lifecycle trend, and emits five ordered logit-scale quantiles.

Run locally with practice data:

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=/path/to/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

No network, third-party packages, randomness, or runtime model calls are used.
