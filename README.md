# catan-marl

Multi-agent RL on Settlers of Catan. Every player is an agent we train. Rules
come from [Catanatron](https://github.com/bcollazo/catanatron) (unmodified,
pinned); the RL side uses [TorchRL](https://github.com/pytorch/rl).

| File | What's in it |
|---|---|
| `catan_env.py` | Catan logic (`CatanCore`), the TorchRL env (`CatanEnv`), robber stats. `python catan_env.py` prints specs and benchmarks speed. |
| `evaluate.py` | Play a policy against Catanatron's bots: win rates, robber-on-leader stats. |
| `test_catan_env.py` | Tests for 3 and 4 players, mini and full board. |

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build is enough
pip install -e ".[dev]"
pytest -q                                                            # 14 passed
```

## Use

```python
from catan_env import CatanEnv, CatanConfig, OBS, ACTION_MASK, ACTIVE

env = CatanEnv(CatanConfig(num_players=4, map_type="MINI", vps_to_win=6), seed=0)
td = env.reset()
td[OBS].shape          # [4, 444]  each player's own view
td[ACTION_MASK].shape  # [4, 208]  legal moves
td[ACTIVE]             # whose turn it is
```

```
python runeval.py checkpoints/iter_0049.pt --games 100
```
Catanatron is GPL-3.0: if this repo is public, license it GPL-3.0 too.
