"""Итоговая оценка всех политик на тестовых сценариях (после train.py и tune_baselines.py).

    python evaluate.py

Результат: results/test_metrics.csv, строка = политика x seed обучения x сценарий,
метрики усреднены по 200 тестовым эпизодам (одни и те же эпизоды для всех политик).
"""
import json
import os
import re
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import pandas as pd
import torch
from stable_baselines3 import PPO

from inventory_rl import config
from inventory_rl.evaluation import rollout, summarize
from inventory_rl.policies import (
    BaseStockPolicy, OracleBaseStockPolicy, RandomPolicy, SB3Policy, SignalBaseStockPolicy,
)


def load_models() -> list:
    """Модели из models/: имя файла {условие}{тег}_s{seed}.zip."""
    models = []
    for path in sorted(Path("models").glob("*.zip")):
        cond, tag, seed = re.fullmatch(r"(base|regime)(_long)?_s(\d+)", path.stem).groups()
        models.append((cond + (tag or ""), int(seed), PPO.load(path, device="cpu"), config.CONDITIONS[cond]))
    return models


def main():
    torch.set_num_threads(1)
    params = json.loads(Path("results/baselines.json").read_text())
    models = load_models()
    rows = []
    for scenario in config.TEST_SCENARIOS:
        heuristics = {
            "random": RandomPolicy(seed=0),
            "base_stock_calm": BaseStockPolicy(*params["base_stock_calm"]),
            "base_stock_static": BaseStockPolicy(*params["base_stock_static"]),
            "base_stock_signal": SignalBaseStockPolicy(*params["base_stock_signal"]),
            "oracle": OracleBaseStockPolicy(*params["oracle"]),
        }
        runs = [(name, None, policy, {}) for name, policy in heuristics.items()]
        runs += [(name, seed, SB3Policy(model), env_kwargs) for name, seed, model, env_kwargs in models]
        for name, seed, policy, env_kwargs in runs:
            metrics = summarize(rollout(policy, config.TEST_SEEDS, scenario=scenario, **env_kwargs))
            rows.append({"policy": name, "seed": seed, "scenario": scenario.name, **metrics})
        print("готов сценарий", scenario.name, flush=True)
    pd.DataFrame(rows).to_csv("results/test_metrics.csv", index=False)


if __name__ == "__main__":
    main()
