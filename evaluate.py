"""Подбор эвристик на отдельных seed и итоговая оценка всех политик на тестовых сценариях.

    python evaluate.py

Результаты: results/baselines.json (параметры эвристик), results/test_metrics.csv
(строка = политика x seed обучения x сценарий; метрики усреднены по 200 тестовым эпизодам).
"""
import itertools
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


def grid_search(make_policy, grid, scenario) -> tuple:
    """Полный перебор параметров эвристики по средней стоимости на эпизодах TUNE_SEEDS."""
    costs = {params: summarize(rollout(make_policy(*params), config.TUNE_SEEDS, scenario=scenario))["cost"]
             for params in grid}
    return min(costs, key=costs.get)


def tune_baselines() -> dict:
    train = config.TRAIN_SCENARIO
    levels = [(s,) for s in range(20, 61)]
    params = {
        "base_stock_calm": grid_search(BaseStockPolicy, levels, config.CALM_SCENARIO),
        "base_stock_static": grid_search(BaseStockPolicy, levels, train),
        "oracle": grid_search(OracleBaseStockPolicy, itertools.product(range(25, 50), range(35, 75)), train),
    }
    # Для трёх параметров полный перебор с шагом 1 долгий: грубая сетка, затем уточнение вокруг лучшей точки.
    calm, shock, thr = grid_search(
        SignalBaseStockPolicy, itertools.product(range(28, 48, 3), range(38, 70, 4), (0.2, 0.35, 0.5, 0.65)), train)
    thresholds = (round(thr - 0.1, 2), thr, round(thr + 0.1, 2))
    params["base_stock_signal"] = grid_search(
        SignalBaseStockPolicy, itertools.product(range(calm - 2, calm + 3), range(shock - 3, shock + 4), thresholds), train)
    return params


def main():
    torch.set_num_threads(1)
    params = tune_baselines()
    Path("results/baselines.json").write_text(json.dumps(params, indent=2))
    print("эвристики:", params)

    models = []
    for path in sorted(Path("models").glob("*.zip")):
        cond, tag, seed = re.fullmatch(r"(base|regime)(_long)?_s(\d+)", path.stem).groups()
        models.append((cond + (tag or ""), int(seed), PPO.load(path, device="cpu"), config.CONDITIONS[cond]))

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
