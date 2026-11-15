"""Подбор уровней эвристик (base-stock) перебором по сетке на эпизодах TUNE_SEEDS.

    python tune_baselines.py

Результат: results/baselines.json.
"""
import itertools
import json
from pathlib import Path

from inventory_rl import config
from inventory_rl.evaluation import rollout, summarize
from inventory_rl.policies import BaseStockPolicy, OracleBaseStockPolicy, SignalBaseStockPolicy


def grid_search(make_policy, grid, scenario) -> tuple:
    """Полный перебор параметров по средней стоимости эпизода."""
    costs = {params: summarize(rollout(make_policy(*params), config.TUNE_SEEDS, scenario=scenario))["cost"]
             for params in grid}
    return min(costs, key=costs.get)


def main():
    train = config.TRAIN_SCENARIO
    levels = [(s,) for s in range(20, 61)]
    params = {
        "base_stock_calm": grid_search(BaseStockPolicy, levels, config.CALM_SCENARIO),
        "base_stock_static": grid_search(BaseStockPolicy, levels, train),
        "oracle": grid_search(OracleBaseStockPolicy, itertools.product(range(28, 46), range(40, 66)), train),
    }
    # Для трёх параметров полный перебор с шагом 1 долгий: грубая сетка, затем уточнение вокруг лучшей точки.
    calm, shock, thr = grid_search(
        SignalBaseStockPolicy, itertools.product(range(28, 48, 3), range(38, 70, 4), (0.2, 0.35, 0.5, 0.65)), train)
    fine_grid = itertools.product(
        range(calm - 2, calm + 3), range(shock - 3, shock + 4), (round(thr - 0.1, 2), thr, round(thr + 0.1, 2)))
    params["base_stock_signal"] = grid_search(SignalBaseStockPolicy, fine_grid, train)

    Path("results").mkdir(exist_ok=True)
    Path("results/baselines.json").write_text(json.dumps(params, indent=2))
    print(params)


if __name__ == "__main__":
    main()
