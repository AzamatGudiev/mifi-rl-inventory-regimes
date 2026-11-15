"""Общие настройки эксперимента: условия, seed, сценарии оценки, гиперпараметры PPO."""
from dataclasses import replace

from .env import CALM, SHOCK, TRAIN_REGIMES, Scenario

# Условия отличаются только признаком режима в наблюдении; всё остальное общее.
CONDITIONS = {
    "base": dict(regime_feature=False),
    "regime": dict(regime_feature=True),
}

# Основной эксперимент и проверка на увеличенном бюджете (тот же код, другой бюджет и тег).
MAIN = dict(tag="", timesteps=500_000, seeds=[0, 1, 2, 3, 4])
LONG = dict(tag="_long", timesteps=1_500_000, seeds=[0, 1, 2])

# Непересекающиеся наборы seed эпизодов. Обучающие среды получают seed 1000*k..1000*k+7.
VAL_SEEDS = range(50_000, 50_050)  # только для кривых обучения, выбора модели по ним нет
TUNE_SEEDS = range(60_000, 60_300)  # подбор уровней эвристик
TEST_SEEDS = range(100_000, 100_200)  # итоговая оценка

N_ENVS = 8
REWARD_SCALE = 20.0  # награда делится на константу только при обучении: оптимальная политика не меняется
EVAL_EVERY = 20_000
PPO_PARAMS = dict(
    n_steps=256,
    batch_size=256,
    n_epochs=10,
    learning_rate=3e-4,
    gamma=0.95,  # горизонт ~20 шагов: больше срока поставки (2) и средней длины шока (10)
    gae_lambda=0.95,
    ent_coef=0.01,
    policy_kwargs=dict(net_arch=[64, 64]),
)

TRAIN_SCENARIO = Scenario("train")
CALM_SCENARIO = Scenario("calm_only", regimes=replace(TRAIN_REGIMES, p_calm_to_shock=0.0), start_regime=CALM)

TEST_SCENARIOS = [
    Scenario("base"),  # то же распределение, что при обучении, новые seed
    Scenario("empty_in_shock", init_inventory=(0, 0), start_regime=SHOCK),
    Scenario("overstock", init_inventory=(100, 100), start_regime=CALM),
    Scenario("frequent_shocks", regimes=replace(TRAIN_REGIMES, p_calm_to_shock=0.12)),
    Scenario("long_shocks", regimes=replace(TRAIN_REGIMES, p_shock_to_calm=0.05)),
    Scenario("strong_shocks", regimes=replace(TRAIN_REGIMES, shock_std=9.0)),
]
