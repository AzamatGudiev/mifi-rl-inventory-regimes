"""Эталонные политики. Политика получает батч наблюдений и список сред и возвращает батч действий (уровней S)."""
import numpy as np

from .env import SHOCK


class RandomPolicy:
    """Случайный целевой уровень на каждом шаге."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs, envs):
        return self.rng.integers(0, envs[0].max_level + 1, size=len(envs))


class BaseStockPolicy:
    """Заказ до постоянного уровня S (base-stock)."""

    def __init__(self, level: int):
        self.level = level

    def __call__(self, obs, envs):
        return np.full(len(envs), self.level)


class SignalBaseStockPolicy:
    """Два уровня S: для шока, если признак режима (вероятность шока) выше порога, иначе для спокойного режима.

    Пользуется тем же признаком, что и агент с признаком режима, - эвристический аналог экспериментального условия.
    """

    def __init__(self, level_calm: int, level_shock: int, threshold: float):
        self.levels = (level_calm, level_shock)
        self.threshold = threshold

    def __call__(self, obs, envs):
        return np.array([self.levels[int(e.shock_probability() > self.threshold)] for e in envs])


class OracleBaseStockPolicy:
    """Два уровня S по истинному текущему режиму. Агентам режим недоступен: это верхний ориентир."""

    def __init__(self, level_calm: int, level_shock: int):
        self.levels = (level_calm, level_shock)

    def __call__(self, obs, envs):
        return np.array([self.levels[int(e.current_regime == SHOCK)] for e in envs])


class SB3Policy:
    """Обученная модель SB3, жадное (детерминированное) действие."""

    def __init__(self, model):
        self.model = model

    def __call__(self, obs, envs):
        return self.model.predict(obs, deterministic=True)[0]
