"""Склад одного товара: спрос переключается между спокойным и шоковым режимом (марковская цепь)."""
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

CALM, SHOCK = 0, 1

# Масштабы для нормировки наблюдения: PPO заметно хуже учится на признаках порядка десятков.
INV_SCALE, DEMAND_SCALE = 50.0, 20.0


@dataclass(frozen=True)
class DemandRegimes:
    """Параметры двух режимов спроса и матрицы переходов. Значения по умолчанию = обучающий процесс."""
    calm_mean: float = 10.0
    calm_std: float = 2.0
    shock_mean: float = 13.0
    shock_std: float = 6.0
    p_calm_to_shock: float = 0.04  # средняя длина спокойного периода 25 шагов
    p_shock_to_calm: float = 0.10  # средняя длина шока 10 шагов

    def stationary_shock_share(self) -> float:
        return self.p_calm_to_shock / (self.p_calm_to_shock + self.p_shock_to_calm)


TRAIN_REGIMES = DemandRegimes()


@dataclass(frozen=True)
class Scenario:
    """Набор начальных состояний и динамика спроса, на которых запускаются эпизоды."""
    name: str
    regimes: DemandRegimes = field(default_factory=DemandRegimes)
    init_inventory: tuple[int, int] = (20, 60)  # начальный запас ~ U{a..b}
    start_regime: int | None = None  # None: режим в начале берётся из стационарного распределения


@dataclass(frozen=True)
class Costs:
    holding: float = 1.0  # за единицу на складе в конце шага
    lost_sale: float = 20.0  # за единицу неудовлетворённого спроса (упущенная маржа + потеря лояльности)
    order: float = 1.0  # закупочная цена единицы


class InventoryRegimeEnv(gym.Env):
    """Периодический пересмотр запаса, потерянные продажи, фиксированный срок поставки.

    Действие - целевой уровень позиции запаса S (склад + заказы в пути); заказ q = max(0, S - позиция).
    Шаг: заказ q уходит поставщику -> приходит заказ, сделанный lead_time шагов назад ->
    реализуется спрос (неудовлетворённый спрос теряется) -> начисляется стоимость.
    Награда = -(хранение + штраф за потерянные продажи + закупка).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: Scenario = Scenario("train"),
        regime_feature: bool = False,
        history_len: int = 4,
        horizon: int = 100,
        lead_time: int = 2,
        max_level: int = 80,
        costs: Costs = Costs(),
    ):
        self.scenario = scenario
        self.regime_feature = regime_feature
        self.history_len = history_len
        self.horizon = horizon
        self.lead_time = lead_time
        self.max_level = max_level
        self.costs = costs

        self.action_space = spaces.Discrete(max_level + 1)
        obs_dim = 1 + lead_time + history_len + int(regime_feature)
        self.observation_space = spaces.Box(0.0, np.inf, shape=(obs_dim,), dtype=np.float32)

    def _sample_paths(self) -> None:
        """Режимы и спрос не зависят от действий, поэтому весь путь генерируется при reset.

        Так при одинаковом seed все политики видят один и тот же спрос (общие случайные числа).
        Первые history_len значений - предыстория до t=0, чтобы наблюдение было определено с первого шага.
        """
        r = self.scenario.regimes
        burn_in = self.history_len
        n = burn_in + self.horizon
        rng = self.np_random
        regimes = np.empty(n, dtype=np.int64)
        if self.scenario.start_regime is None:
            regimes[0] = int(rng.random() < r.stationary_shock_share())
            first = 0
        else:
            first = burn_in  # заданный режим держится всю предысторию и на t=0
            regimes[: first + 1] = self.scenario.start_regime
        for i in range(first + 1, n):
            p_switch = r.p_calm_to_shock if regimes[i - 1] == CALM else r.p_shock_to_calm
            regimes[i] = 1 - regimes[i - 1] if rng.random() < p_switch else regimes[i - 1]
        mean = np.where(regimes == SHOCK, r.shock_mean, r.calm_mean)
        std = np.where(regimes == SHOCK, r.shock_std, r.calm_std)
        self.regimes = regimes
        self.demands = np.maximum(0, np.rint(rng.normal(mean, std))).astype(np.int64)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._sample_paths()
        lo, hi = self.scenario.init_inventory
        self.inventory = int(self.np_random.integers(lo, hi + 1))
        self.pipeline = [0] * self.lead_time  # pipeline[0] придёт на ближайшем шаге
        self.t = 0
        return self._obs(), {}

    def past_demands(self) -> np.ndarray:
        """Последние history_len наблюдённых значений спроса (до текущего шага)."""
        now = self.history_len + self.t
        return self.demands[now - self.history_len: now]

    def shock_probability(self) -> float:
        """Признак режима: вероятность шока на текущем шаге по тем же значениям спроса, что есть в наблюдении.

        Фильтр Гамильтона (прямой проход скрытой марковской модели) с параметрами обучающего процесса.
        Признак - детерминированная функция наблюдения, новой информации он агенту не даёт.
        Считается в лог-шансах, чтобы не было переполнения на больших отклонениях спроса.
        """
        m = TRAIN_REGIMES
        p = m.stationary_shock_share()
        for d in self.past_demands():
            log_lr = (np.log(m.calm_std / m.shock_std)
                      - 0.5 * ((d - m.shock_mean) / m.shock_std) ** 2
                      + 0.5 * ((d - m.calm_mean) / m.calm_std) ** 2)
            post = 1.0 / (1.0 + np.exp(-(np.log(p / (1 - p)) + log_lr)))
            p = post * (1 - m.p_shock_to_calm) + (1 - post) * m.p_calm_to_shock
        return float(p)

    def inventory_position(self) -> int:
        return self.inventory + sum(self.pipeline)

    @property
    def current_regime(self) -> int:
        return int(self.regimes[self.history_len + self.t])

    def _obs(self) -> np.ndarray:
        parts = [
            [self.inventory / INV_SCALE],
            np.asarray(self.pipeline, dtype=float) / INV_SCALE,
            self.past_demands() / DEMAND_SCALE,
        ]
        if self.regime_feature:
            parts.append([self.shock_probability()])
        return np.concatenate(parts).astype(np.float32)

    def step(self, action):
        # Все S не выше текущей позиции дают один и тот же нулевой заказ, поэтому для анализа
        # записывается фактическая позиция после заказа max(S, позиция), а не само действие.
        order = max(0, int(action) - self.inventory_position())
        position = self.inventory_position() + order
        regime = self.current_regime
        demand = int(self.demands[self.history_len + self.t])

        self.pipeline.append(order)
        self.inventory += self.pipeline.pop(0)
        sold = min(self.inventory, demand)
        lost = demand - sold
        self.inventory -= sold

        holding = self.costs.holding * self.inventory
        penalty = self.costs.lost_sale * lost
        purchase = self.costs.order * order
        cost = holding + penalty + purchase

        self.t += 1
        truncated = self.t >= self.horizon  # конец эпизода только по горизонту, терминальных состояний нет
        info = {
            "cost": cost, "holding": holding, "penalty": penalty, "purchase": purchase, "demand": demand,
            "lost": lost, "inventory": self.inventory, "order": order, "position": position, "regime": regime,
        }
        return self._obs(), -cost, False, truncated, info
