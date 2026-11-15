"""Пакетный прогон политики на фиксированных seed и метрики по эпизодам."""
import numpy as np

from .env import SHOCK, InventoryRegimeEnv

TRACE_KEYS = ("cost", "holding", "penalty", "purchase", "demand", "lost", "inventory", "order", "position", "regime")


def rollout(policy, seeds, **env_kwargs) -> dict[str, np.ndarray]:
    """По одному эпизоду на каждый seed, все эпизоды идут параллельно. Возвращает массивы [эпизод, шаг]."""
    envs = [InventoryRegimeEnv(**env_kwargs) for _ in seeds]
    obs = np.stack([env.reset(seed=int(s))[0] for env, s in zip(envs, seeds)])
    horizon = envs[0].horizon
    trace = {k: np.zeros((len(envs), horizon)) for k in TRACE_KEYS}
    for t in range(horizon):
        actions = policy(obs, envs)
        steps = [env.step(a) for env, a in zip(envs, actions)]
        obs = np.stack([s[0] for s in steps])
        for k in TRACE_KEYS:
            trace[k][:, t] = [s[4][k] for s in steps]
    return trace


def _mean(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else float("nan")  # в спокойном сценарии шоковых шагов нет


def summarize(trace: dict[str, np.ndarray]) -> dict[str, float]:
    """Метрики по набору эпизодов. Метрики по режимам считаются по всем шагам этого режима вместе."""
    shock = trace["regime"] == SHOCK
    stockout = trace["lost"] > 0
    return {
        "cost": trace["cost"].sum(1).mean(),
        "holding": trace["holding"].sum(1).mean(),
        "penalty": trace["penalty"].sum(1).mean(),
        "purchase": trace["purchase"].sum(1).mean(),
        "stockout_rate": stockout.mean(),
        "lost_share": trace["lost"].sum() / trace["demand"].sum(),
        "position_calm": _mean(trace["position"][~shock]),
        "position_shock": _mean(trace["position"][shock]),
        "inv_calm": _mean(trace["inventory"][~shock]),
        "inv_shock": _mean(trace["inventory"][shock]),
        "stockout_calm": _mean(stockout[~shock]),
        "stockout_shock": _mean(stockout[shock]),
        "shock_share": shock.mean(),
    }
