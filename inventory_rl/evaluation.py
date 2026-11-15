"""Пакетный прогон политики на фиксированных seed и метрики по эпизодам."""
import numpy as np

from .env import SHOCK, InventoryRegimeEnv

TRACE_KEYS = ("cost", "holding", "penalty", "purchase", "demand", "lost", "inventory", "order", "level", "regime")


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
        "level_calm": trace["level"][~shock].mean(),
        "level_shock": trace["level"][shock].mean(),
        "inv_calm": trace["inventory"][~shock].mean(),
        "inv_shock": trace["inventory"][shock].mean(),
        "stockout_calm": stockout[~shock].mean(),
        "stockout_shock": stockout[shock].mean(),
        "shock_share": shock.mean(),
    }
