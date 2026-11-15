"""Обучение PPO в обоих условиях.

    python train.py                 # основной эксперимент: 2 условия x 5 seed x 500k шагов
    python train.py --long          # проверка на бюджете 1.5M шагов: 2 условия x 3 seed
    python train.py --jobs 4        # несколько запусков параллельно
"""
import argparse
import os
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")

import pandas as pd
import torch
from gymnasium.wrappers import TransformReward
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from inventory_rl import config
from inventory_rl.env import InventoryRegimeEnv
from inventory_rl.evaluation import rollout, summarize
from inventory_rl.policies import SB3Policy


class LearningCurve(BaseCallback):
    """Каждые EVAL_EVERY шагов оценивает жадную политику на фиксированных валидационных эпизодах."""

    def __init__(self, env_kwargs):
        super().__init__()
        self.env_kwargs = env_kwargs
        self.rows = []
        self.start = time.time()

    def _evaluate(self):
        metrics = summarize(rollout(SB3Policy(self.model), config.VAL_SEEDS, **self.env_kwargs))
        self.rows.append({"timesteps": self.num_timesteps, "elapsed_s": time.time() - self.start, **metrics})

    def _on_training_start(self):
        self._evaluate()  # точка 0 = необученная политика

    def _on_step(self):
        if self.num_timesteps - self.rows[-1]["timesteps"] >= config.EVAL_EVERY:
            self._evaluate()
        return True

    def _on_training_end(self):
        if self.rows[-1]["timesteps"] != self.num_timesteps:
            self._evaluate()


def train_one(condition: str, seed: int, timesteps: int, tag: str) -> str:
    torch.set_num_threads(1)
    env_kwargs = dict(scenario=config.TRAIN_SCENARIO, **config.CONDITIONS[condition])
    venv = make_vec_env(
        lambda: TransformReward(InventoryRegimeEnv(**env_kwargs), lambda r: r / config.REWARD_SCALE),
        n_envs=config.N_ENVS,
        seed=1000 * seed,
    )
    model = PPO("MlpPolicy", venv, seed=seed, device="cpu", verbose=0, **config.PPO_PARAMS)
    curve = LearningCurve(env_kwargs)
    model.learn(timesteps, callback=curve)

    name = f"{condition}{tag}_s{seed}"
    model.save(f"models/{name}")
    pd.DataFrame(curve.rows).to_csv(f"results/curves/{name}.csv", index=False)
    return f"{name}: {curve.rows[-1]['elapsed_s'] / 60:.1f} мин, val cost {curve.rows[-1]['cost']:.0f}"


def train_one_star(run: tuple) -> str:
    return train_one(*run)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--long", action="store_true", help="увеличенный бюджет (config.LONG)")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()

    setup = config.LONG if args.long else config.MAIN
    os.makedirs("models", exist_ok=True)
    os.makedirs("results/curves", exist_ok=True)
    runs = [(cond, seed, setup["timesteps"], setup["tag"]) for seed in setup["seeds"] for cond in config.CONDITIONS]
    with Pool(args.jobs) as pool:
        for line in pool.imap_unordered(train_one_star, runs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
