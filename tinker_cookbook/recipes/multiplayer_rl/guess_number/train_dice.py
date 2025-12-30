"""Train a guess-the-number policy with a discriminator-shaped reward.

This script mirrors the DICE-style PPO flow from the `asaran/llm-dice` repo's
`ppo_dice_single_turn.py`, adapted to the guess-number multiplayer RL task.
The discriminator is a lightweight torch model that learns to spot high-quality
(closer-to-correct) guesses and supplies an auxiliary reward term.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

import chz
import torch
from torch import nn
from torch.optim import Adam

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.multiplayer_rl.guess_number.env import (
    GuessNumberDataset,
    GuessNumberDatasetBuilder,
    GuessNumberEnv,
    GuessNumberEnvGroupBuilder,
    _UPPER_BOUND,
)
from tinker_cookbook.rl import train
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.renderers import Renderer, get_renderer


def _normalize(value: float) -> float:
    return value / float(_UPPER_BOUND)


class GuessNumberDiscriminator(nn.Module):
    """Tiny discriminator to judge guess quality.

    Mirrors the single-turn DICE discriminator: it is trained with a BCE loss and
    its logit becomes the shaping reward (log-odds of the guess being correct).
    """

    def __init__(self, learning_rate: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 1))
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = Adam(self.parameters(), lr=learning_rate)

    def forward(self, guess: torch.Tensor, abs_error: torch.Tensor) -> torch.Tensor:
        features = torch.stack([guess, abs_error], dim=-1)
        return self.net(features).squeeze(-1)

    def logit_and_prob(self, guess: int, answer: int) -> tuple[float, float]:
        guess_tensor = torch.tensor(_normalize(float(guess)))
        error_tensor = torch.tensor(_normalize(abs(float(guess) - float(answer))))
        with torch.no_grad():
            logit = self.forward(guess_tensor, error_tensor).item()
        prob = float(torch.sigmoid(torch.tensor(logit)))
        return logit, prob

    def train_on_batch(
        self, guesses: list[int], answers: list[int], labels: list[float], num_steps: int
    ) -> tuple[float, float]:
        if not guesses:
            return 0.0, 0.0

        guess_tensor = torch.tensor([_normalize(float(g)) for g in guesses], dtype=torch.float32)
        error_tensor = torch.tensor(
            [_normalize(abs(float(g) - float(a))) for g, a in zip(guesses, answers, strict=True)],
            dtype=torch.float32,
        )
        label_tensor = torch.tensor(labels, dtype=torch.float32)

        total_loss = 0.0
        total_correct = 0.0
        for _ in range(num_steps):
            logits = self.forward(guess_tensor, error_tensor)
            loss = self.loss_fn(logits, label_tensor)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            total_correct += float((preds == label_tensor).float().mean())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        num_steps_f = float(num_steps)
        return total_loss / num_steps_f, total_correct / num_steps_f


class DiceGuessNumberEnv(GuessNumberEnv):
    """GuessNumberEnv that records parsed guesses for discriminator training."""

    def __init__(self, gold_answer: int, renderer: Renderer):
        super().__init__(gold_answer=gold_answer, renderer=renderer)

    async def step(self, action):
        step_result = await super().step(action)

        # Re-parse the assistant guess that produced this transition so downstream
        # reward shaping doesn't depend on turn order assumptions.
        action_message, _ = self.renderer.parse_response(action)
        guess_value = None
        if isinstance(action_message["content"], str) and action_message["content"].startswith("Guess: "):
            try:
                guess_value = int(action_message["content"].split("Guess: ")[1])
            except ValueError:
                guess_value = None

        if guess_value is not None:
            step_result.metrics["guess_value"] = guess_value
            step_result.metrics["answer"] = self.gold_answer
        return step_result


@dataclass(frozen=True)
class DiceGuessNumberEnvGroupBuilder(GuessNumberEnvGroupBuilder):
    discriminator: GuessNumberDiscriminator
    reward_scale: float
    discriminator_steps: int

    async def make_envs(self) -> Sequence[Env]:
        return [DiceGuessNumberEnv(self.answer, self.renderer) for _ in range(self.num_envs)]

    async def compute_group_rewards(
        self, trajectory_group, env_group: Sequence[Env]
    ) -> list[tuple[float, dict[str, float]]]:
        guesses: list[int] = []
        answers: list[int] = []
        labels: list[float] = []

        for env, traj in zip(env_group, trajectory_group, strict=True):
            assert isinstance(env, GuessNumberEnv)
            for transition in traj.transitions:
                guess = transition.metrics.get("guess_value")
                answer = transition.metrics.get("answer")
                if guess is None or answer is None:
                    continue
                guesses.append(int(guess))
                answers.append(int(answer))
                labels.append(1.0 if guess == answer else 0.0)

        disc_loss, disc_acc = self.discriminator.train_on_batch(
            guesses=guesses, answers=answers, labels=labels, num_steps=self.discriminator_steps
        )

        rewards_and_metrics: list[tuple[float, dict[str, float]]] = []
        for env, traj in zip(env_group, trajectory_group, strict=True):
            assert isinstance(env, GuessNumberEnv)
            disc_reward = 0.0
            last_prob = 0.0
            for transition in traj.transitions:
                guess = transition.metrics.get("guess_value")
                answer = transition.metrics.get("answer")
                if guess is None or answer is None:
                    continue
                logit, prob = self.discriminator.logit_and_prob(int(guess), int(answer))
                disc_reward += self.reward_scale * logit
                last_prob = prob
            rewards_and_metrics.append(
                (
                    disc_reward,
                    {
                        "disc_reward": disc_reward,
                        "disc_loss": disc_loss,
                        "disc_acc": disc_acc,
                        "disc_prob": last_prob,
                    },
                )
            )

        return rewards_and_metrics


@dataclass(frozen=True)
class DiceGuessNumberDataset(GuessNumberDataset):
    discriminator: GuessNumberDiscriminator
    reward_scale: float
    discriminator_steps: int

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        return [
            DiceGuessNumberEnvGroupBuilder(
                answer=self.answers[index * self.batch_size + i],
                renderer=self.renderer,
                num_envs=self.group_size,
                discriminator=self.discriminator,
                reward_scale=self.reward_scale,
                discriminator_steps=self.discriminator_steps,
            )
            for i in range(self.batch_size)
        ]


@chz.chz
class DiceGuessNumberDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    renderer_name: str
    train_group_size: int
    model_name: str
    discriminator: GuessNumberDiscriminator
    reward_scale: float
    discriminator_steps: int = 1
    train_fraction: float = 0.9
    test_group_size: int = 4

    async def __call__(self) -> tuple[RLDataset, RLDataset]:
        renderer = get_renderer(self.renderer_name, get_tokenizer(self.model_name))
        train_numbers, test_numbers = self._get_train_and_test_numbers()
        assert self.batch_size <= len(train_numbers)

        training_dataset = DiceGuessNumberDataset(
            answers=train_numbers,
            renderer=renderer,
            batch_size=self.batch_size,
            group_size=self.train_group_size,
            discriminator=self.discriminator,
            reward_scale=self.reward_scale,
            discriminator_steps=self.discriminator_steps,
        )
        test_dataset = DiceGuessNumberDataset(
            answers=test_numbers,
            renderer=renderer,
            batch_size=len(test_numbers),
            group_size=self.test_group_size,
            discriminator=self.discriminator,
            reward_scale=self.reward_scale,
            discriminator_steps=self.discriminator_steps,
        )
        return training_dataset, test_dataset

    def _get_train_and_test_numbers(self) -> tuple[list[int], list[int]]:
        base_builder = GuessNumberDatasetBuilder(
            batch_size=self.batch_size,
            renderer_name=self.renderer_name,
            train_group_size=self.train_group_size,
            model_name=self.model_name,
            train_fraction=self.train_fraction,
            test_group_size=self.test_group_size,
        )
        return base_builder._get_train_and_test_numbers()


@chz.chz
class CLIConfig:
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    renderer_name: str | None = None
    group_size: int = 8
    batch_size: int = 32
    learning_rate: float = 3e-5
    discriminator_learning_rate: float = 1e-3
    discriminator_steps: int = 2
    reward_scale: float = 0.5
    max_tokens: int = 64
    eval_every: int = 5
    save_every: int = 20
    wandb_project: str | None = None
    wandb_name: str | None = None
    log_path: str | None = None


def build_config(cli_config: CLIConfig) -> train.Config:
    model_name = cli_config.model_name
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )

    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = (
        f"{model_name}-dice-{cli_config.group_size}group-"
        f"{cli_config.batch_size}batch-{cli_config.learning_rate}lr-{date_and_time}"
    )

    log_path = cli_config.log_path or f"/tmp/tinker-examples/guess-number-dice/{run_name}"
    wandb_name = cli_config.wandb_name or run_name

    discriminator = GuessNumberDiscriminator(cli_config.discriminator_learning_rate)

    dataset_builder = DiceGuessNumberDatasetBuilder(
        batch_size=cli_config.batch_size,
        model_name=model_name,
        renderer_name=renderer_name,
        train_group_size=cli_config.group_size,
        discriminator=discriminator,
        reward_scale=cli_config.reward_scale,
        discriminator_steps=cli_config.discriminator_steps,
    )

    return train.Config(
        model_name=model_name,
        log_path=log_path,
        dataset_builder=dataset_builder,
        learning_rate=cli_config.learning_rate,
        max_tokens=cli_config.max_tokens,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        loss_fn="ppo",
    )


def main():
    cli_cfg = chz.entrypoint(CLIConfig)
    cfg = build_config(cli_cfg)
    cli_utils.check_log_dir(cfg.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(cfg))


if __name__ == "__main__":
    main()
