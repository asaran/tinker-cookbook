"""Train a guess-the-number policy with a discriminator-shaped reward.

This script mirrors the DICE-style PPO flow from the `asaran/llm-dice` repo's
`ppo_dice_single_turn.py`, adapted to the guess-number multiplayer RL task.
The discriminator is a lightweight torch model that learns to spot high-quality
(closer-to-correct) guesses and supplies an auxiliary reward term.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, Sequence

import chz
import torch
from torch import nn
from torch.optim import Adam
import tinker

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.renderers import Renderer, get_renderer
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
from tinker_cookbook.supervised.data import conversation_to_datum


def _normalize(value: float) -> float:
    return value / float(_UPPER_BOUND)


class GuessNumberDiscriminator(Protocol):
    async def logit_and_prob(self, guess: int, answer: int) -> tuple[float, float]:
        ...

    async def train_on_batch(
        self, guesses: list[int], answers: list[int], labels: list[float], num_steps: int
    ) -> tuple[float, float]:
        ...


class LocalGuessNumberDiscriminator(nn.Module):
    """Tiny discriminator to judge guess quality.

    Mirrors the single-turn DICE discriminator: it is trained with a BCE loss and
    its logit becomes the shaping reward (log-odds of the guess being correct).
    """

    def __init__(self, learning_rate: float, hidden_size: int = 256):
        super().__init__()
        # Match the llm-dice discriminator capacity (stacked MLP with GELU)
        self.net = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = Adam(self.parameters(), lr=learning_rate)

    def forward(self, guess: torch.Tensor, abs_error: torch.Tensor) -> torch.Tensor:
        features = torch.stack([guess, abs_error], dim=-1)
        return self.net(features).squeeze(-1)

    async def logit_and_prob(self, guess: int, answer: int) -> tuple[float, float]:
        guess_tensor = torch.tensor(_normalize(float(guess)))
        error_tensor = torch.tensor(_normalize(abs(float(guess) - float(answer))))
        with torch.no_grad():
            logit = self.forward(guess_tensor, error_tensor).item()
        prob = float(torch.sigmoid(torch.tensor(logit)))
        return logit, prob

    async def train_on_batch(
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


class TinkerGuessNumberDiscriminator:
    """Discriminator whose updates run on the Tinker API."""

    def __init__(
        self,
        training_client: tinker.TrainingClient,
        renderer: Renderer,
        learning_rate: float,
        max_tokens: int,
    ):
        self.training_client = training_client
        self.renderer = renderer
        self.learning_rate = learning_rate
        self.max_tokens = max_tokens
        self._sampling_client: tinker.SamplingClient | None = None

    @classmethod
    async def create(
        cls,
        model_name: str,
        renderer_name: str,
        learning_rate: float,
        max_tokens: int,
        lora_rank: int,
        base_url: str | None,
    ) -> "TinkerGuessNumberDiscriminator":
        service_client = tinker.ServiceClient(base_url=base_url)
        training_client = await service_client.create_lora_training_client_async(
            model_name, rank=lora_rank
        )
        renderer = get_renderer(renderer_name, training_client.get_tokenizer())
        return cls(
            training_client=training_client,
            renderer=renderer,
            learning_rate=learning_rate,
            max_tokens=max_tokens,
        )

    async def _ensure_sampling_client(self) -> tinker.SamplingClient:
        if self._sampling_client is None:
            self._sampling_client = await self.training_client.save_weights_and_get_sampling_client_async()
        return self._sampling_client

    async def _classify_messages(self, guess: int, answer: int) -> tuple[list[dict], str]:
        messages = [
            {
                "role": "system",
                "content": "Given a guess and answer, output CORRECT if they match, otherwise INCORRECT.",
            },
            {
                "role": "user",
                "content": f"Guess: {guess}. Answer: {answer}. Respond with CORRECT or INCORRECT.",
            },
        ]
        return messages, "CORRECT"

    async def logit_and_prob(self, guess: int, answer: int) -> tuple[float, float]:
        sampling_client = await self._ensure_sampling_client()
        messages, positive_target = await self._classify_messages(guess, answer)
        # Ask for a single-token verdict so we can read logprobs off the first token
        sample_result = await sampling_client.sample_async(
            messages=messages, max_tokens=1, logprobs=True, temperature=0.0
        )
        generation = sample_result.generations[0]
        if not generation.logprobs:
            return 0.0, 0.5
        logprobs = generation.logprobs[0].logprobs
        pos_logprob = logprobs.get(positive_target, logprobs.get(positive_target.lower(), float("-inf")))
        # Use log-sum-exp over observed candidates for a normalized probability
        denom = torch.logsumexp(
            torch.tensor([torch.tensor(v) for v in logprobs.values()], dtype=torch.float32), dim=0
        )
        logit = float(pos_logprob - denom)
        prob = float(torch.exp(torch.tensor(logit)))
        return logit, prob

    async def train_on_batch(
        self, guesses: list[int], answers: list[int], labels: list[float], num_steps: int
    ) -> tuple[float, float]:
        if not guesses:
            return 0.0, 0.0

        datums: list[tinker.Datum] = []
        for guess, answer, label in zip(guesses, answers, labels, strict=True):
            messages, _ = await self._classify_messages(guess, answer)
            target = "CORRECT" if label >= 0.5 else "INCORRECT"
            datum = conversation_to_datum(
                messages=messages
                + [
                    {
                        "role": "assistant",
                        "content": target,
                    }
                ],
                renderer=self.renderer,
                max_length=self.max_tokens,
            )
            datums.append(datum)

        total_loss = 0.0
        for _ in range(num_steps):
            fwd = await self.training_client.forward_backward_async(datums, loss_fn="cross_entropy")
            optim = await self.training_client.optim_step_async(
                tinker.AdamParams(learning_rate=self.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)
            )
            result = await fwd.result_async()
            await optim.result_async()
            total_loss += sum(float(loss["loss"].to_torch().item()) for loss in result.loss_fn_outputs) / max(
                1, len(result.loss_fn_outputs)
            )

        # Refresh sampling client after an update
        self._sampling_client = await self.training_client.save_weights_and_get_sampling_client_async()
        return total_loss / float(num_steps), 0.0


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

        disc_loss, disc_acc = await self.discriminator.train_on_batch(
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
                logit, prob = await self.discriminator.logit_and_prob(int(guess), int(answer))
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
    reward_scale: float
    discriminator_steps: int = 1
    train_fraction: float = 0.9
    test_group_size: int = 4
    discriminator_learning_rate: float = 1e-3
    discriminator_hidden_size: int = 256
    use_tinker_discriminator: bool = False
    discriminator_model_name: str | None = None
    discriminator_renderer_name: str | None = None
    discriminator_lora_rank: int = 32
    discriminator_max_tokens: int = 32
    base_url: str | None = None

    async def __call__(self) -> tuple[RLDataset, RLDataset]:
        renderer = get_renderer(self.renderer_name, get_tokenizer(self.model_name))
        train_numbers, test_numbers = self._get_train_and_test_numbers()
        assert self.batch_size <= len(train_numbers)

        if self.use_tinker_discriminator:
            disc_model_name = self.discriminator_model_name or self.model_name
            disc_renderer_name = self.discriminator_renderer_name or model_info.get_recommended_renderer_name(
                disc_model_name
            )
            discriminator = await TinkerGuessNumberDiscriminator.create(
                model_name=disc_model_name,
                renderer_name=disc_renderer_name,
                learning_rate=self.discriminator_learning_rate,
                max_tokens=self.discriminator_max_tokens,
                lora_rank=self.discriminator_lora_rank,
                base_url=self.base_url,
            )
        else:
            discriminator = LocalGuessNumberDiscriminator(
                learning_rate=self.discriminator_learning_rate,
                hidden_size=self.discriminator_hidden_size,
            )

        training_dataset = DiceGuessNumberDataset(
            answers=train_numbers,
            renderer=renderer,
            batch_size=self.batch_size,
            group_size=self.train_group_size,
            discriminator=discriminator,
            reward_scale=self.reward_scale,
            discriminator_steps=self.discriminator_steps,
        )
        test_dataset = DiceGuessNumberDataset(
            answers=test_numbers,
            renderer=renderer,
            batch_size=len(test_numbers),
            group_size=self.test_group_size,
            discriminator=discriminator,
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
    discriminator_hidden_size: int = 256
    discriminator_steps: int = 2
    reward_scale: float = 0.5
    max_tokens: int = 64
    base_url: str | None = None
    use_tinker_discriminator: bool = False
    discriminator_model_name: str | None = None
    discriminator_renderer_name: str | None = None
    discriminator_lora_rank: int = 32
    discriminator_max_tokens: int = 32
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

    dataset_builder = DiceGuessNumberDatasetBuilder(
        batch_size=cli_config.batch_size,
        model_name=model_name,
        renderer_name=renderer_name,
        train_group_size=cli_config.group_size,
        reward_scale=cli_config.reward_scale,
        discriminator_steps=cli_config.discriminator_steps,
        discriminator_learning_rate=cli_config.discriminator_learning_rate,
        discriminator_hidden_size=cli_config.discriminator_hidden_size,
        use_tinker_discriminator=cli_config.use_tinker_discriminator,
        discriminator_model_name=cli_config.discriminator_model_name,
        discriminator_renderer_name=cli_config.discriminator_renderer_name,
        discriminator_lora_rank=cli_config.discriminator_lora_rank,
        discriminator_max_tokens=cli_config.discriminator_max_tokens,
        base_url=cli_config.base_url,
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
        base_url=cli_config.base_url,
    )


def main():
    cli_cfg = chz.entrypoint(CLIConfig)
    cfg = build_config(cli_cfg)
    cli_utils.check_log_dir(cfg.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(cfg))


if __name__ == "__main__":
    main()
