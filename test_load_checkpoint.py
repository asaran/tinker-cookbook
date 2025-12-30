"""
Example: Load a checkpoint directly from Tinker server and resume training or evaluation.
"""

import asyncio
import json
import sys
import chz

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.chat_sl import chat_datasets
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig


def get_checkpoint_state_path(
    checkpoints_jsonl_path: str,
    checkpoint_name: str = "final",
) -> str:
    """Load checkpoint state_path from checkpoints.jsonl.
    
    Args:
        checkpoints_jsonl_path: Path to the checkpoints.jsonl file
        checkpoint_name: Name of checkpoint to load (e.g., "final", "000060")
    
    Returns:
        The tinker:// URI for the checkpoint state
    """
    with open(checkpoints_jsonl_path, 'r') as f:
        for line in f:
            checkpoint = json.loads(line)
            if checkpoint["name"] == checkpoint_name:
                return checkpoint["state_path"]
    
    raise ValueError(f"Checkpoint '{checkpoint_name}' not found in {checkpoints_jsonl_path}")


def build_config_from_checkpoint(
    checkpoints_jsonl_path: str,
    checkpoint_name: str = "final",
) -> chz.Blueprint[train.Config]:
    """Build training config that loads a checkpoint from Tinker server.
    
    Args:
        checkpoints_jsonl_path: Path to the checkpoints.jsonl file from training run
        checkpoint_name: Name of checkpoint to load (e.g., "final", "000060")
    """
    model_name = "meta-llama/Llama-3.1-8B"
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=32768,
        batch_size=128,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    dataset = chat_datasets.NoRobotsBuilder(common_config=common_config)
    
    # Load checkpoint state path from tinker server (tinker:// URI)
    checkpoint_path = get_checkpoint_state_path(checkpoints_jsonl_path, checkpoint_name)
    
    return chz.Blueprint(train.Config).apply(
        {
            "log_path": "/tmp/tinker-examples/sl_finetune_from_checkpoint",
            "model_name": model_name,
            "load_checkpoint_path": checkpoint_path,  # Load from Tinker remote storage
            "dataset_builder": dataset,
            "learning_rate": 1e-4,
            "lr_schedule": "linear",
            "num_epochs": 1,
            "eval_every": 5,
        }
    )


async def main(
    checkpoints_jsonl_path: str,
    checkpoint_name: str = "final",
):
    """Load a checkpoint from Tinker server and continue training."""
    config = build_config_from_checkpoint(checkpoints_jsonl_path, checkpoint_name)
    
    print(f"Loading checkpoint '{checkpoint_name}' from Tinker server:")
    print(f"  {config.load_checkpoint_path}")
    print(f"Training will resume from this checkpoint")
    print()
    
    # Check if log dir exists and ask user
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    
    # Run training with the loaded checkpoint
    await train.main(config)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_load_checkpoint.py <checkpoints_jsonl_path> [checkpoint_name]")
        print()
        print("Example:")
        print("  python test_load_checkpoint.py /tmp/tinker-examples/sl_basic/checkpoints.jsonl final")
        print("  python test_load_checkpoint.py /tmp/tinker-examples/sl_basic/checkpoints.jsonl 000060")
        sys.exit(1)
    
    checkpoints_jsonl = sys.argv[1]
    checkpoint_name = sys.argv[2] if len(sys.argv) > 2 else "final"
    
    print(f"Loading checkpoint '{checkpoint_name}' from {checkpoints_jsonl}")
    asyncio.run(main(checkpoints_jsonl, checkpoint_name))
