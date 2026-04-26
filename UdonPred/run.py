import argparse
import inspect
import os
import sys

import yaml
from transformers.training_args import TrainingArguments

import wandb
from model.build_model import model_init, suggest
from model.data import DataCollator, get_datasets
from model.trainer import CustomTrainer


CONFIG_DIR = "config"
OUTPUT_DIR = "checkpoints"
OPTIMISED_PARAMETERS_DIR = "optimised_parameters"


deepspeed_config = {
    "zero_optimization": {
        "stage": 1,
        "overlap_comm": True,
        "contiguous_gradients": True,
    },
    "fp16": {"enabled": "auto"},
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
}


def calculate_batch_sizes(total_batch_size, max_single_batch_size):
    """Calculate per-device batch size and gradient accumulation steps.
    
    Splits a total batch size into actual batch size and gradient accumulation
    steps based on the maximum single batch size.
    
    Args:
        total_batch_size: Total desired batch size across all devices.
        max_single_batch_size: Maximum batch size per device.
        
    Returns:
        Tuple[int, int]: A tuple of (per_device_batch_size, gradient_accumulation_steps).
    """
    if total_batch_size <= max_single_batch_size:
        return total_batch_size, 1
    return max_single_batch_size, total_batch_size // max_single_batch_size


def setup_wandb(config, hyperparameters=None):
    """Initialize Weights & Biases (WandB) if configured.
    
    Sets up WandB logging with project configuration and optional hyperparameters.
    Only initializes if wandb configuration is present in the config.
    
    Args:
        config: Full configuration dictionary including wandb settings.
        hyperparameters: Optional dict of hyperparameters to log with WandB config.
                        Defaults to None.
    """
    if "wandb" in config["config"]:
        os.environ["WANDB_PROJECT"] = config["config"]["wandb"]["project"]
        os.environ["WANDB_LOG_MODEL"] = config["config"]["wandb"]["log_model"]
        wandb.init(
            project=os.environ["WANDB_PROJECT"],
            name=config["config"]["run_name"],
            config=config | (hyperparameters or {}),
        )


def get_data(config):
    """Load and prepare datasets based on configuration.
    
    Loads datasets defined in the config and optionally filters the training set
    using a filter file if specified.
    
    Args:
        config: Configuration dictionary containing dataset paths and filter settings.
    
    Returns:
        DatasetDict: Combined dataset with train/validation splits.
    """
    if config["config"]["filter"] is not False:
        with open(config["config"]["filter"]) as f:
            train_filter = [x.strip() for x in f]
    else:
        train_filter = None

    return get_datasets(config, train_filter)


def hp_space(trial):
    """Define non-architecture hyperparameter search space for Optuna optimization.
    
    Creates a hyperparameter space of options unrelated to model architecture 
    that Optuna can sample from during hyperparameter optimization. 
    Suggests values for batch size, learning rate, and learning rate scheduler type.
    
    Args:
        trial: Optuna trial object for suggesting hyperparameter values.
    
    Returns:
        dict: Dictionary of suggested hyperparameters for training.
    """
    config = sys.modules["config"]

    total_batch_size = suggest(trial, "batch_size", config["optimise"]["batch_size"])
    batch_size, gradient_accumulation_steps = calculate_batch_sizes(
        total_batch_size, config["config"]["max_single_batch_size"]
    )

    return {
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": suggest(
            trial, "learning_rate", config["optimise"]["learning_rate"]
        ),
        "lr_scheduler_type": suggest(
            trial, "lr_scheduler", config["optimise"]["lr_scheduler"]
        ),
        # "fp16": suggest(trial, "fp16", config["optimise"]["fp16"]),
    }


def train_model(config, datasets, collator):
    """Train a model with fixed hyperparameters.
    
    Performs model training using specified hyperparameters from a configuration file.
    Handles batch size calculation, training setup, and optional DeepSpeed integration.
    
    Args:
        config: Configuration dictionary with training parameters.
        datasets: Dictionary with 'train' and 'validation' dataset splits.
        collator: Data collator for batching samples.
    """
    with open(config["config"]["hyperparameter_path"], "r") as f:
        hyperparameters = yaml.safe_load(f)

    batch_size, grad_accum_steps = calculate_batch_sizes(
        hyperparameters["batch_size"],
        config["config"]["max_single_batch_size"]
    )
    hyperparameters["per_device_train_batch_size"] = batch_size
    hyperparameters["gradient_accumulation_steps"] = grad_accum_steps

    training_arguments = {
        key: value
        for key, value in hyperparameters.items()
        if key in inspect.signature(TrainingArguments).parameters
    }
    training_arguments["remove_unused_columns"] = False
    training_arguments["report_to"] = (
        "wandb" if "wandb" in config["config"].keys() else None
    )
    training_arguments["output_dir"] = (
        f"{OUTPUT_DIR}/{config['config']['run_name']}"
    )
    training_arguments |= config["config"]["logging"]
    training_arguments |= config["config"]["saving"]
    training_arguments["num_train_epochs"] = config["config"]["num_train_epochs"]

    if config["config"]["deepspeed_zero"]:
        training_arguments["deepspeed"] = deepspeed_config

    config["hyperparameters"] = hyperparameters

    training_arguments = TrainingArguments(**training_arguments)

    model = model_init(hyperparameters)

    trainer = CustomTrainer(
        config=config,
        model=model,
        args=training_arguments,
        data_collator=collator,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
    )

    setup_wandb(config, hyperparameters)

    trainer.train(resume_from_checkpoint=config["config"]["checkpoint"])


def run_optimisation(config, datasets, collator):
    """Run hyperparameter optimization using Optuna.
    
    Performs hyperparameter optimization with optional Optuna samplers and pruners.
    Saves the best hyperparameters found to a YAML file.
    
    Args:
        config: Configuration dictionary including optimization settings.
        datasets: Dictionary with 'train' and 'validation' dataset splits.
        collator: Data collator for batching samples.
    """
    training_arguments = {
        "remove_unused_columns": False,
        "report_to": "wandb" if "wandb" in config["config"].keys() else None,
        "output_dir": f"{OUTPUT_DIR}/{config['config']['run_name']}",
        "per_device_eval_batch_size": config["config"]["max_single_batch_size"],
        "no_cuda": False,
        "metric_for_best_model": "eval_loss",
    }

    training_arguments |= config["config"]["logging"]
    training_arguments["save_strategy"] = "no"
    training_arguments["num_train_epochs"] = config["config"]["num_train_epochs"]

    if config["config"]["deepspeed_zero"]:
        training_arguments["deepspeed"] = deepspeed_config

    training_arguments = TrainingArguments(**training_arguments)

    trainer = CustomTrainer(
        config=config,
        model=None,
        model_init=model_init,
        args=training_arguments,
        data_collator=collator,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
    )

    if "sampler" in config["config"]["optim"]:
        from model.build_model import import_from_string
        config["config"]["optim"]["sampler"] = import_from_string(
            config["config"]["optim"]["sampler"]
        )()

    if "pruner" in config["config"]["optim"]:
        from model.build_model import import_from_string
        config["config"]["optim"]["pruner"] = import_from_string(
            config["config"]["optim"]["pruner"]
        )()

    best_run = trainer.hyperparameter_search(
        hp_space=hp_space,
        compute_objective=lambda metrics: metrics["eval_loss"],
        **config["config"]["optim"],
    )

    os.makedirs(f"{OPTIMISED_PARAMETERS_DIR}/{config['config']['run_name']}/", exist_ok=True)
    with open(
        f"{OPTIMISED_PARAMETERS_DIR}/{config['config']['run_name']}/{best_run.run_id}.yaml",
        "w+",
    ) as f:
        yaml.dump(best_run.hyperparameters, f)


def main(mode: str):
    """Main execution function.
    
    Args:
        mode: Either 'train' or 'optimise'
    """
    config = {}
    for file in os.listdir(CONFIG_DIR):
        name = file.replace(".yaml", "")
        with open(f"{CONFIG_DIR}/{file}", "r") as f:
            config[name] = yaml.safe_load(f)

    sys.modules["config"] = config

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["config"]["cuda_devices"])

    data = get_data(config)

    collator = DataCollator(
        config["config"]["backbone"]["name"],
        config["config"]["backbone"]["tokeniser_type"],
    )

    if mode == "train":
        train_model(config, data, collator)
    elif mode == "optimise":
        run_optimisation(config, data, collator)
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'optimise'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train or optimise a model for disorder prediction"
    )
    parser.add_argument(
        "mode",
        choices=["train", "optimise"],
        help="Mode to run: 'train' for training or 'optimise' for hyperparameter optimization"
    )
    
    args = parser.parse_args()
    main(args.mode)
