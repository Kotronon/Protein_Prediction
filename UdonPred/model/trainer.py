import inspect
import os
import re
from collections import defaultdict
from importlib import import_module
from typing import Optional, Union

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import Trainer
from transformers.trainer import *
from peft import PeftModel

import wandb
from model.data import ClusterSampler

logger = logging.get_logger(__name__)


def get_object(p):
    """Retrieve an object from a scope using dot notation.
    
    Evaluates a scope-qualified path string to retrieve objects from different
    scopes (global, local, or class attributes).
    
    Args:
        p: Path string in format 'scope.object_name.attr.subattr' where scope
           is one of: 'global', 'local', or 'class'.
    
    Returns:
        The retrieved object or nested attribute.
    
    Raises:
        ValueError: If scope is not one of the supported values.
    """
    scope = p.split(".")[0]
    obj_name = p.split(".")[1]
    obj_path = p.split(".")[2:]

    match scope:
        case "global":
            global_context = inspect.currentframe().f_back.f_globals  # type: ignore
            obj = global_context[obj_name]
        case "local":
            local_context = inspect.currentframe().f_back.f_locals  # type: ignore
            obj = local_context[obj_name]
        case "class":
            class_context = inspect.currentframe().f_back.f_locals["self"]  # type: ignore
            obj = getattr(class_context, obj_name)
        case _:
            raise ValueError("Scope must be one of global, local, class")

    for name in obj_path:
        obj = obj[name]

    return obj


class CustomTrainer(Trainer):
    """Custom trainer with support for metrics, losses, and checkpoint loading.
    
    Extends the HuggingFace Trainer class to support custom metric and loss
    computation from configuration, checkpoint loading for LoRA models, and
    custom train dataloader with clustering support.
    
    Attributes:
        config: Configuration dictionary containing data and model settings.
        metrics: Configured metrics indexed by dataset and feature.
        losses: Configured losses indexed by dataset and feature.
        args: Training arguments from parent Trainer.
    """
    def __init__(self, config, **kwargs):
        """Initialize the CustomTrainer.
        
        Args:
            config: Configuration dictionary with 'data' key containing metric/loss config.
            **kwargs: Additional arguments passed to parent Trainer class.
        """
        super().__init__(**kwargs)

        self.__init_metrics(config["data"])
        self.__init_losses(config["data"])
        self.config = config
        self.args.ignore_data_skip = True

        self.uses_wandb = "wandb" in config["config"].keys()

    def __import_from_string(self, import_path: str):
        """Import a class from a dot-notation path.
        
        Args:
            import_path: Full import path in format 'module.path.ClassName'.
        
        Returns:
            The imported class or function.
        """
        module_path, attr_name = import_path.rsplit(".", 1)
        module = import_module(module_path)
        return getattr(module, attr_name)

    def _init_configured_functions(self, data_config, config_key, extra_keys=None):
        """Initialize metric or loss functions from configuration.
        
        Creates function instances from configuration specifications, organizing them
        by dataset and feature. Each function has associated output patterns and arguments.
        
        Args:
            data_config: Data configuration dictionary.
            config_key: Key in data config ('metrics' or 'losses') to process.
            extra_keys: Additional keys to extract from item_config. Defaults to None.
        
        Returns:
            dict: Nested dictionary indexed by [dataset][feature] containing tuples of
                  (function_instance, output_pattern, arguments, *extra_values).
        """
        extra_keys = extra_keys or []
        configured_functions = {}
        for dataset in data_config.keys():
            if config_key not in data_config[dataset]:
                continue
            dataset_functions = {}
            for feature in data_config[dataset][config_key]:
                feature_items = []
                for item_config in data_config[dataset][config_key][feature]:
                    obj = self.__import_from_string(item_config["class"])(
                        **item_config["class_arguments"]
                    )
                    output = item_config["output"]
                    kwargs = item_config["arguments"]
                    entry = [obj, output, kwargs]
                    for key in extra_keys:
                        entry.append(item_config[key])
                    feature_items.append(tuple(entry))
                dataset_functions[feature] = feature_items
            configured_functions[dataset] = dataset_functions
        return configured_functions

    def __init_metrics(self, data_config):
        """Initialize metrics from data configuration.
        
        Args:
            data_config: Data configuration dictionary.
        """
        self.metrics = self._init_configured_functions(data_config, "metrics")

    def __init_losses(self, data_config):
        """Initialize losses from data configuration.
        
        Args:
            data_config: Data configuration dictionary.
        """
        self.losses = self._init_configured_functions(
            data_config, "losses", extra_keys=["weight"]
        )

    def save_prediction_head(self, output_dir):
        """Save the prediction head model weights.
        
        Saves model state excluding backbone weights to the specified directory.
        
        Args:
            output_dir: Directory to save model weights to.
        """
        state_dict = self.model.state_dict()  # type:ignore

        state_dict = {k: v for k, v in state_dict.items() if "backbone" not in k}

        torch.save(
            state_dict,
            os.path.join(output_dir, "pytorch_model.bin"),
        )

    def save_model(self, output_dir=None, _internal_call=False):
        """Save the complete model (backbone and prediction head) with config.
        
        Args:
            output_dir: Directory to save model to. Uses args.output_dir if None.
            _internal_call: Internal flag for Trainer compatibility.
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)  # type:ignore

        if self.model.backbone is not None:
            self.model.backbone.model.save_pretrained(output_dir)  # type:ignore
        self.save_prediction_head(output_dir)

        yaml.dump(self.config, open(output_dir + "/config.yaml", "w"))  # type: ignore

    def _get_dataset_index(self, kw, dataset_name):
        """Get the index of dataset in a batch.
        
        Args:
            kw: Batch dictionary containing 'dataset' key.
            dataset_name: Name of the dataset to find.
        
        Returns:
            int or None: Index of the dataset in the batch, or None if not found.
        """
        ds_indices = np.where(dataset_name == kw["dataset"])[0]
        if len(ds_indices) != 1:
            return None
        return ds_indices[0]

    def _prepare_call_args(self, config_args, kw, to_cpu=False):
        """Prepare arguments for function calls based on config specification.
        
        Converts configuration argument specifications (e.g., 'input.param') to actual
        values from the input dictionary or global context.
        
        Args:
            config_args: List of argument specifications (strings or paths).
            kw: Input batch dictionary.
            to_cpu: If True, move tensors to CPU. Defaults to False.
        
        Returns:
            dict: Dictionary of prepared keyword arguments.
        """
        func_kwargs = {}
        for v in config_args:
            if type(v) is str and v.startswith("input."):
                suffix = v.split(".")[1]
                for key in kw:
                    if re.match(suffix, key):
                        val = kw[key]
                        if to_cpu and isinstance(val, torch.Tensor):
                            val = val.cpu()
                        func_kwargs[key] = val
            else:
                func_kwargs[v.split(".")[1]] = get_object(v)
        return func_kwargs

    def _prepare_feature_inputs(
        self, feature, output_pattern, kw, subset_outputs, ds_idx, to_cpu=False
    ):
        """Prepare feature inputs and outputs for metric/loss computation.
        
        Args:
            feature: Feature name to extract.
            output_pattern: Regex pattern to match output keys.
            kw: Input batch dictionary.
            subset_outputs: Model outputs for a specific dataset.
            ds_idx: Dataset index in the batch.
            to_cpu: If True, move tensors to CPU. Defaults to False.
        
        Returns:
            Tuple[torch.Tensor, dict]: Selected feature tensor and matching outputs.
        """
        selected_features = [
            x for i, x in enumerate(kw[feature]) if kw["dataset_keys"][i] == ds_idx
        ]
        if to_cpu:
            selected_features = [x.cpu() for x in selected_features]

        subset_feature = torch.stack(selected_features)

        subset_feature_output = {}
        for x, val in subset_outputs.items():
            if re.match(output_pattern + r"_\d+", x):
                if to_cpu:
                    val = val.cpu()
                subset_feature_output[x] = val

        return subset_feature, subset_feature_output

    def _get_subset_outputs(self, outputs, kw, ds_idx):
        """Extract model outputs for a specific dataset.
        
        Args:
            outputs: Dictionary of model outputs.
            kw: Input batch dictionary with 'dataset_keys'.
            ds_idx: Dataset index.
        
        Returns:
            dict: Outputs corresponding to the specified dataset.
        """
        return {k: v[kw["dataset_keys"] == ds_idx] for k, v in outputs.items()}

    def calculate_metrics(self, outputs, kw):
        """Calculate metrics from model outputs and targets.
        
        Computes all configured metrics for each dataset and feature.
        
        Args:
            outputs: Model output dictionary.
            kw: Batch dictionary containing targets and metadata.
        
        Returns:
            dict: Calculated metrics keyed by metric name.
        """
        calculated_metrics = {}
        state = "training" if self.model.training else "validation"  # type: ignore
        for ds in self.metrics.keys():
            ds_idx = self._get_dataset_index(kw, ds)
            if ds_idx is None:
                continue

            subset_outputs = self._get_subset_outputs(outputs, kw, ds_idx)

            for feature in self.metrics[ds].keys():
                for metric, output_pattern, metric_args_config in self.metrics[ds][
                    feature
                ]:
                    metric_kwargs = self._prepare_call_args(
                        metric_args_config, kw, to_cpu=True
                    )

                    subset_feature, subset_feature_output = (
                        self._prepare_feature_inputs(
                            feature,
                            output_pattern,
                            kw,
                            subset_outputs,
                            ds_idx,
                            to_cpu=True,
                        )
                    )

                    if (subset_feature != 1).sum() != 0:
                        calculated_metrics[f"{state}/{ds}/{feature}/{str(metric)}"] = (
                            metric(
                                subset_feature_output,
                                subset_feature,
                                **metric_kwargs,
                            ).item()
                        )

        return calculated_metrics

    def calculate_loss(self, outputs, kw):
        """Calculate total weighted loss from model outputs.
        
        Computes all configured losses for each dataset and feature,
        then sums them with their respective weights.
        
        Args:
            outputs: Model output dictionary.
            kw: Batch dictionary containing targets and metadata.
        
        Returns:
            torch.Tensor: Total weighted loss.
        """
        loss = 0
        for ds in self.losses.keys():
            ds_idx = self._get_dataset_index(kw, ds)
            if ds_idx is None:
                continue

            subset_outputs = self._get_subset_outputs(outputs, kw, ds_idx)

            for feature in self.losses[ds].keys():
                for (
                    loss_fnc,
                    output_pattern,
                    loss_args_config,
                    weight,
                ) in self.losses[ds][feature]:
                    subset_feature, subset_feature_output = (
                        self._prepare_feature_inputs(
                            feature,
                            output_pattern,
                            kw,
                            subset_outputs,
                            ds_idx,
                            to_cpu=False,
                        )
                    )

                    loss_kwargs = self._prepare_call_args(
                        loss_args_config, kw, to_cpu=False
                    )

                    loss += weight * loss_fnc(
                        subset_feature_output, subset_feature, **loss_kwargs
                    )

        return loss

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """Compute loss for training step.
        
        Computes custom losses and metrics from model outputs. Scales loss by
        gradient accumulation steps and logs metrics to WandB.
        
        Args:
            model: The model being trained.
            inputs: Input batch dictionary.
            return_outputs: If True, return outputs with loss.
            num_items_in_batch: Optional number of items in batch.
        
        Returns:
            Tensor or Tuple: Loss tensor, or (loss, outputs) if return_outputs=True.
        """
        if (
            self.label_smoother is not None or self.compute_loss_func is not None
        ) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}

        outputs = model(**inputs)
        model_outputs = {k: v for k, v in outputs.items() if k != "loss"}

        metrics = self.calculate_metrics(model_outputs, inputs)

        if self.uses_wandb:
            wandb.log(metrics)

        loss = (
            self.calculate_loss(model_outputs, inputs)
            / self.args.gradient_accumulation_steps
        )

        return (loss, outputs) if return_outputs else loss

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        """Evaluate the model on the evaluation dataset.
        
        Runs evaluation loop, computing metrics and loss. Handles deepspeed
        and FSDP initialization if needed.
        
        Args:
            eval_dataset: Optional evaluation dataset to use. If None, uses self.eval_dataset.
            ignore_keys: Optional list of keys to ignore in outputs.
            metric_key_prefix: Prefix for metric keys. Defaults to 'eval'.
        
        Returns:
            dict: Dictionary of computed metrics.
        """
        eval_dataloader = (
            self.get_eval_dataloader(self.eval_dataset)  # type:ignore
            if eval_dataset is None
            else eval_dataset
        )

        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=eval_dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                or (
                    self.is_fsdp_enabled
                    and self.accelerator.mixed_precision != "fp8"
                    and not self.args.torch_compile
                )
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            if model is not self.model:
                self.model_wrapped = model

            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        if not self.is_in_train:
            if self.args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=self.args.device)
            elif self.args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=self.args.device)

        model.eval()
        if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):  # type:ignore
            self.optimizer.eval()  # type:ignore

        self.callback_handler.eval_dataloader = eval_dataloader  # type: ignore
        eval_dataset = getattr(eval_dataloader, "dataset", None)

        if self.args.past_index >= 0:
            self._past = None

        batch_size = self.args.eval_batch_size

        merged_metrics = defaultdict(list)
        with torch.no_grad():
            for inputs in tqdm(eval_dataloader, total=len(eval_dataloader)):  # type: ignore
                observed_batch_size = find_batch_size(inputs)
                if observed_batch_size is not None:
                    if batch_size is None:
                        batch_size = observed_batch_size

                outputs = model(**inputs)

                merged_metrics["eval_loss"].append(self.calculate_loss(outputs, inputs))

                # outputs = {k: v for k, v in outputs.items() if k != "eval_loss"}

                metrics = self.calculate_metrics(outputs, inputs)
                for k, v in metrics.items():
                    merged_metrics[k].append(v)


            if self.uses_wandb:
                for k, v in merged_metrics.items():
                    wandb.log({k: torch.tensor(v).mean().item()})

        print(
            "\neval_loss",
            torch.tensor(merged_metrics["eval_loss"]).mean().item(),
            end="\n",
        )
        return {"eval_loss": torch.tensor(merged_metrics["eval_loss"]).mean().item()}

    def get_train_dataloader(self):
        """Get training dataloader with optional cluster-based sampling.
        
        Creates a dataloader with support for cluster-based sampling.
        
        Returns:
            DataLoader: Training dataloader.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        if self.config["config"]["cluster"]:
            sampler = ClusterSampler(
                self.train_dataset, self.config["config"]["cluster"]
            )
        else:
            sampler = self._get_train_sampler

        return self._get_dataloader(
            dataset=self.train_dataset,  # type:ignore
            description="Training",
            batch_size=self._train_batch_size,  # type:ignore
            sampler_fn=sampler,  # type:ignore
            is_training=True,
        )

    def _load_from_checkpoint(self, resume_from_checkpoint):
        """Load model from checkpoint with LoRA adapter support.
        
        Handles loading of LoRA adapter and prediction head from checkpoint.

        Args:
            resume_from_checkpoint: Path to checkpoint directory to load from. False if not resuming.
        """
        if self.model.backbone is not None:
            self.model.backbone.model.load_adapter(resume_from_checkpoint, "default")

        self.model.load_state_dict(
            torch.load(
                resume_from_checkpoint + "/pytorch_model.bin", weights_only=True
            ),
            strict=False,
        )
