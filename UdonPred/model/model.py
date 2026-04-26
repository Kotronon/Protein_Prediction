import re
from collections import defaultdict

from peft import LoraConfig, get_peft_model
from torch import nn

from .embedder import Embedder


class UdonPred(nn.Module):
    """Prediction model for protein sequences using pre-trained embeddings.
    
    Combines a backbone embedding model with prediction heads to generate
    predictions for multiple tasks/outputs.
    
    Attributes:
        backbone: Pre-trained embedding model (optional, can be None for inference with provided embeddings).
        prediction_heads: Dictionary of prediction head modules.
        output_keys: Mapping of output names to prediction head indices.
    """
    def __init__(
        self,
        backbone,
        prediction_heads,
        output_keys,
    ):
        """Initialize UdonPred model.
        
        Args:
            backbone: Embedder instance or None for external embeddings.
            prediction_heads: Dictionary of prediction head modules.
            output_keys: List of output names in format 'head_idx' or 'head_-idx'.
        """
        super().__init__()

        self.backbone = backbone

        self.prediction_heads = nn.ModuleDict(prediction_heads)

        self.output_keys = defaultdict(lambda: defaultdict(str))
        for key in output_keys:
            head = key.split("_")[0]
            pos = int(key.split("_")[1])
            if pos < 0:
                pos = len(prediction_heads[head]) + pos
                self.output_keys[head][pos] = key


    def forward(self, **kw):
        """Forward pass through the model.
        
        Processes input sequences through the embedding backbone (if present) and
        prediction heads, generating outputs at specified layers.
        
        Args:
            **kw: Batch dictionary with keys like:
                  - input_ids_i: Tokenized input for i-th sequence group
                  - attention_mask_i: Attention mask for i-th sequence group
                  - embedding_i: Pre-computed embedding for i-th sequence group (if no backbone)
                  - x_i_lens: Sequence lengths
                  - dataset: Dataset name
                  - Other feature tensors for targets
        
        Returns:
            dict: Dictionary of outputs keyed by output names from output_keys.
        """
        num_inputs = (
            len([x for x in kw if re.match(r"^input_ids_\d+$", x)])
            if self.backbone is not None
            else len([x for x in kw if re.match(r"^embedding_\d+$", x)])
        )
        outputs = {}

        for i in range(num_inputs):
            if self.backbone is not None:

                emb = self.backbone(
                    kw[f"input_ids_{i}"], kw[f"attention_mask_{i}"]
                )

                emb = emb * kw[f"attention_mask_{i}"].unsqueeze(-1)
                emb = emb[:, 
                          self.backbone.prefix_token_len:
                          self.backbone.prefix_token_len + max(kw[f"x_{i}_lens"]),
                     :]

            else:
                emb = kw[f"embedding_{i}"]

            for head_name, prediction_head in self.prediction_heads.items():
                current = emb
                for j, layer in enumerate(prediction_head):
                    current = layer(current)
                    if j in self.output_keys[head_name].keys():
                        outputs[f"{self.output_keys[head_name][j]}_{i}"] = current

        return outputs


def load_model(config, finetune, lora_config, prediction_heads, output_keys):
    """Load UdonPred model with optional fine-tuning.
    
    Creates and returns a UdonPred model, optionally with LoRA fine-tuning
    configuration applied to the backbone.
    
    Args:
        config: Configuration dictionary with backbone settings.
        finetune: Whether to enable LoRA fine-tuning on the backbone.
        lora_config: LoRA configuration dictionary.
        prediction_heads: Dictionary of prediction head modules.
        output_keys: List of output names.
    
    Returns:
        UdonPred: The initialized model.
    """
    bb_params = config["config"]["backbone"]
    embedder = Embedder(
        backbone_name=bb_params["name"],
        prefix_token=bb_params["prefix_token"],
        tokeniser_type=bb_params["tokeniser_type"],
        model_type=bb_params["model_type"],
    )

    if finetune:
        embedder.load_embedder()

        peft_config = LoraConfig(**lora_config)
        embedder.model = get_peft_model(embedder.model, peft_config)

        model = UdonPred(
            embedder, prediction_heads, output_keys
        )

        return model
    else:
        return UdonPred(None, prediction_heads, output_keys)


def print_trainable_parameters(model):
    """Print the number of trainable and total parameters in the model.
    
    Counts and displays the number of trainable parameters, total parameters,
    and the percentage of trainable parameters in the model.
    
    Args:
        model: PyTorch model to analyze.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
