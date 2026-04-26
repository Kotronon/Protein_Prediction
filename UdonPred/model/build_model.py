import sys
import copy
from importlib import import_module

from torch import nn

from model.model import load_model


def import_from_string(import_path):
    """Import a class or function from a dot-notation path.
    
    Args:
        import_path: Full import path in format 'module.path.ClassName'.
    
    Returns:
        The imported class or function.
    """
    module_path, attr_name = import_path.rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, attr_name)


def suggest(trial, name, data):
    """Suggest a hyperparameter value from configuration or trial.
    
    For fixed hyperparameters (passed as dict), returns the value directly.
    For search spaces (passed as config dict with 'type'), suggests a value
    based on the data type.
    
    Args:
        trial: Optuna trial object or dict of fixed hyperparameters.
        name: Name of the hyperparameter.
        data: Configuration dict with 'type' and 'values' or a fixed value.
    
    Returns:
        The suggested or fixed hyperparameter value.
    """
    if isinstance(trial, dict):
        if name in trial:
            return trial[name]
        else:
            return False # Default value if not found in dict

    match data["type"]:
        case "bool":
            if len(data["values"]) == 1:
                return data["values"][0]
            return bool(trial.suggest_int(name, 0, 1))
        case "int":
            return trial.suggest_int(name, *data["values"])
        case "float":
            return trial.suggest_float(name, *data["values"])
        case "categorical":
            return trial.suggest_categorical(name, data["values"])


def get_layer(trial, i, layer_type, input_dim, output_dim, layer_params):
    """
    Instantiate a layer from a PyTorch module specification.
    
    Args:
        trial: Optuna trial object or dict of hyperparameters
        i: Layer index
        layer_type: String path to PyTorch layer class (e.g., 'torch.nn.Linear')
        input_dim: Input dimension (used as first argument)
        output_dim: Output dimension (used as second argument)
        layer_params: Dict of parameters to pass to the layer, potentially with
                     hyperparameter suggestions
    
    Returns:
        Instantiated PyTorch layer
    """
    # Import the layer class
    layer_class = import_from_string(layer_type)
    
    # Process layer parameters, suggesting hyperparameters where needed
    processed_params = {}
    for param_name, param_value in layer_params.items():
        if isinstance(param_value, dict) and "type" in param_value:
            # This is a hyperparameter to suggest
            processed_params[param_name] = suggest(
                trial, f"{param_name}_{i}", param_value
            )
        else:
            # This is a fixed parameter
            processed_params[param_name] = param_value
    
    # Create the layer with input_dim and output_dim as first two arguments
    layer = layer_class(input_dim, output_dim, **processed_params)
    return layer


def get_block(trial, i, layer_type, input_dim, output_dim, activation_type, 
              dropout_rate, layer_params):
    """
    Build a block consisting of a layer, activation, and dropout.
    
    Args:
        trial: Optuna trial object or dict of hyperparameters
        i: Layer index
        layer_type: String path to PyTorch layer class
        input_dim: Input dimension
        output_dim: Output dimension
        activation_type: String path to activation class or None
        dropout_rate: Dropout rate (0.0 means no dropout)
        layer_params: Dict of parameters for the layer
    
    Returns:
        nn.Sequential block
    """
    block = []
    
    # Add the layer
    layer = get_layer(trial, i, layer_type, input_dim, output_dim, layer_params)
    block.append(layer)
    
    # Add activation if specified
    if activation_type is not None:
        block.append(import_from_string(activation_type)())
    
    # Add dropout if rate > 0
    if dropout_rate != 0.0:
        block.append(nn.Dropout(dropout_rate))
    
    return nn.Sequential(*block)


def build_prediction_heads(trial, config):
    """Build prediction heads for all configured head names.
    
    Constructs prediction heads based on configuration,
    including per-head pre/post-processing layers and configurable middle layers.
    
    Args:
        trial: Optuna trial object or dict of hyperparameters for layer configuration.
        config: Configuration dictionary with optimise and config keys.
    
    Returns:
        dict: Dictionary of prediction heads indexed by head name, each as nn.Sequential.
    """
    single_head = []

    n_layers = suggest(trial, "n_layers", config["optimise"]["num_layers"])
    input_dim = config["config"]["input_dim"]

    for layer in range(n_layers):
        layer_type = suggest(trial, f"layer_{layer}", config["optimise"]["layer_type"])
        output_dim = suggest(trial, f"dim_{layer}", config["optimise"]["layer_size"])

        activation_type = suggest(
            trial, f"activation_{layer}", config["optimise"]["activation_type"]
        )
        dropout_rate = suggest(
            trial, f"dropout_{layer}", config["optimise"]["dropout_rate"]
        )

        # Get layer-specific parameters, defaults to empty dict if not specified
        layer_params = config["optimise"]["layer_params"].get(layer_type, {})

        block = get_block(
            trial,
            layer,
            layer_type,
            input_dim,
            output_dim,
            activation_type,
            dropout_rate,
            layer_params,
        )
        single_head.append(block)
        input_dim = output_dim

    single_head.append(nn.Linear(input_dim, config["config"]["output_dim"]))
    single_head = nn.Sequential(*single_head)

    prediction_heads = {}
    for head_name in config["config"]["heads"]:
        prediction_head = []
        for op in config["config"]["pre"][head_name]:
            prediction_head.append(import_from_string(op[0])(**op[1]))

        prediction_head += list(copy.deepcopy(single_head))

        for op in config["config"]["post"][head_name]:
            prediction_head.append(import_from_string(op[0])(**op[1]))

        prediction_heads[head_name] = nn.Sequential(*prediction_head)

    return prediction_heads


def model_init(trial):
    """Initialize a UdonPred model for training.
    
    Creates a model instance with hyperparameters suggested by the trial
    (or uses defaults if no trial provided). Determines output keys from
    configured losses and metrics.
    
    Args:
        trial: Optuna trial object for hyperparameter suggestion, or None for defaults.
    
    Returns:
        UdonPred: The initialized model.
    """
    config = sys.modules["config"]

    output_keys = set()
    for ds in config["data"]:
        if config["data"][ds]["fraction"] > 0:
            keys = set()
            for key_losses in config["data"][ds]["losses"].values():
                for loss in key_losses:
                    keys.add(loss["output"])
            for key_metrics in config["data"][ds]["metrics"].values():
                for metric in key_metrics:
                    keys.add(metric["output"])
            output_keys |= keys

    if trial:
        finetune = suggest(trial, "finetune", config["optimise"]["lora"]["finetune"])
        lora_config = {}
        if finetune:
            for param in config["optimise"]["lora"]:
                if param != "finetune":
                    lora_config[param] = suggest(
                        trial,
                        f"lora_{param}",
                        config["optimise"]["lora"][param],
                    )

        prediction_heads = build_prediction_heads(trial, config)
        model = load_model(config, finetune, lora_config, prediction_heads, output_keys)
    else:  # first call of model_init is without trial object
        prediction_heads = {}
        for head_name in config["config"]["heads"]:
            prediction_head = []
            for op in config["config"]["pre"][head_name]:
                prediction_head.append(import_from_string(op[0])(**op[1]))

            prediction_head.append(
                nn.Linear(
                    config["config"]["input_dim"],
                    config["config"]["output_dim"],
                )
            )

            for op in config["config"]["post"][head_name]:
                prediction_head.append(import_from_string(op[0])(**op[1]))

            prediction_heads[head_name] = nn.Sequential(*prediction_head)

        model = load_model(config, False, {}, prediction_heads, output_keys)

    return model
