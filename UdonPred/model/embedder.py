import re
from importlib import import_module

import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from transformers import AutoConfig


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


class Embedder(PreTrainedModel):
    """Embedding model wrapper for protein sequences.
    
    Wraps a pretrained transformer model to provide tokenization and embedding
    functionality for protein sequences. Supports loading tokenizers and models
    dynamically based on configuration.
    
    Attributes:
        backbone_name: Name or path of the pretrained model.
        tokeniser_type: Type of tokenizer to use (e.g., 'T5Tokenizer').
        model_type: Type of model to use (e.g., 'T5EncoderModel').
        prefix_token: Optional prefix to prepend to sequences.
        tokeniser: Loaded tokenizer instance (lazy-loaded).
        model: Loaded embedding model (lazy-loaded).
        prefix_token_len: Number of tokens in the prefix.
    """
    def __init__(
        self,
        backbone_name: str,
        prefix_token: str = "",
        tokeniser_type: str = "T5Tokenizer",
        model_type: str = "T5EncoderModel",
    ):
        """Initialize the Embedder.
        
        Args:
            backbone_name: Name or path of the pretrained model to load.
            prefix_token: Optional prefix string to prepend to sequences.
                         Defaults to empty string.
            tokeniser_type: Name of tokenizer class to use from transformers.
                           Defaults to 'T5Tokenizer'.
            model_type: Name of model class to use from transformers.
                       Defaults to 'T5EncoderModel'.
        """
        config = AutoConfig.from_pretrained(
            backbone_name, trust_remote_code=True
        )
        super().__init__(config)

        self.backbone_name = backbone_name
        self.tokeniser_type = tokeniser_type
        self.model_type = model_type
        self.tokeniser = None
        self.model = None
        self.prefix_token = prefix_token + " " if prefix_token != "" else ""
        self.prefix_token_len = len(self.prefix_token.split(" ")) - 1

    def load_tokeniser(self):
        """Load the tokenizer for the embedding model.
        
        Dynamically imports and loads the specified tokenizer type from the
        transformers library. Sets the tokeniser instance attribute.
        """
        tokeniser_base = getattr(import_module("transformers"), self.tokeniser_type)
        self.tokeniser = tokeniser_base.from_pretrained(
            self.backbone_name,
            use_fast=False,
            do_lower_case=False,
            trust_remote_code=True,
        )

    def load_embedder(self):
        """Load the embedding model and move to device.
        
        Dynamically imports and loads the specified model type from the transformers
        library. Sets the model to evaluation mode and moves it to the appropriate
        device (CUDA, MPS, or CPU).
        """
        backbone_base = getattr(import_module("transformers"), self.model_type)
        self.model = backbone_base.from_pretrained(
            self.backbone_name, trust_remote_code=True
        )
        self.model.config.output_attentions = False
        self.model.config.output_hidden_states = False

        self.model = self.model.eval()
        self.model = self.model.to(device)

    def tokenise(self, seqs):
        """Tokenize protein sequences.
        
        Converts protein sequences to token IDs and attention masks. Replaces
        non-standard amino acids (U, Z, O, B, *) with X before tokenization.
        Adds prefix token if configured.
        
        Args:
            seqs: List of protein sequences as strings.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple of (input_ids, attention_mask)
                tensors from the tokenizer.
        """
        seqs = [
            self.prefix_token + " ".join(list(re.sub(r"[UZOB\*]", "X", seq)))
            for seq in seqs
        ]

        if self.tokeniser is None:
            self.load_tokeniser()

        token_encoding = self.tokeniser.batch_encode_plus(  # type: ignore
            seqs,
            add_special_tokens=True,
            padding="longest",
            return_tensors="pt",
        )

        return token_encoding["input_ids"], token_encoding["attention_mask"]

    def embed(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Generate embeddings for tokenized sequences.
        
        Offline embedding method for inference and data preparation. 
        Processes tokenized input through the embedding model in 
        inference mode and moves the result to CPU.
        
        Args:
            input_ids: Tokenized input IDs.
            attention_mask: Attention mask for valid tokens.
        
        Returns:
            torch.Tensor: Embeddings of shape (batch_size, seq_len, embedding_dim).
        """
        input_ids = torch.as_tensor(input_ids, device=device)
        attention_mask = torch.as_tensor(attention_mask, device=device)

        if self.model is None:
            self.load_embedder()

        with torch.inference_mode():
            embedding_rpr = self.model(  # type: ignore
                input_ids, attention_mask=attention_mask
            ).last_hidden_state.cpu()

        if device.type == "mps":
            torch.mps.empty_cache()

        return embedding_rpr

    def forward(self, input_ids, attention_mask):
        """Forward pass for online embedding (with gradient support).
        
        Online embedding method for use during training.
        
        Args:
            input_ids: Tokenized input IDs.
            attention_mask: Attention mask for valid tokens.
        
        Returns:
            Transformer model output containing hidden states.
        """
        if self.model is None:
            self.load_embedder()

        return self.model(input_ids, attention_mask=attention_mask).last_hidden_state
