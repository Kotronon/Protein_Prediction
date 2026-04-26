import torch
import torch.nn as nn


class Permute(nn.Module):
    """Permutes tensor dimensions.
    
    A simple wrapper around torch.permute (e.g. for use in a cnn layer).
    
    Attributes:
        dims: Tuple of dimension indices for permutation.
    """
    def __init__(self, dims):
        """Initialize the Permute layer.
        
        Args:
            dims: Tuple or list of dimension indices specifying the permutation.
        """
        super().__init__()
        self.dims = dims

    def forward(self, x):
        """Apply permutation to input tensor.
        
        Args:
            x: Input tensor.
        
        Returns:
            torch.Tensor: Permuted tensor.
        """
        return x.permute(self.dims)


class Squeeze(nn.Module):
    """Squeezes or unsqueezes a tensor dimension.
    
    Wrapper around torch.squeeze/unsqueeze (e.g. for use in a cnn layer).
    
    Attributes:
        dim: Dimension to squeeze or unsqueeze.
        unsqueeze: If True, adds a dimension; if False, removes a dimension.
    """
    def __init__(self, dim=-1, unsqueeze=False):
        """Initialize the Squeeze layer.
        
        Args:
            dim: Dimension to operate on. Defaults to -1 (last dimension).
            unsqueeze: If True, add dimension; if False, remove dimension.
                      Defaults to False.
        """
        super().__init__()
        self.dim = dim
        self.unsqueeze = unsqueeze

    def forward(self, x):
        """Apply squeeze or unsqueeze operation.
        
        Args:
            x: Input tensor.
        
        Returns:
            torch.Tensor: Tensor with dimension squeezed or unsqueezed.
        """
        if self.unsqueeze:
            return x.unsqueeze(dim=self.dim)
        else:
            return x.squeeze(dim=self.dim)


class CNNLayer(nn.Module):
    """Convolutional layer.
    
    This layer expects input of shape (batch, input_dim, seq_len) and applies
    a 2D convolution with kernel_size=(kernel_size, 1) to preserve sequence structure.
    
    Args:
        input_dim: Number of input channels
        output_dim: Number of output channels  
        kernel_size: Size of the convolutional kernel
        padding: Padding mode (default: 'same' padding)
    """
    def __init__(self, input_dim, output_dim, kernel_size=3, padding='same'):
        """Initialize the CNNLayer.
        
        Args:
            input_dim: Number of input channels.
            output_dim: Number of output channels.
            kernel_size: Size of the convolutional kernel. Defaults to 3.
            padding: Padding mode. 'same' maintains sequence length; can also be an int
                    or tuple. Defaults to 'same'.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        
        # Convert 'same' padding to actual padding value
        if isinstance(padding, str) and padding.lower() == 'same':
            conv_padding = (kernel_size // 2, 0)
        else:
            conv_padding = (padding, 0) if isinstance(padding, int) else padding
        
        self.seq = nn.Sequential(
            Permute([0, 2, 1]),  # (batch, seq_len, channels) -> (batch, channels, seq_len)
            Squeeze(unsqueeze=True),  # Add spatial dimension: (batch, channels, seq_len) -> (batch, channels, seq_len, 1)
            nn.Conv2d(
                input_dim,
                output_dim,
                kernel_size=(kernel_size, 1),
                padding=conv_padding,
            ),
            Squeeze(),  # Remove spatial dimension: (batch, channels, seq_len, 1) -> (batch, channels, seq_len)
            Permute([0, 2, 1]),  # (batch, channels, seq_len) -> (batch, seq_len, channels)
        )
    
    def forward(self, x):
        """Apply convolution to input.
        
        Args:
            x: Input tensor of shape (batch, seq_len, channels).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, output_dim).
        """
        return self.seq(x)
