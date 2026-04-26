from abc import ABC, abstractmethod

import torch


class Loss(ABC):
    """Base abstract class for loss functions.
    
    Provides common functionality for masking and flattening tensors,
    with abstract methods for loss computation and string representation.
    """
    def __init__(self):
        """Initialize the Loss object."""
        pass

    def flatten_and_mask(self, pred, target, mask_value = 999):
        """Flatten tensors and apply masking.
        
        Flattens prediction and target tensors, then masks out invalid values.
        
        Args:
            pred: Prediction tensor.
            target: Target tensor.
            mask_value: Value indicating invalid targets. Defaults to 999.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Flattened and masked predictions and targets.
        """
        pred = pred.flatten()
        target = target.flatten()
        mask = target != mask_value
        pred = pred[mask].to(torch.float)
        target = target[mask].to(torch.float)
        return pred, target

    @abstractmethod
    def __call__(self, pred, target): ...

    @abstractmethod
    def __repr__(self) -> str:
        """Return string representation of the loss."""
        return "Base Loss Class"


class Pearson(Loss):
    """Pearson correlation coefficient loss.
    
    Measures correlation between predictions and targets.
    """
    def __init__(self):
        """Initialize Pearson loss with torchmetrics implementation."""
        from torchmetrics.regression import PearsonCorrCoef

        self.pc = PearsonCorrCoef()

    def __call__(self, pred, target):
        """Compute Pearson correlation.
        
        Args:
            pred: Dictionary with predictions.
            target: Target tensor.
        
        Returns:
            torch.Tensor: Pearson correlation coefficient.
        """
        pred = pred[list(pred.keys())[0]]
        pred, target = self.flatten_and_mask(pred, target)
        return self.pc(pred, target)

    def __repr__(self):
        return "Pearson"


class Spearman(Loss):
    """Spearman rank correlation loss.
    
    Measures rank correlation between predictions and targets.
    """
    def __init__(self):
        """Initialize Spearman loss with torchmetrics implementation."""
        from torchmetrics.regression import SpearmanCorrCoef

        self.sc = SpearmanCorrCoef()

    def __call__(self, pred, target):
        """Compute Spearman correlation.
        
        Args:
            pred: Dictionary with predictions.
            target: Target tensor.
        
        Returns:
            torch.Tensor: Spearman correlation coefficient.
        """
        pred = pred[list(pred.keys())[0]]
        pred, target = self.flatten_and_mask(pred, target)
        return self.sc(pred, target)

    def __repr__(self):
        return "Spearman"


class MSE(Loss):
    """Mean Squared Error loss."""
    def __init__(self):
        """Initialize MSE loss with torch implementation."""
        from torch.nn import MSELoss

        self.mse = MSELoss()

    def __call__(self, pred, target):
        """Compute MSE loss.
        
        Args:
            pred: Dictionary with predictions.
            target: Target tensor.
        
        Returns:
            torch.Tensor: Mean squared error.
        """
        pred = pred[list(pred.keys())[0]]
        pred, target = self.flatten_and_mask(pred, target)
        return self.mse(pred, target)

    def __repr__(self):
        return "MSE"


class MAE(Loss):
    """Mean Absolute Error loss."""
    def __call__(self, pred, target):
        """Compute MAE loss.
        
        Args:
            pred: Dictionary with predictions.
            target: Target tensor.
        
        Returns:
            torch.Tensor: Mean absolute error.
        """
        pred = pred[list(pred.keys())[0]]
        pred, target = self.flatten_and_mask(pred, target)
        return (pred - target).abs().mean()

    def __repr__(self):
        return "MAE"


class BCE(Loss):
    """Binary Cross-Entropy loss."""
    def __init__(self):
        """Initialize BCE loss with torch implementation."""
        from torch.nn import BCELoss

        self.bce = BCELoss()

    def __call__(self, pred, target):
        """Compute BCE loss.
        
        Args:
            pred: Dictionary with predictions.
            target: Target tensor.
        
        Returns:
            torch.Tensor: Binary cross-entropy loss.
        """
        pred = pred[list(pred.keys())[0]]
        pred, target = self.flatten_and_mask(pred, target)
        return self.bce(pred, target)

    def __repr__(self):
        return "BCE"


class AUROC(Loss):
    """Area Under the Receiver Operating Characteristic curve."""
    def __init__(self, threshold=0.5):
        """Initialize AUROC metric.
        
        Args:
            threshold: Classification threshold for binary conversion of labels. Defaults to 0.5.
        """
        from torchmetrics.classification import BinaryAUROC

        self.auc = BinaryAUROC()
        self.threshold = threshold

    def __call__(self, pred, target):
        """Compute AUROC.
        
        Args:
            pred: Dictionary with predictions.
            target: Target tensor.
        
        Returns:
            torch.Tensor: AUROC score.
        """
        pred = pred[list(pred.keys())[0]]
        pred, target = self.flatten_and_mask(pred, target)
        return self.auc(pred, (target > self.threshold))

    def __repr__(self):
        return "AUROC"


class AUPR(Loss):
    """Area Under the Precision-Recall curve."""
    def __init__(self, threshold=0.5):
        """Initialize AUPR metric.
        
        Args:
            threshold: Classification threshold for binary conversion of labels. Defaults to 0.5.
        """
        from torchmetrics.classification import BinaryAveragePrecision

        self.auc = BinaryAveragePrecision()
        self.threshold = threshold

    def __call__(self, pred, target):
        """Compute AUPR.
        
        Args:
            pred: Dictionary with predictions.
            target: Target tensor.
        
        Returns:
            torch.Tensor: AUPR score.
        """
        pred = pred[list(pred.keys())[0]]
        pred, target = self.flatten_and_mask(pred, target)
        return self.auc(pred, (target > self.threshold))

    def __repr__(self):
        return "AUPR"
