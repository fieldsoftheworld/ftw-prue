import torch
import torch.nn as nn
import torch.nn.functional as F


class FtnmtLoss(nn.Module):
    """
    Fractal Tanimoto (with complement/dual) loss in PyTorch.
    """
    def __init__(self, depth=5, smooth=1e-5, dims=(1, 2, 3)):
        """
        Args:
            depth: fractal depth (>=0)
            smooth: numerical stability term
            dims: dimensions to reduce over (channel, H, W for segmentation)
        """
        super().__init__()
        assert depth >= 0, "depth must be >= 0"

        if depth == 0:
            self.depth = 1
            self.scale = 1.0
        else:
            self.depth = depth
            self.scale = 1.0 / depth

        self.smooth = smooth
        self.dims = dims

    def inner_prod(self, prob, label):
        """Inner product <prob, label>"""
        return torch.sum(prob * label, dim=self.dims)

    def tnmt_base(self, preds, labels):
        """
        Base fractal Tanimoto coefficient (averaged over depths).
        """
        tpl = self.inner_prod(preds, labels)  
        tpp = self.inner_prod(preds, preds)   
        tll = self.inner_prod(labels, labels) 

        num = tpl + self.smooth
        denum = 0.0

        for d in range(self.depth):
            a = 2.0 ** d
            b = -(2.0 * a - 1.0)
            denom_d = a * (tpp + tll) + b * tpl + self.smooth
            denum += 1.0 / denom_d

        result = num * denum * self.scale
        return torch.mean(result) 

    def forward(self, preds, labels):
        """
        preds: (N, C, H, W)
        labels: (N, C, H, W)
        """
        l1 = self.tnmt_base(preds, labels)
        l2 = self.tnmt_base(1.0 - preds, 1.0 - labels)

        sim = 0.5 * (l1 + l2)
        return 1.0 - sim


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss wrapper with task-specific losses.
    """
    def __init__(self, depth=5, n_classes=2):
        super().__init__()

        self.ftnmt_loss = FtnmtLoss(depth=depth, dims=(1, 2, 3))
        self.distance_loss = nn.MSELoss() 
        self.n_classes = n_classes

    def forward(self, predictions, labels):
        """
        Args:
            predictions: list of 3 tensors [segm_pred, boundary_pred, distance_pred]
            labels: list of 3 tensors [segm_label, boundary_label, distance_label]
            
            NOTE: We no longer concatenate labels.
        """
        pred_segm, pred_bound, pred_dist = predictions
        label_segm, label_bound, label_dist = labels

        # Task 1: Segmentation loss (using Ftnmt)
        loss_segm = self.ftnmt_loss(pred_segm, label_segm)
        
        # Task 2: Boundary loss (using Ftnmt)
        loss_bound = self.ftnmt_loss(pred_bound, label_bound)

        # Task 3: Distance loss (MSE)
        loss_dist = self.distance_loss(pred_dist, label_dist)

        total_loss = (loss_segm + loss_bound + loss_dist) / 3.0

        return total_loss, loss_segm, loss_bound, loss_dist