import torch
import torch.nn as nn

class FtnmtLoss(nn.Module):
    def __init__(self, depth=5, smooth=1e-5, dims=(1, 2, 3)):
        super().__init__()
        assert depth >= 0, "depth must be >= 0"
        self.depth = max(1, depth)
        self.scale = 1.0 / self.depth
        self.smooth = smooth
        self.dims = dims

    def inner_prod(self, prob, label, valid_mask):
        """Inner product <prob, label> with valid_mask."""
        return torch.sum(prob * label * valid_mask, dim=self.dims)

    def tnmt_base(self, preds, labels, valid_mask):
        tpl = self.inner_prod(preds, labels, valid_mask)   # intersection
        tpp = self.inner_prod(preds, preds, valid_mask)    # p^2
        tll = self.inner_prod(labels, labels, valid_mask)  # l^2

        num = tpl + self.smooth
        denum = 0.0

        for d in range(self.depth):
            a = 2.0 ** d
            b = -(2.0 * a - 1.0)
            denom_d = a * (tpp + tll) + b * tpl + self.smooth
            denum += 1.0 / denom_d

        result = num * denum * self.scale
        return torch.mean(result)

    def forward(self, preds, labels, valid_mask):
        """
        preds:  (N, C, H, W)
        labels: (N, C, H, W)
        valid_mask: (N, 1, H, W)
        """
        l1 = self.tnmt_base(preds, labels, valid_mask)
        l2 = self.tnmt_base(1.0 - preds, 1.0 - labels, valid_mask)
        sim = 0.5 * (l1 + l2)
        return 1.0 - sim


class MultiTaskLoss(nn.Module):
    def __init__(self, depth=5, seg_weight=1.0, bound_weight=1.0, dist_weight=1.0):
        super().__init__()
        self.ftnmt_loss = FtnmtLoss(depth=depth, dims=(1, 2, 3))
        self.distance_loss = nn.MSELoss(reduction="none")
        self.seg_weight = seg_weight
        self.bound_weight = bound_weight
        self.dist_weight = dist_weight

    def forward(self, predictions, labels, valid_mask):
        pred_segm, pred_bound, pred_dist = predictions
        label_segm, label_bound, label_dist = labels

        # Sanity check for distance regression
        assert pred_dist.shape[1] == 1, f"Expected 1 channel for distance, got {pred_dist.shape[1]}"

        # Losses
        loss_segm = self.ftnmt_loss(pred_segm, label_segm, valid_mask)
        loss_bound = self.ftnmt_loss(pred_bound, label_bound, valid_mask)

        loss_dist_raw = self.distance_loss(pred_dist, label_dist)
        loss_dist = torch.sum(loss_dist_raw * valid_mask) / (torch.sum(valid_mask) + 1e-5)

        # Weighted aggregation
        total_loss = (
            self.seg_weight * loss_segm +
            self.bound_weight * loss_bound +
            self.dist_weight * loss_dist
        ) / (self.seg_weight + self.bound_weight + self.dist_weight)

        return total_loss, loss_segm, loss_bound, loss_dist
