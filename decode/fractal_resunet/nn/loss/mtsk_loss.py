from .ftnmt_loss import ftnmt_loss


class mtsk_loss:
    """
    Here NClasses = 2 by default, for a binary segmentation problem in 1hot representation
    """

    def __init__(self, depth=0, NClasses=2):
        self.ftnmt = ftnmt_loss(depth=depth)
        self.skip = NClasses

    def loss(self, prediction, label):
        pred_segm = prediction[0]
        pred_bound = prediction[1]
        pred_dists = prediction[2]

        label_segm = label[:, : self.skip, :, :]
        label_bound = label[:, self.skip : 2 * self.skip, :, :]
        label_dists = label[:, 2 * self.skip :, :, :]

        loss_segm = self.ftnmt(pred_segm, label_segm)
        loss_bound = self.ftnmt(pred_bound, label_bound)
        loss_dists = self.ftnmt(pred_dists, label_dists)

        return (loss_segm + loss_bound + loss_dists) / 3.0
