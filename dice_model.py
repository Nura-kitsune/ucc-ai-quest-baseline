# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 18:43:23 2023

@author: gorchan2
"""

import torch
import torch.nn as nn
import torchvision
import lightning.pytorch as pl
from metrics import SMAPIoUMetric


class SegModel(pl.LightningModule):
    def __init__(self):
        super(SegModel, self).__init__()
        self.learning_rate = 1e-3
        self.net = torchvision.models.segmentation.fcn_resnet50(num_classes=2)
        self.evaluator = SMAPIoUMetric()

    def dice_loss(self, pred, target, smooth=1.):
        pred = pred.sigmoid()  # Apply sigmoid to get probabilities
        pred = pred.contiguous()
        target = target.contiguous()    

        intersection = (pred * target).sum(dim=2).sum(dim=2)
        
        loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
        
        return loss.mean()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_nb):
        img, mask = batch
        img = img.float()
        mask = mask.float()  # Change mask to float for dice loss
        out = self.forward(img)["out"]
        loss = self.dice_loss(out, mask)
        self.log("loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_nb):
        img, mask = batch
        img = img.float()
        mask = mask.float()  # Change mask to float for dice loss
        out = self.forward(img)["out"]
        loss = self.dice_loss(out, mask)

        probs = torch.sigmoid(out)  # Use sigmoid instead of softmax
        preds = (probs > 0.5).float()  # Threshold probabilities to get binary prediction
        preds = preds.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()

        self.evaluator.process(input={"pred": preds, "gt": mask})

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
        )

    def on_validation_epoch_end(self) -> None:
        metrics = self.evaluator.evaluate(0)
        self.log(
            f"val_high_vegetation_IoU",
            metrics["high_vegetation__IoU"],
            sync_dist=True,
        )
        self.log(f"val_mIoU", metrics["mIoU"], sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
