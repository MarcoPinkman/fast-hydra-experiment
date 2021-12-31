from typing import Any, List

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics.classification.accuracy import Accuracy
from src.models.modules.MLPClassifier import MLPClassifier

class pretrainedLitClassifierModel(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model=MLPClassifier(hparams= self.hparams)
        self.crierion = nn.CrossEntropyLoss()
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()
        
    def forward(self, x: torch.Tensor):
        return self.model(x)
    
    def get_conditional_input(self, X, C_Y):
        new_X = torch.cat([X, C_Y], dim=1).float()
        return new_X
    
    def step(self, batch: Any):
        img_features, label_attr, label_idx = batch
        X_inp = self.get_conditional_input(img_features,label_attr)
        Y_probs = self.forward(X_inp)
        loss = self.criterion(Y_probs, label_idx)
        Y_preds = torch.argmax(Y_probs, dim=1)
        return loss, Y_preds, label_idx
    
     def training_step(self, batch: Any, batch_idx: int):
            loss, preds, targets = self.step(batch)

       
        acc = self.train_accuracy(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}


    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        acc = self.val_accuracy(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}


    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr
        )
 