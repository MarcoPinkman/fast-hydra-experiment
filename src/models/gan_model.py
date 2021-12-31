from typing import Any, List

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics.classification.accuracy import Accuracy
from torch.distributions import uniform, normal
from src.models.modules.discriminator import Discriminator
from src.models.modules.generator import  Generator

class LitGanModel(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.generator=Generator(hparams= self.hparams)
        self.discriminator = Discriminator(hparams= self.hparams)
        
        self.eps_dist = uniform.Uniform(0, 1)
        self.Z_dist = normal.Normal(0, 1)
        self.eps_shape = torch.Size([self.hparams['bs'], 1])
        self.z_shape = torch.Size([self.hparams['bs'], self.hparams['z_dim']])
        
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()
        
    def forward(self, x: torch.Tensor):
        return self.generator(x)
    
    def get_conditional_input(self, X, C_Y):
        new_X = torch.cat([X, C_Y], dim=1).float()
        return new_X
    
    def get_gradient_penalty(self, X_real, X_gen):
        eps = self.eps_dist.sample(self.eps_shape)
        X_penalty = eps * X_real + (1 - eps) * X_gen

        critic_pred = self.discriminaotr(X_penalty)
        grad_outputs = torch.ones(critic_pred.size())
        gradients = autograd.grad(
                outputs=critic_pred, inputs=X_penalty,
                grad_outputs=grad_outputs,
                create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return grad_penalty
    
    def step(self, batch: Any):
        L_gen = 0
        L_disc = 0
        total_L_disc = 0
        img_features, label_attr, label_idx = batch
        X_real = self.get_conditional_input(img_features,label_attr)
        
        for _ in range(self.hparams['n_critic']):
            Z = self.Z_dist.sample(self.z_shape)
            Z = self.get_conditional_input(Z, label_attr)

            X_gen = self.generator(Z)
            X_gen = self.get_conditional_input(X_gen, label_attr)

            # calculate normal GAN loss
            L_disc = (self.discriminator(X_gen) - self.discriminator(X_real)).mean()

            # calculate gradient penalty
            grad_penalty = self.get_gradient_penalty(X_real, X_gen)
            L_disc += self.hparams['lmbda'] * grad_penalty

            # update critic params
            self.optim_D.zero_grad()
            L_disc.backward()
            self.optim_D.step()

            total_L_disc += L_disc.item()

        # =============================================================
        # optimize generator
        # =============================================================
        Z = self.Z_dist.sample(self.z_shape).to(self.device)
        Z = self.get_conditional_input(Z, label_attr)

        X_gen = self.net_G(Z)
        X = torch.cat([X_gen, label_attr], dim=1).float()
        L_gen = -1 * torch.mean(self.net_D(X))

        if use_cls_loss:
            self.classifier.eval()
            Y_pred = F.softmax(self.classifier(X), dim=0)
            log_prob = torch.log(torch.gather(Y_pred, 1, label_idx.unsqueeze(1)))
            L_cls = -1 * torch.mean(log_prob)
            L_gen += self.beta * L_cls

        self.optim_G.zero_grad()
        L_gen.backward()
        self.optim_G.step()

        return total_L_disc, L_gen.item()
    
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
 