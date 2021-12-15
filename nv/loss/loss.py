import torch
import torch.nn as nn
import torch.nn.functional as F

class HiFiLoss(nn.Module):
    def __init__(
        self, criterion_mel=nn.L1Loss(), *args, **kwargs
    ):
        super().__init__()
        self.criterion_mel = criterion_mel
        
    def forward(self, batch, *args, **kwargs):
        mask = batch['mel_mask']
        mask = torch.repeat_interleave(
            batch['mel_mask'],
            repeats=batch["mel"].size(1), #repeat mask for every feature
            dim=0
        ).view(batch["mel"].size())

        pad = batch["mel"].size(-1) - batch["mel_pred"].size(-1)
        batch["mel_pred"] = F.pad(batch["mel_pred"], (0, pad), "constant", -11.5)

        mel_loss = self.criterion_mel(batch["mel_pred"][mask], batch["mel"][mask]) #TODO: add mask
        
        return {
            "G_loss": mel_loss,
            "mel_loss": mel_loss,
        }
