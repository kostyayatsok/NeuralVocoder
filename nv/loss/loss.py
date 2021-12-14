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
        pad = batch["mel"].size(-1) - batch["mel_pred"].size(-1)
        batch["mel_pred"] = F.pad(batch["mel_pred"], (0, pad))

        mel_loss = self.criterion_mel(batch["mel_pred"], batch["mel"]) #TODO: add mask
        
        return {
            "G_loss": mel_loss,
            "mel_loss": mel_loss,
        }
