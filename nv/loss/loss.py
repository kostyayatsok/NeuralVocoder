import torch.nn as nn


class HiFiLoss(nn.Module):
    def __init__(
        self, criterion_mel=nn.L1Loss(), *args, **kwargs
    ):
        super().__init__()
        self.criterion_mel = criterion_mel
        
    def forward(self, batch, *args, **kwargs):

        mask = batch['mel_mask']
        mel_loss = self.criterion_mel(
            batch["wav_pred"][mask], batch["wav"][mask])
        
        return {
            "G_loss": mel_loss,
            "mel_loss": mel_loss,
        }
