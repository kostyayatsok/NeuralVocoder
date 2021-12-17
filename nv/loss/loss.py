import torch
import torch.nn as nn
import torch.nn.functional as F

class HiFiGLoss(nn.Module):
    def __init__(
        self, *args, **kwargs
    ):
        super().__init__()
        
    def forward(self, batch, *args, **kwargs):
        mask = batch['mel_mask']
        mask = torch.repeat_interleave(
            batch['mel_mask'],
            repeats=batch["mel"].size(1), #repeat mask for every feature
            dim=0
        ).view(batch["mel"].size())

        pad = batch["mel"].size(-1) - batch["mel_pred"].size(-1)
        batch["mel_pred"] = F.pad(batch["mel_pred"], (0, pad), "constant", -11.5)

        mel_loss = F.l1_loss(batch["mel_pred"][mask], batch["mel"][mask])

        feature_loss = F.l1_loss(batch["G_MPD_fake"], batch["G_MPD_real"])
        feature_loss += F.l1_loss(batch["G_MSD_fake"], batch["G_MSD_real"])
        
        G_MPD_fake = 0, 0
        G_MSD_fake = 0, 0
        for x in batch['G_MPD_fake']: G_MPD_fake += F.mse_loss(x, torch.ones_like(x))
        for x in batch['G_MSD_fake']: G_MSD_fake += F.mse_loss(x, torch.ones_like(x))
        
        return {
            "mel_loss": mel_loss,
            "feature_loss": feature_loss,
            "G_MPD_fake": G_MPD_fake,
            "G_MSD_fake": G_MSD_fake,
            "G_loss": 45 * mel_loss + 2 * feature_loss + G_MPD_fake + G_MSD_fake,
        }

class HiFiDLoss(nn.Module):
    def __init__(
        self, criterion_mel=nn.L1Loss(), *args, **kwargs
    ):
        super().__init__()
        self.criterion_mel = criterion_mel
        
    def forward(self, batch, *args, **kwargs):
        D_MPD_fake, D_MPD_real = 0, 0
        D_MSD_fake, D_MSD_real = 0, 0
        for x in batch['D_MPD_fake']: D_MPD_fake += F.mse_loss(x, torch.zeros_like(x))
        for x in batch['D_MSD_fake']: D_MSD_fake += F.mse_loss(x, torch.zeros_like(x))
        for x in batch['D_MPD_real']: D_MPD_real += F.mse_loss(x, torch.ones_like(x))
        for x in batch['D_MSD_real']: D_MSD_real += F.mse_loss(x, torch.ones_like(x))
                    
        return {
            "D_MPD_fake": D_MPD_fake,
            "D_MSD_fake": D_MSD_fake,
            "D_MPD_real": D_MPD_real,
            "D_MSD_real": D_MSD_real,
            "D_loss": D_MPD_fake+D_MSD_fake+D_MPD_real+D_MSD_real   
        }
