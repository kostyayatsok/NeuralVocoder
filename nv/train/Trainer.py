from datetime import datetime
import numpy as np
import os
import torch
import nv.loss
import wandb
from nv.data_utils import MelSpectrogram
from nv.data_utils.build_dataloaders import build_dataloaders
from nv.models import HiFiGenerator
from nv.models import MultiScaleDiscriminator, MultiPeriodDiscriminator
from nv.utils import MetricsTracker
from nv.utils.util import write_json
from tqdm import tqdm
import torchaudio
from torchsummary import summary
import torch.nn.functional as F

class Trainer:
    def __init__(self, config):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs(f"{config['save_dir']}/{self.run_id}/", exist_ok=True)
        write_json(config.config, f"{config['save_dir']}/{self.run_id}/config.json")
        
        self.tokenizer = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()
        self.train_loader, self.val_loader = build_dataloaders(config)
        print(f"Use {len(self.train_loader)} batches for training and",
                f"{0 if self.val_loader is None else len(self.val_loader)} for validation.")
            
        self.vocoder = HiFiGenerator(**config["Vocoder"]).to(self.device)
        self.MSD = MultiScaleDiscriminator().to(self.device)
        self.MPD = MultiPeriodDiscriminator(**config["MPD"]).to(self.device)
        print(f"Total vocoder parameters: \
                {sum(p.numel() for p in self.vocoder.parameters())}")
        print(f"Total MSD parameters: \
                {sum(p.numel() for p in self.MSD.parameters())}")
        print(f"Total MPD parameters: \
                {sum(p.numel() for p in self.MPD.parameters())}")

        # summary(self.vocoder, input_size=(80, 100))
        if config.resume is not None:
            print(f"Load vocoder model from checkpoint {config.resume}")
            self.vocoder.load_state_dict(torch.load(config.resume))

        self.G_optimizer = config.init_obj(
            config["optimizer"], torch.optim, self.vocoder.parameters()
        )
        self.G_scheduler = config.init_obj(
            config["scheduler"], torch.optim.lr_scheduler, self.G_optimizer
        )

        self.D_optimizer = config.init_obj(
            config["optimizer"], torch.optim,
            list(self.MPD.parameters()) + list(self.MSD.parameters())
        )
        self.D_scheduler = config.init_obj(
            config["scheduler"], torch.optim.lr_scheduler, self.D_optimizer
        )

        
        self.criterionD = config.init_obj(config["lossD"], nv.loss)
        self.criterionG = config.init_obj(config["lossG"], nv.loss)
        
        self.featurizer =\
            MelSpectrogram(config["MelSpectrogram"]).to(self.device)
        
        if config["wandb"]:
            wandb.init(project=config["wandb_name"])
        self.metrics = MetricsTracker([
            "mel_loss",
            "feature_loss",
            "G_MPD_fake",
            "G_MSD_fake",
            "G_loss",
            "D_MPD_fake",
            "D_MSD_fake",
            "D_MPD_real",
            "D_MSD_real",
            "D_loss",
        ])
        self.step = 0
        
        self.config = config
    
    def train(self):
        for self.epoch in tqdm(range(1, self.config["n_epoch"]+1)):
            self.vocoder.train()
            self.MPD.train()
            self.MSD.train()
            for batch in self.train_loader:
                try:
                    self.G_optimizer.zero_grad()
                    self.D_optimizer.zero_grad()
                    
                    batch = self.process_batch(batch, True)
                    
                    if self.config["wandb"] and\
                        self.step % self.config["wandb"] == 0:
                            self.log_batch(batch)
                    self.step += 1
                except Exception as inst:
                    print(inst)
            
            self.vocoder.eval()
            self.MPD.eval()
            self.MSD.eval()
            if self.val_loader is not None:
                with torch.no_grad():
                    for batch in self.val_loader:
                        try:
                            batch = self.process_batch(batch)
                        except Exception as inst:
                            print(inst)
                if self.config["wandb"]:
                    self.log_batch(batch, mode="val")
            if self.config["wandb"]:
                self.log_test()
            if self.config["save_period"] and\
                self.epoch % self.config["save_period"] == 0:
                    torch.save(
                        self.vocoder.state_dict(),
                        f"{self.config['save_dir']}/"+\
                        f"{self.run_id}/vocoder.pt"
                    )
                    torch.save(
                        self.MPD.state_dict(),
                        f"{self.config['save_dir']}/"+\
                        f"{self.run_id}/MPD.pt"
                    )
                    torch.save(
                        self.MSD.state_dict(),
                        f"{self.config['save_dir']}/"+\
                        f"{self.run_id}/MSD.pt"
                    )
            self.G_scheduler.step()
            self.D_scheduler.step()

    
    def process_batch(self, batch, BACKPROP=False):
        #move tensors to cuda:
        for key in [
            "waveform", "waveform_length", "tokens", "token_lengths"]:
                batch[key] = batch[key].to(self.device)
            
        #creating mel spectrogram from ground truth wav
        batch.update(
            self.featurizer(batch["waveform"], batch["waveform_length"])
        )
        batch['mel_mask'] = self.lens2mask(
            batch['mel'].size(-1), batch['mel_length'])

        #run model
        batch['wav_pred'] = self.vocoder(**batch)
        pad = batch["waveform"].size(-1) - batch["wav_pred"].size(-1)
        batch["wav_pred"] = F.pad(batch["wav_pred"], (0, pad))


        batch['mel_pred'] = self.featurizer(batch["wav_pred"], 42)['mel'] #don't care about length
        
        batch['D_MPD_fake'] = self.MPD(batch["wav_pred"].detach())
        batch['D_MSD_fake'] = self.MSD(batch["wav_pred"].detach())
        batch['D_MPD_real'] = self.MPD(batch["waveform"].detach())
        batch['D_MSD_real'] = self.MSD(batch["waveform"].detach())
        #calculate D loss
        batch.update(
            self.criterionD(batch)
        )
        if BACKPROP:
            batch["D_loss"].backward()
            self.D_optimizer.step()
            batch["lr_D"] = self.D_scheduler.get_last_lr()[0]
            
        batch['G_MPD_fake'] = self.MPD(batch["wav_pred"])
        batch['G_MSD_fake'] = self.MSD(batch["wav_pred"])
        batch['G_MPD_real'] = self.MPD(batch["waveform"])
        batch['G_MSD_real'] = self.MSD(batch["waveform"])

        #calculate loss
        batch.update(
            self.criterionG(batch)
        )
        self.metrics(batch)        
        
        if BACKPROP:
            batch["G_loss"].backward()
            self.G_optimizer.step()
            batch["lr_G"] = self.G_scheduler.get_last_lr()[0]
                    
        return batch
    
    def log_batch(self, batch, mode="train"):
        idx = np.random.randint(batch["mel"].size(0))
        
        mel = batch["mel"][idx].cpu().detach().numpy()
        mel_pred = batch["mel_pred"][idx].cpu().detach().numpy()
        wav = batch["waveform"][idx].cpu().detach().numpy()
        wav_pred = batch["wav_pred"][idx].cpu().detach().numpy()
        
        dict2log = {
            "step": self.step,
            "epoch": self.epoch,
            f"mel_loss_{mode}": self.metrics["mel_loss"],
            f"feature_loss_{mode}": self.metrics["feature_loss"],
            f"G_MPD_fake_{mode}": self.metrics["G_MPD_fake"],
            f"G_MSD_fake_{mode}": self.metrics["G_MSD_fake"],
            f"G_loss_{mode}": self.metrics["G_loss"],
            f"D_MPD_fake_{mode}": self.metrics["D_MPD_fake"],
            f"D_MSD_fake_{mode}": self.metrics["D_MSD_fake"],
            f"D_MPD_real_{mode}": self.metrics["D_MPD_real"],
            f"D_MSD_real_{mode}": self.metrics["D_MSD_real"],
            f"D_loss_{mode}": self.metrics["D_loss"],
            f"orig_mel_{mode}": wandb.Image(mel, caption="Original mel"),
            f"pred_mel_{mode}": wandb.Image(mel_pred, caption="Predicted mel"),
            f"orig_audio_{mode}": wandb.Audio(
                wav, caption="Original audio", sample_rate=22050),
            f"pred_audio_{mode}": wandb.Audio(
                wav_pred,
                caption="Predicted audio", sample_rate=22050),
            f"text_{mode}": wandb.Html(batch["transcript"][idx]),
        }
        if "lr" in batch:
            dict2log["lr"] = batch["lr"]
        wandb.log(dict2log)

    @torch.no_grad()
    def log_test(self):
        return
        self.vocoder.eval()
        for i, text in enumerate(self.config["TestData"]):
            tokens, length = self.tokenizer(text)
            tokens = tokens.to(self.device)
            mask = torch.ones(tokens.size(), dtype=torch.bool, device=self.device)
            mel = self.vocoder.inference(tokens, mask)
            wav = self.vocoder.inference(mel).cpu().detach().numpy()[0]
            mel = mel.cpu().detach().numpy()[0]
            wandb.log({
                "step": self.step,
                "epoch": self.epoch,
                f"Test/mel{i}": wandb.Image(mel, caption=text),
                f"Test/wav{i}": wandb.Audio(
                    wav, caption=text, sample_rate=22050),    
            })
            
    def lens2mask(self, max_len, lens):
        return torch.arange(max_len, device=lens.device)[None, :] <= lens[:, None]

