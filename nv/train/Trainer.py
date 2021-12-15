from datetime import datetime
import numpy as np
import os
import torch
import nv.loss
import wandb
from nv.data_utils import MelSpectrogram
from nv.data_utils.build_dataloaders import build_dataloaders
from nv.models import HiFiGenerator
from nv.utils import MetricsTracker
from nv.utils.util import write_json
from tqdm import tqdm
import torchaudio

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
        print(f"Total model parameters: \
            {sum(p.numel() for p in self.vocoder.parameters())}")
        from torchsummary import summary
        summary(self.vocoder, input_size=(80, 100))
        if config.resume is not None:
            print(f"Load vocoder model from checkpoint {config.resume}")
            self.vocoder.load_state_dict(torch.load(config.resume))
            
        self.G_optimizer = config.init_obj(
            config["optimizer"], torch.optim, self.vocoder.parameters()
        )
        # self.scheduler = config.init_obj(
        #     config["scheduler"], torch.optim.lr_scheduler, self.G_optimizer,
        #     steps_per_epoch=len(self.train_loader), epochs=config["n_epoch"]
        # )
        
        self.criterion = config.init_obj(config["loss"], nv.loss)
        
        self.featurizer =\
            MelSpectrogram(config["MelSpectrogram"]).to(self.device)
        
        if config["wandb"]:
            wandb.init(project=config["wandb_name"])
        self.metrics = MetricsTracker(["G_loss", "mel_loss"])
        self.step = 0
        
        self.config = config
    
    def train(self):
        for self.epoch in tqdm(range(1, self.config["n_epoch"]+1)):
            self.vocoder.train()
            for batch in self.train_loader:
                # try:
                    self.G_optimizer.zero_grad()
                    
                    batch = self.process_batch(batch)
                    self.metrics(batch)
                    batch["G_loss"].backward()
                    self.G_optimizer.step()
                    # self.scheduler.step()
                    # batch["lr"] = self.scheduler.get_last_lr()[0]
                    
                    if self.config["wandb"] and\
                        self.step % self.config["wandb"] == 0:
                            self.log_batch(batch)
                    self.step += 1
                    break
                # except Exception as inst:
                #     print(inst)

#             if self.config["wandb"]:
#                 self.log_batch(batch)
                    
            self.vocoder.eval()
            if self.val_loader is not None:
                with torch.no_grad():
                    for batch in self.val_loader:
                        try:
                            batch = self.process_batch(batch)
                            self.metrics(batch)
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
        

    
    def process_batch(self, batch):
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
        outputs = self.vocoder(**batch)
        batch.update(outputs)

        batch['mel_pred'] = self.featurizer(batch["wav_pred"], 42)['mel'] #don't care about length
        
        #calculate loss
        batch.update(self.criterion(batch))
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
            f"G_loss_{mode}": self.metrics["G_loss"],
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

