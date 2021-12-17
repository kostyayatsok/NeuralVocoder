from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm
from utils import SLOPE, init_weights

class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods: List, *args, **kwargs) -> None:
        super().__init__()
        self.periods = periods
        classifiers = []
        for p in self.periods:
            classifiers.append(PeriodClassifier(p))
        self.classifiers = nn.ModuleList(classifiers)
        
        init_weights(self)
    def forward(self, audio: torch.Tensor) -> List[torch.Tensor]:
        """
            audio -- 1D raw audio of length T
        """
        T = audio.size(1)
        res = []
        for p, classifier in zip(self.periods, self.classifiers):
            x = F.pad(audio, (0, T % p))
            x = x.view(-1, 1, p, T // p)
            x = classifier(x)
            res.append(x)
        return res
    
class PeriodClassifier(nn.Module):
    # TODO weight normalization
    def __init__(self, p: int) -> None:
        super().__init__()
        layers = []
        in_ch = p
        for l in range(1, 5):
            out_ch = 2**(5+l)
            layers.append(nn.Sequential(  
                nn.Conv2d(in_ch, out_ch, (5, 1), stride=(3, 1), padding=(2, 0)),
                nn.LeakyReLU()
            ))
            in_ch = out_ch
        out_ch = 1024
        layers.append(nn.Sequential(  
            nn.Conv2d(in_ch, out_ch, (5, 1), padding=(2, 0)),
            nn.LeakyReLU(),
            nn.Conv2d(out_ch, 1, (3, 1), padding=(2, 0))
        ))
        self.net = nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    

class MultiScaleDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifiers = nn.ModuleList([
            ScaleClassifier(0, spectral_norm),
            ScaleClassifier(1, weight_norm),
            ScaleClassifier(2, weight_norm),
        ])
        
        init_weights(self)
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        res = []
        for classifier in self.classifiers:
            res.append(classifier(x))
        return res

class ScaleClassifier(nn.Module):
    def __init__(self, x2, norm_layer) -> None:
        super().__init__()
        self.pooling = nn.Sequential(
            nn.AvgPool1d(4, 2, padding=2) for _ in range(x2)
        )
        self.net = nn.Sequential(
            norm_layer(nn.Conv1d(1, 128, 15, 1, padding=7)),
            nn.LeakyReLU(SLOPE),
            norm_layer(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            nn.LeakyReLU(SLOPE),
            norm_layer(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            nn.LeakyReLU(SLOPE),
            norm_layer(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            nn.LeakyReLU(SLOPE),
            norm_layer(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            nn.LeakyReLU(SLOPE),
            norm_layer(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            nn.LeakyReLU(SLOPE),
            norm_layer(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
            nn.LeakyReLU(SLOPE),
            norm_layer(nn.Conv1d(1024, 1, 3, 1, padding=1)),
            nn.LeakyReLU(SLOPE),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #torch.flatten(, start_dim=1)
        return self.net(self.pooling(x))